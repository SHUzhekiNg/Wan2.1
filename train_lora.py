import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import yaml
import pickle
from easydict import EasyDict

from wan.modules.model import WanModel
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from wan.configs.shared_config import wan_shared_cfg

from finetune.dataset import ActionDataset
from finetune.models import WanActionModel

def main():
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "finetune/config.yaml")
    with open(config_path, "r") as f:
        args = EasyDict(yaml.safe_load(f))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )
    device = accelerator.device

    # 1. Load Models
    # VAE
    vae = WanVAE(
        vae_pth=os.path.join(args.checkpoint_dir, "Wan2.1_VAE.pth"), 
        device=device,
        dtype=torch.bfloat16
    )
    # vae.model.compile()
    # T5 Encoder
    print("Loading T5 Encoder...")
    t5_encoder = T5EncoderModel(
        text_len=512,
        checkpoint_path=os.path.join(args.checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.checkpoint_dir, "google/umt5-xxl"),
        device=device,
        dtype=torch.bfloat16
    )
    t5_encoder.model.requires_grad_(False)
    t5_encoder.model.eval()

    # CLIP Model
    print("Loading CLIP Model...")
    clip_model = CLIPModel(
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=os.path.join(args.checkpoint_dir, "xlm-roberta-large")
    )
    clip_model.model.requires_grad_(False)
    clip_model.model.eval()

    # Base DiT Model
    print("Loading Base DiT Model...")
    # Load as I2V model
    base_model = WanModel.from_pretrained(
        args.checkpoint_dir, 
        model_type='i2v', 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    base_model.requires_grad_(False) # Freeze base model
    
    # Action Model (LoRA + Action Encoder)
    model = WanActionModel(
        base_model, 
        action_dim=args.action_dim, 
        # state_dim removed
        hidden_dim=1024, 
        lora_rank=args.lora_rank
    )

    # 2. Dataset
    with open(args.data_path, 'rb') as f:
        data_list = pickle.load(f)
    
    dataset = ActionDataset(
        data_list, 
        stats_path=args.get('stats_path'),
        To=args.get('To', 5), 
        Ta=args.get('Ta', 16)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # 3. Optimizer
    # Only optimize parameters that require gradients (LoRA + Action components)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print(f"Process {accelerator.process_index}: Preparing for training with Accelerator...")
    print(f"Process {accelerator.process_index}: Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    print(f"Process {accelerator.process_index}: Starting training...")
    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            video = batch["video"].to(device) # (B, C, T, H, W)
            actions = batch["actions"].to(device) # (B, T, action_dim)
            prompts = batch["prompt"] # List of strings

            with torch.no_grad():
                latents = vae.encode(video) # (B, C, T', H', W')
                context = t5_encoder(prompts, device) 
                
                # Prepare I2V conditions
                # 1. CLIP Features
                # video is (B, C, T, H, W), [-1, 1]
                # Extract first frame as list of (C, 1, H, W)
                first_frames = [v[:, 0:1, :, :] for v in video]
                clip_fea = clip_model.visual(first_frames) # (B, 257, 1280)
                
                # 2. VAE Condition (y)
                # Construct y_video: first frame + zeros
                y_videos = []
                for v in video:
                    # v: (C, T, H, W)
                    y_v = torch.zeros_like(v)
                    y_v[:, 0, :, :] = v[:, 0, :, :] # Copy first frame
                    y_videos.append(y_v)
                
                y_latents = vae.encode(y_videos) # List of (C, T', H', W')
                y_latents = torch.stack(y_latents).to(torch.bfloat16)
                
                # 3. Mask
                # Create mask matching y_latents shape
                # y_latents shape: (B, 16, T', H', W')
                # Mask should be (B, 4, T', H', W')
                
                B_sz, C_sz, T_sz, H_sz, W_sz = y_latents.shape
                
                # Dynamic mask construction
                # We assume the first latent frame corresponds to the first video frame (condition)
                msk = torch.zeros(B_sz, 4, T_sz, H_sz, W_sz, device=device)
                msk[:, :, 0, :, :] = 1.0 # Condition on first latent frame
                
                # Concatenate mask and y_latents
                # y_latents: (B, 16, T', H', W')
                # msk: (B, 4, T', H', W')
                y = torch.cat([msk, y_latents], dim=1) # (B, 20, T', H', W')
                
                # Convert to list of tensors for model input
                y = [y[i] for i in range(B_sz)]

            # Flow Matching Training
            # Sample t in [0, 1]
            # Process batch directly as tensors
            batch_latents = torch.stack(latents).to(torch.bfloat16) # (B, C, T', H', W')
            t = torch.rand((batch_latents.shape[0],), device=device, dtype=torch.bfloat16)
            noise = torch.randn_like(batch_latents, dtype=torch.bfloat16)
            
            t_expanded = t.view(-1, 1, 1, 1, 1)
            x_t = (1 - t_expanded) * noise + t_expanded * batch_latents
            
            target = batch_latents - noise
            
            # Predict velocity
            # Model now accepts batch tensors directly
            with accelerator.autocast():
                # Prepare context as batch tensor
                context_batch = torch.stack([torch.cat([c.to(device), c.new_zeros(512 - c.size(0), c.size(1))]) for c in context])
                # Prepare y as batch tensor
                y_batch = torch.stack(y) if y and len(y) > 0 else None
                
                pred = model(x_t, t, context_batch, seq_len=1024, actions=actions, clip_fea=clip_fea, y=y_batch)
            
            # pred is a list of tensors [C, T', H', W'], convert to batch tensor
            pred_tensor = torch.stack(pred)
            loss = F.mse_loss(pred_tensor, target)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # Save checkpoint
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_lora_weights(os.path.join(args.output_dir, f"lora_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
