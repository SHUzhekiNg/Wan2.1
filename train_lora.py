import os
import torch
import torch.nn.functional as F
import math
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
from finetune.model_action import WanActionModel
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

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
        torch_dtype=torch.bfloat16
    )
    
    # Action Model (LoRA + Action Encoder)
    model = WanActionModel(
        model_type='i2v',
        patch_size=base_model.patch_size,
        text_len=base_model.text_len,
        in_dim=36,  # I2V: 16 (latent) + 20 (4 mask + 16 condition) = 36
        dim=base_model.dim,
        ffn_dim=base_model.ffn_dim,
        freq_dim=base_model.freq_dim,
        text_dim=base_model.text_dim,
        action_dim=args.action_dim,
        action_hidden_dim=1024,
        out_dim=base_model.out_dim,
        num_heads=base_model.num_heads,
        num_layers=base_model.num_layers,
        window_size=base_model.window_size,
        qk_norm=base_model.qk_norm,
        cross_attn_norm=base_model.cross_attn_norm,
        eps=base_model.eps
    ).to(device, dtype=torch.bfloat16)

    # Load weights from base model
    model.load_state_dict(base_model.state_dict(), strict=False)
    del base_model
    torch.cuda.empty_cache()

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze action parameters BEFORE applying LoRA
    for name, param in model.named_parameters():
        if "action_encoder" in name or "action_adaln_proj" in name:
            param.requires_grad = True

    # Apply LoRA - this will add LoRA adapters to specified modules
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters for verification
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")

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
                # Convert batch tensor to list for VAE
                video_list = [video[i] for i in range(video.shape[0])]
                
                latents = vae.encode(video_list) # List of (16, T', H', W')
                context = t5_encoder(prompts, device) 
                
                # Prepare I2V conditions
                # 1. CLIP Features - extract first frame
                first_frames = [v[:, 0:1, :, :] for v in video_list]
                clip_fea = clip_model.visual(first_frames) # (B, 257, 1280)
                
                # 2. VAE Condition (y) - following original I2V implementation
                # For each video, create: first frame + zeros for rest
                y_videos = []
                for v in video_list:
                    # v: (C, T, H, W)
                    y_v = torch.zeros_like(v)
                    y_v[:, 0, :, :] = v[:, 0, :, :] # Copy first frame only
                    y_videos.append(y_v)
                
                y_latents = vae.encode(y_videos) # List of (16, T', H', W')
                
                # 3. Create y = mask + latents for each sample
                y = []
                for y_lat in y_latents:
                    # y_lat: (16, T', H', W')
                    C_lat, T_lat, H_lat, W_lat = y_lat.shape
                    msk = torch.zeros(4, T_lat, H_lat, W_lat, device=device, dtype=y_lat.dtype)
                    msk[:, 0, :, :] = 1.0  # Condition on first latent frame
                    # Concatenate: (20, T', H', W')
                    y_i = torch.cat([msk, y_lat], dim=0)
                    y.append(y_i)

            # Flow Matching Training
            # Sample t in [0, 1]
            B = len(latents)
            t = torch.rand(B, device=device, dtype=torch.bfloat16)
            
            # Create noise and interpolation for each sample
            x_t_list = []
            target_list = []
            for i in range(B):
                noise = torch.randn_like(latents[i], dtype=torch.bfloat16)
                # Flow matching: x_t = (1-t)*noise + t*x_1
                x_t_i = (1 - t[i]) * noise + t[i] * latents[i]
                x_t_list.append(x_t_i)
                # Target is the velocity: v = x_1 - noise
                target_list.append(latents[i] - noise)
            
            # Predict velocity
            with accelerator.autocast():
                # Calculate seq_len based on actual latent dimensions
                # Get sample latent dimensions from first sample
                lat_T, lat_H, lat_W = latents[0].shape[1], latents[0].shape[2], latents[0].shape[3]
                # Access patch_size correctly through PEFT wrapper
                if hasattr(model, 'module'):
                    # Multi-GPU with DDP
                    base = model.module.base_model.model if hasattr(model.module, 'base_model') else model.module
                else:
                    # Single GPU
                    base = model.base_model.model if hasattr(model, 'base_model') else model
                patch_size = base.patch_size
                seq_len = math.ceil((lat_H * lat_W) / (patch_size[1] * patch_size[2]) * lat_T)
                
                # Model forward expects lists
                pred = model(x_t_list, t, context, seq_len=seq_len, actions=actions, clip_fea=clip_fea, y=y)
            
            # pred is a list of tensors, convert to batch for loss computation
            pred_tensor = torch.stack(pred)
            target_tensor = torch.stack(target_list)
            loss = F.mse_loss(pred_tensor, target_tensor)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        # Save checkpoint
        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            # Save only trainable parameters (LoRA + Action)
            state_dict = get_peft_model_state_dict(unwrapped_model)
            torch.save(state_dict, os.path.join(args.output_dir, f"lora_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
