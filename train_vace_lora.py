import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import math
from tqdm import tqdm
import numpy as np
import yaml
import pickle
from easydict import EasyDict

from wan.modules.model import WanAttentionBlock
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.configs.shared_config import wan_shared_cfg
from wan.vace import VaceWanModel

from finetune.dataset import ActionDataset
from finetune.vace_model_action import VaceWanActionModel, ActionEncoder
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

def vace_encode_frames(frames, masks, vae, ref_images=None):
    if ref_images is None:
        ref_images = [None] * len(frames)
    else:
        assert len(frames) == len(ref_images)

    if masks is None:
        latents = vae.encode(frames)
    else:
        masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
        inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
        reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
        inactive = vae.encode(inactive)
        reactive = vae.encode(reactive)
        latents = [
            torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)
        ]

    cat_latents = []
    for latent, refs in zip(latents, ref_images):
        if refs is not None:
            if masks is None:
                ref_latent = vae.encode(refs)
            else:
                ref_latent = vae.encode(refs)
                ref_latent = [
                    torch.cat((u, torch.zeros_like(u)), dim=0)
                    for u in ref_latent
                ]
            assert all([x.shape[1] == 1 for x in ref_latent])
            latent = torch.cat([*ref_latent, latent], dim=1)
        cat_latents.append(latent)
    return cat_latents

def vace_encode_masks(masks, vae_stride=(4, 8, 8), ref_images=None):
    if ref_images is None:
        ref_images = [None] * len(masks)
        
    result_masks = []
    for mask, refs in zip(masks, ref_images):
        c, depth, height, width = mask.shape
        new_depth = int((depth + 3) // vae_stride[0])
        height = 2 * (int(height) // (vae_stride[1] * 2))
        width = 2 * (int(width) // (vae_stride[2] * 2))

        # reshape
        mask = mask[0, :, :, :]
        mask = mask.view(depth, height, vae_stride[1], width,
                            vae_stride[1])  # depth, height, 8, width, 8
        mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
        mask = mask.reshape(vae_stride[1] * vae_stride[2], depth, height,
                            width)  # 8*8, depth, height, width

        # interpolation
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(new_depth, height, width),
            mode='nearest-exact').squeeze(0)

        if refs is not None:
            length = len(refs)
            mask_pad = torch.zeros_like(mask[:, :length, :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
        result_masks.append(mask)
    return result_masks

def vace_latent(z, m):
    return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def print_gpu_utilization():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def save_lora_checkpoint(model, output_path, rank):
    # Extract the full state dict to CPU on rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
    
    if rank == 0:
        # Extract LoRA weights + ActionEncoder (which are the only trainable parts)
        lora_state_dict = get_peft_model_state_dict(model, state_dict=state_dict)
        
        # Manually add ActionEncoder and point-wise modulation parameters
        # These are marked as requires_grad but not captured by get_peft_model_state_dict
        for k, v in state_dict.items():
            if "action_encoder" in k or "action_adaln_proj" in k:
                lora_state_dict[k] = v
                
        torch.save(lora_state_dict, output_path)
        print(f"Saved LoRA and Action weights to {output_path}")

def load_lora_checkpoint(model, checkpoint_path, rank):
    if not checkpoint_path:
        return
    
    if os.path.isdir(checkpoint_path):
        # find the latest lora_epoch_*.pth
        checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith("lora_epoch_") and f.endswith(".pth")]
        if not checkpoints:
            if rank == 0:
                print(f"No lora_epoch_*.pth found in {checkpoint_path}")
            return
        # sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])

    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"Checkpoint {checkpoint_path} not found, skipping.")
        return
    
    if rank == 0:
        print(f"Loading LoRA and Action weights from {checkpoint_path}")
    
    # Load on all ranks to avoid manual broadcasting before FSDP
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    info = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"Loaded checkpoint. Missing keys: {len(info.missing_keys)}, Unexpected keys: {len(info.unexpected_keys)}")

def main():
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "finetune/config.yaml")
    with open(config_path, "r") as f:
        args = EasyDict(yaml.safe_load(f))

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    # 1. Models
    # VAE and T5 are frozen and kept on device (or CPU depending on size)
    print(f"Loading VAE to {device}...")
    vae = WanVAE(
        vae_pth=os.path.join(args.checkpoint_dir, "Wan2.1_VAE.pth"), 
        device=device,
        dtype=dtype
    )
    
    print(f"Loading T5 to {device}...")
    t5_encoder = T5EncoderModel(
        text_len=512,
        checkpoint_path=os.path.join(args.checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.checkpoint_dir, "google/umt5-xxl"),
        device=device,
        dtype=dtype
    )
    t5_encoder.model.requires_grad_(False)
    t5_encoder.model.eval()

    print(f"Initializing VACE LoRA Action Model on Rank {rank}...")
    model = VaceWanActionModel(
        action_dim=args.action_dim,
        action_hidden_dim=args.action_hidden_dim,
        vace_in_dim=args.vace_in_dim,
    ).to(device, dtype=dtype)

    # Load pre-trained weights
    print(f"Loading pre-trained weights into VACE LoRA Action Model on Rank {rank}...")
    model.from_pretrained(args.checkpoint_dir, strict=False)

    # Freeze backbone
    model.requires_grad_(False)
    
    # Unfreeze Action components BEFORE LoRA
    for name, param in model.named_parameters():
        if "action_encoder" in name:
            param.requires_grad = True

    # Apply LoRA
    print(f"Applying LoRA to VACE Model on Rank {rank}...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.to(dtype)
    
    # Load LoRA checkpoint if provided
    if args.get("resume_checkpoint"):
        load_lora_checkpoint(model, args.resume_checkpoint, rank)

    # FSDP Wrapping
    # Note: we wrap the blocks specifically for memory efficiency
    # auto_wrap_policy = partial(
    #     lambda_auto_wrap_policy, 
    #     lambda_fn=lambda m: m in model.base_model.model.blocks or m in model.base_model.model.vace_blocks
    # )
    
    auto_wrap_policy = partial(
        lambda_auto_wrap_policy,
        lambda_fn=lambda m: isinstance(m, (VaceWanActionModel))
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )

    print(f"Wrapping model with FSDP on Rank {rank}...")
    model = FSDP(
        module=model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
        mixed_precision=mixed_precision_policy,
        sync_module_states=True,
        use_orig_params=True
    )

    # Optimizer (on Rank 0 only the trainable params)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 2. Dataset
    print(f"Loading dataset from {args.data_path} on Rank {rank}...")
    with open(args.data_path, 'rb') as f:
        data_list = pickle.load(f)
    
    dataset = ActionDataset(
        data_list, 
        stats_path=args.get('stats_path'),
        To=args.To,
        Ta=args.Ta
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # 3. Training Loop
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs...")
        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, disable=rank != 0)
        
        for batch in pbar:
            video = batch["video"].to(device).permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
            actions = batch["actions"].to(device) # (B, Ta, action_dim)
            prompts = batch["prompt"]

            with torch.no_grad():
                video_list = [video[i] for i in range(video.shape[0])]
                latents = vae.encode(video_list)
                context = t5_encoder(prompts, device)
                
                # VACE conditioning
                # Create pixel-level mask: 1 = Target (hidden), 0 = Context (visible)
                B, C, T, H, W = video.shape
                pixel_mask = torch.ones(B, 1, T, H, W, device=device, dtype=dtype)
                if args.To > 0:
                    pixel_mask[:, :, :args.To, :, :] = 0.0
                
                mask_list = [pixel_mask[i] for i in range(B)]
                z0 = vace_encode_frames(video_list, mask_list, vae=vae)
                vae_stride = (4, 8, 8)
                m0 = vace_encode_masks(mask_list, vae_stride=vae_stride)
                y = vace_latent(z0, m0)

            # Sample t and create noisy latents
            B = len(latents)
            t = torch.rand(B, device=device, dtype=dtype)
            
            x_t_list, target_list = [], []
            for i in range(B):
                noise = torch.randn_like(latents[i], dtype=dtype)
                x_t = (1 - t[i]) * noise + t[i] * latents[i].to(dtype)
                x_t_list.append(x_t)
                target_list.append(latents[i].to(dtype) - noise)
            
            # Predict
            lat_T, lat_H, lat_W = latents[0].shape[1], latents[0].shape[2], latents[0].shape[3]
            patch_size = model.module.base_model.model.patch_size
            seq_len = math.ceil((lat_H * lat_W) / (patch_size[1] * patch_size[2]) * lat_T)
            
            with amp.autocast(dtype=torch.bfloat16):
                pred = model(
                    x=x_t_list, 
                    t=t, 
                    vace_context=y,
                    context=context, 
                    seq_len=seq_len, 
                    actions=actions
                )
                
                loss = F.mse_loss(torch.stack(pred), torch.stack(target_list))
            # print_gpu_utilization() # batch size 1: Allocated: 67.30 GB | Reserved: 74.55 GB
            loss.backward()
            optimizer.step()
            # print_gpu_utilization() # batch size 1: Allocated: 17.57 GB | Reserved: 46.79 GB
            optimizer.zero_grad()

            if rank == 0:
                loss_value = loss.item()
                pbar.set_description(f"Epoch {epoch} Loss: {loss_value:.4f}")
                
                # Log to tensorboard
                writer.add_scalar('Loss/train', loss_value, global_step)
                writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            global_step += 1

        # Save Checkpoint
        dist.barrier()
        if rank == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_lora_checkpoint(model, os.path.join(args.output_dir, f"lora_epoch_{epoch}.pth"), rank)
        dist.barrier()
    
    # Close tensorboard writer
    if rank == 0:
        writer.close()
        print("Training completed. TensorBoard logs saved to", log_dir)

if __name__ == "__main__":
    main()
