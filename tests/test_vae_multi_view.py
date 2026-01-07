import os
import sys
import torch

import pickle
import yaml
import numpy as np
import imageio
from PIL import Image

# Add workspace root to sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace_root)

from finetune.dataset import ActionDataset
from wan.modules.vae import WanVAE

def save_video(video_tensor, output_path, fps=10):
    """
    video_tensor: (C, T, H, W), normalized to [-1, 1]
    """
    video_tensor = (video_tensor + 1.0) / 2.0 * 255.0
    video_tensor = video_tensor.clamp(0, 255).to(torch.uint8)
    video_tensor = video_tensor.permute(1, 2, 3, 0).cpu().numpy() # (T, H, W, C)
    
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in video_tensor:
        writer.append_data(frame)
    writer.close()

def main():
    # Configuration
    config_path = os.path.join(workspace_root, 'finetune/config.yaml')
    checkpoint_dir = "/project/peilab/licheng/models/Wan2.1-I2V-14B-720P"
    vae_checkpoint = os.path.join(checkpoint_dir, 'Wan2.1_VAE.pth')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config.get('data_path')
    stats_path = config.get('stats_path')
    To = config.get('To', 5)
    Ta = config.get('Ta', 16)
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # Create datasets for three views
    cameras = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    datasets = {}
    for cam in cameras:
        datasets[cam] = ActionDataset(
            data_list=data,
            stats_path=stats_path,
            To=To,
            Ta=Ta,
            camera_name=cam
        )
    
    print(f"Dataset length: {len(datasets[cameras[0]])}")
    
    # Get first sample from each
    idx = 10
    videos = []
    for cam in cameras:
        item = datasets[cam][idx]
        videos.append(item['video']) # (C, T, H, W)
    
    # Concatenate vertically
    # Each video is (3, T, 256, 256)
    # Concatenated: (3, T, 768, 256)
    full_video = torch.cat(videos, dim=2)
    print(f"Full video shape: {full_video.shape}")
    
    # Initialize VAE
    print(f"Initializing VAE from {vae_checkpoint}...")
    vae = WanVAE(
        vae_pth=vae_checkpoint,
        device=device,
        dtype=torch.bfloat16
    )
    
    # Encode and Decode
    print("Running VAE encode/decode...")
    full_video_cuda = full_video.to(device).to(torch.bfloat16)
    
    with torch.no_grad():
        # WanVAE.encode expects a list of videos [C, T, H, W]
        latents = vae.encode([full_video_cuda])
        reconstructed = vae.decode(latents)
    
    recon_video = reconstructed[0]
    print(f"Reconstructed video shape: {recon_video.shape}")
    
    # Save results
    os.makedirs(os.path.join(workspace_root, 'tests/outputs'), exist_ok=True)
    orig_path = os.path.join(workspace_root, 'tests/outputs/orig_multi_view.mp4')
    recon_path = os.path.join(workspace_root, 'tests/outputs/recon_multi_view.mp4')
    
    print(f"Saving original video to {orig_path}...")
    save_video(full_video, orig_path)
    
    print(f"Saving reconstructed video to {recon_path}...")
    save_video(recon_video, recon_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
