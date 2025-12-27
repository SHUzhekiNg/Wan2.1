import pickle
import yaml
import numpy as np
import os
import torch
from dataset import ActionDataset

def test_dataset_class(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config.get('data_path')
    stats_path = config.get('stats_path')
    To = config.get('To', 5)
    Ta = config.get('Ta', 16)
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Initializing ActionDataset with To={To}, Ta={Ta}, stats_path={stats_path}")
    dataset = ActionDataset(
        data_list=data,
        stats_path=stats_path,
        To=To,
        Ta=Ta
    )
    
    print(f"Dataset length (windows): {len(dataset)}")
    
    if len(dataset) > 0:
        # Test first item
        print("\nTesting __getitem__(0)...")
        item = dataset[0]
        
        video = item['video']
        actions = item['actions']
        prompt = item['prompt']
        
        print(f"Video shape: {video.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Prompt: {prompt}")
        
        # Verify shapes
        expected_frames = To + Ta
        if video.shape[1] != expected_frames:
            print(f"ERROR: Video frames {video.shape[1]} != expected {expected_frames}")
        else:
            print("Video frame count correct.")
            
        if actions.shape[0] != Ta:
            print(f"ERROR: Action steps {actions.shape[0]} != expected {Ta}")
        else:
            print("Action steps correct.")
            
        # Check normalization range
        print(f"Video range: [{video.min():.4f}, {video.max():.4f}]")
        print(f"Action range: [{actions.min():.4f}, {actions.max():.4f}]")
        
        # Check padding logic
        # If index 0 corresponds to start=0.
        # vs = 0 - 5 + 1 = -4.
        # Frames: -4, -3, -2, -1, 0, ..., 16.
        # Indices used: 0, 0, 0, 0, 0, ..., 16.
        # So first 5 frames should be identical?
        
        first_5_frames = video[:, :5, :, :]
        # Check if they are identical
        diff = (first_5_frames[:, 1:] - first_5_frames[:, :-1]).abs().sum()
        if diff == 0:
            print("Padding verification: First 5 frames are identical (Correct for start=0).")
        else:
            print(f"Padding verification: First 5 frames are NOT identical. Diff: {diff}")

if __name__ == "__main__":
    config_file = "/disk2/licheng/code/Wan2.1/finetune/config.yaml"
    test_dataset_class(config_file)
