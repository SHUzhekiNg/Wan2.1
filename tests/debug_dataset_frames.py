import os
import sys
import pickle
import yaml
import torch

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(workspace_root)

from finetune.dataset import ActionDataset

def main():
    config_path = os.path.join(workspace_root, 'finetune/config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config.get('data_path')
    stats_path = config.get('stats_path')
    To = config.get('To', 5)
    Ta = config.get('Ta', 16)
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    dataset = ActionDataset(
        data_list=data,
        stats_path=stats_path,
        To=To,
        Ta=Ta,
        camera_name='cam_high'
    )
    
    print(f"Dataset config: To={To}, Ta={Ta}")
    print(f"Expected frames: {To + Ta} frames")
    print(f"Dataset length: {len(dataset)} windows\n")
    
    # Check first window
    if len(dataset) > 0:
        traj_idx, start = dataset.windows[0]
        print(f"Window 0: traj_idx={traj_idx}, start={start}")
        
        vs = start - To + 1
        ve = start + To + Ta
        print(f"Frame range: vs={vs}, ve={ve}, total={ve - vs} frames")
        
        # Show which trajectory indices are used
        trajectory = data['results'][traj_idx]['trajectory']
        print(f"Trajectory length: {len(trajectory)}")
        print("\nFrame mapping:")
        for i in range(vs, ve):
            idx_to_use = max(0, i)
            idx_to_use = min(idx_to_use, len(trajectory) - 1)
            marker = " <-- DUPLICATE" if idx_to_use == 0 and i < 0 else ""
            print(f"  Frame {i:2d} -> trajectory[{idx_to_use:2d}]{marker}")
        
        # Get actual item
        item = dataset[0]
        video = item['video']
        print(f"\nActual video shape: {video.shape}")
        print(f"Expected: (3, {To + Ta}, H, W)")
        
        # Check for frame duplication by comparing pixel values
        video_np = video.cpu().numpy()
        duplicates = 0
        for i in range(1, video.shape[1]):
            if (video_np[:, i] == video_np[:, 0]).all():
                duplicates += 1
        
        print(f"\nFrames identical to first frame: {duplicates}")

if __name__ == "__main__":
    main()
