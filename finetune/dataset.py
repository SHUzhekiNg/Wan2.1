import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class ActionDataset(Dataset):
    def __init__(self, data_list, stats_path=None, To=5, Ta=16, stride=1, camera_name='cam_high', transform=None):
        """
        Args:
            data_list: List of 'result' dicts, each containing a 'trajectory'.
            stats_path: Path to dataset statistics json.
            dataset_path: Path to the original dataset file (used for saving stats).
            To: History Observation Frames.
            Ta: Future Prediction Action Steps.
            stride: Stride for sliding window.
            camera_name: Which camera view to use.
            transform: Image transformations.
        """
        self.data_list = data_list
        self.To = To
        self.Ta = Ta
        self.stride = stride
        self.camera_name = camera_name
        self.transform = transform
        
        # Determine action dim from data
        self.action_dim = 0
        if 'results' in self.data_list and len(self.data_list['results']) > 0:
             # Find first non-empty trajectory
             for res in self.data_list['results']:
                 traj = res.get('trajectory', [])
                 if len(traj) > 0:
                     self.action_dim = len(traj[0]['action'])
                     break

        # Load stats
        self.q01 = None
        self.q99 = None
        stats_loaded = False
        
        if stats_path and os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    stats = json.load(f)
                
                # Logic adapted from SimpleVLAWebDataset to handle different stats formats
                q01_temp = None
                q99_temp = None
                
                if "action" in stats and "min" in stats["action"]:
                     q01_temp = np.asarray(stats["action"]["min"], np.float32)
                     q99_temp = np.asarray(stats["action"]["max"], np.float32)
                elif "action" in stats and "q01" in stats["action"]:
                     q01_temp = np.asarray(stats["action"]["q01"], np.float32)
                     q99_temp = np.asarray(stats["action"]["q99"], np.float32)
                else:
                    # Try nested key (like in libero stats)
                    key = list(stats.keys())[0]
                    if "action" in stats[key]:
                        if "min" in stats[key]["action"]:
                            q01_temp = np.asarray(stats[key]["action"]["min"], np.float32)
                            q99_temp = np.asarray(stats[key]["action"]["max"], np.float32)
                        else:
                            q01_temp = np.asarray(stats[key]["action"]["q01"], np.float32)
                            q99_temp = np.asarray(stats[key]["action"]["q99"], np.float32)
                
                if q01_temp is not None:
                    if self.action_dim > 0 and q01_temp.shape[0] != self.action_dim:
                        print(f"Warning: Loaded stats dimension {q01_temp.shape[0]} does not match data action dim {self.action_dim}. Ignoring loaded stats.")
                    else:
                        self.q01 = q01_temp
                        self.q99 = q99_temp
                        stats_loaded = True
                        
            except Exception as e:
                print(f"Warning: Failed to load stats from {stats_path}: {e}. Using default normalization.")
        
        if not stats_loaded and self.action_dim > 0:
            print("Calculating stats from dataset...")
            all_actions = []
            # Sample a subset if dataset is too large to avoid OOM or slow startup
            # But for accuracy, we should use all.
            count = 0
            for res in self.data_list['results']:
                traj = res.get('trajectory', [])
                for step in traj:
                    all_actions.append(step['action'])
                    count += 1
            
            if count > 0:
                all_actions = np.array(all_actions)
                self.q01 = all_actions.min(axis=0).astype(np.float32)
                self.q99 = all_actions.max(axis=0).astype(np.float32)
                print(f"Calculated stats: min={self.q01}, max={self.q99}")

                # Handle constant values (max == min) to avoid division by zero
                # This is handled in __getitem__ by checking denom == 0
            else:
                print("Warning: No actions found in dataset to calculate stats.")

        # Pre-calculate windows
        self.windows = []
        if 'results' in self.data_list:
            results = self.data_list['results']
        else:
            results = [] 
            
        for traj_idx, result in enumerate(results):
            trajectory = result.get('trajectory', [])
            total_steps = len(trajectory)
            
            for start in range(0, total_steps - self.Ta + 1, self.stride):
                self.windows.append((traj_idx, start))
                
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        traj_idx, start = self.windows[idx]
        result = self.data_list['results'][traj_idx]
        trajectory = result['trajectory']
        
        # Calculate indices
        vs = start - self.To + 1
        ve = start + self.Ta + 1
        
        # Collect frames and actions
        frames = []
        actions = []
        
        # Video frames: [vs, ve)
        for i in range(vs, ve):
            idx_to_use = max(0, i) 
            idx_to_use = min(idx_to_use, len(trajectory) - 1)
            
            step = trajectory[idx_to_use]
            obs = step['observation']
            img_data = obs['images'][self.camera_name]
            
            # Normalize image to [-1, 1]
            if isinstance(img_data, np.ndarray):
                if img_data.shape[0] == 3: # (C, H, W)
                    img = torch.from_numpy(img_data).float() / 127.5 - 1.0
                else: # (H, W, C)
                    img = torch.from_numpy(img_data).permute(2, 0, 1).float() / 127.5 - 1.0
            else:
                img = torch.zeros((3, 256, 256)) 
                
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Actions: [start, start + Ta)
        for i in range(start, start + self.Ta):
            idx_to_use = min(i, len(trajectory) - 1)
            step = trajectory[idx_to_use]
            act = step['action']
            actions.append(act)
            
        # Stack
        video = torch.stack(frames).to(torch.bfloat16) # (T, C, H, W)
        video = video.permute(1, 0, 2, 3) # (C, T, H, W)
        
        actions_np = np.array(actions)
        
        # Normalize actions to [-1, 1]
        if self.q01 is not None and self.q99 is not None:
            denom = self.q99 - self.q01
            denom[denom == 0] = 1.0
            actions_norm = 2 * ((actions_np - self.q01) / denom) - 1
            actions = torch.from_numpy(actions_norm).to(torch.bfloat16)
        else:
            actions = torch.from_numpy(actions_np).to(torch.bfloat16)

        # Prompt
        prompt = trajectory[0]['observation'].get('prompt', "")
        
        return {
            "video": video,
            "actions": actions,
            "prompt": prompt
        }
