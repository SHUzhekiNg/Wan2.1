import pickle
import json
import numpy as np
import argparse
import os

def calculate_stats(input_path, output_path):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    all_actions = []
    
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print("Error: Unknown data format. Expected a dict with 'results' or a list of trajectories.")
        return

    print(f"Processing {len(results)} trajectories...")
    for traj_idx, res in enumerate(results):
        trajectory = res.get('trajectory', [])
        for step in trajectory:
            action = step.get('action')
            if action is not None:
                all_actions.append(action)

    if not all_actions:
        print("Error: No actions found in the dataset.")
        return

    all_actions = np.array(all_actions)
    print(f"Total action steps: {len(all_actions)}")
    print(f"Action dimension: {all_actions.shape[1]}")

    # Calculate statistics
    q01 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)
    mean = np.mean(all_actions, axis=0)
    std = np.std(all_actions, axis=0)
    min_val = np.min(all_actions, axis=0)
    max_val = np.max(all_actions, axis=0)

    stats = {
        "action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_val.tolist(),
            "max": max_val.tolist(),
        }
    }

    print(f"Saving statistics to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate action statistics from a pkl file.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input .pkl file")
    parser.add_argument("--output", "-o", type=str, default="action_stats.json", help="Path to the output .json file")
    
    args = parser.parse_args()
    
    calculate_stats(args.input, args.output)

# python finetune/calculate_stats.py --input path/to/your/data.pkl --output finetune/your_stats.json