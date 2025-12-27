import argparse
import os
from huggingface_hub import snapshot_download, hf_hub_download

def download_dataset(repo_id, local_dir, repo_type="dataset", filename=None):
    """
    Download a dataset or specific file from Hugging Face.
    
    Args:
        repo_id (str): The Hugging Face repo ID (e.g., "username/dataset-name").
        local_dir (str): Local directory to save the data.
        repo_type (str): Type of repo ("dataset", "model", or "space").
        filename (str, optional): If provided, only download this specific file.
    """
    print(f"Starting download from {repo_id} to {local_dir}...")
    
    os.makedirs(local_dir, exist_ok=True)
    
    if filename:
        # Download a single file
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=local_dir
        )
        print(f"Downloaded single file to: {path}")
    else:
        # Download the entire repository
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            # ignore_patterns=["*.msgpack", "*.h5"], # Optional: skip large files you don't need
        )
        print(f"Downloaded full repo to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Hugging Face")
    parser.add_argument("--repo", type=str, required=True, help="HF Repo ID (e.g. 'OpenDriveLab/DriveLM')")
    parser.add_argument("--out", type=str, default="./data", help="Local output directory")
    parser.add_argument("--type", type=str, default="dataset", choices=["dataset", "model"], help="Repo type")
    parser.add_argument("--file", type=str, default=None, help="Specific file to download (optional)")
    
    args = parser.parse_args()
    
    try:
        download_dataset(args.repo, args.out, args.type, args.file)
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nTip: If the repo is private, make sure to run 'huggingface-cli login' first.")
