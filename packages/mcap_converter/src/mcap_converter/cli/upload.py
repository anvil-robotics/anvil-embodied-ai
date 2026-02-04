#!/usr/bin/env python3
"""
Upload LeRobot Dataset to Hugging Face Hub

This tool uploads a local LeRobot dataset to Hugging Face Hub
for sharing and training.

Prerequisites:
1. Login to Hugging Face first:
   $ huggingface-cli login

2. Or set HF_TOKEN environment variable:
   $ export HF_TOKEN=your_token_here

3. Verify login:
   $ huggingface-cli whoami
"""

import argparse
from pathlib import Path

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import huggingface_hub
except ImportError as e:
    print("[ERROR] Error: Missing required packages")
    print(f"Run: pip install lerobot huggingface-hub")
    print(f"Details: {e}")
    import sys
    sys.exit(1)


def upload_dataset(
    dataset_path: str,
    repo_id: str,
    private: bool = False,
    force: bool = False,
):
    """
    Upload dataset to Hugging Face Hub

    Args:
        dataset_path: Path to local dataset directory
        repo_id: Repository ID (e.g., "anvil-robot/workcell_test_120301")
        private: Make repository private
        force: Force push even if remote exists
    """
    print("=" * 70)
    print("UPLOAD DATASET TO HUGGING FACE HUB")
    print("=" * 70)

    dataset_dir = Path(dataset_path)

    # Validate dataset exists
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"\nDataset: {dataset_path}")
    print(f"Repo ID: {repo_id}")
    print(f"Private: {private}")
    print("=" * 70)

    try:
        # Check HuggingFace authentication
        print("\n[1/5] Checking Hugging Face authentication...")
        try:
            user_info = huggingface_hub.whoami()
            username = user_info['name']
            print(f"[OK] Logged in as: {username}")

            # Validate repo_id namespace matches user
            repo_namespace = repo_id.split('/')[0] if '/' in repo_id else username
            if repo_namespace != username:
                print(f"\n[WARNING] WARNING: Repository namespace mismatch!")
                print(f"  Logged in user: {username}")
                print(f"  Repo namespace: {repo_namespace}")
                print(f"\nYou can only upload to your own namespace: {username}/...")

                # Suggest correct repo_id
                dataset_name = repo_id.split('/')[-1]
                suggested_repo_id = f"{username}/{dataset_name}"
                print(f"\nSuggested repo_id: {suggested_repo_id}")

                response = input(f"\nUse {suggested_repo_id} instead? [Y/n]: ")
                if response.lower() != 'n':
                    repo_id = suggested_repo_id
                    print(f"[OK] Using: {repo_id}")
                else:
                    print("Upload cancelled")
                    return False

        except Exception as e:
            print(f"\n[ERROR] Not logged in to Hugging Face")
            print("\nPlease login first:")
            print("  $ huggingface-cli login")
            print("\nOr set token:")
            print("  $ export HF_TOKEN=your_token_here")
            print(f"\nError details: {e}")
            return False

        # Load dataset
        print(f"\n[2/5] Loading dataset...")
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)

        print(f"[OK] Dataset loaded")
        print(f"  - Episodes: {dataset.num_episodes}")
        print(f"  - Frames: {dataset.num_frames}")
        print(f"  - Robot: {dataset.meta.robot_type}")

        # Check if repo exists
        print(f"\n[3/5] Checking if repository exists...")
        try:
            repo_info = huggingface_hub.repo_info(
                repo_id=repo_id,
                repo_type="dataset",
            )
            exists = True
            print(f"[WARNING] Repository already exists: {repo_id}")

            if not force:
                response = input("Continue and overwrite? [y/N]: ")
                if response.lower() != 'y':
                    print("Upload cancelled")
                    return False
        except huggingface_hub.utils.RepositoryNotFoundError:
            exists = False
            print(f"[OK] Repository does not exist (will create new)")
        except Exception as e:
            print(f"[WARNING] Could not check repository: {e}")

        # Create repository first
        print(f"\n[4/5] Creating Hugging Face repository...")
        try:
            url = huggingface_hub.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            print(f"[OK] Repository ready: {url}")
        except Exception as e:
            print(f"[ERROR] Failed to create repository: {e}")
            print("\nPossible issues:")
            print("  1. No permission for this namespace")
            print("  2. Invalid repository name")
            print("  3. Token needs 'write' permission")
            return False

        # Upload dataset
        print(f"\n[5/5] Uploading dataset to Hugging Face Hub...")
        print("This may take several minutes depending on dataset size...")

        dataset.push_to_hub(
            repo_id=repo_id,
            token=True,
            private=private,
        )

        print("\n" + "=" * 70)
        print("[OK] Dataset uploaded successfully!")
        print("=" * 70)
        print(f"\nDataset URL: https://huggingface.co/datasets/{repo_id}")
        print(f"\nTo use in training:")
        print(f"  lerobot-train --dataset.repo_id={repo_id}")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Upload LeRobot dataset to Hugging Face Hub"
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help='Path to local dataset directory'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        help='Repository ID (e.g., anvil-robot/dataset_name). If not provided, uses dataset directory name'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force upload without confirmation if repo exists'
    )
    parser.add_argument(
        '--hf-user',
        type=str,
        help='Hugging Face username (default: auto-detect from login)'
    )

    args = parser.parse_args()

    # Determine repo_id
    if args.repo_id:
        repo_id = args.repo_id
    else:
        dataset_name = Path(args.dataset_path).name

        # Get HF username
        if args.hf_user:
            username = args.hf_user
        else:
            try:
                user_info = huggingface_hub.whoami()
                username = user_info['name']
            except:
                username = "anvil-robot"

        repo_id = f"{username}/{dataset_name}"

    # Upload
    success = upload_dataset(
        dataset_path=args.dataset_path,
        repo_id=repo_id,
        private=args.private,
        force=args.force,
    )

    if not success:
        exit(1)


if __name__ == '__main__':
    main()
