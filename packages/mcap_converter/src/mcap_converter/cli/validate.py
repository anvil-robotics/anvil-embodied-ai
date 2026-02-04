#!/usr/bin/env python3
"""
Test Converted LeRobot Dataset

This script loads the dataset and displays basic information to verify successful conversion.
"""

import argparse
from pathlib import Path

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print("[ERROR] Error: Please install lerobot first")
    print(f"Run: pip install lerobot")
    print(f"Details: {e}")
    import sys
    sys.exit(1)


def test_dataset(repo_id: str, root: str):
    """Test if dataset can be loaded normally"""

    print("=" * 70)
    print("LeRobot Dataset Test")
    print("=" * 70)
    print(f"Repo ID: {repo_id}")
    print(f"Root: {root}")
    print("=" * 70)

    try:
        # Load dataset
        print("\n[1/5] Load dataset...")
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        print("[OK] Dataset loaded successfully")

        # Display basic information
        print("\n[2/5] Dataset Basic Information:")
        print(f"  - Total episodes: {dataset.num_episodes}")
        print(f"  - Total frames: {dataset.num_frames}")
        print(f"  - FPS: {dataset.fps}")
        print(f"  - Robot type: {dataset.meta.robot_type}")

        # Display features
        print("\n[3/5] Dataset Features:")
        for feat_name, feat_info in dataset.features.items():
            print(f"  - {feat_name}:")
            print(f"      dtype: {feat_info.get('dtype', 'N/A')}")
            print(f"      shape: {feat_info.get('shape', 'N/A')}")

        # Test reading first frame
        print("\n[4/5] Test reading data...")
        if len(dataset) > 0:
            frame = dataset[0]
            print(f"[OK] Successfully read first frame")
            print(f"  Available keys: {list(frame.keys())}")

            # Display feature shapes
            print("\n  Shape of each feature:")
            for key, value in frame.items():
                if hasattr(value, 'shape'):
                    print(f"    - {key}: {value.shape}")
                elif hasattr(value, '__len__'):
                    print(f"    - {key}: len={len(value)}")
                else:
                    print(f"    - {key}: {type(value).__name__}")
        else:
            print("[WARNING] Warning: Dataset is empty")

        # Test batch reading
        print("\n[5/5] Test batch reading...")
        num_test_frames = min(10, len(dataset))
        if num_test_frames > 0:
            for i in range(num_test_frames):
                frame = dataset[i]
            print(f"[OK] Successfully read {num_test_frames} frames")

        # Statistics
        if hasattr(dataset.meta, 'stats') and dataset.meta.stats:
            print("\n[Additional] Statistics:")
            for key, stats in dataset.meta.stats.items():
                if isinstance(stats, dict):
                    print(f"  - {key}:")
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, list):
                            print(f"      {stat_name}: [{len(stat_value)} values]")
                        else:
                            print(f"      {stat_name}: {stat_value}")

        print("\n" + "=" * 70)
        print("[OK] All tests passed! Dataset can be used normally")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LeRobot Dataset")
    parser.add_argument(
        '--repo_id',
        type=str,
        default='anvil_robot/manipulation_v1',
        help='Dataset repository ID'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='output_dataset',
        help='Dataset root directory'
    )

    args = parser.parse_args()

    # Check if directory exists
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"[ERROR] Error: Directory not found {args.root}")
        print("Please run first convert.py create dataset")
        exit(1)

    # Run test
    success = test_dataset(args.repo_id, args.root)

    if not success:
        exit(1)


if __name__ == '__main__':
    main()
