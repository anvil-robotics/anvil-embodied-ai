#!/usr/bin/env python3
"""Generate a minimal dummy model for CI testing."""

import argparse
import json
from pathlib import Path
import struct

def generate_dummy_model(output_dir: Path):
    """Generate a minimal dummy model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config.json
    config = {
        "model_type": "act",
        "input_shapes": {
            "observation.images.cam_wrist": [3, 480, 640],
            "observation.state": [14]
        },
        "output_shapes": {
            "action": [14]
        },
        "chunk_size": 100,
        "n_obs_steps": 1,
        "hidden_dim": 64,
        "dim_feedforward": 128,
        "n_heads": 4,
        "n_layers": 2
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create minimal safetensors file
    # This is a simplified version - just creates a valid safetensors header
    # Real implementation would use safetensors library
    create_minimal_safetensors(output_dir / "model.safetensors")

    print(f"Dummy model created at {output_dir}")

def create_minimal_safetensors(filepath: Path):
    """Create a minimal valid safetensors file."""
    # Minimal safetensors: header + small tensor
    import numpy as np

    try:
        from safetensors.numpy import save_file
        # Create small random tensors
        tensors = {
            "dummy_weight": np.random.randn(64, 64).astype(np.float32),
            "dummy_bias": np.random.randn(64).astype(np.float32),
        }
        save_file(tensors, str(filepath))
    except ImportError:
        # Fallback: create empty file with note
        filepath.write_text("# Placeholder - install safetensors to generate real weights")
        print("Warning: safetensors not installed, created placeholder file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy test model")
    parser.add_argument("--output", "-o", type=Path,
                       default=Path("model_zoo/test/dummy_model"),
                       help="Output directory")
    args = parser.parse_args()
    generate_dummy_model(args.output)
