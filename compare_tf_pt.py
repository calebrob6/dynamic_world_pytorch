#!/usr/bin/env python3
"""Compare the TF SavedModel and our PyTorch port on the same random input.

Loads both models, runs them on a fixed random tensor, and prints absolute
difference statistics. Useful for confirming that the bundled PyTorch weights
still match the TF SavedModel after any change to either side.

Usage:
    git clone https://github.com/google/dynamicworld.git /tmp/dynamicworld
    pip install tensorflow
    python compare_tf_pt.py --tf-model /tmp/dynamicworld/model/forward
"""

import os

# Force TF onto CPU; one forward pass is fast enough and TF GPU JIT is flaky.
# Must be set before tensorflow is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore
import torch

from dynamic_world import DynamicWorld


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tf-model",
        type=Path,
        default=Path("weights/tf_forward"),
        help="Path to the TF SavedModel `forward/` directory (default: %(default)s).",
    )
    ap.add_argument(
        "--pt-weights",
        type=Path,
        default=Path("weights/dynamic_world.pt"),
        help="Path to the PyTorch state dict (default: %(default)s).",
    )
    ap.add_argument("--height", type=int, default=60)
    ap.add_argument("--width", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Loading TF SavedModel from {args.tf_model} ...")
    tf_model = tf.saved_model.load(args.tf_model)
    sig = list(tf_model.signatures.values())[0]
    _, sig_kwargs = sig.structured_input_signature
    tf_input_name = next(iter(sig_kwargs))

    print(f"Loading PyTorch model from {args.pt_weights} ...")
    pt_model = DynamicWorld.from_pretrained(args.pt_weights)

    rng = np.random.RandomState(args.seed)
    x_np = rng.randn(1, args.height, args.width, 9).astype(np.float32)
    print(f"Random input shape (NHWC) = {x_np.shape} (seed={args.seed})")

    x_pt = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        y_pt = pt_model(x_pt).numpy().transpose(0, 2, 3, 1)
    y_tf = list(sig(**{tf_input_name: tf.constant(x_np)}).values())[0].numpy()

    diff = np.abs(y_pt - y_tf)
    print()
    print("Numerical comparison:")
    print(f"  TF  output mean/std = {y_tf.mean():+.4f} / {y_tf.std():.4f}")
    print(f"  PT  output mean/std = {y_pt.mean():+.4f} / {y_pt.std():.4f}")
    print(f"  max |Δ|             = {diff.max():.4e}")
    print(f"  mean |Δ|            = {diff.mean():.4e}")
    if diff.max() < 1e-4:
        print("  ✅ Bit-exact match (within float32 rounding noise)")
        return 0
    print("  ⚠ Outputs diverge more than expected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
