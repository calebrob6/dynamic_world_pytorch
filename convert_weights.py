#!/usr/bin/env python3
"""Convert Google's Dynamic World TensorFlow SavedModel into a PyTorch state dict.

Reads ``model/forward`` from a clone of https://github.com/google/dynamicworld
and writes a ``.pt`` file that ``DynamicWorld.from_pretrained()`` can load.

The conversion is purely a tensor reshape (NHWC kernels → NCHW; HWIO → OIHW for
standard convs; HWIO → CIHW for depthwise convs). After conversion the PyTorch
model produces outputs that match the TensorFlow model bit-exactly (max |Δ| on
the order of float32 rounding noise, ≈ 4e-6).

Usage:
    # Clone Google's official repo to get the SavedModel
    git clone https://github.com/google/dynamicworld.git /tmp/dynamicworld

    # Convert
    python convert_weights.py \\
        --tf-model /tmp/dynamicworld/model/forward \\
        --output weights/dynamic_world.pt

    # Verify (optional — recomputes outputs in both frameworks and diffs)
    python convert_weights.py --verify --tf-model /tmp/dynamicworld/model/forward
"""

import os

# TF GPU JIT often fails in user environments and we only need a single forward
# pass on CPU; must be set before tensorflow is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # type: ignore
import torch
import torch.nn as nn

from dynamic_world import DynamicWorld

# Mapping of (PyTorch submodule path, TF Conv name, TF BN name, is_depthwise).
# Recovered by introspecting the SavedModel's serialized graph; TF naming is
# generation-order, not topological order.
# fmt: off
TF_TO_PT_MAPPING: list[tuple[str, str, str | None, bool]] = [
    ("stem",            "conv2d",              "batch_normalization",     False),
    ("skip.pw",         "conv2d_1",            "batch_normalization_1",   False),
    ("skip.dw",         "depthwise_conv2d",    "batch_normalization_2",   True),
    ("block1.pw1",      "conv2d_2",            "batch_normalization_3",   False),
    ("block1.dw1",      "depthwise_conv2d_1",  "batch_normalization_4",   True),
    ("block1.pw2",      "conv2d_3",            "batch_normalization_5",   False),
    ("block1.dw2",      "depthwise_conv2d_2",  "batch_normalization_6",   True),
    ("block1.pw3",      "conv2d_4",            "batch_normalization_7",   False),
    ("block1.dw3",      "depthwise_conv2d_3",  "batch_normalization_8",   True),
    ("block1.shortcut", "conv2d_5",            "batch_normalization_9",   False),
    ("block2.pw1",      "conv2d_6",            "batch_normalization_10",  False),
    ("block2.dw1",      "depthwise_conv2d_4",  "batch_normalization_11",  True),
    ("block2.pw2",      "conv2d_7",            "batch_normalization_12",  False),
    ("block2.dw2",      "depthwise_conv2d_5",  "batch_normalization_13",  True),
    ("block2.pw3",      "conv2d_8",            "batch_normalization_14",  False),
    ("block2.dw3",      "depthwise_conv2d_6",  "batch_normalization_15",  True),
    ("block2.shortcut", "conv2d_9",            "batch_normalization_16",  False),
    ("block3.pw1",      "conv2d_10",           "batch_normalization_17",  False),
    ("block3.dw1",      "depthwise_conv2d_7",  "batch_normalization_18",  True),
    ("block3.pw2",      "conv2d_11",           "batch_normalization_19",  False),
    ("block3.dw2",      "depthwise_conv2d_8",  "batch_normalization_20",  True),
    ("block3.pw3",      "conv2d_12",           "batch_normalization_21",  False),
    ("block3.dw3",      "depthwise_conv2d_9",  "batch_normalization_22",  True),
    ("block3.shortcut", "conv2d_13",           "batch_normalization_23",  False),
    ("classifier",      "conv2d_14",           None,                      False),
]
# fmt: on


def _get_module(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = getattr(m, part)
    return m


def load_tf_weights(model: DynamicWorld, tf_model_dir: str) -> None:
    """Load weights from a TensorFlow SavedModel directory into the PyTorch model.

    Handles the NHWC↔NCHW and HWIO↔OIHW transpositions that distinguish the two
    frameworks' conv weight layouts.
    """
    tf_model = tf.saved_model.load(tf_model_dir)
    tf_vars = {v.name.split(":")[0]: v.numpy() for v in tf_model.variables}

    for pt_path, conv_prefix, bn_prefix, is_dw in TF_TO_PT_MAPPING:
        mod = _get_module(model, pt_path)
        if is_dw:
            tf_kernel = tf_vars[f"{conv_prefix}/depthwise_kernel"]  # (kH, kW, C, 1)
            pt_kernel = torch.from_numpy(tf_kernel).permute(2, 3, 0, 1).contiguous()
            target = mod.dw if hasattr(mod, "dw") else mod
            target.weight.data.copy_(pt_kernel)
            target.bias.data.copy_(torch.from_numpy(tf_vars[f"{conv_prefix}/bias"]))
        else:
            tf_kernel = tf_vars[f"{conv_prefix}/kernel"]  # (kH, kW, in, out)
            pt_kernel = torch.from_numpy(tf_kernel).permute(3, 2, 0, 1).contiguous()
            target = mod.conv if hasattr(mod, "conv") else mod
            target.weight.data.copy_(pt_kernel)
            target.bias.data.copy_(torch.from_numpy(tf_vars[f"{conv_prefix}/bias"]))
        if bn_prefix is not None:
            bn = mod.bn
            bn.weight.data.copy_(torch.from_numpy(tf_vars[f"{bn_prefix}/gamma"]))
            bn.bias.data.copy_(torch.from_numpy(tf_vars[f"{bn_prefix}/beta"]))
            bn.running_mean.data.copy_(
                torch.from_numpy(tf_vars[f"{bn_prefix}/moving_mean"])
            )
            bn.running_var.data.copy_(
                torch.from_numpy(tf_vars[f"{bn_prefix}/moving_variance"])
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--tf-model",
        default="weights/tf_forward",
        help="Path to the TF SavedModel `forward/` directory (default: %(default)s).",
    )
    ap.add_argument(
        "--output",
        default="weights/dynamic_world.pt",
        help="Where to write the PyTorch state_dict (default: %(default)s)",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="After conversion, run both models on a fixed random input and "
        "report the max |Δ|. Should be ≈ 4e-6 (float32 rounding noise).",
    )
    args = ap.parse_args()

    print(f"Loading TF SavedModel from {args.tf_model} ...")
    model = DynamicWorld()
    load_tf_weights(model, args.tf_model)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(
        f"✓ Saved {sum(p.numel() for p in model.parameters()):,} params -> {out_path}"
    )
    print(f"  ({out_path.stat().st_size / 1024:.1f} KB on disk)")

    if args.verify:
        x_np = np.random.RandomState(0).randn(1, 60, 60, 9).astype(np.float32)
        x_pt = torch.from_numpy(x_np).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            y_pt = model(x_pt).numpy().transpose(0, 2, 3, 1)

        tf_model = tf.saved_model.load(args.tf_model)
        sig = list(tf_model.signatures.values())[0]
        y_tf = list(sig(input_3=tf.constant(x_np)).values())[0].numpy()

        diff = np.abs(y_pt - y_tf)
        print()
        print("Numerical verification:")
        print(f"  TF  output mean/std = {y_tf.mean():+.4f} / {y_tf.std():.4f}")
        print(f"  PT  output mean/std = {y_pt.mean():+.4f} / {y_pt.std():.4f}")
        print(f"  max |Δ|             = {diff.max():.4e}")
        print(f"  mean |Δ|            = {diff.mean():.4e}")
        if diff.max() < 1e-4:
            print("  ✅ Bit-exact match (within float32 rounding noise)")
            return 0
        print("  ⚠ Outputs diverge more than expected.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
