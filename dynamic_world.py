#!/usr/bin/env python3
"""PyTorch port of Google's Dynamic World forward model.

Dynamic World (Brown et al., 2022) is a fully-convolutional Sentinel-2 land
cover classifier published as a TensorFlow SavedModel at
https://github.com/google/dynamicworld. This module reimplements the
``forward`` graph in PyTorch and matches the official model bit-exactly when
the converted weights are loaded.

The model uses a stem, antialiased 2× downsample, three residual blocks with a
skip branch, bilinear upsample, channel L2 normalization, and a 1×1 classifier.
Each residual block sums a three-layer pointwise/depthwise main path with a
projected shortcut.

Quick start:

    from dynamic_world import DynamicWorld
    model = DynamicWorld.from_pretrained("weights/dynamic_world.pt").eval()
    # x: float32 tensor of shape (B, 9, H, W) — already normalized!
    logits = model(x)
    probs = logits.softmax(dim=1)
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNMish(nn.Module):
    """Conv2d → BN → Mish (default 1×1 / VALID; pass padding=1 for SAME 3×3)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | str = 0,
        groups: int = 1,
        bias: bool = True,
        bn_eps: float = 1e-3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=0.01)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWMish(nn.Module):
    """3×3 SAME-padded depthwise conv → BN → Mish."""

    def __init__(self, ch: int, bn_eps: float = 1e-3):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=True)
        self.bn = nn.BatchNorm2d(ch, eps=bn_eps, momentum=0.01)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.dw(x)))


class ResidualBlock(nn.Module):
    """Custom inverted-residual block used in Dynamic World.

    Main path interleaves three (Conv1×1 + DW3×3) pairs (each followed by
    BN + Mish), so each block contains 3 pointwise convs and 3 depthwise convs.
    The shortcut is a single Conv1×1 → BN → Mish from the block's input to the
    output width. The two paths are summed (no activation after the add).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pw1 = ConvBNMish(in_ch, out_ch, kernel_size=1)
        self.dw1 = DWMish(out_ch)
        self.pw2 = ConvBNMish(out_ch, out_ch, kernel_size=1)
        self.dw2 = DWMish(out_ch)
        self.pw3 = ConvBNMish(out_ch, out_ch, kernel_size=1)
        self.dw3 = DWMish(out_ch)
        self.shortcut = ConvBNMish(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.dw3(self.pw3(self.dw2(self.pw2(self.dw1(self.pw1(x))))))
        return main + self.shortcut(x)


class SkipBranch(nn.Module):
    """Encoder→decoder skip: Conv1×1 → BN → Mish → DW3×3 → BN → Mish."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pw = ConvBNMish(in_ch, out_ch, kernel_size=1)
        self.dw = DWMish(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))


# Dynamic World class names, in label order (channels of the output tensor).
# These are the names from Brown et al. (2022). Note that the EE
# GOOGLE/DYNAMICWORLD/V1 collection uses shorter probability band names that
# differ for two classes: `built` (here: `built_area`) and `bare` (here:
# `bare_ground`). Channel order is identical, so band-by-index comparison works.
CLASS_NAMES = (
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built_area",
    "bare_ground",
    "snow_and_ice",
)


# Sentinel-2 bands consumed by Dynamic World (must be supplied in this order).
SENTINEL2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12")


class DynamicWorld(nn.Module):
    """Google Dynamic World forward model in PyTorch.

    Args:
        in_ch: number of input channels (9 for Sentinel-2).
        num_classes: 9 land-cover classes for the published model.
        m: channel multiplier (1.5 in the published model).
        b: stem-width multiplier (2 in the published model).
        base: base channel width (32 in the published model).
        scale_factor: spatial downsample factor inside the network. The
            published model uses 2.0 (the internal feature map runs at
            H/2 × W/2). The TF SavedModel uses ``tf.image.resize`` with
            ``method='triangle'`` and ``antialias=True``; we use
            ``F.interpolate(mode='bilinear', antialias=True)`` with the same
            explicit output size, which matches bit-exactly at this scale.
    """

    def __init__(
        self,
        in_ch: int = 9,
        num_classes: int = 9,
        m: float = 1.5,
        b: float = 2.0,
        base: int = 32,
        scale_factor: float = 2.0,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.scale_factor = scale_factor

        stem_c = int(round(b * base))  # 64
        skip_c = int(round(m * base))  # 48
        mid_c = int(round(b * m * base))  # 96
        high_c = int(round(b * m * m * base))  # 144

        self.stem = ConvBNMish(in_ch, stem_c, kernel_size=3, padding=1)
        self.skip = SkipBranch(stem_c, skip_c)
        self.block1 = ResidualBlock(stem_c, mid_c)
        self.block2 = ResidualBlock(mid_c, high_c)
        self.block3 = ResidualBlock(high_c + skip_c, high_c)
        self.classifier = nn.Conv2d(high_c, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = self.stem(x)
        down_size = (int(H / self.scale_factor), int(W / self.scale_factor))
        x_down = F.interpolate(
            x,
            size=down_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        skip_out = self.skip(x_down)
        b1 = self.block1(x_down)
        b2 = self.block2(b1)
        cat = torch.cat([b2, skip_out], dim=1)
        b3 = self.block3(cat)
        up = F.interpolate(b3, size=(H, W), mode="bilinear", align_corners=False)
        normed = F.normalize(up, p=2, dim=1, eps=1e-12)
        return self.classifier(normed)

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path = "weights/dynamic_world.pt",
        map_location: str | torch.device | None = "cpu",
        **kwargs,
    ) -> "DynamicWorld":
        """Build a DynamicWorld and load converted PyTorch weights.

        Returns the model in eval mode. ``weights_path`` defaults to the
        bundled ``weights/dynamic_world.pt``, which was produced by
        ``convert_weights.py`` from the official Google SavedModel. See the
        README for verification details.
        """
        model = cls(**kwargs)
        state = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state)
        model.eval()
        return model
