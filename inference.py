#!/usr/bin/env python3
"""Run Dynamic World PyTorch inference on a local 9-band Sentinel-2 GeoTIFF."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from dynamic_world import CLASS_NAMES, SENTINEL2_BANDS, DynamicWorld

NORM_PERCENTILES = np.array(
    [
        [1.7417268007636313, 2.023298706048351],
        [1.7261204997060209, 2.038905204308012],
        [1.6798346251414997, 2.179592821212937],
        [1.7734969472909623, 2.2890068333026603],
        [2.289154079164943, 2.6171674549378166],
        [2.382939712192371, 2.773418590375327],
        [2.3828939530384052, 2.7578332604178284],
        [2.1952484264967844, 2.789092484314204],
        [1.554812948247501, 2.4140534947492487],
    ],
    dtype=np.float32,
)


def find_one(pattern: str, directory: Path) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        names = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple files match {pattern!r}:\n  {names}")
    return matches[0] if matches else None


def scene_id_from_input(path: Path) -> str | None:
    prefix = "S2_L1C_DWbands_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else None


def default_input() -> Path:
    path = find_one("S2_L1C_DWbands_*.tif", Path("data"))
    if path is None:
        raise FileNotFoundError(
            "No input was provided and no data/S2_L1C_DWbands_*.tif file exists."
        )
    return path


def normalize_sentinel2(stack: np.ndarray) -> np.ndarray:
    """Apply Dynamic World's normalization to a raw S2 stack shaped (9, H, W)."""
    x = np.log(stack * 0.005 + 1.0)
    x = (x - NORM_PERCENTILES[:, 0, None, None]) / NORM_PERCENTILES[:, 1, None, None]
    x = np.exp(x * 5.0 - 1.0)
    return x / (x + 1.0)


def predict_full(
    model: DynamicWorld, normed: np.ndarray, device: torch.device
) -> np.ndarray:
    x = torch.from_numpy(normed).unsqueeze(0).to(device)
    with torch.inference_mode():
        return model(x).softmax(dim=1)[0].cpu().numpy().astype(np.float32)


def predict_tiled(
    model: DynamicWorld,
    normed: np.ndarray,
    device: torch.device,
    tile_size: int,
) -> np.ndarray:
    _, height, width = normed.shape
    output = np.zeros((len(CLASS_NAMES), height, width), dtype=np.float32)
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            output[:, y0:y1, x0:x1] = predict_full(
                model,
                normed[:, y0:y1, x0:x1],
                device,
            )
    return output


def write_probs(path: Path, probs: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="float32",
        count=len(CLASS_NAMES),
        nodata=None,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(probs.astype(np.float32))
        for index, name in enumerate(CLASS_NAMES, start=1):
            dst.set_band_description(index, name)


def write_label(path: Path, label: np.ndarray, profile: dict) -> None:
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=None,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(label.astype(np.uint8), 1)
        dst.set_band_description(1, "label")


def compare_outputs(
    probs: np.ndarray,
    label: np.ndarray,
    official_probs_path: Path | None,
    official_label_path: Path | None,
) -> None:
    if official_probs_path is None and official_label_path is None:
        print("No official Dynamic World outputs found; skipping comparison.")
        return

    print("\nComparison to exported Dynamic World outputs:")

    if official_probs_path is not None:
        with rasterio.open(official_probs_path) as src:
            official_probs = src.read().astype(np.float32)
        if official_probs.shape != probs.shape:
            raise ValueError(
                f"Probability shape mismatch: predicted {probs.shape}, "
                f"official {official_probs.shape}"
            )
        diff = np.abs(probs - official_probs)
        official_argmax = official_probs.argmax(axis=0).astype(np.uint8)
        print(f"  official probs: {official_probs_path}")
        print(f"  mean abs probability diff: {float(diff.mean()):.9f}")
        print(f"  max abs probability diff:  {float(diff.max()):.9f}")
        print(
            "  p50/p90/p95/p99 abs diff: "
            + " ".join(f"{v:.9f}" for v in np.percentile(diff, [50, 90, 95, 99]))
        )
        print(
            f"  probability argmax agreement: {100.0 * (label == official_argmax).mean():.5f}%"
        )

    if official_label_path is not None:
        with rasterio.open(official_label_path) as src:
            official_label = src.read(1).astype(np.uint8)
        if official_label.shape != label.shape:
            raise ValueError(
                f"Label shape mismatch: predicted {label.shape}, "
                f"official {official_label.shape}"
            )
        matching = label == official_label
        print(f"  official label: {official_label_path}")
        print(f"  label agreement: {100.0 * matching.mean():.5f}%")
        print(f"  differing label pixels: {int((~matching).sum())} / {label.size}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="9-band S2 GeoTIFF. Defaults to data/S2_L1C_DWbands_*.tif.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("weights/dynamic_world.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix. Defaults to PyTorch_DynamicWorld_<scene_id>.",
    )
    parser.add_argument(
        "--official-probs",
        type=Path,
        default=None,
        help="Optional exported Dynamic World probability GeoTIFF to compare.",
    )
    parser.add_argument(
        "--official-label",
        type=Path,
        default=None,
        help="Optional exported Dynamic World label GeoTIFF to compare.",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, etc."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help=(
            "Tile size for memory-limited inference. Default 0 runs the full "
            "image at once; tiled inference can differ slightly at tile edges."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    input_path = args.input or default_input()
    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id = scene_id_from_input(input_path)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = (
            f"PyTorch_DynamicWorld_{scene_id}"
            if scene_id is not None
            else f"{input_path.stem}_dynamic_world"
        )

    official_probs_path = args.official_probs
    official_label_path = args.official_label
    if scene_id is not None:
        official_probs_path = official_probs_path or find_one(
            f"DynamicWorld_V1_{scene_id}_probs.tif",
            input_path.parent,
        )
        official_label_path = official_label_path or find_one(
            f"DynamicWorld_V1_{scene_id}_label.tif",
            input_path.parent,
        )

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    with rasterio.open(input_path) as src:
        if src.count != len(SENTINEL2_BANDS):
            raise ValueError(f"Expected 9 bands, found {src.count}: {input_path}")
        descriptions = tuple(desc or "" for desc in src.descriptions)
        if any(descriptions) and descriptions != SENTINEL2_BANDS:
            raise ValueError(
                f"Expected band descriptions {SENTINEL2_BANDS}, found {descriptions}"
            )
        stack = src.read().astype(np.float32)
        profile = src.profile.copy()

    print(f"Input: {input_path}")
    print(f"Input shape: {stack.shape}")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")

    normed = normalize_sentinel2(stack).astype(np.float32)
    model = DynamicWorld.from_pretrained(args.weights).to(device).eval()
    if args.tile_size > 0:
        probs = predict_tiled(model, normed, device, args.tile_size)
    else:
        probs = predict_full(model, normed, device)
    label = probs.argmax(axis=0).astype(np.uint8)

    probs_path = output_dir / f"{output_prefix}_probs.tif"
    label_path = output_dir / f"{output_prefix}_label.tif"
    write_probs(probs_path, probs, profile)
    write_label(label_path, label, profile)

    print(f"Wrote probabilities: {probs_path}")
    print(f"Wrote labels:        {label_path}")
    compare_outputs(probs, label, official_probs_path, official_label_path)


if __name__ == "__main__":
    main()
