"""
Pick a low-cloud Sentinel-2 L1C scene over Redmond, WA in summer 2024
and export to Google Drive or download directly from Earth Engine:
  1. The 9 Sentinel-2 bands that Dynamic World consumes
     (B2,B3,B4,B5,B6,B7,B8,B11,B12) -- in the source image's native UTM
     projection, all resampled bilinearly to the 10 m grid of B2 to match
     the DW preprocessing.
  2. The matching Dynamic World V1 prediction, split into two files because
     the label band is integer and probability bands are float32:
       - <id>_label.tif  (uint8, single band: 'label')
       - <id>_probs.tif  (float32, 9 class probability bands)

Usage:
    python ee_export_example.py --auth --project YOUR_EE_PROJECT
    python ee_export_example.py --project YOUR_EE_PROJECT
    python ee_export_example.py --project YOUR_EE_PROJECT --half-deg 0.02 --download-dir data/ee
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import ee
from ee.ee_exception import EEException

# Dynamic World's 9 input bands from Sentinel-2 L1C (TOA reflectance).
DW_INPUT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# Dynamic World V1 probability band names as exposed by the EE collection.
# Note: these differ from `dynamic_world.CLASS_NAMES` at indices 6 and 7 --
# EE uses `built` / `bare`; the paper (and our model) uses `built_area` /
# `bare_ground`. Channel order is identical.
DW_PROB_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

# Redmond, WA city center; 0.07° (~7.8 km) half-side bounding box around it.
REDMOND_LON, REDMOND_LAT = -122.1215, 47.6740
HALF_DEG = 0.07

DRIVE_FOLDER = "EarthEngineExports"
EXPORT_SCALE_M = 10


def get_region(lon, lat, half_deg):
    return ee.Geometry.Rectangle(
        [
            lon - half_deg,
            lat - half_deg,
            lon + half_deg,
            lat + half_deg,
        ]
    )


def pick_scene(region, start, end):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    n = coll.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No Sentinel-2 L1C scenes found over the selected region between "
            f"{start} and {end}."
        )
    return ee.Image(coll.first()), n


def get_matching_dw(s2_image):
    """DW V1 image system:index equals the source S2 L1C image's system:index."""
    s2_id = s2_image.get("system:index")
    dw_coll = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filter(
        ee.Filter.eq("system:index", s2_id)
    )
    if dw_coll.size().getInfo() == 0:
        return None
    return ee.Image(dw_coll.first())


def download_image(image, description, region, crs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{description}.tif"
    params = {
        "name": description,
        "region": region,
        "scale": EXPORT_SCALE_M,
        "crs": crs,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    try:
        url = image.getDownloadURL(params)
    except EEException as exc:
        raise RuntimeError(
            "Direct Earth Engine downloads are limited to small requests "
            "(currently 32 MB and 10,000 pixels per side). Use the default "
            "Drive export mode for larger AOIs."
        ) from exc
    with urllib.request.urlopen(url) as response, path.open("wb") as dst:
        shutil.copyfileobj(response, dst)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run ee.Authenticate() before initializing (one-time setup).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud project ID to use for Earth Engine initialization.",
    )
    parser.add_argument("--start", default="2024-06-01")
    # End date is exclusive in EE; pass the day after the last day you want.
    parser.add_argument("--end", default="2024-10-01")
    parser.add_argument("--lon", type=float, default=REDMOND_LON)
    parser.add_argument("--lat", type=float, default=REDMOND_LAT)
    parser.add_argument("--half-deg", type=float, default=HALF_DEG)
    parser.add_argument("--folder", default=DRIVE_FOLDER)
    parser.add_argument(
        "--download-dir",
        type=Path,
        help=(
            "Download GeoTIFFs directly from Earth Engine into this directory "
            "instead of starting Google Drive export tasks. Direct downloads "
            "only work for small AOIs."
        ),
    )
    args = parser.parse_args()

    if args.auth:
        ee.Authenticate()

    ee.Initialize(project=args.project)

    region = get_region(args.lon, args.lat, args.half_deg)

    s2_scene, n_candidates = pick_scene(region, args.start, args.end)
    dw_scene = get_matching_dw(s2_scene)

    s2_id = s2_scene.get("system:index").getInfo()
    s2_date = ee.Date(s2_scene.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    s2_cloud = s2_scene.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    spacecraft = s2_scene.get("SPACECRAFT_NAME").getInfo()

    # Use the source image's native UTM projection so we don't reproject to
    # EPSG:4326 on export. Pin everything to the B2 (10 m) grid.
    b2_proj = s2_scene.select("B2").projection()
    crs = b2_proj.crs().getInfo()  # e.g. "EPSG:32610" for UTM 10N

    print("Selected S2 L1C scene:")
    print(f"  candidates in window : {n_candidates}")
    print(f"  system:index         : {s2_id}")
    print(f"  date                 : {s2_date}")
    print(f"  cloud %              : {s2_cloud:.2f}")
    print(f"  spacecraft           : {spacecraft}")
    print(f"  native CRS           : {crs}")
    print(f"  matching DW          : {'yes' if dw_scene is not None else 'NO'}")

    if dw_scene is None:
        print(
            "ERROR: no Dynamic World prediction is available for this scene. "
            "DW skips heavily-clouded scenes; try a different date window.",
            file=sys.stderr,
        )
        sys.exit(1)

    dw_algorithm_version = dw_scene.get("dynamicworld_algorithm_version").getInfo()
    qa_algorithm_version = dw_scene.get("qa_algorithm_version").getInfo()
    print(f"  DW algorithm version : {dw_algorithm_version}")
    print(f"  DW QA version        : {qa_algorithm_version}")

    # S2 9 DW-input bands: bilinear-resample 20 m bands to the 10 m B2 grid.
    s2_export = (
        s2_scene.select(DW_INPUT_BANDS)
        .resample("bilinear")
        .reproject(crs=crs, scale=EXPORT_SCALE_M)
    )

    # DW outputs split: integer label vs float32 probabilities.
    dw_label_export = dw_scene.select(["label"]).toUint8()
    dw_prob_export = dw_scene.select(DW_PROB_BANDS).toFloat()

    s2_desc = f"S2_L1C_DWbands_{s2_id}"
    dw_label_desc = f"DynamicWorld_V1_{s2_id}_label"
    dw_prob_desc = f"DynamicWorld_V1_{s2_id}_probs"

    if args.download_dir is not None:
        print()
        print("Downloading directly from Earth Engine:")
        s2_path = download_image(s2_export, s2_desc, region, crs, args.download_dir)
        dw_label_path = download_image(
            dw_label_export, dw_label_desc, region, crs, args.download_dir
        )
        dw_prob_path = download_image(
            dw_prob_export, dw_prob_desc, region, crs, args.download_dir
        )
        print(f"  - {s2_path}")
        print(f"  - {dw_label_path}")
        print(f"  - {dw_prob_path}")
        return

    common = dict(
        folder=args.folder,
        region=region,
        scale=EXPORT_SCALE_M,
        crs=crs,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )

    s2_task = ee.batch.Export.image.toDrive(
        image=s2_export,
        description=s2_desc,
        fileNamePrefix=s2_desc,
        **common,
    )
    dw_label_task = ee.batch.Export.image.toDrive(
        image=dw_label_export,
        description=dw_label_desc,
        fileNamePrefix=dw_label_desc,
        **common,
    )
    dw_prob_task = ee.batch.Export.image.toDrive(
        image=dw_prob_export,
        description=dw_prob_desc,
        fileNamePrefix=dw_prob_desc,
        **common,
    )

    s2_task.start()
    dw_label_task.start()
    dw_prob_task.start()

    print()
    print("Started Drive export tasks:")
    print(f"  - {s2_desc}        (task id: {s2_task.id})")
    print(f"  - {dw_label_desc}  (task id: {dw_label_task.id})")
    print(f"  - {dw_prob_desc}   (task id: {dw_prob_task.id})")
    print(f"All will appear in your Drive under folder: {args.folder}")
    print()
    print("Track progress: https://code.earthengine.google.com/tasks")
    print("Or run:         earthengine task list")


if __name__ == "__main__":
    main()
