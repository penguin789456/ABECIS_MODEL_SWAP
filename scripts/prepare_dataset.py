"""
Dataset preparation script — must be run before any training.

Steps:
    1. Validate all 458 rgb/BW image pairs (handles mixed .jpg/.JPG)
    2. Generate data/splits/{train,val,test}.txt (70/15/15, seed 42)
    3. Convert binary BW masks to COCO-format JSON for Detectron2
       → outputs/coco_annotations/{train,val,test}.json

Usage:
    conda activate CrackSeg
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dataset_root concreteCrackSegmentationDataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.split import generate_splits


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_pairs(dataset_root: str) -> list[str]:
    """
    Check every rgb image has a matching BW mask.
    Returns a sorted list of valid stems.
    """
    rgb_dir = Path(dataset_root) / "rgb"
    bw_dir = Path(dataset_root) / "BW"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb/ directory not found: {rgb_dir}")
    if not bw_dir.exists():
        raise FileNotFoundError(f"BW/ directory not found: {bw_dir}")

    valid, missing = [], []
    for p in sorted(rgb_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        bw_p = bw_dir / f"{p.stem}.jpg"
        if not bw_p.exists():
            bw_p = bw_dir / f"{p.stem}.JPG"
        if bw_p.exists():
            valid.append(p.stem)
        else:
            missing.append(p.name)

    if missing:
        print(f"WARNING: {len(missing)} images have no BW mask:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print(f"Valid pairs: {len(valid)}  |  Missing masks: {len(missing)}")
    return valid


# ---------------------------------------------------------------------------
# COCO annotation conversion
# ---------------------------------------------------------------------------

def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a 2D binary mask as COCO RLE (using pycocotools if available)."""
    try:
        import pycocotools.mask as mask_util
        rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except ImportError:
        # Fallback: simple uncompressed polygon (bbox only, no fine mask)
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if not rows.any():
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        w, h = int(cmax - cmin + 1), int(rmax - rmin + 1)
        # Return as bitmask (uncompressed) — Detectron2 can handle both
        return {
            "counts": binary_mask[rmin:rmax+1, cmin:cmax+1].ravel().tolist(),
            "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        }


def build_coco_json(
    stems: list[str],
    dataset_root: str,
    split_name: str,
    out_path: Path,
) -> None:
    """Convert BW masks to a COCO-format JSON file."""
    rgb_dir = Path(dataset_root) / "rgb"
    bw_dir = Path(dataset_root) / "BW"

    rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    images, annotations = [], []
    ann_id = 1

    for img_id, stem in enumerate(stems, start=1):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            continue
        bw_p = bw_dir / f"{stem}.jpg"
        if not bw_p.exists():
            bw_p = bw_dir / f"{stem}.JPG"

        W, H = Image.open(rgb_p).size
        images.append({
            "id": img_id,
            "file_name": rgb_p.name,
            "width": W,
            "height": H,
        })

        bw_arr = np.array(Image.open(bw_p).convert("L")) > 127
        if not bw_arr.any():
            continue  # No crack — skip annotation

        rle = mask_to_rle(bw_arr)
        if rle is None:
            continue

        # Bounding box from mask
        rows = np.any(bw_arr, axis=1)
        cols = np.any(bw_arr, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
        area = int(bw_arr.sum())

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": bbox,
            "area": area,
            "iscrowd": 1,  # 1 = RLE format in COCO
        })
        ann_id += 1

    coco = {
        "info": {"description": f"Concrete Crack Segmentation — {split_name}"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"Written COCO JSON: {out_path}  ({len(images)} images, {len(annotations)} annotations)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dataset_root: str, splits_dir: str, coco_dir: str) -> None:
    print("=" * 60)
    print("Step 1: Validate image pairs")
    print("=" * 60)
    valid_stems = validate_pairs(dataset_root)

    print("\n" + "=" * 60)
    print("Step 2: Generate train/val/test splits")
    print("=" * 60)
    generate_splits(dataset_root=dataset_root, output_dir=splits_dir, seed=42)

    print("\n" + "=" * 60)
    print("Step 3: Build COCO annotation JSONs for Detectron2")
    print("=" * 60)
    splits_path = Path(splits_dir)
    coco_path = Path(coco_dir)

    for split_name in ("train", "val", "test"):
        txt_file = splits_path / f"{split_name}.txt"
        with open(txt_file) as f:
            stems = [ln.strip() for ln in f if ln.strip()]
        build_coco_json(stems, dataset_root, split_name, coco_path / f"{split_name}.json")

    print("\nDataset preparation complete.")
    print(f"  Split files : {splits_dir}/{{train,val,test}}.txt")
    print(f"  COCO JSONs  : {coco_dir}/{{train,val,test}}.json")
    print("\nNext step: commit data/splits/ to git, then run training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare crack segmentation dataset")
    parser.add_argument("--dataset_root", default="concreteCrackSegmentationDataset")
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--coco_dir", default="outputs/coco_annotations")
    args = parser.parse_args()

    main(args.dataset_root, args.splits_dir, args.coco_dir)
