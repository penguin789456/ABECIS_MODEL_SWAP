# -*- coding: utf-8 -*-
"""
Merge external crack datasets (DeepCrack / CRACK500) into a combined
dataset directory, then regenerate train/val splits.

Test set is FROZEN — only the original 70 test images from the Mendeley
dataset are used for final evaluation. All external data goes to train only.

Supported source formats
------------------------
DeepCrack (https://github.com/yhlleo/DeepCrack)
    Variant A  data/{train,test}/image/  +  data/{train,test}/label/
    Variant B  {train,test}_img/  +  {train,test}_lab/

CRACK500 (https://github.com/fyangneil/pavement-crack-detection)
    {train,val,test}data/ — each folder has *.jpg (image) + *.png (mask)
    with the same stem.

Generic
    --generic_img_dir / --generic_mask_dir
    Any two directories with matching filenames (supports jpg/png/bmp).

Usage (run from project root, CrackSeg env)
-------------------------------------------
# 1. Merge ONLY DeepCrack
python scripts/prepare_external.py \\
    --deepcrack_dir /path/to/DeepCrack \\
    --output_root   concreteCrackSegmentationDataset_merged

# 2. Merge ONLY CRACK500
python scripts/prepare_external.py \\
    --crack500_dir /path/to/CRACK500 \\
    --output_root  concreteCrackSegmentationDataset_merged

# 3. Both at once
python scripts/prepare_external.py \\
    --deepcrack_dir /path/to/DeepCrack \\
    --crack500_dir  /path/to/CRACK500 \\
    --output_root   concreteCrackSegmentationDataset_merged

# 4. After merging: retrain PP-LiteSeg on the expanded dataset
#    python scripts/precompute_patches.py --config configs/final/ppliteseg.yaml \\
#        --dataset_root concreteCrackSegmentationDataset_merged \\
#        --splits_dir   data/splits_merged
#    python training/train_crackseg.py --config configs/final/ppliteseg.yaml \\
#        [edit yaml to point dataset.root and dataset.splits_dir to _merged versions]

Output
------
<output_root>/
    rgb/   — all colour images (JPEG)
    BW/    — all binary masks  (JPEG, white=crack)

data/splits_merged/
    train.txt  — original train + all external images
    val.txt    — original val   (unchanged)
    test.txt   — original test  (FROZEN, unchanged)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _stem_index(directory: Path) -> dict[str, Path]:
    """Return {stem.lower(): path} for all image files in a directory."""
    return {
        p.stem.lower(): p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    }


def _save_rgb(img_path: Path, dst_dir: Path, stem: str) -> str:
    """Copy/convert an image to <dst_dir>/rgb/<stem>.jpg. Returns stem."""
    img = Image.open(img_path).convert("RGB")
    out = dst_dir / "rgb" / f"{stem}.jpg"
    img.save(out, "JPEG", quality=95)
    return stem


def _save_mask(mask_path: Path, dst_dir: Path, stem: str) -> None:
    """
    Convert a mask to binary (white=crack) and save as
    <dst_dir>/BW/<stem>.jpg.

    Handles:
    - Greyscale masks (threshold at 127)
    - RGB masks where crack is a specific colour (treated as non-zero)
    - Already-binary masks
    """
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    # Binarise: any non-zero pixel = crack
    binary = ((arr > 10).astype(np.uint8)) * 255
    out = dst_dir / "BW" / f"{stem}.jpg"
    Image.fromarray(binary).save(out, "JPEG", quality=95)


def _init_output(output_root: Path) -> None:
    (output_root / "rgb").mkdir(parents=True, exist_ok=True)
    (output_root / "BW").mkdir(parents=True, exist_ok=True)


def _copy_original(original_root: Path, output_root: Path) -> list[str]:
    """
    Copy all original Mendeley dataset images → output_root.
    Returns list of stems copied.
    """
    rgb_dir = original_root / "rgb"
    bw_dir = original_root / "BW"
    rgb_idx = _stem_index(rgb_dir)
    copied = []
    for stem, rgb_p in rgb_idx.items():
        bw_p = bw_dir / f"{stem}.jpg"
        if not bw_p.exists():
            bw_p = bw_dir / f"{stem}.JPG"
        if not bw_p.exists():
            continue
        dst_rgb = output_root / "rgb" / f"{stem}.jpg"
        dst_bw  = output_root / "BW"  / f"{stem}.jpg"
        if not dst_rgb.exists():
            shutil.copy2(rgb_p, dst_rgb)
        if not dst_bw.exists():
            shutil.copy2(bw_p, dst_bw)
        copied.append(stem)
    print(f"  [original] copied {len(copied)} pairs")
    return copied


# ── DeepCrack ──────────────────────────────────────────────────────────────────

def _locate_deepcrack_splits(root: Path) -> list[tuple[Path, Path]]:
    """
    Try to find (img_dir, label_dir) pairs for DeepCrack.
    Returns a list of (image_folder, label_folder) to process.
    """
    candidates: list[tuple[Path, Path]] = []

    # Variant A: data/{train,test}/image/ + data/{train,test}/label/
    for split in ("train", "test"):
        img_d = root / "data" / split / "image"
        lbl_d = root / "data" / split / "label"
        if img_d.exists() and lbl_d.exists():
            candidates.append((img_d, lbl_d))

    if candidates:
        return candidates

    # Variant B: {train,test}_img/ + {train,test}_lab/
    for split in ("train", "test"):
        img_d = root / f"{split}_img"
        lbl_d = root / f"{split}_lab"
        if img_d.exists() and lbl_d.exists():
            candidates.append((img_d, lbl_d))

    if candidates:
        return candidates

    # Variant C: flat image/ + label/ at root
    img_d = root / "image"
    lbl_d = root / "label"
    if img_d.exists() and lbl_d.exists():
        return [(img_d, lbl_d)]

    # Variant D: flat img/ + gt/ at root
    img_d = root / "img"
    lbl_d = root / "gt"
    if img_d.exists() and lbl_d.exists():
        return [(img_d, lbl_d)]

    raise FileNotFoundError(
        f"Cannot locate DeepCrack image/label directories inside {root}.\n"
        "Expected one of:\n"
        "  data/train/image + data/train/label\n"
        "  train_img + train_lab\n"
        "  image + label\n"
        "  img + gt"
    )


def merge_deepcrack(root: Path, output_root: Path, prefix: str = "dc") -> list[str]:
    print(f"\n[DeepCrack] reading from: {root}")
    pairs = _locate_deepcrack_splits(root)
    added: list[str] = []
    for img_dir, lbl_dir in pairs:
        img_idx = _stem_index(img_dir)
        lbl_idx = _stem_index(lbl_dir)
        matched = set(img_idx) & set(lbl_idx)
        print(f"  {img_dir.name}: {len(matched)} matched pairs")
        for stem in sorted(matched):
            new_stem = f"{prefix}_{stem}"
            _save_rgb(img_idx[stem], output_root, new_stem)
            _save_mask(lbl_idx[stem], output_root, new_stem)
            added.append(new_stem)
    print(f"  [DeepCrack] total added: {len(added)}")
    return added


# ── CRACK500 ───────────────────────────────────────────────────────────────────

def _locate_crack500_splits(root: Path) -> list[tuple[Path, Path | None]]:
    """
    Returns list of (img_dir, mask_dir_or_None).
    If mask_dir is None, masks are co-located with images (same stem, .png).
    """
    candidates: list[tuple[Path, Path | None]] = []

    # Format A: {train,val,test}data/ — images (.jpg) and masks (.png) co-located
    for split in ("train", "val", "test"):
        d = root / f"{split}data"
        if d.exists():
            candidates.append((d, None))

    if candidates:
        return candidates

    # Format B: separate img/ and mask/ subdirectories per split
    for split in ("train", "val", "test"):
        img_d  = root / split / "img"
        mask_d = root / split / "mask"
        if img_d.exists() and mask_d.exists():
            candidates.append((img_d, mask_d))

    if candidates:
        return candidates

    # Format C: flat directories
    img_d  = root / "img"
    mask_d = root / "mask"
    if img_d.exists() and mask_d.exists():
        return [(img_d, mask_d)]

    # Format D: everything flat in root
    if any(p.suffix.lower() == ".jpg" for p in root.iterdir() if p.is_file()):
        return [(root, None)]

    raise FileNotFoundError(
        f"Cannot locate CRACK500 image directories inside {root}.\n"
        "Expected one of:\n"
        "  traindata/ valdata/ testdata/  (co-located jpg+png)\n"
        "  train/img + train/mask\n"
        "  img/ + mask/"
    )


def merge_crack500(root: Path, output_root: Path, prefix: str = "c5") -> list[str]:
    print(f"\n[CRACK500] reading from: {root}")
    splits = _locate_crack500_splits(root)
    added: list[str] = []
    for img_dir, mask_dir in splits:
        img_idx = {
            p.stem.lower(): p
            for p in sorted(img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
        }
        if mask_dir is not None:
            mask_idx = _stem_index(mask_dir)
        else:
            # Co-located: mask has same stem but .png extension
            mask_idx = {
                p.stem.lower(): p
                for p in sorted(img_dir.iterdir())
                if p.is_file() and p.suffix.lower() == ".png"
            }

        matched = set(img_idx) & set(mask_idx)
        print(f"  {img_dir.name}: {len(matched)} matched pairs")
        for stem in sorted(matched):
            new_stem = f"{prefix}_{stem}"
            _save_rgb(img_idx[stem], output_root, new_stem)
            _save_mask(mask_idx[stem], output_root, new_stem)
            added.append(new_stem)
    print(f"  [CRACK500] total added: {len(added)}")
    return added


# ── Generic ────────────────────────────────────────────────────────────────────

def merge_generic(
    img_dir: Path, mask_dir: Path, output_root: Path, prefix: str = "ext"
) -> list[str]:
    print(f"\n[Generic] img={img_dir}  mask={mask_dir}")
    img_idx  = _stem_index(img_dir)
    mask_idx = _stem_index(mask_dir)
    matched  = set(img_idx) & set(mask_idx)
    print(f"  matched pairs: {len(matched)}")
    added: list[str] = []
    for stem in sorted(matched):
        new_stem = f"{prefix}_{stem}"
        _save_rgb(img_idx[stem], output_root, new_stem)
        _save_mask(mask_idx[stem], output_root, new_stem)
        added.append(new_stem)
    print(f"  [Generic] total added: {len(added)}")
    return added


# ── Splits ─────────────────────────────────────────────────────────────────────

def regenerate_splits(
    original_splits_dir: Path,
    output_splits_dir: Path,
    external_stems: list[str],
    seed: int = 42,
) -> None:
    """
    Strategy:
    - test.txt   : FROZEN (copied from original, unchanged)
    - val.txt    : FROZEN (copied from original, unchanged)
    - train.txt  : original train + ALL external stems
    """
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    # Frozen: test and val
    for split in ("test", "val"):
        src = original_splits_dir / f"{split}.txt"
        dst = output_splits_dir / f"{split}.txt"
        shutil.copy2(src, dst)
        n = len(src.read_text().strip().splitlines())
        print(f"  {split}.txt  : {n} images (FROZEN)")

    # Train: original + external
    orig_train = (original_splits_dir / "train.txt").read_text().strip().splitlines()
    orig_train = [s.strip() for s in orig_train if s.strip()]

    # Shuffle external stems so they're not all at the end
    rng = random.Random(seed)
    ext = list(external_stems)
    rng.shuffle(ext)

    combined = orig_train + ext
    (output_splits_dir / "train.txt").write_text("\n".join(combined) + "\n")
    print(f"  train.txt  : {len(orig_train)} original + {len(ext)} external = {len(combined)} total")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge external crack datasets into combined training data"
    )
    parser.add_argument(
        "--original_root", default="concreteCrackSegmentationDataset",
        help="Path to original Mendeley dataset (default: concreteCrackSegmentationDataset)"
    )
    parser.add_argument(
        "--output_root", default="concreteCrackSegmentationDataset_merged",
        help="Output directory for merged dataset"
    )
    parser.add_argument(
        "--deepcrack_dir", default=None,
        help="Path to downloaded DeepCrack dataset directory"
    )
    parser.add_argument(
        "--crack500_dir", default=None,
        help="Path to downloaded CRACK500 dataset directory"
    )
    parser.add_argument(
        "--generic_img_dir", default=None,
        help="Generic: path to folder containing colour images"
    )
    parser.add_argument(
        "--generic_mask_dir", default=None,
        help="Generic: path to folder containing binary masks"
    )
    parser.add_argument(
        "--generic_prefix", default="ext",
        help="Filename prefix for generic dataset images (default: ext)"
    )
    # CFD (Crack Forest Dataset)
    parser.add_argument(
        "--cfd_img_dir", default=None,
        help="CFD: path to cfd_image/ folder (extracted from cfd_image.zip)"
    )
    parser.add_argument(
        "--cfd_mask_dir", default=None,
        help="CFD: path to seg_gt/ folder (extracted from cfd_gt.zip)"
    )
    # GAPS384
    parser.add_argument(
        "--gaps384_img_dir", default=None,
        help="GAPS384: path to croppedimg/ folder (extracted from croppedimg.zip)"
    )
    parser.add_argument(
        "--gaps384_mask_dir", default=None,
        help="GAPS384: path to croppedgt/ folder (extracted from croppedgt.zip)"
    )
    parser.add_argument(
        "--original_splits_dir", default="data/splits",
        help="Existing split files to use as base (default: data/splits)"
    )
    parser.add_argument(
        "--output_splits_dir", default="data/splits_v1",
        help="Where to write merged split files (default: data/splits_v1)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = PROJECT_ROOT / args.output_root
    _init_output(output_root)

    print("=" * 60)
    print(f"Output directory : {output_root}")
    print("=" * 60)

    # 1. Copy original dataset
    print("\n[Step 1] Copy original Mendeley dataset")
    original_root = PROJECT_ROOT / args.original_root
    _copy_original(original_root, output_root)

    # 2. Merge external datasets
    external_stems: list[str] = []

    if args.deepcrack_dir:
        stems = merge_deepcrack(Path(args.deepcrack_dir), output_root, prefix="dc")
        external_stems.extend(stems)

    if args.crack500_dir:
        stems = merge_crack500(Path(args.crack500_dir), output_root, prefix="c5")
        external_stems.extend(stems)

    if args.generic_img_dir and args.generic_mask_dir:
        stems = merge_generic(
            Path(args.generic_img_dir),
            Path(args.generic_mask_dir),
            output_root,
            prefix=args.generic_prefix,
        )
        external_stems.extend(stems)

    if args.cfd_img_dir and args.cfd_mask_dir:
        stems = merge_generic(
            Path(args.cfd_img_dir),
            Path(args.cfd_mask_dir),
            output_root,
            prefix="cfd",
        )
        external_stems.extend(stems)

    if args.gaps384_img_dir and args.gaps384_mask_dir:
        stems = merge_generic(
            Path(args.gaps384_img_dir),
            Path(args.gaps384_mask_dir),
            output_root,
            prefix="gaps",
        )
        external_stems.extend(stems)

    if not external_stems:
        print("\nNo external datasets provided. Only original data copied.")
        print("Provide --deepcrack_dir and/or --crack500_dir to add external data.")

    # 3. Regenerate splits
    print("\n[Step 3] Regenerate splits")
    original_splits_dir = PROJECT_ROOT / args.original_splits_dir
    output_splits_dir   = PROJECT_ROOT / args.output_splits_dir
    regenerate_splits(original_splits_dir, output_splits_dir, external_stems, args.seed)

    # 4. Summary
    n_rgb  = len(list((output_root / "rgb").iterdir()))
    n_bw   = len(list((output_root / "BW").iterdir()))
    print(f"\n{'=' * 60}")
    print(f"  Merged dataset : {output_root}")
    print(f"  rgb/ images    : {n_rgb}")
    print(f"  BW/ masks      : {n_bw}")
    print(f"  Split files    : {output_splits_dir}")
    print(f"{'=' * 60}")
    print("""
Next steps:
  1. Precompute patches for the merged dataset:
       python scripts/precompute_patches.py \\
           --config configs/final/ppliteseg.yaml \\
           --dataset_root concreteCrackSegmentationDataset_merged \\
           --splits_dir   data/splits_merged \\
           --output_dir   data/patches_merged

  2. Update your config yaml to point to the merged data:
       dataset:
         root:        concreteCrackSegmentationDataset_merged
         splits_dir:  data/splits_merged
         precomputed_dir: data/patches_merged

  3. Retrain:
       python training/train_crackseg.py --config configs/final/ppliteseg.yaml
""")


if __name__ == "__main__":
    main()
