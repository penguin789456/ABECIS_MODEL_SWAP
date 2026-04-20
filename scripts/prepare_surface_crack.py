# -*- coding: utf-8 -*-
"""
Merge Surface Crack Detection (SCD) ABECIS pseudo-labels into dataset_merged_v3.

Source data
-----------
Surface Crack Detection (Özgenel, Mendeley):
  <scd_dir>/Positive/                       ← 20,000 cracked images (227×227)
  <scd_dir>/Positive/Crack_Analysis/Confident/  ← ABECIS high-confidence masks
  <scd_dir>/Positive/Crack_Analysis/Possible/   ← ABECIS low-confidence masks

Processing pipeline
-------------------
1. Copy dataset_merged_v2 → dataset_merged_v3  (v2 unchanged)
2. For each SCD image with an ABECIS mask:
     • Resize image  227×227 → 512×512 (LANCZOS)
     • Binarise mask  (pixel > MASK_THRESHOLD → 255)
     • Resize mask   227×227 → 512×512 (NEAREST — no interpolation on labels)
     • Filter: skip if crack% < MIN_CRACK_PCT or > MAX_CRACK_PCT
     • Save as scd_{stem}.jpg  in v3/rgb/ and v3/BW/
3. Generate splits_v3:
     - test.txt  → FROZEN (identical to splits_v2/test.txt)
     - val.txt   → unchanged (Mendeley val only)
     - train.txt → splits_v2/train.txt + all new SCD stems

Usage (run from project root, CrackSeg env)
-------------------------------------------
python scripts/prepare_surface_crack.py \\
    --scd_dir "H:\\ChihleeMaster\\Surface Crack Detection" \\
    --confidence confident
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SIZE      = (512, 512)   # resize all SCD images/masks to this
MASK_THRESHOLD   = 50           # binarise: >50 → crack  (ignores JPEG artifacts ≤~30)
MIN_CRACK_PCT    = 0.005        # 0.5 % — discard near-empty masks (ABECIS miss)
MAX_CRACK_PCT    = 0.60         # 60 % — discard near-full masks (likely false alarm)
SCD_PREFIX       = "scd"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _copy_v2_to_v3(v2_root: Path, v3_root: Path) -> list[str]:
    """
    Copy every rgb/BW pair from v2 → v3.
    Skips files that already exist (safe to re-run).
    Returns list of stems copied.
    """
    (v3_root / "rgb").mkdir(parents=True, exist_ok=True)
    (v3_root / "BW").mkdir(parents=True, exist_ok=True)

    stems: list[str] = []
    for rgb_p in tqdm(sorted((v2_root / "rgb").iterdir()), desc="copy v2→v3"):
        if rgb_p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        stem = rgb_p.stem
        # Prefer .jpg mask, fall back to .png
        bw_p = v2_root / "BW" / f"{stem}.jpg"
        if not bw_p.exists():
            bw_p = v2_root / "BW" / f"{stem}.png"
        if not bw_p.exists():
            continue

        dst_rgb = v3_root / "rgb" / rgb_p.name
        dst_bw  = v3_root / "BW"  / bw_p.name
        if not dst_rgb.exists():
            shutil.copy2(rgb_p, dst_rgb)
        if not dst_bw.exists():
            shutil.copy2(bw_p, dst_bw)
        stems.append(stem)

    print(f"  [v2 copy] {len(stems)} pairs → {v3_root}")
    return stems


def _process_scd(
    scd_dir: Path,
    v3_root: Path,
    confidence: str = "confident",
) -> list[str]:
    """
    Convert SCD ABECIS outputs to v3 training pairs.
    Returns list of new stems added.
    """
    (v3_root / "rgb").mkdir(parents=True, exist_ok=True)
    (v3_root / "BW").mkdir(parents=True, exist_ok=True)

    positive_dir = scd_dir / "Positive"

    # Select confidence tiers
    conf_folders: list[tuple[Path, str]] = []
    if confidence in ("confident", "both"):
        conf_folders.append(
            (positive_dir / "Crack_Analysis" / "Confident", "")
        )
    if confidence in ("possible", "both"):
        # Use distinct prefix to avoid stem clashes when using both tiers
        conf_folders.append(
            (positive_dir / "Crack_Analysis" / "Possible", "p_")
        )

    added: list[str] = []
    skipped_notfound = 0
    skipped_filter   = 0

    for conf_dir, tier_pfx in conf_folders:
        if not conf_dir.exists():
            print(f"  [SCD] WARNING: {conf_dir} not found — skipping")
            continue

        mask_files = sorted(conf_dir.glob("*_mask.jpg"))
        print(f"\n  [SCD] {conf_dir.name}: {len(mask_files)} mask files")

        for mask_path in tqdm(mask_files, desc=f"  SCD {conf_dir.name[:10]}"):
            # "00006_mask.jpg" → stem "00006"
            stem     = mask_path.stem.replace("_mask", "")
            orig_img = positive_dir / f"{stem}.jpg"

            if not orig_img.exists():
                skipped_notfound += 1
                continue

            # ── Load & process mask ────────────────────────────────────────
            mask_arr = np.array(Image.open(mask_path).convert("L"))
            binary   = ((mask_arr > MASK_THRESHOLD).astype(np.uint8)) * 255

            # Quality filter
            crack_pct = binary.mean() / 255.0
            if crack_pct < MIN_CRACK_PCT or crack_pct > MAX_CRACK_PCT:
                skipped_filter += 1
                continue

            # ── Resize ────────────────────────────────────────────────────
            img_resized  = Image.open(orig_img).convert("RGB").resize(
                TARGET_SIZE, Image.LANCZOS
            )
            mask_resized = Image.fromarray(binary).resize(
                TARGET_SIZE, Image.NEAREST       # NEAREST preserves binary edges
            )

            # ── Save ──────────────────────────────────────────────────────
            new_stem = f"{SCD_PREFIX}_{tier_pfx}{stem}"
            img_resized.save(
                v3_root / "rgb" / f"{new_stem}.jpg", "JPEG", quality=95
            )
            mask_resized.save(
                v3_root / "BW"  / f"{new_stem}.jpg", "JPEG", quality=95
            )
            added.append(new_stem)

    print(
        f"\n  [SCD total] added={len(added)}, "
        f"not_found={skipped_notfound}, "
        f"filtered={skipped_filter}"
    )
    return added


def _generate_splits(
    v2_splits_dir: Path,
    v3_splits_dir: Path,
    new_stems: list[str],
) -> None:
    """
    Build splits_v3:
      test.txt  — FROZEN copy from v2
      val.txt   — copy from v2 (Mendeley val, no SCD)
      train.txt — v2 train + all SCD stems
    """
    v3_splits_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(v2_splits_dir / "test.txt", v3_splits_dir / "test.txt")
    shutil.copy2(v2_splits_dir / "val.txt",  v3_splits_dir / "val.txt")

    v2_train  = [
        ln.strip()
        for ln in (v2_splits_dir / "train.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    all_train = v2_train + new_stems
    (v3_splits_dir / "train.txt").write_text(
        "\n".join(all_train) + "\n", encoding="utf-8"
    )

    n_test = len((v3_splits_dir / "test.txt").read_text().strip().splitlines())
    n_val  = len((v3_splits_dir / "val.txt").read_text().strip().splitlines())
    print(f"\n  [splits_v3]")
    print(f"    train : {len(v2_train):>5} (v2)  +  {len(new_stems):>5} (SCD)  "
          f"= {len(all_train):>6}")
    print(f"    val   : {n_val:>5}  (frozen)")
    print(f"    test  : {n_test:>5}  (FROZEN)")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dataset_merged_v3 = v2 + SCD pseudo-labels"
    )
    parser.add_argument(
        "--scd_dir",
        default=r"H:\ChihleeMaster\Surface Crack Detection",
        help="Root of Surface Crack Detection dataset",
    )
    parser.add_argument(
        "--v2_root",
        default="dataset_merged_v2",
    )
    parser.add_argument(
        "--v2_splits_dir",
        default="data/splits_v2",
    )
    parser.add_argument(
        "--output_root",
        default="dataset_merged_v3",
    )
    parser.add_argument(
        "--output_splits_dir",
        default="data/splits_v3",
    )
    parser.add_argument(
        "--confidence",
        choices=["confident", "possible", "both"],
        default="confident",
        help="ABECIS confidence tier(s) to include",
    )
    args = parser.parse_args()

    scd_dir       = Path(args.scd_dir)
    v2_root       = PROJECT_ROOT / args.v2_root
    v2_splits     = PROJECT_ROOT / args.v2_splits_dir
    v3_root       = PROJECT_ROOT / args.output_root
    v3_splits     = PROJECT_ROOT / args.output_splits_dir

    print("=" * 60)
    print("Preparing dataset_merged_v3")
    print(f"  SCD dir    : {scd_dir}")
    print(f"  v2 root    : {v2_root}")
    print(f"  v3 root    : {v3_root}")
    print(f"  confidence : {args.confidence}")
    print(f"  target sz  : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  mask thr   : > {MASK_THRESHOLD}")
    print(f"  crack% ok  : {MIN_CRACK_PCT*100:.1f}% – {MAX_CRACK_PCT*100:.0f}%")
    print("=" * 60)

    # Step 1 – copy v2 → v3
    print("\n[Step 1] Copying dataset_merged_v2 → dataset_merged_v3 ...")
    v2_stems = _copy_v2_to_v3(v2_root, v3_root)

    # Step 2 – process SCD
    print(f"\n[Step 2] Converting SCD pseudo-labels ...")
    scd_stems = _process_scd(scd_dir, v3_root, confidence=args.confidence)

    # Step 3 – splits
    print(f"\n[Step 3] Generating splits_v3 ...")
    _generate_splits(v2_splits, v3_splits, scd_stems)

    total = len(v2_stems) + len(scd_stems)
    print(f"\n{'='*60}")
    print(f"dataset_merged_v3 ready  —  {total} total image pairs")
    print(f"  v2 base : {len(v2_stems)}")
    print(f"  SCD new : {len(scd_stems)}")
    print(f"\nNext:")
    print(f"  python scripts/precompute_patches.py "
          f"--config configs/final/ppliteseg_v3.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
