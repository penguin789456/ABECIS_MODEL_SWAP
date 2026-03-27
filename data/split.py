"""
Generate deterministic 70 / 15 / 15 train / val / test splits.

Usage:
    python data/split.py --dataset_root concreteCrackSegmentationDataset \
                         --output_dir data/splits --seed 42

Output:
    data/splits/train.txt   (~320 stems)
    data/splits/val.txt     (~69 stems)
    data/splits/test.txt    (~69 stems)

The generated .txt files should be committed to git so all models
are evaluated on the same frozen test set.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def generate_splits(
    dataset_root: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    rgb_dir = Path(dataset_root) / "rgb"
    bw_dir = Path(dataset_root) / "BW"

    # Collect stems from rgb/ (case-insensitive extension matching)
    stems: list[str] = []
    for p in sorted(rgb_dir.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # Verify a matching BW mask exists
            bw_path = bw_dir / f"{p.stem}.jpg"
            if not bw_path.exists():
                bw_path = bw_dir / f"{p.stem}.JPG"
            if bw_path.exists():
                stems.append(p.stem)
            else:
                print(f"WARNING: no BW mask found for {p.name} — skipped")

    total = len(stems)
    print(f"Valid image pairs found: {total}")

    rng = random.Random(seed)
    rng.shuffle(stems)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_stems = stems[:n_train]
    val_stems = stems[n_train : n_train + n_val]
    test_stems = stems[n_train + n_val :]

    print(f"Split sizes  →  train: {len(train_stems)}  val: {len(val_stems)}  test: {len(test_stems)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        (out / f"{name}.txt").write_text("\n".join(split) + "\n")
        print(f"Written: {out / (name + '.txt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate train/val/test split files")
    parser.add_argument("--dataset_root", default="concreteCrackSegmentationDataset")
    parser.add_argument("--output_dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_splits(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
