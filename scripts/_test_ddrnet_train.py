"""
Quick sanity check: 1 epoch of DDRNet training + validation.
Run: python scripts/_test_ddrnet_train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

cfg_text = """
model:
  name: ddrnet
  arch_type: DDRNet-23-slim
  num_classes: 1

training:
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4
  batch_size: 4
  epochs: 1
  warmup_epochs: 0
  scheduler: cosine
  seed: 42
  output_dir: outputs/_test_ddrnet

loss:
  type: bce_dice
  bce_weight: 0.5
  dice_weight: 0.5

dataset:
  root: concreteCrackSegmentationDataset
  splits_dir: data/splits
  patch_size: 512
  overlap: 128
  num_workers: 2
  prefetch_factor: 2
  pin_memory: false
  persistent_workers: false
  precomputed_dir: null
  oversample_positive: false

checkpoint:
  save_dir: outputs/_test_ddrnet/checkpoints
  save_best_metric: iou
  save_every_n_epochs: 999

evaluation:
  threshold: 0.5
"""

cfg = yaml.safe_load(cfg_text)

if __name__ == "__main__":
    from training.train_crackseg import train
    train(cfg)
    print("\n=== DDRNet 1-epoch test PASSED ===")
