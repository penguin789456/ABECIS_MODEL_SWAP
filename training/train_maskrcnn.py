"""
Mask R-CNN training script using Detectron2.

Runs inside the CrackPre conda environment.

Prerequisites:
    1. Run scripts/prepare_dataset.py first to generate:
       - data/splits/{train,val,test}.txt
       - outputs/coco_annotations/{train,val,test}.json
    2. conda activate CrackPre
    3. pip install -e detectron2 --no-build-isolation

Usage:
    python training/train_maskrcnn.py --config configs/maskrcnn.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(cfg: dict) -> None:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.model_zoo import model_zoo

    ds_cfg = cfg["dataset"]
    d2_cfg = cfg["dataset_detectron2"]
    tr_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]

    coco_dir = Path(ds_cfg["coco_annotations_dir"])
    rgb_dir = str(Path(ds_cfg["root"]) / "rgb")

    # Register train / val datasets
    for split in ("train", "val"):
        register_coco_instances(
            f"crack_{split}",
            {},
            str(coco_dir / f"{split}.json"),
            rgb_dir,
        )

    # Build Detectron2 config
    d2 = get_cfg()
    d2.merge_from_file(model_zoo.get_config_file(cfg["model"]["config_file"]))
    d2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg["model"]["config_file"])

    d2.DATASETS.TRAIN = ("crack_train",)
    d2.DATASETS.TEST = ("crack_val",)

    d2.SOLVER.BASE_LR = tr_cfg["lr"]
    d2.SOLVER.MOMENTUM = tr_cfg.get("momentum", 0.9)
    d2.SOLVER.WEIGHT_DECAY = tr_cfg["weight_decay"]
    d2.SOLVER.MAX_ITER = tr_cfg["max_iter"]
    d2.SOLVER.WARMUP_ITERS = tr_cfg["warmup_iters"]
    d2.SOLVER.STEPS = tuple(tr_cfg["lr_steps"])
    d2.SOLVER.IMS_PER_BATCH = tr_cfg["batch_size"]

    d2.MODEL.ROI_HEADS.NUM_CLASSES = 1  # binary: crack
    d2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    d2.INPUT.MASK_FORMAT = d2_cfg.get("mask_format", "bitmask")
    d2.SEED = tr_cfg["seed"]

    out_dir = Path(ckpt_cfg["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    d2.OUTPUT_DIR = str(out_dir)

    d2.TEST.EVAL_PERIOD = ckpt_cfg.get("eval_period", 500)

    trainer = DefaultTrainer(d2)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"\nTraining complete. Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN with Detectron2")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
