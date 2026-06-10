"""
Mask R-CNN training script using Detectron2.

Runs inside the CrackPre conda environment.

Prerequisites:
    1. Run scripts/prepare_dataset.py first to generate COCO annotations.
    2. conda activate CrackPre
    3. pip install -e detectron2 --no-build-isolation

Usage:
    python training/train_maskrcnn.py --config configs/maskrcnn.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def write_run_info(cfg: dict, run_dir: Path) -> None:
    import torch

    tr_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    ck_cfg = cfg["checkpoint"]

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda or "?"
    else:
        gpu_name = "CPU"
        cuda_ver = "N/A"

    lines = [
        "=" * 60,
        "  Training Run Information",
        "=" * 60,
        f"  Generated     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Config        : {cfg.get('_config_path', 'N/A')}",
        f"  Run ID        : {run_dir.name}",
        f"  Run Directory : {run_dir}",
        "",
        "  Model",
        f"  name          : {model_cfg.get('name', 'maskrcnn')}",
        f"  config_file   : {model_cfg.get('config_file', '?')}",
        "",
        "  Training",
        f"  optimizer     : {tr_cfg.get('optimizer', 'sgd')}",
        f"  lr            : {tr_cfg['lr']}",
        f"  momentum      : {tr_cfg.get('momentum', 0.9)}",
        f"  weight_decay  : {tr_cfg['weight_decay']}",
        f"  batch_size    : {tr_cfg['batch_size']}",
        f"  max_iter      : {tr_cfg['max_iter']}",
        f"  warmup_iters  : {tr_cfg['warmup_iters']}",
        f"  lr_steps      : {tr_cfg['lr_steps']}",
        f"  seed          : {tr_cfg.get('seed', 42)}",
        "",
        "  Dataset",
        f"  root          : {ds_cfg.get('root', '?')}",
        f"  splits_dir    : {ds_cfg.get('splits_dir', '?')}",
        f"  coco_ann_dir  : {ds_cfg.get('coco_annotations_dir', '?')}",
        "",
        "  Checkpoint",
        f"  save_dir      : {ck_cfg.get('save_dir', '?')}",
        f"  eval_period   : {ck_cfg.get('eval_period', 0)} iter",
        "",
        "  System",
        f"  hostname      : {socket.gethostname()}",
        f"  platform      : {platform.platform()}",
        f"  Python        : {platform.python_version()}",
        f"  CUDA          : {cuda_ver}",
        f"  GPU           : {gpu_name}",
        "=" * 60,
    ]

    info_path = run_dir / "train_info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Train info      -> {info_path}")


def export_detectron2_metrics(metrics_path: Path, csv_path: Path) -> None:
    if not metrics_path.exists():
        print(f"metrics.json not found, skip CSV export: {metrics_path}")
        return

    rows: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print(f"No metrics rows found in {metrics_path}")
        return

    fieldnames = [
        "iteration",
        "total_loss",
        "loss_cls",
        "loss_box_reg",
        "loss_mask",
        "loss_rpn_cls",
        "loss_rpn_loc",
        "lr",
        "time",
        "data_time",
        "eta_seconds",
        "max_mem",
        "bbox/AP",
        "bbox/AP50",
        "segm/AP",
        "segm/AP50",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"CSV log         -> {csv_path}")


def main(cfg: dict) -> None:
    from detectron2.config import get_cfg
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator
    from detectron2.model_zoo import model_zoo

    ds_cfg = cfg["dataset"]
    d2_cfg = cfg["dataset_detectron2"]
    tr_cfg = cfg["training"]
    ck_cfg = cfg["checkpoint"]

    coco_dir = Path(ds_cfg["coco_annotations_dir"])
    rgb_dir = str(Path(ds_cfg["root"]) / "rgb")

    for split in ("train", "val"):
        register_coco_instances(
            f"crack_{split}",
            {},
            str(coco_dir / f"{split}.json"),
            rgb_dir,
        )

    stable_out_dir = Path(ck_cfg["save_dir"])
    stable_out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = stable_out_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    write_run_info(cfg, run_dir)

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

    d2.MODEL.ROI_HEADS.NUM_CLASSES = 1
    d2.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    d2.INPUT.MASK_FORMAT = d2_cfg.get("mask_format", "bitmask")
    d2.SEED = tr_cfg["seed"]
    d2.OUTPUT_DIR = str(run_dir)
    d2.TEST.EVAL_PERIOD = ck_cfg.get("eval_period", 500)

    class CrackTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            output_folder = output_folder or str(run_dir / "inference")
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    trainer = CrackTrainer(d2)
    trainer.resume_or_load(resume=False)
    trainer.train()

    export_detectron2_metrics(run_dir / "metrics.json", run_dir / "train_log.csv")

    final_model = run_dir / "model_final.pth"
    if final_model.exists():
        shutil.copy2(final_model, stable_out_dir / "best.pth")

    print(f"\nTraining complete. Run directory: {run_dir}")
    print(f"Stable best     : {stable_out_dir / 'best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN with Detectron2")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(Path(args.config).resolve())

    main(cfg)
