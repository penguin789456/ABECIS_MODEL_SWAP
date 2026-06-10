"""
Mask R-CNN training using torchvision without Detectron2.

Runs inside the CrackSeg conda environment.

Usage:
    conda activate CrackSeg
    python training/train_maskrcnn_tv.py --config configs/maskrcnn_tv.yaml
"""

from __future__ import annotations

import argparse
import csv
import platform
import socket
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_instance import CrackInstanceDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_maskrcnn(num_classes: int = 2, pretrained: bool = True):
    """Build Mask R-CNN ResNet-50 FPN with COCO pretrained weights."""
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def compute_binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_pixel_metrics(model, loader, device, threshold: float = 0.5) -> dict[str, float]:
    """Merge predicted instance masks into a binary mask and compute pixel metrics."""
    model.eval()
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for image, output, target in zip(imgs, outputs, targets):
                if target["masks"].shape[0] > 0:
                    height, width = target["masks"].shape[-2:]
                else:
                    height, width = image.shape[-2:]

                pred_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
                if len(output["scores"]) > 0:
                    keep = output["scores"] >= threshold
                    if keep.any():
                        pred_masks = output["masks"][keep, 0] > 0.5
                        if pred_masks.shape[-2:] != (height, width):
                            import torch.nn.functional as F

                            pred_masks = F.interpolate(
                                pred_masks.float().unsqueeze(1),
                                size=(height, width),
                                mode="nearest",
                            ).squeeze(1).bool()
                        pred_mask = pred_masks.any(dim=0)

                gt_masks = target["masks"].to(device)
                gt_mask = (
                    gt_masks.any(dim=0)
                    if gt_masks.shape[0] > 0
                    else torch.zeros((height, width), dtype=torch.bool, device=device)
                )

                total_tp += int((pred_mask & gt_mask).sum().item())
                total_fp += int((pred_mask & ~gt_mask).sum().item())
                total_fn += int((~pred_mask & gt_mask).sum().item())

    return compute_binary_metrics(total_tp, total_fp, total_fn)


def write_run_info(
    cfg: dict,
    run_dir: Path,
    train_ds: CrackInstanceDataset,
    val_ds: CrackInstanceDataset,
    device: torch.device,
) -> None:
    tr_cfg = cfg["training"]
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    ck_cfg = cfg["checkpoint"]

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
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
        f"  pretrained    : {model_cfg.get('pretrained', tr_cfg.get('pretrained', True))}",
        "",
        "  Training",
        f"  epochs        : {tr_cfg['epochs']}",
        f"  batch_size    : {tr_cfg['batch_size']}",
        f"  optimizer     : {tr_cfg.get('optimizer', 'sgd')}",
        f"  lr            : {tr_cfg['lr']}",
        f"  momentum      : {tr_cfg.get('momentum', 0.9)}",
        f"  weight_decay  : {tr_cfg.get('weight_decay', 1e-4)}",
        f"  warmup_epochs : {tr_cfg.get('warmup_epochs', 0)}",
        f"  lr_steps      : {tr_cfg.get('lr_steps', [])}",
        f"  num_workers   : {tr_cfg.get('num_workers', 0)}",
        f"  seed          : {tr_cfg.get('seed', 42)}",
        "",
        "  Dataset",
        f"  root          : {ds_cfg.get('root', '?')}",
        f"  splits_dir    : {ds_cfg.get('splits_dir', '?')}",
        f"  train images  : {len(train_ds)}",
        f"  val images    : {len(val_ds)}",
        "",
        "  Checkpoint",
        f"  save_dir      : {ck_cfg.get('save_dir', '?')}",
        f"  eval_period   : {ck_cfg.get('eval_period_epochs', 5)} epochs",
        "",
        "  System",
        f"  hostname      : {socket.gethostname()}",
        f"  platform      : {platform.platform()}",
        f"  Python        : {platform.python_version()}",
        f"  PyTorch       : {torch.__version__}",
        f"  CUDA          : {cuda_ver}",
        f"  GPU           : {gpu_name}",
        "=" * 60,
    ]

    info_path = run_dir / "train_info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Train info      -> {info_path}")


def main(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    ck_cfg = cfg["checkpoint"]

    train_ds = CrackInstanceDataset(
        split_file=Path(ds_cfg["splits_dir"]) / "train.txt",
        dataset_root=ds_cfg["root"],
        train=True,
    )
    val_ds = CrackInstanceDataset(
        split_file=Path(ds_cfg["splits_dir"]) / "val.txt",
        dataset_root=ds_cfg["root"],
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=tr_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_maskrcnn(num_classes=2, pretrained=tr_cfg.get("pretrained", True))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=tr_cfg["lr"],
        momentum=tr_cfg.get("momentum", 0.9),
        weight_decay=tr_cfg.get("weight_decay", 1e-4),
    )

    num_epochs = tr_cfg["epochs"]
    warmup_epochs = tr_cfg.get("warmup_epochs", 3)
    lr_steps = tr_cfg.get("lr_steps", [int(num_epochs * 0.6), int(num_epochs * 0.8)])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_steps,
        gamma=0.1,
    )

    stable_out_dir = Path(ck_cfg["save_dir"])
    stable_out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = stable_out_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    write_run_info(cfg, run_dir, train_ds, val_ds, device)

    log_path = run_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        [
            "epoch",
            "train_loss",
            "iou",
            "dice",
            "precision",
            "recall",
            "lr",
            "epoch_time_s",
            "gpu_mem_gb",
        ]
    )
    print(f"CSV log         -> {log_path}")

    best_iou = 0.0
    eval_period = ck_cfg.get("eval_period_epochs", 5)

    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            model.train()

            if epoch <= warmup_epochs:
                warmup_factor = epoch / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = tr_cfg["lr"] * warmup_factor

            total_loss = 0.0
            n_batches = 0

            for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                loss_dict = model(imgs, targets)
                losses = sum(loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += losses.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if epoch > warmup_epochs:
                scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]
            gpu_mem = (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda"
                else 0.0
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            metrics: dict[str, float | str] = {
                "iou": "",
                "dice": "",
                "precision": "",
                "recall": "",
            }

            if epoch % eval_period == 0 or epoch == num_epochs:
                metrics = evaluate_pixel_metrics(
                    model,
                    val_loader,
                    device,
                    threshold=cfg.get("evaluation", {}).get("threshold", 0.5),
                )
                print(
                    f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                    f"IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} "
                    f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
                )

                if float(metrics["iou"]) > best_iou:
                    best_iou = float(metrics["iou"])
                    best_payload = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "iou": float(metrics["iou"]),
                        "dice": float(metrics["dice"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                    }
                    torch.save(best_payload, run_dir / "best.pth")
                    torch.save(best_payload, stable_out_dir / "best.pth")
                    print(f"  -> Saved best.pth (IoU={best_iou:.4f})")

                if epoch % (eval_period * 2) == 0:
                    payload = {"epoch": epoch, "model": model.state_dict()}
                    torch.save(payload, run_dir / f"epoch_{epoch:03d}.pth")
                    torch.save(payload, stable_out_dir / f"epoch_{epoch:03d}.pth")
            else:
                print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}")

            csv_writer.writerow(
                [
                    epoch,
                    f"{avg_loss:.4f}",
                    f"{metrics['iou']:.4f}" if metrics["iou"] != "" else "",
                    f"{metrics['dice']:.4f}" if metrics["dice"] != "" else "",
                    f"{metrics['precision']:.4f}" if metrics["precision"] != "" else "",
                    f"{metrics['recall']:.4f}" if metrics["recall"] != "" else "",
                    f"{lr:.6f}",
                    f"{epoch_time:.1f}",
                    f"{gpu_mem:.2f}",
                ]
            )
            log_file.flush()
    finally:
        log_file.close()

    print(f"\nBest Val IoU: {best_iou:.4f}")
    print(f"Run directory   : {run_dir}")
    print(f"CSV log         : {log_path}")
    print(f"Stable best     : {stable_out_dir / 'best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN (torchvision)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(Path(args.config).resolve())

    main(cfg)
