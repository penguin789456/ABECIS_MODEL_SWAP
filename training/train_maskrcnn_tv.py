"""
Mask R-CNN training using torchvision — NO Detectron2 / NO C++ compilation.

Runs inside the CrackSeg conda environment (same as the other three models).
Uses torchvision.models.detection.maskrcnn_resnet50_fpn with a COCO-pretrained
backbone, fine-tuned on the crack instance dataset.

Usage:
    conda activate CrackSeg
    python training/train_maskrcnn_tv.py --config configs/maskrcnn_tv.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset_instance import CrackInstanceDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_maskrcnn(num_classes: int = 2, pretrained: bool = True):
    """Build Mask R-CNN ResNet-50 FPN with COCO pretrained backbone."""
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def evaluate_pixel_iou(model, loader, device, threshold: float = 0.5) -> float:
    """Merge all predicted instance masks → binary semantic mask, compute pixel IoU."""
    model.eval()
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Val", leave=False):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for output, target in zip(outputs, targets):
                # Use GT mask dimensions as canonical size
                if target["masks"].shape[0] > 0:
                    H, W = target["masks"].shape[-2:]
                else:
                    H, W = imgs[0].shape[-2:]

                # Predicted semantic mask: union of all instance masks above threshold
                pred_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
                if len(output["scores"]) > 0:
                    keep = output["scores"] >= threshold
                    if keep.any():
                        pred_masks = output["masks"][keep, 0] > 0.5
                        # Guard against EXIF shape mismatch between model output and GT
                        if pred_masks.shape[-2:] != (H, W):
                            import torch.nn.functional as F
                            pred_masks = F.interpolate(
                                pred_masks.float().unsqueeze(1),
                                size=(H, W), mode="nearest"
                            ).squeeze(1).bool()
                        pred_mask = pred_masks.any(dim=0)

                # GT semantic mask
                gt_masks = target["masks"].to(device)
                gt_mask = gt_masks.any(dim=0) if gt_masks.shape[0] > 0 else torch.zeros(
                    (H, W), dtype=torch.bool, device=device
                )

                tp = (pred_mask & gt_mask).sum().item()
                fp = (pred_mask & ~gt_mask).sum().item()
                fn = (~pred_mask & gt_mask).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    return float(iou)


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
        optimizer, milestones=lr_steps, gamma=0.1
    )

    out_dir = Path(ck_cfg["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    eval_period = ck_cfg.get("eval_period_epochs", 5)

    for epoch in range(1, num_epochs + 1):
        model.train()
        # Warmup LR
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = tr_cfg["lr"] * warmup_factor

        total_loss = 0.0
        n_batches = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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

        # Periodic evaluation
        if epoch % eval_period == 0 or epoch == num_epochs:
            iou = evaluate_pixel_iou(model, val_loader, device)
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | Val IoU={iou:.4f}")
            if iou > best_iou:
                best_iou = iou
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "iou": iou},
                    out_dir / "best.pth",
                )
                print(f"  -> Saved best.pth (IoU={iou:.4f})")

            # Periodic checkpoint
            if epoch % (eval_period * 2) == 0:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict()},
                    out_dir / f"epoch_{epoch:03d}.pth",
                )
        else:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}")

    print(f"\nBest Val IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN (torchvision)")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
