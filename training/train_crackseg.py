"""
Unified training script for DeepLabV3+, PP-LiteSeg, and PIDNet.

Usage:
    conda activate CrackSeg
    python training/train_crackseg.py --config configs/deeplabv3plus.yaml
    python training/train_crackseg.py --config configs/ppliteseg.yaml
    python training/train_crackseg.py --config configs/pidnet.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import CrackDataset
from data.transforms import get_train_transforms, get_val_transforms
from evaluation.metrics import compute_metrics
from models.losses import BCEDiceLoss
from training.lr_scheduler import build_scheduler


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_cfg: dict) -> torch.nn.Module:
    name = model_cfg["name"].lower()

    if name == "deeplabv3plus":
        from models.deeplabv3plus import DeepLabV3Plus
        return DeepLabV3Plus(pretrained=model_cfg.get("pretrained", True))

    if name == "ppliteseg":
        # Requires the zh320 repo to be cloned at project root
        zh320_root = Path("realtime-semantic-segmentation-pytorch")
        if not zh320_root.exists():
            raise FileNotFoundError(
                "PP-LiteSeg requires zh320 repo. "
                "Run: git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch"
            )
        sys.path.insert(0, str(zh320_root))
        from models.ppliteseg import PPLiteSeg  # type: ignore[import]
        return PPLiteSeg(
            num_classes=1,
            backbone=model_cfg.get("backbone", "STDC1"),
        )

    if name == "pidnet":
        zh320_root = Path("realtime-semantic-segmentation-pytorch")
        if not zh320_root.exists():
            raise FileNotFoundError(
                "PIDNet requires zh320 repo. "
                "Run: git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch"
            )
        sys.path.insert(0, str(zh320_root))
        from models.pidnet import PIDNet  # type: ignore[import]
        return PIDNet(
            num_classes=1,
            variant=model_cfg.get("variant", "pidnet_s"),
        )

    raise ValueError(f"Unknown model name: {name!r}. Choose from deeplabv3plus / ppliteseg / pidnet")


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > threshold).float()
            all_preds.append(preds.view(-1).numpy())
            all_gts.append(masks.view(-1).numpy())

    import numpy as np
    return compute_metrics(
        np.concatenate(all_preds).astype(bool),
        np.concatenate(all_gts).astype(bool),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    # Reproducibility
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    # Datasets
    ds_cfg = cfg["dataset"]
    splits = ds_cfg["splits_dir"]
    train_ds = CrackDataset(
        ds_cfg["root"],
        f"{splits}/train.txt",
        patch_size=ds_cfg["patch_size"],
        overlap=ds_cfg["overlap"],
        transform=get_train_transforms(ds_cfg["patch_size"]),
    )
    val_ds = CrackDataset(
        ds_cfg["root"],
        f"{splits}/val.txt",
        patch_size=ds_cfg["patch_size"],
        overlap=ds_cfg["overlap"],
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=ds_cfg["num_workers"],
        pin_memory=ds_cfg["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg["num_workers"],
    )

    # Model, loss, optimiser, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    loss_cfg = cfg["loss"]
    criterion = BCEDiceLoss(
        bce_weight=loss_cfg["bce_weight"],
        dice_weight=loss_cfg["dice_weight"],
    )

    tr_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=tr_cfg["warmup_epochs"],
        total_epochs=tr_cfg["epochs"],
    )

    # Checkpoint directory
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # AMP scaler (BF16 for Blackwell Tensor Cores)
    scaler = GradScaler()

    # TensorBoard writer
    model_name = cfg["model"]["name"]
    tb_dir = Path(tr_cfg.get("output_dir", "outputs")) / "runs" / model_name
    writer = SummaryWriter(log_dir=str(tb_dir))

    # CSV log
    log_path = ckpt_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "train_loss", "iou", "dice", "precision", "recall"])

    print(f"TensorBoard logs → {tb_dir}")
    print(f"CSV log         → {log_path}")

    best_iou = 0.0
    save_every = cfg["checkpoint"].get("save_every_n_epochs", 10)

    for epoch in range(1, tr_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['epochs']}"):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast(dtype=torch.bfloat16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Periodic validation
        if epoch % 5 == 0 or epoch == tr_cfg["epochs"]:
            metrics = validate(model, val_loader, device)
            iou = metrics["iou"]
            print(
                f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                f"IoU={iou:.4f} Dice={metrics['dice']:.4f} "
                f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
            )
            writer.add_scalar("Metrics/IoU", iou, epoch)
            writer.add_scalar("Metrics/Dice", metrics["dice"], epoch)
            writer.add_scalar("Metrics/Precision", metrics["precision"], epoch)
            writer.add_scalar("Metrics/Recall", metrics["recall"], epoch)
            csv_writer.writerow([
                epoch, f"{avg_loss:.4f}", f"{iou:.4f}",
                f"{metrics['dice']:.4f}", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
            ])
            log_file.flush()

            if iou > best_iou:
                best_iou = iou
                torch.save(model.state_dict(), ckpt_dir / "best.pth")
                print(f"  -> Saved best checkpoint (IoU={best_iou:.4f})")

        if epoch % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")
        else:
            print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")

    writer.close()
    log_file.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crack segmentation model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg)
