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
import time
from pathlib import Path

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Allow imports from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add zh320 repo to sys.path at module level so DataLoader worker subprocesses
# (which re-import this module via multiprocessing spawn) can also find it.
# Append AFTER project root so local models/ (losses.py etc.) still takes priority.
_ZH320_ROOT = _PROJECT_ROOT / "realtime-semantic-segmentation-pytorch"
if _ZH320_ROOT.exists() and str(_ZH320_ROOT) not in sys.path:
    sys.path.append(str(_ZH320_ROOT))

from data.dataset import CrackDataset, PrecomputedCrackDataset
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
        zh320_root = (Path(__file__).resolve().parent.parent / "realtime-semantic-segmentation-pytorch")
        if not zh320_root.exists():
            raise FileNotFoundError(
                "PP-LiteSeg requires zh320 repo. "
                "Run: git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch"
            )
        sys.path.insert(0, str(zh320_root))
        import importlib; importlib.invalidate_caches()
        # Temporarily hide the local 'models' package so Python finds zh320's instead
        _saved = {k: v for k, v in list(sys.modules.items())
                  if k == "models" or k.startswith("models.")}
        for k in _saved:
            del sys.modules[k]
        try:
            from models.pp_liteseg import PPLiteSeg  # type: ignore[import]
        finally:
            sys.modules.update(_saved)  # restore local models
        return PPLiteSeg(
            num_class=1,
            encoder_type=model_cfg.get("backbone", "STDC1").lower(),
        )

    if name in ("pidnet", "ddrnet"):
        import importlib
        zh320_root = (Path(__file__).resolve().parent.parent / "realtime-semantic-segmentation-pytorch")
        if not zh320_root.exists():
            raise FileNotFoundError(
                "DDRNet requires zh320 repo. "
                "Run: git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch"
            )

        # zh320 models use relative imports (from .modules import ...) so they need the
        # full package context. Strategy:
        #   1. Temporarily move zh320_root to sys.path[0] (before project root)
        #   2. Remove local 'models' from sys.modules cache
        #   3. Import → zh320's models/ package is found
        #   4. Restore sys.path and sys.modules
        zh320_str = str(zh320_root)
        _path_backup = sys.path[:]
        if zh320_str in sys.path:
            sys.path.remove(zh320_str)
        sys.path.insert(0, zh320_str)
        importlib.invalidate_caches()

        _saved = {k: v for k, v in list(sys.modules.items())
                  if k == "models" or k.startswith("models.")}
        for k in _saved:
            del sys.modules[k]
        try:
            from models.ddrnet import DDRNet  # type: ignore[import]
        finally:
            # Restore sys.path and local models cache
            sys.path[:] = _path_backup
            sys.modules.update(_saved)

        # DDRNet-23-slim: 5.6M params, dual-branch real-time segmentation
        arch_type = model_cfg.get("arch_type", "DDRNet-23-slim")
        return DDRNet(num_class=1, arch_type=arch_type)

    raise ValueError(f"Unknown model name: {name!r}. Choose from deeplabv3plus / ppliteseg / pidnet")


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    threshold: float = 0.5,
) -> tuple[dict[str, float], float]:
    """Returns (metrics_dict, avg_val_loss). Uses running TP/FP/FN to avoid OOM."""
    model.eval()
    running_loss = 0.0
    tp = fp = fn = 0
    eps = 1e-7

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            running_loss += criterion(logits, masks).item()

            preds = (torch.sigmoid(logits) > threshold).bool()
            gt    = masks.bool()
            tp += int((preds &  gt).sum())
            fp += int((preds & ~gt).sum())
            fn += int((~preds & gt).sum())

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)

    metrics = {"iou": iou, "dice": dice, "precision": precision, "recall": recall}
    return metrics, running_loss / len(loader)


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
    precomputed_dir = ds_cfg.get("precomputed_dir")
    if precomputed_dir and (Path(precomputed_dir) / "train" / "rgb").exists():
        print(f"[dataset] Using pre-computed patches from {precomputed_dir}")
        train_ds = PrecomputedCrackDataset(
            str(Path(precomputed_dir) / "train"),
            transform=get_train_transforms(ds_cfg["patch_size"]),
            oversample_positive=ds_cfg.get("oversample_positive", False),
            positive_weight=ds_cfg.get("positive_weight", 5.0),
        )
        val_ds = PrecomputedCrackDataset(
            str(Path(precomputed_dir) / "val"),
            transform=get_val_transforms(),
        )
    else:
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

    persistent = ds_cfg.get("persistent_workers", False) and ds_cfg["num_workers"] > 0
    prefetch = ds_cfg.get("prefetch_factor", 2)

    # Weighted sampler: oversample patches that contain cracks.
    # Only activate when the dataset actually loaded non-uniform weights from
    # metadata.json — if metadata is missing, _sample_weights is None and
    # get_sample_weights() returns all-ones, in which case WeightedRandomSampler
    # with replacement=True would silently skip a fraction of patches each epoch.
    _use_weighted_sampler = (
        ds_cfg.get("oversample_positive", False)
        and isinstance(train_ds, PrecomputedCrackDataset)
        and train_ds._sample_weights is not None   # non-uniform weights available
    )
    if _use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            sampler=sampler,               # mutually exclusive with shuffle=True
            num_workers=ds_cfg["num_workers"],
            pin_memory=ds_cfg["pin_memory"],
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
        print("[train] WeightedRandomSampler enabled for positive patch oversampling")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=ds_cfg["num_workers"],
            pin_memory=ds_cfg["pin_memory"],
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=ds_cfg["num_workers"],
        pin_memory=ds_cfg.get("pin_memory", False),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    # Model, loss, optimiser, scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    loss_cfg = cfg["loss"]
    loss_type = loss_cfg.get("type", "bce_dice")
    if loss_type == "focal_tversky":
        from models.losses import FocalTverskyLoss
        criterion = FocalTverskyLoss(
            alpha=loss_cfg.get("alpha", 0.3),
            beta=loss_cfg.get("beta", 0.7),
            gamma=loss_cfg.get("gamma", 0.75),
        )
    elif loss_type == "focal_dice":
        from models.losses import FocalDiceLoss
        criterion = FocalDiceLoss(
            gamma=loss_cfg.get("gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.25),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
    else:
        criterion = BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", None),
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

    # Timestamped run ID — shared by TensorBoard, checkpoints, and CSV
    model_name = cfg["model"]["name"]
    resume_path = cfg.get("resume")
    stable_ckpt_dir = Path(cfg["checkpoint"]["save_dir"])

    if resume_path:
        # Resume: reuse original run folder so TensorBoard curve is continuous
        ckpt_dir = Path(resume_path).parent
        run_ts = ckpt_dir.name
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = stable_ckpt_dir / run_ts

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # AMP scaler (BF16 for Blackwell Tensor Cores)
    scaler = GradScaler()

    # TensorBoard writer — same timestamp as checkpoint folder
    tb_dir = Path(tr_cfg.get("output_dir", "outputs")) / "runs" / model_name / run_ts
    writer = SummaryWriter(log_dir=str(tb_dir))

    # CSV log — complete record for thesis
    log_path   = ckpt_dir / "train_log.csv"
    csv_mode   = "a" if (log_path.exists() and cfg.get("resume")) else "w"
    log_file   = open(log_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    if csv_mode == "w":
        csv_writer.writerow([
            "epoch", "train_loss", "val_loss",
            "iou", "dice", "precision", "recall",
            "lr", "epoch_time_s", "gpu_mem_gb",
        ])

    print(f"TensorBoard logs → {tb_dir}")
    print(f"CSV log         → {log_path}")

    best_iou   = 0.0
    start_epoch = 1
    save_every  = cfg["checkpoint"].get("save_every_n_epochs", 10)

    # Resume from checkpoint if specified
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_iou    = ckpt.get("best_iou", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] Loaded {resume_path}  (epoch {ckpt['epoch']}, best IoU={best_iou:.4f})")

    for epoch in range(start_epoch, tr_cfg["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['epochs']}"):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast(dtype=torch.bfloat16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        gpu_mem = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Validation every epoch
        threshold = cfg.get("evaluation", {}).get("threshold", 0.5)
        metrics, avg_val_loss = validate(model, val_loader, device, criterion,
                                         threshold=threshold)
        iou = metrics["iou"]

        print(
            f"Epoch {epoch:3d} | "
            f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f} | "
            f"IoU={iou:.4f}  Dice={metrics['dice']:.4f}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f} | "
            f"lr={lr:.2e}  t={epoch_time:.0f}s  mem={gpu_mem:.1f}GB"
        )

        # TensorBoard
        writer.add_scalar("Loss/train",           avg_train_loss,       epoch)
        writer.add_scalar("Loss/val",             avg_val_loss,         epoch)
        writer.add_scalar("Metrics/IoU",          iou,                  epoch)
        writer.add_scalar("Metrics/Dice",         metrics["dice"],      epoch)
        writer.add_scalar("Metrics/Precision",    metrics["precision"], epoch)
        writer.add_scalar("Metrics/Recall",       metrics["recall"],    epoch)
        writer.add_scalar("System/LR",            lr,                   epoch)
        writer.add_scalar("System/epoch_time_s",  epoch_time,           epoch)
        writer.add_scalar("System/gpu_mem_gb",    gpu_mem,              epoch)

        # CSV
        csv_writer.writerow([
            epoch,
            f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}",
            f"{iou:.4f}", f"{metrics['dice']:.4f}",
            f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
            f"{lr:.6f}", f"{epoch_time:.1f}", f"{gpu_mem:.2f}",
        ])
        log_file.flush()

        # Checkpoint: best val IoU
        if iou > best_iou:
            best_iou = iou
            ckpt_payload = {
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_iou":  best_iou,
            }
            # 1. Timestamped run folder (for per-run history)
            torch.save(ckpt_payload, ckpt_dir / "best.pth")
            # 2. Stable root path consumed by inference_crackseg.py
            stable_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_payload, stable_ckpt_dir / "best.pth")
            print(f"  -> Saved best.pth (IoU={best_iou:.4f})")

        # Checkpoint: periodic
        if epoch % save_every == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler":    scaler.state_dict(),
                "best_iou":  best_iou,
            }, ckpt_dir / f"epoch_{epoch:03d}.pth")

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint : {ckpt_dir / 'best.pth'}")
    print(f"CSV log         : {log_path}")

    writer.close()
    log_file.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crack segmentation model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = args.resume

    train(cfg)
