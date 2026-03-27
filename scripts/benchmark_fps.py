"""
FPS and latency benchmarking for CrackSeg models.

Warms up 10 batches, then times 100 forward passes and reports:
  - Mean / std FPS
  - Mean / std inference time per image (ms)
  - Parameter count
  - Model file size (MB)

Usage:
    conda activate CrackSeg
    python scripts/benchmark_fps.py --config configs/deeplabv3plus.yaml
    python scripts/benchmark_fps.py --config configs/ppliteseg.yaml --batch_size 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train_crackseg import build_model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark(
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int = 512,
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 100,
) -> dict:
    model.eval()
    dummy = torch.randn(batch_size, 3, patch_size, patch_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms)
    per_image_ms = times_arr / batch_size

    return {
        "batch_size": batch_size,
        "mean_batch_ms": float(times_arr.mean()),
        "std_batch_ms": float(times_arr.std()),
        "mean_per_image_ms": float(per_image_ms.mean()),
        "std_per_image_ms": float(per_image_ms.std()),
        "fps": float(batch_size * 1000.0 / times_arr.mean()),
    }


def main(cfg: dict, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    ckpt_path = Path(cfg["checkpoint"]["save_dir"]) / "best.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path} — benchmarking untrained model")

    params = count_parameters(model)
    model_mb = ckpt_path.stat().st_size / 1e6 if ckpt_path.exists() else 0.0

    patch_size = cfg["dataset"].get("patch_size", 512)
    results = benchmark(model, device, patch_size=patch_size, batch_size=batch_size)

    model_name = cfg["model"]["name"]
    print(f"\n{'=' * 50}")
    print(f"Model        : {model_name}")
    print(f"Device       : {device}")
    print(f"Batch size   : {batch_size}")
    print(f"Patch size   : {patch_size}×{patch_size}")
    print(f"Parameters   : {params:,}")
    print(f"Model size   : {model_mb:.1f} MB")
    print(f"FPS          : {results['fps']:.1f}")
    print(f"Per-image    : {results['mean_per_image_ms']:.2f} ± {results['std_per_image_ms']:.2f} ms")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model FPS and latency")
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.batch_size)
