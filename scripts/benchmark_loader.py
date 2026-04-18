"""
DataLoader + forward/backward throughput benchmark.

Usage:
    python scripts/benchmark_loader.py \
        --config configs/final/ddrnet.yaml \
        --batches 30

Tests all combinations of batch_size × num_workers, prints a summary table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── GPU warm-up helper ────────────────────────────────────────────────────────
def _warmup(model, device, batch_size, n=3):
    import torch
    model.train()
    dummy = torch.randn(batch_size, 3, 512, 512, device=device)
    for _ in range(n):
        out = model(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out.mean().backward()
    torch.cuda.synchronize()


# ── Single benchmark run ──────────────────────────────────────────────────────
def benchmark(cfg: dict, batch_size: int, num_workers: int, n_batches: int) -> dict:
    import torch
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    from data.dataset import PrecomputedCrackDataset, CrackDataset
    from data.transforms import get_train_transforms as get_train_transform

    ds_cfg = cfg["dataset"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build dataset ─────────────────────────────────────────────────────────
    precomputed_dir = ds_cfg.get("precomputed_dir")
    if precomputed_dir and Path(precomputed_dir).exists():
        dataset = PrecomputedCrackDataset(
            patches_dir=str(Path(precomputed_dir) / "train"),
            transform=get_train_transform(),
            oversample_positive=ds_cfg.get("oversample_positive", True),
            positive_weight=ds_cfg.get("positive_weight", 5.0),
        )
        mode = "precomputed"
    else:
        dataset = CrackDataset(
            dataset_root=ds_cfg["root"],
            split_file=str(Path(ds_cfg["splits_dir"]) / "train.txt"),
            patch_size=ds_cfg.get("patch_size", 512),
            overlap=ds_cfg.get("overlap", 128),
            transform=get_train_transform(),
        )
        mode = "on-the-fly"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(ds_cfg.get("prefetch_factor", 4) if num_workers > 0 else None),
        drop_last=True,
    )

    # ── Build model ───────────────────────────────────────────────────────────
    import importlib, importlib.util

    zh320_root = PROJECT_ROOT / "realtime-semantic-segmentation-pytorch"
    zh320_str = str(zh320_root)
    if zh320_str in sys.path:
        sys.path.remove(zh320_str)
    sys.path.insert(0, zh320_str)
    importlib.invalidate_caches()
    _saved = {k: v for k, v in list(sys.modules.items())
              if k == "models" or k.startswith("models.")}
    for k in _saved:
        del sys.modules[k]
    try:
        from models.ddrnet import DDRNet
    finally:
        sys.path.remove(zh320_str)
        sys.modules.update(_saved)

    model = DDRNet(num_class=1, arch_type="DDRNet-23-slim").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    _warmup(model, device, batch_size)

    # ── Timed loop ────────────────────────────────────────────────────────────
    model.train()
    loader_iter = iter(loader)
    times = []
    samples_per_iter = []

    for i in range(n_batches):
        try:
            imgs, masks = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            imgs, masks = next(loader_iter)

        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.bfloat16):
            out = model(imgs)
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= 3:  # skip first 3 (warm-up I/O)
            times.append(t1 - t0)
            samples_per_iter.append(imgs.size(0))

    if not times:
        return {}

    avg_t = sum(times) / len(times)
    avg_samples = sum(samples_per_iter) / len(samples_per_iter)
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "mode": mode,
        "avg_s_per_it": avg_t,
        "samples_per_sec": avg_samples / avg_t,
        "it_per_sec": 1.0 / avg_t,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/final/ddrnet.yaml")
    parser.add_argument("--batches", type=int, default=30,
                        help="Timed batches per config (skip first 3)")
    parser.add_argument("--batch_sizes",  default="32,64,128,256")
    parser.add_argument("--num_workers",  default="0,2,4")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    batch_sizes  = [int(x) for x in args.batch_sizes.split(",")]
    worker_list  = [int(x) for x in args.num_workers.split(",")]
    combos = [(b, w) for b in batch_sizes for w in worker_list]

    print(f"\n{'='*66}")
    print(f"  DDRNet benchmark -- {len(combos)} configs x {args.batches} batches each")
    print(f"{'='*66}")
    print(f"  {'batch':>6}  {'workers':>7}  {'s/it':>8}  {'it/s':>7}  {'img/s':>8}  mode")
    print(f"  {'-'*60}")

    results = []
    for batch_size, num_workers in combos:
        label = f"batch={batch_size:3d} workers={num_workers}"
        print(f"  >> Testing {label} ...", flush=True)
        try:
            r = benchmark(cfg, batch_size, num_workers, args.batches)
            if r:
                results.append(r)
                print(f"  {'':>3}{batch_size:>6}  {num_workers:>7}  "
                      f"{r['avg_s_per_it']:>8.3f}  "
                      f"{r['it_per_sec']:>7.2f}  "
                      f"{r['samples_per_sec']:>8.1f}  "
                      f"{r['mode']}")
        except Exception as e:
            print(f"  FAIL {label}: {e}")

    if not results:
        print("No results.")
        return

    best = max(results, key=lambda r: r["samples_per_sec"])
    print(f"\n{'='*66}")
    print(f"  FASTEST: batch_size={best['batch_size']}  num_workers={best['num_workers']}")
    print(f"           {best['samples_per_sec']:.1f} img/s  ({best['avg_s_per_it']*1000:.0f} ms/it)")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
