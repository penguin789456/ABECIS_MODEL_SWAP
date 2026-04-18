# ABECIS_MODEL_SWAP — Concrete Crack Segmentation

Comparative study of lightweight semantic segmentation models vs. the ABECIS Mask R-CNN baseline on a concrete crack dataset, as part of a Master's thesis.

**Research question:** Can a lightweight semantic segmentation model replace ABECIS's instance segmentation framework with minimal accuracy trade-off while drastically reducing computational cost?

| Model | Role | Params | Test IoU |
|-------|------|--------|----------|
| Mask R-CNN R50-FPN | ABECIS baseline | 44M | (pending) |
| DeepLabV3 (MobileNetV3-Large) | Semantic seg baseline | 11M | (pending) |
| DDRNet-23-slim | Dual-branch real-time | 5.6M | **0.3912** |
| PP-LiteSeg-T (STDC1) | Primary lightweight model | 5M | **0.4391** |

All models evaluated with unified **pixel-level IoU** on the same frozen test set (70 Mendeley images).

---

## Environment Setup

All models run in a single **CrackSeg** conda environment.

```bash
conda env create -f CrackSeg_env.yaml
conda activate CrackSeg

# Required for PP-LiteSeg and DDRNet
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
```

---

## Dataset

> The dataset is **not included** in this repository (excluded via `.gitignore`).

Three dataset tiers are used:

| Version | Folder | Contents | Train images |
|---------|--------|----------|-------------|
| v0 | `concreteCrackSegmentationDataset/` | Mendeley (458 images) | 320 |
| v1 | `dataset_merged_v1/` | v0 + DeepCrack (537) | 857 |
| v2 | `dataset_merged_v2/` | v1 + CRACK500 (1,896) + CFD (118) + GAPS384 (509) | 3,380 |

**Test set is frozen at the original 70 Mendeley images for all experiments.**

### Prepare base dataset (v0)

Place files at `concreteCrackSegmentationDataset/rgb/` and `.../BW/`, then:

```bash
conda activate CrackSeg
python scripts/prepare_dataset.py
```

### Merge external datasets (v1 / v2)

```bash
# v1: Mendeley + DeepCrack
python scripts/prepare_external.py \
    --deepcrack_dir /path/to/DeepCrack \
    --output_root dataset_merged_v1 \
    --output_splits_dir data/splits_v1

# v2: all sources
python scripts/prepare_external.py \
    --deepcrack_dir    /path/to/DeepCrack \
    --crack500_dir     /path/to/CRACK500/traincrop \
    --cfd_img_dir      /path/to/CFD/cfd_image \
    --cfd_mask_dir     /path/to/CFD/seg_gt \
    --gaps384_img_dir  /path/to/GAPS384/croppedimg \
    --gaps384_mask_dir /path/to/GAPS384/croppedgt \
    --output_root      dataset_merged_v2 \
    --output_splits_dir data/splits_v2
```

### Precompute patch cache (required before training)

```bash
# Precompute for a specific config
python scripts/precompute_patches.py --config configs/final/ppliteseg_v2.yaml
```

Patches are saved as `.npy` files under `data/patches_*/` and loaded directly at training time, eliminating CPU bottleneck and keeping GPU utilization stable.

---

## Training

```bash
conda activate CrackSeg

# Baseline (Mendeley only)
python training/train_crackseg.py --config configs/final/ppliteseg.yaml

# v1 experiment (+ DeepCrack)
python training/train_crackseg.py --config configs/final/ppliteseg_v1.yaml

# v2 experiment (full merged dataset)
python training/train_crackseg.py --config configs/final/ppliteseg_v2.yaml

# Resume from checkpoint
python training/train_crackseg.py --config configs/final/ppliteseg_v2.yaml \
    --resume outputs/checkpoints/ppliteseg_v2/best.pth
```

### Training outputs

Each run creates a timestamped folder under `outputs/checkpoints/{model}/{timestamp}/`:

```
outputs/checkpoints/ppliteseg_v2/
├── best.pth              ← best val IoU checkpoint
├── epoch_010.pth         ← periodic checkpoint (every 10 epochs)
├── train_log.csv         ← per-epoch metrics (loss/IoU/Dice/P/R/lr/time/mem)
└── train_info.txt        ← training parameters snapshot
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 + cosine annealing |
| Warmup | 5 epochs |
| Batch Size | 32 |
| Epochs | 150 |
| Loss | BCEDiceLoss (BCE 0.5 + Dice 0.5) |
| Patch Size | 512×512, overlap 128px |
| Oversampling | WeightedRandomSampler, positive_weight=5.0 |

### Monitor training

```bash
# TensorBoard
tensorboard --logdir outputs/runs --port 6006

# GPU utilization
nvidia-smi -l 1
```

---

## Evaluation

### Threshold sweep (find optimal inference threshold)

```bash
python scripts/threshold_sweep.py --config configs/final/ppliteseg_v2.yaml
```

### Run inference + evaluation

```bash
# Generate predictions
python evaluation/inference_crackseg.py --config configs/final/ppliteseg_v2.yaml

# Compute metrics
python evaluation/evaluate.py

# Results
cat outputs/results/metrics_summary.csv
```

---

## Project Structure

```
├── configs/final/
│   ├── ppliteseg.yaml              # PP-LiteSeg baseline (v0)
│   ├── ppliteseg_v1.yaml           # PP-LiteSeg + DeepCrack
│   ├── ppliteseg_v2.yaml           # PP-LiteSeg full merged dataset
│   ├── ddrnet.yaml
│   ├── deeplabv3_mobilenet.yaml
│   └── maskrcnn.yaml
├── data/
│   ├── dataset.py                  # CrackDataset / PrecomputedCrackDataset
│   ├── transforms.py               # Albumentations pipelines
│   ├── splits/                     # v0 frozen splits (committed)
│   ├── splits_v1/  splits_v2/      # merged splits (gitignored)
│   ├── patches/                    # v0 patch cache (gitignored)
│   ├── patches_v1/ patches_v2/     # merged patch cache (gitignored)
├── models/
│   ├── deeplabv3_mobilenet.py
│   └── losses.py                   # BCEDiceLoss, FocalTverskyLoss, FocalDiceLoss
├── training/
│   ├── train_crackseg.py           # Unified trainer (all models)
│   ├── train_maskrcnn_tv.py        # torchvision Mask R-CNN trainer
│   └── lr_scheduler.py
├── evaluation/
│   ├── metrics.py                  # IoU, Dice, Precision, Recall, clDice
│   ├── inference_crackseg.py
│   ├── inference_maskrcnn_tv.py
│   └── evaluate.py
├── scripts/
│   ├── prepare_dataset.py          # Validate pairs, generate splits
│   ├── prepare_external.py         # Merge external datasets
│   ├── precompute_patches.py       # Build .npy patch cache
│   ├── threshold_sweep.py          # Find optimal inference threshold
│   ├── benchmark_loader.py         # DataLoader throughput benchmark
│   └── benchmark_fps.py            # Model FPS benchmark
├── concreteCrackSegmentationDataset/   # gitignored — place manually
├── dataset_merged_v1/                  # gitignored — generated by prepare_external.py
├── dataset_merged_v2/                  # gitignored — generated by prepare_external.py
├── outputs/                            # gitignored — created at runtime
├── CrackSeg_env.yaml
└── LICENSE
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Lower `batch_size` in config YAML |
| `No module named 'models.losses'` in worker | Fixed — Windows multiprocessing requires PYTHONPATH propagation |
| albumentations update warning | Harmless; set `NO_ALBUMENTATIONS_UPDATE=1` to suppress |
| PP-LiteSeg/DDRNet import error | Clone zh320 repo to project root |
| GPU utilization pulsing | Run `precompute_patches.py` first — eliminates CPU bottleneck |

---

## References

- zh320 Realtime Segmentation Repo: https://github.com/zh320/realtime-semantic-segmentation-pytorch
- DeepCrack: https://github.com/yhlleo/DeepCrack
- CRACK500: https://github.com/fyangneil/pavement-crack-detection

## License

Apache License 2.0 — see [LICENSE](LICENSE).
