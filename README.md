# Lightweight Concrete Crack Segmentation

Training code for a master's-thesis study comparing **lightweight semantic
segmentation models** (PP-LiteSeg-T, DDRNet-23-slim, DeepLabV3-MobileNetV3)
against a **Mask R-CNN** baseline for concrete crack detection.

**Research question:** can a lightweight semantic segmentation model replace an
instance-segmentation framework with minimal accuracy loss while drastically
reducing compute, enabling real-time edge deployment (e.g. NVIDIA Jetson)?

> This repository contains **training and validation code only**. The
> cross-dataset inference / benchmarking pipeline and its outputs live outside
> this repo.

---

## 1. Installation (venv)

Python 3.11 is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 2. Install PyTorch (CUDA-specific — pick ONE):
#    RTX 50-series (Blackwell, sm_120) — CUDA 12.8 nightly:
pip install --pre torch==2.12.0.dev20260327+cu128 torchvision==0.26.0.dev20260326+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128
#    Stable GPUs (CUDA 12.1):
#    pip install torch torchvision

# 3. Install the remaining dependencies
pip install -r requirements.txt

# 4. Clone the backbone repo for PP-LiteSeg / DDRNet (kept out of this repo)
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch

# 5. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Add the project root **and** the cloned backbone repo to `PYTHONPATH` when running
(PP-LiteSeg / DDRNet are imported from the latter):

```bash
export PYTHONPATH="$PWD:$PWD/realtime-semantic-segmentation-pytorch"   # Windows: set / $env:
```

---

## 2. Training data

Datasets are **not committed** (excluded via `.gitignore`); place or generate
them locally. Only the frozen split lists (`data/splits/*.txt`) are versioned.

| Version | Folder | Contents | Train imgs | Role |
|---------|--------|----------|-----------:|------|
| **v0** | `concreteCrackSegmentationDataset/` | Mendeley CCSD — 458 wall-crack images (`rgb/` + `BW/` masks) | 320 | Primary |
| v3 | `dataset_merged_v3/` | v0 + Surface-Crack-Detection confident pseudo-labels | ~15,380 | Data-augmentation study |
| v1 / v2 | `dataset_merged_v1` / `_v2` | + DeepCrack / + CRACK500 + GAPS384 | — | Abandoned |

- **Split:** 70 / 15 / 15 train/val/test, `seed=42` (`data/split.py`). The
  **test set is frozen** at 70 original Mendeley images
  (`data/splits/test.txt`) and shared across all experiments.
- **Patches:** models train on **512×512 patches, 128 px overlap** (stride 384),
  extracted from the full-resolution images. Online augmentation only
  (`data/transforms.py`) — flips, ±45° rotation, brightness/contrast, CLAHE,
  sharpen, Gaussian noise/blur — so the sample count equals the patch count, not
  a fixed "augmented image count".

```bash
# Prepare base v0: place rgb/ and BW/ under concreteCrackSegmentationDataset/, then
python scripts/prepare_dataset.py
# Precompute the 512×512 patch cache (required before training; writes data/patches*/)
python scripts/precompute_patches.py --config configs/final/ppliteseg.yaml
```

Original dataset sources: Mendeley CCSD (concrete walls), Surface Crack Detection
(v3 pseudo-labels), CFD / Crack Forest Dataset (cross-domain, road surface).

---

## 3. Training

```bash
python training/train_crackseg.py --config configs/final/ppliteseg.yaml       # PP-LiteSeg-T (v0 baseline)
python training/train_crackseg.py --config configs/final/ppliteseg_v3.yaml    # + SCD pseudo-labels
python training/train_crackseg.py --config configs/final/ddrnet.yaml          # DDRNet-23-slim
python training/train_crackseg.py --config configs/final/ddrnet_ftl.yaml      # DDRNet + Focal Tversky loss
python training/train_crackseg.py --config configs/final/deeplabv3_mobilenet.yaml
python training/train_maskrcnn_tv.py --config configs/final/maskrcnn.yaml     # torchvision Mask R-CNN
# Resume:  --resume outputs/checkpoints/{model}/best.pth
```

Each run writes `outputs/checkpoints/{model}/{timestamp}/`: `best.pth`, periodic
`epoch_NNN.pth`, `train_log.csv`, `train_info.txt`. Monitor with
`tensorboard --logdir outputs/runs`.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer / LR | AdamW / 1e-4 + cosine annealing, 5-epoch warmup |
| Batch / Epochs | 32 / 150 |
| Loss | `BCEDiceLoss` (BCE 0.5 + Dice 0.5); FTL variant α=0.2 β=0.8 γ=0.75 |
| Patch | 512×512, 128 px overlap |
| Oversampling | `WeightedRandomSampler`, positive_weight = 5.0 |

---

## 4. Validation

Validation metrics run each epoch inside the trainer (`evaluation/metrics.py`:
IoU, Dice, Precision, Recall, clDice). To sweep the decision threshold on the
validation set, benchmark throughput, or score pre-generated predictions:

```bash
# Per-model optimal threshold (sweeps on the validation split)
python scripts/threshold_sweep.py --config configs/final/ppliteseg.yaml \
    --checkpoint outputs/checkpoints/ppliteseg/best.pth
# FPS / latency / parameter count / model size
python scripts/benchmark_fps.py --config configs/final/ppliteseg.yaml --batch_size 1
# Score a folder of prediction masks against ground truth
python evaluation/evaluate.py --dataset_root concreteCrackSegmentationDataset \
    --predictions_dir outputs/preds/ppliteseg --test_split data/splits/test.txt \
    --output_dir outputs/eval/ppliteseg
```

Per-model optimal thresholds (from the validation sweep):
PP-LiteSeg 0.65 · DeepLabV3 0.45 · DDRNet / DDRNet-FTL 0.50.

Export a trained checkpoint to TorchScript for deployment:

```bash
python scripts/export_ppliteseg_torchscript.py     # → outputs/checkpoints/ppliteseg/ppliteseg_torchscript.pt
```

---

## 5. Project structure

```
├── configs/
│   ├── base.yaml              # shared defaults
│   └── final/                 # authoritative training configs (ppliteseg / ddrnet / ddrnet_ftl / deeplabv3 / maskrcnn / *_v1..v3)
├── data/
│   ├── dataset.py             # CrackDataset + PrecomputedCrackDataset (.npy patch cache)
│   ├── dataset_instance.py    # instance-mask dataset (Mask R-CNN)
│   ├── transforms.py          # Albumentations train/val pipelines
│   ├── split.py               # 70/15/15 split (seed=42)
│   └── splits/                # frozen train/val/test.txt (versioned)
├── models/
│   ├── deeplabv3_mobilenet.py
│   └── losses.py              # BCEDice / FocalTversky / FocalDice
├── training/
│   ├── train_crackseg.py      # unified semantic-segmentation trainer
│   ├── train_maskrcnn_tv.py   # torchvision Mask R-CNN trainer
│   └── lr_scheduler.py        # warmup + CosineAnnealingLR
├── evaluation/
│   ├── metrics.py             # IoU / Dice / P / R / clDice
│   ├── evaluate.py            # checkpoint evaluation
│   └── inference_*.py         # single-image inference helpers
├── scripts/
│   ├── prepare_dataset.py / prepare_external.py / prepare_surface_crack.py
│   ├── precompute_patches.py  # build 512×512 patch cache
│   ├── threshold_sweep.py / benchmark_fps.py / benchmark_loader.py
│   └── export_*_torchscript.py
├── requirements.txt
├── CLAUDE.md                  # project status / results / notes
└── LICENSE
```

**Gitignored** (not in repo): datasets (`concreteCrackSegmentationDataset/`,
`dataset_merged_v*/`), patch caches (`data/patches*/`), training outputs
(`outputs/`), and the cloned backbone repo
(`realtime-semantic-segmentation-pytorch/`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'models.losses'` / `data.dataset` | Add the repo root to `PYTHONPATH` |
| `PP-LiteSeg` / `DDRNet` import error | Clone the zh320 backbone repo and add it to `PYTHONPATH` |
| `CUDA out of memory` | Lower `batch_size` in the config YAML |
| GPU utilization pulsing / slow | Run `scripts/precompute_patches.py` first |
| `torch.cuda.is_available()` is False on RTX 50-series | Use the CUDA 12.8 nightly wheel (see Installation) |

## References

- PP-LiteSeg / DDRNet backbones: https://github.com/zh320/realtime-semantic-segmentation-pytorch
- Mendeley Concrete Crack Segmentation Dataset · CFD (Crack Forest Dataset)

## License

Apache License 2.0 — see [LICENSE](LICENSE).
