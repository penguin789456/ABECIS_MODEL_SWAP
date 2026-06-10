# ABECIS_MODEL_SWAP — Lightweight Concrete Crack Segmentation

Master's thesis project comparing **lightweight semantic segmentation models** against the **ABECIS Mask R-CNN** baseline for concrete crack detection.

**Research question:** Can a lightweight semantic segmentation model replace ABECIS's instance segmentation framework with minimal accuracy trade-off while drastically reducing compute, enabling real-time edge deployment (e.g. NVIDIA Jetson, drones)?

**Key finding:** On the frozen Mendeley test set, ABECIS reaches the highest accuracy (IoU 0.50) but only **0.02 FPS on CPU**, while PP-LiteSeg-T reaches IoU 0.44 at **122 FPS on GPU** — ~10% accuracy gap for a ~6000× speed-up.

---

## 🚀 Quick Start (for new maintainers)

Read these in order to get oriented:

1. **`CLAUDE.md`** — single source of truth: full status, results tables, env caveats, oral-defense talking points (Chinese).
2. **This README** — reproduction steps (English).
3. **`infernce/README.md`** — the cross-dataset inference pipeline.
4. **`RESEARCH_SUMMARY.md`** — research narrative for external readers.

Two pipelines live here:
- **Training** (`training/`, `configs/`, `data/`) — trains the segmentation models.
- **Inference & evaluation** (`infernce/`, `ground_truth/`) — runs patch sliding-window inference across datasets and computes pixel-level metrics + clDice. Outputs land in `H:\ChihleeMaster\dev\final_outputs\` (machine-specific path, hardcoded in scripts).

---

## Results

### Main comparison — Mendeley frozen test set (70 images, 512 patch sliding-window)

| Model | IoU | Recall | clDice | FPS | Device |
|-------|-----|--------|--------|-----|--------|
| **ABECIS (Mask R-CNN R50-FPN)** | **0.5042** | 0.7273 | **0.8123** | 0.02 | CPU |
| PP-LiteSeg-T (STDC1, ~5M) | 0.4392 | — | — | 122 | GPU |
| DeepLabV3 (MobileNetV3-L, ~11M) | 0.4358 | — | — | 91 | GPU |
| DDRNet-23-slim (~5.6M) | 0.3908 | — | — | 90 | GPU |
| DDRNet-23-slim FTL | 0.3280 | — | — | — | GPU |

> Patch sliding-window matches the 512×512 training scale. Naively resizing the 4032×3024 Mendeley images to 1024 collapses thin cracks and drops IoU to ~0.01 — see CLAUDE.md.

### Cross-domain — CFD (Crack Forest Dataset, 118 images with GT, unseen)

| Model | Micro IoU | Recall | clDice | FPS |
|-------|-----------|--------|--------|-----|
| DDRNet | 0.3174 | 0.6250 | 0.5697 | 42.4 |
| DDRNet FTL | 0.2862 | 0.7107 | 0.6148 | 44.5 |
| DeepLabV3 | 0.2365 | 0.5977 | 0.4368 | 44.9 |
| PP-LiteSeg-T | 0.2145 | 0.3771 | 0.3214 | 38.1 |
| ABECIS (native 640×480) | 0.2142 | 0.4851 | 0.4352 | 1.31 |

FPS measured on RTX 5060 Ti (segmentation models, GPU) / i5-11400 (ABECIS, CPU).

---

## Environments

Two conda environments are required (ABECIS's Detectron2 is incompatible with the RTX 5060 Ti's sm_120):

| Env | Used for | Key versions | Device |
|-----|----------|--------------|--------|
| `CrackSeg` | All segmentation models (train + infer) | torch 2.12 dev + cu128 | GPU (sm_120 OK) |
| `ABECIS` | Detectron2 Mask R-CNN inference | torch 2.4.1 + cu121 | **CPU only** |

```bash
# CrackSeg
conda env create -f CrackSeg_env.yaml && conda activate CrackSeg
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch   # PP-LiteSeg / DDRNet
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**⚠️ Running scripts on Windows:**
- Do **not** use `conda run -n` — it crashes with a cp950 encoding error. Call the env's python directly: `G:\conda\envs\CrackSeg\python.exe ...`
- Always set `PYTHONIOENCODING=utf-8`. For segmentation models also set `PYTHONPATH` to the project root + the zh320 repo.

---

## Dataset

> Datasets are **not committed** (excluded via `.gitignore`). Place / generate them locally.

| Version | Folder | Contents | Train imgs | Status |
|---------|--------|----------|-----------|--------|
| v0 | `concreteCrackSegmentationDataset/` | Mendeley CCSD (458) | 320 | ✅ primary |
| v3 | `dataset_merged_v3/` | v0 + SCD confident pseudo-labels | ~15,380 | ✅ augmentation study |
| v1 / v2 | `dataset_merged_v1`/`_v2` | + DeepCrack / +CRACK500+CFD+GAPS384 | — | ❌ abandoned (stopped at ep16) |

- **Test set is frozen** at the 70 original Mendeley images (`data/splits/test.txt`), shared across all experiments.
- **Cross-domain test:** CFD (`H:\ChihleeMaster\CrackK500\CFD`, 118 of 155 have GT), fully unseen.

```bash
# Prepare base v0: place rgb/ and BW/ under concreteCrackSegmentationDataset/, then
python scripts/prepare_dataset.py
# Precompute 512×512 patch cache (required before training)
python scripts/precompute_patches.py --config configs/final/ppliteseg.yaml
```

---

## Training

```bash
conda activate CrackSeg
python training/train_crackseg.py --config configs/final/ppliteseg.yaml          # baseline v0
python training/train_crackseg.py --config configs/final/ppliteseg_v3.yaml       # + SCD pseudo
python training/train_crackseg.py --config configs/final/ddrnet_ftl.yaml         # DDRNet + Focal Tversky
# Resume:  --resume outputs/checkpoints/{model}/best.pth
```

Each run writes `outputs/checkpoints/{model}/{timestamp}/`: `best.pth`, periodic `epoch_NNN.pth`, `train_log.csv`, `train_info.txt`.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer / LR | AdamW / 1e-4 + cosine, warmup 5 ep |
| Batch / Epochs | 32 / 150 |
| Loss | BCEDiceLoss (BCE 0.5 + Dice 0.5); FTL variant α=0.2 β=0.8 γ=0.75 |
| Patch | 512×512, overlap 128px |
| Oversampling | WeightedRandomSampler, positive_weight=5.0 |

Monitor: `tensorboard --logdir outputs/runs --port 6006`

---

## Inference & Evaluation Pipeline

Export trained checkpoints to TorchScript, run patch sliding-window inference, then evaluate against GT. Full details in [`infernce/README.md`](infernce/README.md).

```bash
# 1. Export checkpoint → TorchScript .pt
G:\conda\envs\CrackSeg\python.exe scripts/export_ppliteseg_torchscript.py

# 2. Patch sliding-window inference (512×512, overlap 128) — per model
PYTHONIOENCODING=utf-8 PYTHONPATH="<repo>;<repo>\realtime-semantic-segmentation-pytorch" \
  G:\conda\envs\CrackSeg\python.exe infernce/mendeley_patch_inference.py --model ppliteseg --device cuda
PYTHONIOENCODING=utf-8 G:\conda\envs\ABECIS\python.exe \
  infernce/abecis_mendeley_patch_inference.py --threshold 0.8 --device cpu --resume

# 3. Ground-truth evaluation (IoU/Dice/P/R + clDice)
PYTHONIOENCODING=utf-8 G:\conda\envs\CrackSeg\python.exe \
  ground_truth/evaluate_gt_with_cldice.py --model ppliteseg_mendeley_patch --dataset mendeley --device gpu
```

Per-model inference thresholds (from validation sweep): PP-LiteSeg 0.65, DeepLabV3 0.45, DDRNet/FTL 0.50, ABECIS 0.80.

ABECIS is detection-based: each patch is inferred independently and instance masks are **unioned** into a binary full-image mask (no cross-patch NMS issue). On CFD it runs at native 640×480 (small enough to skip patching).

---

## Project Structure

```
├── configs/final/        # training YAMLs (ppliteseg / ddrnet / ddrnet_ftl / deeplabv3 / maskrcnn_tv / *_v3)
├── data/                 # dataset.py, transforms.py, splits/ (v0 committed), patches_*/ (gitignored)
├── models/               # deeplabv3_mobilenet.py, losses.py (BCEDice / FocalTversky / FocalDice)
├── training/             # train_crackseg.py (unified), train_maskrcnn_tv.py, lr_scheduler.py
├── evaluation/           # metrics.py (incl. clDice), inference_*.py, evaluate.py
├── scripts/              # prepare_dataset / prepare_external / precompute_patches / threshold_sweep
│   └── export_*_torchscript.py   # checkpoint → TorchScript .pt (for infernce/)
├── infernce/             # ⭐ cross-dataset patch inference pipeline (see infernce/README.md)
├── ground_truth/         # GT evaluation incl. clDice (--dataset cfd|mendeley)
├── archive/experiments/  # one-off sweep / detection-rate / report scripts (kept for traceability)
├── CLAUDE.md             # ⭐ project status, results, env caveats, defense notes (read first)
├── RESEARCH_SUMMARY.md   # research narrative
├── CrackSeg_env.yaml
└── LICENSE
```

Gitignored (not in repo): datasets (`concreteCrackSegmentationDataset/`, `dataset_merged_v*/`), patch caches (`data/patches_v*/`), `outputs/`, cloned repos (`realtime-semantic-segmentation-pytorch/`, `detectron2/`).

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `conda run` cp950 / UnicodeEncodeError | Call env python.exe directly + set `PYTHONIOENCODING=utf-8` |
| ABECIS Mendeley patch inference interrupted | Re-run with `--resume` (skips already-written pred_masks) |
| `CUDA out of memory` | Lower `batch_size` in config YAML |
| `No module named 'models.losses'` in worker | Ensure `PYTHONPATH` includes the repo root |
| PP-LiteSeg/DDRNet import error | Clone zh320 repo to project root, add to PYTHONPATH |
| GPU utilization pulsing | Run `precompute_patches.py` first |

---

## References

- zh320 realtime segmentation: https://github.com/zh320/realtime-semantic-segmentation-pytorch
- DeepCrack: https://github.com/yhlleo/DeepCrack · CRACK500: https://github.com/fyangneil/pavement-crack-detection

## License

Apache License 2.0 — see [LICENSE](LICENSE).
