# ABECIS_MODEL_SWAP — Concrete Crack Segmentation

Comparative study of four deep learning segmentation models on a custom concrete crack dataset, as part of a Master's thesis.

| Model | Role | Environment |
|-------|------|-------------|
| Mask R-CNN R50-FPN | Instance segmentation baseline | CrackPre (Detectron2) |
| DeepLabV3+ | Semantic segmentation baseline | CrackSeg |
| PP-LiteSeg-T (STDC1) | Primary lightweight model | CrackSeg |
| PIDNet-S | Real-time comparison model | CrackSeg |

---

## Dataset Setup

> **The dataset is NOT included in this repository** (excluded via `.gitignore`).

Place the dataset files at the following paths before running any scripts:

```
concreteCrackSegmentationDataset/
├── rgb/        ← 458 original concrete images (.jpg / .JPG)
├── BW/         ← 458 binary masks (.jpg, white pixel = crack)
└── testing/    ← 22 additional evaluation images
```

After placing the dataset, validate and generate the train/val/test splits:

```bash
conda activate CrackSeg
python scripts/prepare_dataset.py
```

This will:
- Validate all 458 rgb/BW image pairs (handles mixed .jpg/.JPG extensions)
- Generate frozen split files at `data/splits/{train,val,test}.txt` (70% / 15% / 15%)
- Convert binary masks to COCO-format JSON for Detectron2 (`outputs/coco_annotations/`)

> **Important:** Commit `data/splits/` to git after generation. All four models must be evaluated on the same frozen test set.

---

## Environment Setup

### CrackSeg — DeepLabV3+, PP-LiteSeg, PIDNet

```bash
conda env create -f CrackSeg_env.yaml
conda activate CrackSeg
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
```

### CrackPre — Mask R-CNN / Detectron2

```bash
# Windows: install C++ Build Tools first
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

conda env create -n CrackPre -f ComplateENV.yaml
conda activate CrackPre
pip install -e detectron2 --no-build-isolation
```

> If `import detectron2` fails, run `Fixdetectron2.bat`.

---

## Quick Start

```bash
# 1. Prepare dataset (CrackSeg env)
python scripts/prepare_dataset.py

# 2. Train all three CrackSeg models
scripts\run_train_crackseg.bat

# 3. Train Mask R-CNN (switch to CrackPre env)
scripts\run_train_maskrcnn.bat

# 4. Run inference + unified evaluation
scripts\run_eval_all.bat
# Results → outputs/results/metrics_summary.csv
```

---

## Development Tools

### Jupyter Lab

用於執行 `notebooks/` 下的資料探索與結果視覺化筆記本。

```bash
conda activate CrackSeg
jupyter lab --notebook-dir=notebooks --no-browser --port=8888
```

開啟後在瀏覽器訪問：`http://localhost:8888`

| Notebook | 用途 |
|----------|------|
| `01_data_exploration.ipynb` | 資料集統計與影像瀏覽 |
| `02_patch_visualization.ipynb` | 512×512 patch 切分視覺化 |
| `03_results_comparison.ipynb` | 四個模型評估結果比較 |

### TensorBoard

用於訓練時監控 loss 與 metrics 曲線。訓練腳本需先輸出 log 至 `outputs/runs/`。

```bash
conda activate CrackSeg
tensorboard --logdir=outputs/runs --port=6006
```

開啟後在瀏覽器訪問：`http://localhost:6006`

> **注意**：需先執行訓練腳本產生 log 資料，TensorBoard 才有內容可顯示。

---

## Project Structure

```
├── data/
│   ├── dataset.py              # PyTorch Dataset: 512×512 patch extraction
│   ├── transforms.py           # Albumentations augmentation pipelines
│   ├── split.py                # 70/15/15 split generator
│   └── splits/                 # Frozen split index files (committed to git)
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── configs/
│   ├── base.yaml               # Shared defaults
│   ├── deeplabv3plus.yaml
│   ├── ppliteseg.yaml
│   ├── pidnet.yaml
│   └── maskrcnn.yaml
├── models/
│   ├── deeplabv3plus.py        # torchvision wrapper, binary output
│   └── losses.py               # BCEDiceLoss
├── training/
│   ├── train_crackseg.py       # Unified trainer (DeepLabV3+, PP-LiteSeg, PIDNet)
│   ├── train_maskrcnn.py       # Detectron2 trainer (CrackPre env)
│   └── lr_scheduler.py         # Warmup + cosine annealing
├── evaluation/
│   ├── metrics.py              # IoU, Dice, Precision, Recall
│   ├── postprocess.py          # Skeletonization, crack length, continuity
│   ├── inference_crackseg.py   # Generate PNG masks from CrackSeg models
│   ├── inference_maskrcnn.py   # Generate PNG masks from Detectron2
│   └── evaluate.py             # Env-agnostic unified evaluator → CSV
├── scripts/
│   ├── prepare_dataset.py      # Validate pairs, generate splits + COCO JSON
│   ├── benchmark_fps.py        # FPS / latency benchmarking
│   ├── run_train_crackseg.bat
│   ├── run_train_maskrcnn.bat
│   └── run_eval_all.bat
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_patch_visualization.ipynb
│   └── 03_results_comparison.ipynb
├── outputs/                    # gitignored — created at runtime
│   ├── checkpoints/
│   ├── predictions/
│   ├── logs/
│   ├── results/
│   └── coco_annotations/
├── concreteCrackSegmentationDataset/   # gitignored — must be downloaded manually
├── CrackSeg_env.yaml
├── ComplateENV.yaml
└── LICENSE
```

---

## Training Configuration

| Hyperparameter | PyTorch Models | Mask R-CNN |
|----------------|---------------|------------|
| Optimizer | AdamW | SGD |
| Learning Rate | 1e-4 + cosine | 0.001 |
| Batch Size | 8 | 2 |
| Epochs | 100 + 5 warmup | 50 |
| Loss | BCE + Dice | Cross-Entropy |

## Evaluation Metrics

- **Segmentation:** IoU (primary), Dice, Precision, Recall
- **Efficiency:** FPS, inference time (ms), parameter count, model size
- **Post-processing:** Skeletonization, crack length estimation, continuity analysis

---

## Results

| Model | IoU | Dice | Precision | Recall | FPS |
|-------|-----|------|-----------|--------|-----|
| Mask R-CNN | — | — | — | — | — |
| DeepLabV3+ | — | — | — | — | — |
| PP-LiteSeg-T | — | — | — | — | — |
| PIDNet-S | — | — | — | — | — |

*Results will be filled after experiments complete.*

---

## References

- PP-LiteSeg: [ppliteseg.pdf](PP-LiteSeg.pdf)
- Detectron2: [detectron2.pdf](detectron2.pdf)
- zh320 Realtime Segmentation Repo: https://github.com/zh320/realtime-semantic-segmentation-pytorch
- ABECIS Dataset: *(cite source here)*

## License

Apache License 2.0 — see [LICENSE](LICENSE).
