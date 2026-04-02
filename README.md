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

## Training Guide

### 前置確認

開始訓練前，確保以下條件已滿足：

- [ ] `concreteCrackSegmentationDataset/` 已放置於專案根目錄
- [ ] `data/splits/train.txt`、`val.txt`、`test.txt` 已存在（執行過 `prepare_dataset.py`）
- [ ] `outputs/coco_annotations/train.json`、`val.json`、`test.json` 已存在（Mask R-CNN 需要）
- [ ] `CrackSeg` conda 環境已建立
- [ ] `CrackPre` conda 環境已建立（Mask R-CNN 才需要）

```bash
# 驗證 splits 是否存在
ls data/splits/
# 應輸出：test.txt  train.txt  val.txt
```

---

### Step 1：訓練 CrackSeg 三個模型（DeepLabV3+ / PP-LiteSeg-T / PIDNet-S）

這三個模型共用同一個訓練腳本，透過不同的 config 檔切換。

#### 方法 A：一鍵執行（推薦）

```bat
REM 從專案根目錄執行（Windows CMD）
scripts\run_train_crackseg.bat
```

腳本會依序訓練三個模型，任一個失敗即停止。

#### 方法 B：單獨訓練某個模型

```bash
conda activate CrackSeg

# 訓練 DeepLabV3+（ResNet-101 backbone）
python training/train_crackseg.py --config configs/deeplabv3plus.yaml

# 訓練 PP-LiteSeg-T（STDC1 backbone，主要模型）
python training/train_crackseg.py --config configs/ppliteseg.yaml

# 訓練 PIDNet-S（對照模型）
python training/train_crackseg.py --config configs/pidnet.yaml
```

> **注意**：PP-LiteSeg 與 PIDNet 需要先 clone zh320 repo 至專案根目錄：
> ```bash
> git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
> ```

#### 訓練過程輸出

```
Epoch   5 | loss=0.3821 | IoU=0.5214 Dice=0.6851 P=0.7102 R=0.6613
  -> Saved best checkpoint (IoU=0.5214)
Epoch  10 | loss=0.2934 | IoU=0.6033 Dice=0.7523 P=0.7801 R=0.7261
  -> Saved best checkpoint (IoU=0.6033)
...
```

- 每 **5 個 epoch** 執行一次 validation，輸出 IoU / Dice / Precision / Recall
- 每 **10 個 epoch** 儲存週期性 checkpoint
- Val IoU 有提升時自動儲存 `best.pth`

#### Checkpoint 輸出位置

```
outputs/checkpoints/
├── deeplabv3plus/
│   ├── best.pth          ← Val IoU 最高的權重（用於評估）
│   ├── epoch_010.pth
│   ├── epoch_020.pth
│   └── ...
├── ppliteseg/
│   └── best.pth
└── pidnet/
    └── best.pth
```

#### 訓練超參數（CrackSeg 三個模型共用）

| 超參數 | 值 |
|--------|-----|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| LR Schedule | 5 epoch warmup + CosineAnnealingLR |
| Batch Size | 8 |
| Epochs | 100 |
| Loss | BCEDiceLoss（BCE 0.5 + Dice 0.5）|
| Input Size | 512 × 512 patch |
| Augmentation | HFlip、VFlip、Rotate ±10°、亮度/對比、Gaussian Noise |

---

### Step 2：訓練 Mask R-CNN（CrackPre 環境）

Mask R-CNN 使用 Detectron2 框架，需切換至 `CrackPre` 環境，且需要 COCO JSON 格式的標註。

#### 方法 A：一鍵執行

```bat
REM 從專案根目錄執行（Windows CMD）
scripts\run_train_maskrcnn.bat
```

#### 方法 B：手動執行

```bash
conda activate CrackPre
python training/train_maskrcnn.py --config configs/maskrcnn.yaml
```

#### 訓練過程輸出

Detectron2 每 **500 iterations** 執行一次 validation（COCO AP 指標），並在 `outputs/checkpoints/maskrcnn/` 下儲存 checkpoint。

```
outputs/checkpoints/maskrcnn/
├── model_final.pth       ← 最終權重
├── model_0004999.pth
├── metrics.json          ← 完整訓練 log
└── last_checkpoint
```

#### 訓練超參數（Mask R-CNN）

| 超參數 | 值 |
|--------|-----|
| Optimizer | SGD |
| Base LR | 0.001 |
| Momentum | 0.9 |
| Batch Size | 2 images/iter |
| Max Iterations | 8000 |
| Warmup Iterations | 1000 |
| LR Decay Steps | 5000, 7000 |
| Backbone | ResNet-50 FPN（COCO pretrained）|
| Loss | Cross-Entropy（cls + bbox + mask）|

---

### Step 3：監控訓練（TensorBoard）

> 需要訓練腳本輸出 TensorBoard log。目前訓練腳本以 stdout 輸出指標，若需視覺化請手動整合 `SummaryWriter`。

啟動 TensorBoard：

```bash
conda activate CrackSeg
tensorboard --logdir=outputs/runs --port=6006
# 開啟瀏覽器：http://localhost:6006
```

---

### Step 4：推論與評估

三個模型都訓練完成後，執行完整的推論與評估流程：

```bat
scripts\run_eval_all.bat
```

流程說明：

1. **CrackSeg 推論**（`CrackSeg` env）：對 test set 產生 PNG binary mask
   ```
   outputs/predictions/deeplabv3plus/
   outputs/predictions/ppliteseg/
   outputs/predictions/pidnet/
   ```

2. **Mask R-CNN 推論**（`CrackPre` env）：產生 PNG binary mask
   ```
   outputs/predictions/maskrcnn/
   ```

3. **統一評估**（`CrackSeg` env）：計算所有模型指標，輸出 CSV
   ```
   outputs/results/metrics_summary.csv
   ```

`metrics_summary.csv` 格式：

| model | iou | dice | precision | recall | fps | inference_ms |
|-------|-----|------|-----------|--------|-----|--------------|
| deeplabv3plus | — | — | — | — | — | — |
| ppliteseg | — | — | — | — | — | — |
| pidnet | — | — | — | — | — | — |
| maskrcnn | — | — | — | — | — | — |

---

### 常見問題

**Q：訓練中斷如何繼續？**

目前 CrackSeg 訓練腳本不支援 resume，需從頭訓練。Mask R-CNN 可透過 Detectron2 的 `resume_or_load` 機制自動從最後的 checkpoint 繼續：
```bash
# 修改 train_maskrcnn.py 中的 resume=False → resume=True
trainer.resume_or_load(resume=True)
```

**Q：GPU 記憶體不足（OOM）怎麼辦？**

在對應的 config YAML 中調低 `batch_size`：
```yaml
# configs/deeplabv3plus.yaml
training:
  batch_size: 4  # 從 8 調低至 4
```

**Q：PP-LiteSeg / PIDNet 提示找不到模組？**

確認 zh320 repo 已 clone 至專案根目錄：
```bash
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
# 確認目錄名稱為 realtime-semantic-segmentation-pytorch
```

**Q：Windows 執行 bat 腳本出現編碼錯誤？**

設定 UTF-8 後再執行：
```bat
chcp 65001
scripts\run_train_crackseg.bat
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
