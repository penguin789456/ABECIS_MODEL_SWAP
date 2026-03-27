# CLAUDE.md

## 專案狀態
- **分支**：`dev/Model_First_traing`
- **Scaffold 完成**：所有模組已建立（data/, configs/, models/, training/, evaluation/, scripts/, notebooks/）
- **Splits 完成**：train(320) / val(68) / test(70)，已 commit
- **待完成**：手動執行訓練（見下方「訓練教學」）
- **已修正**（2026-03-27）：
  - DeepLabV3+ `in_channels` 從 backbone 動態讀取（commit `fc383e0`）
  - albumentations 鎖定 1.4.21，避免 2.x breaking change（commit `e581144`）

## 架構概覽
```
data/dataset.py          ← CrackDataset（512×512 patch, LRU cache, .jpg/.JPG fix）
data/transforms.py       ← Albumentations train/val/test pipelines
data/split.py            ← 70/15/15 split generator（seed=42）
configs/                 ← base.yaml + 4 model overrides
models/deeplabv3plus.py  ← torchvision wrapper, binary output
models/losses.py         ← BCEDiceLoss（weighted BCE + soft Dice）
training/train_crackseg.py      ← unified trainer（DeepLabV3+/PP-LiteSeg/PIDNet）
training/train_maskrcnn.py      ← Detectron2 DefaultTrainer subclass
training/lr_scheduler.py        ← warmup(5ep) + CosineAnnealingLR
evaluation/metrics.py           ← IoU, Dice, Precision, Recall
evaluation/postprocess.py       ← skeletonize, crack_length, continuity_score
evaluation/inference_crackseg.py ← CrackSeg env → PNG masks
evaluation/inference_maskrcnn.py ← CrackPre env → PNG masks
evaluation/evaluate.py          ← env-agnostic evaluator → metrics_summary.csv
scripts/prepare_dataset.py      ← 第一步執行：validate + splits + COCO JSON
scripts/run_eval_all.bat        ← 兩個 env 推論 + 評估
```

## 雙環境推論流程
```
CrackPre: inference_maskrcnn.py → outputs/predictions/maskrcnn/
CrackSeg: inference_crackseg.py → outputs/predictions/{deeplabv3+,ppliteseg,pidnet}/
                    ↓
          evaluate.py（numpy/PIL only）
                    ↓
          outputs/results/metrics_summary.csv
```

## 資料集
```
concreteCrackSegmentationDataset/
├── rgb/   # 原始影像（458 張）
└── BW/    # binary mask（同檔名，白=裂縫）
```
切分：70% train / 15% val / 15% test  
切片：512×512，重疊 128px  
增強（train only）：水平翻轉、旋轉 ±10°、亮度/對比、隨機雜訊

## 模型
| 模型 | 角色 | 來源 |
|------|------|------|
| Mask R-CNN R50-FPN | ABECIS 基準 | Detectron2 |
| DeepLabV3+ | 語意分割基準 | torchvision.models.segmentation |
| PP-LiteSeg-T (STDC1) | 主要模型 | zh320/realtime-semantic-segmentation-pytorch |
| PIDNet | 對照模型 | zh320/realtime-semantic-segmentation-pytorch |

> PP-LiteSeg 與 PIDNet 使用同一 repo，確保訓練框架一致：
> https://github.com/zh320/realtime-semantic-segmentation-pytorch

## 訓練設定
| 超參數 | PyTorch 模型 | Mask R-CNN |
|--------|-------------|------------|
| Optimizer | AdamW | SGD |
| LR | 1e-4 + cosine | 0.001 |
| Batch | 8 | 2 |
| Epochs | 100 + 5 warmup | 50 |
| Loss | BCE + Dice | Cross-Entropy |

## 評估指標
- 分割：IoU（主）、Dice、Precision、Recall
- 效率：FPS、推論時間(ms)、參數量、模型大小
- 後處理：skeletonization、裂縫長度估計、連續性分析

## 訓練教學

### 前置確認
```bash
conda activate CrackSeg
cd H:\碩士\dev\ABECIS_MODEL_SWAP

# 確認 CUDA 正常
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 訓練各模型（依序執行）

```bash
# 1. DeepLabV3+（語意分割基準）
python training/train_crackseg.py --config configs/deeplabv3plus.yaml

# 2. PP-LiteSeg（需先 clone zh320 repo，見下方）
python training/train_crackseg.py --config configs/ppliteseg.yaml

# 3. PIDNet
python training/train_crackseg.py --config configs/pidnet.yaml

# 4. Mask R-CNN（切換環境）
conda activate CrackPre
python training/train_maskrcnn.py --config configs/maskrcnn.yaml
```

### PP-LiteSeg / PIDNet 前置（首次執行）
```bash
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
```

### 監控訓練

**TensorBoard**（開新終端機）
```bash
conda activate CrackSeg
tensorboard --logdir outputs/runs --port 6006
# 瀏覽器開啟 http://localhost:6006
```

**GPU 使用狀況**
```bash
nvidia-smi -l 1
```

**Jupyter Lab**（分析 notebooks）
```bash
conda activate CrackSeg
jupyter lab --notebook-dir=notebooks --port 8888
# 瀏覽器開啟 http://localhost:8888
```

### 輸出位置
```
outputs/
├── checkpoints/{deeplabv3plus,ppliteseg,pidnet,maskrcnn}/
│   ├── best.pth          ← 最佳 val IoU checkpoint
│   └── epoch_10.pth ...  ← 每 10 epoch 儲存
├── predictions/          ← 推論後 PNG masks
└── results/
    └── metrics_summary.csv
```

### 常見錯誤

| 錯誤訊息 | 解法 |
|---------|------|
| `CUDA out of memory` | 降低 batch_size：`configs/deeplabv3plus.yaml` → `batch_size: 4` |
| `in_channels mismatch` | 確認使用最新代碼（`git pull`），已於 fc383e0 修正 |
| albumentations update 警告 | 無害，可忽略；或設 `NO_ALBUMENTATIONS_UPDATE=1` |
| `No module named detectron2` | Mask R-CNN 需切換至 `CrackPre` 環境 |
| PP-LiteSeg/PIDNet import error | 確認已 clone zh320 repo 至專案根目錄 |

---

## 環境

### 兩個獨立 Conda 環境

| 環境 | 用途 | 備註 |
|------|------|------|
| `CrackPre` | Mask R-CNN（Detectron2） | 現有環境，見 ComplateENV.yaml |
| `CrackSeg` | PP-LiteSeg、PIDNet、DeepLabV3+ | 新建，見下方 yaml |

評估統一在獨立腳本中進行，兩個環境各自輸出 binary mask 後再合併計算指標。

### CrackPre 環境建立

```bash
# 1. 安裝 C++ Build Tools（Windows 必要）
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 2. 建立環境
conda env create -n CrackPre -f ComplateENV.yaml
conda activate CrackPre

# 3. 安裝 Detectron2（本地 editable）
pip install -e detectron2 --no-build-isolation
```

> 若 `import detectron2` 失敗，執行 `Fixdetectron2.bat`。

### CrackSeg 環境建立

```bash
conda env create -f CrackSeg_env.yaml
conda activate CrackSeg
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
```

### CrackSeg_env.yaml

```yaml
name: CrackSeg
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - torch==2.1.2
      - torchvision==0.16.2
      - numpy
      - opencv-python
      - pillow
      - albumentations
      - torchmetrics
      - loguru
      - tqdm
      - pyyaml
      - matplotlib
      - scikit-image       # skeletonization 後處理用
      - scipy
      - pandas
```

> torch 2.1.2 為 zh320 repo 已驗證可用的版本（官方要求 >= 1.8.1）。
