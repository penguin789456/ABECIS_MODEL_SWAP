# CLAUDE.md

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
