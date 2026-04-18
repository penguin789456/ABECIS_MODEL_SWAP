# CLAUDE.md

## 專案狀態（2026-04-19）
- **分支**：`main`
- **環境**：所有模型統一在 `CrackSeg` 環境，不再需要 CrackPre/Detectron2
- **基準已完成**：PP-LiteSeg-T IoU=0.4391、DDRNet-23-slim IoU=0.3912
- **進行中**：PP-LiteSeg v1（Mendeley+DeepCrack）/ v2（全資料集）訓練，各 150 epochs
- **Patch cache 就緒**：patches_v1（28,789）、patches_v2（34,279）

## 架構概覽
```
data/dataset.py              ← CrackDataset / PrecomputedCrackDataset（.npy patch cache）
data/transforms.py           ← Albumentations train/val/test pipelines
data/split.py                ← 70/15/15 split generator（seed=42）
configs/final/               ← 訓練用 YAML（ppliteseg / v1 / v2 / ddrnet / deeplabv3 / maskrcnn）
models/deeplabv3_mobilenet.py  ← torchvision wrapper, binary output
models/losses.py             ← BCEDiceLoss / FocalTverskyLoss / FocalDiceLoss
training/train_crackseg.py   ← unified trainer（全模型）→ best.pth + train_log.csv + train_info.txt
training/train_maskrcnn_tv.py← torchvision Mask R-CNN trainer
training/lr_scheduler.py     ← warmup(5ep) + CosineAnnealingLR
evaluation/metrics.py        ← IoU, Dice, Precision, Recall, clDice
evaluation/postprocess.py    ← skeletonize（備用，未在評估流程中使用）
evaluation/inference_crackseg.py  ← CrackSeg env → PNG masks（語意分割）
evaluation/inference_maskrcnn_tv.py ← CrackSeg env → PNG masks（Mask R-CNN）
evaluation/evaluate.py       ← env-agnostic evaluator → metrics_summary.csv
scripts/prepare_dataset.py   ← validate + splits + COCO JSON
scripts/prepare_external.py  ← 合併外部資料集（DeepCrack/CRACK500/CFD/GAPS384）
scripts/precompute_patches.py← 預計算 512×512 patch cache → data/patches_*/
scripts/threshold_sweep.py   ← 閾值 sweep，找最佳 inference threshold
scripts/benchmark_loader.py  ← DataLoader 吞吐量測試（batch × workers）
scripts/benchmark_fps.py     ← 模型 FPS / latency benchmark
```

## 推論流程（單一 CrackSeg 環境）
```
CrackSeg: inference_maskrcnn_tv.py  → outputs/predictions/maskrcnn/
CrackSeg: inference_crackseg.py     → outputs/predictions/{deeplabv3_mobilenet,ppliteseg,ddrnet}/
                    ↓
          evaluate.py（numpy/PIL only）
                    ↓
          outputs/results/metrics_summary.csv
```

## 資料集階層

| 版本 | 資料夾 | 內容 | 訓練圖數 |
|------|--------|------|---------|
| v0 | `concreteCrackSegmentationDataset/` | Mendeley 原始 | 320 |
| v1 | `dataset_merged_v1/` | + DeepCrack (537) | 857 |
| v2 | `dataset_merged_v2/` | + CRACK500(1896) + CFD(118) + GAPS384(509) | 3,380 |

測試集：**70 張 Mendeley 原始影像（全版本共用，FROZEN）**

**Patch cache 位置：**
```
data/patches/      ← v0（原始 Mendeley）
data/patches_v1/   ← v1（28,789 train）
data/patches_v2/   ← v2（34,279 train）
data/splits/       ← v0 splits
data/splits_v1/    ← v1 splits
data/splits_v2/    ← v2 splits
```

## 模型

| 模型 | 角色 | 來源 | 環境 |
|------|------|------|------|
| Mask R-CNN R50-FPN | ABECIS 基準（~44M） | torchvision.models.detection | CrackSeg |
| DeepLabV3 (MobileNetV3-Large) | 語意分割基準（~11M） | torchvision.models.segmentation | CrackSeg |
| DDRNet-23-slim | 雙分支即時分割（~5.6M） | zh320/realtime-semantic-segmentation-pytorch | CrackSeg |
| PP-LiteSeg-T (STDC1) | 主要輕量模型（~5M） | zh320/realtime-semantic-segmentation-pytorch | CrackSeg |

## 訓練設定

| 超參數 | 值 |
|--------|-----|
| Optimizer | AdamW |
| LR | 1e-4 + cosine annealing |
| Warmup | 5 epochs |
| Batch Size | 32 |
| Epochs | 150 |
| Loss | BCEDiceLoss（BCE 0.5 + Dice 0.5） |
| Patch Size | 512×512，overlap 128px |
| Oversample | WeightedRandomSampler，positive_weight=5.0 |

## 訓練指令

```bash
conda activate CrackSeg
cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP

# 基準（Mendeley only）
python training/train_crackseg.py --config configs/final/ppliteseg.yaml

# v1（+ DeepCrack）
python training/train_crackseg.py --config configs/final/ppliteseg_v1.yaml

# v2（全資料集）
python training/train_crackseg.py --config configs/final/ppliteseg_v2.yaml

# Resume
python training/train_crackseg.py --config configs/final/ppliteseg_v2.yaml \
    --resume outputs/checkpoints/ppliteseg_v2/best.pth
```

## 訓練輸出

每次訓練在 `outputs/checkpoints/{model}/{timestamp}/` 產生：
```
best.pth          ← 最佳 val IoU checkpoint
epoch_010.pth     ← 每 10 epoch
train_log.csv     ← 每 epoch 指標（loss/IoU/Dice/P/R/lr/time/mem）
train_info.txt    ← 訓練參數快照（model/training/loss/dataset/system）
```

## 評估指標
- 分割：IoU（主）、Dice、Precision、Recall、clDice（中心線 Dice）
- 效率：FPS、推論時間(ms)、參數量、模型大小
- 後處理：clDice 驗證分割品質（`evaluation/metrics.py` 內實作）

## 實驗結果

| 模型 | 資料集 | Test IoU | FPS | 參數量 | Config |
|------|--------|----------|-----|--------|--------|
| PP-LiteSeg-T | v0（基準） | **0.4391** | 30+ | 5M | `ppliteseg.yaml` |
| PP-LiteSeg-T | v1（+DeepCrack） | 訓練中 | — | 5M | `ppliteseg_v1.yaml` |
| PP-LiteSeg-T | v2（全資料） | 訓練中 | — | 5M | `ppliteseg_v2.yaml` |
| DDRNet-23-slim | v0（基準） | **0.3912** | ~40 | 5.6M | `ddrnet.yaml` |
| DeepLabV3-MobileNetV3 | v0 | 訓練中 | ~30 | 11M | `deeplabv3_mobilenet.yaml` |
| Mask R-CNN R50-FPN | v0 | 訓練中 | ~3 | 44M | `maskrcnn.yaml` |

## 研究核心命題

> **在低犧牲精度的前提下，以輕量語意分割模型取代 ABECIS 的實例分割框架，大幅降低運算資源需求，並探討部署於邊緣運算平台（如 NVIDIA Jetson）的可行性。**

### 研究缺口說明

ABECIS 採用 Mask R-CNN 實現高精度裂縫偵測，然而其龐大的運算需求（44M 參數、~3 FPS）
使系統必須依賴高規格 GPU 電腦執行推論，難以嵌入無人機或行動裝置進行現場即時處理。
本研究探討能否以輕量語意分割模型在低犧牲精度的前提下，大幅降低運算資源需求，
實現邊緣裝置上的即時裂縫偵測。

### 評估方法說明

ABECIS 原論文採用 Instance Segmentation，IoU 以實例邊界框為單位計算。
本研究採用**統一的像素級 IoU**，對所有模型在相同測試集重新評估，確保比較公平。
補充指標 clDice 從中心線吻合度驗證分割品質。

### 口試答辯標準說法

> 「ABECIS 原論文使用 Instance Segmentation 框架，IoU 以實例邊界框為單位計算，這在偵測離散物件時是標準做法。
> 本研究的核心問題是：**在低犧牲精度的前提下，能否以語意分割模型取代實例分割框架，同時獲得即時推論能力？**
> 為了使比較具備控制變數，我們將 Mask R-CNN 納入本研究作為基準模型，在相同測試集以相同 pixel-level IoU 重新評估。
> 在此統一標準下，PP-LiteSeg 以 __ 倍速度換取約 __% 的精度差距，具備邊緣裝置即時部署潛力。」

### IoU 偏低的口試應對

**核心原則：IoU 數值低是研究發現，不是研究失敗。**

1. **指標本質**：像素級 IoU 對 1~3px 寬裂縫本質嚴苛；文獻中 0.4~0.6 屬正常範圍
2. **相對比較有效**：四模型排序與精度-速度 tradeoff 分析仍成立
3. **主動改善**：外部資料（v1/v2 實驗）、損失函數調整、知識蒸餾

研究貢獻定位：「建立可重現的輕量化裂縫分割比較框架，系統性分析不同架構在精度、速度與部署成本之間的權衡關係」

### 待補充（訓練完成後填入）
- [ ] PP-LiteSeg v1 Test IoU（threshold sweep 後）
- [ ] PP-LiteSeg v2 Test IoU（threshold sweep 後）
- [ ] Mask R-CNN pixel IoU
- [ ] DeepLabV3-MobileNetV3 pixel IoU
- [ ] 各模型實測 FPS

---

## 完整工作流程

### 新資料集加入
```bash
# 1. 合併
python scripts/prepare_external.py \
    --deepcrack_dir    H:\ChihleeMaster\DeepCrack \
    --crack500_dir     H:\ChihleeMaster\CrackK500\CRACK500\traincrop \
    --cfd_img_dir      H:\ChihleeMaster\CrackK500\CFD\cfd_image \
    --cfd_mask_dir     H:\ChihleeMaster\CrackK500\CFD\seg_gt \
    --gaps384_img_dir  H:\ChihleeMaster\CrackK500\GAPS384\croppedimg \
    --gaps384_mask_dir H:\ChihleeMaster\CrackK500\GAPS384\croppedgt \
    --output_root      dataset_merged_v2 \
    --output_splits_dir data/splits_v2

# 2. 預計算 patches
python scripts/precompute_patches.py --config configs/final/ppliteseg_v2.yaml

# 3. 訓練
python training/train_crackseg.py --config configs/final/ppliteseg_v2.yaml
```

### 閾值 Sweep（訓練完成後）
```bash
python scripts/threshold_sweep.py --config configs/final/ppliteseg_v2.yaml
```

### 監控
```bash
tensorboard --logdir outputs/runs --port 6006
nvidia-smi -l 1
```

---

## 常見錯誤

| 錯誤訊息 | 解法 |
|---------|------|
| `CUDA out of memory` | config 調低 `batch_size: 16` |
| `No module named 'models.losses'` in worker | 已修正（Windows spawn 需 PYTHONPATH 傳遞） |
| albumentations update 警告 | 無害；或設 `NO_ALBUMENTATIONS_UPDATE=1` |
| PP-LiteSeg/DDRNet import error | 確認已 clone zh320 repo 至專案根目錄 |
| `cache up-to-date, skipping` | 正常，patch cache 已存在不需重新計算 |

---

## 環境

### CrackSeg 環境建立
```bash
conda env create -f CrackSeg_env.yaml
conda activate CrackSeg
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
```

### 驗證 CUDA
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
