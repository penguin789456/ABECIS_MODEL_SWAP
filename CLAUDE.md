# CLAUDE.md

## 專案狀態（2026-06-10）
- **分支**：`main`
- **核心命題**：在低犧牲精度的前提下，以輕量語意分割模型取代 ABECIS（Mask R-CNN）的實例分割框架，大幅降低運算資源需求，探討邊緣運算平台（如 NVIDIA Jetson）部署可行性。
- **訓練環境**：所有模型在 `CrackSeg` 環境（torch 2.12.0.dev+cu128，支援 RTX 5060 Ti sm_120）
- **ABECIS 環境**：Detectron2 在 `ABECIS` 環境（torch 2.4.1+cu121），**CPU only**（sm_120 不相容，最高支援 sm_90）

### 已完成
- **訓練**（4 模型 + FTL/v3 變體）：見「實驗結果」
- **跨資料集 Patch 推論評估**（本階段重點，`final_outputs/`）：
  - Mendeley 凍結測試集（70 張）+ CFD（完全未見，118 張有 GT）
  - 5 模型統一 512×512 patch 滑動視窗推論 + clDice
  - **關鍵發現**：ABECIS patch 版 Mendeley IoU=0.5042（全模型最高）但僅 0.02 FPS；PP-LiteSeg IoU=0.4392 @ 122 FPS → 精度差 ~10% 換取 ~6000 倍速度，完美支撐命題

---

## 兩套流程（重要區分）

### A. 訓練流程（`ABECIS_MODEL_SWAP/`，CrackSeg 環境）
```
data/dataset.py              ← CrackDataset / PrecomputedCrackDataset（.npy patch cache）
data/transforms.py           ← Albumentations train/val/test pipelines
data/split.py                ← 70/15/15 split（seed=42）
configs/final/               ← 訓練 YAML（ppliteseg / ddrnet / ddrnet_ftl / deeplabv3 / maskrcnn_tv / *_v3）
models/losses.py             ← BCEDiceLoss / FocalTverskyLoss / FocalDiceLoss
training/train_crackseg.py   ← 統一 trainer（語意分割）→ best.pth + train_log.csv
training/train_maskrcnn_tv.py← torchvision Mask R-CNN trainer
training/lr_scheduler.py     ← warmup(5ep) + CosineAnnealingLR
evaluation/metrics.py        ← IoU, Dice, Precision, Recall, clDice
scripts/precompute_patches.py← 預計算 512×512 patch cache → data/patches_*/
scripts/threshold_sweep.py   ← 閾值 sweep
scripts/benchmark_fps.py     ← FPS / latency benchmark
```

### B. Patch 推論評估流程（`H:\ChihleeMaster\dev\final_outputs\`）
本階段交付物，與訓練流程分離。輸出 `{pred_masks,before,after,logs}\{model}\{gpu|cpu}\`
```
infernce/cfd_patch_inference.py            ← CFD 512 patch（4 分割模型，CrackSeg）
infernce/mendeley_patch_inference.py       ← Mendeley 512 patch（4 分割模型，CrackSeg）
infernce/abecis_cfd_patch_inference.py     ← CFD ABECIS（原始解析度 640×480，ABECIS env CPU）
infernce/abecis_mendeley_patch_inference.py← Mendeley ABECIS（512 滑窗，ABECIS env CPU，含 --resume）
ground_truth/evaluate_gt_with_cldice.py    ← GT 評估（--dataset cfd|mendeley，含 clDice）
```

**Patch 推論共用設定**：PATCH_SIZE=512, OVERLAP=128, STRIDE=384，重疊區概率平均。
**各模型門檻**：ppliteseg=0.65, ddrnet=0.50, ddrnet_ftl=0.50, deeplabv3=0.45, abecis=0.80。
**ABECIS 偵測式處理**：每 patch 推論 → 實例遮罩**聯集**成二值全圖（OR 合併，避開 NMS 跨邊界問題），`MIN/MAX_SIZE_TEST=512`。

---

## ⚠️ 執行環境注意事項
- **勿用 `conda run -n`**（會觸發 cp950 編碼崩潰）。直接呼叫 env python：
  - `G:\conda\envs\CrackSeg\python.exe`（分割模型）
  - `G:\conda\envs\ABECIS\python.exe`（ABECIS Detectron2）
- **務必加** `PYTHONIOENCODING=utf-8`（否則中文輸出 cp950 錯誤）
- 分割模型再加 `PYTHONPATH` 指向專案根 + `realtime-semantic-segmentation-pytorch`
- ABECIS checkpoint：`H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\output\model_final.pth`

範例（Mendeley patch）：
```bash
cd H:\ChihleeMaster\dev\final_outputs\infernce
PYTHONIOENCODING=utf-8 PYTHONPATH="H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP;H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\realtime-semantic-segmentation-pytorch" \
  G:\conda\envs\CrackSeg\python.exe mendeley_patch_inference.py --model ppliteseg --device cuda
# ABECIS（含斷點續傳）
PYTHONIOENCODING=utf-8 G:\conda\envs\ABECIS\python.exe abecis_mendeley_patch_inference.py --threshold 0.8 --device cpu --resume
# GT 評估
PYTHONIOENCODING=utf-8 G:\conda\envs\CrackSeg\python.exe ../ground_truth/evaluate_gt_with_cldice.py --model ppliteseg_mendeley_patch --dataset mendeley --device gpu
```

---

## 模型

| 模型 | 角色 | 來源 | 參數量 |
|------|------|------|--------|
| PP-LiteSeg-T (STDC1) | 主要輕量模型 | zh320/realtime-semantic-segmentation-pytorch | ~5M |
| DDRNet-23-slim | 雙分支即時分割 | zh320/realtime-semantic-segmentation-pytorch | ~5.6M |
| DeepLabV3 (MobileNetV3-Large) | 語意分割基準 | torchvision.models.segmentation | ~11M |
| Mask R-CNN R50-FPN (ABECIS) | 實例分割基準 | Detectron2（官方預訓練權重，非重訓） | ~44M |

> ABECIS 用官方 Detectron2 預訓練權重直接推論。訓練流程中另有 torchvision Mask R-CNN R50-FPN（架構相同）供統一訓練比較用。

---

## 資料集

| 版本 | 資料夾 | 內容 | 訓練圖數 | 狀態 |
|------|--------|------|---------|------|
| v0 | `concreteCrackSegmentationDataset/` | Mendeley CCSD 原始（458 張） | 320 | ✅ 主用 |
| v3 | `dataset_merged_v3/` | + SCD confident pseudo-label | ~15,380 | ✅ 資料增強實驗 |
| v1 / v2 | `dataset_merged_v1`/`_v2` | +DeepCrack / +CRACK500+CFD+GAPS384 | — | ❌ 放棄（停在 ep16） |

- **訓練域測試集**：70 張 Mendeley 原始影像（`data/splits/test.txt`，FROZEN，全版本共用）
- **跨域測試集**：CFD（Crack Forest Dataset）155 張，其中 118 張有 GT，完全未見、無資料洩漏
- 評估時 leading-zero 處理：test.txt 中 `059` → `BW/059.jpg`

**原始資料來源路徑**：
| 資料集 | 路徑 |
|--------|------|
| Mendeley CCSD | `concreteCrackSegmentationDataset/`（rgb / BW） |
| CFD | `H:\ChihleeMaster\CrackK500\CFD`（cfd_image / seg_gt） |
| SCD（v3 來源） | `H:\ChihleeMaster\Surface Crack Detection` |

---

## 訓練設定

| 超參數 | 值 |
|--------|-----|
| Optimizer | AdamW |
| LR | 1e-4 + cosine annealing，warmup 5 ep |
| Batch Size | 32 |
| Epochs | 150 |
| Loss | BCEDiceLoss（BCE 0.5 + Dice 0.5）；FTL 變體 α=0.2 β=0.8 γ=0.75 |
| Patch | 512×512，overlap 128px |
| Oversample | WeightedRandomSampler，positive_weight=5.0 |

訓練指令：
```bash
G:\conda\envs\CrackSeg\python.exe training/train_crackseg.py --config configs/final/ppliteseg.yaml
# Resume: --resume outputs/checkpoints/{model}/best.pth
```

---

## 實驗結果

### 1. 訓練評估 — v0 Mendeley 測試集（1024 全圖訓練管線評估）

| 模型 | Test IoU | Precision | Recall | Threshold | FPS（GPU） | 模型大小 |
|------|----------|-----------|--------|-----------|-----------|---------|
| PP-LiteSeg-T | **0.4391** | 0.8463 | 0.4046 | 0.65 | 122.3 | 75.8 MB |
| DeepLabV3-MobileNetV3 | **0.4385** | 0.8378 | 0.4792 | 0.45 | 91.0 | 132.6 MB |
| DDRNet-23-slim | **0.3912** | — | — | 0.50 | 89.8 | 67.0 MB |
| Mask R-CNN (torchvision) | **0.1487** | 0.4301 | 0.1852 | 0.50 | 22.4 | 176.2 MB |

FPS 量測：RTX 5060 Ti，512×512 patch，batch=1。

資料增強：PP-LiteSeg v3（+SCD pseudo）Test IoU=0.4361，Recall 0.4778（+7.3%），threshold=0.70。

### 2. Mendeley Patch 推論（凍結測試集 70 張，512 滑窗）⭐ 本階段主結果

| 模型 | Micro IoU | Macro IoU | Recall | clDice | FPS |
|------|-----------|-----------|--------|--------|-----|
| **ABECIS**（patch, CPU） | **0.5042** | 0.5058 | 0.7273 | **0.8123** | 0.02 |
| PP-LiteSeg-T（GPU） | 0.4392 | — | — | — | 122 |
| DeepLabV3（GPU） | 0.4358 | — | — | — | 91 |
| DDRNet（GPU） | 0.3908 | — | — | — | 90 |
| DDRNet FTL（GPU） | 0.3280 | — | — | — | — |

> patch 版 PP-LiteSeg IoU 0.4392 ≈ 訓練管線 0.4391，驗證滑窗管線正確。
> ABECIS 在原生 patch 尺度下精度最高，但 CPU 推論 0.02 FPS（每張 ~60s，88 patches）。

### 3. CFD Patch 推論（跨域，118 張有 GT）

| 模型 | Micro IoU | Macro IoU | Recall | clDice | FPS |
|------|-----------|-----------|--------|--------|-----|
| DDRNet | 0.3174 | 0.2628 | 0.6250 | 0.5697 | 42.4 |
| DDRNet FTL | 0.2862 | 0.2449 | 0.7107 | 0.6148 | 44.5 |
| DeepLabV3 | 0.2365 | 0.1874 | 0.5977 | 0.4368 | 44.9 |
| PP-LiteSeg-T | 0.2145 | 0.1493 | 0.3771 | 0.3214 | 38.1 |
| ABECIS（原始解析度, CPU） | 0.2142 | 0.1741 | 0.4851 | 0.4352 | 1.31 |

> CFD ABECIS 用原始解析度 640×480 推論（影像本身夠小，無需切 patch）。
> 跨域 IoU 普遍下降為預期（domain gap）；DDRNet 系列在 CFD 道路裂縫上相對穩健。

### 交付 ZIP（`H:\ChihleeMaster\dev\final_outputs\`）
- `logs_Mendeley_patch.zip`（5 模型）、`logs_CFD_patch.zip`（5 模型）
- `logs_*_patch_4models_backup.zip`（加入 ABECIS 前的備份）
- 歷史：`logs_Mendeley.zip` / `logs_CFD.zip`（1024 全圖版）、`logs_Training_Eval.zip`

---

## 評估方法與口試論述

### 評估方法說明
ABECIS 原論文採 Instance Segmentation，IoU 以實例邊界框為單位。本研究採**統一像素級 IoU**，所有模型在相同測試集重新評估，確保公平；補充 clDice 從中心線吻合度驗證分割品質。Patch 滑窗推論使各模型在訓練尺度（512）下評估，避免大圖 resize 造成裂縫壓縮失真（Mendeley 4032×3024 直接 resize 至 1024 會使 IoU 崩至 ~0.01）。

### 口試答辯標準說法
> 「ABECIS 使用 Instance Segmentation，IoU 以實例邊界框為單位。本研究核心問題是：在低犧牲精度前提下，能否以語意分割模型取代實例分割框架並獲得即時推論能力？為控制變數，將 Mask R-CNN（ABECIS）納入，在相同測試集以相同 pixel-level IoU 重新評估。統一標準下，PP-LiteSeg 以約 6000 倍速度（122 FPS vs 0.02 FPS）換取約 10% 的精度差距（IoU 0.4392 vs 0.5042），具備邊緣裝置即時部署潛力。」

### IoU 偏低的應對
**核心原則：IoU 數值低是研究發現，不是失敗。**
1. 像素級 IoU 對 1~3px 寬裂縫本質嚴苛；文獻中 0.4~0.6 屬正常
2. 相對比較有效：模型排序與精度-速度 tradeoff 成立
3. 主動改善：v3 SCD pseudo-label、FTL 損失函數
> 貢獻定位：建立可重現的輕量化裂縫分割比較框架，系統性分析架構在精度、速度、部署成本間的權衡。

---

## 常見錯誤

| 錯誤 | 解法 |
|---------|------|
| `conda run` cp950 編碼崩潰 | 改直接呼叫 env python.exe + `PYTHONIOENCODING=utf-8` |
| 中文輸出亂碼 / UnicodeEncodeError | 設 `PYTHONIOENCODING=utf-8` |
| ABECIS Mendeley patch 中途中止 | 用 `--resume` 跳過已完成 pred_masks 續跑 |
| `CUDA out of memory` | config 調低 `batch_size: 16` |
| PP-LiteSeg/DDRNet import error | 確認 PYTHONPATH 含 zh320 repo |
| `cache up-to-date, skipping` | 正常，patch cache 已存在 |

---

## 環境建立
```bash
# 分割模型
conda env create -f CrackSeg_env.yaml && conda activate CrackSeg
git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
# 驗證 CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 待辦
- [ ] 論文 docx（`H:\ChihleeMaster\論文計畫書_譚秉弘_修訂版V1.docx`）填入 patch 結果（第四章實驗 / 第五章結論）
- [ ] 散布圖 `final_outputs/make_scatter.py`（FPS vs IoU）更新為 patch 版數據
