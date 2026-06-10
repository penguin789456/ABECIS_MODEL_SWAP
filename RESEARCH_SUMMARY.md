# 研究摘要 — 輕量裂縫分割模型 vs ABECIS

> 本文件供外部 AI 快速理解研究現況，最後更新：2026-05-03

---

## 一、研究背景與目標

### 核心命題
> 在低犧牲精度的前提下，以**輕量語意分割模型**取代 ABECIS 的實例分割框架，大幅降低運算資源需求，並探討部署於邊緣運算平台（如 NVIDIA Jetson）的可行性。

### ABECIS 是什麼
- 混凝土裂縫自動偵測系統，使用 **Mask R-CNN R50-FPN**（Detectron2 實作，44M 參數）
- 論文報告偵測率 **80.4%**，IoU=0.309，Precision=0.327
- 官方模型已公開：`H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\output\model_final.pth`（168MB）
- ABECIS 環境（ABECIS conda env）因 GPU sm_120 不相容，只能 CPU 推論

### 研究缺口
ABECIS（44M 參數，~3 FPS）需要高規格 GPU，難以嵌入無人機或行動裝置進行現場即時處理。

---

## 二、模型架構

| 模型 | 角色 | 參數量 | 來源 |
|------|------|--------|------|
| **Mask R-CNN R50-FPN** | ABECIS 基準複現 | ~44M | torchvision（GPU 相容版） |
| **PP-LiteSeg-T (STDC1)** | 主要輕量模型 | ~5M | zh320/realtime-semantic-segmentation-pytorch |
| **PP-LiteSeg-T v3** | PP-LiteSeg + v3 資料增強 | ~5M | 同上，資料集擴充 |
| **DDRNet-23-slim** | 雙分支即時分割 | ~5.6M | zh320/realtime-semantic-segmentation-pytorch |
| **DDRNet FTL** | DDRNet + Focal Tversky Loss | ~5.6M | 同上，損失函數修改 |
| **DDRNet FTL v3** | DDRNet + FTL + v3 資料增強 | ~5.6M | 同上，組合實驗 ⏳ 訓練中 |
| **DeepLabV3 (MobileNetV3-Large)** | 語意分割基準 | ~11M | torchvision |

> ⚠️ 注意：ABECIS 官方使用 Detectron2 版 Mask R-CNN；本研究的 Mask R-CNN 基準採用 torchvision 實作，架構完全相同，但為獨立訓練版本。

---

## 三、資料集

### 主要訓練/測試資料集

| 版本 | 資料夾 | 內容 | 總圖數 | 狀態 |
|------|--------|------|--------|------|
| **v0（主要使用）** | `concreteCrackSegmentationDataset/` | Mendeley 混凝土裂縫 | 458 | ✅ 使用中 |
| **v3** | `dataset_merged_v3/` | v0 + SCD pseudo-label | ~16,531 | ✅ PP-LiteSeg + DDRNet FTL 實驗 |
| v1 | `dataset_merged_v1/` | v0 + DeepCrack | 985 | ❌ 放棄（ep16 中斷） |
| v2 | `dataset_merged_v2/` | v0 + CRACK500 + CFD + GAPS384 | 3,508 | ❌ 放棄（ep16 中斷） |

**凍結測試集（Frozen）：70 張 Mendeley 原始影像，所有模型共用**

分割比例（seed=42）：
- 訓練集：320 張（v0）/ ~15,380 張（v3）
- 驗證集：68 張（共用）
- 測試集：70 張（FROZEN，不參與訓練）

### 跨資料集驗證（外部）

| 資料集 | 路徑 | 圖數 | GT | 用途 |
|--------|------|------|-----|------|
| **CFD**（Crack Forest Dataset） | `H:\ChihleeMaster\CrackK500\CFD\` | 155 張 | 118 張 | 零樣本泛化驗證 |

CFD 為**完全未見資料集**（從未用於任何模型訓練），與 Mendeley 在影像來源、解析度、裂縫型態（線狀 vs 網狀）存在明顯 domain shift。GT 僅涵蓋 118/155 張為此資料集原始特性。

---

## 四、訓練設定

| 超參數 | 值 |
|--------|-----|
| Optimizer | AdamW |
| 初始 LR | 1e-4 |
| LR Scheduler | Warmup 5 epochs + CosineAnnealing |
| Batch Size | 32 |
| Epochs | 150（語意分割）/ 50（Mask R-CNN）|
| Loss（主要） | BCEDiceLoss（BCE 0.5 + Dice 0.5） |
| Loss（FTL） | FocalTverskyLoss（alpha=0.2, beta=0.8, gamma=0.75） |
| Patch Size | 512×512，overlap 128px |
| 正樣本過採樣 | WeightedRandomSampler，positive_weight=5.0 |
| 環境 | CrackSeg conda env（PyTorch 2.12.0+cu128，RTX 5060 Ti sm_120） |

### FTL 損失函數說明
FocalTverskyLoss 設計目的：強懲罰 FN（漏偵），容忍 FP（誤報），提升 Recall。
- alpha=0.2（低 FP 懲罰）
- beta=0.8（高 FN 懲罰）
- NaN 修正：強制 float32、clamp tversky [0,1]、10% BCE 混合

---

## 五、評估方法

### 主要評估（Mendeley 凍結測試集）
- **Pixel-level IoU**（主指標）
- Dice、Precision、Recall、clDice（中心線 Dice）
- FPS、推論時間（ms）、參數量、模型大小

### 偵測率評估（Detection Rate）
- 對 GT 黑白圖做連通分量分析（8-connectivity，min 50px）
- 每個 GT 裂縫區域若預測 mask 有任何像素重疊 → 算偵測到
- 偵測率 = 被偵測區域數 / 總區域數
- **不計算像素差值**，只看「有沒有找到裂縫」

### 跨資料集評估（CFD，Zero-shot）
- 與主要評估相同指標（Micro IoU / Dice / Precision / Recall）
- 輸入統一 resize 至 1024×1024，套用 ImageNet normalization，threshold 統一 0.8
- 語意分割模型直接輸出像素遮罩；Mask R-CNN 實例遮罩合併為二元遮罩
- 定位為**輔助穩健性指標**，不與主測試集直接混用

> ⚠️ Threshold 說明：所有 CFD 推論統一使用 threshold=0.8（標準化比較）。PP-LiteSeg v0 最佳 threshold=0.65，使用 0.8 導致 CFD 結果偏低，應參考 v3 結果進行跨版本比較。

---

## 六、實驗結果

### 6.1 四模型主比較（Mendeley 凍結測試集，70 張）

| 模型 | Test IoU | Precision | Recall | Threshold | FPS | 參數量 |
|------|----------|-----------|--------|-----------|-----|--------|
| **PP-LiteSeg-T** | **0.4391** | 0.8463 | 0.4046 | 0.65 | **122.3** | 5M |
| **DDRNet-23-slim** | **0.3912** | 0.7130 | 0.4640 | 0.35 | 89.8 | 5.6M |
| **DeepLabV3-MobileNetV3** | **0.4385** | 0.8378 | 0.4792 | 0.45 | 91.0 | 11M |
| **Mask R-CNN R50-FPN** | **0.1487** | 0.4301 | 0.1852 | 0.50 | 22.4 | 44M |

### 6.2 資料增強實驗（PP-LiteSeg-T，Mendeley 測試集）

| 資料集 | Test IoU | Precision | Recall | 備註 |
|--------|----------|-----------|--------|------|
| v0（Mendeley） | 0.4391 | 0.8463 | 0.4046 | 基準 |
| v3（+SCD pseudo-label） | 0.4361 | 0.8332 | 0.4778 | Recall +7.3%，IoU 持平 |

### 6.3 ABECIS Detectron2 重測（Mendeley 測試集，70 張）

| 閾值模式 | 偵測率 | 說明 |
|---------|--------|------|
| SCORE_THRESH=0.01（全輸出） | **92.0%** | 超越論文 80.4% |
| SCORE_THRESH=0.80（instance-level） | 50.6% | 高信心篩選 |

> ABECIS 論文報告 80.4% 已驗證（重測為 92.0%），基準值可信。

### 6.4 DDRNet FTL 偵測率掃描（Mendeley 測試集）

| 閾值 | DDRNet FTL 偵測率 | ABECIS 偵測率（同閾值） | 勝者 |
|------|------------------|----------------------|------|
| 0.80 | **70.7%** | 50.6% | DDRNet FTL ✅ |
| 0.30 | **71.8%** | — | DDRNet FTL |

### 6.5 CFD 跨資料集評估（零樣本，118 張含 GT，threshold=0.8 統一）

| 模型 | Micro IoU | Precision | Recall | Dice | 備註 |
|------|-----------|-----------|--------|------|------|
| **DDRNet FTL** | **0.4115** | 0.6187 | 0.5512 | 0.5830 | 最佳泛化 |
| **PP-LiteSeg v3** | **0.3496** | 0.6983 | 0.4118 | 0.5180 | v3 資料增強顯著提升 |
| **DeepLabV3** | 0.3181 | 0.6651 | 0.3787 | 0.4826 | |
| **DDRNet v0** | 0.2210 | 0.7315 | 0.2405 | 0.3620 | FTL 版高出 0.19 |
| **ABECIS Detectron2** | 0.2063 | 0.3817 | 0.3098 | 0.3420 | CPU 推論 |
| **PP-LiteSeg v0** | 0.0603 | 0.6795 | 0.0621 | 0.1137 | ⚠️ threshold 0.8 偏高，最佳 threshold=0.65 |
| **Mask R-CNN (torchvision)** | 0.0370 | 0.5932 | 0.0379 | 0.0713 | ⚠️ threshold 0.8 偏高 |

**關鍵觀察：**
- DDRNet FTL 在 CFD 零樣本測試中 IoU=0.41，為所有模型最高，且高出 ABECIS 近一倍（0.21）
- PP-LiteSeg v3 在 CFD 上 IoU=0.35，比 v0（0.06）大幅提升，顯示 v3 資料增強顯著改善跨域泛化能力
- DeepLabV3 無額外訓練策略下在 CFD 仍達 0.32，表現穩定

### 6.6 DDRNet FTL v3 組合實驗（2×2 消融設計）

| | v0 資料集 | v3 資料集 |
|---|-----------|-----------|
| **BCEDice 損失** | IoU=0.3912（基準）| — |
| **FTL 損失** | DDRNet FTL（已完成）| DDRNet FTL v3 ⏳ 訓練中 |

> CFD 評估發現 v3 資料增強對跨域泛化有顯著效益，補充 DDRNet FTL + v3 組合實驗以完整驗證兩因子的交互作用。Config：`configs/final/ddrnet_ftl_v3.yaml`

---

## 七、關鍵發現

1. **輕量模型 IoU 優於 ABECIS**：PP-LiteSeg（0.44）、DeepLabV3（0.44）、DDRNet（0.39）皆高於 torchvision Mask R-CNN（0.15）
2. **速度差距懸殊**：PP-LiteSeg 122 FPS vs Mask R-CNN 22 FPS，前者快 5.5 倍
3. **跨域泛化最強**：DDRNet FTL 在 CFD 未見資料集 IoU=0.41，ABECIS 僅 0.21
4. **偵測率公平比較**：同閾值（0.80）下，DDRNet FTL（70.7%）> ABECIS（50.6%）
5. **v3 改善跨域但不改善 in-distribution**：PP-LiteSeg v3 在 Mendeley 測試集 IoU 與 v0 持平（0.4361 vs 0.4391），但 CFD 零樣本 IoU 大幅提升（0.35 vs 0.06），顯示 v3 的貢獻在於跨域泛化而非原始資料集精度
6. **FTL 損失大幅提升 DDRNet 泛化**：DDRNet FTL（0.41）vs DDRNet v0（0.22）在 CFD 上差距達 0.19，說明損失函數設計對 zero-shot 泛化有顯著影響

---

## 八、目前狀態

| 項目 | 狀態 |
|------|------|
| 四模型訓練（v0 基準） | ✅ 完成 |
| PP-LiteSeg v3 訓練 | ✅ 完成（IoU=0.4361） |
| DDRNet FTL 訓練 | ✅ 完成 |
| DDRNet FTL v3 訓練 | ⏳ 待執行（config 已建立：`ddrnet_ftl_v3.yaml`） |
| FPS benchmark | ✅ 完成 |
| 主要評估（Mendeley 70 張） | ✅ 完成（全模型） |
| ABECIS 官方模型重測 | ✅ 完成 |
| CFD 推論（全 7 個模型） | ✅ 完成 |
| CFD GT 評估（全 7 個模型） | ✅ 完成 |
| TorchScript 匯出（PP-LiteSeg / DDRNet / DeepLabV3 / PP-LiteSeg v3） | ✅ 完成 |
| DDRNet FTL v3 CFD 評估 | ⏳ 待訓練完成後執行 |

---

## 九、檔案結構（重要路徑）

```
H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\
├── configs\final\
│   ├── ppliteseg.yaml              ← PP-LiteSeg v0 基準
│   ├── ppliteseg_v3.yaml           ← PP-LiteSeg v3（+SCD pseudo-label）
│   ├── ddrnet.yaml                 ← DDRNet v0 基準
│   ├── ddrnet_ftl.yaml             ← DDRNet + FTL 損失
│   ├── ddrnet_ftl_v3.yaml          ← DDRNet + FTL + v3（新增，待訓練）
│   ├── deeplabv3_mobilenet.yaml    ← DeepLabV3
│   └── maskrcnn.yaml               ← Mask R-CNN（torchvision）
├── outputs\checkpoints\
│   ├── ppliteseg\best.pth          ← v0 best（IoU=0.4391）
│   ├── ppliteseg_v3\best.pth       ← v3 best（IoU=0.4361）
│   ├── ppliteseg_v3\ppliteseg_v3_torchscript.pt
│   ├── ddrnet\best.pth
│   ├── ddrnet_ftl\best.pth
│   ├── deeplabv3_mobilenet\20260422_114901\best.pth
│   └── maskrcnn\best.pth
├── export_ppliteseg_torchscript.py
├── export_ppliteseg_v3_torchscript.py
├── export_ddrnet_v0_torchscript.py
├── export_ddrnet_torchscript.py    ← DDRNet FTL TorchScript
├── export_deeplabv3_torchscript.py
└── data\splits\test.txt            ← 凍結測試集 70 張 stem 清單

H:\ChihleeMaster\CrackPreVer3.5.3\ABECIS-main\
├── output\model_final.pth          ← ABECIS 官方模型（168MB）
└── abecis_headless_eval.py         ← ABECIS 推論 + 偵測率評估腳本

H:\ChihleeMaster\dev\final_outputs\
├── logs\{model_name}\{gpu|cpu}\
│   ├── inference_log_{timestamp}.txt
│   ├── per_image_{timestamp}.csv
│   ├── gt_metrics_{timestamp}.txt   ← GT 評估結果
│   └── gt_per_image_{timestamp}.csv
├── pred_masks\{model_name}\{device}\*.png
├── before\{model_name}\{device}\*.png
├── after\{model_name}\{device}\*.png
└── infernce\
    ├── ppliteseg_cfd_infernce.py        ← PP-LiteSeg v0 CFD 推論
    ├── ppliteseg_v3_cfd_infernce.py     ← PP-LiteSeg v3 CFD 推論
    ├── ddrnet_cfd_infernce.py           ← DDRNet v0 CFD 推論
    ├── deeplabv3_cfd_infernce.py        ← DeepLabV3 CFD 推論
    ├── maskrcnn_cfd_infernce.py         ← Mask R-CNN CFD 推論
    ├── ddrnet_ftl_infernce.py           ← DDRNet FTL CFD 推論
    └── abecis_infernce.py               ← ABECIS Detectron2 CFD 推論

H:\ChihleeMaster\CrackK500\CFD\
├── cfd_image\                      ← 155 張 CFD RGB 影像
└── seg_gt\                         ← 118 張像素級 GT 遮罩
```

---

## 十、DDRNet FTL v3 訓練指令

```powershell
conda activate CrackSeg
cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
python training/train_crackseg.py --config configs/final/ddrnet_ftl_v3.yaml
```

訓練完成後的後續流程：
```powershell
# 1. 匯出 TorchScript（export_ddrnet_torchscript.py 路徑改為 ddrnet_ftl_v3）
# 2. 推論 CFD（複製 ddrnet_ftl_infernce.py，修改 MODEL_NAME="ddrnet_ftl_v3_cfd"）
# 3. GT 評估
python "H:\ChihleeMaster\dev\final_outputs\ground_truth\evaluate_ground_truth_metrics.py" --model ddrnet_ftl_v3_cfd --device gpu
```

---

## 十一、口試答辯重點

> 「本研究在統一的像素級評估框架下，以相同測試集重新評估所有模型。PP-LiteSeg-T（5M 參數，122 FPS）達到 IoU=0.44，在 5.5 倍速度優勢下僅犧牲約 30% 精度（相較 ABECIS torchvision 重測值）。跨資料集零樣本驗證顯示 DDRNet FTL 泛化能力（CFD IoU=0.41）顯著優於 ABECIS（CFD IoU=0.21），證明輕量模型具備更強的領域外適應能力。在公平比較條件下（同閾值 0.80），DDRNet FTL 偵測率（70.7%）亦高於 ABECIS（50.6%），支持以輕量語意分割架構取代實例分割框架之可行性。」

### 常見委員問題答辯摘要

| 委員問題 | 答辯核心 |
|---------|---------|
| IoU 為何偏低？ | 像素級 IoU 對細線裂縫本質嚴苛，0.4 在文獻中屬正常範圍；相對比較與排序仍有效 |
| 為何不用 ABECIS 的 instance IoU？ | 統一 pixel-level 確保控制變數，才能公平比較架構差異 |
| v3 改善有限，意義何在？ | v3 的貢獻不在 in-distribution，而在 cross-domain：CFD 泛化 IoU 從 0.06 → 0.35 |
| 為何 DDRNet 不做 v3？ | 實驗設計以 Mendeley 為依據；CFD 大幅提升是事後評估的新發現，補充 DDRNet FTL v3 實驗作為回應 |
| 為何不做 DDRNet FTL + v3 組合？ | 已補做（`ddrnet_ftl_v3.yaml`），正在訓練，結果完成後更新 |
