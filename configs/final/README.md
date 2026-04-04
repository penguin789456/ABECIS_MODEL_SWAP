# configs/final/ — 定稿設定檔

此目錄存放每個模型的**定稿設定檔**，包含最終超參數與已確認的評估結果。

`configs/` 下的工作設定檔可能在實驗過程中被修改；
此目錄的設定檔為各模型的唯一事實來源（single source of truth）。

**PP-LiteSeg 關鍵發現（2026-04-02）：**
- 最佳 IoU = 0.3769 @ threshold = 0.65（細粒度 sweep 確認）
- FN 懲罰損失（FocalTversky、pos_weight）導致模型退化（IoU → 0.018）
- 正樣本過採樣（5x）= +0.027 IoU，為最有效的單一改善
- Threshold cliff：< 0.52 全猜正；plateau 0.58–0.68（delta < 0.003）

---

## 進度

| 檔案 | 模型 | 狀態 | Best IoU | Threshold | Checkpoint |
|------|------|------|---------|-----------|------------|
| `ppliteseg.yaml` | PP-LiteSeg-T | ✅ 完成 | **0.3769** | **0.65** | `20260402_141530/best.pth` |
| `deeplabv3plus.yaml` | DeepLabV3+ | 🔄 訓練中 | — | — | — |
| `ddrnet.yaml` | DDRNet-23-slim | ⏳ 待訓練 | — | — | — |
| `maskrcnn.yaml` | Mask R-CNN | ⏳ 待訓練（CrackPre） | — | — | — |

---

## 使用方式

**重現 PP-LiteSeg 推論：**
```bash
conda activate CrackSeg
python evaluation/inference_crackseg.py --config configs/final/ppliteseg.yaml \
    --checkpoint outputs/checkpoints/ppliteseg/20260402_141530/best.pth
```

**重跑 threshold sweep 驗證：**
```bash
python scripts/threshold_sweep.py --config configs/final/ppliteseg.yaml \
    --checkpoint outputs/checkpoints/ppliteseg/20260402_141530/best.pth
```
