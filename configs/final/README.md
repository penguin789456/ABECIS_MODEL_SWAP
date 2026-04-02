# configs/final/ — 定稿設定檔

此目錄存放每個模型**已完成訓練、結果已確認**的最終設定檔。

`configs/` 下的工作設定檔可能在實驗過程中被修改；
此目錄的設定檔為唯讀參考，不應再修改。

---

## 進度

| 檔案 | 模型 | 狀態 | Best IoU | Threshold | Checkpoint |
|------|------|------|---------|-----------|------------|
| `ppliteseg.yaml` | PP-LiteSeg-T | ✅ 完成 | 0.3755 | 0.60 | `20260402_141530/best.pth` |
| `deeplabv3plus.yaml` | DeepLabV3+ | ⏳ 待訓練 | — | — | — |
| `pidnet.yaml` | PIDNet-S | ⏳ 待訓練 | — | — | — |
| `maskrcnn.yaml` | Mask R-CNN | ⏳ 待訓練 | — | — | — |

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
