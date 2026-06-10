# infernce/ — 推論腳本

跨資料集（Mendeley 凍結測試集 / CFD 跨域）的推論管線。
輸出統一寫至 `H:\ChihleeMaster\dev\final_outputs\{pred_masks,before,after,logs}\{model}\{gpu|cpu}\`，
再由 `../ground_truth/evaluate_gt_with_cldice.py` 評估（含 clDice）。

## ⚠️ 執行注意
- 勿用 `conda run -n`（cp950 崩潰），直接呼叫 env python：
  - 分割模型：`G:\conda\envs\CrackSeg\python.exe`
  - ABECIS：`G:\conda\envs\ABECIS\python.exe`（Detectron2，CPU only）
- 務必加 `PYTHONIOENCODING=utf-8`；分割模型再加 `PYTHONPATH`（專案根 + zh320 repo）

## Patch 滑窗推論（512×512, overlap=128, stride=384）⭐ 主結果
| 腳本 | 模型 | 資料集 | 環境 |
|------|------|--------|------|
| `mendeley_patch_inference.py` | 4 分割模型 | Mendeley 70 張 | CrackSeg |
| `cfd_patch_inference.py` | 4 分割模型 | CFD 全圖 | CrackSeg |
| `abecis_mendeley_patch_inference.py` | ABECIS（實例遮罩聯集，含 `--resume`） | Mendeley | ABECIS CPU |
| `abecis_cfd_patch_inference.py` | ABECIS（原始解析度 640×480） | CFD | ABECIS CPU |

## 全圖推論（1024×1024，歷史對照）
- `mendeley_inference.py` — 分割模型 Mendeley 全圖
- `abecis_infernce.py` / `abecis_mendeley_infernce.py` — ABECIS 全圖
- `{ppliteseg,ddrnet,ddrnet_ftl,deeplabv3,maskrcnn,ppliteseg_v3}_cfd_infernce.py`、`crackseg_cfd_infernce.py` — 各模型 CFD 全圖

> 範例指令與完整結果見專案根 `CLAUDE.md`。
