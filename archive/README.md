# archive/

一次性實驗、分析與報告腳本的歸檔處。這些腳本已完成其任務，
結果已記錄於 `CLAUDE.md` / `RESEARCH_SUMMARY.md` 與 `outputs/` 日誌中，
保留供日後回溯，不屬於主要訓練/推論流程。

## experiments/
- `*_thresh_sweep*.py` / `asymmetric_thresh_sweep.py` — 各模型推論閾值掃描
- `*_detection_rate.py` / `detection_rate_*.py` — 裂縫偵測率分析
- `eval_ddrnet_t03.py` — DDRNet 特定閾值評估
- `run_ddrnet_ftl_*.py` — DDRNet-FTL 一次性執行/統計腳本
- `compare_abecis_vs_ppliteseg.py` — ABECIS vs PP-LiteSeg 比較
- `ppliteseg_fine_sweep.py` — PP-LiteSeg 細粒度閾值掃描
- `abecis_llm_filter.py` / `abecis_model_filter.py` / `abecis_report_generator.py` — ABECIS 結果過濾與報告生成

> 正式推論評估流程見 `H:\ChihleeMaster\dev\final_outputs\`（patch 滑窗推論 + GT 評估）。
