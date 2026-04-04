@echo off
REM ============================================================
REM  Unified Inference + Evaluation Pipeline
REM  Uses configs/final/ (finalized thresholds)
REM  Run from project root: scripts\run_eval_all.bat
REM ============================================================
REM  Prerequisites:
REM    - All 4 models trained (checkpoints in outputs/checkpoints/)
REM    - Threshold sweep done for each CrackSeg model
REM    - configs/final/*.yaml thresholds updated
REM ============================================================

echo ============================================================
echo  [CrackPre] Mask R-CNN inference
echo ============================================================
call conda activate CrackPre

python evaluation/inference_maskrcnn.py --config configs/final/maskrcnn.yaml
if %errorlevel% neq 0 ( echo ERROR: Mask R-CNN inference failed. & pause & exit /b %errorlevel% )

call conda deactivate

echo.
echo ============================================================
echo  [CrackSeg] Semantic segmentation inference (final thresholds)
echo ============================================================
call conda activate CrackSeg

python evaluation/inference_crackseg.py --config configs/final/deeplabv3plus.yaml
if %errorlevel% neq 0 ( echo ERROR: DeepLabV3+ inference failed. & pause & exit /b %errorlevel% )

python evaluation/inference_crackseg.py --config configs/final/ppliteseg.yaml
if %errorlevel% neq 0 ( echo ERROR: PP-LiteSeg inference failed. & pause & exit /b %errorlevel% )

python evaluation/inference_crackseg.py --config configs/final/ddrnet.yaml
if %errorlevel% neq 0 ( echo ERROR: DDRNet inference failed. & pause & exit /b %errorlevel% )

echo.
echo ============================================================
echo  [CrackSeg] Unified evaluation (IoU/Dice/Precision/Recall/clDice)
echo ============================================================

python evaluation/evaluate.py
if %errorlevel% neq 0 ( echo ERROR: Evaluation failed. & pause & exit /b %errorlevel% )

call conda deactivate

echo.
echo ============================================================
echo  Done. Results: outputs/results/metrics_summary.csv
echo ============================================================
pause
