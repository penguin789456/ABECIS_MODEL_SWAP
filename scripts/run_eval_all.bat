@echo off
REM Run inference for all models and compute unified evaluation metrics
REM Run from the project root: scripts\run_eval_all.bat

echo ============================================================
echo  [CrackSeg] Running inference for CrackSeg models
echo ============================================================
call conda activate CrackSeg

python evaluation/inference_crackseg.py --config configs/deeplabv3plus.yaml
if %errorlevel% neq 0 ( echo ERROR: DeepLabV3+ inference failed. & pause & exit /b %errorlevel% )

python evaluation/inference_crackseg.py --config configs/ppliteseg.yaml
if %errorlevel% neq 0 ( echo ERROR: PP-LiteSeg inference failed. & pause & exit /b %errorlevel% )

python evaluation/inference_crackseg.py --config configs/pidnet.yaml
if %errorlevel% neq 0 ( echo ERROR: PIDNet inference failed. & pause & exit /b %errorlevel% )

call conda deactivate

echo.
echo ============================================================
echo  [CrackPre] Running inference for Mask R-CNN
echo ============================================================
call conda activate CrackPre

python evaluation/inference_maskrcnn.py --config configs/maskrcnn.yaml
if %errorlevel% neq 0 ( echo ERROR: Mask R-CNN inference failed. & pause & exit /b %errorlevel% )

call conda deactivate

echo.
echo ============================================================
echo  [CrackSeg] Computing unified evaluation metrics
echo ============================================================
call conda activate CrackSeg

python evaluation/evaluate.py
if %errorlevel% neq 0 ( echo ERROR: Evaluation failed. & pause & exit /b %errorlevel% )

call conda deactivate

echo.
echo ============================================================
echo  Evaluation complete.
echo  Results: outputs/results/metrics_summary.csv
echo ============================================================
pause
