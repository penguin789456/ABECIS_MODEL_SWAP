@echo off
chcp 65001 > nul
REM Train DeepLabV3-MobileNetV3 in CrackSeg environment
REM Run from the project root: scripts\run_train_crackseg.bat
REM
REM PP-LiteSeg and DDRNet are already trained.
REM This script trains only DeepLabV3-MobileNetV3 (the remaining model).
REM To retrain PP-LiteSeg / DDRNet, uncomment the relevant blocks below.

echo ============================================================
echo  Activating CrackSeg conda environment
echo ============================================================
call conda activate CrackSeg

echo.
echo ============================================================
echo  Training DeepLabV3-MobileNetV3 (~11M params)
echo ============================================================
python training/train_crackseg.py --config configs/deeplabv3_mobilenet.yaml
if %errorlevel% neq 0 (
    echo ERROR: DeepLabV3-MobileNetV3 training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  [SKIP] PP-LiteSeg-T - already trained (Test IoU=0.4391)
echo  To retrain: python training/train_crackseg.py --config configs/ppliteseg.yaml
echo ============================================================

echo.
echo ============================================================
echo  [SKIP] DDRNet-23-slim - already trained (Test IoU=0.3912)
echo  To retrain: python training/train_crackseg.py --config configs/final/ddrnet.yaml
echo ============================================================

echo.
echo ============================================================
echo  DeepLabV3-MobileNetV3 training complete.
echo  Checkpoints: outputs/checkpoints/deeplabv3_mobilenet/
echo  Next: run scripts\run_eval_all.bat to generate metrics_summary.csv
echo ============================================================
call conda deactivate
pause
