@echo off
REM Train Mask R-CNN using Detectron2 in the CrackPre environment
REM Run from the project root: scripts\run_train_maskrcnn.bat

echo ============================================================
echo  Activating CrackPre conda environment
echo ============================================================
call conda activate CrackPre

echo.
echo ============================================================
echo  Training Mask R-CNN R50-FPN
echo ============================================================
python training/train_maskrcnn.py --config configs/maskrcnn.yaml
if %errorlevel% neq 0 (
    echo ERROR: Mask R-CNN training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Mask R-CNN training complete.
echo  Checkpoints: outputs/checkpoints/maskrcnn/
echo ============================================================
call conda deactivate
pause
