@echo off
REM Train DeepLabV3+, PP-LiteSeg, and PIDNet in the CrackSeg environment
REM Run from the project root: scripts\run_train_crackseg.bat

echo ============================================================
echo  Activating CrackSeg conda environment
echo ============================================================
call conda activate CrackSeg

echo.
echo ============================================================
echo  Training DeepLabV3+
echo ============================================================
python training/train_crackseg.py --config configs/deeplabv3_mobilenet.yaml
if %errorlevel% neq 0 (
    echo ERROR: DeepLabV3+ training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Training PP-LiteSeg-T (STDC1)
echo ============================================================
python training/train_crackseg.py --config configs/ppliteseg.yaml
if %errorlevel% neq 0 (
    echo ERROR: PP-LiteSeg training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  Training PIDNet-S
echo ============================================================
python training/train_crackseg.py --config configs/pidnet.yaml
if %errorlevel% neq 0 (
    echo ERROR: PIDNet training failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo  All CrackSeg models trained successfully.
echo  Checkpoints: outputs/checkpoints/
echo ============================================================
call conda deactivate
pause
