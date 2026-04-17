@echo off
chcp 65001 > nul
echo ============================================================
echo  Train Mask R-CNN (torchvision) — CrackSeg env
echo ============================================================
call conda activate CrackSeg
cd /d H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
python training\train_maskrcnn_tv.py --config configs\maskrcnn_tv.yaml
pause
