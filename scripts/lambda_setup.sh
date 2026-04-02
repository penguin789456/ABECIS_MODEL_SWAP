#!/bin/bash
# ============================================================
# Lambda GPU Cloud 環境建立腳本
# 在 Lambda 實例上 SSH 後執行：
#   bash ~/ABECIS_MODEL_SWAP/scripts/lambda_setup.sh
# ============================================================

set -e
cd ~/ABECIS_MODEL_SWAP

echo "=============================="
echo " Step 1: 安裝 Python 套件"
echo "=============================="
pip install -q \
    "albumentations==1.4.21" \
    torchvision==0.16.2 \
    torchmetrics \
    loguru \
    tqdm \
    pyyaml \
    matplotlib \
    scikit-image \
    scipy \
    pandas \
    tensorboard

echo ""
echo "=============================="
echo " Step 2: Clone zh320 repo（PP-LiteSeg / PIDNet）"
echo "=============================="
if [ ! -d "realtime-semantic-segmentation-pytorch" ]; then
    git clone https://github.com/zh320/realtime-semantic-segmentation-pytorch
    echo "Clone 完成"
else
    echo "已存在，跳過"
fi

echo ""
echo "=============================="
echo " Step 3: 驗證 CUDA"
echo "=============================="
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=============================="
echo " 環境建立完成，可以開始訓練"
echo "=============================="
