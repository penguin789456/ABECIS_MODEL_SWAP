"""
匯出 DDRNet checkpoint 為 TorchScript 模型

用途：
  將 best.pth 權重 + DDRNet 架構匯出成可獨立載入的 TorchScript .pt

執行方式：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python export_ddrnet_torchscript.py
"""

import sys
from pathlib import Path

import torch
import yaml

# 讓 Python 找得到你的專案模組
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "realtime-semantic-segmentation-pytorch"))

from training.train_crackseg import build_model


# ── 路徑設定 ─────────────────────────────────────────────
CFG_PATH = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ddrnet_ftl.yaml")

CKPT_PATH = Path(
    r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\ddrnet_ftl\best.pth"
)

OUT_PATH = Path(
    r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\ddrnet_ftl\ddrnet_ftl_torchscript.pt"
)


def main():
    device = torch.device("cpu")

    print(f"Config     : {CFG_PATH}")
    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Output     : {OUT_PATH}")

    # 1. 讀 config
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 建立模型架構
    model = build_model(cfg["model"]).to(device)

    # 3. 載入 checkpoint 權重
    ckpt = torch.load(CKPT_PATH, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    # 4. 建立 dummy input
    # 你的 whole-image 推論預設是 1024x1024，所以這裡用同樣尺寸
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device)

    # 5. 匯出 TorchScript
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # 6. 儲存
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(OUT_PATH))

    print("\n✅ TorchScript 匯出完成")
    print(f"模型檔案：{OUT_PATH}")

    # 7. 簡單測試能不能重新載入
    loaded = torch.jit.load(str(OUT_PATH), map_location=device)
    loaded.eval()

    with torch.no_grad():
        test_output = loaded(dummy_input)

    if isinstance(test_output, (tuple, list)):
        test_output = test_output[0]

    print(f"測試輸出 shape：{tuple(test_output.shape)}")


if __name__ == "__main__":
    main()