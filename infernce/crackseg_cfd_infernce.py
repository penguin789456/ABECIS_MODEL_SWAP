"""
語意分割模型 CFD 推論腳本（PP-LiteSeg / DDRNet / DeepLabV3）
使用 CFD 完全未見資料集，patch-based stitch inference

支援模型：
  --model ppliteseg        → PP-LiteSeg-T
  --model ddrnet           → DDRNet-23-slim (BCEDice)
  --model deeplabv3        → DeepLabV3-MobileNetV3

啟動方式（CrackSeg env）：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python "H:\ChihleeMaster\dev\final_outputs\infernce\crackseg_cfd_infernce.py" --model ppliteseg --device cuda
  python "H:\ChihleeMaster\dev\final_outputs\infernce\crackseg_cfd_infernce.py" --model ddrnet    --device cuda
  python "H:\ChihleeMaster\dev\final_outputs\infernce\crackseg_cfd_infernce.py" --model deeplabv3 --device cuda

輸出：
  H:\ChihleeMaster\dev\final_outputs\
    logs\{model_name}\{device}\inference_log.txt
    logs\{model_name}\{device}\per_image.csv
    before\{model_name}\{device}\*.png
    after\{model_name}\{device}\*.png
    pred_masks\{model_name}\{device}\*.png
"""

import os, sys, csv, time, shutil, argparse, platform, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# ── 專案路徑 ─────────────────────────────────────────────────────────────────
PROJECT = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "realtime-semantic-segmentation-pytorch"))

from training.train_crackseg import build_model
from data.transforms import get_test_transforms

# ── CFD 資料集 ───────────────────────────────────────────────────────────────
CFD_RGB_DIR = Path(r"H:\ChihleeMaster\CrackK500\CFD\cfd_image")
OUT_BASE    = Path(r"H:\ChihleeMaster\dev\final_outputs")

# ── 模型設定表 ────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ppliteseg": {
        "cfg"   : PROJECT / "configs/final/ppliteseg.yaml",
        "ckpt"  : PROJECT / "outputs/checkpoints/ppliteseg/best.pth",
        "name"  : "ppliteseg_cfd",
        "thresh": 0.65,
    },
    "ddrnet": {
        "cfg"   : PROJECT / "configs/final/ddrnet.yaml",
        "ckpt"  : PROJECT / "outputs/checkpoints/ddrnet/best.pth",
        "name"  : "ddrnet_cfd",
        "thresh": 0.35,
    },
    "deeplabv3": {
        "cfg"   : PROJECT / "configs/final/deeplabv3_mobilenet.yaml",
        "ckpt"  : PROJECT / "outputs/checkpoints/deeplabv3_mobilenet/20260422_114901/best.pth",
        "name"  : "deeplabv3_cfd",
        "thresh": 0.45,
    },
}


def get_cpu_name():
    try:
        if platform.system().lower() == "windows":
            out = subprocess.check_output(["wmic", "cpu", "get", "Name"],
                                          stderr=subprocess.DEVNULL, text=True)
            lines = [x.strip() for x in out.splitlines()
                     if x.strip() and x.strip().lower() != "name"]
            if lines: return lines[0]
    except Exception: pass
    return platform.processor() or platform.machine()


try:
    import psutil
    _PSUTIL = True
except Exception:
    psutil = None
    _PSUTIL = False


def ram_mb():
    if _PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    return 0.0


def gpu_mb(device):
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def overlay_mask(rgb, mask, color=(220, 50, 50), alpha=0.5):
    out = rgb.copy().astype(np.float32)
    crack = mask >= 128
    for c, v in enumerate(color):
        out[:, :, c] = np.where(crack, out[:, :, c] * (1-alpha) + v*alpha, out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)


def stitch_patches(image, model, device, transform, patch_size=512, overlap=128, threshold=0.5):
    H, W = image.shape[:2]
    stride = patch_size - overlap
    logit_sum = np.zeros((H, W), np.float32)
    count_map = np.zeros((H, W), np.float32)

    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if not ys or ys[-1] + patch_size < H: ys.append(max(H - patch_size, 0))
    if not xs or xs[-1] + patch_size < W: xs.append(max(W - patch_size, 0))

    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image[y:y+patch_size, x:x+patch_size]
                ph = patch_size - patch.shape[0]
                pw = patch_size - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0,ph),(0,pw),(0,0)), mode='reflect')
                aug = transform(image=patch)
                tensor = aug['image'].unsqueeze(0).float().to(device)
                logit = model(tensor).squeeze().cpu().numpy()
                ah = min(patch_size, H-y)
                aw = min(patch_size, W-x)
                logit_sum[y:y+ah, x:x+aw] += logit[:ah, :aw]
                count_map[y:y+ah, x:x+aw] += 1.0

    avg_logit = logit_sum / np.maximum(count_map, 1.0)
    prob = 1.0 / (1.0 + np.exp(-avg_logit))
    return (prob > threshold).astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="模型名稱：ppliteseg / ddrnet / deeplabv3")
    parser.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=None,
                        help="覆蓋預設 threshold")
    args = parser.parse_args()

    mc = MODEL_CONFIGS[args.model]
    model_name = mc["name"]
    threshold  = args.threshold if args.threshold is not None else mc["thresh"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，改用 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    device_suffix = "gpu" if device.type == "cuda" else "cpu"

    # ── 建立輸出目錄（新結構）────────────────────────────────────────────────
    log_dir       = OUT_BASE / "logs"      / model_name / device_suffix
    before_dir    = OUT_BASE / "before"    / model_name / device_suffix
    after_dir     = OUT_BASE / "after"     / model_name / device_suffix
    pred_masks_dir = OUT_BASE / "pred_masks" / model_name / device_suffix
    for d in [log_dir, before_dir, after_dir, pred_masks_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "inference_log.txt"
    csv_path = log_dir / "per_image.csv"

    # ── 載入模型 ─────────────────────────────────────────────────────────────
    print(f"Model    : {args.model} ({model_name})")
    print(f"Ckpt     : {mc['ckpt']}")
    print(f"Device   : {device}  |  Threshold: {threshold}")

    with open(mc["cfg"], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_obj = build_model(cfg["model"]).to(device)
    ckpt = torch.load(mc["ckpt"], map_location=device)
    model_obj.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model_obj.eval()
    print("Model loaded.\n")

    transform = get_test_transforms()
    patch_size = cfg["dataset"].get("patch_size", 512)
    overlap    = cfg["dataset"].get("overlap", 128)

    rgb_files = sorted([p for p in CFD_RGB_DIR.iterdir()
                        if p.suffix.lower() in (".jpg",".jpeg",".png")])
    print(f"CFD images: {len(rgb_files)} 張")

    # ── 推論 ─────────────────────────────────────────────────────────────────
    log_lines = [
        f"語意分割 CFD 推論記錄 — {model_name}",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型        : {args.model}",
        f"Checkpoint  : {mc['ckpt']}",
        f"Device      : {device}",
        f"Threshold   : {threshold}",
        f"Patch size  : {patch_size}  Overlap: {overlap}",
        f"CFD 影像數  : {len(rgb_files)} 張",
        "資料集      : CFD（完全未見，無資料洩漏）",
        f"CPU         : {get_cpu_name()}",
        "-" * 80,
    ]
    csv_rows = []

    success = fail = crack_n = no_crack_n = total_px_crack = 0
    cov_list = elapsed_list = []
    cov_list, elapsed_list = [], []

    t_infer = time.perf_counter()
    for rgb_p in tqdm(rgb_files, desc=f"{args.model} CFD"):
        stem = rgb_p.stem
        try:
            t0 = time.perf_counter()
            rgb = np.array(Image.open(rgb_p).convert("RGB"))

            mask = stitch_patches(rgb, model_obj, device, transform,
                                  patch_size=patch_size, overlap=overlap,
                                  threshold=threshold)

            shutil.copy(str(rgb_p), before_dir / f"{stem}.png")
            Image.fromarray(mask).save(pred_masks_dir / f"{stem}.png")
            overlay = overlay_mask(rgb, mask)
            Image.fromarray(overlay).save(after_dir / f"{stem}.png")

            crack_px = int((mask >= 128).sum())
            coverage = crack_px / (mask.shape[0] * mask.shape[1]) * 100
            elapsed  = time.perf_counter() - t0

            success += 1
            total_px_crack += crack_px
            cov_list.append(coverage)
            elapsed_list.append(elapsed)
            (crack_n if crack_px > 0 else no_crack_n).__class__  # dummy
            if crack_px > 0: crack_n += 1
            else: no_crack_n += 1

            csv_rows.append({
                "image_id": stem, "status": "OK",
                "crack_pixels": crack_px,
                "coverage_%": round(coverage, 6),
                "judgement": "有裂縫" if crack_px > 0 else "無裂縫",
                "elapsed_sec": round(elapsed, 6),
                "ram_mb": round(ram_mb(), 2),
                "gpu_mb": round(gpu_mb(device), 2),
            })
        except Exception as e:
            fail += 1
            csv_rows.append({"image_id": stem, "status": f"FAILED:{e}",
                             "crack_pixels":"","coverage_%":"","judgement":"",
                             "elapsed_sec":"","ram_mb":"","gpu_mb":""})

    infer_elapsed = time.perf_counter() - t_infer
    avg_t   = float(np.mean(elapsed_list)) if elapsed_list else 0
    avg_cov = float(np.mean(cov_list)) if cov_list else 0

    log_lines += [
        f"成功     : {success} 張  |  失敗: {fail} 張",
        f"有裂縫   : {crack_n} 張  |  無裂縫: {no_crack_n} 張",
        f"判斷率   : {crack_n/success*100:.2f}%" if success else "判斷率: N/A",
        f"平均耗時 : {avg_t:.4f} 秒/張",
        f"總耗時   : {infer_elapsed:.2f} 秒",
        f"平均覆蓋率: {avg_cov:.4f}%",
        "-" * 80,
        f"before → {before_dir}",
        f"after  → {after_dir}",
        f"masks  → {pred_masks_dir}",
        f"CSV    → {csv_path}",
    ]

    fields = ["image_id","status","crack_pixels","coverage_%","judgement",
              "elapsed_sec","ram_mb","gpu_mb"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(csv_rows)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n✅ 完成 — {model_name}")
    print(f"   log       → {log_path}")
    print(f"   pred_masks→ {pred_masks_dir}")


if __name__ == "__main__":
    main()
