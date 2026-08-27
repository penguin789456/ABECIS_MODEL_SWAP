# -*- coding: utf-8 -*-
"""
Mendeley 凍結測試集推論腳本（TorchScript 通用版）

讀取 data/splits/test.txt（70 張）對應的 Mendeley rgb 圖片，
以 1024×1024 全圖推論，儲存 pred_masks。

用法：
  conda activate CrackSeg
  python mendeley_inference.py --model ppliteseg   --device cuda
  python mendeley_inference.py --model ddrnet       --device cuda
  python mendeley_inference.py --model ddrnet_ftl   --device cuda
  python mendeley_inference.py --model deeplabv3    --device cuda
  python mendeley_inference.py --model ppliteseg   --device cpu

--model 選項：ppliteseg / ddrnet / ddrnet_ftl / deeplabv3
"""

import argparse
import csv
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ── 路徑常數 ──────────────────────────────────────────────────────────────────
CKPT_BASE   = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints")
RGB_DIR     = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\rgb")
TEST_SPLIT  = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt")
OUT_BASE    = Path(r"H:\ChihleeMaster\dev\final_outputs")
TARGET_SIZE = (1024, 1024)

# ImageNet 正規化
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── 模型設定 ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "ppliteseg": {
        "pt"        : CKPT_BASE / "ppliteseg" / "ppliteseg_torchscript.pt",
        "threshold" : 0.65,
        "out_name"  : "ppliteseg_mendeley",
    },
    "ddrnet": {
        "pt"        : CKPT_BASE / "ddrnet" / "ddrnet_torchscript.pt",
        "threshold" : 0.50,
        "out_name"  : "ddrnet_mendeley",
    },
    "ddrnet_ftl": {
        "pt"        : CKPT_BASE / "ddrnet_ftl" / "ddrnet_ftl_torchscript.pt",
        "threshold" : 0.50,
        "out_name"  : "ddrnet_ftl_mendeley",
    },
    "deeplabv3": {
        "pt"        : CKPT_BASE / "deeplabv3_mobilenet" / "deeplabv3_torchscript.pt",
        "threshold" : 0.45,
        "out_name"  : "deeplabv3_mendeley",
    },
}


def get_cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"], encoding="utf-8", errors="ignore"
            )
            return out.strip().splitlines()[-1].strip()
    except Exception:
        pass
    return platform.processor()


def build_rgb_index(folder: Path) -> dict:
    """stem.lower() → Path，處理大小寫混用"""
    return {
        p.stem.lower(): p
        for p in folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    }


def load_model(pt_path: Path, device: torch.device) -> torch.jit.ScriptModule:
    model = torch.jit.load(str(pt_path), map_location=device)
    model.eval()
    return model


def infer_image(model, pil_img: Image.Image, device: torch.device, threshold: float) -> np.ndarray:
    """回傳 orig-size 的二元 mask (np.uint8 0/255)"""
    orig_w, orig_h = pil_img.size
    img = pil_img.resize(TARGET_SIZE, Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(img), mean=MEAN, std=STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    if (orig_w, orig_h) != TARGET_SIZE[::-1]:
        mask = np.array(Image.fromarray(mask).resize((orig_w, orig_h), Image.NEAREST))
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="覆蓋預設 threshold")
    args = parser.parse_args()

    cfg       = MODEL_REGISTRY[args.model]
    pt_path   = cfg["pt"]
    threshold = args.threshold if args.threshold is not None else cfg["threshold"]
    out_name  = cfg["out_name"]

    device_str    = args.device
    device        = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    device_suffix = "gpu" if device.type == "cuda" else "cpu"

    # ── 讀 test split ─────────────────────────────────────────────────────────
    stems_raw = [l.strip() for l in TEST_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    stems     = [s.zfill(3) if s.isdigit() else s for s in stems_raw]
    rgb_index = build_rgb_index(RGB_DIR)

    image_paths = []
    for stem in stems:
        p = rgb_index.get(stem.lower())
        if p is not None:
            image_paths.append((stem, p))
        else:
            print(f"⚠ 找不到圖片：{stem}")

    n_images = len(image_paths)
    print(f"模型      : {pt_path}")
    print(f"裝置      : {device}")
    print(f"Threshold : {threshold}")
    print(f"影像數    : {n_images} 張（test.txt）")

    # ── 輸出目錄 ──────────────────────────────────────────────────────────────
    pred_dir   = OUT_BASE / "pred_masks" / out_name / device_suffix
    before_dir = OUT_BASE / "before"     / out_name / device_suffix
    after_dir  = OUT_BASE / "after"      / out_name / device_suffix
    log_dir    = OUT_BASE / "logs"       / out_name / device_suffix
    for d in [pred_dir, before_dir, after_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"inference_log_{ts}.txt"
    csv_path = log_dir / f"per_image_{ts}.csv"

    # ── 載入模型 ──────────────────────────────────────────────────────────────
    model = load_model(pt_path, device)

    # GPU 記憶體初始
    gpu_mem_start = torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    # ── 推論 ──────────────────────────────────────────────────────────────────
    rows         = []
    latencies    = []
    crack_count  = 0
    fail_count   = 0
    total_px     = 0

    t_proc_start = time.perf_counter()

    for stem, img_path in tqdm(image_paths, desc=out_name):
        try:
            t0    = time.perf_counter()
            orig  = Image.open(img_path).convert("RGB")
            mask  = infer_image(model, orig, device, threshold)
            elapsed = time.perf_counter() - t0

            has_crack   = mask.any()
            crack_px    = int(mask.astype(bool).sum())
            total_px_im = orig.size[0] * orig.size[1]

            if has_crack:
                crack_count += 1
            total_px += crack_px
            latencies.append(elapsed)

            # before / after
            orig.save(before_dir / f"{stem}.png")
            overlay = orig.copy().convert("RGBA")
            red     = Image.new("RGBA", orig.size, (255, 0, 0, 0))
            alpha   = Image.fromarray((mask.astype(np.uint8) * 128), "L")
            red.putalpha(alpha)
            overlay = Image.alpha_composite(overlay, red)
            overlay.convert("RGB").save(after_dir / f"{stem}.png")

            # pred_mask
            Image.fromarray(mask).save(pred_dir / f"{stem}.png")

            rows.append({
                "stem": stem, "status": "OK",
                "latency_s": round(elapsed, 6),
                "has_crack": int(has_crack),
                "crack_px":  crack_px,
                "total_px":  total_px_im,
                "cover_pct": round(crack_px / total_px_im * 100, 4),
            })

        except Exception as e:
            fail_count += 1
            rows.append({"stem": stem, "status": f"ERROR: {e}"})

    t_total = time.perf_counter() - t_proc_start

    # GPU peak
    gpu_peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    # ── CSV ───────────────────────────────────────────────────────────────────
    fields = ["stem", "status", "latency_s", "has_crack", "crack_px", "total_px", "cover_pct"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ── 統計 ──────────────────────────────────────────────────────────────────
    n_ok  = len(latencies)
    avg_t = float(np.mean(latencies)) if latencies else 0
    max_t = float(np.max(latencies))  if latencies else 0
    min_t = float(np.min(latencies))  if latencies else 0
    fps   = 1.0 / avg_t if avg_t > 0 else 0

    sep  = "-" * 100
    lines = [
        f"Mendeley 凍結測試集 推論記錄 — {out_name}",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型        : {pt_path}",
        f"資料來源    : {RGB_DIR}（凍結測試集 test.txt）",
        f"資料集      : Mendeley CCSD（訓練域測試集，70 張）",
        f"閾值        : {threshold}",
        f"輸入尺寸    : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}",
        f"影像數      : {n_images}",
        f"輸出目錄    : {OUT_BASE}",
        sep,
        "硬體資訊",
        f"運算裝置    : {device}",
        f"CPU         : {get_cpu_name()}",
        f"CUDA 可用   : {torch.cuda.is_available()}",
    ]
    if device.type == "cuda":
        lines += [
            f"GPU         : {torch.cuda.get_device_name(0)}",
            f"GPU 記憶體  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
        ]
    lines += [
        f"PyTorch     : {torch.__version__}",
        sep,
        "整體統計",
        f"成功處理    : {n_ok} / {n_images} 張",
        f"失敗        : {fail_count} 張",
        f"有裂縫判定  : {crack_count} 張",
        f"推論平均    : {avg_t*1000:.3f} ms/張",
        f"FPS         : {fps:.2f}",
        f"最快        : {min_t*1000:.3f} ms",
        f"最慢        : {max_t*1000:.3f} ms",
        f"總耗時      : {t_total:.3f} 秒",
        f"GPU peak    : {gpu_peak:.2f} MB" if device.type == "cuda" else "",
        sep,
        f"pred_masks → {pred_dir}",
        f"before     → {before_dir}",
        f"after      → {after_dir}",
        f"CSV        → {csv_path}",
    ]

    report = "\n".join(l for l in lines if l is not None)
    print(report)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 完成 → {log_path}")


if __name__ == "__main__":
    main()
