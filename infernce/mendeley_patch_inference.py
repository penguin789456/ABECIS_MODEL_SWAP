# -*- coding: utf-8 -*-
"""
Mendeley 凍結測試集 — 512×512 Patch 滑動視窗推論腳本（TorchScript 通用版）

與訓練條件一致：patch_size=512, overlap=128（stride=384）
符合「閾值掃描於 validation 決定各模型最適門檻」之論文設計

用法：
  conda activate CrackSeg
  python mendeley_patch_inference.py --model ppliteseg  --device cuda
  python mendeley_patch_inference.py --model ddrnet      --device cuda
  python mendeley_patch_inference.py --model ddrnet_ftl  --device cuda
  python mendeley_patch_inference.py --model deeplabv3   --device cuda

輸出（不覆蓋全圖結果）：
  pred_masks/{model}_mendeley_patch/{device}/
  logs/{model}_mendeley_patch/{device}/inference_log_{ts}.txt
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

# ── 路徑 ──────────────────────────────────────────────────────────────────────
CKPT_BASE  = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints")
RGB_DIR    = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\rgb")
TEST_SPLIT = Path(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt")
OUT_BASE   = Path(r"H:\ChihleeMaster\dev\final_outputs")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

PATCH_SIZE = 512
OVERLAP    = 128
STRIDE     = PATCH_SIZE - OVERLAP   # 384

# ── 模型設定（各自最適門檻） ──────────────────────────────────────────────────
MODEL_REGISTRY = {
    "ppliteseg": {
        "pt"       : CKPT_BASE / "ppliteseg" / "ppliteseg_torchscript.pt",
        "threshold": 0.65,
        "out_name" : "ppliteseg_mendeley_patch",
    },
    "ddrnet": {
        "pt"       : CKPT_BASE / "ddrnet" / "ddrnet_torchscript.pt",
        "threshold": 0.50,
        "out_name" : "ddrnet_mendeley_patch",
    },
    "ddrnet_ftl": {
        "pt"       : CKPT_BASE / "ddrnet_ftl" / "ddrnet_ftl_torchscript.pt",
        "threshold": 0.50,
        "out_name" : "ddrnet_ftl_mendeley_patch",
    },
    "deeplabv3": {
        "pt"       : CKPT_BASE / "deeplabv3_mobilenet" / "deeplabv3_torchscript.pt",
        "threshold": 0.45,
        "out_name" : "deeplabv3_mendeley_patch",
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
    return {p.stem.lower(): p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")}


def infer_patch_sliding(
    model,
    pil_img: Image.Image,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    """
    512×512 滑動視窗推論，重疊區域取平均概率後套用門檻。

    Returns:
        binary mask (np.uint8, 0 or 255) at original image size
    """
    orig_w, orig_h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))  # H × W × 3

    prob_sum = np.zeros((orig_h, orig_w), dtype=np.float32)
    count_map = np.zeros((orig_h, orig_w), dtype=np.float32)

    # 產生起始座標列表（確保覆蓋右 / 下邊緣）
    def get_starts(length: int) -> list:
        starts = list(range(0, max(length - PATCH_SIZE, 0) + 1, STRIDE))
        # 最後一個 patch 必須貼齊右/下緣
        last = max(length - PATCH_SIZE, 0)
        if starts[-1] != last:
            starts.append(last)
        return starts

    ys = get_starts(orig_h) if orig_h >= PATCH_SIZE else [0]
    xs = get_starts(orig_w) if orig_w >= PATCH_SIZE else [0]

    for y1 in ys:
        y2 = min(y1 + PATCH_SIZE, orig_h)
        ph = y2 - y1
        for x1 in xs:
            x2 = min(x1 + PATCH_SIZE, orig_w)
            pw = x2 - x1

            # 裁取 patch；若不足 512 則 pad 到 512
            patch = img_np[y1:y2, x1:x2]
            if ph < PATCH_SIZE or pw < PATCH_SIZE:
                pad = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
                pad[:ph, :pw] = patch
                patch = pad

            tensor = TF.normalize(
                TF.to_tensor(Image.fromarray(patch)),
                mean=MEAN, std=STD
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                logit = model(tensor)
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()  # 512 × 512

            # 只寫回有效區域
            prob_sum[y1:y2, x1:x2]  += prob[:ph, :pw]
            count_map[y1:y2, x1:x2] += 1.0

    avg_prob = prob_sum / np.maximum(count_map, 1e-7)
    return (avg_prob > threshold).astype(np.uint8) * 255


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
    device        = torch.device(device_str if (device_str == "cuda" and
                                                torch.cuda.is_available()) else "cpu")
    device_suffix = "gpu" if device.type == "cuda" else "cpu"

    # ── 讀 test split ─────────────────────────────────────────────────────────
    stems_raw = [l.strip() for l in
                 TEST_SPLIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    stems     = [s.zfill(3) if s.isdigit() else s for s in stems_raw]
    rgb_index = build_rgb_index(RGB_DIR)
    image_paths = [(s, rgb_index[s.lower()]) for s in stems if s.lower() in rgb_index]
    n_images = len(image_paths)

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

    print(f"模型      : {pt_path}")
    print(f"裝置      : {device}")
    print(f"Threshold : {threshold}")
    print(f"Patch     : {PATCH_SIZE}×{PATCH_SIZE}，overlap={OVERLAP}，stride={STRIDE}")
    print(f"影像數    : {n_images} 張（test.txt）")

    # ── 載入模型 ──────────────────────────────────────────────────────────────
    model = torch.jit.load(str(pt_path), map_location=device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── 推論 ──────────────────────────────────────────────────────────────────
    rows      = []
    latencies = []
    crack_cnt = 0
    fail_cnt  = 0

    t_start = time.perf_counter()

    for stem, img_path in tqdm(image_paths, desc=out_name):
        try:
            t0   = time.perf_counter()
            orig = Image.open(img_path).convert("RGB")
            mask = infer_patch_sliding(model, orig, device, threshold)
            elapsed = time.perf_counter() - t0

            crack_px    = int(mask.astype(bool).sum())
            total_px_im = orig.size[0] * orig.size[1]
            has_crack   = crack_px > 0
            if has_crack:
                crack_cnt += 1
            latencies.append(elapsed)

            # before / after / pred_mask
            orig.save(before_dir / f"{stem}.png")
            overlay = orig.copy().convert("RGBA")
            red     = Image.new("RGBA", orig.size, (255, 0, 0, 0))
            alpha   = Image.fromarray((mask.astype(np.uint8) * 128), "L")
            red.putalpha(alpha)
            Image.alpha_composite(overlay, red).convert("RGB").save(
                after_dir / f"{stem}.png")
            Image.fromarray(mask).save(pred_dir / f"{stem}.png")

            rows.append({
                "stem": stem, "status": "OK",
                "latency_s" : round(elapsed, 6),
                "has_crack" : int(has_crack),
                "crack_px"  : crack_px,
                "total_px"  : total_px_im,
                "cover_pct" : round(crack_px / total_px_im * 100, 4),
            })
        except Exception as e:
            fail_cnt += 1
            rows.append({"stem": stem, "status": f"ERROR: {e}"})

    t_total = time.perf_counter() - t_start
    gpu_peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0

    # ── CSV ───────────────────────────────────────────────────────────────────
    fields = ["stem","status","latency_s","has_crack","crack_px","total_px","cover_pct"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # ── 統計 ──────────────────────────────────────────────────────────────────
    n_ok  = len(latencies)
    avg_t = float(np.mean(latencies)) if latencies else 0
    fps   = 1.0 / avg_t if avg_t > 0 else 0
    sep   = "-" * 100

    lines = [
        f"Mendeley 凍結測試集 Patch 推論記錄 — {out_name}",
        f"時間        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"模型        : {pt_path}",
        f"資料來源    : {RGB_DIR}（test.txt 70 張）",
        f"推論方式    : 512×512 滑動視窗，overlap=128，stride=384",
        f"閾值        : {threshold}",
        f"影像數      : {n_images}",
        sep,
        "硬體資訊",
        f"運算裝置    : {device}",
        f"CPU         : {get_cpu_name()}",
        f"CUDA 可用   : {torch.cuda.is_available()}",
    ]
    if device.type == "cuda":
        lines += [
            f"GPU         : {torch.cuda.get_device_name(0)}",
            f"GPU 記憶體  : {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB",
        ]
    lines += [
        f"PyTorch     : {torch.__version__}",
        sep,
        "整體統計",
        f"成功處理    : {n_ok} / {n_images} 張",
        f"失敗        : {fail_cnt} 張",
        f"有裂縫判定  : {crack_cnt} 張",
        f"推論平均    : {avg_t*1000:.3f} ms/張",
        f"FPS         : {fps:.2f}",
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
