"""
DDRNet FTL 推論腳本 — 統一輸出版
輸出目錄：H:\ChihleeMaster\dev\final_outputs\
  logs/   → ddrnet_ftl_log.txt（每張影像結果）
  before/ → 原始 RGB 影像
  after/  → 原始影像 + 紅色裂縫遮罩疊合圖

啟動方式（CrackSeg conda env）：
  conda activate CrackSeg
  cd H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP
  python run_ddrnet_ftl_final.py

選擇性：指定閾值（預設 0.30）
  python run_ddrnet_ftl_final.py --threshold 0.30
"""
import sys, os, argparse, shutil, platform, time
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'realtime-semantic-segmentation-pytorch'))

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from data.transforms import get_test_transforms
from training.train_crackseg import build_model

# ── 設定 ────────────────────────────────────────────────────────────────────
CFG_PATH   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ddrnet_ftl.yaml'
CKPT_PATH  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\checkpoints\ddrnet_ftl\20260501_181112\best.pth'
OUT_BASE   = Path(r'H:\ChihleeMaster\dev\final_outputs')
MODEL_NAME = 'ddrnet_ftl'

def get_cpu_name() -> str:
    """嘗試取得 CPU 型號；若失敗則回傳 platform.processor()。"""
    cpu_name = os.environ.get('PROCESSOR_IDENTIFIER', '').strip()
    if cpu_name:
        return cpu_name
    try:
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'Unknown CPU'


def get_hardware_info(device: torch.device) -> dict:
    """取得本次推論使用的硬體資訊。"""
    info = {
        'device': str(device),
        'cpu_name': get_cpu_name(),
        'gpu_name': 'N/A',
        'cuda_available': torch.cuda.is_available(),
    }
    if device.type == 'cuda' and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info.update({
            'gpu_name': torch.cuda.get_device_name(idx),
            'cuda_device_index': idx,
            'gpu_memory_gb': prop.total_memory / (1024 ** 3),
        })
    return info


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(220, 50, 50), alpha=0.5) -> np.ndarray:
    """mask 白色區域疊合紅色到原始影像上"""
    out = rgb.copy().astype(np.float32)
    crack = mask >= 128
    for c, v in enumerate(color):
        out[:, :, c] = np.where(crack,
            out[:, :, c] * (1 - alpha) + v * alpha,
            out[:, :, c])
    return out.clip(0, 255).astype(np.uint8)

def stitch_patches(image, model, device, transform, patch_size=512, overlap=128, threshold=0.30):
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
                ph, pw = patch_size - patch.shape[0], patch_size - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0,ph),(0,pw),(0,0)), mode='reflect')
                aug = transform(image=patch)
                tensor = aug['image'].unsqueeze(0).float().to(device)
                logit = model(tensor).squeeze().cpu().numpy()
                ah, aw = min(patch_size, H-y), min(patch_size, W-x)
                logit_sum[y:y+ah, x:x+aw] += logit[:ah,:aw]
                count_map[y:y+ah, x:x+aw] += 1.0
    avg_logit = logit_sum / np.maximum(count_map, 1.0)
    prob = 1.0 / (1.0 + np.exp(-avg_logit))
    return (prob > threshold).astype(np.uint8) * 255

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.30,
                        help='推論閾值（預設 0.30）')
    args = parser.parse_args()
    threshold = args.threshold

    # 建立輸出目錄
    log_dir    = OUT_BASE / 'logs'
    before_dir = OUT_BASE / 'before' / MODEL_NAME
    after_dir  = OUT_BASE / 'after'  / MODEL_NAME
    for d in [log_dir, before_dir, after_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 載入設定
    with open(CFG_PATH, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hardware_info = get_hardware_info(device)
    total_start_time = time.perf_counter()

    print(f'Device : {device}')
    print(f'CPU    : {hardware_info["cpu_name"]}')
    print(f'GPU    : {hardware_info["gpu_name"]}')
    print(f'Threshold : {threshold}')
    print(f'Checkpoint: {CKPT_PATH}')

    # 載入模型
    model = build_model(cfg['model']).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    print('Model loaded.\n')

    transform = get_test_transforms()
    ds_cfg    = cfg['dataset']
    rgb_dir   = Path(ds_cfg['root']) / 'rgb'
    split_file = Path(ds_cfg['splits_dir']) / 'test.txt'

    with open(split_file) as f:
        stems = [l.strip() for l in f if l.strip()]

    rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir()
                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}

    # Log 檔
    log_path = log_dir / f'{MODEL_NAME}_log.txt'
    log_lines = [
        f'DDRNet FTL 推論記錄',
        f'時間      : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Checkpoint: {CKPT_PATH}',
        f'閾值      : {threshold}',
        f'運算裝置  : {hardware_info["device"]}',
        f'CPU       : {hardware_info["cpu_name"]}',
        f'GPU       : {hardware_info["gpu_name"]}',
        f'影像數    : {len(stems)}',
        f'輸出目錄  : {OUT_BASE}',
        '-' * 78,
        f'{"影像ID":<10}  {"判斷":>8}  {"裂縫像素數":>10}  {"覆蓋率":>8}  {"耗時秒":>8}',
        '-' * 78,
    ]

    total_images = len(stems)
    processed_images = 0
    missing_images = 0
    crack_detected_images = 0
    total_crack_px = 0
    total_pixels = 0
    per_image_times = []

    print(f'Processing {len(stems)} images...')
    for stem in tqdm(stems, desc='DDRNet FTL'):
        image_start_time = time.perf_counter()
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
            missing_images += 1
            log_lines.append(f'{stem:<10}  RGB not found')
            continue

        rgb = np.array(Image.open(rgb_p).convert('RGB'))
        mask = stitch_patches(rgb, model, device, transform,
                              patch_size=ds_cfg['patch_size'],
                              overlap=ds_cfg['overlap'],
                              threshold=threshold)

        # 儲存原始影像（before）
        shutil.copy(str(rgb_p), before_dir / f'{stem}.png')

        # 儲存疊合影像（after）
        overlay = overlay_mask(rgb, mask)
        Image.fromarray(overlay).save(after_dir / f'{stem}.png')

        # 統計
        image_elapsed = time.perf_counter() - image_start_time
        per_image_times.append(image_elapsed)

        crack_px  = int((mask >= 128).sum())
        image_pixels = int(mask.shape[0] * mask.shape[1])
        coverage  = crack_px / image_pixels * 100
        judgement = '有裂縫' if crack_px > 0 else '無裂縫'

        processed_images += 1
        total_crack_px += crack_px
        total_pixels += image_pixels
        if crack_px > 0:
            crack_detected_images += 1

        log_lines.append(
            f'{stem:<10}  {judgement:>8}  {crack_px:>10}px  {coverage:>7.2f}%  {image_elapsed:>8.2f}'
        )

    total_elapsed = time.perf_counter() - total_start_time
    judgement_rate = (crack_detected_images / processed_images * 100) if processed_images else 0.0
    avg_coverage = (total_crack_px / total_pixels * 100) if total_pixels else 0.0
    avg_time = (sum(per_image_times) / len(per_image_times)) if per_image_times else 0.0

    log_lines += [
        '-' * 78,
        '整體統計',
        f'總影像數          : {total_images}',
        f'成功處理影像數    : {processed_images}',
        f'缺少 RGB 影像數   : {missing_images}',
        f'判定有裂縫影像數  : {crack_detected_images}',
        f'判定無裂縫影像數  : {processed_images - crack_detected_images}',
        f'判斷率            : {judgement_rate:.2f}%  # 有裂縫判定張數 / 成功處理張數',
        f'總裂縫像素數      : {total_crack_px}px',
        f'平均覆蓋率        : {avg_coverage:.4f}%',
        f'總耗時            : {total_elapsed:.2f} 秒',
        f'平均每張耗時      : {avg_time:.2f} 秒/張',
        f'運算裝置          : {hardware_info["device"]}',
        f'CPU               : {hardware_info["cpu_name"]}',
        f'GPU               : {hardware_info["gpu_name"]}',
        f'CUDA 可用          : {hardware_info["cuda_available"]}',
    ]
    if hardware_info.get('gpu_memory_gb') is not None:
        log_lines.append(f'GPU 記憶體         : {hardware_info["gpu_memory_gb"]:.2f} GB')

    log_lines += [
        '-' * 78,
        f'完成：{processed_images}/{total_images} 張',
        f'before 目錄: {before_dir}',
        f'after  目錄: {after_dir}',
    ]

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print(f'\n✅ 完成')
    print(f'   logs  → {log_path}')
    print(f'   before→ {before_dir}')
    print(f'   after → {after_dir}')

if __name__ == '__main__':
    main()
