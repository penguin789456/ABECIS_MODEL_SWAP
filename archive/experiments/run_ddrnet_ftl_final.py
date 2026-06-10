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
import sys, os, argparse, shutil
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
    print(f'Device : {device}')
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
        f'影像數    : {len(stems)}',
        f'輸出目錄  : {OUT_BASE}',
        '-' * 50,
        f'{"影像ID":<10}  {"裂縫像素數":>10}  {"覆蓋率":>8}',
        '-' * 50,
    ]

    print(f'Processing {len(stems)} images...')
    for stem in tqdm(stems, desc='DDRNet FTL'):
        rgb_p = rgb_index.get(stem.lower())
        if rgb_p is None:
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
        crack_px  = int((mask >= 128).sum())
        coverage  = crack_px / (mask.shape[0] * mask.shape[1]) * 100
        log_lines.append(f'{stem:<10}  {crack_px:>10}px  {coverage:>7.2f}%')

    log_lines += [
        '-' * 50,
        f'完成：{len(stems)} 張',
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
