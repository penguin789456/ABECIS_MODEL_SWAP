"""
逐圖 pixel-level IoU 比較：ABECIS 預測 vs PP-LiteSeg 預測
輸出：comparison_per_image.csv
"""
import sys, os, csv
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from PIL import Image

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
PPLITESEG_DIR  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ppliteseg'
ABECIS_DIR     = r'H:\ChihleeMaster\abecis_predictions'
GT_MASK_DIR    = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS_DIR     = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits'
OUT_CSV        = r'H:\ChihleeMaster\comparison_per_image.csv'

# ── 讀取測試集 image id ───────────────────────────────────────────────────────
test_txt = os.path.join(SPLITS_DIR, 'test.txt')
with open(test_txt) as f:
    test_ids = [line.strip() for line in f if line.strip()]
print(f'Test set: {len(test_ids)} images')

# ── pixel-level IoU 計算 ──────────────────────────────────────────────────────
def pixel_iou(pred_arr, gt_arr):
    """Both arrays should be binary (0/1) numpy arrays of same shape."""
    pred = pred_arr.astype(bool)
    gt   = gt_arr.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = tp + fp + fn
    if union == 0:
        return 1.0  # both empty → perfect
    return float(tp) / float(union)

def pixel_precision_recall(pred_arr, gt_arr):
    pred = pred_arr.astype(bool)
    gt   = gt_arr.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)

def load_binary(path, threshold=128):
    img = Image.open(path).convert('L')
    arr = np.array(img)
    return (arr >= threshold).astype(np.uint8)

# ── 逐圖比較 ─────────────────────────────────────────────────────────────────
rows = []
missing = []

for img_id in test_ids:
    # GT mask path (BW folder, same filename as rgb)
    gt_candidates = [
        os.path.join(GT_MASK_DIR, f'{img_id}.jpg'),
        os.path.join(GT_MASK_DIR, f'{img_id}.png'),
    ]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)

    # PP-LiteSeg prediction
    pp_path = os.path.join(PPLITESEG_DIR, f'{img_id}.png')

    # ABECIS prediction (filename = {id}_pred.png)
    ab_path = os.path.join(ABECIS_DIR, f'{img_id}_pred.png')

    if not gt_path:
        missing.append(f'GT missing: {img_id}')
        continue
    if not os.path.exists(pp_path):
        missing.append(f'PP-LiteSeg missing: {img_id}')
        continue
    if not os.path.exists(ab_path):
        missing.append(f'ABECIS missing: {img_id}')
        continue

    gt   = load_binary(gt_path)
    pp   = load_binary(pp_path)
    ab   = load_binary(ab_path)

    # Resize to GT size if needed
    if pp.shape != gt.shape:
        pp_img = Image.fromarray(pp * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        pp = (np.array(pp_img) >= 128).astype(np.uint8)
    if ab.shape != gt.shape:
        ab_img = Image.fromarray(ab * 255).resize((gt.shape[1], gt.shape[0]), Image.NEAREST)
        ab = (np.array(ab_img) >= 128).astype(np.uint8)

    pp_iou  = pixel_iou(pp, gt)
    ab_iou  = pixel_iou(ab, gt)
    pp_prec, pp_rec = pixel_precision_recall(pp, gt)
    ab_prec, ab_rec = pixel_precision_recall(ab, gt)

    gt_crack_px  = int(gt.sum())
    pp_pred_px   = int(pp.sum())
    ab_pred_px   = int(ab.sum())

    rows.append({
        'image_id':      img_id,
        'gt_crack_px':   gt_crack_px,
        'pp_pred_px':    pp_pred_px,
        'ab_pred_px':    ab_pred_px,
        'pp_iou':        round(pp_iou, 6),
        'ab_iou':        round(ab_iou, 6),
        'pp_beats_ab':   1 if pp_iou > ab_iou else 0,
        'pp_prec':       round(pp_prec, 4),
        'pp_rec':        round(pp_rec, 4),
        'ab_prec':       round(ab_prec, 4),
        'ab_rec':        round(ab_rec, 4),
    })

# ── 輸出 CSV ──────────────────────────────────────────────────────────────────
fieldnames = ['image_id','gt_crack_px','pp_pred_px','ab_pred_px',
              'pp_iou','ab_iou','pp_beats_ab','pp_prec','pp_rec','ab_prec','ab_rec']

with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

# ── 摘要 ─────────────────────────────────────────────────────────────────────
if rows:
    pp_ious = [r['pp_iou'] for r in rows]
    ab_ious = [r['ab_iou'] for r in rows]
    pp_precs = [r['pp_prec'] for r in rows]
    ab_precs = [r['ab_prec'] for r in rows]
    pp_recs  = [r['pp_rec']  for r in rows]
    ab_recs  = [r['ab_rec']  for r in rows]
    pp_wins  = sum(r['pp_beats_ab'] for r in rows)

    print(f'\n{"指標":<20} {"PP-LiteSeg":>12} {"ABECIS(pixel)":>14}')
    print('-'*50)
    print(f'{"Mean pixel IoU":<20} {np.mean(pp_ious):>12.4f} {np.mean(ab_ious):>14.4f}')
    print(f'{"Mean Precision":<20} {np.mean(pp_precs):>12.4f} {np.mean(ab_precs):>14.4f}')
    print(f'{"Mean Recall":<20} {np.mean(pp_recs):>12.4f} {np.mean(ab_recs):>14.4f}')
    print(f'{"PP wins (per img)":<20} {pp_wins:>12} / {len(rows)}')
    print(f'\n（ABECIS 原始 instance IoU = 0.303，← 與 pixel IoU 不可直接比較）')

if missing:
    print('\n[WARN] Missing files:')
    for m in missing:
        print(' ', m)

print(f'\nSaved: {OUT_CSV}')
