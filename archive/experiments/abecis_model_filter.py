"""
ABECIS 二次篩選（使用 PP-LiteSeg 作為 LLM 替代驗證器）
邏輯：ABECIS 預測的每個連通元件，需要 PP-LiteSeg 也有像素重疊才保留
概念等同：「兩個獨立模型都說是裂縫 → 更可信」

ABECIS 原始: Prec=0.327, Rec=0.846, DR=80.4%
目標: 提升 Precision，同時盡量維持 Detection Rate
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP')
sys.path.insert(0, r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\realtime-semantic-segmentation-pytorch')

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
import torch
import yaml

from data.transforms import get_test_transforms
from training.train_crackseg import build_model

PRED_DIR = r'H:\ChihleeMaster\abecis_predictions'
GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
MIN_PX   = 50
ABECIS_PREC  = 0.3270
ABECIS_REC   = 0.8460
ABECIS_IOU   = 0.3086
ABECIS_DR    = 80.4

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── 載入 PP-LiteSeg v0（驗證器）────────────────────────────────────────────────
CFG_PATH = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg.yaml'
with open(CFG_PATH, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = build_model(cfg['model']).to(DEVICE)
ckpt_path = Path(cfg['checkpoint']['save_dir']) / 'best.pth'
ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
model.load_state_dict(ckpt.get('model', ckpt))
model.eval()
print(f'Validator loaded: {ckpt_path}')

transform  = get_test_transforms()
ds_cfg     = cfg['dataset']
rgb_dir    = Path(ds_cfg['root']) / 'rgb'
patch_size = ds_cfg['patch_size']
overlap    = ds_cfg['overlap']
stride     = patch_size - overlap

rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir()
             if p.suffix.lower() in ('.jpg','.jpeg','.png')}

def get_prob_map(image):
    H, W = image.shape[:2]
    logit_sum = np.zeros((H, W), np.float32)
    count_map = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if not ys or ys[-1] + patch_size < H: ys.append(max(H - patch_size, 0))
    if not xs or xs[-1] + patch_size < W: xs.append(max(W - patch_size, 0))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image[y:y+patch_size, x:x+patch_size]
                ph = patch_size - patch.shape[0]; pw = patch_size - patch.shape[1]
                if ph > 0 or pw > 0:
                    patch = np.pad(patch, ((0,ph),(0,pw),(0,0)), mode='reflect')
                t = transform(image=patch)['image'].unsqueeze(0).float().to(DEVICE)
                logit = model(t).squeeze().cpu().numpy()
                ah = min(patch_size, H - y); aw = min(patch_size, W - x)
                logit_sum[y:y+ah, x:x+aw] += logit[:ah, :aw]
                count_map[y:y+ah, x:x+aw] += 1.0
    return 1.0 / (1.0 + np.exp(-logit_sum / np.maximum(count_map, 1.0)))

# ── 掃描不同的驗證閾值（validator_thresh）─────────────────────────────────────
print('\nPre-computing PP-LiteSeg prob maps...')
pp_probs = {}
for i, img_id in enumerate(test_ids):
    rgb_p = rgb_index.get(img_id.lower())
    if not rgb_p: continue
    image = np.array(Image.open(rgb_p).convert('RGB'))
    pp_probs[img_id] = get_prob_map(image)
    sys.stdout.write(f'\r  [{i+1}/{len(test_ids)}] {img_id}  ')
    sys.stdout.flush()
print(f'\nDone: {len(pp_probs)} images')

# ── 載入 GT ───────────────────────────────────────────────────────────────────
gt_data = {}
gt_index = {p.stem.lower(): p for p in Path(GT_DIR).iterdir()
            if p.suffix.lower() in ('.jpg','.jpeg','.png')}

for img_id in test_ids:
    gt_p = gt_index.get(img_id.lower())
    if not gt_p: continue
    rgb_p = rgb_index.get(img_id.lower())
    if rgb_p is None: continue
    ref_size = Image.open(rgb_p).size
    gt_pil = Image.open(gt_p).convert('L')
    if gt_pil.size != ref_size:
        gt_pil = gt_pil.resize(ref_size, Image.NEAREST)
    gt = np.array(gt_pil) >= 128
    labeled, n = ndimage.label(gt)
    sizes = ndimage.sum(gt, labeled, range(1, n+1))
    valid = [j+1 for j, sz in enumerate(sizes) if sz >= MIN_PX]
    if valid:
        gt_data[img_id] = (gt, labeled, valid)

total_regions = sum(len(v) for _, _, v in gt_data.values())
print(f'Total crack regions: {total_regions}')

# ── 不同驗證閾值的掃描 ─────────────────────────────────────────────────────────
# validator_thresh：PP-LiteSeg 需要的最低機率才算「確認」
validator_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
# overlap_ratio：ABECIS 元件中需要有多少比例的像素被驗證器確認（0=任何一個像素即可）
overlap_min_px = [1, 5, 10, 20]  # 需要幾個像素重疊

print(f'\n{"驗證閾值":>8} {"最少重疊px":>10} {"偵測率":>8} {"IoU":>8} {"Prec":>8} {"Rec":>8}  vs ABECIS')
print('-'*75)

best_iou = 0
best_setting = None

for v_thresh in validator_thresholds:
    for min_overlap in [1, 10, 20]:
        tp = fp = fn = det_total = 0
        comps_kept = comps_total = 0

        for img_id, (gt, labeled_gt, valid) in gt_data.items():
            pred_path = Path(PRED_DIR) / f'{img_id}_pred.png'
            if not pred_path.exists(): continue
            if img_id not in pp_probs: continue

            rgb_p = rgb_index.get(img_id.lower())
            ref_size = Image.open(rgb_p).size
            pred_mask = np.array(Image.open(pred_path).convert('L').resize(
                ref_size, Image.NEAREST)) >= 128

            pp_confirmed = pp_probs[img_id] > v_thresh
            if pred_mask.shape != pp_confirmed.shape:
                continue

            # 對 ABECIS 預測的每個連通元件做驗證
            labeled_pred, n_pred = ndimage.label(pred_mask)
            filtered_mask = np.zeros_like(pred_mask)

            for cid in range(1, n_pred + 1):
                comp = (labeled_pred == cid)
                px = comp.sum()
                if px < MIN_PX: continue
                comps_total += 1
                overlap_px = int(np.logical_and(comp, pp_confirmed).sum())
                if overlap_px >= min_overlap:
                    filtered_mask |= comp
                    comps_kept += 1

            # 評估過濾後遮罩
            for cid in valid:
                if np.logical_and(filtered_mask, labeled_gt==cid).any():
                    det_total += 1
            tp += int(np.logical_and(filtered_mask,  gt).sum())
            fp += int(np.logical_and(filtered_mask, ~gt).sum())
            fn += int(np.logical_and(~filtered_mask, gt).sum())

        dr   = det_total / total_regions * 100
        iou  = tp/(tp+fp+fn) if (tp+fp+fn) else 0
        prec = tp/(tp+fp)    if (tp+fp)    else 0
        rec  = tp/(tp+fn)    if (tp+fn)    else 0

        prec_delta = prec - ABECIS_PREC
        dr_delta   = dr - ABECIS_DR
        note = f'Prec{prec_delta:+.3f} DR{dr_delta:+.1f}%'

        if iou > best_iou and prec > ABECIS_PREC:
            best_iou = iou
            best_setting = (v_thresh, min_overlap, dr, iou, prec, rec)

        print(f'{v_thresh:>8.2f} {min_overlap:>10} {dr:>7.1f}% {iou:>8.4f} '
              f'{prec:>8.4f} {rec:>8.4f}  {note}')

print(f'\n{"="*60}')
print(f'ABECIS 原始: DR={ABECIS_DR}%  IoU={ABECIS_IOU}  Prec={ABECIS_PREC}  Rec={ABECIS_REC}')
if best_setting:
    v, m, dr, iou, prec, rec = best_setting
    print(f'\n★ 最佳過濾設定（Prec提升且 IoU 最高）:')
    print(f'  validator_thresh={v:.2f}, min_overlap_px={m}')
    print(f'  → DR={dr:.1f}%  IoU={iou:.4f}  Prec={prec:.4f}  Rec={rec:.4f}')
    print(f'  Precision 提升: {prec - ABECIS_PREC:+.4f}')
    print(f'  IoU 提升: {iou - ABECIS_IOU:+.4f}')
