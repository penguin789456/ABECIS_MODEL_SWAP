"""
測試三種提升偵測率的方法（不需重新訓練）
1. 形態學膨脹（Dilation）後處理
2. 多模型 OR 融合（Ensemble）
3. 不同膨脹半徑比較
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation, generate_binary_structure

PRED_DIRS = {
    'ppliteseg':           r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ppliteseg',
    'deeplabv3_mobilenet': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\deeplabv3_mobilenet',
    'ddrnet':              r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ddrnet',
    'maskrcnn':            r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\maskrcnn',
}
GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
MIN_PX   = 50

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

def load_binary(path, threshold=128):
    img = Image.open(path).convert('L')
    return (np.array(img) >= threshold).astype(bool)

def detection_rate(pred, gt_labeled, valid_comps):
    detected = sum(
        np.logical_and(pred, gt_labeled == cid).any()
        for cid in valid_comps
    )
    return detected, len(valid_comps) - detected

def dilate_mask(mask, radius):
    if radius == 0:
        return mask
    struct = generate_binary_structure(2, 1)
    return binary_dilation(mask, structure=struct, iterations=radius)

# ── 收集所有 GT 連通區域 ──────────────────────────────────────────────────────
all_gt     = {}  # img_id -> (labeled, valid_comps)
all_preds  = {m: {} for m in PRED_DIRS}

for img_id in test_ids:
    gt_path = next((p for p in [
        os.path.join(GT_DIR, f'{img_id}.jpg'),
        os.path.join(GT_DIR, f'{img_id}.png'),
    ] if os.path.exists(p)), None)
    if not gt_path:
        continue

    gt = load_binary(gt_path)
    labeled, n = ndimage.label(gt)
    sizes = ndimage.sum(gt, labeled, range(1, n+1))
    valid = [i+1 for i, sz in enumerate(sizes) if sz >= MIN_PX]
    if not valid:
        continue
    all_gt[img_id] = (labeled, valid)

    for model, d in PRED_DIRS.items():
        p = os.path.join(d, f'{img_id}.png')
        if os.path.exists(p):
            pred = load_binary(p)
            if pred.shape != gt.shape:
                pred = (np.array(Image.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST)) >= 0.5)
            all_preds[model][img_id] = pred

total_regions = sum(len(v) for _, v in all_gt.values())

# ── 方法 1：DDRNet 單模型，不同膨脹半徑 ──────────────────────────────────────
print(f'\n【方法 1】DDRNet 膨脹後處理（total regions={total_regions}）')
print(f'{"膨脹半徑":>8} {"偵測到":>8} {"漏掉":>6} {"偵測率":>8} {"漏檢率":>8}')
print('-'*50)
for radius in [0, 3, 5, 8, 12, 20]:
    det_total = 0
    for img_id, (labeled, valid) in all_gt.items():
        if img_id not in all_preds['ddrnet']:
            continue
        pred_d = dilate_mask(all_preds['ddrnet'][img_id], radius)
        d, _ = detection_rate(pred_d, labeled, valid)
        det_total += d
    missed = total_regions - det_total
    rate   = det_total / total_regions
    print(f'{radius:>8}px {det_total:>8} {missed:>6} {rate:>8.1%} {1-rate:>8.1%}')

# ── 方法 2：多模型 OR 融合 ────────────────────────────────────────────────────
print(f'\n【方法 2】OR 融合組合')
print(f'{"組合":^40} {"偵測到":>8} {"偵測率":>8}')
print('-'*55)

combos = [
    ('DDRNet only',                      ['ddrnet']),
    ('DDRNet + PP-LiteSeg',             ['ddrnet','ppliteseg']),
    ('DDRNet + DeepLabV3',              ['ddrnet','deeplabv3_mobilenet']),
    ('DDRNet + MaskRCNN',               ['ddrnet','maskrcnn']),
    ('PP + DL + DDR (3模型)',           ['ppliteseg','deeplabv3_mobilenet','ddrnet']),
    ('全4模型 OR',                       ['ppliteseg','deeplabv3_mobilenet','ddrnet','maskrcnn']),
]

for name, models in combos:
    det_total = 0
    for img_id, (labeled, valid) in all_gt.items():
        preds_avail = [all_preds[m][img_id] for m in models if img_id in all_preds[m]]
        if not preds_avail:
            continue
        combined = np.logical_or.reduce(preds_avail)
        d, _ = detection_rate(combined, labeled, valid)
        det_total += d
    rate = det_total / total_regions
    print(f'{name:<40} {det_total:>8} {rate:>8.1%}')

# ── 方法 3：DDRNet 膨脹 + OR 融合組合 ────────────────────────────────────────
print(f'\n【方法 3】DDRNet 膨脹(r=8) + OR 融合')
print(f'{"組合":^40} {"偵測到":>8} {"偵測率":>8}')
print('-'*55)

combos2 = [
    ('DDRNet dil8 only',                       {'ddrnet':8}),
    ('DDRNet dil8 + PP-LiteSeg dil3',         {'ddrnet':8,'ppliteseg':3}),
    ('DDRNet dil8 + DeepLabV3 dil3',          {'ddrnet':8,'deeplabv3_mobilenet':3}),
    ('DDRNet dil8 + 全3語意模型 dil3',         {'ddrnet':8,'ppliteseg':3,'deeplabv3_mobilenet':3}),
    ('全4模型 dil5',                            {'ddrnet':5,'ppliteseg':5,'deeplabv3_mobilenet':5,'maskrcnn':5}),
]

for name, model_radii in combos2:
    det_total = 0
    for img_id, (labeled, valid) in all_gt.items():
        preds_avail = []
        for m, r in model_radii.items():
            if img_id in all_preds[m]:
                preds_avail.append(dilate_mask(all_preds[m][img_id], r))
        if not preds_avail:
            continue
        combined = np.logical_or.reduce(preds_avail)
        d, _ = detection_rate(combined, labeled, valid)
        det_total += d
    rate = det_total / total_regions
    print(f'{name:<40} {det_total:>8} {rate:>8.1%}')

print(f'\n（ABECIS Detectron2 baseline: 144/179 = 80.4%）')
