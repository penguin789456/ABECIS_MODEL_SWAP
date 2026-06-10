import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
from scipy import ndimage

GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
DIRS = {
    'ddrnet t=0.5 (原始)': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ddrnet',
    'ddrnet t=0.3 (新)':   r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ddrnet_t03',
    'abecis_detectron2':   r'H:\ChihleeMaster\abecis_predictions',
}
MIN_PX = 50

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

def load_bin(path, thresh=128):
    img = Image.open(path).convert('L')
    return (np.array(img) >= thresh).astype(bool)

def get_path(name, img_id):
    d = DIRS[name]
    if 'abecis' in name:
        return os.path.join(d, f'{img_id}_pred.png')
    return os.path.join(d, f'{img_id}.png')

stats = {n: {'det':0,'miss':0,'tp':0,'fp':0,'fn':0} for n in DIRS}

for img_id in test_ids:
    gt_path = next((p for p in [
        os.path.join(GT_DIR, f'{img_id}.jpg'),
        os.path.join(GT_DIR, f'{img_id}.png'),
    ] if os.path.exists(p)), None)
    if not gt_path: continue

    gt = load_bin(gt_path)
    labeled, n = ndimage.label(gt)
    sizes = ndimage.sum(gt, labeled, range(1, n+1))
    valid = [i+1 for i, sz in enumerate(sizes) if sz >= MIN_PX]
    if not valid: continue

    for name in DIRS:
        p = get_path(name, img_id)
        if not os.path.exists(p): continue
        pred = load_bin(p)
        if pred.shape != gt.shape:
            pred = (np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)) >= 0.5)

        # 漏檢率
        for cid in valid:
            comp = (labeled == cid)
            if np.logical_and(pred, comp).any():
                stats[name]['det'] += 1
            else:
                stats[name]['miss'] += 1

        # pixel IoU
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()
        stats[name]['tp'] += int(tp)
        stats[name]['fp'] += int(fp)
        stats[name]['fn'] += int(fn)

print(f'\n{"模型":<28} {"偵測率":>8} {"漏檢率":>8} {"pixel IoU":>10} {"Precision":>10} {"Recall":>8}')
print('-'*78)
for name, s in stats.items():
    total = s['det'] + s['miss']
    det_r = s['det'] / total if total else 0
    iou   = s['tp'] / (s['tp']+s['fp']+s['fn']) if (s['tp']+s['fp']+s['fn']) else 0
    prec  = s['tp'] / (s['tp']+s['fp']) if (s['tp']+s['fp']) else 0
    rec   = s['tp'] / (s['tp']+s['fn']) if (s['tp']+s['fn']) else 0
    print(f'{name:<28} {det_r:>8.1%} {1-det_r:>8.1%} {iou:>10.4f} {prec:>10.4f} {rec:>8.4f}')
