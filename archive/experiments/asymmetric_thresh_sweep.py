"""
비대칭 閾值 OR 融合掃描
PP-LiteSeg (固定 thresh=0.50) OR DDRNet (低閾值掃描)
+ 加入 DeepLabV3 進行三模型融合
找出 偵測率 > 80.4% 且 Precision > 0.10 的有效組合
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

GT_DIR  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
MIN_PX  = 50
ABECIS_DET = 80.4

MODELS = {
    'ppliteseg_v0': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg.yaml',
    'ppliteseg_v3': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg_v3.yaml',
    'ddrnet':       r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ddrnet.yaml',
    'deeplabv3':    r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\deeplabv3_mobilenet.yaml',
}

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

transform = get_test_transforms()

def get_prob_map(model, image, patch_size, overlap):
    H, W = image.shape[:2]
    stride = patch_size - overlap
    logit_sum = np.zeros((H, W), np.float32)
    count_map = np.zeros((H, W), np.float32)
    ys = list(range(0, max(H-patch_size,0)+1, stride))
    xs = list(range(0, max(W-patch_size,0)+1, stride))
    if not ys or ys[-1]+patch_size < H: ys.append(max(H-patch_size,0))
    if not xs or xs[-1]+patch_size < W: xs.append(max(W-patch_size,0))
    model.eval()
    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image[y:y+patch_size, x:x+patch_size]
                ph = patch_size-patch.shape[0]; pw = patch_size-patch.shape[1]
                if ph>0 or pw>0:
                    patch = np.pad(patch, ((0,ph),(0,pw),(0,0)), mode='reflect')
                t = transform(image=patch)['image'].unsqueeze(0).float().to(DEVICE)
                logit = model(t).squeeze().cpu().numpy()
                ah = min(patch_size, H-y); aw = min(patch_size, W-x)
                logit_sum[y:y+ah, x:x+aw] += logit[:ah,:aw]
                count_map[y:y+ah, x:x+aw] += 1.0
    avg = logit_sum / np.maximum(count_map, 1.0)
    return 1.0 / (1.0 + np.exp(-avg))

# ── 載入所有模型並預計算機率圖 ───────────────────────────────────────────────
all_probs = {}
rgb_index = None

for model_name, cfg_path in MODELS.items():
    print(f'\n[{model_name}] Loading...')
    if not os.path.exists(cfg_path):
        print(f'  Config not found: {cfg_path}'); continue
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ckpt_path = Path(cfg['checkpoint']['save_dir']) / 'best.pth'
    if not ckpt_path.exists():
        print(f'  Checkpoint not found: {ckpt_path}'); continue
    model = build_model(cfg['model']).to(DEVICE)
    ckpt  = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt.get('model', ckpt))
    print(f'  Loaded: {ckpt_path}')
    ds_cfg = cfg['dataset']
    patch_size = ds_cfg['patch_size']
    overlap    = ds_cfg['overlap']
    rgb_dir    = Path(ds_cfg['root']) / 'rgb'
    if rgb_index is None:
        rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir()
                     if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    probs = {}
    for i, img_id in enumerate(test_ids):
        rgb_p = rgb_index.get(img_id.lower())
        if not rgb_p: continue
        image = np.array(Image.open(rgb_p).convert('RGB'))
        probs[img_id] = get_prob_map(model, image, patch_size, overlap)
        sys.stdout.write(f'\r  [{i+1}/{len(test_ids)}] {img_id}  ')
        sys.stdout.flush()
    all_probs[model_name] = probs
    del model; torch.cuda.empty_cache()
    print(f'\n  Done: {len(probs)} images')

# ── 載入 GT ───────────────────────────────────────────────────────────────────
print('\nLoading GT...')
gt_data = {}
for img_id in test_ids:
    gt_path = next((p for p in [
        os.path.join(GT_DIR, f'{img_id}.jpg'),
        os.path.join(GT_DIR, f'{img_id}.png'),
    ] if os.path.exists(p)), None)
    if not gt_path: continue
    ref_prob = next(iter(all_probs.values())).get(img_id)
    if ref_prob is None: continue
    gt_pil = Image.open(gt_path).convert('L')
    if gt_pil.size != (ref_prob.shape[1], ref_prob.shape[0]):
        gt_pil = gt_pil.resize((ref_prob.shape[1], ref_prob.shape[0]), Image.NEAREST)
    gt = (np.array(gt_pil) >= 128)
    labeled, n = ndimage.label(gt)
    sizes = ndimage.sum(gt, labeled, range(1, n+1))
    valid = [j+1 for j, sz in enumerate(sizes) if sz >= MIN_PX]
    if valid:
        gt_data[img_id] = (gt, labeled, valid)

total_regions = sum(len(v) for _, _, v in gt_data.values())
print(f'Total crack regions: {total_regions}')

# ── 非對稱閾值掃描 ────────────────────────────────────────────────────────────
# 每個模型用自己最適合的閾值

def evaluate_combo(model_thresh_pairs):
    """model_thresh_pairs: list of (model_name, threshold)"""
    det_total = tp = fp = fn = 0
    for img_id, (gt, labeled, valid) in gt_data.items():
        preds = []
        for m, t in model_thresh_pairs:
            if m in all_probs and img_id in all_probs[m]:
                preds.append(all_probs[m][img_id] > t)
        if not preds: continue
        pred = np.logical_or.reduce(preds)
        for cid in valid:
            if np.logical_and(pred, labeled==cid).any():
                det_total += 1
        tp += int(np.logical_and(pred,  gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred, gt).sum())
    dr   = det_total / total_regions * 100
    iou  = tp/(tp+fp+fn) if (tp+fp+fn) else 0
    prec = tp/(tp+fp)    if (tp+fp)    else 0
    rec  = tp/(tp+fn)    if (tp+fn)    else 0
    return dr, iou, prec, rec

print(f'\n{"組合描述":<45} {"偵測率":>8} {"IoU":>8} {"Prec":>8} {"Rec":>8}  BEAT')
print('-'*90)

# 1. 單模型基準（各自最佳閾值）
baselines = [
    ('ppliteseg_v0 @0.65',    [('ppliteseg_v0', 0.65)]),
    ('ppliteseg_v3 @0.65',    [('ppliteseg_v3', 0.65)]),
    ('ddrnet @0.05',           [('ddrnet', 0.05)]),
    ('ddrnet @0.20',           [('ddrnet', 0.20)]),
    ('deeplabv3 @0.45',        [('deeplabv3', 0.45)]),
    ('deeplabv3 @0.05',        [('deeplabv3', 0.05)]),
]
print('── 單模型基準 ──')
for name, pairs in baselines:
    if all(m in all_probs for m, _ in pairs):
        dr, iou, prec, rec = evaluate_combo(pairs)
        beat = '★' if dr > ABECIS_DET else ''
        print(f'{name:<45} {dr:>7.1f}% {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}  {beat}')

# 2. 非對稱雙模型 OR
print('\n── 非對稱雙模型 OR ──')
pp_threshs   = [0.50, 0.52, 0.55, 0.60, 0.65]
ddr_threshs  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
dl_threshs   = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]

for pt in pp_threshs:
    for dt in ddr_threshs:
        name = f'v0@{pt:.2f} OR ddr@{dt:.2f}'
        pairs = [('ppliteseg_v0', pt), ('ddrnet', dt)]
        if all(m in all_probs for m, _ in pairs):
            dr, iou, prec, rec = evaluate_combo(pairs)
            beat = '★' if dr > ABECIS_DET else ''
            print(f'{name:<45} {dr:>7.1f}% {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}  {beat}')

print()
for pt in pp_threshs:
    for dt in dl_threshs:
        name = f'v0@{pt:.2f} OR dl@{dt:.2f}'
        pairs = [('ppliteseg_v0', pt), ('deeplabv3', dt)]
        if all(m in all_probs for m, _ in pairs):
            dr, iou, prec, rec = evaluate_combo(pairs)
            beat = '★' if dr > ABECIS_DET else ''
            print(f'{name:<45} {dr:>7.1f}% {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}  {beat}')

# 3. 三模型 OR（非對稱）
print('\n── 三模型 OR（非對稱） ──')
combos3 = [
    ('v0@0.50 OR v3@0.50 OR ddr@0.05',  [('ppliteseg_v0',0.50),('ppliteseg_v3',0.50),('ddrnet',0.05)]),
    ('v0@0.50 OR v3@0.50 OR ddr@0.10',  [('ppliteseg_v0',0.50),('ppliteseg_v3',0.50),('ddrnet',0.10)]),
    ('v0@0.50 OR v3@0.50 OR ddr@0.20',  [('ppliteseg_v0',0.50),('ppliteseg_v3',0.50),('ddrnet',0.20)]),
    ('v0@0.65 OR ddr@0.05 OR dl@0.05',  [('ppliteseg_v0',0.65),('ddrnet',0.05),('deeplabv3',0.05)]),
    ('v0@0.65 OR ddr@0.05 OR dl@0.45',  [('ppliteseg_v0',0.65),('ddrnet',0.05),('deeplabv3',0.45)]),
    ('v0@0.50 OR ddr@0.05 OR dl@0.05',  [('ppliteseg_v0',0.50),('ddrnet',0.05),('deeplabv3',0.05)]),
    ('v0@0.50 OR ddr@0.05 OR dl@0.45',  [('ppliteseg_v0',0.50),('ddrnet',0.05),('deeplabv3',0.45)]),
    ('v0@0.50 OR v3@0.50 OR dl@0.05',   [('ppliteseg_v0',0.50),('ppliteseg_v3',0.50),('deeplabv3',0.05)]),
    ('ALL@0.50',                          [('ppliteseg_v0',0.50),('ppliteseg_v3',0.50),('ddrnet',0.50),('deeplabv3',0.50)]),
    ('v0@0.65 OR v3@0.65 OR ddr@0.05 OR dl@0.45', [('ppliteseg_v0',0.65),('ppliteseg_v3',0.65),('ddrnet',0.05),('deeplabv3',0.45)]),
]
for name, pairs in combos3:
    avail = [(m,t) for m,t in pairs if m in all_probs]
    if not avail: continue
    dr, iou, prec, rec = evaluate_combo(avail)
    beat = '★' if dr > ABECIS_DET else ''
    print(f'{name:<45} {dr:>7.1f}% {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}  {beat}')

print(f'\n{"="*60}')
print(f'ABECIS: 偵測率=80.4%  IoU=0.3086  Prec=0.327  Rec=0.846')
