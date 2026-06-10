"""
多模型機率圖閾值掃描 + OR 融合
目標：找到超過 ABECIS 80.4% 偵測率的設定
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

MODELS = {
    'ppliteseg_v0': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg.yaml',
    'ppliteseg_v3': r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg_v3.yaml',
    'ddrnet':       r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ddrnet.yaml',
}

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

transform = get_test_transforms()

def get_prob_map(model, image, patch_size, overlap):
    H, W   = image.shape[:2]
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
all_probs = {}   # model_name -> {img_id -> prob_map}
rgb_index = None

for model_name, cfg_path in MODELS.items():
    print(f'\n[{model_name}] Loading...')
    if not os.path.exists(cfg_path):
        print(f'  Config not found: {cfg_path}')
        continue

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(cfg['checkpoint']['save_dir']) / 'best.pth'
    if not ckpt_path.exists():
        print(f'  Checkpoint not found: {ckpt_path}')
        continue

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

# ── 閾值 + 融合掃描 ───────────────────────────────────────────────────────────
ABECIS_DET = 80.4

thresholds  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
combos = [
    ('ppliteseg_v0',                    ['ppliteseg_v0']),
    ('ppliteseg_v3',                    ['ppliteseg_v3']),
    ('ddrnet',                          ['ddrnet']),
    ('v0 OR v3',                        ['ppliteseg_v0','ppliteseg_v3']),
    ('v0 OR ddrnet',                    ['ppliteseg_v0','ddrnet']),
    ('v3 OR ddrnet',                    ['ppliteseg_v3','ddrnet']),
    ('v0 OR v3 OR ddrnet',              ['ppliteseg_v0','ppliteseg_v3','ddrnet']),
]

print(f'\n{"組合":<24} {"閾值":>6} {"偵測率":>8} {"漏檢率":>8} {"IoU":>8} {"Prec":>7} {"Rec":>7}  BEAT_ABECIS')
print('-'*88)

best_results = []

for combo_name, models_in_combo in combos:
    avail = [m for m in models_in_combo if m in all_probs]
    if not avail:
        continue
    for thresh in thresholds:
        det_total = tp = fp = fn = 0
        for img_id, (gt, labeled, valid) in gt_data.items():
            preds = [all_probs[m][img_id] > thresh
                     for m in avail if img_id in all_probs[m]]
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
        beat = '★ YES' if dr > ABECIS_DET else ''
        if dr > ABECIS_DET or thresh in [0.05, 0.10, 0.20, 0.50]:
            print(f'{combo_name:<24} {thresh:>6.2f} {dr:>7.1f}% {100-dr:>7.1f}% {iou:>8.4f} {prec:>7.4f} {rec:>7.4f}  {beat}')
        if dr > ABECIS_DET:
            best_results.append((combo_name, thresh, dr, iou, prec, rec))

print(f'\n{"="*60}')
print(f'ABECIS baseline: 偵測率={ABECIS_DET}%  IoU=0.3086  Prec=0.327  Rec=0.846')
if best_results:
    print(f'\n★ 超越 ABECIS 的設定：')
    for r in best_results:
        print(f'  {r[0]} @ thresh={r[1]:.2f} → 偵測率={r[2]:.1f}%  IoU={r[3]:.4f}')
else:
    print('\n未找到超越 ABECIS 的設定，需重新訓練。')
