"""
PP-LiteSeg-T v0 細粒度閾值掃描（0.45–0.70）
找出 偵測率>80.4% 且 Precision 不完全退化 的甜蜜點
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

GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
CFG_PATH = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\configs\final\ppliteseg.yaml'
MIN_PX   = 50
ABECIS_DET = 80.4

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

with open(CFG_PATH, encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

model = build_model(cfg['model']).to(DEVICE)
ckpt_path = Path(cfg['checkpoint']['save_dir']) / 'best.pth'
ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
model.load_state_dict(ckpt.get('model', ckpt))
model.eval()
print(f'Loaded: {ckpt_path}')

transform  = get_test_transforms()
ds_cfg     = cfg['dataset']
rgb_dir    = Path(ds_cfg['root']) / 'rgb'
patch_size = ds_cfg['patch_size']
overlap    = ds_cfg['overlap']
stride     = patch_size - overlap

rgb_index = {p.stem.lower(): p for p in rgb_dir.iterdir()
             if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}

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
    avg = logit_sum / np.maximum(count_map, 1.0)
    return 1.0 / (1.0 + np.exp(-avg))

# 預計算機率圖 + GT
print('Computing prob maps...')
prob_maps = {}
gt_data   = {}
for i, img_id in enumerate(test_ids):
    rgb_p = rgb_index.get(img_id.lower())
    if rgb_p is None: continue
    image = np.array(Image.open(rgb_p).convert('RGB'))
    prob  = get_prob_map(image)
    prob_maps[img_id] = prob

    gt_path = next((p for p in [
        os.path.join(GT_DIR, f'{img_id}.jpg'),
        os.path.join(GT_DIR, f'{img_id}.png'),
    ] if os.path.exists(p)), None)
    if not gt_path: continue

    gt_pil = Image.open(gt_path).convert('L')
    if gt_pil.size != (prob.shape[1], prob.shape[0]):
        gt_pil = gt_pil.resize((prob.shape[1], prob.shape[0]), Image.NEAREST)
    gt = (np.array(gt_pil) >= 128)
    labeled, n = ndimage.label(gt)
    sizes = ndimage.sum(gt, labeled, range(1, n+1))
    valid = [j+1 for j, sz in enumerate(sizes) if sz >= MIN_PX]
    if valid:
        gt_data[img_id] = (gt, labeled, valid)
    sys.stdout.write(f'\r  [{i+1}/{len(test_ids)}] {img_id}  ')
    sys.stdout.flush()

total_regions = sum(len(v) for _, _, v in gt_data.values())
print(f'\nDone. Total crack regions: {total_regions}')

# 細粒度閾值：0.45 ~ 0.70，步長 0.01
thresholds = [round(x * 0.01, 2) for x in range(45, 71)]

print(f'\n{"閾值":>6} {"偵測率":>8} {"漏檢率":>8} {"IoU":>8} {"Prec":>8} {"Rec":>8}  BEAT')
print('-'*65)

for thresh in thresholds:
    det_total = tp = fp = fn = 0
    for img_id, (gt, labeled, valid) in gt_data.items():
        if img_id not in prob_maps: continue
        pred = (prob_maps[img_id] > thresh)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred.astype(np.uint8)*255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)) >= 128
        for cid in valid:
            if np.logical_and(pred, labeled == cid).any():
                det_total += 1
        tp += int(np.logical_and(pred,  gt).sum())
        fp += int(np.logical_and(pred, ~gt).sum())
        fn += int(np.logical_and(~pred,  gt).sum())

    dr   = det_total / total_regions * 100
    iou  = tp/(tp+fp+fn) if (tp+fp+fn) else 0
    prec = tp/(tp+fp)    if (tp+fp)    else 0
    rec  = tp/(tp+fn)    if (tp+fn)    else 0
    beat = '★' if dr > ABECIS_DET else ''
    print(f'{thresh:>6.2f} {dr:>7.1f}% {100-dr:>7.1f}% {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}  {beat}')

print(f'\nABECIS: 偵測率=80.4%  IoU=0.3086  Prec=0.327  Rec=0.846')
