"""
裂縫連通區域漏偵測分析
=======================
對每張 GT mask 做連通元件分析，檢查每個裂縫區域是否被各模型偵測到。
「偵測到」= 預測 mask 與該連通區域有任何像素重疊（overlap > 0）
輸出：detection_rate_summary.csv + detection_rate_per_image.csv
"""
import sys, os, csv
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image
from scipy import ndimage

# ── 路徑 ──────────────────────────────────────────────────────────────────────
PRED_DIRS = {
    'ppliteseg':          r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ppliteseg',
    'deeplabv3_mobilenet':r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\deeplabv3_mobilenet',
    'ddrnet':             r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\ddrnet',
    'maskrcnn':           r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\maskrcnn',
    'abecis_detectron2':  r'H:\ChihleeMaster\abecis_predictions',
}
GT_DIR    = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS    = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
OUT_DIR   = r'H:\ChihleeMaster'
MIN_COMPONENT_PX = 50   # 小於此像素數的連通區域視為雜訊，忽略

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

def load_binary(path, threshold=128):
    img = Image.open(path).convert('L')
    return (np.array(img) >= threshold).astype(np.uint8)

def get_pred_path(model, img_id):
    d = PRED_DIRS[model]
    if model == 'abecis_detectron2':
        return os.path.join(d, f'{img_id}_pred.png')
    return os.path.join(d, f'{img_id}.png')

# ── 逐圖分析 ─────────────────────────────────────────────────────────────────
per_image_rows = []
model_stats = {m: {'total':0,'detected':0,'missed':0} for m in PRED_DIRS}

for img_id in test_ids:
    gt_candidates = [os.path.join(GT_DIR, f'{img_id}.jpg'),
                     os.path.join(GT_DIR, f'{img_id}.png')]
    gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
    if not gt_path:
        continue

    gt = load_binary(gt_path)

    # 連通元件分析（4-connectivity）
    labeled, n_components = ndimage.label(gt)
    component_sizes = ndimage.sum(gt, labeled, range(1, n_components+1))

    # 過濾雜訊小區域
    valid_components = [i+1 for i, sz in enumerate(component_sizes)
                        if sz >= MIN_COMPONENT_PX]
    n_valid = len(valid_components)

    if n_valid == 0:
        continue  # 此圖無有效裂縫區域

    row = {'image_id': img_id, 'n_crack_regions': n_valid}

    for model in PRED_DIRS:
        pred_path = get_pred_path(model, img_id)
        if not os.path.exists(pred_path):
            row[f'{model}_detected'] = 'N/A'
            row[f'{model}_missed']   = 'N/A'
            row[f'{model}_rate']     = 'N/A'
            continue

        pred = load_binary(pred_path)
        if pred.shape != gt.shape:
            pred_img = Image.fromarray(pred*255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)
            pred = (np.array(pred_img) >= 128).astype(np.uint8)

        detected = 0
        for comp_id in valid_components:
            comp_mask = (labeled == comp_id)
            # 若預測與此連通區域有任何重疊 → 偵測到
            if np.logical_and(pred.astype(bool), comp_mask).any():
                detected += 1

        missed = n_valid - detected
        rate   = detected / n_valid if n_valid > 0 else 1.0

        row[f'{model}_detected'] = detected
        row[f'{model}_missed']   = missed
        row[f'{model}_rate']     = round(rate, 4)

        model_stats[model]['total']    += n_valid
        model_stats[model]['detected'] += detected
        model_stats[model]['missed']   += missed

    per_image_rows.append(row)

# ── 摘要輸出 ──────────────────────────────────────────────────────────────────
print(f'\n{"模型":<25} {"偵測到":>8} {"漏掉":>6} {"總計":>6} {"偵測率":>8} {"漏檢率":>8}')
print('-'*65)
summary_rows = []
for model, s in model_stats.items():
    if s['total'] == 0:
        continue
    rate     = s['detected'] / s['total']
    miss_rate = s['missed']  / s['total']
    print(f'{model:<25} {s["detected"]:>8} {s["missed"]:>6} {s["total"]:>6} '
          f'{rate:>8.1%} {miss_rate:>8.1%}')
    summary_rows.append({
        'model': model,
        'total_crack_regions': s['total'],
        'detected': s['detected'],
        'missed': s['missed'],
        'detection_rate': round(rate, 4),
        'miss_rate': round(miss_rate, 4),
    })

# ── 儲存 CSV ─────────────────────────────────────────────────────────────────
summary_csv = os.path.join(OUT_DIR, 'detection_rate_summary.csv')
with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['model','total_crack_regions','detected',
                                       'missed','detection_rate','miss_rate'])
    w.writeheader()
    w.writerows(summary_rows)

img_csv = os.path.join(OUT_DIR, 'detection_rate_per_image.csv')
if per_image_rows:
    fields = list(per_image_rows[0].keys())
    with open(img_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_image_rows)

print(f'\n（最小連通區域門檻：{MIN_COMPONENT_PX} px）')
print(f'Saved: {summary_csv}')
print(f'Saved: {img_csv}')
