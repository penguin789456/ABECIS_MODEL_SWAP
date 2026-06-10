"""
Mask R-CNN（ABECIS 基準）偵測率計算
直接讀取已存的 PNG 預測結果，計算連通區域偵測率
不需要重跑模型推論
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from PIL import Image
import cv2

GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
PRED_DIR = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions\maskrcnn'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
MIN_PX   = 50  # 最小裂縫區域像素數

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

print(f'測試集影像數: {len(test_ids)}')
print(f'預測結果目錄: {PRED_DIR}')
print(f'最小區域大小: {MIN_PX} px\n')

total_regions = 0
detected      = 0
tp = fp = fn  = 0
missing_pred  = []

for img_id in test_ids:
    # 讀取 GT mask
    gt_path = next((os.path.join(GT_DIR, f'{img_id}{ext}')
                    for ext in ['.jpg', '.png'] if os.path.exists(os.path.join(GT_DIR, f'{img_id}{ext}'))), None)
    if not gt_path:
        continue

    # 讀取預測 mask
    pred_path = os.path.join(PRED_DIR, f'{img_id}.png')
    if not os.path.exists(pred_path):
        missing_pred.append(img_id)
        continue

    gt_pil   = Image.open(gt_path).convert('L')
    pred_pil = Image.open(pred_path).convert('L')

    # Resize pred to match GT if needed
    if pred_pil.size != gt_pil.size:
        pred_pil = pred_pil.resize(gt_pil.size, Image.NEAREST)

    gt   = np.array(gt_pil)   >= 128
    pred = np.array(pred_pil) >= 128

    # 找 GT 連通分量（cv2 快速實作）
    gt_u8 = gt.astype(np.uint8) * 255
    n, labeled, stats, _ = cv2.connectedComponentsWithStats(gt_u8, connectivity=8)
    # stats[i] = [x, y, w, h, area]；label 0 為背景
    valid_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_PX]

    # 計算各連通區域是否被偵測到
    for cid in valid_ids:
        total_regions += 1
        region_mask = (labeled == cid)
        if np.logical_and(pred, region_mask).any():
            detected += 1

    # 像素級統計
    tp += int(np.logical_and(pred,  gt).sum())
    fp += int(np.logical_and(pred, ~gt).sum())
    fn += int(np.logical_and(~pred,  gt).sum())

# 計算指標
dr   = detected / total_regions if total_regions else 0
iou  = tp / (tp+fp+fn) if (tp+fp+fn) else 0
prec = tp / (tp+fp)    if (tp+fp)    else 0
rec  = tp / (tp+fn)    if (tp+fn)    else 0

print('='*55)
print(f'  Mask R-CNN（torchvision）偵測率分析')
print('='*55)
print(f'  總裂縫區域數  : {total_regions}')
print(f'  成功偵測區域  : {detected}')
print(f'  漏偵測區域    : {total_regions - detected}')
print(f'  偵測率        : {dr:.1%}')
print(f'  漏檢率        : {1-dr:.1%}')
print()
print(f'  像素級指標:')
print(f'  IoU           : {iou:.4f}')
print(f'  Precision     : {prec:.4f}')
print(f'  Recall        : {rec:.4f}')
print('='*55)
print()
print(f'  ABECIS 論文報告: 偵測率=80.4%, IoU=0.3086, Prec=0.327')
print(f'  本研究 MaskRCNN: 偵測率={dr:.1%},  IoU={iou:.4f}, Prec={prec:.4f}')

if missing_pred:
    print(f'\n  ⚠ 缺少預測結果: {missing_pred}')
