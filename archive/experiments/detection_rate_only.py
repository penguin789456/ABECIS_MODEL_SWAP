"""
純偵測率計算：只看每條裂縫有沒有被偵測到
不計算像素差值（IoU / Precision / Recall）

評估邏輯：
  GT 黑白圖 → 連通分量分析 → 每個 >=50px 的區域
  預測黑白圖 → 檢查是否有任何像素重疊
  偵測率 = 被碰到的區域數 / 總區域數
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from pathlib import Path
from PIL import Image
import cv2

GT_DIR  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
BASE    = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\outputs\predictions'
MIN_PX  = 50

with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]

# ── GT 連通分量 ───────────────────────────────────────────────────────────────
gt_data = {}
for img_id in test_ids:
    gt_path = next((os.path.join(GT_DIR, f'{img_id}{ext}')
                    for ext in ['.jpg', '.png']
                    if os.path.exists(os.path.join(GT_DIR, f'{img_id}{ext}'))), None)
    if not gt_path:
        continue
    gt = np.array(Image.open(gt_path).convert('L')) >= 128
    n, labeled, stats, _ = cv2.connectedComponentsWithStats(
        gt.astype(np.uint8) * 255, connectivity=8)
    valid = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= MIN_PX]
    if valid:
        gt_data[img_id] = (gt, labeled, valid)

total_regions = sum(len(v) for _, _, v in gt_data.values())
print(f'測試集: {len(test_ids)} 張  |  GT 裂縫區域 (≥{MIN_PX}px): {total_regions} 個\n')

# ── 計算函數 ──────────────────────────────────────────────────────────────────
def calc_dr(pred_dir, label):
    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        print(f'  {label:<42} {"—":>6}  (目錄不存在)')
        return 0
    det = missing = 0
    for img_id, (gt, labeled, valid) in gt_data.items():
        pred_path = pred_dir / f'{img_id}.png'
        if not pred_path.exists():
            missing += 1
            continue
        pred = np.array(Image.open(pred_path).convert('L'))
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST))
        pred_bin = pred >= 128
        for cid in valid:
            if np.logical_and(pred_bin, labeled == cid).any():
                det += 1
    dr  = det / total_regions if total_regions else 0
    tag = '★ 超越論文' if dr > 0.804 else ('△ 接近' if dr > 0.70 else '')
    miss_note = f'  ⚠{missing}張無預測' if missing else ''
    print(f'  {label:<42} {dr:>6.1%}  ({det:>3}/{total_regions}) {tag}{miss_note}')
    return dr

# ── 各模型 ────────────────────────────────────────────────────────────────────
print(f'  {"模型 / 設定":<42} {"偵測率":>6}  {"偵測/總計"}')
print('  ' + '-' * 62)

calc_dr(f'{BASE}\\abecis_detectron2', 'ABECIS Detectron2  (thresh=0.01 全輸出)')
calc_dr(f'{BASE}\\abecis_t080',       'ABECIS Detectron2  (thresh=0.80)')
calc_dr(f'{BASE}\\maskrcnn',          'Mask R-CNN torchvision  (thresh=0.50)')
calc_dr(f'{BASE}\\ppliteseg',         'PP-LiteSeg-T v0  (thresh=0.65)')
calc_dr(f'{BASE}\\ddrnet',            'DDRNet BCEDice  (thresh=0.50)')
calc_dr(f'{BASE}\\deeplabv3_mobilenet','DeepLabV3-MobileNetV3  (thresh=0.45)')

print()
print(f'  {"── 參考基準 ──":<42}')
print(f'  {"ABECIS 論文報告":<42} {"80.4%":>6}  (論文原始測試集，非本研究重測)')
print()
print('說明：')
print('  • 偵測率 = 預測 mask 有接觸到的 GT 裂縫區域數 / 總區域數')
print('  • 只要預測圖在該裂縫區域內有任何一個白像素即算偵測到')
print('  • 不計算像素重疊精度，只看「有沒有找到」')
print('  • DDRNet FTL 訓練中，masks 尚未產生')
