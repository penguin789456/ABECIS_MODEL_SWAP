"""
ABECIS LLM 二次篩選：用 Claude Vision 過濾假陽性，提升 Precision
流程：
  1. 載入 ABECIS 預測遮罩 → 找各連通元件
  2. 每個元件裁切原始圖片 + 紅色標示
  3. 送 Claude API 詢問「這是混凝土裂縫嗎？」
  4. 保留 YES 元件，過濾 NO（假陽性）
  5. 重新計算 Precision / IoU / Detection Rate

使用方式：
  set ANTHROPIC_API_KEY=sk-ant-...
  python abecis_llm_filter.py [--test N] [--model haiku|sonnet]
"""
import sys, os, base64, argparse, json, time
sys.stdout.reconfigure(encoding='utf-8')

# 從 .env 讀取 API key（嘗試腳本目錄和當前目錄）
for _env_path in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
    os.path.join(os.getcwd(), '.env'),
    r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\.env',
]:
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    if _k.strip() not in os.environ:
                        os.environ[_k.strip()] = _v.strip()
        break

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage
import anthropic

# ── 路徑設定 ─────────────────────────────────────────────────────────────────
PRED_DIR = r'H:\ChihleeMaster\abecis_predictions'
RGB_DIR  = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\rgb'
GT_DIR   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\concreteCrackSegmentationDataset\BW'
SPLITS   = r'H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP\data\splits\test.txt'
MIN_PX   = 50     # 最小元件像素數
PAD      = 80     # 裁切邊框 padding（px）

# ── 解析參數 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=int, default=0,
                    help='僅處理前 N 張圖（0=全部）')
parser.add_argument('--model', default='haiku',
                    choices=['haiku','sonnet'],
                    help='使用的 Claude 模型（haiku 快且便宜，sonnet 更準）')
parser.add_argument('--save_dir', default='outputs/abecis_filtered',
                    help='儲存過濾後遮罩的目錄')
args = parser.parse_args()

MODEL_MAP = {
    'haiku':  'claude-haiku-4-5',
    'sonnet': 'claude-sonnet-4-5',
}
MODEL_ID = MODEL_MAP[args.model]

api_key = os.environ.get('ANTHROPIC_API_KEY', '')
if not api_key:
    print('ERROR: 請先設定 ANTHROPIC_API_KEY 環境變數')
    print('  Windows CMD:  set ANTHROPIC_API_KEY=sk-ant-...')
    print('  PowerShell:   $env:ANTHROPIC_API_KEY="sk-ant-..."')
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)

# ── 讀取測試集 ID ─────────────────────────────────────────────────────────────
with open(SPLITS) as f:
    test_ids = [l.strip() for l in f if l.strip()]
if args.test > 0:
    test_ids = test_ids[:args.test]

# RGB 索引
rgb_index = {p.stem.lower(): p for p in Path(RGB_DIR).iterdir()
             if p.suffix.lower() in ('.jpg','.jpeg','.png')}

# 輸出目錄
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

print(f'Model: {MODEL_ID}')
print(f'Processing {len(test_ids)} images...\n')

# ── LLM 詢問函數 ──────────────────────────────────────────────────────────────
def ask_is_crack(img_crop_pil: Image.Image, img_id: str, comp_id: int) -> bool:
    """送裁切圖到 Claude Vision，回傳 True=裂縫 / False=假陽性"""
    import io
    buf = io.BytesIO()
    img_crop_pil.save(buf, format='JPEG', quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode()

    prompt = (
        "You are a concrete crack detection expert reviewing automated detection results. "
        "The image shows a cropped region from a concrete surface. "
        "The RED highlighted area is what an automated system flagged as a potential crack. "
        "\n\nCracks in concrete appear as thin, elongated, irregular lines or networks. "
        "False positives are typically round blobs, texture patterns, joints, stains, or shadows "
        "with no linear crack features.\n\n"
        "Question: Is the RED region a crack or part of a crack in the concrete? "
        "Answer YES if there is ANY crack feature visible in or near the red region. "
        "Answer NO only if you are CONFIDENT there is NO crack — it is clearly a false detection. "
        "When in doubt, answer YES. "
        "Reply with exactly ONE word: YES or NO."
    )
    try:
        resp = client.messages.create(
            model=MODEL_ID,
            max_tokens=10,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image',
                     'source': {'type': 'base64',
                                'media_type': 'image/jpeg',
                                'data': b64}},
                    {'type': 'text', 'text': prompt}
                ]
            }]
        )
        answer = resp.content[0].text.strip().upper()
        return answer.startswith('Y')
    except Exception as e:
        print(f'  API error [{img_id} comp{comp_id}]: {e}')
        return True  # 保守：出錯時保留

# ── 主處理迴圈 ────────────────────────────────────────────────────────────────
stats = {
    'total_components': 0,
    'kept': 0,
    'filtered': 0,
    'api_calls': 0,
    'errors': 0,
}
filter_log = []  # [{img_id, comp_id, bbox, is_crack, px_count}]

for idx, img_id in enumerate(test_ids):
    pred_path = Path(PRED_DIR) / f'{img_id}_pred.png'
    rgb_p     = rgb_index.get(img_id.lower())
    if not pred_path.exists() or rgb_p is None:
        continue

    # 載入遮罩與原圖
    pred_mask = np.array(Image.open(pred_path).convert('L')) >= 128
    rgb_img   = Image.open(rgb_p).convert('RGB')
    rgb_arr   = np.array(rgb_img)

    # 確保遮罩尺寸一致
    if pred_mask.shape != (rgb_arr.shape[0], rgb_arr.shape[1]):
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8)*255)
                 .resize((rgb_arr.shape[1], rgb_arr.shape[0]), Image.NEAREST)
        ) >= 128

    H, W = pred_mask.shape

    # 連通元件分析
    labeled, n = ndimage.label(pred_mask)
    if n == 0:
        # 無預測，直接存空遮罩
        Image.fromarray(np.zeros((H, W), np.uint8)).save(save_dir / f'{img_id}_filtered.png')
        continue

    keep_mask = np.zeros((H, W), bool)
    img_kept = img_filtered = 0

    for cid in range(1, n+1):
        comp = (labeled == cid)
        px   = comp.sum()
        if px < MIN_PX:
            continue  # 太小，直接丟棄

        stats['total_components'] += 1

        # 計算 bounding box + padding
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        r0 = max(rows[0]  - PAD, 0)
        r1 = min(rows[-1] + PAD + 1, H)
        c0 = max(cols[0]  - PAD, 0)
        c1 = min(cols[-1] + PAD + 1, W)

        # 裁切並畫紅色高亮
        crop_arr = rgb_arr[r0:r1, c0:c1].copy()
        comp_crop = comp[r0:r1, c0:c1]
        crop_arr[comp_crop] = [255, 50, 50]  # 紅色標示

        crop_pil = Image.fromarray(crop_arr)

        # LLM 判斷
        is_crack = ask_is_crack(crop_pil, img_id, cid)
        stats['api_calls'] += 1

        filter_log.append({
            'img_id': img_id,
            'comp_id': cid,
            'px_count': int(px),
            'bbox': [int(r0), int(r1), int(c0), int(c1)],
            'llm_verdict': 'YES' if is_crack else 'NO',
        })

        if is_crack:
            keep_mask |= comp
            img_kept += 1
            stats['kept'] += 1
        else:
            img_filtered += 1
            stats['filtered'] += 1

    # 儲存過濾後遮罩
    Image.fromarray(keep_mask.astype(np.uint8) * 255).save(
        save_dir / f'{img_id}_filtered.png')

    print(f'[{idx+1:3d}/{len(test_ids)}] {img_id}  '
          f'components={n}  kept={img_kept}  filtered={img_filtered}')

# ── 儲存 filter log ───────────────────────────────────────────────────────────
log_path = save_dir / 'filter_log.json'
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(filter_log, f, ensure_ascii=False, indent=2)
print(f'\nFilter log saved: {log_path}')
print(f'Stats: total_components={stats["total_components"]}  '
      f'kept={stats["kept"]}  filtered={stats["filtered"]}')

# ── 評估過濾後的 Metrics ──────────────────────────────────────────────────────
print('\n' + '='*60)
print('評估過濾後的 ABECIS 指標...')

gt_index = {}
for p in Path(GT_DIR).iterdir():
    if p.suffix.lower() in ('.jpg','.jpeg','.png'):
        gt_index[p.stem.lower()] = p

tp = fp = fn = 0
det_filtered = det_original = 0
total_regions = 0

with open(SPLITS) as f:
    all_test_ids = [l.strip() for l in f if l.strip()]

for img_id in all_test_ids:
    gt_p = gt_index.get(img_id.lower())
    if not gt_p:
        continue

    # GT
    gt_pil = Image.open(gt_p).convert('L')
    rgb_p = rgb_index.get(img_id.lower())
    if rgb_p is None:
        continue
    ref_size = Image.open(rgb_p).size  # (W, H)
    if gt_pil.size != ref_size:
        gt_pil = gt_pil.resize(ref_size, Image.NEAREST)
    gt = np.array(gt_pil) >= 128

    labeled_gt, n_gt = ndimage.label(gt)
    sizes_gt = ndimage.sum(gt, labeled_gt, range(1, n_gt+1))
    valid_gt = [j+1 for j, sz in enumerate(sizes_gt) if sz >= MIN_PX]
    total_regions += len(valid_gt)

    # 原始 ABECIS 預測
    pred_orig_path = Path(PRED_DIR) / f'{img_id}_pred.png'
    if pred_orig_path.exists():
        orig_mask = np.array(Image.open(pred_orig_path).convert('L').resize(
            ref_size, Image.NEAREST)) >= 128
        for cid in valid_gt:
            if np.logical_and(orig_mask, labeled_gt==cid).any():
                det_original += 1

    # 過濾後遮罩
    filt_path = save_dir / f'{img_id}_filtered.png'
    if not filt_path.exists():
        # 未處理的圖（測試模式或無預測），用原始遮罩
        if pred_orig_path.exists():
            filt_mask = np.array(Image.open(pred_orig_path).convert('L').resize(
                ref_size, Image.NEAREST)) >= 128
        else:
            filt_mask = np.zeros_like(gt)
    else:
        filt_mask = np.array(Image.open(filt_path).convert('L').resize(
            ref_size, Image.NEAREST)) >= 128

    for cid in valid_gt:
        if np.logical_and(filt_mask, labeled_gt==cid).any():
            det_filtered += 1

    tp += int(np.logical_and(filt_mask,  gt).sum())
    fp += int(np.logical_and(filt_mask, ~gt).sum())
    fn += int(np.logical_and(~filt_mask, gt).sum())

dr_orig = det_original / total_regions * 100 if total_regions else 0
dr_filt = det_filtered / total_regions * 100 if total_regions else 0
iou_filt  = tp/(tp+fp+fn) if (tp+fp+fn) else 0
prec_filt = tp/(tp+fp)    if (tp+fp)    else 0
rec_filt  = tp/(tp+fn)    if (tp+fn)    else 0

print(f'\nTotal crack regions: {total_regions}')
print(f'\n{"指標":<20} {"ABECIS原始":>15} {"LLM過濾後":>15}')
print('-'*52)
print(f'{"偵測率":<20} {dr_orig:>14.1f}% {dr_filt:>14.1f}%')
print(f'{"IoU":<20} {"0.3086":>15} {iou_filt:>15.4f}')
print(f'{"Precision":<20} {"0.3270":>15} {prec_filt:>15.4f}')
print(f'{"Recall":<20} {"0.8460":>15} {rec_filt:>15.4f}')
print(f'\nComponents filtered out: {stats["filtered"]}/{stats["total_components"]}')
print(f'API calls: {stats["api_calls"]}')
