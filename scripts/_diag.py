import json
from pathlib import Path

base = Path("H:/ChihleeMaster/dev/ABECIS_MODEL_SWAP/data/patches")

for split in ("train", "val"):
    meta_path = base / split / "metadata.json"
    if not meta_path.exists():
        print(f"[{split}] no metadata.json")
        continue
    meta = json.loads(meta_path.read_text())
    ratios = list(meta.values())
    n = len(ratios)
    zero = sum(1 for r in ratios if r == 0)
    gt001 = sum(1 for r in ratios if r > 0.001)
    gt005 = sum(1 for r in ratios if r > 0.005)
    gt01  = sum(1 for r in ratios if r > 0.01)
    gt05  = sum(1 for r in ratios if r > 0.05)
    pos   = [r for r in ratios if r > 0]
    mean_all = sum(ratios) / n
    mean_pos = sum(pos) / len(pos) if pos else 0
    print(f"[{split}]")
    print(f"  total={n}, zero={zero} ({100*zero/n:.1f}%)")
    print(f"  crack>0.1%={gt001} ({100*gt001/n:.1f}%)")
    print(f"  crack>0.5%={gt005} ({100*gt005/n:.1f}%)")
    print(f"  crack>1%  ={gt01}  ({100*gt01/n:.1f}%)")
    print(f"  crack>5%  ={gt05}  ({100*gt05/n:.1f}%)")
    print(f"  mean_crack={mean_all*100:.3f}%, pos_mean={mean_pos*100:.3f}%, max={max(ratios)*100:.2f}%")
    sorted_r = sorted(ratios, reverse=True)
    print(f"  top-5 ratios: {[round(x*100,2) for x in sorted_r[:5]]}%")
    print()

# Check train log final epoch
log_candidates = list(Path("H:/ChihleeMaster/dev/ABECIS_MODEL_SWAP/outputs/checkpoints/ppliteseg").rglob("train_log.csv"))
if log_candidates:
    latest = max(log_candidates, key=lambda p: p.stat().st_mtime)
    lines = latest.read_text(encoding="utf-8").strip().splitlines()
    print(f"Train log: {latest}")
    print(f"Header: {lines[0]}")
    print(f"Last 3 epochs:")
    for line in lines[-3:]:
        print(f"  {line}")
