import numpy as np, sys
sys.path.insert(0, 'H:/ChihleeMaster/dev/ABECIS_MODEL_SWAP')
from evaluation.metrics import compute_cldice, compute_metrics_2d

# Perfect match
a = np.zeros((64, 64), bool)
a[32, 10:54] = True
print("perfect clDice:", round(compute_cldice(a, a), 4))

# Broken crack (5px gap) — Dice stays high but clDice drops
b = a.copy()
b[32, 28:33] = False
m = compute_metrics_2d(b, a)
print("broken  Dice:", round(m['dice'], 4), " clDice:", round(m['cldice'], 4))

# Both empty
print("both empty clDice:", compute_cldice(np.zeros((64,64),bool), np.zeros((64,64),bool)))
print("All keys:", list(m.keys()))
