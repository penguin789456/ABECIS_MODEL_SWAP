"""
Local test: verify DDRNet can be loaded from zh320 repo
and instantiated with num_classes=1.
"""
import importlib.util, inspect, sys
from pathlib import Path

zh320_root = Path(__file__).resolve().parent.parent / "realtime-semantic-segmentation-pytorch"
print(f"zh320 root  : {zh320_root}")
print(f"ddrnet.py   : {(zh320_root / 'models' / 'ddrnet.py').exists()}")

# Clear local models from cache
for k in list(sys.modules):
    if k == "models" or k.startswith("models."):
        del sys.modules[k]

# Put zh320 FIRST so its 'models' package takes priority
if str(zh320_root) not in sys.path:
    sys.path.insert(0, str(zh320_root))
importlib.invalidate_caches()

# Now regular import works (relative imports in ddrnet.py need the package context)
from models.ddrnet import DDRNet  # type: ignore[import]
mod_DDRNet = DDRNet

print(f"signature   : {inspect.signature(DDRNet.__init__)}")
print(f"import OK")

# Instantiate with binary output
import torch

# num_class=1 (binary), arch_type default = DDRNet-23-slim (~5.7M params)
model = DDRNet(num_class=1)
model.eval()
dummy = torch.zeros(1, 3, 512, 512)
with torch.no_grad():
    out = model(dummy)

# During eval, output may be a single tensor or tuple depending on use_aux
print(f"raw output type: {type(out)}")
if isinstance(out, (list, tuple)):
    print(f"  [0] shape: {out[0].shape}")
    print(f"  [1] shape: {out[1].shape if len(out) > 1 else 'N/A'}")
    main_out = out[0]
else:
    print(f"  shape: {out.shape}")
    main_out = out

print(f"main output shape: {main_out.shape}  (expected [1, 1, 512, 512])")
print("PASS" if main_out.shape == torch.Size([1, 1, 512, 512]) else f"FAIL: {main_out.shape}")

# Count params
total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"params: {total:.1f}M")
