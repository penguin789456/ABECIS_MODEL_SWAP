"""Quick launcher to train DeepLabV3+ — avoids conda multiline issue."""
import subprocess, sys, os

os.chdir(r"H:\ChihleeMaster\dev\ABECIS_MODEL_SWAP")

cmd = [
    sys.executable,
    "training/train_crackseg.py",
    "--config", "configs/deeplabv3_mobilenet.yaml",
]

print("Starting DeepLabV3+ training...")
print("CMD:", " ".join(cmd))
result = subprocess.run(cmd, check=False)
sys.exit(result.returncode)
