"""Run the real diagonal-preprocessor U-Net standalone to reproduce the crash."""
import faulthandler
import os
import sys

faulthandler.enable()
sys.path.insert(0, ".")

import cv2

ckpt = sys.argv[1] if len(sys.argv) > 1 else "weights/epoch_040.pth"
device = sys.argv[2] if len(sys.argv) > 2 else None

from detection.unet_inference import run_unet_pipeline

img = cv2.imread("input/1.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("image:", rgb.shape, "ckpt:", ckpt, "device:", device, flush=True)

res = run_unet_pipeline(rgb, ckpt, device=device)
if res is None:
    print("run_unet_pipeline returned None", flush=True)
else:
    wm = res.get("wall_mask")
    print("OK wall px:", 0 if wm is None else int((wm > 0).sum()), flush=True)
