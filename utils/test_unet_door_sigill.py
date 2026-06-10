"""Minimal repro: run UNetDoorDetector inference in-process to isolate SIGILL."""
import sys
import cv2

sys.path.insert(0, ".")

from detection.unet_doors import UNetDoorDetector

img = cv2.imread("input/1.png")
print("image:", None if img is None else img.shape, flush=True)

det = UNetDoorDetector(ckpt_path="weights/epoch_040.pth", img_size=1024)
ok = det.initialize()
print("init:", ok, flush=True)

mask = det._run_inference(img)
print("inference ok, door px:", 0 if mask is None else int((mask > 0).sum()), flush=True)
