"""Minimal torch CPU conv repro for the SIGILL/0xC000001D crash."""
import os
import sys

print("ATEN_CPU_CAPABILITY =", os.environ.get("ATEN_CPU_CAPABILITY"), flush=True)

import torch
print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available(), flush=True)

torch.set_num_threads(int(os.environ.get("TEST_THREADS", "4")))

# Mimic the U-Net first layers: conv on a ~1000px image
x = torch.randn(1, 3, 1000, 1000)
conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
with torch.no_grad():
    for i in range(3):
        y = conv(x)
        print(f"conv pass {i}: out {tuple(y.shape)} sum={float(y.sum()):.1f}", flush=True)
print("OK", flush=True)
