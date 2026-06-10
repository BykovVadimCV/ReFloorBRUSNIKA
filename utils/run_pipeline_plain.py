"""Minimal pipeline driver: no rich console, plain logging. Usage:
    python utils/run_pipeline_plain.py <image> <output_dir> [--enable-unet]
"""
import faulthandler
import logging
import sys

faulthandler.enable()
sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
)

from core.config import PipelineConfig
from pipeline import FloorplanPipeline

image = sys.argv[1]
outdir = sys.argv[2]
enable_unet = "--enable-unet" in sys.argv

cfg = PipelineConfig(debug_mode=True, enable_unet=enable_unet)
pipe = FloorplanPipeline(cfg)
pipe.initialize()
res = pipe.process(image, outdir)
print(
    f"DONE walls={res.n_walls} doors={res.n_doors} windows={res.n_windows} "
    f"rooms={len(res.rooms)} sh3d={res.sh3d_path}",
    flush=True,
)
