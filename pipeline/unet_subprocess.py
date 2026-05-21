"""Run U-Net inference in an isolated subprocess so crashes don't kill the pipeline."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _unet_worker(img_path, result_path, checkpoint_path, device, img_size,
                 confidence, min_wall_area, min_door_area, min_window_area):
    """Worker function for subprocess — runs U-Net inference.

    Reads input image from *img_path* (numpy .npy file), writes results
    to *result_path* (pickle file).  If inference crashes, the subprocess
    dies and no result file is written — the parent detects this.
    """
    import pickle
    try:
        img_rgb = np.load(img_path)
        from detection.unet_inference import run_unet_pipeline
        result = run_unet_pipeline(
            image_rgb=img_rgb,
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            confidence=confidence,
            min_wall_area=min_wall_area,
            min_door_area=min_door_area,
            min_window_area=min_window_area,
        )
        if result is not None:
            out = {
                "wall_bboxes": result["wall_bboxes"],
                "door_bboxes": result["door_bboxes"],
                "window_bboxes": result["window_bboxes"],
                "wall_mask": result["wall_mask"],
            }
            with open(result_path, "wb") as f:
                pickle.dump(out, f)
    except Exception as e:
        import pickle
        with open(result_path, "wb") as f:
            pickle.dump({"error": str(e)}, f)


def _run_unet_subprocess(img_rgb, checkpoint_path, device, img_size,
                         confidence, min_wall_area, min_door_area,
                         min_window_area, timeout=120):
    """Run U-Net in a subprocess so crashes don't kill the main process.

    Uses temp files for IPC instead of multiprocessing.Manager (which
    spawns its own subprocess and fails on Windows with rich imports).

    Returns the result dict or None on failure/crash/timeout.
    """
    import multiprocessing as mp
    import pickle
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="unet_")
    img_path = os.path.join(tmp_dir, "input.npy")
    result_path = os.path.join(tmp_dir, "result.pkl")

    try:
        np.save(img_path, img_rgb)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_unet_worker,
            args=(img_path, result_path, checkpoint_path, device, img_size,
                  confidence, min_wall_area, min_door_area, min_window_area),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            logger.warning("U-Net subprocess timed out after %ds, killing", timeout)
            proc.kill()
            proc.join(5)
            return None

        if proc.exitcode != 0:
            logger.warning("U-Net subprocess crashed with exit code %s "
                           "(0x%08X) — falling back to color pipeline",
                           proc.exitcode,
                           proc.exitcode & 0xFFFFFFFF if proc.exitcode < 0
                           else proc.exitcode)
            return None

        if not os.path.exists(result_path):
            logger.warning("U-Net subprocess returned no results")
            return None

        with open(result_path, "rb") as f:
            result = pickle.load(f)

        if "error" in result:
            logger.warning("U-Net subprocess error: %s", result["error"])
            return None

        return result

    except Exception as e:
        logger.warning("U-Net subprocess setup failed: %s", e)
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


