"""Wall-thickness-based scale normalization for U-Net inference sizing.

Compact floorplans (few rooms) render walls proportionally thick.  After
crop-to-floorplan + square-pad the drawing fills the canvas, so a wall stroke
can span ~2x the pixel thickness it does on a roomier plan.  The wall U-Net was
trained in a narrower thickness band, so out-of-band strokes segment poorly
(ragged edges → rect_decompose shredding).

This module measures the dominant wall stroke thickness directly and derives a
U-Net input dimension that lands that thickness in a target band.  A text-height
fallback is provided for plans where the ink distance-transform is unreliable
(solid-fill / heavily furnished): across the Brusnika template family wall
thickness ≈ ``wall_to_glyph`` × OCR glyph height, so glyph height is a usable
secondary scale anchor.

All functions are pure (cv2 + numpy only) to keep them importable from both the
detection and pipeline layers without cycles.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Plausible wall-stroke thickness band (px, at native/inference resolution).
# Estimates outside this are treated as unreliable and trigger the fallback.
_PLAUSIBLE_WT = (6.0, 220.0)


def estimate_wall_stroke_thickness_px(
    img_bgr: np.ndarray,
    ink_percentile: float = 85.0,
) -> Optional[float]:
    """Median stroke thickness (px) of the thickest dark ink in a floorplan.

    Otsu-binarise ink, then take a distance transform: each ink pixel's value is
    its distance to the nearest background pixel, i.e. the medial-axis radius of
    the stroke it sits in.  The upper ``ink_percentile`` of those distances
    picks out the ridge of the *thickest* strokes (the structural walls, not
    thin furniture/annotation lines); doubling the median radius gives a full
    thickness.

    Returns ``None`` when the image has no ink (blank) so callers can fall back.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    if np.mean(binary) > 127:  # ensure ink = 255
        binary = cv2.bitwise_not(binary)
    if not np.any(binary):
        return None
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    vals = dt[binary > 0]
    if vals.size == 0:
        return None
    ridge = vals[vals > np.percentile(vals, ink_percentile)]
    if ridge.size == 0:
        return None
    return float(np.median(ridge)) * 2.0


def median_glyph_height_px(ocr_bboxes: Optional[List]) -> Optional[float]:
    """Median height of OCR label boxes, clustered to the dominant band.

    ``ocr_bboxes`` items are ``(text, x1, y1, x2, y2)`` tuples.  Label sizes are
    multi-modal (big area number vs small room numbers vs superscript ``м²``),
    so the median of the dominant ±60% band is used rather than the raw median.
    Returns ``None`` when too few boxes are supplied.
    """
    if not ocr_bboxes:
        return None
    heights = []
    for item in ocr_bboxes:
        if len(item) < 5:
            continue
        h = float(item[4]) - float(item[2])
        if h > 1:
            heights.append(h)
    if len(heights) < 3:
        return None
    arr = np.asarray(heights, dtype=np.float64)
    med = float(np.median(arr))
    band = arr[(arr > med * 0.6) & (arr < med * 1.6)]
    return float(np.median(band)) if band.size else med


def unet_size_for_wall_thickness(
    img_bgr: np.ndarray,
    target_wall_px: float = 30.0,
    *,
    ocr_bboxes: Optional[List] = None,
    wall_to_glyph: float = 1.8,
    size_min: int = 512,
    size_max: Optional[int] = None,
    multiple: int = 32,
) -> Tuple[int, Optional[float], str]:
    """Choose a square U-Net input dimension that normalizes wall thickness.

    The U-Net transform is ``Resize(size, size)``, so a native wall of thickness
    ``wt`` (measured on an ``H×W`` image) appears in the tensor at roughly
    ``wt * size / ((H+W)/2)``.  Solving for the target gives the returned size.

    ``size_max`` defaults to the current native size ``(H+W)//2`` so this only
    ever *thins* over-thick walls; it never upscales thin-line plans (a separate
    concern).  The size is snapped to a multiple of ``multiple`` for the encoder
    and clamped to ``[size_min, size_max]``.

    Returns ``(size, measured_wt, source)`` where ``source`` is one of
    ``"distance-transform"``, ``"ocr-glyph"`` or ``"native-fallback"``.
    """
    h, w = img_bgr.shape[:2]
    native_avg = (h + w) / 2.0
    native_size = (h + w) // 2
    if size_max is None:
        size_max = native_size

    wt = estimate_wall_stroke_thickness_px(img_bgr)
    source = "distance-transform"
    if wt is None or not (_PLAUSIBLE_WT[0] <= wt <= _PLAUSIBLE_WT[1]):
        gh = median_glyph_height_px(ocr_bboxes)
        if gh is not None:
            wt = wall_to_glyph * gh
            source = "ocr-glyph"
        else:
            return native_size, wt, "native-fallback"

    size = target_wall_px * native_avg / wt
    size = int(round(size / multiple) * multiple)
    size = max(size_min, min(size, size_max))
    return size, wt, source
