"""
Diagonal wall detection for the ReFloorBRUSNIKA pipeline.

Uses OpenCV's Line Segment Detector (LSD) to find non-axis-aligned line
segments in the floorplan image, filters them against existing axis-aligned
walls to avoid double-counting, and returns them as WallSegment objects with
is_diagonal=True.

Only called by pipeline.py Stage 1c when the solid-colour wall path is active
(used_unet_grey=False); LSD is unreliable on synthetic grey-fill images.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from core.config import PipelineConfig
from core.models import BBox, WallSegment

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────

AXIS_TOLERANCE_DEG: float = 20.0       # reject if within this many ° of H or V
MIN_LENGTH_PX: float = 40.0            # reject short segments
MAX_COLOR_DISTANCE: float = 55.0       # max Euclidean RGB distance from wall colour
MIN_WALL_MASK_COVERAGE: float = 0.35   # min fraction of sampled pts on wall mask
MAX_AREA_FRACTION: float = 0.08        # reject if AABB > 8% of image → frame borders
BBOX_OVERLAP_REJECT: float = 0.60      # reject if AABB overlaps existing wall by ≥60%
THICKNESS_MIN: float = 3.0             # clamp LSD width (pixels)
THICKNESS_MAX: float = 60.0


# ── Private helpers ───────────────────────────────────────────────────────────

def _sample_color_along_line(
    img: np.ndarray,
    x1: float, y1: float,
    x2: float, y2: float,
    n: int = 15,
) -> Tuple[float, float, float]:
    """Return mean BGR at *n* evenly-spaced points along the line."""
    t = np.linspace(0.0, 1.0, n)
    xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, img.shape[1] - 1)
    ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, img.shape[0] - 1)
    samples = img[ys, xs].astype(float)   # (n, 3)
    mean = samples.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _color_distance(a: Tuple[float, float, float],
                    b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two BGR triples."""
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _overlaps_existing(
    bbox: Tuple[float, float, float, float],
    walls: List[WallSegment],
    threshold: float = BBOX_OVERLAP_REJECT,
) -> bool:
    """
    Return True if the detected line's AABB overlaps any existing wall's
    AABB by at least *threshold* fraction of the detected line's bbox area.
    """
    ax1, ay1, ax2, ay2 = bbox
    area_a = (ax2 - ax1) * (ay2 - ay1)
    if area_a <= 0:
        return False
    for w in walls:
        bx1, by1, bx2, by2 = w.bbox.x1, w.bbox.y1, w.bbox.x2, w.bbox.y2
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter / area_a >= threshold:
            return True
    return False


def _sample_mask_along_line(
    mask: np.ndarray,
    x1: float, y1: float,
    x2: float, y2: float,
    n: int = 20,
) -> float:
    """Return fraction of sampled points that land on mask pixels (> 0)."""
    t = np.linspace(0.0, 1.0, n)
    xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, mask.shape[0] - 1)
    hits = int(np.sum(mask[ys, xs] > 0))
    return hits / n


# ── Public API ────────────────────────────────────────────────────────────────

def detect_diagonal_walls(
    img: np.ndarray,
    config: PipelineConfig,
    existing_walls: List[WallSegment],
    wall_mask: Optional[np.ndarray] = None,
    axis_tolerance_deg: float = AXIS_TOLERANCE_DEG,
    min_length_px: float = MIN_LENGTH_PX,
    max_color_distance: float = MAX_COLOR_DISTANCE,
    min_wall_mask_coverage: float = MIN_WALL_MASK_COVERAGE,
) -> List[WallSegment]:
    """
    Detect non-axis-aligned wall segments using OpenCV's LSD.

    Segments that are:
      - too short (< min_length_px)
      - near-horizontal or near-vertical (within axis_tolerance_deg of 0/90°)
      - covering more than 8% of image area (frame borders)
      - heavily overlapping an already-detected axis-aligned wall
      - not on the wall mask / not wall-coloured
    are rejected.

    Args:
        img: Original BGR image (after pre-processing / crop / deskew).
        config: Pipeline configuration (supplies wall colour and pixels_to_cm).
        existing_walls: Axis-aligned walls already detected in Stage 1.
        wall_mask: Optional binary mask (uint8, 255 = wall pixel).  When
            provided, this is the primary filter; colour distance is used as
            fallback when the mask is absent and auto_wall_color is False.
        axis_tolerance_deg: Degrees from horizontal/vertical within which a
            segment is considered axis-aligned (and rejected here).
        min_length_px: Minimum line length in pixels.
        max_color_distance: Maximum Euclidean BGR distance from the configured
            wall colour (used only when wall_mask is None and auto_wall_color
            is False).
        min_wall_mask_coverage: Minimum fraction of sampled points along the
            line that must fall on wall-mask pixels.

    Returns:
        List of WallSegment objects with is_diagonal=True.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape[:2]
    image_area = float(H * W)

    # ── Build LSD detector ────────────────────────────────────────────────────
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except AttributeError:
        # Older OpenCV builds without REFINE_STD constant
        lsd = cv2.createLineSegmentDetector()

    try:
        raw = lsd.detect(gray)
    except Exception as exc:
        logger.warning("LSD detection failed: %s", exc)
        return []

    if raw is None or raw[0] is None or len(raw[0]) == 0:
        logger.debug("LSD returned no segments")
        return []

    lines_raw = raw[0]       # shape (N, 1, 4): x1,y1,x2,y2
    widths_raw = raw[1]      # shape (N, 1)   or None

    results: List[WallSegment] = []

    for i, line in enumerate(lines_raw):
        x1, y1, x2, y2 = line[0]

        # ── a. Length filter ──────────────────────────────────────────────
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length_px:
            continue

        # ── b. Axis-alignment filter ──────────────────────────────────────
        # angle in [0, 180); deviation from nearest axis (H=0° or V=90°)
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        deviation = min(angle_deg % 90.0, 90.0 - angle_deg % 90.0)
        if deviation < axis_tolerance_deg:
            continue

        # ── c. Area / border filter ───────────────────────────────────────
        bx1 = min(x1, x2)
        by1 = min(y1, y2)
        bx2 = max(x1, x2)
        by2 = max(y1, y2)
        bbox_area = (bx2 - bx1) * (by2 - by1)
        if bbox_area > MAX_AREA_FRACTION * image_area:
            continue

        # ── d. Overlap with existing axis-aligned walls ───────────────────
        if _overlaps_existing((bx1, by1, bx2, by2), existing_walls):
            continue

        # ── e. Wall-content filter (mask or colour) ───────────────────────
        if wall_mask is not None:
            coverage = _sample_mask_along_line(wall_mask, x1, y1, x2, y2)
            if coverage < min_wall_mask_coverage:
                continue
        elif not config.auto_wall_color:
            # Only use colour filter when the user has pinned a specific colour
            mean_bgr = _sample_color_along_line(img, x1, y1, x2, y2)
            wall_bgr = config.wall_color_bgr  # (B, G, R) tuple
            if _color_distance(mean_bgr, wall_bgr) > max_color_distance:
                continue

        # ── f–g. Thickness ────────────────────────────────────────────────
        if widths_raw is not None and i < len(widths_raw):
            raw_width = float(widths_raw[i][0])
        else:
            raw_width = THICKNESS_MIN
        thickness = max(THICKNESS_MIN, min(THICKNESS_MAX, raw_width))

        # ── h. Build WallSegment ──────────────────────────────────────────
        seg = WallSegment(
            id=f"diag-{uuid4()}",
            bbox=BBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
            thickness=thickness,
            is_structural=True,
            is_diagonal=True,
            outline=[(int(x1), int(y1)), (int(x2), int(y2))],
        )
        results.append(seg)

    logger.info(
        "detect_diagonal_walls: %d LSD segments → %d diagonal walls accepted",
        len(lines_raw), len(results),
    )
    return results
