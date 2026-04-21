"""
Diagonal wall detection for the ReFloorBRUSNIKA pipeline.

Uses OpenCV's Line Segment Detector (LSD) to find non-axis-aligned line
segments in the floorplan image, filters them against existing axis-aligned
walls to avoid double-counting, and returns them as WallSegment objects with
is_diagonal=True.

Each accepted segment is refined in two steps:
  1. Both endpoints are extended slightly outward along the wall direction.
  2. The extended endpoints are snapped to the face of the nearest nearby
     axis-aligned wall (ray–segment intersection), so diagonal walls connect
     cleanly to their neighbours rather than stopping short.

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

ENDPOINT_EXTEND_PX: float = 8.0        # always push each endpoint this far outward
ENDPOINT_SNAP_RADIUS: float = 30.0     # max distance to search for a wall face to snap to


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
    samples = img[ys, xs].astype(float)
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
    return int(np.sum(mask[ys, xs] > 0)) / n


def _snap_endpoint_to_wall_face(
    px: float, py: float,
    dx: float, dy: float,
    existing_walls: List[WallSegment],
    extend_px: float = ENDPOINT_EXTEND_PX,
    snap_radius: float = ENDPOINT_SNAP_RADIUS,
) -> Tuple[float, float]:
    """
    Push point (px, py) outward along (dx, dy) until it meets a wall face,
    or by *extend_px* if no face is found within *snap_radius*.

    Casts a ray from (px, py) in direction (dx, dy) and finds the nearest
    axis-aligned wall face intersection within snap_radius.  Only faces that
    the ray actually reaches in the positive direction are considered.

    The wall face is modelled as the outer edge of the wall's bbox:
      • horizontal wall (width ≥ height): top edge (y=bbox.y1) and bottom edge (y=bbox.y2)
      • vertical wall (height > width):  left edge (x=bbox.x1) and right edge (x=bbox.x2)

    The endpoint is clamped to the wall's long-axis extent so the snap only
    fires when the diagonal actually reaches that section of the wall.
    """
    best_t = extend_px  # default: always extend by at least this far

    for wall in existing_walls:
        if wall.is_diagonal:
            continue

        bx1, by1 = wall.bbox.x1, wall.bbox.y1
        bx2, by2 = wall.bbox.x2, wall.bbox.y2

        # Skip walls whose centre is too far away to matter
        wcx, wcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        half_diag = math.hypot(bx2 - bx1, by2 - by1) / 2
        if math.hypot(px - wcx, py - wcy) > snap_radius + half_diag:
            continue

        ww = bx2 - bx1
        wh = by2 - by1

        if ww >= wh:
            # Horizontal wall: snap to top (by1) or bottom (by2) face
            if abs(dy) > 1e-6:
                for face_y in (by1, by2):
                    t = (face_y - py) / dy
                    if 0.0 < t <= snap_radius:
                        ix = px + t * dx
                        # Check x is within the wall's horizontal extent (±half-thickness)
                        half_t = max(wall.thickness / 2, 2.0)
                        if bx1 - half_t <= ix <= bx2 + half_t:
                            if t < best_t:
                                best_t = t
        else:
            # Vertical wall: snap to left (bx1) or right (bx2) face
            if abs(dx) > 1e-6:
                for face_x in (bx1, bx2):
                    t = (face_x - px) / dx
                    if 0.0 < t <= snap_radius:
                        iy = py + t * dy
                        half_t = max(wall.thickness / 2, 2.0)
                        if by1 - half_t <= iy <= by2 + half_t:
                            if t < best_t:
                                best_t = t

    return px + best_t * dx, py + best_t * dy


def _refine_endpoints(
    x1: float, y1: float,
    x2: float, y2: float,
    existing_walls: List[WallSegment],
    extend_px: float = ENDPOINT_EXTEND_PX,
    snap_radius: float = ENDPOINT_SNAP_RADIUS,
) -> Tuple[float, float, float, float]:
    """
    Extend both endpoints outward and snap each to the nearest wall face.

    Endpoint 1 is pushed in direction (x1→x2 reversed), endpoint 2 in
    direction (x1→x2 forward).
    """
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1.0:
        return x1, y1, x2, y2

    # Unit direction from 1 → 2
    udx = (x2 - x1) / length
    udy = (y2 - y1) / length

    # Extend endpoint 1 outward: direction is reversed (−udx, −udy)
    nx1, ny1 = _snap_endpoint_to_wall_face(
        x1, y1, -udx, -udy, existing_walls, extend_px, snap_radius
    )
    # Extend endpoint 2 outward: direction is forward (+udx, +udy)
    nx2, ny2 = _snap_endpoint_to_wall_face(
        x2, y2, +udx, +udy, existing_walls, extend_px, snap_radius
    )
    return nx1, ny1, nx2, ny2


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

    After acceptance, each segment's endpoints are extended slightly and
    snapped to the face of any nearby axis-aligned wall so the diagonal wall
    connects cleanly into the surrounding wall geometry.

    Segments that are:
      - too short (< min_length_px)
      - near-horizontal or near-vertical (within axis_tolerance_deg of 0/90°)
      - covering more than 8% of image area (frame borders)
      - heavily overlapping an already-detected axis-aligned wall
      - not on the wall mask / not wall-coloured
    are rejected.

    Args:
        img: Original BGR image (after pre-processing / crop / deskew).
        config: Pipeline configuration (wall colour, pixels_to_cm, etc.).
        existing_walls: Axis-aligned walls already detected in Stage 1.
        wall_mask: Optional binary mask (uint8, 255 = wall pixel).  Primary
            filter when provided; colour distance used as fallback when absent
            and auto_wall_color is False.
        axis_tolerance_deg: Degrees from H/V within which a segment is
            treated as axis-aligned (and rejected here).
        min_length_px: Minimum segment length in pixels.
        max_color_distance: Max Euclidean BGR distance from wall colour (used
            only when wall_mask is None and auto_wall_color is False).
        min_wall_mask_coverage: Minimum fraction of sampled points on the mask.

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
        lsd = cv2.createLineSegmentDetector()

    try:
        raw = lsd.detect(gray)
    except Exception as exc:
        logger.warning("LSD detection failed: %s", exc)
        return []

    if raw is None or raw[0] is None or len(raw[0]) == 0:
        logger.debug("LSD returned no segments")
        return []

    lines_raw = raw[0]    # shape (N, 1, 4): x1,y1,x2,y2
    widths_raw = raw[1]   # shape (N, 1) or None

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
        bx1 = min(x1, x2);  by1 = min(y1, y2)
        bx2 = max(x1, x2);  by2 = max(y1, y2)
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
            mean_bgr = _sample_color_along_line(img, x1, y1, x2, y2)
            if _color_distance(mean_bgr, config.wall_color_bgr) > max_color_distance:
                continue

        # ── f. Thickness ──────────────────────────────────────────────────
        if widths_raw is not None and i < len(widths_raw):
            raw_width = float(widths_raw[i][0])
        else:
            raw_width = THICKNESS_MIN
        thickness = max(THICKNESS_MIN, min(THICKNESS_MAX, raw_width))

        # ── g. Extend endpoints and snap to nearby wall faces ─────────────
        # Push each endpoint slightly outward along the wall direction, then
        # snap to the face of any nearby axis-aligned wall.  This makes the
        # diagonal wall meet its neighbours cleanly rather than stopping short.
        x1r, y1r, x2r, y2r = _refine_endpoints(x1, y1, x2, y2, existing_walls)

        # Recompute AABB after refinement
        bx1 = min(x1r, x2r);  by1 = min(y1r, y2r)
        bx2 = max(x1r, x2r);  by2 = max(y1r, y2r)

        # ── h. Build WallSegment ──────────────────────────────────────────
        seg = WallSegment(
            id=f"diag-{uuid4()}",
            bbox=BBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
            thickness=thickness,
            is_structural=True,
            is_diagonal=True,
            outline=[(int(round(x1r)), int(round(y1r))),
                     (int(round(x2r)), int(round(y2r)))],
        )
        results.append(seg)

    logger.info(
        "detect_diagonal_walls: %d LSD segments → %d diagonal walls accepted",
        len(lines_raw), len(results),
    )
    return results
