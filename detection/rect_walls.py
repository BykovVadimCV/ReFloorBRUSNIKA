"""
detection/rect_walls.py — rectangle-decomposition wall detector.

The new wall-detection pipeline:

    Deskewed BGR image
        │
        ▼
    Binary U-Net (epoch_20.pth) → wall_mask
        │
        ▼
    LSD on wall_mask → diagonal-presence test
        │
        ▼
    rect_decompose.decompose()
        │   • initial_bleed_weight=10, bleed_weight=1.5
        │   • coverage_stop=0.93, max_overlap=0.5
        │   • axis_only=True when no diagonals detected
        │   • axis_gap snaps near-axis features to 0/45/90/135
        ▼
    Snap pass: extend collinear rectangle endpoints to meet
        │
        ▼
    Rect → WallSegment  (shorter dim = thickness, longer dim = length)

The output is a list of `WallSegment` instances compatible with the rest of
the pipeline. Diagonal walls (angle not within ±diag_axis_tol of 0°/90°)
have ``is_diagonal=True`` and carry their two centerline endpoints in
``outline``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.config import PipelineConfig
from core.models import BBox, DetectionResult, WallSegment

import rect_decompose as rd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable defaults — exposed via PipelineConfig
# ---------------------------------------------------------------------------

DEFAULT_AXIS_TOL_DEG    = 3.0      # walls within ±this of 0/90 are axis-aligned
DEFAULT_DIAG_LSD_RATIO  = 0.05     # if ≥5% of LSD line-length is diagonal,
                                   # allow diagonal rectangles
DEFAULT_DIAG_LSD_MIN_LEN = 25.0    # min LSD segment length to count
DEFAULT_SNAP_GAP_FACTOR  = 2.0     # extend if gap ≤ factor × max_thickness
DEFAULT_SNAP_GAP_FLOOR   = 6.0     # but never below this many pixels
DEFAULT_SNAP_ANGLE_DEG   = 5.0     # rectangles must agree within this angle


# ---------------------------------------------------------------------------
# Centerline geometry
# ---------------------------------------------------------------------------

def rect_centerline(
    rect: rd.Rect,
) -> Tuple[Tuple[float, float], Tuple[float, float], float, float]:
    """
    Compute the centerline endpoints of a rotated rectangle.

    The longer side is the wall's length; the shorter side is its thickness.
    Returns ``(p1, p2, thickness, wall_angle_deg)``.

    ``wall_angle_deg`` is the orientation of the centerline in image space,
    NOT the rectangle's own ``angle`` field — the two differ by 90° when
    ``h > w``.
    """
    rad = math.radians(rect.angle)
    cos, sin = math.cos(rad), math.sin(rad)

    if rect.w >= rect.h:
        # Length runs along the rect's local +x axis
        thickness = float(rect.h)
        half_len  = rect.w / 2.0
        dx, dy    = half_len * cos, half_len * sin
        wall_angle = rect.angle
    else:
        # Length runs along local +y, which is +x rotated by 90°
        thickness = float(rect.w)
        half_len  = rect.h / 2.0
        dx, dy    = -half_len * sin, half_len * cos
        wall_angle = rect.angle + 90.0

    p1 = (rect.cx - dx, rect.cy - dy)
    p2 = (rect.cx + dx, rect.cy + dy)
    return p1, p2, thickness, wall_angle % 180.0


# ---------------------------------------------------------------------------
# Diagonal-presence test (LSD over wall mask)
# ---------------------------------------------------------------------------

def detect_has_diagonals(
    wall_mask: np.ndarray,
    *,
    axis_tol_deg: float    = DEFAULT_AXIS_TOL_DEG,
    diag_ratio_thr: float  = DEFAULT_DIAG_LSD_RATIO,
    min_segment_len: float = DEFAULT_DIAG_LSD_MIN_LEN,
) -> bool:
    """
    Return True when the wall mask contains a non-trivial amount of
    diagonal line segments (defined as: angle deviates from the nearest
    cardinal axis by more than ``axis_tol_deg``).

    The diagonal "amount" is measured as a length-weighted ratio:

        sum(len of off-axis segments) / sum(len of all segments) ≥ diag_ratio_thr

    This is 0 when the floorplan is purely orthogonal — common for
    Brusnika layouts — and lets the decomposer skip the diagonal search.
    """
    if wall_mask is None or wall_mask.size == 0:
        return False

    src = (wall_mask > 0).astype(np.uint8) * 255

    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except Exception:
        try:
            lsd = cv2.createLineSegmentDetector()
        except Exception as exc:
            logger.warning("LSD unavailable: %s — assuming no diagonals", exc)
            return False

    try:
        raw = lsd.detect(src)
    except Exception as exc:
        logger.warning("LSD detect failed: %s — assuming no diagonals", exc)
        return False

    if raw is None or raw[0] is None or len(raw[0]) == 0:
        return False

    total_len = 0.0
    diag_len  = 0.0
    for seg in raw[0]:
        x1, y1, x2, y2 = seg[0]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < min_segment_len:
            continue
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        dev = min(ang % 90.0, 90.0 - (ang % 90.0))
        total_len += L
        if dev > axis_tol_deg:
            diag_len += L

    if total_len <= 0.0:
        return False

    ratio = diag_len / total_len
    logger.info(
        "LSD diagonal ratio: %.1f%% (threshold %.1f%%) → %s",
        ratio * 100.0, diag_ratio_thr * 100.0,
        "diagonals present" if ratio >= diag_ratio_thr else "axis-only",
    )
    return ratio >= diag_ratio_thr


# ---------------------------------------------------------------------------
# Centerline-collinearity snapping
# ---------------------------------------------------------------------------

@dataclass
class _RectInfo:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    thickness: float
    angle_deg: float          # 0..180
    rect: rd.Rect             # original rect kept for downstream use


def _line_intersection(
    p1a: Tuple[float, float], p2a: Tuple[float, float],
    p1b: Tuple[float, float], p2b: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersection of line(p1a,p2a) and line(p1b,p2b). None if parallel."""
    x1, y1 = p1a; x2, y2 = p2a
    x3, y3 = p1b; x4, y4 = p2b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _angle_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def snap_collinear_endpoints(
    infos: List[_RectInfo],
    *,
    snap_gap_factor: float = DEFAULT_SNAP_GAP_FACTOR,
    snap_gap_floor:  float = DEFAULT_SNAP_GAP_FLOOR,
    snap_angle_deg:  float = DEFAULT_SNAP_ANGLE_DEG,
) -> None:
    """
    For every pair of rectangles whose centerline angles agree to within
    ``snap_angle_deg`` and whose nearest endpoints are within a small gap,
    move both endpoints to their lines' intersection.  Modifies ``infos``
    in place — widths are NOT changed, only endpoints are shifted.

    Handles:
        - Parallel walls that don't quite touch  →  meet at common point
        - T-junctions where one wall's tip lands on the other's centerline
        - Corners where two walls meet at ~90°
    """
    n = len(infos)
    for i in range(n):
        a = infos[i]
        gap_a = max(snap_gap_floor, snap_gap_factor * a.thickness)

        for j in range(n):
            if i == j:
                continue
            b = infos[j]
            gap_b = max(snap_gap_floor, snap_gap_factor * b.thickness)
            gap_thr = max(gap_a, gap_b)

            # Find the closest endpoint pair (a_endpoint, b_endpoint)
            pairs = [
                ('p1', a.p1, 'p1', b.p1),
                ('p1', a.p1, 'p2', b.p2),
                ('p2', a.p2, 'p1', b.p1),
                ('p2', a.p2, 'p2', b.p2),
            ]
            ka, pa, kb, pb = min(pairs, key=lambda t: _dist(t[1], t[3]))
            d = _dist(pa, pb)
            if d > gap_thr:
                continue

            adiff = _angle_dist(a.angle_deg, b.angle_deg)

            if adiff <= snap_angle_deg:
                # Nearly parallel — snap both to the midpoint of pa,pb
                meet = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
            else:
                # Different angles — find true line intersection
                meet = _line_intersection(a.p1, a.p2, b.p1, b.p2)
                if meet is None:
                    continue
                # Sanity: the intersection must be near both endpoints,
                # otherwise we'd snap to a point far from where the walls
                # actually live.
                if _dist(meet, pa) > gap_thr * 2.5 or _dist(meet, pb) > gap_thr * 2.5:
                    continue

            # Apply the shift
            if ka == 'p1':
                a.p1 = meet
            else:
                a.p2 = meet
            if kb == 'p1':
                b.p1 = meet
            else:
                b.p2 = meet


# ---------------------------------------------------------------------------
# Rectangle → WallSegment conversion
# ---------------------------------------------------------------------------

def _rect_to_wall_segment(info: _RectInfo, idx: int,
                           axis_tol_deg: float = DEFAULT_AXIS_TOL_DEG) -> WallSegment:
    """Build a WallSegment from snapped centerline endpoints."""
    p1, p2 = info.p1, info.p2

    # Axis-aligned classification: distance to nearest cardinal axis
    dev = min(info.angle_deg % 90.0, 90.0 - (info.angle_deg % 90.0))
    is_diagonal = dev > axis_tol_deg

    # AABB enclosing the centerline (used by downstream visualisation/IoU code)
    x1 = min(p1[0], p2[0]); x2 = max(p1[0], p2[0])
    y1 = min(p1[1], p2[1]); y2 = max(p1[1], p2[1])

    # Pad the AABB by half-thickness on the short side so it has non-zero
    # width when the wall is purely axis-aligned (otherwise BBox.area = 0).
    half = info.thickness / 2.0
    if x2 - x1 < half * 2:
        x1 -= half; x2 += half
    if y2 - y1 < half * 2:
        y1 -= half; y2 += half

    bbox = BBox(float(x1), float(y1), float(x2), float(y2))
    return WallSegment(
        id=f"wall-{idx:04d}",
        bbox=bbox,
        thickness=float(info.thickness),
        is_structural=False,
        is_outer=False,
        is_diagonal=bool(is_diagonal),
        outline=[(int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1])))],
    )


# ---------------------------------------------------------------------------
# Rasterisation helper for downstream room-finding
# ---------------------------------------------------------------------------

def rasterise_wall_segments(
    walls: List[WallSegment], shape: Tuple[int, int],
) -> np.ndarray:
    """
    Stamp every wall segment as a filled rotated rectangle on a uint8 mask
    (255 = wall).  Each segment's four corners are computed from the
    centerline endpoints and thickness so the output has sharp right-angle
    corners — no rounded line-cap artefacts.
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for w in walls:
        if not w.outline or len(w.outline) < 2:
            continue
        (x1, y1), (x2, y2) = w.outline[0], w.outline[1]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 0.5:
            continue
        half = max(0.5, w.thickness / 2.0)
        # Unit normal perpendicular to the wall direction
        nx, ny = -dy / L * half, dx / L * half
        corners = np.array([
            [x1 + nx, y1 + ny],
            [x2 + nx, y2 + ny],
            [x2 - nx, y2 - ny],
            [x1 - nx, y1 - ny],
        ], dtype=np.int32)
        cv2.fillConvexPoly(out, corners, 255)
    return out


# ---------------------------------------------------------------------------
# Enclosed-space filtering
# ---------------------------------------------------------------------------

def find_enclosed_spaces(
    img_bgr: np.ndarray,
    wall_mask: Optional[np.ndarray] = None,
    *,
    close_kernel_size: int = 7,
    close_iters: int = 2,
) -> np.ndarray:
    """
    Identify white spaces in the image that are fully enclosed by walls.

    Algorithm
    ---------
    1. Grayscale → invert (walls/text become bright, white background dark).
    2. Otsu-binarise → binary mask (walls = 255, background = 0).
    3. Morphological *close* with a ``close_kernel_size`` kernel to bridge
       small gaps in the wall structure (the "cap" step).
    4. OR with a dilated copy of ``wall_mask`` to reinforce boundaries.
    5. Flood-fill from all border pixels → marks exterior background as 128.
    6. Connected components on the remaining 0-pixels → enclosed regions.

    Returns
    -------
    labels : ndarray (H, W), int32
        Each enclosed region has a unique positive label ≥ 1.
        Exterior / wall pixels have label 0.
    """
    H, W = img_bgr.shape[:2]

    # 1. Invert grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray

    # 2. Otsu binarise
    _, binary = cv2.threshold(
        inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. Close to bridge small wall gaps
    k = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=close_iters)

    # 4. Reinforce with dilated wall_mask
    if wall_mask is not None and wall_mask.size > 0:
        wm = (wall_mask > 0).astype(np.uint8) * 255
        dil_k = np.ones((3, 3), dtype=np.uint8)
        wm_dil = cv2.dilate(wm, dil_k, iterations=2)
        closed = cv2.bitwise_or(closed, wm_dil)

    # 5. Flood-fill from all border background pixels → exterior = 128
    filled = closed.copy()
    flood_seed = np.zeros((H + 2, W + 2), dtype=np.uint8)   # required by cv2

    def _try_fill(x: int, y: int) -> None:
        if 0 <= y < H and 0 <= x < W and filled[y, x] == 0:
            cv2.floodFill(filled, flood_seed, (x, y), 128)

    for x in range(W):
        _try_fill(x, 0)
        _try_fill(x, H - 1)
    for y in range(H):
        _try_fill(0, y)
        _try_fill(W - 1, y)

    # 6. Remaining 0-pixels are enclosed → connected-component label them
    enclosed_bin = (filled == 0).astype(np.uint8)
    _, labels = cv2.connectedComponents(enclosed_bin, connectivity=8)
    return labels.astype(np.int32)


def refine_mask_by_enclosed_spaces(
    wall_mask: np.ndarray,
    img_bgr: np.ndarray,
    ocr_bboxes: Optional[List] = None,
    *,
    wall_overlap_threshold: float = 0.50,
    close_kernel_size: int = 7,
    close_iters: int = 2,
    debug_img_path: Optional[str] = None,
) -> np.ndarray:
    """
    Refine *wall_mask* using enclosed-space analysis.

    For every enclosed region (a white space surrounded by walls):

    * **No OCR label inside + ≥ wall_overlap_threshold wall-mask overlap**
      → structural cavity; fill the whole region as wall (255).
    * **OCR label centre inside**
      → labelled room; clear any stray wall pixels (set to 0).

    Parameters
    ----------
    wall_mask  : H×W uint8, 255 = wall
    img_bgr    : BGR image (text regions already erased)
    ocr_bboxes : list of ``(x1, y1, x2, y2)`` or ``(text, x1, y1, x2, y2)``
    wall_overlap_threshold : fraction of region pixels that must be wall
                             for the "fill as wall" branch to fire

    Returns
    -------
    Refined wall_mask (a copy — the input is not modified).
    """
    if wall_mask is None:
        return wall_mask

    refined = wall_mask.copy()
    H, W = refined.shape[:2]

    # Normalise ocr_bboxes to plain (x1, y1, x2, y2) int tuples
    boxes: List[Tuple[int, int, int, int]] = []
    if ocr_bboxes:
        for item in ocr_bboxes:
            if len(item) == 4:
                boxes.append((int(item[0]), int(item[1]),
                               int(item[2]), int(item[3])))
            elif len(item) >= 5:
                # (text, x1, y1, x2, y2) — skip text token
                boxes.append((int(item[1]), int(item[2]),
                               int(item[3]), int(item[4])))

    # Precompute OCR bbox centres once
    ocr_centres: List[Tuple[float, float]] = []
    for (bx1, by1, bx2, by2) in boxes:
        ocr_centres.append(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0))

    # Get enclosed region labels
    labels = find_enclosed_spaces(
        img_bgr, wall_mask,
        close_kernel_size=close_kernel_size,
        close_iters=close_iters,
    )

    n_labels = int(labels.max())
    if n_labels == 0:
        logger.debug("refine_mask_by_enclosed_spaces: no enclosed regions found")
        return refined

    filled_count = 0
    cleared_count = 0

    for region_id in range(1, n_labels + 1):
        region_mask = labels == region_id
        region_area = int(np.count_nonzero(region_mask))
        if region_area == 0:
            continue

        # Check if any OCR bbox centre falls inside this region
        has_ocr = False
        for (cx, cy) in ocr_centres:
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= yi < H and 0 <= xi < W and region_mask[yi, xi]:
                has_ocr = True
                break

        if has_ocr:
            # Labelled room — clear stray wall pixels
            refined[region_mask] = 0
            cleared_count += 1
        else:
            # No room label — check wall-mask overlap
            overlap = int(np.count_nonzero(wall_mask[region_mask] > 0))
            overlap_frac = overlap / region_area
            if overlap_frac >= wall_overlap_threshold:
                refined[region_mask] = 255
                filled_count += 1

    logger.info(
        "refine_mask_by_enclosed_spaces: %d structural regions filled, "
        "%d room regions cleared  (total enclosed=%d)",
        filled_count, cleared_count, n_labels,
    )

    # ── Optional debug image ───────────────────────────────────────────────
    # Shows every enclosed white region found by the flood-fill step,
    # each painted with a distinct hue so the user can verify detection.
    # A thin white border is drawn around each region, and a small label
    # prints the region's pixel area.  OCR bounding boxes are shown as
    # dashed white rectangles.
    if debug_img_path:
        try:
            # Faded greyscale base so the coloured regions stand out
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Dim slightly so overlays are vivid against it
            dbg = (base * 0.45).astype(np.uint8)

            alpha = 0.75   # overlay opacity

            for region_id in range(1, n_labels + 1):
                rmask = (labels == region_id)
                region_px = int(np.count_nonzero(rmask))
                if region_px == 0:
                    continue

                # Assign a unique, evenly-spaced hue (HSV → BGR)
                hue = int((region_id * 37) % 180)       # cycles through hue wheel
                hsv_pixel = np.array([[[hue, 230, 240]]], dtype=np.uint8)
                bgr_color = tuple(
                    int(v) for v in cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
                )

                # Flood-fill the region with its colour
                overlay = np.zeros_like(dbg)
                overlay[rmask] = bgr_color
                dbg = cv2.addWeighted(dbg, 1.0 - alpha, overlay, alpha, 0)

                # Draw a 1-px white contour around the region border
                region_u8 = rmask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(dbg, contours, -1, (255, 255, 255), 1)

                # Label the centroid with the pixel area
                ys, xs = np.where(rmask)
                cx_r, cy_r = int(xs.mean()), int(ys.mean())
                label_txt = str(region_px)
                (tw, th), _ = cv2.getTextSize(
                    label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )
                # Dark backing rectangle for legibility
                cv2.rectangle(
                    dbg,
                    (cx_r - tw // 2 - 1, cy_r - th - 1),
                    (cx_r + tw // 2 + 1, cy_r + 1),
                    (0, 0, 0), -1,
                )
                cv2.putText(
                    dbg, label_txt,
                    (cx_r - tw // 2, cy_r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )

            # OCR bounding boxes — white outlines so we can see which
            # regions have been identified as labelled rooms
            for (bx1, by1, bx2, by2) in boxes:
                cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

            cv2.imwrite(debug_img_path, dbg)
            logger.info(
                "Enclosed-space debug image saved → %s  (%d regions)",
                debug_img_path, n_labels,
            )
        except Exception as _dbe:
            logger.warning("Could not save enclosed-space debug image: %s", _dbe)

    return refined


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class RectWallDetector:
    """
    Wall detector that produces axis-and-diagonal rectangles via U-Net
    inference + greedy rectangle decomposition.

    Use::

        det = RectWallDetector(config)
        det.initialize()                  # loads U-Net once
        result = det.detect(img_bgr)      # → DetectionResult
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._unet_loaded = False

    # ------------------------------------------------------------------
    # Init / model loading is delegated to the existing _run_unet helper.
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        # Loading happens lazily inside _run_unet on the first call.
        self._unet_loaded = True

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(
        self,
        img_bgr: np.ndarray,
        *,
        wall_mask: Optional[np.ndarray] = None,
        ocr_bboxes: Optional[List] = None,
        debug_img_path: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect walls in ``img_bgr``.  If ``wall_mask`` is supplied it is used
        directly; otherwise the binary U-Net is invoked.

        Parameters
        ----------
        img_bgr    : BGR image (text regions ideally already erased).
        wall_mask  : optional pre-computed binary wall mask; if None the
                     U-Net checkpoint is used.
        ocr_bboxes : list of ``(x1, y1, x2, y2)`` or ``(text, x1, y1, x2, y2)``
                     bounding boxes for OCR text regions.  Used by the
                     enclosed-space refinement step to distinguish rooms
                     (labelled) from structural cavities (unlabelled).
        """
        cfg = self.config
        H, W = img_bgr.shape[:2]

        # ── 1. Raw wall mask (U-Net) ───────────────────────────────────
        if wall_mask is None:
            wall_mask = self._run_binary_unet(img_bgr)

        if wall_mask is None or int(np.count_nonzero(wall_mask)) == 0:
            logger.warning("RectWallDetector: empty wall mask — no walls returned")
            return DetectionResult(
                walls=[], openings=[], wall_mask=wall_mask,
                raw_wall_mask=wall_mask,
                outline_mask=None, source="rect_walls",
            )

        # Preserve the raw U-Net output for diagnostics / downstream code
        # that may want to compare against post-processed masks.
        raw_wall_mask = wall_mask.copy()

        # ── 1a. Cap step — morphological close on the wall mask ────────
        # Bridges small gaps in U-Net wall coverage so rect_decompose sees
        # connected wall material.  This is the "cap" preprocessing step
        # applied directly to wall_mask (not just to the enclosed-space
        # analysis mask).
        cap_k = max(1, int(getattr(cfg, "rect_cap_kernel_size", 5)))
        cap_i = max(1, int(getattr(cfg, "rect_cap_iters", 2)))
        if cap_k > 1 and cap_i > 0:
            kernel = np.ones((cap_k, cap_k), dtype=np.uint8)
            wall_mask = cv2.morphologyEx(
                wall_mask, cv2.MORPH_CLOSE, kernel, iterations=cap_i,
            )
            logger.info(
                "Cap step (MORPH_CLOSE k=%d iters=%d): "
                "%.1f%% wall pixels (was %.1f%% raw)",
                cap_k, cap_i,
                100.0 * int(np.count_nonzero(wall_mask)) / wall_mask.size,
                100.0 * int(np.count_nonzero(raw_wall_mask)) / raw_wall_mask.size,
            )

        # ── 1b. Enclosed-space refinement ─────────────────────────────
        # Identify enclosed white spaces in the image (rooms / structural
        # cavities) and apply targeted fixes:
        #   • Spaces with an OCR label inside → labelled room; clear stray
        #     wall pixels so room boundaries are clean.
        #   • Spaces without OCR label AND ≥overlap_thr wall-mask overlap →
        #     structural cavity; fill entirely as wall.
        overlap_thr = float(
            getattr(cfg, "rect_enclosed_overlap_threshold", 0.50)
        )
        try:
            wall_mask = refine_mask_by_enclosed_spaces(
                wall_mask, img_bgr, ocr_bboxes,
                wall_overlap_threshold=overlap_thr,
                close_kernel_size=cap_k,
                close_iters=cap_i,
                debug_img_path=debug_img_path,
            )
        except Exception as _rme:
            logger.warning(
                "refine_mask_by_enclosed_spaces failed (%s) — skipping", _rme
            )

        # ── 2. Diagonal-presence test ──────────────────────────────────
        axis_tol  = getattr(cfg, "rect_axis_tol_deg",      DEFAULT_AXIS_TOL_DEG)
        diag_thr  = getattr(cfg, "rect_diag_lsd_ratio",    DEFAULT_DIAG_LSD_RATIO)
        diag_minl = getattr(cfg, "rect_diag_lsd_min_len",  DEFAULT_DIAG_LSD_MIN_LEN)

        has_diagonals = detect_has_diagonals(
            wall_mask,
            axis_tol_deg    = axis_tol,
            diag_ratio_thr  = diag_thr,
            min_segment_len = diag_minl,
        )
        force_axis_only = not has_diagonals

        # ── 3. Rect decomposition ──────────────────────────────────────
        mask01 = (wall_mask > 0).astype(np.uint8)

        rects: List[rd.Rect] = rd.decompose(
            mask01,
            angle_steps          = getattr(cfg, "rect_angle_steps",         12),
            rect_penalty         = getattr(cfg, "rect_penalty",             0.02),
            max_grid_dim         = getattr(cfg, "rect_max_grid_dim",        200),
            coverage_stop        = getattr(cfg, "rect_coverage_stop",       0.93),
            bleed_weight         = getattr(cfg, "rect_bleed_weight",        1.5),
            initial_bleed_weight = getattr(cfg, "rect_initial_bleed_weight", 10.0),
            bleed_decay          = getattr(cfg, "rect_bleed_decay",         1.0),
            axis_only            = force_axis_only,
            axis_gap             = getattr(cfg, "rect_axis_gap",            0.0),
            max_overlap          = getattr(cfg, "rect_max_overlap",         0.5),
            max_spill_fraction   = getattr(cfg, "rect_max_spill",           0.15),
            refine               = True,
            verbose              = False,
        )

        logger.info(
            "RectWallDetector: %d rectangles  (axis_only=%s, diagonals=%s)",
            len(rects), force_axis_only, has_diagonals,
        )

        # ── 4. Snap collinear endpoints ────────────────────────────────
        infos: List[_RectInfo] = []
        for r in rects:
            p1, p2, thick, wall_ang = rect_centerline(r)
            infos.append(_RectInfo(
                p1=p1, p2=p2, thickness=thick, angle_deg=wall_ang, rect=r,
            ))

        snap_collinear_endpoints(
            infos,
            snap_gap_factor = getattr(cfg, "rect_snap_gap_factor", DEFAULT_SNAP_GAP_FACTOR),
            snap_gap_floor  = getattr(cfg, "rect_snap_gap_floor",  DEFAULT_SNAP_GAP_FLOOR),
            snap_angle_deg  = getattr(cfg, "rect_snap_angle_deg",  DEFAULT_SNAP_ANGLE_DEG),
        )

        # ── 5. WallSegments ────────────────────────────────────────────
        walls = [_rect_to_wall_segment(info, idx, axis_tol_deg=axis_tol)
                 for idx, info in enumerate(infos)]

        # ── 6. Build outline_mask (raster of placed rect walls), used
        #      later to OR with the U-Net mask for room finding.
        outline_mask = rasterise_wall_segments(walls, (H, W))

        return DetectionResult(
            walls=walls,
            openings=[],
            wall_mask=wall_mask,
            raw_wall_mask=raw_wall_mask,
            outline_mask=outline_mask,
            source="rect_walls",
        )

    # ------------------------------------------------------------------
    # U-Net (binary, epoch_20.pth)
    # ------------------------------------------------------------------
    def _run_binary_unet(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the binary wall U-Net following the standard inference protocol:

            logits = model(inp)
            probs  = softmax(logits, dim=1)
            conf_map, pred_argmax = probs.max(dim=1)
            pred[conf_map < threshold] = 0          # uncertain → background
            wall_mask = (pred == 1)                 # class 1 = wall

        Returns a uint8 H×W mask (255 = wall, 0 = background) in the
        original image resolution, with NO morphological post-processing
        so the caller (rect_decompose) gets the raw probability-thresholded
        output.
        """
        ckpt = getattr(self.config, "rect_unet_ckpt_path",
                       "weights/epoch_20.pth")
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        if not ckpt_path.exists():
            logger.error("Binary U-Net checkpoint not found: %s", ckpt_path)
            return None

        try:
            import torch
            from detect_unet import build_model_from_checkpoint, get_val_transform
        except Exception as exc:
            logger.warning("Binary U-Net dependencies unavailable: %s", exc)
            return None

        size       = getattr(self.config, "rect_unet_img_size", 1024)
        confidence = getattr(self.config, "unet_confidence",    0.5)
        device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Load model ──────────────────────────────────────────────────
        try:
            model, _num_classes = build_model_from_checkpoint(
                str(ckpt_path), device
            )
            model.eval()
        except Exception as exc:
            logger.warning("Binary U-Net model load failed: %s", exc)
            return None

        # ── Pre-process ─────────────────────────────────────────────────
        h_orig, w_orig = img_bgr.shape[:2]
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transform = get_val_transform(size)
        inp       = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

        # ── Inference ───────────────────────────────────────────────────
        try:
            with torch.no_grad():
                logits               = model(inp)
                probs                = torch.softmax(logits, dim=1)
                conf_map, pred_argmax = probs.max(dim=1)

            conf_np = conf_map.squeeze(0).cpu().numpy()    # (H_model, W_model)
            pred_np = pred_argmax.squeeze(0).cpu().numpy() # (H_model, W_model)

            # Uncertain pixels → background (class 0)
            pred_np = pred_np.copy()
            pred_np[conf_np < confidence] = 0

        except Exception as exc:
            logger.warning("Binary U-Net inference failed: %s", exc)
            return None
        finally:
            try:
                del inp, logits, probs, conf_map, pred_argmax
            except Exception:
                pass

        # ── Resize to original resolution ───────────────────────────────
        pred_full = cv2.resize(
            pred_np.astype(np.uint8), (w_orig, h_orig),
            interpolation=cv2.INTER_NEAREST,
        )

        # Binary model: class 0 = background, class 1 = wall
        wall_mask = (pred_full == 1).astype(np.uint8) * 255

        logger.info(
            "Binary U-Net (%s): %.1f%% wall pixels  (conf≥%.2f, %d classes)",
            ckpt_path.name,
            100.0 * int(np.count_nonzero(wall_mask)) / wall_mask.size,
            confidence, _num_classes,
        )
        return wall_mask
