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
        │   • coverage_stop=0.97, max_overlap=0.5
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

import contextlib
import io
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from skimage.morphology import skeletonize

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
# Enclosed-space filtering — skeleton-based pipe-end capping
# ---------------------------------------------------------------------------

def _trace_skel(skel: np.ndarray, r0: int, c0: int, max_steps: int):
    """Walk a skeleton path from endpoint (r0, c0) for up to max_steps steps."""
    H, W = skel.shape
    visited = {(r0, c0)}
    path    = [(r0, c0)]
    cur     = (r0, c0)
    for _ in range(max_steps):
        cr, cc = cur
        nbrs = [
            (cr + dr, cc + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
            and 0 <= cr + dr < H
            and 0 <= cc + dc < W
            and skel[cr + dr, cc + dc]
            and (cr + dr, cc + dc) not in visited
        ]
        if not nbrs or len(nbrs) > 1:
            break
        visited.add(nbrs[0])
        path.append(nbrs[0])
        cur = nbrs[0]
    return path


def _local_wall_dir(path, local_steps: int = 15):
    """
    Estimate wall direction from the first ``local_steps`` skeleton steps.

    Returns a unit vector (dr, dc) pointing OUTWARD from the wall toward the
    opening, or None if the path is too short.
    """
    n = min(local_steps, len(path) - 1)
    if n < 2:
        return None
    d = np.array(path[0], float) - np.array(path[n], float)
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else None


def _ray_hit(
    binary: np.ndarray,
    dist_black: np.ndarray,
    r0: int,
    c0: int,
    perp_r: float,
    perp_c: float,
    max_pipe_width: int,
    min_pipe_r: float,
    max_wall_skip: int = 12,
):
    """
    Shoot a ray from (r0, c0) in direction (perp_r, perp_c).

    Phases
    ------
    1. Skip source-wall white pixels (up to max_wall_skip).
    2. Cross the black channel; require dist_transform ≥ min_pipe_r to confirm
       a genuine interior.
    3. Land on the opposing white wall → return (hit_r, hit_c).

    Returns None if the sequence is not completed within max_pipe_width steps.
    """
    H, W = binary.shape
    in_source  = True
    white_skip = 0
    traversed  = False

    for step in range(1, max_pipe_width + 1):
        nr = int(round(r0 + perp_r * step))
        nc = int(round(c0 + perp_c * step))
        if not (0 <= nr < H and 0 <= nc < W):
            return None

        is_white = binary[nr, nc] > 0

        if in_source:
            if is_white:
                white_skip += 1
                if white_skip > max_wall_skip:
                    return None
                continue
            else:
                in_source = False

        if is_white:
            return (nr, nc) if traversed else None

        if dist_black[nr, nc] >= min_pipe_r:
            traversed = True

    return None


def find_caps(
    binary: np.ndarray,
    *,
    min_wall_len:   int   = 30,
    max_pipe_width: int   = 50,
    min_pipe_r:     float = 3.0,
    look_ahead:     int   = 80,
    local_steps:    int   = 15,
    cap_thickness:  Optional[int] = None,
    merge_radius:   int   = 6,
    min_space_size: float = 0.0,
) -> list:
    """
    Detect open pipe ends in a binary wall image and return cap segments.

    Parameters
    ----------
    binary          : uint8 H×W, 255 = wall, 0 = open channel.
    min_wall_len    : minimum skeleton-path length to qualify as a pipe wall.
    max_pipe_width  : maximum channel width searched for the opposing wall.
    min_pipe_r      : minimum distance-transform value needed to confirm a
                      real channel interior.
    look_ahead      : skeleton trace length used for path-length check.
    local_steps     : first N skeleton steps used for direction estimation.
    cap_thickness   : drawn line thickness (None → auto from segment length).
    merge_radius    : suppress duplicate caps whose hit points are within this
                      many pixels of an already-capped location.
    min_space_size  : minimum open-channel area (fraction of image) required
                      for a cap to be placed.  0.0 disables the check.

    Returns
    -------
    List of dicts with keys ``'p1'``, ``'p2'`` (col, row) and ``'thickness'``.
    """
    dist_black = cv2.distanceTransform(
        (binary == 0).astype(np.uint8) * 255, cv2.DIST_L2, 5
    )

    H_img, W_img = binary.shape[:2]
    min_space_px = int(min_space_size * H_img * W_img) if min_space_size > 0.0 else 0
    if min_space_px > 0:
        channel_mask = (binary == 0).astype(np.uint8)
        n_spaces, space_labels = cv2.connectedComponents(channel_mask, connectivity=8)
        space_areas = np.bincount(space_labels.ravel(), minlength=n_spaces)
    else:
        space_labels = None
        space_areas  = None

    skel = skeletonize(binary > 0).astype(np.uint8)
    k    = np.ones((3, 3), np.uint8)
    k[1, 1] = 0
    nc_       = cv2.filter2D(skel, -1, k)
    endpoints = np.argwhere((skel > 0) & (nc_ == 1))

    caps:         list                   = []
    capped_exact: set                    = set()
    capped_hits:  List[Tuple[int, int]]  = []

    def _near_capped_hit(r: int, c: int) -> bool:
        return any(
            abs(r - hr) + abs(c - hc) <= merge_radius
            for hr, hc in capped_hits
        )

    for r, c in endpoints:
        if (r, c) in capped_exact or _near_capped_hit(r, c):
            continue

        path = _trace_skel(skel, r, c, look_ahead)
        if len(path) < min_wall_len:
            continue

        d = _local_wall_dir(path, local_steps)
        if d is None:
            continue
        dr, dc = d

        for sign in (1, -1):
            perp_r = sign * (-dc)
            perp_c = sign *   dr

            hit = _ray_hit(
                binary, dist_black, r, c,
                perp_r, perp_c,
                max_pipe_width, min_pipe_r,
            )
            if hit is None:
                continue

            hr, hc = hit
            if _near_capped_hit(hr, hc):
                continue

            seg_len = max(1, int(np.hypot(hr - r, hc - c)))

            # ≥ 60% of the segment must be black (genuine channel, not bulge)
            black_count = sum(
                1
                for t in range(1, seg_len)
                if binary[
                    int(round(r  + (hr - r)  * t / seg_len)),
                    int(round(c  + (hc - c)  * t / seg_len)),
                ] == 0
            )
            if black_count < seg_len * 0.6:
                continue

            if min_space_px > 0:
                mid_r = int(round((r  + hr) / 2))
                mid_c = int(round((c  + hc) / 2))
                space_label = space_labels[mid_r, mid_c]
                if space_label == 0 or space_areas[space_label] < min_space_px:
                    continue

            thick = cap_thickness if cap_thickness is not None \
                    else max(2, seg_len // 6)

            caps.append({"p1": (c, r), "p2": (hc, hr), "thickness": thick})
            capped_exact.add((r, c))
            capped_hits.append((r, c))
            capped_hits.append((hr, hc))
            break

    logger.info("find_caps: %d open pipe end(s) capped", len(caps))
    return caps


def apply_caps(binary: np.ndarray, caps: list) -> np.ndarray:
    """Draw white cap lines onto *binary* (copy) and return it."""
    result = binary.copy()
    for cap in caps:
        cv2.line(result, cap["p1"], cap["p2"], 255, cap["thickness"])
    return result


def _cap_pipe_openings(
    binary: np.ndarray,
    *,
    min_wall_len:   int   = 30,
    max_pipe_width: int   = 50,
    min_pipe_r:     float = 3.0,
    look_ahead:     int   = 80,
    local_steps:    int   = 15,
    cap_thickness:  Optional[int] = None,
    merge_radius:   int   = 6,
    min_space_size: float = 0.0,
) -> np.ndarray:
    """
    Close open pipe ends in *binary* (255 = open space, 0 = wall).

    Inverts so walls are white, calls :func:`find_caps`, draws caps, then
    inverts back.
    """
    wall_binary  = cv2.bitwise_not(binary)
    caps         = find_caps(
        wall_binary,
        min_wall_len=min_wall_len,
        max_pipe_width=max_pipe_width,
        min_pipe_r=min_pipe_r,
        look_ahead=look_ahead,
        local_steps=local_steps,
        cap_thickness=cap_thickness,
        merge_radius=merge_radius,
        min_space_size=min_space_size,
    )
    wall_capped = apply_caps(wall_binary, caps)
    return cv2.bitwise_not(wall_capped)


def find_enclosed_spaces(
    img_bgr: np.ndarray,
    wall_mask: Optional[np.ndarray] = None,   # kept for API compat, unused
    *,
    min_wall_len:   int   = 30,
    max_pipe_width: int   = 50,
    min_pipe_r:     float = 3.0,
    look_ahead:     int   = 80,
    local_steps:    int   = 15,
    cap_thickness:  Optional[int] = None,
    merge_radius:   int   = 6,
    min_space_size: float = 0.0,
) -> np.ndarray:
    """
    Label every white connected region in the floor-plan image.

    Algorithm
    ---------
    1. Grayscale + Otsu binarise: walls → 0, open space → 255.
    2. Seal open pipe ends via skeleton-based capping (:func:`_cap_pipe_openings`).
    3. ``cv2.connectedComponents`` on the capped binary.

    Returns
    -------
    labels : ndarray (H, W), int32
        0 = wall/dark pixels, 1..N = individual white regions.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = _cap_pipe_openings(
        binary,
        min_wall_len=min_wall_len,
        max_pipe_width=max_pipe_width,
        min_pipe_r=min_pipe_r,
        look_ahead=look_ahead,
        local_steps=local_steps,
        cap_thickness=cap_thickness,
        merge_radius=merge_radius,
        min_space_size=min_space_size,
    )

    _, labels = cv2.connectedComponents(binary, connectivity=8)
    return labels.astype(np.int32)


def _save_enclosed_debug_image(
    path: str,
    img_bgr: np.ndarray,
    labels: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> None:
    """
    Write a colour-coded debug image: each enclosed region gets a unique hue
    (golden-angle HSV spacing), walls stay dark grey, pixel counts are
    annotated at centroids (up to 200 regions), OCR bboxes outlined in white.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_dbg = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        n_regions = int(labels.max())
        lut = np.zeros((n_regions + 1, 3), dtype=np.uint8)
        lut[0] = (60, 60, 60)
        for i in range(1, n_regions + 1):
            hue = int((i * 137.508) % 360) // 2
            hsv = np.uint8([[[hue, 220, 220]]])
            lut[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

        colored = lut[labels]
        colored[binary_dbg == 0] = (60, 60, 60)

        region_stats = []
        for i in range(1, n_regions + 1):
            rmask = labels == i
            area = int(np.count_nonzero(rmask))
            if area == 0:
                continue
            ys, xs = np.where(rmask)
            region_stats.append((area, i, int(xs.mean()), int(ys.mean())))
        region_stats.sort(reverse=True)

        for area, _idx, cx_r, cy_r in region_stats[:200]:
            txt = str(area)
            (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(colored, txt, (cx_r - tw // 2 + 1, cy_r + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(colored, txt, (cx_r - tw // 2, cy_r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        for (bx1, by1, bx2, by2) in boxes:
            cv2.rectangle(colored, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

        cv2.imwrite(path, colored)
        logger.info(
            "Enclosed-space debug image saved → %s  (%d regions, %d labelled)",
            path, n_regions, len(region_stats),
        )
    except Exception as exc:
        logger.warning("Could not save enclosed-space debug image: %s", exc)


def refine_mask_by_enclosed_spaces(
    wall_mask: np.ndarray,
    img_bgr: np.ndarray,
    ocr_bboxes: Optional[List] = None,
    *,
    wall_overlap_threshold: float = 0.50,
    min_wall_len:   int   = 30,
    max_pipe_width: int   = 50,
    min_pipe_r:     float = 3.0,
    look_ahead:     int   = 80,
    local_steps:    int   = 15,
    cap_thickness:  Optional[int] = None,
    merge_radius:   int   = 6,
    min_space_size: float = 0.0,
    debug_img_path: Optional[str] = None,
) -> np.ndarray:
    """
    Refine *wall_mask* using enclosed-space analysis.

    Open pipe ends are sealed with the skeleton-based cap algorithm before
    region labelling, so partially-open rooms are treated as enclosed.

    For every enclosed region:
    * No OCR label inside + ≥ wall_overlap_threshold wall-mask overlap
      → structural cavity; fill entirely as wall (255).
    * OCR label inside or sparse wall coverage → clear to 0.

    Parameters
    ----------
    wall_mask              : H×W uint8, 255 = wall.
    img_bgr                : BGR image (text regions already erased).
    ocr_bboxes             : list of (x1,y1,x2,y2) or (text,x1,y1,x2,y2).
    wall_overlap_threshold : fraction of region pixels that must be wall.
    min_wall_len … min_space_size : forwarded to :func:`find_enclosed_spaces`.
    debug_img_path         : optional path for colour-coded debug image.

    Returns
    -------
    Refined wall_mask (copy — input not modified).
    """
    if wall_mask is None:
        return wall_mask

    refined = wall_mask.copy()
    H, W = refined.shape[:2]

    # Normalise ocr_bboxes
    boxes: List[Tuple[int, int, int, int]] = []
    if ocr_bboxes:
        for item in ocr_bboxes:
            if len(item) == 4:
                boxes.append((int(item[0]), int(item[1]),
                               int(item[2]), int(item[3])))
            elif len(item) >= 5:
                boxes.append((int(item[1]), int(item[2]),
                               int(item[3]), int(item[4])))

    ocr_centres: List[Tuple[float, float]] = [
        ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
        for (bx1, by1, bx2, by2) in boxes
    ]

    labels = find_enclosed_spaces(
        img_bgr, wall_mask,
        min_wall_len=min_wall_len,
        max_pipe_width=max_pipe_width,
        min_pipe_r=min_pipe_r,
        look_ahead=look_ahead,
        local_steps=local_steps,
        cap_thickness=cap_thickness,
        merge_radius=merge_radius,
        min_space_size=min_space_size,
    )

    n_labels = int(labels.max())
    if n_labels == 0:
        logger.info(
            "refine_mask_by_enclosed_spaces: no enclosed regions found "
            "(find_enclosed_spaces returned 0 labels)"
        )
        return refined

    logger.info(
        "refine_mask_by_enclosed_spaces: %d enclosed regions found, "
        "threshold=%.2f, ocr_boxes=%d",
        n_labels, wall_overlap_threshold, len(boxes),
    )

    filled_count = 0
    cleared_count = 0

    for region_id in range(1, n_labels + 1):
        region_mask = labels == region_id
        region_area = int(np.count_nonzero(region_mask))
        if region_area == 0:
            continue

        has_ocr = False
        for (cx, cy) in ocr_centres:
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= yi < H and 0 <= xi < W and region_mask[yi, xi]:
                has_ocr = True
                break

        overlap = int(np.count_nonzero(wall_mask[region_mask] > 0))
        overlap_frac = overlap / region_area

        if not has_ocr and overlap_frac >= wall_overlap_threshold:
            refined[region_mask] = 255
            filled_count += 1
            logger.info(
                "  Region %d: area=%d px, wall_overlap=%.1f%% (≥%.0f%%), "
                "has_ocr=False → FILLED entirely as wall",
                region_id, region_area, overlap_frac * 100,
                wall_overlap_threshold * 100,
            )
        else:
            refined[region_mask] = 0
            cleared_count += 1
            reason = (
                "has OCR label inside"
                if has_ocr
                else f"wall_overlap={overlap_frac*100:.1f}% < threshold {wall_overlap_threshold*100:.0f}%"
            )
            logger.info(
                "  Region %d: area=%d px, wall_overlap=%.1f%%, "
                "has_ocr=%s → CLEARED (%s)",
                region_id, region_area, overlap_frac * 100, has_ocr, reason,
            )

    logger.info(
        "refine_mask_by_enclosed_spaces: %d structural regions filled, "
        "%d room regions cleared  (total enclosed=%d)",
        filled_count, cleared_count, n_labels,
    )

    if debug_img_path:
        _save_enclosed_debug_image(debug_img_path, img_bgr, labels, boxes)

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
        log_file_path: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect walls in ``img_bgr``.  If ``wall_mask`` is supplied it is used
        directly; otherwise the binary U-Net is invoked.

        Parameters
        ----------
        img_bgr       : BGR image (text regions ideally already erased).
        wall_mask     : optional pre-computed binary wall mask; if None the
                        U-Net checkpoint is used.
        ocr_bboxes    : list of ``(x1, y1, x2, y2)`` or ``(text, x1, y1, x2, y2)``
                        bounding boxes for OCR text regions.  Used by the
                        enclosed-space refinement step to distinguish rooms
                        (labelled) from structural cavities (unlabelled).
        log_file_path : optional path for an exhaustive rect-decompose log
                        file.  All decompose parameters, rd.decompose verbose
                        output, per-rect stats, and final coverage are written
                        there.  Does not affect any return values.
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
        total_wall_px = int(np.count_nonzero(mask01))
        total_px      = mask01.size

        # Collect all decompose params for logging
        _p_angle_steps   = getattr(cfg, "rect_angle_steps",          12)
        _p_penalty       = getattr(cfg, "rect_penalty",              0.0)
        _p_max_grid      = getattr(cfg, "rect_max_grid_dim",         200)
        _p_cov_stop      = getattr(cfg, "rect_coverage_stop",        0.99)
        _p_bleed_w       = getattr(cfg, "rect_bleed_weight",         1.5)
        _p_init_bleed    = getattr(cfg, "rect_initial_bleed_weight", 10.0)
        _p_bleed_decay   = getattr(cfg, "rect_bleed_decay",          1.0)
        _p_axis_gap      = getattr(cfg, "rect_axis_gap",             0.0)
        _p_max_overlap   = getattr(cfg, "rect_max_overlap",          0.3)
        _p_max_spill     = getattr(cfg, "rect_max_spill",            0.15)

        log_lines: List[str] = []
        log_lines.append("=" * 70 + "\n")
        log_lines.append("RECT DECOMPOSE EXHAUSTIVE LOG\n")
        log_lines.append("=" * 70 + "\n\n")
        log_lines.append(f"Image size          : {W} x {H} px  ({total_px} total)\n")
        log_lines.append(f"Wall pixels (input) : {total_wall_px} / {total_px}  "
                         f"({100.0*total_wall_px/max(1,total_px):.2f}%)\n")
        log_lines.append("\nDecompose parameters:\n")
        log_lines.append(f"  coverage_stop        = {_p_cov_stop}  (target: "
                         f"{_p_cov_stop*100:.1f}% of wall pixels covered)\n")
        log_lines.append(f"  rect_penalty         = {_p_penalty}\n")
        log_lines.append(f"  max_overlap          = {_p_max_overlap}\n")
        log_lines.append(f"  max_spill_fraction   = {_p_max_spill}\n")
        log_lines.append(f"  bleed_weight         = {_p_bleed_w}\n")
        log_lines.append(f"  initial_bleed_weight = {_p_init_bleed}\n")
        log_lines.append(f"  bleed_decay          = {_p_bleed_decay}\n")
        log_lines.append(f"  angle_steps          = {_p_angle_steps}\n")
        log_lines.append(f"  axis_only            = {force_axis_only}\n")
        log_lines.append(f"  axis_gap             = {_p_axis_gap}\n")
        log_lines.append(f"  max_grid_dim         = {_p_max_grid}\n")
        log_lines.append(f"  has_diagonals        = {has_diagonals}\n")
        log_lines.append("\n--- rd.decompose verbose output ---\n")

        _verbose_buf = io.StringIO()
        with contextlib.redirect_stdout(_verbose_buf):
            rects: List[rd.Rect] = rd.decompose(
                mask01,
                angle_steps          = _p_angle_steps,
                rect_penalty         = _p_penalty,
                max_grid_dim         = _p_max_grid,
                coverage_stop        = _p_cov_stop,
                bleed_weight         = _p_bleed_w,
                initial_bleed_weight = _p_init_bleed,
                bleed_decay          = _p_bleed_decay,
                axis_only            = force_axis_only,
                axis_gap             = _p_axis_gap,
                max_overlap          = _p_max_overlap,
                max_spill_fraction   = _p_max_spill,
                refine               = True,
                verbose              = True,
            )
        _verbose_text = _verbose_buf.getvalue()
        log_lines.append(_verbose_text if _verbose_text else "(no verbose output from rd.decompose)\n")

        # ── Post-decompose coverage analysis ──────────────────────────
        log_lines.append("\n--- Post-decompose analysis ---\n")
        log_lines.append(f"Rectangles placed    : {len(rects)}\n")

        # Rasterise all rects to measure actual covered wall pixels
        _covered = np.zeros((H, W), dtype=np.uint8)
        for _r in rects:
            _corners = _r.corners().astype(np.int32)
            cv2.fillPoly(_covered, [_corners], 1)
        _covered_wall_px   = int(np.count_nonzero((_covered > 0) & (mask01 > 0)))
        _covered_total_px  = int(np.count_nonzero(_covered > 0))
        _coverage_frac     = _covered_wall_px / max(1, total_wall_px)
        _gap_px            = total_wall_px - _covered_wall_px
        log_lines.append(f"Wall px covered      : {_covered_wall_px} / {total_wall_px}  "
                         f"({_coverage_frac*100:.2f}%)\n")
        log_lines.append(f"Target coverage      : {_p_cov_stop*100:.1f}%\n")
        log_lines.append(f"Gap to target        : {(_p_cov_stop - _coverage_frac)*100:.2f}pp\n")
        log_lines.append(f"Uncovered wall px    : {_gap_px}\n")
        log_lines.append(f"Extra (spill) px     : {_covered_total_px - _covered_wall_px} "
                         f"(outside wall mask)\n")

        # ── Diagnostic overlay image ───────────────────────────────────
        # WHITE  = wall ∩ rect
        # GREEN  = wall pixel not covered by any rect
        # RED    = rect pixel outside wall mask
        # BLACK  = everything else
        if debug_img_path:
            try:
                _wall_bin = mask01 > 0
                _rect_bin = _covered > 0
                _diag = np.zeros((H, W, 3), dtype=np.uint8)
                _diag[_wall_bin & _rect_bin]  = (255, 255, 255)  # white
                _diag[_wall_bin & ~_rect_bin] = (0,   255,   0)  # green
                _diag[_rect_bin & ~_wall_bin] = (0,     0, 255)  # red
                _diag_path = os.path.splitext(debug_img_path)[0] + "_wall_rect_diag.png"
                cv2.imwrite(_diag_path, _diag)
                logger.info("RectWallDetector: wall/rect diagnostic image → %s", _diag_path)
            except Exception as _de:
                logger.warning("RectWallDetector: could not save diagnostic image: %s", _de)

        if len(rects) > 0:
            _areas = []
            for _r in rects:
                _c = _r.corners()
                _xs, _ys = _c[:, 0], _c[:, 1]
                _areas.append(float((_xs.max()-_xs.min()) * (_ys.max()-_ys.min())))
            log_lines.append(f"\nPer-rect area stats (px²):\n")
            log_lines.append(f"  min={min(_areas):.0f}  max={max(_areas):.0f}  "
                             f"mean={sum(_areas)/len(_areas):.0f}\n")
            log_lines.append("\nAll rects (x1,y1,x2,y2  angle_deg  area_px²):\n")
            for _idx, _r in enumerate(rects):
                _c = _r.corners()
                _xs, _ys = _c[:, 0], _c[:, 1]
                _ang = getattr(_r, 'angle', 0.0)
                _a   = (_xs.max()-_xs.min()) * (_ys.max()-_ys.min())
                log_lines.append(
                    f"  rect[{_idx:3d}]: "
                    f"({_xs.min():.0f},{_ys.min():.0f})-"
                    f"({_xs.max():.0f},{_ys.max():.0f})  "
                    f"angle={_ang:.1f}°  area={_a:.0f}\n"
                )

        log_lines.append("\n" + "=" * 70 + "\n")

        # Write log file (always; path derived from debug_img_path if not given)
        _lf_path = log_file_path
        if not _lf_path and debug_img_path:
            _lf_path = os.path.splitext(debug_img_path)[0] + "_rectdecompose.log"
        if _lf_path:
            try:
                with open(_lf_path, "w", encoding="utf-8") as _lf:
                    _lf.writelines(log_lines)
                logger.info("RectWallDetector: exhaustive decompose log → %s", _lf_path)
            except Exception as _le:
                logger.warning("RectWallDetector: could not write log file: %s", _le)

        logger.info(
            "RectWallDetector: %d rectangles, coverage=%.2f%% / target %.1f%%  "
            "(axis_only=%s, diagonals=%s)",
            len(rects), _coverage_frac * 100, _p_cov_stop * 100,
            force_axis_only, has_diagonals,
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

        # Store raw rect corners for the overlay debug image.
        # Captured before endpoint snapping so the polygons reflect exactly
        # what rect_decompose placed, without post-hoc adjustments.
        wall_rect_polygons = [r.corners().astype(np.float32) for r in rects]

        return DetectionResult(
            walls=walls,
            openings=[],
            wall_mask=wall_mask,
            raw_wall_mask=raw_wall_mask,
            outline_mask=outline_mask,
            wall_rect_polygons=wall_rect_polygons,
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
