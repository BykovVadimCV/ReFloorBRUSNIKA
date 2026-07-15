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
        │   • coverage_stop=0.93, max_overlap=1.0 (disabled)
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
import datetime
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

from core import rect_decompose as rd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable defaults — exposed via PipelineConfig
# ---------------------------------------------------------------------------

DEFAULT_AXIS_TOL_DEG    = 3.0      # walls within ±this of 0/90 are axis-aligned
DEFAULT_DIAG_LSD_RATIO  = 0.12     # if ≥12% of LSD line-length is diagonal,
                                   # allow diagonal rectangles (else axis-only)
DEFAULT_DIAG_LSD_AXIS_TOL = 7.0    # LSD lines within ±this of 0/90 do NOT count
                                   # as diagonal evidence — they get straightened
                                   # (see rect_straighten_max_dev_deg) anyway, so
                                   # they must not flip the plan into diagonal mode
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
    snap_gap_factor:    float = DEFAULT_SNAP_GAP_FACTOR,
    snap_gap_floor:     float = DEFAULT_SNAP_GAP_FLOOR,
    snap_angle_deg:     float = DEFAULT_SNAP_ANGLE_DEG,
    max_extend_frac:    float = 0.10,
) -> None:
    """
    Extend wall endpoints at corners / T-junctions to close the perimeter.

    Rules
    -----
    * Each rect_decompose rectangle remains its own independent wall — parallel
      walls (angle difference ≤ snap_angle_deg) are NEVER merged or moved
      toward each other.
    * Only walls that meet at a genuine angle (corner / T-junction) are
      eligible for snapping.
    * Even then, each endpoint may only move by at most ``max_extend_frac``
      of that wall's own length (default 10%).  Larger shifts are rejected.
    * Only endpoints whose raw gap is already within the threshold are touched;
      the threshold is small (factor × thickness, floor-capped) so only
      near-misses at real junctions qualify.

    Modifies ``infos`` in place — only endpoint coordinates are shifted,
    wall thickness is never changed.
    """
    n = len(infos)
    for i in range(n):
        a     = infos[i]
        len_a = _dist(a.p1, a.p2)
        gap_a = max(snap_gap_floor, snap_gap_factor * a.thickness)

        for j in range(n):
            if i == j:
                continue
            b     = infos[j]
            len_b = _dist(b.p1, b.p2)
            gap_b = max(snap_gap_floor, snap_gap_factor * b.thickness)
            gap_thr = max(gap_a, gap_b)

            # Find the closest endpoint pair
            pairs = [
                ('p1', a.p1, 'p1', b.p1),
                ('p1', a.p1, 'p2', b.p2),
                ('p2', a.p2, 'p1', b.p1),
                ('p2', a.p2, 'p2', b.p2),
            ]
            ka, pa, kb, pb = min(pairs, key=lambda t: _dist(t[1], t[3]))
            if _dist(pa, pb) > gap_thr:
                continue

            adiff = _angle_dist(a.angle_deg, b.angle_deg)

            # Parallel walls → merging is forbidden; skip entirely.
            if adiff <= snap_angle_deg:
                continue

            # Corner / T-junction: extend both endpoints to the line
            # intersection so the perimeter closes cleanly.
            meet = _line_intersection(a.p1, a.p2, b.p1, b.p2)
            if meet is None:
                continue

            # The intersection must sit near both endpoints (not far away on
            # the infinite line extension).
            if _dist(meet, pa) > gap_thr * 2.5 or _dist(meet, pb) > gap_thr * 2.5:
                continue

            # Hard limit: each wall may be extended by at most
            # max_extend_frac of its own length (default 10%).
            # This prevents long-range endpoint teleportation.
            ext_a = _dist(pa, meet)
            ext_b = _dist(pb, meet)
            if len_a > 0 and ext_a > max_extend_frac * len_a:
                continue
            if len_b > 0 and ext_b > max_extend_frac * len_b:
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

    # ── Second pass: T-junctions (endpoint → mid-span of another wall) ──
    # The endpoint-pair pass above only closes corners where two wall ENDS
    # nearly meet. A wall ending just short of another wall's SIDE has no
    # nearby endpoint, so it stayed open and exported as a disconnected
    # slab. Extend such endpoints along their own wall line until they
    # reach the crossing wall's centerline.
    for i in range(n):
        a = infos[i]
        len_a = _dist(a.p1, a.p2)
        if len_a < 1e-6:
            continue
        gap_a = max(snap_gap_floor, snap_gap_factor * a.thickness)

        for key in ('p1', 'p2'):
            pa = getattr(a, key)
            other = a.p2 if key == 'p1' else a.p1
            best_meet = None
            best_ext = float('inf')

            for j in range(n):
                if j == i:
                    continue
                b = infos[j]
                if _angle_dist(a.angle_deg, b.angle_deg) <= snap_angle_deg:
                    continue
                len_b = _dist(b.p1, b.p2)
                if len_b < 1e-6:
                    continue

                meet = _line_intersection(a.p1, a.p2, b.p1, b.p2)
                if meet is None:
                    continue
                # Meet must land within b's span (a genuine T, not a far
                # extension past b's end).
                t = (((meet[0] - b.p1[0]) * (b.p2[0] - b.p1[0])
                      + (meet[1] - b.p1[1]) * (b.p2[1] - b.p1[1]))
                     / (len_b * len_b))
                if t < -0.05 or t > 1.05:
                    continue
                # Extend outward only: the meet must lie beyond pa as seen
                # from the opposite endpoint (never fold the wall back).
                if _dist(other, meet) < len_a - 1e-6:
                    continue
                # Allowed extension: cross the face gap plus half of b's
                # body to reach its centerline.
                ext = _dist(pa, meet)
                max_ext = max(gap_a,
                              snap_gap_factor * b.thickness,
                              snap_gap_floor) + b.thickness / 2.0
                if ext > max_ext or ext >= best_ext:
                    continue
                best_ext = ext
                best_meet = meet

            if best_meet is not None and best_ext > 1e-6:
                if key == 'p1':
                    a.p1 = best_meet
                else:
                    a.p2 = best_meet


# ---------------------------------------------------------------------------
# Minor-diagonal suppression
# ---------------------------------------------------------------------------

def _suppress_minor_diagonals(
    infos: List[_RectInfo],
    *,
    axis_tol_deg: float = DEFAULT_AXIS_TOL_DEG,
    junction_radius_factor: float = 2.0,
    junction_radius_floor: float = 12.0,
    short_frac: float = 0.5,
    max_diag_frac: float = 0.34,
) -> int:
    """
    Snap small, isolated diagonal walls to the nearest cardinal axis.

    A diagonal wall's centerline is rotated to pure horizontal/vertical about
    its own midpoint (length preserved) only when ALL of the following hold:

      (a) every wall sharing a junction with it is axis-aligned (no diagonal
          neighbours) and at least one such neighbour exists;
      (b) it is short relative to the median wall length
          (length < ``short_frac`` × median);
      (c) the plan is mostly orthogonal — diagonal walls are a small minority
          (≤ ``max_diag_frac`` of all walls).

    Rationale: genuine diagonal architecture comes in runs of mutually-aligned
    diagonal walls; a lone short diagonal wedged between axis walls in an
    otherwise-orthogonal plan is almost always a decomposition artefact, so we
    "keep it to itself" as an axis-aligned segment instead.

    Modifies ``infos`` in place.  Endpoint snapping should be re-run afterwards
    so a rotated wall still meets its neighbours at the corner.  Returns the
    number of walls snapped.
    """
    if not infos:
        return 0

    def _dev(a: float) -> float:
        return min(a % 90.0, 90.0 - (a % 90.0))

    diag_idx = [i for i, info in enumerate(infos)
                if _dev(info.angle_deg) > axis_tol_deg]
    if not diag_idx:
        return 0

    # (c) global orthogonality gate — bail out on genuinely diagonal plans.
    if len(diag_idx) / len(infos) > max_diag_frac:
        return 0

    lengths = [_dist(info.p1, info.p2) for info in infos]
    median_len = float(np.median(lengths)) if lengths else 0.0
    if median_len <= 0.0:
        return 0

    snapped = 0
    for i in diag_idx:
        a = infos[i]
        len_a = _dist(a.p1, a.p2)

        # (b) short relative to the typical wall
        if len_a >= short_frac * median_len:
            continue

        # (a) every junction neighbour must be axis-aligned
        jr = max(junction_radius_floor, junction_radius_factor * a.thickness)
        has_axis_neighbour = False
        has_diag_neighbour = False
        for j, b in enumerate(infos):
            if j == i:
                continue
            touch = min(
                _dist(a.p1, b.p1), _dist(a.p1, b.p2),
                _dist(a.p2, b.p1), _dist(a.p2, b.p2),
            )
            if touch > jr:
                continue
            if _dev(b.angle_deg) > axis_tol_deg:
                has_diag_neighbour = True
                break
            has_axis_neighbour = True
        if has_diag_neighbour or not has_axis_neighbour:
            continue

        # All conditions met — rotate the centerline to the nearest axis about
        # its midpoint, preserving length.
        mid = ((a.p1[0] + a.p2[0]) / 2.0, (a.p1[1] + a.p2[1]) / 2.0)
        half = len_a / 2.0
        ang_mod = a.angle_deg % 180.0
        dev_horizontal = min(ang_mod, 180.0 - ang_mod)
        dev_vertical = abs(ang_mod - 90.0)
        if dev_horizontal <= dev_vertical:
            a.p1 = (mid[0] - half, mid[1])
            a.p2 = (mid[0] + half, mid[1])
            a.angle_deg = 0.0
        else:
            a.p1 = (mid[0], mid[1] - half)
            a.p2 = (mid[0], mid[1] + half)
            a.angle_deg = 90.0
        snapped += 1

    return snapped


# ---------------------------------------------------------------------------
# Near-axis straightening (unconditional)
# ---------------------------------------------------------------------------

def _straighten_near_axis_walls(
    infos: List[_RectInfo],
    *,
    max_dev_deg: float = 7.0,
) -> int:
    """
    Force every wall whose centerline lies within ``max_dev_deg`` of a cardinal
    axis to be exactly horizontal or vertical.

    Unlike :func:`_suppress_minor_diagonals`, this is **unconditional**: it does
    not look at neighbours, relative length, or the global diagonal fraction, and
    it accepts whatever change in mask coverage results.  A wall tilted by only a
    few degrees is almost always a decomposition wobble on a wall the architect
    drew straight; leaving it at, say, 4° gets it classified ``is_diagonal`` by
    everything downstream (parent-wall finding, opening placement, and the
    ``_drop_diagonal_openings`` filter all reject diagonals), which silently
    drops doors/windows that sit on it.

    The centerline is rotated about its own midpoint to the nearest axis with
    its length preserved.  Modifies ``infos`` in place; endpoint snapping should
    be re-run afterwards so a straightened wall still meets its neighbours at the
    corner.  Returns the number of walls straightened.
    """
    if max_dev_deg <= 0.0:
        return 0

    straightened = 0
    for a in infos:
        ang_mod = a.angle_deg % 180.0
        dev_horizontal = min(ang_mod, 180.0 - ang_mod)
        dev_vertical = abs(ang_mod - 90.0)
        dev = min(dev_horizontal, dev_vertical)
        if dev <= 1e-9 or dev > max_dev_deg:
            continue

        mid = ((a.p1[0] + a.p2[0]) / 2.0, (a.p1[1] + a.p2[1]) / 2.0)
        half = _dist(a.p1, a.p2) / 2.0
        if dev_horizontal <= dev_vertical:
            a.p1 = (mid[0] - half, mid[1])
            a.p2 = (mid[0] + half, mid[1])
            a.angle_deg = 0.0
        else:
            a.p1 = (mid[0], mid[1] - half)
            a.p2 = (mid[0], mid[1] + half)
            a.angle_deg = 90.0
        straightened += 1

    return straightened


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


# Re-export canonical implementation from core.ocr_utils.
# All callers that import _is_numeric_ocr_label from this module continue
# to work; the logic lives in one place.
from core.ocr_utils import is_numeric_ocr_label as _is_numeric_ocr_label  # noqa: E402

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
    min_space_size: float = 0.001,
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

    # Normalise ocr_bboxes — keep only boxes whose text is a numeric label
    # (e.g. room area "12.5", "8 м²").  Non-numeric text (room names like
    # "Кухня", "WC") and coordinate-only 4-tuples are ignored so they do
    # not prevent structural cavities from being filled as wall.
    boxes: List[Tuple[int, int, int, int]] = []
    if ocr_bboxes:
        for item in ocr_bboxes:
            if len(item) >= 5:
                # (text, x1, y1, x2, y2) — only keep if text is numeric
                if _is_numeric_ocr_label(str(item[0])):
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
# Door + OCR-text exclusion for the wall mask
# (mirrors test_cap.py exactly so both pipelines produce identical masks)
# ---------------------------------------------------------------------------

def _run_door_window_pixel_mask(
    img_bgr: np.ndarray,
    ckpt_path: str = "weights/epoch_040.pth",
    img_size: int = 1024,
    confidence: float = 0.5,
    include_windows: bool = True,
) -> Optional[np.ndarray]:
    """
    Run the 4-class U-Net (epoch_040.pth) and return a uint8 mask of door
    pixels (class 3) and optionally window pixels (class 2).
    Returns None on failure.
    """
    h, w = img_bgr.shape[:2]
    if not os.path.exists(ckpt_path):
        logger.warning("Door-mask checkpoint not found: %s", ckpt_path)
        return None

    try:
        import torch
        from detection.unet_inference import build_model_from_checkpoint, get_val_transform

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, num_classes = build_model_from_checkpoint(ckpt_path, dev)
        model = model.to(dev).eval()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transform = get_val_transform(img_size)
        inp = transform(image=img_rgb)["image"].unsqueeze(0).to(dev)

        with torch.no_grad():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)
            conf_map, pred_idx = probs.max(dim=1)

        conf_np = conf_map.squeeze(0).cpu().numpy()
        pred_np = pred_idx.squeeze(0).cpu().numpy().copy()
        pred_np[conf_np < confidence] = 0

        pred_full = cv2.resize(pred_np.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
        door_px = (pred_full == 3) if num_classes >= 4 else np.zeros((h, w), bool)
        win_px  = (pred_full == 2) if (include_windows and num_classes >= 3) \
                  else np.zeros((h, w), bool)
        return (door_px | win_px).astype(np.uint8) * 255

    except Exception as exc:
        logger.warning("Door-mask U-Net failed: %s", exc)
        return None


def apply_door_text_filter(
    wall_mask: np.ndarray,
    img_bgr: np.ndarray,
    ocr_bboxes: Optional[List] = None,
    *,
    door_ckpt_path: str = "weights/epoch_040.pth",
    img_size: int = 1024,
    confidence: float = 0.5,
    door_margin: int = 10,
    text_margin: int = 5,
    include_windows: bool = True,
    door_pixel_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply the same door + OCR-text exclusion to *wall_mask* that test_cap.py
    builds for its cap-detection input.  Returns a filtered copy.

    Pipeline (matches test_cap.py):
      1. Compute door+window pixel mask via epoch_040 (or use *door_pixel_mask*).
      2. Dilate by *door_margin* px so thin predictions cover full opening.
      3. Subtract wall pixels — wall U-Net takes priority over door U-Net.
      4. Build text mask by dilating each OCR word bbox by *text_margin* px.
      5. Zero out (door ∪ text) regions in wall_mask.
    """
    h, w = wall_mask.shape[:2]
    refined = wall_mask.copy()

    # ── 1-3. Door / window exclusion ───────────────────────────────────
    if door_pixel_mask is None:
        door_pixel_mask = _run_door_window_pixel_mask(
            img_bgr,
            ckpt_path       = door_ckpt_path,
            img_size        = img_size,
            confidence      = confidence,
            include_windows = include_windows,
        )

    door_mask = np.zeros((h, w), dtype=np.uint8)
    if door_pixel_mask is not None and np.any(door_pixel_mask > 0):
        door_mask = door_pixel_mask.copy()
        if door_margin > 0:
            k = np.ones((door_margin * 2 + 1, door_margin * 2 + 1), dtype=np.uint8)
            door_mask = cv2.dilate(door_mask, k, iterations=1)
        # Wall takes priority — keep wall pixels even if the door U-Net
        # predicted them as a door.
        door_mask[refined > 0] = 0

    # ── 4. OCR text exclusion ──────────────────────────────────────────
    text_mask = np.zeros((h, w), dtype=np.uint8)
    if ocr_bboxes:
        for item in ocr_bboxes:
            # Accept (x1,y1,x2,y2) or (text,x1,y1,x2,y2)
            if len(item) >= 5:
                x1, y1, x2, y2 = int(item[1]), int(item[2]), int(item[3]), int(item[4])
            elif len(item) == 4:
                x1, y1, x2, y2 = map(int, item)
            else:
                continue
            x1 = max(0, x1 - text_margin);  y1 = max(0, y1 - text_margin)
            x2 = min(w, x2 + text_margin);  y2 = min(h, y2 + text_margin)
            if x2 > x1 and y2 > y1:
                text_mask[y1:y2, x1:x2] = 255

    # ── 5. Apply combined exclusion ────────────────────────────────────
    excl = (door_mask > 0) | (text_mask > 0)
    refined[excl] = 0

    n_door = int(np.count_nonzero(door_mask > 0))
    n_text = int(np.count_nonzero(text_mask > 0))
    n_excl = int(np.count_nonzero(excl))
    logger.info(
        "apply_door_text_filter: excluded %d wall px "
        "(door=%d, text=%d, combined=%d)",
        int(np.count_nonzero(wall_mask > 0)) - int(np.count_nonzero(refined > 0)),
        n_door, n_text, n_excl,
    )
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
        Detect walls in ``img_bgr``.

        A full pipeline log is always written to a file.  The path is chosen
        as (in priority order):

            1. ``log_file_path`` if explicitly given
            2. Stem of ``debug_img_path`` + ``_rect_walls.log``
            3. ``rect_walls_detect.log`` in the current working directory
        """
        cfg   = self.config
        H, W  = img_bgr.shape[:2]
        total_px = H * W

        # ── Logging helpers ────────────────────────────────────────────
        _ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _LL: List[str] = []

        def _L(msg: str, *args) -> None:
            line = (msg % args) if args else msg
            _LL.append(line + "\n")
            logger.info(line)

        def _LS(title: str) -> None:
            bar = "─" * 68
            _LL.append(f"\n{bar}\n  {title}\n{bar}\n")

        def _write_log() -> None:
            lf = log_file_path
            if not lf and debug_img_path:
                lf = os.path.splitext(debug_img_path)[0] + "_rect_walls.log"
            if not lf:
                lf = "rect_walls_detect.log"
            try:
                with open(lf, "w", encoding="utf-8") as _f:
                    _f.writelines(_LL)
                logger.info("RectWallDetector: pipeline log → %s", lf)
            except Exception as _we:
                logger.warning("RectWallDetector: could not write log: %s", _we)

        # ── Header ────────────────────────────────────────────────────
        _LL.append("=" * 70 + "\n")
        _LL.append("RECT WALL DETECTOR — FULL PIPELINE LOG\n")
        _LL.append(f"  Run time  : {_ts}\n")
        _LL.append(f"  Image     : {W} x {H} px  ({total_px:,} total pixels)\n")
        _LL.append("=" * 70 + "\n")

        # ── Config dump ───────────────────────────────────────────────
        _LS("Configuration")
        _L("  U-Net checkpoint (walls)   : %s",
           getattr(cfg, "rect_unet_ckpt_path", "weights/epoch_20.pth"))
        _L("  U-Net confidence threshold : %.2f",
           getattr(cfg, "unet_confidence", 0.5))
        _L("  enclosed_overlap_threshold : %.2f",
           getattr(cfg, "rect_enclosed_overlap_threshold", 0.25))
        _L("  morph-close cap            : k=%d  iters=%d",
           getattr(cfg, "rect_cap_kernel_size", 9),
           getattr(cfg, "rect_cap_iters", 2))
        _L("  door+text filter           : enabled=%s  door_ckpt=%s",
           getattr(cfg, "enable_wall_door_text_filter", True),
           getattr(cfg, "unet_checkpoint_path", "weights/epoch_040.pth"))
        _L("  rect_max_spill             : %.2f  (%.0f%% outside-mask tolerance)",
           getattr(cfg, "rect_max_spill", 0.15),
           getattr(cfg, "rect_max_spill", 0.15) * 100)
        _L("  rect_coverage_stop         : %.2f  (stop at %.0f%% wall-px covered)",
           getattr(cfg, "rect_coverage_stop", 0.93),
           getattr(cfg, "rect_coverage_stop", 0.93) * 100)
        _L("  rect_max_overlap           : %.2f",
           getattr(cfg, "rect_max_overlap", 1.0))
        _L("  bleed_weight / initial     : %.1f / %.1f  decay=%.1f",
           getattr(cfg, "rect_bleed_weight", 15.0),
           getattr(cfg, "rect_initial_bleed_weight", 15.0),
           getattr(cfg, "rect_bleed_decay", 0.0))
        _L("  snap  gap_factor=%.2f  floor=%.1f px  max_extend=%.0f%%",
           getattr(cfg, "rect_snap_gap_factor", 0.3),
           getattr(cfg, "rect_snap_gap_floor", 5.0),
           getattr(cfg, "rect_snap_max_extend_frac", 0.10) * 100)
        _L("  OCR bboxes supplied        : %d",
           len(ocr_bboxes) if ocr_bboxes else 0)

        # ── Stage 1: Raw wall mask ─────────────────────────────────────
        _LS("Stage 1 — Raw wall mask (epoch_20 binary U-Net)")
        if wall_mask is None:
            _L("  Source : running binary U-Net (%s)",
               getattr(cfg, "rect_unet_ckpt_path", "weights/epoch_20.pth"))
            wall_mask = self._run_binary_unet(
                img_bgr, ocr_bboxes=ocr_bboxes, debug_img_path=debug_img_path)
            if wall_mask is not None:
                _n = int(np.count_nonzero(wall_mask))
                _L("  Output : %d wall px / %d  (%.1f%%)",
                   _n, total_px, 100.0 * _n / total_px)
            else:
                _L("  Output : None — U-Net failed to load or run")
        else:
            _n = int(np.count_nonzero(wall_mask))
            _L("  Source : pre-supplied wall_mask  (%d px, %.1f%%)",
               _n, 100.0 * _n / total_px)

        if wall_mask is None or int(np.count_nonzero(wall_mask)) == 0:
            if wall_mask is None:
                _L("  STOP: wall mask is None (U-Net checkpoint missing or inference error)")
            else:
                _L("  STOP: wall mask is entirely zero (U-Net found no wall pixels)")
            _L("  → returning empty DetectionResult with 0 walls")
            _write_log()
            logger.warning("RectWallDetector: empty wall mask — no walls returned")
            return DetectionResult(
                walls=[], openings=[], wall_mask=wall_mask,
                raw_wall_mask=wall_mask, outline_mask=None, source="rect_walls",
            )

        raw_wall_mask = wall_mask.copy()
        _px_raw = int(np.count_nonzero(raw_wall_mask))

        # ── Stage 1a: Structural denoise (strip AI splatter / speckle) ──
        # The raw U-Net mask carries ragged noise and isolated blobs that bleed
        # beyond the true wall structure.  Two conservative, structure-aware
        # steps clean it before any gap-bridging morphology runs:
        #
        #   1. Morphological OPEN with an image-scaled kernel — erodes thin
        #      tendrils and the narrow bridges that connect splatter to the wall
        #      body, while leaving thicker wall bodies intact.
        #   2. Connected-component area floor — drops speckle blobs (including
        #      splatter just disconnected by step 1) that are too small to be
        #      part of the wall network.
        #
        # Both default ON; kernel and area floor scale with the image so they
        # behave consistently across resolutions.  (The enclosed-space refine
        # heuristic that previously lived here was removed — Stage 1d's skeleton
        # cap pass is the authoritative enclosed-space step.)
        _LS("Stage 1a — Structural denoise (open + small-component removal)")
        if getattr(cfg, "rect_denoise_enable", True):
            _px_pre_dn = int(np.count_nonzero(wall_mask))

            _open_k = int(getattr(cfg, "rect_denoise_open_kernel", 0)) \
                or max(3, int(round(min(H, W) * 0.006)))
            if _open_k >= 2:
                _dn_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (_open_k, _open_k))

                # ── Hatch-fill probe ───────────────────────────────────
                # Walls drawn as diagonal-stroke hatching produce a mask of
                # thin parallel stripes. The OPEN below would erase them
                # wholesale ("filtering throws away major wall chunks") and
                # the leftover stripes would trip the LSD diagonal gate.
                # Probe first: if the OPEN would remove most of the mask,
                # the mask is stroke-textured, so consolidate the strokes
                # into solid wall bodies with a CLOSE before denoising.
                _probe = cv2.morphologyEx(
                    wall_mask, cv2.MORPH_OPEN, _dn_kernel, iterations=1)
                _survive = (int(np.count_nonzero(_probe))
                            / max(1, int(np.count_nonzero(wall_mask))))
                _hatch_thr = float(getattr(cfg, "rect_hatch_survival_thr", 0.45))
                if _survive < _hatch_thr:
                    _close_k = max(2 * _open_k + 1, 7)
                    _hk = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (_close_k, _close_k))
                    wall_mask = cv2.morphologyEx(
                        wall_mask, cv2.MORPH_CLOSE, _hk, iterations=2)
                    _L("  Hatch probe: only %.0f%% of mask would survive OPEN "
                       "(< %.0f%%) → stroke-textured mask; consolidated with "
                       "CLOSE k=%d ×2 before denoise",
                       _survive * 100, _hatch_thr * 100, _close_k)
                else:
                    _L("  Hatch probe: %.0f%% of mask survives OPEN — solid "
                       "mask, no consolidation needed", _survive * 100)

                wall_mask = cv2.morphologyEx(
                    wall_mask, cv2.MORPH_OPEN, _dn_kernel, iterations=1)
                _L("  MORPH_OPEN k=%d: strips tendrils/splatter thinner than ~%d px",
                   _open_k, _open_k // 2)
            else:
                _L("  MORPH_OPEN skipped (kernel < 2 px)")

            _min_area = int(getattr(cfg, "rect_denoise_min_area_px", 0)) \
                or max(1, int(round(total_px * 0.0005)))
            _nc, _lbl, _st, _ = cv2.connectedComponentsWithStats(
                (wall_mask > 0).astype(np.uint8), connectivity=8)
            _dropped = 0
            if _nc > 1 and _min_area > 0:
                _keep = np.zeros(_nc, dtype=bool)
                for _i in range(1, _nc):
                    if _st[_i, cv2.CC_STAT_AREA] >= _min_area:
                        _keep[_i] = True
                    else:
                        _dropped += 1
                wall_mask = np.where(_keep[_lbl], 255, 0).astype(np.uint8)
            _L("  Component area floor=%d px: dropped %d small blob(s)",
               _min_area, _dropped)

            _px_post_dn = int(np.count_nonzero(wall_mask))
            _L("  Before: %d px  →  After: %d px  (removed %d px, %.1f%%)",
               _px_pre_dn, _px_post_dn, _px_pre_dn - _px_post_dn,
               100.0 * (_px_pre_dn - _px_post_dn) / max(1, _px_pre_dn))
        else:
            _L("  SKIPPED — rect_denoise_enable=False in config")

        # ── Stage 1b: Morphological close ─────────────────────────────
        _LS("Stage 1b — Morphological close (bridge small gaps)")
        cap_k = max(1, int(getattr(cfg, "rect_cap_kernel_size", 9)))
        cap_i = max(1, int(getattr(cfg, "rect_cap_iters", 2)))
        if cap_k > 1 and cap_i > 0:
            _px_pre_morph = int(np.count_nonzero(wall_mask))
            kernel = np.ones((cap_k, cap_k), dtype=np.uint8)
            wall_mask = cv2.morphologyEx(
                wall_mask, cv2.MORPH_CLOSE, kernel, iterations=cap_i,
            )
            _px_post_morph = int(np.count_nonzero(wall_mask))
            _L("  MORPH_CLOSE k=%d iters=%d: %d → %d px  (Δ +%d px)",
               cap_k, cap_i, _px_pre_morph, _px_post_morph,
               _px_post_morph - _px_pre_morph)
            _L("  Effect: bridges gaps up to ~%d px wide inside wall regions", cap_k // 2)
        else:
            _L("  SKIPPED — cap_kernel_size=%d or cap_iters=%d too small", cap_k, cap_i)

        # ── Stage 1c: Door + OCR-text filter ──────────────────────────
        _LS("Stage 1c — Door + OCR-text pixel exclusion")
        if getattr(cfg, "enable_wall_door_text_filter", True):
            _px_pre_filt = int(np.count_nonzero(wall_mask))
            # Dynamic U-Net input size: average of image dimensions
            _unet_size = (H + W) // 2
            try:
                wall_mask = apply_door_text_filter(
                    wall_mask, img_bgr, ocr_bboxes,
                    door_ckpt_path  = getattr(cfg, "unet_checkpoint_path",
                                              "weights/epoch_040.pth"),
                    img_size        = _unet_size,
                    confidence      = getattr(cfg, "unet_confidence", 0.5),
                    door_margin     = getattr(cfg, "wall_filter_door_margin", 10),
                    text_margin     = getattr(cfg, "wall_filter_text_margin", 5),
                    include_windows = getattr(cfg, "wall_filter_include_windows", True),
                )
                _px_post_filt = int(np.count_nonzero(wall_mask))
                _L("  Door ckpt   : %s",
                   getattr(cfg, "unet_checkpoint_path", "weights/epoch_040.pth"))
                _L("  U-Net size  : %d px  (dynamic: (H+W)//2 = (%d+%d)//2)",
                   _unet_size, H, W)
                _L("  Margins     : door=%d px  text=%d px  include_windows=%s",
                   getattr(cfg, "wall_filter_door_margin", 10),
                   getattr(cfg, "wall_filter_text_margin", 5),
                   getattr(cfg, "wall_filter_include_windows", True))
                _L("  Before: %d px  →  After: %d px  (removed %d px)",
                   _px_pre_filt, _px_post_filt, _px_pre_filt - _px_post_filt)
                _L("  Removed pixels correspond to door/window openings and OCR text regions")
            except Exception as _fe:
                _L("  WARNING: apply_door_text_filter FAILED (%s) — step skipped", _fe)
                logger.warning("apply_door_text_filter failed (%s) — skipping", _fe)
        else:
            _L("  SKIPPED — enable_wall_door_text_filter=False in config")

        # ── Stage 1d: Skeleton cap + directional conservative fill ────────
        #
        # Four guards before any background pixel is filled as wall:
        #
        #   Guard A — pre-cap snapshot
        #     Record which background pixels were already "exterior" (touching
        #     the image border) BEFORE drawing cap lines.  A region whose pixels
        #     were <5% exterior before caps is a pre-existing room enclosure —
        #     it was already enclosed and was NOT created by the cap.  Skip it.
        #
        #   Guard B — size limit (cap_fill_max_area_px, default 5 000 px)
        #     Rooms are 5 000–100 000+ px; genuine pipe/corner gaps are tiny.
        #
        #   Guard C — OCR numeric label
        #     Any component containing the centre of a numeric OCR label is a
        #     labelled room interior — never fill.
        #
        #   Guard D — directional / smaller-side
        #     For each cap line we sample both perpendicular sides and identify
        #     which background component sits on each side.  We only fill the
        #     SMALLER of the two sides.  This ensures the algorithm always fills
        #     towards the pipe/corner interior, never towards the open room.
        #     Components that are not on the smaller side of any cap are skipped.
        _LS("Stage 1d — Skeleton pipe-end capping + directional conservative fill")
        _L("  Params: min_wall_len=30  max_pipe_width=50 px  min_pipe_r=3.0"
           "  look_ahead=80  merge_radius=6")
        _max_fill_px = int(getattr(cfg, "cap_fill_max_area_px", 5000))
        _L("  Guards: A=pre-existing-room  B=size≤%d px"
           "  C=no-OCR-label  D=smaller-side-of-cap", _max_fill_px)
        try:
            _sk_caps = find_caps(
                wall_mask,
                min_wall_len=30, max_pipe_width=50, min_pipe_r=3.0,
                look_ahead=80, local_steps=15, merge_radius=6,
            )
            _L("  Caps placed: %d", len(_sk_caps))
            for _ci, _cap in enumerate(_sk_caps):
                _cx1, _cy1 = _cap["p1"]; _cx2, _cy2 = _cap["p2"]
                _clen_i = int(math.hypot(_cx2 - _cx1, _cy2 - _cy1))
                _orient = "H" if abs(_cx2 - _cx1) >= abs(_cy2 - _cy1) else "V"
                _L("    cap[%d]  (%d,%d)→(%d,%d)  len=%d px  %s  thick=%d px",
                   _ci, _cx1, _cy1, _cx2, _cy2, _clen_i, _orient, _cap["thickness"])

            if _sk_caps:
                # ── Guard A: snapshot pre-cap exterior state ──────────────
                _pre_bg = (wall_mask == 0).astype(np.uint8)
                _pre_nc, _pre_lbl = cv2.connectedComponents(_pre_bg, connectivity=4)
                _pre_ext: set = set()
                _pre_ext.update(np.unique(_pre_lbl[0,  :]).tolist())
                _pre_ext.update(np.unique(_pre_lbl[-1, :]).tolist())
                _pre_ext.update(np.unique(_pre_lbl[:,  0]).tolist())
                _pre_ext.update(np.unique(_pre_lbl[:, -1]).tolist())
                _pre_ext.discard(0)
                _was_ext = np.isin(_pre_lbl, list(_pre_ext))
                _L("  Pre-cap interior components (rooms, will be blocked by Guard A): %d",
                   max(0, (_pre_nc - 1) - len(_pre_ext)))

                # ── Apply all cap lines ────────────────────────────────────
                wall_mask = apply_caps(wall_mask, _sk_caps)

                # ── Post-cap background components ────────────────────────
                _cap_bg = (wall_mask == 0).astype(np.uint8)
                _cap_nc, _cap_lbl = cv2.connectedComponents(_cap_bg, connectivity=4)
                _ext_lbl: set = set()
                _ext_lbl.update(np.unique(_cap_lbl[0,  :]).tolist())
                _ext_lbl.update(np.unique(_cap_lbl[-1, :]).tolist())
                _ext_lbl.update(np.unique(_cap_lbl[:,  0]).tolist())
                _ext_lbl.update(np.unique(_cap_lbl[:, -1]).tolist())
                _ext_lbl.discard(0)

                # Fast per-label area lookup via bincount
                _area_arr = np.bincount(_cap_lbl.ravel(), minlength=_cap_nc)

                # ── Guard D: for each cap, mark the smaller perpendicular side
                # p1/p2 are (col, row); _cap_lbl is indexed [row, col].
                _fillable_labels: set = set()
                for _cap in _sk_caps:
                    _ccx1, _ccy1 = _cap["p1"]   # col, row
                    _ccx2, _ccy2 = _cap["p2"]
                    _cdx  = _ccx2 - _ccx1        # col direction
                    _cdy  = _ccy2 - _ccy1        # row direction
                    _cclen = math.hypot(_cdx, _cdy)
                    if _cclen < 1:
                        continue
                    # Perpendicular unit vector (col, row)
                    _pn_col = -_cdy / _cclen
                    _pn_row =  _cdx / _cclen
                    _mid_col = (_ccx1 + _ccx2) / 2.0
                    _mid_row = (_ccy1 + _ccy2) / 2.0

                    # Sample both perpendicular sides; walk outward until we
                    # hit a background pixel (label > 0) or wall.
                    _side_lbl = [None, None]
                    for _si, _sign in enumerate((1, -1)):
                        for _step in range(2, 20):
                            _sc = int(round(_mid_col + _sign * _pn_col * _step))
                            _sr = int(round(_mid_row + _sign * _pn_row * _step))
                            if not (0 <= _sr < H and 0 <= _sc < W):
                                break
                            _lv = int(_cap_lbl[_sr, _sc])
                            if _lv > 0:           # background component found
                                _side_lbl[_si] = _lv
                                break

                    _la, _lb = _side_lbl
                    if _la is not None and _lb is not None and _la != _lb:
                        # Both sides: fill the smaller one
                        if _area_arr[_la] <= _area_arr[_lb]:
                            _fillable_labels.add(_la)
                            _L("  Guard D cap: side_a label=%d area=%d"
                               " ≤ side_b label=%d area=%d → mark A as fillable",
                               _la, _area_arr[_la], _lb, _area_arr[_lb])
                        else:
                            _fillable_labels.add(_lb)
                            _L("  Guard D cap: side_b label=%d area=%d"
                               " < side_a label=%d area=%d → mark B as fillable",
                               _lb, _area_arr[_lb], _la, _area_arr[_la])
                    elif _la is not None and _lb is None:
                        _fillable_labels.add(_la)   # one side is wall — fill bg side
                        _L("  Guard D cap: only side_a label=%d area=%d"
                           " (other side is wall) → mark as fillable",
                           _la, _area_arr[_la])
                    elif _lb is not None and _la is None:
                        _fillable_labels.add(_lb)
                        _L("  Guard D cap: only side_b label=%d area=%d"
                           " (other side is wall) → mark as fillable",
                           _lb, _area_arr[_lb])
                    else:
                        _L("  Guard D cap: both sides are wall — no fill")

                # ── Guard C: numeric OCR centre list ─────────────────────
                _ocr_num_ctrs = []
                for _ob in (ocr_bboxes or []):
                    if len(_ob) >= 5 and _is_numeric_ocr_label(str(_ob[0])):
                        _ocr_num_ctrs.append((
                            int((_ob[1] + _ob[3]) // 2),   # cx (col)
                            int((_ob[2] + _ob[4]) // 2),   # cy (row)
                        ))

                # ── Evaluate each interior component ──────────────────────
                _n_filled_total = 0
                _skip_preexist  = 0
                _skip_size      = 0
                _skip_ocr       = 0
                _skip_direction = 0

                for _lid in range(1, _cap_nc):
                    if _lid in _ext_lbl:
                        continue   # exterior — never fill

                    _area = int(_area_arr[_lid])

                    # Guard A
                    _rgn = (_cap_lbl == _lid)
                    _ext_frac = float(np.count_nonzero(_was_ext[_rgn])) / max(1, _area)
                    if _ext_frac < 0.05:
                        _skip_preexist += 1
                        _L("    SKIP label=%d area=%d ext_frac=%.1f%%"
                           " → pre-existing room (Guard A)",
                           _lid, _area, _ext_frac * 100)
                        continue

                    # Guard B
                    if _area > _max_fill_px:
                        _skip_size += 1
                        _L("    SKIP label=%d area=%d > %d px (Guard B)",
                           _lid, _area, _max_fill_px)
                        continue

                    # Guard C
                    _has_ocr = any(
                        0 <= _cy < H and 0 <= _cx < W and _rgn[_cy, _cx]
                        for _cx, _cy in _ocr_num_ctrs
                    )
                    if _has_ocr:
                        _skip_ocr += 1
                        _L("    SKIP label=%d area=%d → OCR label inside (Guard C)",
                           _lid, _area)
                        continue

                    # Guard D
                    if _lid not in _fillable_labels:
                        _skip_direction += 1
                        _L("    SKIP label=%d area=%d → not smaller side of any cap"
                           " (Guard D)", _lid, _area)
                        continue

                    # All guards passed — fill as wall
                    wall_mask[_rgn] = 255
                    _n_filled_total += _area
                    _L("    FILL  label=%d area=%d px ext_frac=%.1f%%",
                       _lid, _area, _ext_frac * 100)

                _n_post_int = (_cap_nc - 1) - len(_ext_lbl)
                _L("  Summary: post-cap interior=%d  skip A=%d B=%d C=%d D=%d"
                   "  filled_px=%d",
                   _n_post_int, _skip_preexist, _skip_size, _skip_ocr,
                   _skip_direction, _n_filled_total)
            else:
                _L("  No caps placed — no qualifying open pipe ends found")
                _L("  Reasons: skeleton path < min_wall_len=30, no opposing wall"
                   " within 50 px, or gap not ≥60%% black pixels")
        except Exception as _sce:
            _L("  WARNING: skeleton cap step FAILED (%s) — step skipped", _sce)
            logger.warning("Skeleton cap step failed (%s) — skipping", _sce)

        _px_final_mask = int(np.count_nonzero(wall_mask))
        _L("  Final wall mask: %d px  (%.1f%% of image;  raw U-Net was %d px)",
           _px_final_mask, 100.0 * _px_final_mask / total_px, _px_raw)

        # ── Stage 2: Diagonal-presence test ───────────────────────────
        _LS("Stage 2 — Diagonal-presence test (LSD on wall mask)")
        axis_tol  = getattr(cfg, "rect_axis_tol_deg",     DEFAULT_AXIS_TOL_DEG)
        diag_thr  = getattr(cfg, "rect_diag_lsd_ratio",   DEFAULT_DIAG_LSD_RATIO)
        diag_minl = getattr(cfg, "rect_diag_lsd_min_len", DEFAULT_DIAG_LSD_MIN_LEN)
        # The LSD test uses its OWN, more permissive axis tolerance (not the
        # tight classification tol): a line within rect_diag_lsd_axis_tol of an
        # axis is near-axis wobble that Stage 4a straightens, so it must not be
        # counted as diagonal evidence and flip the whole plan into the costly
        # (and artefact-prone) diagonal decomposition path.
        diag_axis_tol = getattr(cfg, "rect_diag_lsd_axis_tol",
                                DEFAULT_DIAG_LSD_AXIS_TOL)
        _L("  diag_axis_tol=%.1f°  diag_ratio_thr=%.1f%%  min_segment_len=%.0f px"
           "  (classification axis_tol=%.1f°)",
           diag_axis_tol, diag_thr * 100, diag_minl, axis_tol)
        has_diagonals = detect_has_diagonals(
            wall_mask,
            axis_tol_deg=diag_axis_tol, diag_ratio_thr=diag_thr,
            min_segment_len=diag_minl,
        )
        force_axis_only = not has_diagonals
        _L("  Result: diagonals_present=%s  →  axis_only=%s",
           has_diagonals, force_axis_only)
        if force_axis_only:
            _L("  Decomposer restricted to 0°/90° only (no diagonal rectangles)")
        else:
            _L("  Decomposer will try all %d angle steps (diagonal walls detected)",
               getattr(cfg, "rect_angle_steps", 12))

        # ── Stage 3: Rect decomposition ───────────────────────────────
        _LS("Stage 3 — Rectangle decomposition (rd.decompose)")
        mask01        = (wall_mask > 0).astype(np.uint8)
        total_wall_px = int(np.count_nonzero(mask01))

        _p_angle_steps = getattr(cfg, "rect_angle_steps",          12)
        _p_penalty     = getattr(cfg, "rect_penalty",              0.0)
        _p_max_grid    = getattr(cfg, "rect_max_grid_dim",         200)
        _p_cov_stop    = getattr(cfg, "rect_coverage_stop",        0.93)
        _p_bleed_w     = getattr(cfg, "rect_bleed_weight",         15.0)
        _p_init_bleed  = getattr(cfg, "rect_initial_bleed_weight", 15.0)
        _p_bleed_decay = getattr(cfg, "rect_bleed_decay",          0.0)
        _p_axis_gap    = getattr(cfg, "rect_axis_gap",             0.0)
        _p_max_overlap = getattr(cfg, "rect_max_overlap",          1.0)
        _p_max_spill   = getattr(cfg, "rect_max_spill",            0.15)
        _p_diag_tol    = getattr(cfg, "rect_diag_axis_tol",        8.0)
        _p_diag_spen   = getattr(cfg, "rect_diag_score_penalty",   0.4)
        _p_diag_open   = getattr(cfg, "rect_diag_overlap_penalty", 6.0)
        _p_rescue      = getattr(cfg, "rect_thin_rescue",          True)
        _p_rescue_thk  = getattr(cfg, "rect_thin_rescue_max_thickness", 20.0)
        _p_rescue_len  = getattr(cfg, "rect_thin_rescue_min_len",  25.0)
        _p_rescue_area = getattr(cfg, "rect_thin_rescue_min_area", 40)
        _p_rescue_frg  = getattr(cfg, "rect_thin_rescue_fringe_px", 2)

        _L("  Wall pixels entering decompose: %d / %d  (%.2f%%)",
           total_wall_px, total_px, 100.0 * total_wall_px / max(1, total_px))
        _L("  coverage_stop      = %.2f → stop when ≥%.0f%% of wall px covered",
           _p_cov_stop, _p_cov_stop * 100)
        _L("  max_spill_fraction = %.2f → reject rect if >%.0f%% pixels outside mask",
           _p_max_spill, _p_max_spill * 100)
        _L("  max_overlap        = %.2f → reject rect if >%.0f%% already covered",
           _p_max_overlap, _p_max_overlap * 100)
        _L("  bleed_weight=%.1f  initial=%.1f  decay=%.1f",
           _p_bleed_w, _p_init_bleed, _p_bleed_decay)
        _L("  rect_penalty=%.2f  angle_steps=%d  axis_only=%s  axis_gap=%.1f"
           "  max_grid_dim=%d",
           _p_penalty, _p_angle_steps, force_axis_only, _p_axis_gap, _p_max_grid)
        if not force_axis_only:
            _L("  diagonal penalties : axis_tol=%.1f°  score_penalty=%.2f"
               "  overlap_penalty=%.1f/px (demote thin diagonals over walls)",
               _p_diag_tol, _p_diag_spen, _p_diag_open)
        if _p_rescue:
            _L("  thin rescue        = ON  (max_thickness=%.0fpx  min_len=%.0fpx"
               "  min_area=%dpx²  fringe=%dpx) — full-res re-search of the"
               " abandoned residual", _p_rescue_thk, _p_rescue_len,
               _p_rescue_area, _p_rescue_frg)
        else:
            _L("  thin rescue        = OFF (rect_thin_rescue=False)")
        _LL.append("\n--- rd.decompose verbose output (start) ---\n")

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
                diag_axis_tol        = _p_diag_tol,
                diag_score_penalty   = _p_diag_spen,
                diag_overlap_penalty = _p_diag_open,
                refine               = True,
                verbose              = True,
                thin_rescue          = _p_rescue,
                thin_rescue_max_thickness = _p_rescue_thk,
                thin_rescue_min_len  = _p_rescue_len,
                thin_rescue_min_area = _p_rescue_area,
                thin_rescue_fringe_px = _p_rescue_frg,
            )
        _verbose_text = _verbose_buf.getvalue()
        _LL.append(_verbose_text if _verbose_text
                   else "(no verbose output from rd.decompose)\n")
        _LL.append("--- rd.decompose verbose output (end) ---\n\n")

        # Coverage analysis + stop-reason explanation
        _covered = np.zeros((H, W), dtype=np.uint8)
        for _r in rects:
            cv2.fillPoly(_covered, [_r.corners().astype(np.int32)], 1)
        _covered_wall_px  = int(np.count_nonzero((_covered > 0) & (mask01 > 0)))
        _covered_total_px = int(np.count_nonzero(_covered > 0))
        _coverage_frac    = _covered_wall_px / max(1, total_wall_px)
        _spill_px         = _covered_total_px - _covered_wall_px

        _L("  Rectangles placed      : %d", len(rects))
        _L("  Wall px covered        : %d / %d  (%.2f%%)",
           _covered_wall_px, total_wall_px, _coverage_frac * 100)
        _L("  Target coverage        : %.1f%%", _p_cov_stop * 100)
        _L("  Uncovered wall px      : %d  (%.2f pp below target)",
           total_wall_px - _covered_wall_px,
           max(0.0, _p_cov_stop - _coverage_frac) * 100)
        _L("  Spill px (outside mask): %d  (%.1f%% of placed rect area)",
           _spill_px, 100.0 * _spill_px / max(1, _covered_total_px))

        if _coverage_frac >= _p_cov_stop:
            _L("  STOP REASON: Coverage target reached"
               " (%.2f%% ≥ %.1f%%) — decomposition complete",
               _coverage_frac * 100, _p_cov_stop * 100)
        elif total_wall_px == 0:
            _L("  STOP REASON: Wall mask was empty before decompose")
        else:
            _uncov = total_wall_px - _covered_wall_px
            _L("  STOP REASON: No more valid rectangles could be placed")
            _L("    ~%d wall px remain uncovered but every candidate rect was rejected:", _uncov)
            _L("    • Exceeded max_spill_fraction=%.2f"
               " (>%.0f%% of rect pixels landed outside the wall mask)",
               _p_max_spill, _p_max_spill * 100)
            _L("    • OR exceeded max_overlap=%.2f"
               " (>%.0f%% of rect area was already covered)", _p_max_overlap,
               _p_max_overlap * 100)
            _L("    • OR scored negative after bleed penalty (weight=%.1f)", _p_bleed_w)
            _L("    Possible remedies: raise rect_coverage_stop, lower"
               " rect_max_spill, or improve the wall mask quality")

        if rects:
            _areas = []
            for _r in rects:
                _c = _r.corners(); _xs, _ys = _c[:, 0], _c[:, 1]
                _areas.append(float((_xs.max() - _xs.min()) * (_ys.max() - _ys.min())))
            _L("  Per-rect AABB area (px²): min=%.0f  max=%.0f  mean=%.0f",
               min(_areas), max(_areas), sum(_areas) / len(_areas))
            _LL.append("\n  All rects (AABB x1,y1–x2,y2 | angle° | AABB area):\n")
            for _idx, _r in enumerate(rects):
                _c = _r.corners(); _xs, _ys = _c[:, 0], _c[:, 1]
                _ang = getattr(_r, "angle", 0.0)
                _a   = (_xs.max() - _xs.min()) * (_ys.max() - _ys.min())
                _LL.append(
                    f"    rect[{_idx:3d}]: "
                    f"({_xs.min():.0f},{_ys.min():.0f})–"
                    f"({_xs.max():.0f},{_ys.max():.0f})"
                    f"  angle={_ang:.1f}°  area={_a:.0f}\n"
                )

        if debug_img_path:
            try:
                _wall_bin = mask01 > 0; _rect_bin = _covered > 0
                _diag = np.zeros((H, W, 3), dtype=np.uint8)
                _diag[_wall_bin & _rect_bin]  = (255, 255, 255)  # white  = hit
                _diag[_wall_bin & ~_rect_bin] = (0,   255,   0)  # green  = missed wall
                _diag[_rect_bin & ~_wall_bin] = (0,     0, 255)  # red    = spill
                _diag_path = os.path.splitext(debug_img_path)[0] + "_wall_rect_diag.png"
                cv2.imwrite(_diag_path, _diag)
                _L("  Diagnostic overlay → %s  (white=hit, green=missed, red=spill)",
                   _diag_path)
            except Exception as _de:
                _L("  WARNING: could not save diagnostic image: %s", _de)

        # ── Stage 4: Snap endpoints ────────────────────────────────────
        _LS("Stage 4 — Endpoint snapping (corner/T-junction only)")
        infos: List[_RectInfo] = []
        for r in rects:
            p1, p2, thick, wall_ang = rect_centerline(r)
            infos.append(_RectInfo(p1=p1, p2=p2, thickness=thick,
                                   angle_deg=wall_ang, rect=r))

        # ── Stage 4a: Near-axis straightening ─────────────────────────
        # BEFORE any snapping: force walls tilted by less than
        # rect_straighten_max_dev_deg (default 7°) to exact 0°/90°,
        # unconditionally and regardless of the resulting coverage change.
        # A wall the architect drew straight but the decomposer placed a few
        # degrees off would otherwise be classified is_diagonal downstream,
        # which makes parent-wall finding / opening placement reject it and
        # silently drops the doors and windows sitting on it.
        _straighten_dev = getattr(cfg, "rect_straighten_max_dev_deg", 7.0)
        if _straighten_dev > 0.0:
            _n_straightened = _straighten_near_axis_walls(
                infos, max_dev_deg=_straighten_dev,
            )
            _L("  Stage 4a — Near-axis straightening (<%.1f° → axis): "
               "%d wall(s) straightened (coverage change ignored)",
               _straighten_dev, _n_straightened)
        else:
            _L("  Stage 4a — Near-axis straightening SKIPPED "
               "(rect_straighten_max_dev_deg=%.1f)", _straighten_dev)

        _snap_gf  = getattr(cfg, "rect_snap_gap_factor",     DEFAULT_SNAP_GAP_FACTOR)
        _snap_fl  = getattr(cfg, "rect_snap_gap_floor",       DEFAULT_SNAP_GAP_FLOOR)
        _snap_ang = getattr(cfg, "rect_snap_angle_deg",       DEFAULT_SNAP_ANGLE_DEG)
        _snap_ext = getattr(cfg, "rect_snap_max_extend_frac", 0.10)
        _L("  Parallel walls (angle_diff ≤ %.1f°) → NEVER merged", _snap_ang)
        _L("  Gap threshold: max(%.1f px, %.2f × thickness)", _snap_fl, _snap_gf)
        _L("  Max extension per wall: %.0f%% of its own length", _snap_ext * 100)

        _before_pts = [(tuple(i.p1), tuple(i.p2)) for i in infos]
        snap_collinear_endpoints(
            infos,
            snap_gap_factor = _snap_gf,
            snap_gap_floor  = _snap_fl,
            snap_angle_deg  = _snap_ang,
            max_extend_frac = _snap_ext,
        )
        _after_pts = [(tuple(i.p1), tuple(i.p2)) for i in infos]
        _snapped = sum(1 for b, a in zip(_before_pts, _after_pts) if b != a)
        _L("  Walls with ≥1 endpoint moved: %d / %d", _snapped, len(infos))
        if _snapped == 0:
            _L("  No snapping — all endpoint gaps exceeded threshold or"
               " only parallel walls were close enough")

        # ── Stage 4b: Minor-diagonal suppression ──────────────────────
        # Snap small, isolated diagonal walls to the nearest axis when the plan
        # is mostly orthogonal and their junction neighbours are all axis-
        # aligned.  Only fires in the mixed case (real diagonals elsewhere +
        # a small spurious one locally) — on purely-orthogonal plans Stage 2
        # already forced axis_only so no diagonals exist to suppress.
        _LS("Stage 4b — Minor-diagonal suppression")
        if getattr(cfg, "rect_suppress_minor_diagonals", True):
            _n_diag_before = sum(
                1 for i in infos
                if min(i.angle_deg % 90.0, 90.0 - (i.angle_deg % 90.0)) > axis_tol
            )
            _n_diag_snapped = _suppress_minor_diagonals(
                infos,
                axis_tol_deg=axis_tol,
                short_frac=getattr(cfg, "rect_minor_diag_short_frac", 0.5),
                max_diag_frac=getattr(cfg, "rect_minor_diag_max_frac", 0.34),
            )
            _L("  Diagonal walls: %d present → %d snapped to axis",
               _n_diag_before, _n_diag_snapped)
            if _n_diag_snapped:
                # Re-run corner/T-junction snapping so the newly axis-aligned
                # walls still close the perimeter at their junctions.
                snap_collinear_endpoints(
                    infos,
                    snap_gap_factor=_snap_gf,
                    snap_gap_floor=_snap_fl,
                    snap_angle_deg=_snap_ang,
                    max_extend_frac=_snap_ext,
                )
                _L("  Re-ran endpoint snapping after diagonal suppression")
        else:
            _L("  SKIPPED — rect_suppress_minor_diagonals=False in config")

        # ── Stage 5: WallSegments + output ────────────────────────────
        _LS("Stage 5 — WallSegment conversion + output")
        walls = [_rect_to_wall_segment(info, idx, axis_tol_deg=axis_tol)
                 for idx, info in enumerate(infos)]
        _n_diag = sum(1 for w in walls if w.is_diagonal)
        _L("  Total wall segments : %d  (%d axis-aligned, %d diagonal)",
           len(walls), len(walls) - _n_diag, _n_diag)
        if walls:
            _thk = [w.thickness for w in walls]
            _L("  Thickness (px)      : min=%.1f  max=%.1f  mean=%.1f",
               min(_thk), max(_thk), sum(_thk) / len(_thk))
            _lens = [
                _dist(tuple(w.outline[0]), tuple(w.outline[1]))
                for w in walls if w.outline and len(w.outline) >= 2
            ]
            if _lens:
                _L("  Length (px)         : min=%.1f  max=%.1f  mean=%.1f",
                   min(_lens), max(_lens), sum(_lens) / len(_lens))

        outline_mask       = rasterise_wall_segments(walls, (H, W))
        wall_rect_polygons = [r.corners().astype(np.float32) for r in rects]

        # ── Summary ───────────────────────────────────────────────────
        _LS("Pipeline summary")
        _L("  Image                  : %d × %d px", W, H)
        _L("  Raw U-Net wall px      : %d  (%.1f%%)",
           _px_raw, 100.0 * _px_raw / total_px)
        _L("  Final filtered mask px : %d  (%.1f%%)",
           _px_final_mask, 100.0 * _px_final_mask / total_px)
        _L("  Rects from decompose   : %d  (coverage %.2f%% / target %.1f%%)",
           len(rects), _coverage_frac * 100, _p_cov_stop * 100)
        _L("  Wall segments output   : %d  (%d snapped)", len(walls), _snapped)
        _LL.append("=" * 70 + "\n")

        _write_log()

        logger.info(
            "RectWallDetector: %d walls, coverage=%.2f%%/target %.1f%%"
            "  axis_only=%s  diagonals=%s",
            len(walls), _coverage_frac * 100, _p_cov_stop * 100,
            force_axis_only, has_diagonals,
        )

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
    # Debug: softmax confidence heatmap (RdYlGn), matching the Colab viewer
    # ------------------------------------------------------------------
    @staticmethod
    def _save_conf_heatmap(
        conf_np: np.ndarray,
        orig_hw: Tuple[int, int],
        debug_img_path: str,
    ) -> None:
        """Render the U-Net softmax confidence map as a RdYlGn heatmap PNG.

        Equivalent to the Colab ``imshow(conf, cmap='RdYlGn', vmin=0, vmax=1)``
        panel.  Green = high confidence, red = low.  Saved next to the other
        wall debug images as ``<stem>_wall_conf_heatmap.png``.
        """
        h_orig, w_orig = orig_hw

        # Upsample the model-resolution confidence map to original resolution.
        conf_full = cv2.resize(
            conf_np.astype(np.float32), (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR,
        )
        conf_u8 = np.clip(conf_full * 255.0, 0, 255).astype(np.uint8)

        # RdYlGn look-up table (5 matplotlib anchors, RGB) → 256-entry BGR LUT.
        anchors = np.array([
            [165,   0,  38],   # 0.00 dark red
            [244, 109,  67],   # 0.25 orange
            [255, 255, 191],   # 0.50 pale yellow
            [166, 217, 106],   # 0.75 light green
            [  0, 104,  55],   # 1.00 dark green
        ], dtype=np.float32)
        xs  = np.linspace(0.0, 1.0, len(anchors))
        pos = np.linspace(0.0, 1.0, 256)
        lut_rgb = np.stack(
            [np.interp(pos, xs, anchors[:, c]) for c in range(3)], axis=1)
        lut_bgr = lut_rgb[:, ::-1].astype(np.uint8).reshape(256, 1, 3)

        heat = cv2.applyColorMap(conf_u8, lut_bgr)

        # Colourbar strip on the right so the scale reads without a viewer.
        bar_w = max(18, w_orig // 40)
        ramp  = np.linspace(255, 0, h_orig).astype(np.uint8).reshape(h_orig, 1)
        ramp  = np.repeat(ramp, bar_w, axis=1)
        bar   = cv2.applyColorMap(ramp, lut_bgr)
        cv2.putText(bar, "1.0", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, "0.0", (2, h_orig - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
        out = np.hstack([heat, bar])

        heat_path = os.path.splitext(debug_img_path)[0] + "_wall_conf_heatmap.png"
        cv2.imwrite(heat_path, out)
        logger.info("U-Net confidence heatmap → %s", heat_path)

    # ------------------------------------------------------------------
    # U-Net (binary, epoch_20.pth)
    # ------------------------------------------------------------------
    def _run_binary_unet(
        self, img_bgr: np.ndarray, ocr_bboxes: Optional[List] = None,
        debug_img_path: Optional[str] = None,
    ) -> Optional[np.ndarray]:
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
            from detection.unet_inference import build_model_from_checkpoint, get_val_transform
        except Exception as exc:
            logger.warning("Binary U-Net dependencies unavailable: %s", exc)
            return None

        # ── Get image dimensions for dynamic U-Net sizing ─────────────────
        h_orig, w_orig = img_bgr.shape[:2]
        # Dynamic size: average of image dimensions instead of fixed 1024
        size       = (h_orig + w_orig) // 2
        _size_src  = "dynamic (H+W)//2"
        # Optional: fixed training-size resize (Colab val_transform protocol).
        # Resize every image to the exact square the binary model was trained
        # at (rect_unet_img_size), so the wall stroke lands back in the model's
        # trained thickness distribution.  Helps compact plans (input/12) whose
        # walls render proportionally thick under the dynamic (H+W)//2 size.
        # Takes precedence over wall-thickness-norm.  Gated off by default.
        if getattr(self.config, "enable_unet_fixed_size", False):
            size      = int(getattr(self.config, "rect_unet_img_size", 1024))
            _size_src = "fixed (rect_unet_img_size)"
            logger.info(
                "U-Net fixed-size resize: %d px (native (%d+%d)//2=%d)",
                size, h_orig, w_orig, (h_orig + w_orig) // 2,
            )
        # Optional: normalize the input dimension so the dominant wall stroke
        # lands at ~unet_target_wall_px in the resized tensor (compact plans
        # render walls proportionally thick).  Gated off by default.
        elif getattr(self.config, "enable_wall_thickness_norm", False):
            try:
                from core.scale_norm import unet_size_for_wall_thickness
                _norm_size, _wt, _src = unet_size_for_wall_thickness(
                    img_bgr,
                    target_wall_px=getattr(self.config,
                                           "unet_target_wall_px", 30.0),
                    ocr_bboxes=ocr_bboxes,
                    wall_to_glyph=getattr(self.config,
                                          "wall_to_glyph_ratio", 1.8),
                    size_min=getattr(self.config, "unet_size_min", 512),
                )
                logger.info(
                    "Wall-thickness norm: wt~%s px (%s) → U-Net size %d "
                    "(was %d)",
                    f"{_wt:.1f}" if _wt else "n/a", _src, _norm_size, size,
                )
                size = _norm_size
                _size_src = f"wall-thickness-norm ({_src})"
            except Exception as _wtn_exc:
                logger.warning(
                    "Wall-thickness norm failed (%s) — using dynamic size",
                    _wtn_exc,
                )
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

            # Debug: save the softmax confidence map as a RdYlGn heatmap,
            # matching the Colab inference viewer (imshow(conf, cmap='RdYlGn',
            # vmin=0, vmax=1)).  Only in debug runs (debug_img_path set).
            if debug_img_path:
                try:
                    self._save_conf_heatmap(
                        conf_np, (h_orig, w_orig), debug_img_path)
                except Exception as _he:
                    logger.warning("Could not save U-Net conf heatmap: %s", _he)

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
            "Binary U-Net (%s): %.1f%% wall pixels  (conf≥%.2f, %d classes, "
            "input_size=%d px [%s], native (%d+%d)//2=%d)",
            ckpt_path.name,
            100.0 * int(np.count_nonzero(wall_mask)) / wall_mask.size,
            confidence, _num_classes, size, _size_src,
            h_orig, w_orig, (h_orig + w_orig) // 2,
        )
        return wall_mask
