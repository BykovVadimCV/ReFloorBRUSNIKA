"""Approximate a binary mask with a minimal set of overlapping rotated rectangles."""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Live progress display
# ---------------------------------------------------------------------------

def _progress(step: int, n_rects: int, covered: int, total: int,
               min_gain: int, detail: str = "") -> None:
    frac = covered / max(total, 1)
    w    = 16
    bar  = "█" * int(frac * w) + "░" * (w - int(frac * w))
    sys.stdout.write(f"\r#{step} [{bar}] {100*frac:4.1f}% r={n_rects} thr={min_gain}  {detail:<24}")
    sys.stdout.flush()

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    cx: float
    cy: float
    w:  float
    h:  float
    angle: float   # degrees, counter-clockwise

    def corners(self) -> np.ndarray:
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        hw, hh = self.w / 2, self.h / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
        rot = np.array([[cos, -sin], [sin, cos]])
        return local @ rot.T + np.array([self.cx, self.cy])

    def __str__(self):
        return (f"Rect(center=({self.cx:.1f},{self.cy:.1f})  "
                f"size=({self.w:.1f}x{self.h:.1f})  angle={self.angle:.1f}°)")


# ---------------------------------------------------------------------------
# Rasterise a Rect onto a (H, W) grid
# ---------------------------------------------------------------------------

def rasterise(rect: Rect, H: int, W: int) -> np.ndarray:
    """Boolean (H, W) mask: True where pixel centres lie inside rect."""
    rad = math.radians(-rect.angle)
    cos, sin = math.cos(rad), math.sin(rad)
    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs - rect.cx
    dy = ys - rect.cy
    lx =  dx * cos - dy * sin
    ly =  dx * sin + dy * cos
    return (np.abs(lx) <= rect.w / 2) & (np.abs(ly) <= rect.h / 2)


# ---------------------------------------------------------------------------
# Build a 2-D weight grid in the rotated coordinate frame
# ---------------------------------------------------------------------------

def build_weight_grid(
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    angle: float,
    max_grid_dim: int = 200,
    bleed_weight: float = 1.0,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Project every pixel into the coordinate frame rotated by `angle` degrees
    and accumulate weights:
        +1             for pixels that are still uncovered (in `remaining`)
        -bleed_weight  for pixels that are outside the original mask
         0             for pixels already covered (in mask but not remaining)

    Returns
    -------
    grid       : 2-D int32 array of shape (grid_h, grid_w)
    lx_min_f   : un-rotated x origin (float, in rotated-frame px)
    ly_min_f   : un-rotated y origin (float, in rotated-frame px)
    scale      : factor applied to rotated coords → grid indices
                 (grid index i  ↔  rotated coord  i / scale + origin)
    """
    H, W = remaining.shape
    rad  = math.radians(-angle)
    cos, sin = math.cos(rad), math.sin(rad)

    ys, xs = np.mgrid[0:H, 0:W]
    lx_f = (xs * cos - ys * sin).ravel()
    ly_f = (xs * sin + ys * cos).ravel()

    lx_min_f, lx_max_f = float(lx_f.min()), float(lx_f.max())
    ly_min_f, ly_max_f = float(ly_f.min()), float(ly_f.max())

    span  = max(lx_max_f - lx_min_f, ly_max_f - ly_min_f, 1.0)
    scale = min(1.0, max_grid_dim / span)

    lx_i = np.round((lx_f - lx_min_f) * scale).astype(np.int32)
    ly_i = np.round((ly_f - ly_min_f) * scale).astype(np.int32)

    grid_w = int(lx_i.max()) + 1
    grid_h = int(ly_i.max()) + 1

    weights  = remaining.ravel().astype(np.float32) - bleed_weight * outside_mask.ravel().astype(np.float32)
    flat_idx = ly_i * grid_w + lx_i

    grid_flat = np.bincount(flat_idx, weights=weights.astype(float),
                            minlength=grid_h * grid_w)
    grid = grid_flat.reshape(grid_h, grid_w).astype(np.int32)

    return grid, lx_min_f, ly_min_f, scale


# ---------------------------------------------------------------------------
# Vectorised Kadane's algorithm on a 1-D array
# ---------------------------------------------------------------------------

def kadane(arr: np.ndarray) -> Tuple[int, int, int]:
    """
    Maximum-sum contiguous subarray of `arr`.

    Uses the O(n) cumulative-sum trick:
        best gain ending at index j  =  cumsum[j+1] - min(cumsum[0 .. j])

    Returns (max_sum, start, end) — inclusive, 0-indexed.
    Returns (0, 0, 0) if every subarray sum is ≤ 0.
    """
    cs  = np.concatenate([[0], np.cumsum(arr)])
    min_prefix = np.minimum.accumulate(cs[:-1])   # running min of prefix sums
    gain = cs[1:] - min_prefix
    j    = int(np.argmax(gain))
    val  = int(gain[j])
    if val <= 0:
        return 0, 0, 0
    i = int(np.argmin(cs[: j + 1]))
    return val, i, j


# ---------------------------------------------------------------------------
# Maximum-weight subrectangle — fully vectorised, no Python row loop
# ---------------------------------------------------------------------------

def max_weight_subrect(grid: np.ndarray) -> Tuple[int, int, int, int, int]:
    """
    Find the axis-aligned rectangle in `grid` with the largest element sum.

    Vectorised approach — eliminates the Python-level double loop entirely:

    1. Build all K = rows*(rows+1)/2 column sums at once using row prefix
       sums.  Each column sum is a length-cols vector; together they form a
       (K, cols) matrix.

    2. Run Kadane's algorithm on the entire matrix in one shot:
       cumsum along axis=1, running minimum of the prefix, then argmax of
       the gain.  Every operation is a single numpy call.

    3. Recover row/column indices only for the single best entry.

    Memory: O(K × cols) ≈ 23 MB for a 200×200 grid — fine.
    Speed:  ~6 numpy calls regardless of grid size vs. K Python iterations
            in the loop-based version.

    Returns (best_sum, r1, c1, r2, c2) — all inclusive, 0-indexed.
    """
    rows, cols = grid.shape

    # ---- Step 1: all (r1, r2) column sums --------------------------------
    # row_cs[r, c] = grid[0..r, c].sum()
    row_cs = np.cumsum(grid, axis=0, dtype=np.float32)   # (rows, cols)

    # Enumerate every (r1, r2) pair with r2 >= r1
    r1_idx = np.array([r1 for r1 in range(rows) for _ in range(r1, rows)],
                      dtype=np.int32)                     # (K,)
    r2_idx = np.array([r2 for r1 in range(rows) for r2 in range(r1, rows)],
                      dtype=np.int32)                     # (K,)
    K = len(r1_idx)

    col_sums = row_cs[r2_idx].copy()                     # (K, cols)
    has_top  = r1_idx > 0
    col_sums[has_top] -= row_cs[r1_idx[has_top] - 1]    # subtract prefix above r1

    # ---- Step 2: batch Kadane on (K, cols) matrix ------------------------
    # Prefix sums with a leading zero column
    cs = np.empty((K, cols + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(col_sums, axis=1, out=cs[:, 1:])           # (K, cols+1)

    min_prefix = np.minimum.accumulate(cs[:, :-1], axis=1)   # (K, cols)
    gain       = cs[:, 1:] - min_prefix                       # (K, cols)

    best_j     = np.argmax(gain, axis=1)                      # (K,)
    best_val_k = gain[np.arange(K), best_j]                   # (K,)

    best_k   = int(np.argmax(best_val_k))
    best_val = int(best_val_k[best_k])

    if best_val <= 0:
        return 0, 0, 0, 0, 0

    # ---- Step 3: recover indices for the single best entry ---------------
    r1 = int(r1_idx[best_k])
    r2 = int(r2_idx[best_k])
    c2 = int(best_j[best_k])
    c1 = int(np.argmin(cs[best_k, : c2 + 1]))

    return best_val, r1, c1, r2, c2


# ---------------------------------------------------------------------------
# Best rectangle for a single angle
# ---------------------------------------------------------------------------

def best_rect_at_angle(
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    angle: float,
    max_grid_dim: int = 200,
    bleed_weight: float = 1.0,
) -> Tuple[Optional[Rect], int]:
    """
    Return the highest-scoring Rect at `angle` and its pixel score,
    or (None, 0) if no positive-score rect exists.
    """
    grid, lx_min_f, ly_min_f, scale = build_weight_grid(
        remaining, outside_mask, angle, max_grid_dim, bleed_weight)
    result = max_weight_subrect(grid)
    val, r1, c1, r2, c2 = result

    if val <= 0:
        return None, 0

    # Grid indices → rotated-frame coordinates (un-scale + un-offset)
    cx_l = (c1 + c2) / 2.0 / scale + lx_min_f
    cy_l = (r1 + r2) / 2.0 / scale + ly_min_f

    # Rotated frame → image space
    rad_inv = math.radians(angle)
    ci, si  = math.cos(rad_inv), math.sin(rad_inv)
    cx = cx_l * ci - cy_l * si
    cy = cx_l * si + cy_l * ci

    w = (c2 - c1 + 1) / scale
    h = (r2 - r1 + 1) / scale
    return Rect(cx, cy, float(w), float(h), angle), val


# ---------------------------------------------------------------------------
# PCA angle of the remaining mask
# ---------------------------------------------------------------------------

def pca_angle(mask: np.ndarray) -> float:
    """
    Principal-axis direction of ON-pixels, in degrees ∈ [0°, 180°).
    Falls back to 0° when fewer than 2 pixels are present.
    """
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0
    dx = xs - xs.mean()
    dy = ys - ys.mean()
    cov = np.array([[np.dot(dx, dx), np.dot(dx, dy)],
                    [np.dot(dx, dy), np.dot(dy, dy)]])
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, 1]   # eigenvector for the largest eigenvalue
    return math.degrees(math.atan2(v[1], v[0])) % 180


# ---------------------------------------------------------------------------
# Angle schedule: uniform base + fine grid around the PCA axis
# ---------------------------------------------------------------------------

# Angles that are always included and receive an exclusion gap around them
# so the schedule snaps to these "preferred" directions instead of drifting
# to near-axis angles.
_PREFERRED_ANGLES = [0.0, 45.0, 90.0, 135.0]


def _near_preferred(angle: float, gap: float) -> bool:
    """True if `angle` is within `gap` degrees of any preferred angle (mod 180)."""
    a = angle % 180
    for p in _PREFERRED_ANGLES:
        diff = abs(a - p)
        diff = min(diff, 180.0 - diff)
        if diff < gap:
            return True
    return False


def build_angle_list(
    remaining: np.ndarray,
    angle_steps: int,
    n_pca_fine: int = 8,
    pca_spread: float = 15.0,
    axis_only: bool = False,
    axis_gap: float = 0.0,
) -> List[float]:
    """
    Combine `angle_steps` uniformly-spaced angles in [0°, 180°) with a
    fine grid of `n_pca_fine` angles within ±pca_spread° of the PCA axis.
    Duplicates are removed (rounded to 2 decimal places).

    When `axis_only` is True, return only [0°, 90°] regardless of other
    parameters — no diagonal or PCA-guided angles are considered.

    When `axis_gap` > 0, any generated angle that falls within `axis_gap`
    degrees of a preferred angle (0°, 45°, 90°, 135°) is removed from the
    schedule, and the preferred angles themselves are always injected.
    This creates dead-zones around the special directions so that near-axis
    or near-diagonal features are snapped to them rather than covered by a
    slightly-off angle.
    """
    if axis_only:
        return [0.0, 90.0]

    uniform = [i * 180.0 / angle_steps for i in range(angle_steps)]
    pca     = pca_angle(remaining)
    fine    = [
        pca + pca_spread * (k / (n_pca_fine - 1) - 0.5)
        for k in range(n_pca_fine)
    ]

    candidates = list(uniform) + list(fine)

    if axis_gap > 0.0:
        # Drop anything in the dead-zone, then add the preferred angles back.
        candidates = [a for a in candidates if not _near_preferred(a, axis_gap)]
        candidates += [p for p in _PREFERRED_ANGLES]

    return sorted({round(a % 180, 2) for a in candidates})


# ---------------------------------------------------------------------------
# Coordinate-descent refinement of a placed rectangle
# ---------------------------------------------------------------------------

def refine_rect(
    rect: Rect,
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    H: int,
    W: int,
    iters: int = 40,
    bleed_weight: float = 1.0,
    axis_only: bool = False,
) -> Tuple[Rect, int]:
    """
    Hill-climb (cx, cy, w, h, angle) to maximise

        score = #(pixels ∩ remaining) − #(pixels ∩ outside_mask)

    Uses a doubling/halving step schedule so it explores broadly first
    then converges to sub-pixel precision.

    When `axis_only` is True, the angle is locked and not refined — the
    rectangle stays at exactly 0° or 90°.
    """
    def sc(r: Rect) -> float:
        p = rasterise(r, H, W)
        return (int(np.count_nonzero(p & remaining))
                - bleed_weight * int(np.count_nonzero(p & outside_mask)))

    best   = rect
    best_s = sc(rect)
    # When axis_only, omit 'angle' so refinement never drifts off-axis.
    steps  = {'cx': 2.0, 'cy': 2.0, 'w': 4.0, 'h': 4.0}
    if not axis_only:
        steps['angle'] = 5.0

    for _ in range(iters):
        improved = False
        for attr, delta in list(steps.items()):
            for sign in (+1, -1):
                new_val = getattr(best, attr) + sign * delta
                kwargs  = dict(cx=best.cx, cy=best.cy,
                               w=best.w,  h=best.h, angle=best.angle)
                if attr in ('w', 'h'):
                    new_val = max(1.0, new_val)
                kwargs[attr] = new_val if attr != 'angle' else new_val % 180
                cand  = Rect(**kwargs)
                cand_s = sc(cand)
                if cand_s > best_s:
                    best_s, best, improved = cand_s, cand, True

        if not improved:
            steps = {k: v / 2.0 for k, v in steps.items()}
            if max(steps.values()) < 0.25:
                break

    return best, best_s


# ---------------------------------------------------------------------------
# Overlap helper
# ---------------------------------------------------------------------------

def _overlap_fraction(rect: Rect, placed_union: np.ndarray,
                      H: int, W: int) -> float:
    """
    Return the fraction of `rect`'s pixels that lie inside `placed_union`
    (the union of every previously-placed rectangle).  Returns 0 when the
    rectangle has no pixels or no rectangles have been placed yet.
    """
    if not placed_union.any():
        return 0.0
    pix  = rasterise(rect, H, W)
    area = int(np.count_nonzero(pix))
    if area == 0:
        return 0.0
    return int(np.count_nonzero(pix & placed_union)) / area


def _spill_fraction(rect: Rect, outside_mask: np.ndarray,
                    H: int, W: int) -> float:
    """
    Return the fraction of `rect`'s pixels that fall OUTSIDE the original
    wall mask (i.e. the bleed ratio).  0.0 = perfect containment.
    """
    pix  = rasterise(rect, H, W)
    area = int(np.count_nonzero(pix))
    if area == 0:
        return 0.0
    return int(np.count_nonzero(pix & outside_mask)) / area


# ---------------------------------------------------------------------------
# Main decomposition
# ---------------------------------------------------------------------------

def decompose(
    mask: np.ndarray,
    angle_steps:          int   = 40,
    rect_penalty:         float = 0.02,
    max_grid_dim:         int   = 200,
    coverage_stop:        float = 0.995,
    bleed_weight:         float = 3.0,
    initial_bleed_weight: Optional[float] = None,
    bleed_decay:          float = 1.0,
    axis_only:            bool  = False,
    axis_gap:             float = 0.0,
    max_overlap:          float = 0.5,
    max_spill_fraction:   float = 0.15,
    refine:               bool  = True,
    verbose:              bool  = True,
) -> List[Rect]:
    """
    Greedy rectangle decomposition.  Runs until no rectangle clears the
    penalty threshold — the algorithm decides how many rectangles to use.

    Parameters
    ----------
    mask                 : 2-D boolean / 0-1 uint8 array (H × W).
    angle_steps          : number of uniformly-spaced angles to test per step.
    rect_penalty         : minimum *net* gain (as a fraction of total mask pixels)
                           required to place another rectangle.  Higher values force
                           fewer, larger rectangles; lower values allow more precise
                           coverage.  Default 0.02 = 2 % of mask pixels.
    max_grid_dim         : cap on the rotated weight-grid size (both axes).  Larger
                           values = higher precision but slower per step.  200 is
                           fine for masks up to ~600×600 px.
    bleed_weight         : penalty multiplier for pixels outside the mask
                           (floor value when initial_bleed_weight is used).
    initial_bleed_weight : if provided, the bleed penalty starts at this value
                           for the first rectangle and decreases by `bleed_decay`
                           per placed rectangle until it reaches `bleed_weight`.
                           This allows early large rectangles to stay tightly
                           inside the mask while later small detail rectangles
                           are more permissive.  Set higher than `bleed_weight`
                           to activate (e.g. initial_bleed_weight=10.0,
                           bleed_weight=1.0, bleed_decay=0.5).
    bleed_decay          : how much the bleed penalty decreases per placed
                           rectangle when initial_bleed_weight is active.
                           Default 1.0 (drops by 1 per rect).  Use a smaller
                           value (e.g. 0.1) for a slow descent over many rects,
                           or a larger value (e.g. 5.0) for a rapid drop after
                           the first few large rectangles.
    axis_only            : if True, only test 0° and 90° angles — no diagonal
                           or PCA-guided search.  Refinement also locks the
                           angle so rectangles stay strictly axis-aligned.
    axis_gap             : when > 0 and axis_only is False, any generated angle
                           within this many degrees of 0°, 45°, 90°, or 135°
                           is replaced by the preferred angle itself.  Creates
                           dead-zones that snap near-axis features to exact
                           preferred directions (default 0 = disabled).
    max_overlap          : maximum fraction of a candidate rectangle's own
                           pixels that may overlap with the union of already-
                           placed rectangles.  Default 0.5 (≤50% redundancy).
                           Set to 1.0 to disable the constraint entirely.
                           Candidates exceeding this threshold are rejected
                           during the angle search; if no candidate at any
                           angle satisfies the constraint, the loop terminates.
    max_spill_fraction   : maximum fraction of a candidate rectangle's pixels
                           that may fall OUTSIDE the original mask (i.e. bleed
                           into non-wall territory).  Default 0.15 (15%).
                           Hard-rejects any candidate — before and after
                           refinement — whose outside-pixel ratio exceeds this
                           limit.  Set to 1.0 to disable.
    refine               : if True, hill-climb each placed rectangle.
    verbose              : print live progress.

    Returns
    -------
    List of Rect objects whose union approximates the mask.
    """
    H, W       = mask.shape
    mask       = mask.astype(bool)
    outside    = ~mask
    total      = int(np.count_nonzero(mask))

    # Determine the starting bleed weight; falls back to bleed_weight when
    # initial_bleed_weight is not specified or is not larger.
    bw_start = bleed_weight
    if initial_bleed_weight is not None and initial_bleed_weight > bleed_weight:
        bw_start = float(initial_bleed_weight)

    remaining  = mask.copy()
    placed_union = np.zeros((H, W), dtype=bool)   # union of all placed rects
    chosen: List[Rect] = []
    step = 0

    while True:
        n_left   = int(np.count_nonzero(remaining))
        if n_left == 0 or (total - n_left) / total >= coverage_stop:
            if verbose:
                pct = 100 * (total - n_left) / total
                print(f"\n✓ Stopped at {pct:.1f}% coverage after {step} rectangle(s).")
            break
        step += 1
        covered  = total - n_left
        # Recompute threshold from *remaining* pixels each step so it shrinks
        # as coverage improves — prevents early stopping on small residual regions.
        min_gain = max(1, int(rect_penalty * n_left))

        # Decreasing bleed: start tight on early large rects, relax for later
        # small detail rects.  Decreases by bleed_decay per placed rectangle,
        # floors at bleed_weight (the user-specified minimum penalty).
        current_bleed = max(bleed_weight, bw_start - float(len(chosen)) * bleed_decay)

        angles = build_angle_list(remaining, angle_steps,
                                  axis_only=axis_only, axis_gap=axis_gap)

        best_score: int            = min_gain - 1
        best_rect:  Optional[Rect] = None

        for i, angle in enumerate(angles):
            if verbose:
                _progress(step, len(chosen), covered, total, min_gain,
                          f"angle {angle:5.1f}° ({i+1}/{len(angles)})  bw={current_bleed:.1f}")
            rect, s = best_rect_at_angle(remaining, outside, angle, max_grid_dim, current_bleed)
            if rect is not None and s > best_score:
                # Hard-reject if the rect bleeds too much outside the mask.
                if max_spill_fraction < 1.0:
                    if _spill_fraction(rect, outside, H, W) > max_spill_fraction:
                        continue
                # Reject candidates that overlap previously-placed rectangles
                # by more than max_overlap of their own area.
                if max_overlap < 1.0 and chosen:
                    if _overlap_fraction(rect, placed_union, H, W) > max_overlap:
                        continue
                best_score, best_rect = s, rect

        if best_rect is None:
            if verbose:
                print(f"\n✗ No rect clears ≥{min_gain}px threshold — done.")
            break

        if refine:
            if verbose:
                _progress(step, len(chosen), covered, total, min_gain, "refining…")
            best_rect, best_score = refine_rect(
                best_rect, remaining, outside, H, W,
                bleed_weight=current_bleed, axis_only=axis_only)
            if best_score < min_gain:
                if verbose:
                    print(f"\n✗ Refined rect ({best_score}px) below threshold — done.")
                break
            # Refinement can drift the rectangle; re-check spill constraint.
            if max_spill_fraction < 1.0:
                sf = _spill_fraction(best_rect, outside, H, W)
                if sf > max_spill_fraction:
                    if verbose:
                        print(f"\n✗ Refined rect spills {100*sf:.0f}% (>"
                              f"{100*max_spill_fraction:.0f}%) — done.")
                    break
            # Re-check the overlap constraint after the hill-climb completes.
            if max_overlap < 1.0 and chosen:
                ovf = _overlap_fraction(best_rect, placed_union, H, W)
                if ovf > max_overlap:
                    if verbose:
                        print(f"\n✗ Refined rect overlaps {100*ovf:.0f}% (>"
                              f"{100*max_overlap:.0f}%) — done.")
                    break

        chosen.append(best_rect)
        rect_pixels = rasterise(best_rect, H, W)
        placed_union |= rect_pixels
        remaining    &= ~rect_pixels
        covered = total - int(np.count_nonzero(remaining))

        if verbose:
            # Print final committed state for this step, then newline
            _progress(step, len(chosen), covered, total, min_gain,
                      f"score={best_score}")
            print(f"\n    └─ {best_rect}")

    return chosen


# ---------------------------------------------------------------------------
# Visualisation  (unchanged from v1, minor label update)
# ---------------------------------------------------------------------------

PALETTE = [
    ( 74, 142, 247), (232,  76,  76), ( 45, 184, 125), (240, 160,  48),
    (155,  89, 182), (224,  92, 176), ( 39, 184, 200), (139, 195,  74),
    (255, 112,  67), (  0, 188, 212), (141, 110,  99), ( 96, 125, 139),
]


def visualise(
    mask: np.ndarray,
    rects: List[Rect],
    out_path: str = "result.png",
    show: bool = True,
) -> Image.Image:
    H, W = mask.shape
    mask = mask.astype(bool)   # ensure boolean for ~ operator throughout

    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

    union_arr      = np.zeros((H, W, 3), dtype=np.uint8)
    union_arr[~mask] = (230, 230, 230)

    for i, rect in enumerate(rects):
        col  = PALETTE[i % len(PALETTE)]
        pix  = rasterise(rect, H, W)
        for c in range(3):
            union_arr[:, :, c] = np.where(
                pix,
                np.clip(union_arr[:, :, c] * 0.5 + col[c] * 0.5, 0, 255).astype(np.uint8),
                union_arr[:, :, c],
            )

    union_img = Image.fromarray(union_arr)
    draw = ImageDraw.Draw(union_img)
    for i, rect in enumerate(rects):
        col     = PALETTE[i % len(PALETTE)]
        corners = rect.corners()
        pts     = [(float(corners[j, 0]), float(corners[j, 1])) for j in range(4)]
        pts.append(pts[0])
        draw.line(pts, fill=col, width=1)

    union_mask = np.zeros((H, W), dtype=bool)
    for rect in rects:
        union_mask |= rasterise(rect, H, W)

    err_arr = np.full((H, W, 3), 230, dtype=np.uint8)
    err_arr[ mask & ~union_mask] = ( 45, 184, 125)   # missed  → green
    err_arr[~mask &  union_mask] = (232,  76,  76)   # bleed   → red
    err_arr[ mask &  union_mask] = (255, 255, 255)   # correct → white
    err_img = Image.fromarray(err_arr)

    gap     = 8
    total_w = W * 3 + gap * 2
    out     = Image.new("RGB", (total_w, H + 20), (245, 245, 245))
    out.paste(mask_img,  (0,           0))
    out.paste(union_img, (W + gap,     0))
    out.paste(err_img,   (2*(W + gap), 0))

    d = ImageDraw.Draw(out)
    for x, label in [
        (2, "mask"), (W + gap + 2, "rectangles"), (2*(W + gap) + 2, "error")
    ]:
        d.text((x, H + 4), label, fill=(80, 80, 80))

    out.save(out_path)
    print(f"\nSaved → {out_path}")

    total    = int(np.count_nonzero(mask))
    sym_diff = int(np.count_nonzero(mask ^ union_mask))
    print(f"Rectangles used : {len(rects)}")
    print(f"Symmetric diff  : {sym_diff} px  ({100*sym_diff/max(total,1):.1f}% of mask)")

    if show:
        try:
            out.show()
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# Synthetic test mask
# ---------------------------------------------------------------------------

def make_test_mask(H: int = 64, W: int = 64) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx, ry, rx = H // 2, W // 2, H // 3, W // 4
    ys, xs = np.ogrid[:H, :W]
    mask[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1] = 1
    for i in range(H):
        j = i * W // H
        if 0 <= j < W:
            mask[i, max(0, j - 3):min(W, j + 4)] = 1
    return mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_mask(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(
        description="Decompose a binary mask into a minimal set of rotated rectangles.")
    p.add_argument("image",           nargs="?",
                   help="Path to binary image (PNG/BMP/…)")
    p.add_argument("--angle-steps",   type=int,   default=2,
                   help="Uniform angle steps in [0°,180°) (default 120)")
    p.add_argument("--rect-penalty",  type=float, default=0.02,
                   help="Min net gain fraction to place a rectangle (default 0.02)")
    p.add_argument("--max-grid-dim",  type=int,   default=200,
                   help="Cap on rotated weight-grid size (default 200). "
                        "Raise for higher precision on large masks, lower to speed up.")
    p.add_argument("--bleed-weight",          type=float, default=4.0,
                   help="Penalty multiplier for pixels outside mask (floor; default 4.0)")
    p.add_argument("--initial-bleed-weight",  type=float, default=None,
                   metavar="BW",
                   help="Starting bleed penalty for the first rectangle; decreases "
                        "by --bleed-decay per placed rect down to --bleed-weight. "
                        "Set higher than --bleed-weight to activate "
                        "(e.g. --initial-bleed-weight 12). Default: disabled.")
    p.add_argument("--bleed-decay",     type=float, default=1.0,
                   metavar="D",
                   help="How much the bleed penalty drops per placed rectangle "
                        "when --initial-bleed-weight is active (default 1.0). "
                        "Use 0.1 for a slow descent, 5.0 for a fast one.")
    p.add_argument("--axis-only",       action="store_true",
                   help="Only test 0° and 90° angles — no diagonal or PCA-guided "
                        "search, and refinement keeps angle locked.")
    p.add_argument("--axis-gap",        type=float, default=0.0,
                   metavar="DEG",
                   help="Dead-zone half-width (degrees) around 0°, 45°, 90°, 135°. "
                        "Any generated angle within this gap is dropped and the "
                        "preferred angle is used instead, snapping near-axis features "
                        "to exact preferred directions (default 0 = disabled).")
    p.add_argument("--max-overlap",     type=float, default=0.5,
                   metavar="F",
                   help="Maximum fraction of a candidate rectangle's area that may "
                        "overlap with the union of previously-placed rectangles "
                        "(default 0.5 = 50%%). Use 1.0 to disable the constraint.")
    p.add_argument("--max-spill-fraction", type=float, default=0.15,
                   metavar="F",
                   help="Maximum fraction of a candidate rectangle's pixels that "
                        "may fall outside the wall mask (bleed / spillage). "
                        "Default 0.15 (15%%). Use 1.0 to disable.")
    p.add_argument("--coverage-stop",  type=float, default=0.98,
                   help="Stop once this fraction of mask is covered (default 0.95 = 95%)")
    p.add_argument("--no-refine",     action="store_true",
                   help="Disable coordinate-descent refinement")
    p.add_argument("--output",        default="result.png",
                   help="Output visualisation path (default result.png)")
    p.add_argument("--no-show",       action="store_true",
                   help="Don't open the result image")
    args = p.parse_args()

    if args.image:
        print(f"Loading mask from {args.image}")
        mask = load_mask(args.image)
    else:
        print("No image supplied — using synthetic test mask.")
        mask = make_test_mask()

    total = int(np.count_nonzero(mask))
    print(f"Mask shape : {mask.shape}  ON pixels: {total}")
    print(f"Settings   : angle_steps={args.angle_steps}"
          f"  rect_penalty={args.rect_penalty}"
          f"  max_grid_dim={args.max_grid_dim}"
          f"  coverage_stop={args.coverage_stop}"
          f"  bleed_weight={args.bleed_weight}"
          f"  initial_bleed_weight={args.initial_bleed_weight}"
          f"  bleed_decay={args.bleed_decay}"
          f"  axis_only={args.axis_only}"
          f"  axis_gap={args.axis_gap}"
          f"  max_overlap={args.max_overlap}"
          f"  refine={not args.no_refine}\n")

    rects = decompose(
        mask,
        angle_steps          = args.angle_steps,
        rect_penalty         = args.rect_penalty,
        max_grid_dim         = args.max_grid_dim,
        coverage_stop        = args.coverage_stop,
        bleed_weight         = args.bleed_weight,
        initial_bleed_weight = args.initial_bleed_weight,
        bleed_decay          = args.bleed_decay,
        axis_only            = args.axis_only,
        axis_gap             = args.axis_gap,
        max_overlap          = args.max_overlap,
        max_spill_fraction   = args.max_spill_fraction,
        refine               = not args.no_refine,
        verbose              = True,
    )

    print(f"\n--- {len(rects)} rectangle(s) found ---")
    for i, r in enumerate(rects):
        print(f"  [{i+1}] {r}")

    visualise(mask, rects, out_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()