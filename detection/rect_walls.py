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
    ) -> DetectionResult:
        """
        Detect walls in ``img_bgr``.  If ``wall_mask`` is supplied it is used
        directly; otherwise the binary U-Net is invoked.
        """
        cfg = self.config
        H, W = img_bgr.shape[:2]

        # ── 1. Wall mask ───────────────────────────────────────────────
        if wall_mask is None:
            wall_mask = self._run_binary_unet(img_bgr)

        if wall_mask is None or int(np.count_nonzero(wall_mask)) == 0:
            logger.warning("RectWallDetector: empty wall mask — no walls returned")
            return DetectionResult(
                walls=[], openings=[], wall_mask=wall_mask,
                outline_mask=None, source="rect_walls",
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
