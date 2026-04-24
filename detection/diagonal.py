"""
Diagonal wall detection for the ReFloorBRUSNIKA pipeline.

Full deskew-based pipeline
--------------------------

Phase 1 — analyse the original image
  1. LSD → filter by minimum length and frame-border area
  2. Wall-mask filter  (UNet combined mask, if provided)
  3. OCR text filter   (docTR, optional — silently skipped if unavailable)
  4. Angle clustering → find dominant perpendicular pair → deskew angle

Phase 2 — deskewed image  (when a perpendicular pair is found)
  5. Deskew image + wall mask through the same affine transform M
  6. Transform OCR boxes through M  (no second inference or OCR call)
  7. LSD on deskewed image → wall-mask filter → text-mask filter
  8. Keep only axis-aligned (H/V) segments: these were diagonal in the
     original image and became horizontal/vertical after deskewing
  9. Cluster the deskewed segments; for any remaining diagonal clusters
     run sub-region processing: crop → rotate so diagonal → 0° → LSD
     → keep H/V → rotate back to deskewed-image coords
 10. Inverse-transform all accepted endpoints back to original-image coords
 11. Snap endpoints to the nearest face of an existing axis-aligned wall
 12. Return as WallSegment(is_diagonal=True)

Fallback — when no perpendicular pair is detected
  Return raw diagonal LSD segments from Phase 1 with endpoint snapping.

Only called by pipeline.py Stage 1c when the solid-colour wall path is
active (used_unet_grey=False); LSD is unreliable on synthetic grey-fill
images produced by the hollow-wall U-Net path.
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

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

AXIS_TOLERANCE_DEG: float = 20.0      # reject if within this many ° of H or V
MIN_LENGTH_PX: float = 40.0           # reject short segments
CLUSTER_TOL_DEG: float = 5.0          # merge clusters within this angular distance
PERP_TOL_DEG: float = 20.0            # tolerance for detecting perpendicular pair
MIN_WALL_MASK_COVERAGE: float = 0.35  # min quad-overlap fraction with wall mask
FILL_HALF_W: float = 14.0             # half-width of segment quad for mask checks
OCR_OVERLAP_REJECT: float = 0.30      # reject segment if ≥ this fraction overlaps text
OCR_CONFIDENCE: float = 0.30          # min docTR word confidence to trust
OCR_PADDING_PX: int = 14              # pixel padding added around each OCR word box
MAX_COLOR_DISTANCE: float = 55.0      # Euclidean BGR distance threshold (colour fallback)
MAX_AREA_FRACTION: float = 0.08       # reject if AABB > 8 % of image (frame borders)
BBOX_OVERLAP_REJECT: float = 0.60     # skip if AABB overlaps existing wall by ≥ 60 %
THICKNESS_MIN: float = 3.0
THICKNESS_MAX: float = 60.0
ENDPOINT_EXTEND_PX: float = 8.0       # always push endpoint outward by at least this far
ENDPOINT_SNAP_RADIUS: float = 30.0    # max ray distance to snap to a wall face
SUBREGION_PADDING: int = 60           # padding around diagonal cluster bounding box
SUBREGION_MIN_LEN: float = 20.0       # min LSD segment length inside sub-region crops
DIAG_CLUSTER_MIN_LEN: float = 100.0   # skip tiny residual diagonal clusters in phase 2

# ---------------------------------------------------------------------------
# OCR model cache  (loaded once per process)
# ---------------------------------------------------------------------------

_ocr_model = None   # None = not yet tried; False = tried but failed


def _get_ocr_model():
    """Return a cached docTR predictor, or None if unavailable."""
    global _ocr_model
    if _ocr_model is None:
        try:
            from doctr.models import ocr_predictor
            logger.info("[diagonal] Loading docTR OCR model ...")
            _ocr_model = ocr_predictor(pretrained=True)
            logger.info("[diagonal] docTR model ready")
        except Exception as exc:
            logger.info("[diagonal] docTR unavailable (%s) — text filter disabled", exc)
            _ocr_model = False
    return _ocr_model if _ocr_model else None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Segment angle in [0, 180)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _angle_dist(a: float, b: float) -> float:
    """Circular angular distance mod 180°."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _deviation_from_axis(angle: float) -> float:
    """Minimum angular distance from H (0°) or V (90°)."""
    return min(angle % 90.0, 90.0 - angle % 90.0)


def _weighted_mean_angle(members: list) -> float:
    """Length-weighted circular mean angle (handles 0°/180° wrap-around)."""
    sx = sum(s['length'] * math.cos(math.radians(2.0 * s['angle_deg'])) for s in members)
    sy = sum(s['length'] * math.sin(math.radians(2.0 * s['angle_deg'])) for s in members)
    return (math.degrees(math.atan2(sy, sx)) % 360.0) / 2.0 % 180.0


def _segment_quad(x1: float, y1: float, x2: float, y2: float,
                  half_w: float) -> Optional[np.ndarray]:
    """4-corner int32 rotated-rectangle polygon for a thick line segment."""
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 0.5:
        return None
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    px, py = -uy, ux
    return np.array([
        [x1 + half_w * px, y1 + half_w * py],
        [x1 - half_w * px, y1 - half_w * py],
        [x2 - half_w * px, y2 - half_w * py],
        [x2 + half_w * px, y2 + half_w * py],
    ], dtype=np.int32)


def _transform_pt(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Apply a 2×3 affine matrix to a single point."""
    v = M @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


# ---------------------------------------------------------------------------
# Colour / overlap helpers  (kept for fallback colour-distance filter)
# ---------------------------------------------------------------------------

def _sample_color_along_line(img: np.ndarray,
                             x1: float, y1: float,
                             x2: float, y2: float,
                             n: int = 15) -> Tuple[float, float, float]:
    t = np.linspace(0.0, 1.0, n)
    xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, img.shape[1] - 1)
    ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, img.shape[0] - 1)
    mean = img[ys, xs].astype(float).mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def _color_distance(a: Tuple, b: Tuple) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def _overlaps_existing(bbox: Tuple[float, float, float, float],
                       walls: List[WallSegment],
                       threshold: float = BBOX_OVERLAP_REJECT) -> bool:
    ax1, ay1, ax2, ay2 = bbox
    area_a = (ax2 - ax1) * (ay2 - ay1)
    if area_a <= 0:
        return False
    for w in walls:
        bx1, by1, bx2, by2 = w.bbox.x1, w.bbox.y1, w.bbox.x2, w.bbox.y2
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        if inter_w * inter_h / area_a >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _detect_ocr_boxes(img_bgr: np.ndarray) -> List[Tuple]:
    """
    Run docTR on img_bgr; return padded word boxes as (x1, y1, x2, y2, text).
    Returns [] when docTR is unavailable or finds nothing.
    """
    model = _get_ocr_model()
    if model is None:
        return []
    try:
        from doctr.io import DocumentFile
    except ImportError:
        return []

    H, W = img_bgr.shape[:2]
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        return []
    try:
        doc = DocumentFile.from_images([buf.tobytes()])
        res = model(doc)
    except Exception as exc:
        logger.warning("[diagonal] OCR inference error: %s", exc)
        return []

    boxes: List[Tuple] = []
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if hasattr(word, 'confidence') and word.confidence < OCR_CONFIDENCE:
                        continue
                    (x0, y0), (x1, y1) = word.geometry
                    px1 = max(0, int(x0 * W) - OCR_PADDING_PX)
                    py1 = max(0, int(y0 * H) - OCR_PADDING_PX)
                    px2 = min(W, int(x1 * W) + OCR_PADDING_PX)
                    py2 = min(H, int(y1 * H) + OCR_PADDING_PX)
                    boxes.append((px1, py1, px2, py2, word.value))
    return boxes


def _build_text_mask(H: int, W: int, boxes: List[Tuple]) -> np.ndarray:
    """Rasterise padded text boxes into a binary mask (1 = inside text region)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 1
    return mask


def _transform_boxes(boxes: List[Tuple], M: np.ndarray) -> List[Tuple]:
    """Transform (x1,y1,x2,y2,...) boxes through a 2×3 affine M; result is AABB."""
    result = []
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rest = box[4:]
        corners = np.array(
            [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]],
            dtype=float).T          # (3, 4)
        tc = (M @ corners).T        # (4, 2)
        nx1 = int(tc[:, 0].min()); ny1 = int(tc[:, 1].min())
        nx2 = int(tc[:, 0].max()); ny2 = int(tc[:, 1].max())
        result.append((nx1, ny1, nx2, ny2) + rest)
    return result


# ---------------------------------------------------------------------------
# LSD  +  segment-level mask filters
# ---------------------------------------------------------------------------

def _run_lsd(gray: np.ndarray, min_length: float,
             axis_tolerance: float) -> List[dict]:
    """
    Run LSD on a greyscale image; return all segments above min_length as
    dicts with keys: x1 y1 x2 y2 length angle_deg deviation_deg width is_diagonal.
    """
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except AttributeError:
        lsd = cv2.createLineSegmentDetector()
    try:
        raw = lsd.detect(gray)
    except Exception as exc:
        logger.warning("[diagonal] LSD failed: %s", exc)
        return []
    if raw is None or raw[0] is None or len(raw[0]) == 0:
        return []

    lines_raw, widths_raw = raw[0], raw[1]
    segs: List[dict] = []
    for i, line in enumerate(lines_raw):
        x1, y1, x2, y2 = line[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        angle = _angle_deg(x1, y1, x2, y2)
        dev   = _deviation_from_axis(angle)
        width = (float(widths_raw[i][0])
                 if widths_raw is not None and i < len(widths_raw) else 1.0)
        segs.append(dict(
            x1=x1, y1=y1, x2=x2, y2=y2,
            length=length, angle_deg=angle,
            deviation_deg=dev, width=width,
            is_diagonal=(dev >= axis_tolerance),
        ))
    return segs


def _filter_by_mask(segs: List[dict], mask: np.ndarray,
                    fill_half_w: float = FILL_HALF_W,
                    min_coverage: float = MIN_WALL_MASK_COVERAGE) -> List[dict]:
    """Keep segments whose filled quad overlaps the mask by >= min_coverage."""
    H, W = mask.shape[:2]
    qm   = np.zeros((H, W), dtype=np.uint8)
    kept: List[dict] = []
    for seg in segs:
        hw   = max(fill_half_w, seg.get('width', 1.0))
        quad = _segment_quad(seg['x1'], seg['y1'], seg['x2'], seg['y2'], hw)
        if quad is None:
            kept.append(seg)
            continue
        qm[:] = 0
        cv2.fillPoly(qm, [quad], 1)
        total = int(qm.sum())
        if total == 0:
            kept.append(seg)
            continue
        overlap = int(np.sum((qm > 0) & (mask > 0)))
        if overlap / total >= min_coverage:
            kept.append(seg)
    return kept


def _filter_by_text(segs: List[dict], text_mask: np.ndarray,
                    fill_half_w: float = FILL_HALF_W,
                    max_overlap: float = OCR_OVERLAP_REJECT) -> List[dict]:
    """Reject segments whose filled quad overlaps the text mask by >= max_overlap."""
    H, W = text_mask.shape[:2]
    qm   = np.zeros((H, W), dtype=np.uint8)
    kept: List[dict] = []
    for seg in segs:
        hw   = max(fill_half_w, seg.get('width', 1.0))
        quad = _segment_quad(seg['x1'], seg['y1'], seg['x2'], seg['y2'], hw)
        if quad is None:
            kept.append(seg)
            continue
        qm[:] = 0
        cv2.fillPoly(qm, [quad], 1)
        total = int(qm.sum())
        if total == 0:
            kept.append(seg)
            continue
        overlap = int(np.sum((qm > 0) & (text_mask > 0)))
        if overlap / total < max_overlap:
            kept.append(seg)
    return kept


# ---------------------------------------------------------------------------
# Clustering  +  deskew
# ---------------------------------------------------------------------------

def _cluster_segments(segs: List[dict], cluster_tol: float,
                      axis_tolerance: float) -> List[dict]:
    """
    Greedy angle-clustering of segments.  Uses length-weighted circular mean
    with doubled-angle trick to handle 0°/180° wrap-around cleanly.
    """
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: s['angle_deg'])
    groups  = [[ordered[0]]]
    for seg in ordered[1:]:
        if _angle_dist(seg['angle_deg'],
                       _weighted_mean_angle(groups[-1])) <= cluster_tol:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    # Merge first and last group if they wrap around 180°
    if len(groups) >= 2:
        if _angle_dist(_weighted_mean_angle(groups[0]),
                       _weighted_mean_angle(groups[-1])) <= cluster_tol:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    result = []
    for members in groups:
        mean_a = _weighted_mean_angle(members)
        result.append(dict(
            members      = members,
            cluster_angle= mean_a,
            cluster_size = len(members),
            total_length = sum(s['length'] for s in members),
            is_diagonal  = (_deviation_from_axis(mean_a) >= axis_tolerance),
        ))
    return sorted(result, key=lambda c: c['cluster_angle'])


def _find_deskew_angle(clusters: List[dict],
                       perp_tol: float = PERP_TOL_DEG) -> Optional[float]:
    """
    Find the correction angle that aligns the two most dominant clusters with
    H/V axes.  Returns None when the dominant pair is not ~perpendicular.
    """
    if len(clusters) < 2:
        return None
    ranked          = sorted(clusters, key=lambda c: c['total_length'], reverse=True)
    primary_angle   = ranked[0]['cluster_angle']
    secondary_angle = ranked[1]['cluster_angle']
    apart = _angle_dist(primary_angle, secondary_angle)
    if abs(apart - 90.0) > perp_tol:
        return None
    dev = primary_angle % 90.0
    return -dev if dev <= 45.0 else (90.0 - dev)


def _deskew(img: np.ndarray,
            angle_deg: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Rotate image CCW by angle_deg with canvas expansion and white fill.
    Returns (rotated_img, M_2x3, (new_width, new_height)).
    """
    h, w   = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a  = abs(math.cos(math.radians(angle_deg)))
    sin_a  = abs(math.sin(math.radians(angle_deg)))
    new_w  = int(math.ceil(h * sin_a + w * cos_a))
    new_h  = int(math.ceil(h * cos_a + w * sin_a))
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    return rotated, M, (new_w, new_h)


def _rotate_mask(mask: np.ndarray, M: np.ndarray,
                 new_size: Tuple[int, int]) -> np.ndarray:
    """Apply the same affine to a class-id or binary mask (INTER_NEAREST)."""
    return cv2.warpAffine(mask, M, new_size,
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


# ---------------------------------------------------------------------------
# Endpoint snapping  (ray-cast to nearest axis-aligned wall face)
# ---------------------------------------------------------------------------

def _snap_endpoint_to_wall_face(
    px: float, py: float,
    dx: float, dy: float,
    existing_walls: List[WallSegment],
    extend_px: float = ENDPOINT_EXTEND_PX,
    snap_radius: float = ENDPOINT_SNAP_RADIUS,
) -> Tuple[float, float]:
    """
    Push (px, py) outward along (dx, dy) until it touches a wall face, or
    by extend_px if no face is found within snap_radius.
    """
    best_t = extend_px  # always extend by at least this much
    for wall in existing_walls:
        if wall.is_diagonal:
            continue
        bx1, by1 = wall.bbox.x1, wall.bbox.y1
        bx2, by2 = wall.bbox.x2, wall.bbox.y2
        # Quick distance cull
        wcx, wcy  = (bx1 + bx2) / 2, (by1 + by2) / 2
        half_diag = math.hypot(bx2 - bx1, by2 - by1) / 2
        if math.hypot(px - wcx, py - wcy) > snap_radius + half_diag:
            continue
        ww, wh = bx2 - bx1, by2 - by1
        half_t = max(getattr(wall, 'thickness', 0) / 2, 2.0)
        if ww >= wh:                         # horizontal wall: top/bottom faces
            if abs(dy) > 1e-6:
                for face_y in (by1, by2):
                    t = (face_y - py) / dy
                    if 0.0 < t <= snap_radius:
                        ix = px + t * dx
                        if bx1 - half_t <= ix <= bx2 + half_t:
                            best_t = min(best_t, t)
        else:                                # vertical wall: left/right faces
            if abs(dx) > 1e-6:
                for face_x in (bx1, bx2):
                    t = (face_x - px) / dx
                    if 0.0 < t <= snap_radius:
                        iy = py + t * dy
                        if by1 - half_t <= iy <= by2 + half_t:
                            best_t = min(best_t, t)
    return px + best_t * dx, py + best_t * dy


def _refine_endpoints(
    x1: float, y1: float,
    x2: float, y2: float,
    existing_walls: List[WallSegment],
) -> Tuple[float, float, float, float]:
    """Extend both endpoints outward and snap each to the nearest wall face."""
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1.0:
        return x1, y1, x2, y2
    udx, udy = (x2 - x1) / length, (y2 - y1) / length
    nx1, ny1 = _snap_endpoint_to_wall_face(x1, y1, -udx, -udy, existing_walls)
    nx2, ny2 = _snap_endpoint_to_wall_face(x2, y2, +udx, +udy, existing_walls)
    return nx1, ny1, nx2, ny2


# ---------------------------------------------------------------------------
# Sub-region diagonal processing
# ---------------------------------------------------------------------------

def _process_diagonal_cluster(
    img_bgr: np.ndarray,
    cluster: dict,
    padding: int = SUBREGION_PADDING,
    axis_tolerance: float = AXIS_TOLERANCE_DEG,
) -> List[dict]:
    """
    Crop the cluster bounding box, rotate so the cluster direction → 0°, run
    LSD to find H/V segments inside, then rotate results back to the input
    image's coordinate system.

    Returns list of dicts {p1:(x,y), p2:(x,y), thickness:float} in input-image
    pixel coordinates (not original-image coords — callers must apply M_inv).
    """
    H_img, W_img = img_bgr.shape[:2]
    angle   = cluster['cluster_angle']
    members = cluster['members']

    xs = [v for s in members for v in (s['x1'], s['x2'])]
    ys = [v for s in members for v in (s['y1'], s['y2'])]
    if not xs:
        return []

    bx1 = max(0,     int(min(xs)) - padding)
    by1 = max(0,     int(min(ys)) - padding)
    bx2 = min(W_img, int(max(xs)) + padding)
    by2 = min(H_img, int(max(ys)) + padding)
    patch = img_bgr[by1:by2, bx1:bx2].copy()
    ph, pw = patch.shape[:2]
    if ph < 10 or pw < 10:
        return []

    # Forward rotation: bring cluster direction to 0° (horizontal)
    rot   = -angle
    cx, cy = pw / 2.0, ph / 2.0
    Mf    = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    cos_a = abs(math.cos(math.radians(rot)))
    sin_a = abs(math.sin(math.radians(rot)))
    nw    = int(math.ceil(ph * sin_a + pw * cos_a))
    nh    = int(math.ceil(ph * cos_a + pw * sin_a))
    Mf[0, 2] += (nw - pw) / 2.0
    Mf[1, 2] += (nh - ph) / 2.0
    rotated = cv2.warpAffine(patch, Mf, (nw, nh),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    # LSD on rotated patch — keep only H/V segments
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except AttributeError:
        lsd = cv2.createLineSegmentDetector()
    raw = lsd.detect(gray_rot)
    if raw is None or raw[0] is None:
        return []

    Mi     = cv2.invertAffineTransform(Mf)
    result = []
    for i, line in enumerate(raw[0]):
        x1r, y1r, x2r, y2r = line[0]
        if math.hypot(x2r - x1r, y2r - y1r) < SUBREGION_MIN_LEN:
            continue
        if _deviation_from_axis(_angle_deg(x1r, y1r, x2r, y2r)) > axis_tolerance:
            continue
        w = float(raw[1][i][0]) if raw[1] is not None and i < len(raw[1]) else 2.0
        # Inverse-rotate back to patch coords, then shift to input-image coords
        p1r = Mi @ np.array([x1r, y1r, 1.0])
        p2r = Mi @ np.array([x2r, y2r, 1.0])
        result.append(dict(
            p1=(float(p1r[0]) + bx1, float(p1r[1]) + by1),
            p2=(float(p2r[0]) + bx1, float(p2r[1]) + by1),
            thickness=max(2.0, w),
        ))
    return result


# ---------------------------------------------------------------------------
# WallSegment builder
# ---------------------------------------------------------------------------

def _make_wall_segment(x1: float, y1: float,
                       x2: float, y2: float,
                       thickness: float) -> WallSegment:
    bx1, by1 = min(x1, x2), min(y1, y2)
    bx2, by2 = max(x1, x2), max(y1, y2)
    return WallSegment(
        id=f"diag-{uuid4()}",
        bbox=BBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
        thickness=max(THICKNESS_MIN, min(THICKNESS_MAX, thickness)),
        is_structural=True,
        is_diagonal=True,
        outline=[(int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2)))],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_diagonal_walls(
    img: np.ndarray,
    config: PipelineConfig,
    existing_walls: List[WallSegment],
    wall_mask: Optional[np.ndarray] = None,
    axis_tolerance_deg: float = AXIS_TOLERANCE_DEG,
    min_length_px: float = MIN_LENGTH_PX,
) -> List[WallSegment]:
    """
    Detect non-axis-aligned (diagonal) wall segments using a deskew-based
    pipeline.  See module docstring for the full algorithm.

    Args:
        img:            Original BGR image (after pre-processing / crop).
        config:         Pipeline configuration (wall colour, scale, etc.).
        existing_walls: Axis-aligned walls already found in Stage 1.
        wall_mask:      Optional binary UNet wall mask (any positive value = wall).
        axis_tolerance_deg: Degrees from H/V within which a segment is axis-aligned.
        min_length_px:  Minimum accepted LSD segment length.

    Returns:
        List of WallSegment with is_diagonal=True, in original-image coordinates.
    """
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W        = img.shape[:2]
    image_area  = float(H * W)

    # Normalise wall mask to 0/1 uint8 regardless of input range
    wall_mask_bin: Optional[np.ndarray] = None
    if wall_mask is not None:
        wall_mask_bin = (wall_mask > 0).astype(np.uint8)

    # ── Phase 1: analyse original image ──────────────────────────────────────

    segs = _run_lsd(gray, min_length_px, axis_tolerance_deg)

    # Drop frame-border segments (huge bounding box → probably image edge artefact)
    segs = [
        s for s in segs
        if (max(s['x1'], s['x2']) - min(s['x1'], s['x2'])) *
           (max(s['y1'], s['y2']) - min(s['y1'], s['y2']))
           <= MAX_AREA_FRACTION * image_area
    ]

    # Wall-content filter
    if wall_mask_bin is not None:
        segs = _filter_by_mask(segs, wall_mask_bin)
    elif not config.auto_wall_color:
        # Colour-distance fallback when no mask and wall colour is explicit
        wall_bgr = config.wall_color_bgr
        segs = [
            s for s in segs
            if _color_distance(
                _sample_color_along_line(img, s['x1'], s['y1'], s['x2'], s['y2']),
                wall_bgr,
            ) <= MAX_COLOR_DISTANCE
        ]

    # OCR text filter — silently skipped if docTR is not installed
    text_boxes = _detect_ocr_boxes(img)
    if text_boxes:
        logger.info("[diagonal] Phase 1 OCR: %d word regions found", len(text_boxes))
        segs = _filter_by_text(segs, _build_text_mask(H, W, text_boxes))

    # Cluster all segments and look for a dominant perpendicular pair
    clusters     = _cluster_segments(segs, CLUSTER_TOL_DEG, axis_tolerance_deg)
    deskew_angle = _find_deskew_angle(clusters)

    logger.info(
        "[diagonal] Phase 1: %d segs, %d clusters, deskew=%.2f deg",
        len(segs), len(clusters),
        deskew_angle if deskew_angle is not None else 0.0,
    )

    # ── Fallback: no dominant perpendicular pair ──────────────────────────────
    if deskew_angle is None:
        results: List[WallSegment] = []
        for seg in segs:
            if not seg['is_diagonal']:
                continue
            bbox = (min(seg['x1'], seg['x2']), min(seg['y1'], seg['y2']),
                    max(seg['x1'], seg['x2']), max(seg['y1'], seg['y2']))
            if _overlaps_existing(bbox, existing_walls):
                continue
            x1r, y1r, x2r, y2r = _refine_endpoints(
                seg['x1'], seg['y1'], seg['x2'], seg['y2'], existing_walls)
            results.append(_make_wall_segment(x1r, y1r, x2r, y2r, seg['width']))
        logger.info("[diagonal] Fallback: %d diagonal walls (no perpendicular pair)",
                    len(results))
        return results

    # ── Phase 2: deskewed image ───────────────────────────────────────────────

    img_desk, M, new_size = _deskew(img, deskew_angle)
    W_desk, H_desk        = new_size   # OpenCV convention: (width, height)

    # Rotate wall mask with INTER_NEAREST to preserve 0/1 values
    wall_mask_desk: Optional[np.ndarray] = (
        _rotate_mask(wall_mask_bin, M, new_size) if wall_mask_bin is not None else None
    )

    # Transform OCR boxes into deskewed coordinate space (no second OCR call)
    text_boxes_desk = _transform_boxes(text_boxes, M) if text_boxes else []
    text_mask_desk: Optional[np.ndarray] = (
        _build_text_mask(H_desk, W_desk, text_boxes_desk) if text_boxes_desk else None
    )

    # LSD on deskewed image
    gray_desk = cv2.cvtColor(img_desk, cv2.COLOR_BGR2GRAY)
    segs_desk = _run_lsd(gray_desk, min_length_px, axis_tolerance_deg)

    if wall_mask_desk is not None:
        segs_desk = _filter_by_mask(segs_desk, wall_mask_desk)
    if text_mask_desk is not None:
        segs_desk = _filter_by_text(segs_desk, text_mask_desk)

    # H/V segments in deskewed space = walls that were diagonal in original image
    segs_hv = [s for s in segs_desk if not s['is_diagonal']]

    logger.info(
        "[diagonal] Phase 2: %d segs on deskewed image, %d are H/V (former diagonals)",
        len(segs_desk), len(segs_hv),
    )

    # Inverse affine: deskewed coords → original-image coords
    M_inv   = cv2.invertAffineTransform(M)
    results = []

    # ── Primary: H/V segments → back to original coords ──────────────────────
    for seg in segs_hv:
        ox1, oy1 = _transform_pt(M_inv, seg['x1'], seg['y1'])
        ox2, oy2 = _transform_pt(M_inv, seg['x2'], seg['y2'])
        bbox = (min(ox1, ox2), min(oy1, oy2), max(ox1, ox2), max(oy1, oy2))
        if _overlaps_existing(bbox, existing_walls):
            continue
        ox1r, oy1r, ox2r, oy2r = _refine_endpoints(ox1, oy1, ox2, oy2, existing_walls)
        results.append(_make_wall_segment(ox1r, oy1r, ox2r, oy2r, seg['width']))

    # ── Secondary: remaining diagonal clusters in deskewed space ─────────────
    # These are walls at angles other than the primary diagonal direction
    # (e.g. a 30° wall in an image whose dominant diagonal was 45°).
    clusters_desk = _cluster_segments(segs_desk, CLUSTER_TOL_DEG, axis_tolerance_deg)
    for cluster in clusters_desk:
        if not cluster['is_diagonal']:
            continue
        if cluster['total_length'] < DIAG_CLUSTER_MIN_LEN:
            continue  # skip tiny residual noise clusters
        sub_walls = _process_diagonal_cluster(
            img_desk, cluster, SUBREGION_PADDING, axis_tolerance_deg)
        for sw in sub_walls:
            # Transform from deskewed-image coords back to original-image coords
            p1x, p1y = _transform_pt(M_inv, sw['p1'][0], sw['p1'][1])
            p2x, p2y = _transform_pt(M_inv, sw['p2'][0], sw['p2'][1])
            bbox = (min(p1x, p2x), min(p1y, p2y), max(p1x, p2x), max(p1y, p2y))
            if _overlaps_existing(bbox, existing_walls):
                continue
            ox1r, oy1r, ox2r, oy2r = _refine_endpoints(p1x, p1y, p2x, p2y,
                                                         existing_walls)
            results.append(_make_wall_segment(ox1r, oy1r, ox2r, oy2r,
                                              sw['thickness']))

    logger.info("[diagonal] Phase 2 total: %d diagonal walls returned", len(results))
    return results
