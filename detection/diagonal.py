"""
Diagonal pre-processor for the ReFloorBRUSNIKA pipeline.

Single public entry point: ``preprocess_floorplan(image_path, ...)``.

What this module does, in order, on the input image:

1. Crop to the largest connected ink structure (drops legend / margins).
2. Correct small scanner skew (≤5°) so subsequent stages see a roughly
   axis-aligned image.
3. docTR OCR in normal orientation, then again on a 90° CW rotation
   (to pick up vertical wall-length labels).  Both label sets are
   stored in current-image coordinates.
4. Erase every text region from a working copy of the image so that
   text doesn't pollute U-Net or LSD.
5. U-Net inference → pred_mask (0 bg, 1 wall, 2 window, 3 door) and
   binary wall mask.
6. LSD on the text-erased image, filtered to segments whose oriented
   quad sufficiently overlaps the wall mask.
7. Greedy length-weighted angle clustering on those segments.
8. Pick the longest two perpendicular clusters → deskew correction.
9. Rotate image, U-Net masks, and OCR labels through the same affine
   matrix M.
10. Re-cluster on the deskewed image and find clusters that are STILL
    diagonal (axis deviation ≥ AXIS_TOL_DEG).
11. For each residual diagonal cluster, paint a clean grey filled quad
    on the deskewed image: crop the cluster's pred_mask region with
    padding → rotate so the cluster angle becomes 0° → take the
    axis-aligned bbox of each connected component (any of wall /
    window / door) → inverse-rotate the bbox corners back to deskewed
    coords → fillPoly with grey.  This way the existing axis-aligned
    pipeline downstream sees the diagonal walls (and diagonal windows /
    doors, which are merged into "wall") as ordinary rectangular ink.
12. Save the final deskewed-and-painted image to
    ``<input_dir>/deskewed/<basename>`` (creating the directory).
13. Return the deskewed path and OCR labels in deskewed coordinates.

The pipeline driver (pipeline.py) calls this once per input image and
treats the returned path as the canonical input for everything that
follows.  No second OCR or U-Net run is needed downstream for OCR or
text-erasure purposes — those products are handed off directly.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

GREY_BGR: Tuple[int, int, int] = (180, 180, 180)   # painted-diagonal colour

LSD_MIN_LEN_PX: float       = 30.0
AXIS_TOL_DEG: float         = 15.0      # |dev from H/V| ≥ this ⇒ diagonal
CLUSTER_TOL_DEG: float      = 5.0       # angular merge threshold
PERP_TOL_DEG: float         = 20.0      # deviation from 90° apart
MIN_DESKEW_DEG: float       = 3.0       # below this ⇒ no deskew

WALL_QUAD_HALF_W: float     = 14.0      # half-width of segment quad for mask test
WALL_MASK_MIN_OVERLAP: float = 0.25     # fraction of quad on wall_mask to keep

DIAG_PATCH_PAD_PX: int      = 30
DIAG_MIN_COMPONENT_AREA: int = 60

OCR_CONFIDENCE: float       = 0.30
OCR_PADDING_PX: int         = 14

UNET_DEFAULT_CKPT: str      = "weights/epoch_040.pth"
UNET_DEFAULT_SIZE: int      = 1024


# ---------------------------------------------------------------------------
# OCR — docTR
# ---------------------------------------------------------------------------

_ocr_model: Any = None   # None = not tried; False = tried, unavailable


def _get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        try:
            from doctr.models import ocr_predictor
            logger.info("[diagonal] Loading docTR OCR model ...")
            _ocr_model = ocr_predictor(pretrained=True)
            logger.info("[diagonal] docTR ready")
        except Exception as exc:
            logger.info("[diagonal] docTR unavailable (%s)", exc)
            _ocr_model = False
    return _ocr_model if _ocr_model else None


def _ocr_image(img_bgr: np.ndarray) -> List[Tuple]:
    """
    docTR on img_bgr.
    Returns: List[(text, x1, y1, x2, y2)] in image coordinates.
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
        logger.warning("[diagonal] OCR inference failed: %s", exc)
        return []

    out: List[Tuple] = []
    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if (hasattr(word, 'confidence')
                            and word.confidence < OCR_CONFIDENCE):
                        continue
                    (x0, y0), (x1, y1) = word.geometry
                    px1 = max(0, int(x0 * W) - OCR_PADDING_PX)
                    py1 = max(0, int(y0 * H) - OCR_PADDING_PX)
                    px2 = min(W, int(x1 * W) + OCR_PADDING_PX)
                    py2 = min(H, int(y1 * H) + OCR_PADDING_PX)
                    out.append((str(word.value), px1, py1, px2, py2))
    return out


def _ocr_both_orientations(img_bgr: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Run docTR in normal orientation and on a 90° CW rotation; return both
    label lists in CURRENT-IMAGE coordinates (the rotated set has been
    un-rotated back).
    """
    H, W = img_bgr.shape[:2]
    normal = _ocr_image(img_bgr)

    rotated = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    raw_rot = _ocr_image(rotated)
    del rotated

    rot_unrotated: List[Tuple] = []
    for item in raw_rot:
        if len(item) < 5:
            continue
        text, rx1, ry1, rx2, ry2 = item
        # CW inverse: rotated (rx, ry) -> original (ry, H-1-rx)
        ox1 = min(int(ry1), int(ry2))
        oy1 = H - 1 - max(int(rx1), int(rx2))
        ox2 = max(int(ry1), int(ry2))
        oy2 = H - 1 - min(int(rx1), int(rx2))
        rot_unrotated.append((text, ox1, oy1, ox2, oy2))
    return normal, rot_unrotated


def _erase_text(img_bgr: np.ndarray,
                label_lists: List[List[Tuple]],
                pad: int = 3) -> np.ndarray:
    """Return a copy of img_bgr with every text bbox filled white."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for src in label_lists:
        for item in src or []:
            if len(item) < 5:
                continue
            x1 = max(0, int(item[1]) - pad)
            y1 = max(0, int(item[2]) - pad)
            x2 = min(W, int(item[3]) + pad)
            y2 = min(H, int(item[4]) + pad)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _angle_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _deviation_from_axis(angle: float) -> float:
    return min(angle % 90.0, 90.0 - angle % 90.0)


def _weighted_mean_angle(members: List[Dict]) -> float:
    sx = sum(m['length'] * math.cos(math.radians(2.0 * m['angle_deg'])) for m in members)
    sy = sum(m['length'] * math.sin(math.radians(2.0 * m['angle_deg'])) for m in members)
    return (math.degrees(math.atan2(sy, sx)) % 360.0) / 2.0 % 180.0


def _segment_quad(x1: float, y1: float, x2: float, y2: float,
                  half_w: float) -> Optional[np.ndarray]:
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


def _transform_pts(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 2×3 affine to (N,2) array of points, return (N,2)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, M).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Cropping & small-skew (vendored copies of pipeline.py helpers, so this
# module can preprocess without importing pipeline.py)
# ---------------------------------------------------------------------------

def _crop_to_floorplan(img: np.ndarray,
                       fg_thresh: int = 230,
                       min_fill: float = 0.03,
                       margin: int = 12) -> np.ndarray:
    """Crop to the largest dark-ink connected component."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, fg = cv2.threshold(gray, fg_thresh, 255, cv2.THRESH_BINARY_INV)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return img
    H, W = img.shape[:2]
    best_idx = 0
    best_area = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > best_area:
            best_area = a
            best_idx = i
    if best_area < min_fill * H * W:
        return img
    x = stats[best_idx, cv2.CC_STAT_LEFT]
    y = stats[best_idx, cv2.CC_STAT_TOP]
    w = stats[best_idx, cv2.CC_STAT_WIDTH]
    h = stats[best_idx, cv2.CC_STAT_HEIGHT]
    if w >= W * 0.98 and h >= H * 0.98:
        return img
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)
    return img[y1:y2, x1:x2].copy()


def _small_skew_correct(img: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Hough-based small skew correction. Returns img unchanged if |θ|<0.3°."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120,
                             minLineLength=80, maxLineGap=12)
    if lines is None or len(lines) == 0:
        return img
    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = ((a + 90.0) % 180.0) - 90.0   # fold to (-90, 90]
        # only horizontal-ish lines for skew estimation
        if abs(a) <= max_angle:
            angles.append(a)
    if len(angles) < 5:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return img
    return _rotate(img, angle, border_value=(255, 255, 255), interp=cv2.INTER_LINEAR)[0]


# ---------------------------------------------------------------------------
# LSD + filtering + clustering
# ---------------------------------------------------------------------------

def _run_lsd(gray: np.ndarray, min_len: float = LSD_MIN_LEN_PX,
             axis_tol: float = AXIS_TOL_DEG) -> List[Dict]:
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except AttributeError:
        try:
            lsd = cv2.createLineSegmentDetector()
        except Exception as exc:
            logger.warning("[diagonal] LSD unavailable: %s", exc)
            return []
    try:
        raw = lsd.detect(gray)
    except Exception as exc:
        logger.warning("[diagonal] LSD detect failed: %s", exc)
        return []
    if raw is None or raw[0] is None or len(raw[0]) == 0:
        return []
    lines, widths = raw[0], raw[1]
    out: List[Dict] = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < min_len:
            continue
        a = _angle_deg(x1, y1, x2, y2)
        dev = _deviation_from_axis(a)
        w = (float(widths[i][0])
             if widths is not None and i < len(widths) else 1.0)
        out.append(dict(
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            length=L, angle_deg=a, deviation_deg=dev,
            width=w, is_diagonal=(dev >= axis_tol),
        ))
    return out


def _filter_segments_by_mask(segs: List[Dict], mask: np.ndarray,
                              half_w: float = WALL_QUAD_HALF_W,
                              min_overlap: float = WALL_MASK_MIN_OVERLAP
                              ) -> List[Dict]:
    if not segs or mask is None:
        return segs
    H, W = mask.shape[:2]
    bin_mask = (mask > 0).astype(np.uint8)
    quad_buf = np.zeros((H, W), dtype=np.uint8)
    kept: List[Dict] = []
    for s in segs:
        hw = max(half_w, s.get('width', 1.0))
        quad = _segment_quad(s['x1'], s['y1'], s['x2'], s['y2'], hw)
        if quad is None:
            kept.append(s); continue
        quad_buf[:] = 0
        cv2.fillPoly(quad_buf, [quad], 1)
        total = int(quad_buf.sum())
        if total == 0:
            kept.append(s); continue
        overlap = int(np.sum((quad_buf > 0) & (bin_mask > 0)))
        if overlap / total >= min_overlap:
            kept.append(s)
    return kept


def _cluster_segments(segs: List[Dict],
                      cluster_tol: float = CLUSTER_TOL_DEG,
                      axis_tol: float = AXIS_TOL_DEG) -> List[Dict]:
    if not segs:
        return []
    ordered = sorted(segs, key=lambda s: s['angle_deg'])
    groups: List[List[Dict]] = [[ordered[0]]]
    for s in ordered[1:]:
        if _angle_dist(s['angle_deg'],
                       _weighted_mean_angle(groups[-1])) <= cluster_tol:
            groups[-1].append(s)
        else:
            groups.append([s])
    if len(groups) >= 2:
        if _angle_dist(_weighted_mean_angle(groups[0]),
                       _weighted_mean_angle(groups[-1])) <= cluster_tol:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
    out = []
    for members in groups:
        ma = _weighted_mean_angle(members)
        out.append(dict(
            members=members,
            cluster_angle=ma,
            cluster_size=len(members),
            total_length=sum(s['length'] for s in members),
            is_diagonal=(_deviation_from_axis(ma) >= axis_tol),
        ))
    return sorted(out, key=lambda c: c['cluster_angle'])


def _find_deskew_angle(clusters: List[Dict],
                       perp_tol: float = PERP_TOL_DEG,
                       min_angle: float = MIN_DESKEW_DEG) -> Optional[float]:
    if len(clusters) < 2:
        return None
    ranked = sorted(clusters, key=lambda c: c['total_length'], reverse=True)
    primary = ranked[0]
    if not primary['is_diagonal']:
        return None
    secondary = ranked[1]
    if abs(_angle_dist(primary['cluster_angle'],
                        secondary['cluster_angle']) - 90.0) > perp_tol:
        return None
    dev = primary['cluster_angle'] % 90.0
    correction = -dev if dev <= 45.0 else (90.0 - dev)
    return correction if abs(correction) >= min_angle else None


# ---------------------------------------------------------------------------
# Affine rotation helpers
# ---------------------------------------------------------------------------

def _rotate(img: np.ndarray, angle_deg: float,
            border_value=(255, 255, 255),
            interp: int = cv2.INTER_LINEAR
            ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Rotate CCW by angle_deg, expand canvas. Returns (rotated, M, (W,H))."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(math.cos(math.radians(angle_deg)))
    sin_a = abs(math.sin(math.radians(angle_deg)))
    new_w = int(math.ceil(h * sin_a + w * cos_a))
    new_h = int(math.ceil(h * cos_a + w * sin_a))
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             flags=interp,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=border_value)
    return rotated, M, (new_w, new_h)


def _rotate_mask(mask: np.ndarray, M: np.ndarray,
                  size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask, M, size,
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _transform_labels(labels: List[Tuple], M: np.ndarray) -> List[Tuple]:
    """Pass (text, x1, y1, x2, y2) labels through an affine matrix.
    Result bbox is the AABB of the four transformed corners."""
    out: List[Tuple] = []
    for item in labels or []:
        if len(item) < 5:
            out.append(item); continue
        text = item[0]
        x1, y1, x2, y2 = float(item[1]), float(item[2]), float(item[3]), float(item[4])
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        new = _transform_pts(M, corners)
        nx1, ny1 = float(new[:, 0].min()), float(new[:, 1].min())
        nx2, ny2 = float(new[:, 0].max()), float(new[:, 1].max())
        out.append((text, int(nx1), int(ny1), int(nx2), int(ny2)))
    return out


# ---------------------------------------------------------------------------
# Diagonal painting
# ---------------------------------------------------------------------------

def _paint_diagonal_clusters(
    img: np.ndarray,
    pred_mask: np.ndarray,
    clusters: List[Dict],
    grey: Tuple[int, int, int] = GREY_BGR,
    padding: int = DIAG_PATCH_PAD_PX,
    min_area: int = DIAG_MIN_COMPONENT_AREA,
) -> int:
    """
    Paint each diagonal cluster's wall / window / door region as a filled
    grey quadrilateral on `img`, in place.

    Approach (per cluster):
      1. AABB of all member segment endpoints, padded.
      2. Crop pred_mask at that bbox.
      3. Rotate the crop by -cluster_angle (so the cluster runs ~horizontal).
      4. For wall/window/door classes, find connected components on the
         rotated mask; for each, take its axis-aligned bbox in the rotated
         frame, inverse-rotate the four corners, translate to image
         coords, and fill the resulting quadrilateral with grey.

    Returns the number of quads painted.
    """
    if pred_mask is None or img is None or not clusters:
        return 0
    H, W = img.shape[:2]
    n_painted = 0
    DIAG_CLASSES = (1, 2, 3)   # 1 wall, 2 window, 3 door — all become grey

    for cluster in clusters:
        if not cluster.get('is_diagonal', False):
            continue
        members = cluster.get('members') or []
        if not members:
            continue

        # 1. AABB of all member endpoints, padded
        xs: List[float] = []
        ys: List[float] = []
        for s in members:
            xs.extend([s['x1'], s['x2']])
            ys.extend([s['y1'], s['y2']])
        bx1 = max(0, int(min(xs)) - padding)
        by1 = max(0, int(min(ys)) - padding)
        bx2 = min(W, int(max(xs)) + padding)
        by2 = min(H, int(max(ys)) + padding)
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            continue

        # 2. Crop pred_mask at bbox
        crop = pred_mask[by1:by2, bx1:bx2]
        ch, cw = crop.shape[:2]

        # 3. Rotate so cluster_angle → 0°.
        # cluster_angle is in [0, 180); we want to rotate the crop CCW by
        # `-cluster_angle` (i.e., warpAffine uses CCW positive). The wall
        # then runs roughly along the x-axis in the rotated frame.
        rot_angle = float(cluster['cluster_angle'])
        cx, cy = cw / 2.0, ch / 2.0
        Mr = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        cos_a = abs(math.cos(math.radians(rot_angle)))
        sin_a = abs(math.sin(math.radians(rot_angle)))
        new_w = int(math.ceil(ch * sin_a + cw * cos_a))
        new_h = int(math.ceil(ch * cos_a + cw * sin_a))
        Mr[0, 2] += (new_w - cw) / 2.0
        Mr[1, 2] += (new_h - ch) / 2.0
        rot = cv2.warpAffine(crop, Mr, (new_w, new_h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Mr_inv = cv2.invertAffineTransform(Mr)

        # 4. Connected components per class, paint each axis-aligned bbox
        for cls in DIAG_CLASSES:
            class_mask = (rot == cls).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
            n_cc, _lbls, stats, _ = cv2.connectedComponentsWithStats(class_mask, 8)
            for i in range(1, n_cc):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                rx = int(stats[i, cv2.CC_STAT_LEFT])
                ry = int(stats[i, cv2.CC_STAT_TOP])
                rw = int(stats[i, cv2.CC_STAT_WIDTH])
                rh = int(stats[i, cv2.CC_STAT_HEIGHT])

                corners_rot = np.array([
                    [rx,      ry     ],
                    [rx + rw, ry     ],
                    [rx + rw, ry + rh],
                    [rx,      ry + rh],
                ], dtype=np.float32)
                corners_crop = _transform_pts(Mr_inv, corners_rot)
                corners_img  = corners_crop + np.array([bx1, by1], dtype=np.float32)

                cv2.fillPoly(img, [corners_img.astype(np.int32)], grey)
                n_painted += 1
    return n_painted


# ---------------------------------------------------------------------------
# U-Net wrapper
# ---------------------------------------------------------------------------

def _run_unet(img_bgr: np.ndarray, ckpt_path: str,
               img_size: int = UNET_DEFAULT_SIZE
               ) -> Optional[Dict[str, Any]]:
    """Run the project U-Net on a BGR image. Returns its native dict, or None."""
    try:
        from detect_unet import run_unet_pipeline
    except Exception as exc:
        logger.info("[diagonal] U-Net module unavailable: %s", exc)
        return None
    if not os.path.exists(ckpt_path):
        logger.info("[diagonal] U-Net checkpoint not found: %s", ckpt_path)
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        return run_unet_pipeline(rgb, ckpt_path, img_size=img_size)
    except Exception as exc:
        logger.warning("[diagonal] U-Net inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocess_floorplan(
    image_path: str,
    unet_ckpt_path: str = UNET_DEFAULT_CKPT,
    output_dir: Optional[str] = None,
    grey: Tuple[int, int, int] = GREY_BGR,
) -> Dict[str, Any]:
    """
    Run the full pre-processing pipeline on the floorplan at *image_path*.

    Output: writes a deskewed / text-erased / diagonal-painted PNG to
    ``<output_dir>/<basename>`` (defaults to ``<input_dir>/deskewed/``)
    and returns:

        {
          "deskewed_path": str,        # absolute path of the saved file
          "normal_labels": [(text,x1,y1,x2,y2), ...],   # in deskewed coords
          "rotated_labels": [(text,x1,y1,x2,y2), ...],  # in deskewed coords
          "deskew_angle":  Optional[float],  # CCW degrees, or None
          "n_diagonal_painted": int,
          "image_shape":   (H, W),
        }

    Even when no deskew is needed the output file is still written so that
    pipeline.py can always source from one location.
    """
    image_path = os.path.abspath(image_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(image_path), "deskewed")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(output_dir, os.path.basename(image_path)))

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    base = Path(image_path).stem

    # 1. Crop, 2. small-skew
    img = _crop_to_floorplan(img)
    img = _small_skew_correct(img)

    # 3. OCR (both orientations)
    normal_labels, rotated_labels = _ocr_both_orientations(img)
    logger.info("[diagonal:%s] OCR collected: %d normal + %d rotated",
                base, len(normal_labels), len(rotated_labels))

    # 4. Erase text → working image
    img_clean = _erase_text(img, [normal_labels, rotated_labels])

    # 5. U-Net on text-erased image
    unet = _run_unet(img_clean, unet_ckpt_path)
    if unet is None or 'wall_mask' not in unet or 'pred_mask' not in unet:
        logger.warning("[diagonal:%s] U-Net unavailable — saving image as-is",
                       base)
        cv2.imwrite(out_path, img)
        return {
            "deskewed_path":      out_path,
            "normal_labels":      normal_labels,
            "rotated_labels":     rotated_labels,
            "deskew_angle":       None,
            "n_diagonal_painted": 0,
            "image_shape":        (img.shape[0], img.shape[1]),
        }

    wall_mask = unet['wall_mask']
    pred_mask = unet['pred_mask']

    # 6 + 7. LSD on text-erased gray, filter, cluster
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    segs = _run_lsd(gray)
    segs = _filter_segments_by_mask(segs, wall_mask)
    clusters = _cluster_segments(segs)
    logger.info("[diagonal:%s] LSD: %d wall-aligned segs in %d clusters",
                base, len(segs), len(clusters))

    # 8. Deskew angle
    deskew = _find_deskew_angle(clusters)

    if deskew is None:
        # No tilt — the image is already aligned. Still need to scan for
        # residual diagonals (bay windows, cut corners) to paint grey.
        out_img = img.copy()
        n_painted = _paint_diagonal_clusters(out_img, pred_mask, clusters,
                                              grey=grey)
        cv2.imwrite(out_path, out_img)
        logger.info("[diagonal:%s] no deskew needed; %d diagonal patches painted",
                    base, n_painted)
        return {
            "deskewed_path":      out_path,
            "normal_labels":      normal_labels,
            "rotated_labels":     rotated_labels,
            "deskew_angle":       None,
            "n_diagonal_painted": n_painted,
            "image_shape":        (out_img.shape[0], out_img.shape[1]),
        }

    # 9. Deskew everything through M
    img_desk, M, new_size = _rotate(img, deskew,
                                     border_value=(255, 255, 255),
                                     interp=cv2.INTER_LINEAR)
    pred_desk = _rotate_mask(pred_mask, M, new_size)
    wall_desk = _rotate_mask(wall_mask, M, new_size)
    normal_labels  = _transform_labels(normal_labels,  M)
    rotated_labels = _transform_labels(rotated_labels, M)

    # 10. Find residual diagonals on the deskewed image
    img_desk_clean = _erase_text(img_desk, [normal_labels, rotated_labels])
    gray_desk = cv2.cvtColor(img_desk_clean, cv2.COLOR_BGR2GRAY)
    segs_d = _run_lsd(gray_desk)
    segs_d = _filter_segments_by_mask(segs_d, wall_desk)
    clusters_d = _cluster_segments(segs_d)

    # 11. Paint residual diagonals on the deskewed image
    n_painted = _paint_diagonal_clusters(img_desk, pred_desk, clusters_d,
                                          grey=grey)

    # 12. Save
    cv2.imwrite(out_path, img_desk)
    logger.info("[diagonal:%s] deskewed by %.3f deg, %d diagonal patches painted → %s",
                base, deskew, n_painted, out_path)

    return {
        "deskewed_path":      out_path,
        "normal_labels":      normal_labels,
        "rotated_labels":     rotated_labels,
        "deskew_angle":       deskew,
        "n_diagonal_painted": n_painted,
        "image_shape":        (img_desk.shape[0], img_desk.shape[1]),
    }


# ---------------------------------------------------------------------------
# Backward-compat shims (still imported by pipeline.py during the
# transition).  The new code path is preprocess_floorplan().
# ---------------------------------------------------------------------------

def compute_deskew(*args, **kwargs):   # pragma: no cover
    raise NotImplementedError(
        "compute_deskew has been replaced by preprocess_floorplan(); "
        "update pipeline.py to call preprocess_floorplan() instead.")


def transform_ocr_labels(labels, M):
    """Public wrapper retained for callers that already have an M matrix."""
    return _transform_labels(labels, M)


def detect_diagonal_walls(*args, **kwargs):   # pragma: no cover
    """Removed — diagonal walls are now baked into the deskewed image as
    grey filled rectangles by ``preprocess_floorplan``."""
    return []
