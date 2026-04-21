"""
Rotated-frame wall / door / window detection for diagonal building sections.

Background
----------
Standard axis-aligned detectors (projection histograms, run-length analysis,
CNN models trained on rectilinear floor plans) work best when walls are
horizontal or vertical.  When a floor-plan section contains diagonal walls,
those detectors either miss openings entirely or produce badly-fitted bounding
boxes.

Strategy
--------
1.  Collect the diagonal walls already found by ``detect_diagonal_walls``
    (Stage 1c) and extract their direction vectors.
2.  Cluster those directions into one or more dominant rotation angles
    (using a circular-mean approach on the half-circle [0°, 180°)).
3.  For each cluster angle θ:
      a.  Rotate the full image (and optional wall mask) by −θ so that the
          diagonal walls in that cluster become axis-aligned (horizontal).
      b.  Call the supplied axis-aligned detector function.  This is expected
          to be the same function used for the regular wall / door / window
          pass — no diagonal detection is performed here.
      c.  Back-project every detected segment from the rotated frame to the
          original frame as an **oriented bounding box** (OBB).  The four
          OBB corners are stored in ``WallSegment.outline``; the ``bbox``
          field is set to the axis-aligned bounding box of those corners.
4.  Deduplicate any cross-cluster overlaps and return the full list.

Coordinate conventions
-----------------------
All angles are measured counter-clockwise from the positive X axis, in
degrees, mapped to [0°, 180°) because walls are undirected line segments.
Rotation is applied around the **centre** of the original image; the canvas
is expanded so no pixels are lost during rotation.

Usage in pipeline.py (Stage 1c continuation)
---------------------------------------------
::

    from diagonal_rotation import detect_in_rotated_frames

    # After the regular Stage 1 wall pass (axis-aligned walls in
    # `existing_walls`) and the diagonal detection pass:
    diagonal_walls = detect_diagonal_walls(img, config, existing_walls,
                                           wall_mask=wall_mask)

    def _standard_detector(rotated_img, cfg, prior_walls, rotated_mask):
        # e.g. your existing detect_walls / detect_doors / detect_windows
        walls   = detect_walls(rotated_img, cfg, prior_walls, rotated_mask)
        doors   = detect_doors(rotated_img, cfg, walls)
        windows = detect_windows(rotated_img, cfg, walls)
        return walls + doors + windows

    rotated_segs = detect_in_rotated_frames(
        img, diagonal_walls, _standard_detector, config,
        wall_mask=wall_mask,
    )
    all_segments.extend(rotated_segs)
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from core.models import BBox, WallSegment

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────

ANGLE_CLUSTER_TOL_DEG: float = 10.0
"""Two diagonal walls are treated as belonging to the same rotational cluster
if their canonical angles differ by less than this many degrees."""

DEDUP_OBB_OVERLAP_THRESH: float = 0.55
"""Fraction of the smaller segment's OBB area that must overlap a
larger segment's OBB for the smaller one to be considered a duplicate."""

BORDER_FILL_VALUE: int = 255
"""Pixel value used to pad the canvas when rotating (white background,
matching typical floor-plan backgrounds)."""


# ── Angle utilities ───────────────────────────────────────────────────────────

def _canonical_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the undirected angle of the segment in [0°, 180°)."""
    a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return a


def _circular_mean_half(angles_deg: List[float]) -> float:
    """
    Compute the mean of a list of undirected angles in [0°, 180°).

    Doubles each angle to map to [0°, 360°), takes the vector mean,
    then halves back.  Handles the wraparound at 0°/180° correctly.
    """
    if not angles_deg:
        return 0.0
    doubled = [2.0 * a for a in angles_deg]
    sin_sum = sum(math.sin(math.radians(d)) for d in doubled)
    cos_sum = sum(math.cos(math.radians(d)) for d in doubled)
    mean_doubled = math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0
    return mean_doubled / 2.0


def _angular_distance_half(a: float, b: float) -> float:
    """Smallest difference between two undirected angles in [0°, 180°)."""
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def _cluster_angles(
    angles_deg: List[float],
    tol_deg: float = ANGLE_CLUSTER_TOL_DEG,
) -> List[List[float]]:
    """
    Group undirected angles into clusters whose members are within
    *tol_deg* of the running cluster mean.

    Returns a list of clusters, each cluster being a list of the original
    angle values assigned to it.  Clusters are sorted by descending size
    (largest / most-represented direction first).
    """
    if not angles_deg:
        return []

    clusters: List[List[float]] = []   # list of [mean, [members]]
    cluster_means: List[float] = []

    for a in angles_deg:
        best_idx = -1
        best_dist = tol_deg  # must beat threshold to join existing cluster

        for i, mean in enumerate(cluster_means):
            d = _angular_distance_half(a, mean)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx == -1:
            clusters.append([a])
            cluster_means.append(a)
        else:
            clusters[best_idx].append(a)
            # Update cluster mean incrementally
            cluster_means[best_idx] = _circular_mean_half(clusters[best_idx])

    # Sort largest cluster first
    clusters.sort(key=len, reverse=True)
    return clusters


# ── Image rotation helpers ────────────────────────────────────────────────────

def _rotation_matrix_expanded(
    img_h: int, img_w: int, angle_deg: float
) -> Tuple[np.ndarray, int, int]:
    """
    Compute an affine rotation matrix that rotates the image by *angle_deg*
    counter-clockwise around its centre **and** expands the canvas so the
    entire original image remains visible.

    Returns
    -------
    M : np.ndarray, shape (2, 3)
        Affine matrix ready for ``cv2.warpAffine``.
    new_w, new_h : int
        Canvas dimensions after expansion.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    rad = math.radians(angle_deg)
    cos_a = abs(math.cos(rad))
    sin_a = abs(math.sin(rad))

    new_w = int(img_h * sin_a + img_w * cos_a)
    new_h = int(img_h * cos_a + img_w * sin_a)

    # Standard OpenCV rotation matrix around (cx, cy)
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Shift so the whole image stays inside the expanded canvas
    M[0, 2] += (new_w - img_w) / 2.0
    M[1, 2] += (new_h - img_h) / 2.0

    return M, new_w, new_h


def _rotate_image_and_mask(
    img: np.ndarray,
    angle_deg: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, int, int]:
    """
    Rotate *img* (and optionally *mask*) by *angle_deg* CCW with canvas expansion.

    Returns
    -------
    rot_img : np.ndarray
        Rotated BGR image.
    rot_mask : np.ndarray or None
        Rotated mask (same canvas size), or None if mask was None.
    M : np.ndarray
        The 2×3 affine matrix used (needed for back-projection).
    new_w, new_h : int
        Size of the expanded canvas.
    """
    H, W = img.shape[:2]
    M, new_w, new_h = _rotation_matrix_expanded(H, W, angle_deg)

    rot_img = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(BORDER_FILL_VALUE,) * img.shape[2] if img.ndim == 3
                    else BORDER_FILL_VALUE,
    )

    rot_mask = None
    if mask is not None:
        rot_mask = cv2.warpAffine(
            mask, M, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return rot_img, rot_mask, M, new_w, new_h


# ── Back-projection helpers ───────────────────────────────────────────────────

def _invert_affine(M: np.ndarray) -> np.ndarray:
    """Return the 2×3 inverse of an affine transformation matrix."""
    # cv2.invertAffineTransform works directly on 2×3 matrices
    return cv2.invertAffineTransform(M)


def _transform_point(
    px: float, py: float, M: np.ndarray
) -> Tuple[float, float]:
    """Apply a 2×3 affine matrix M to point (px, py)."""
    nx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    ny = M[1, 0] * px + M[1, 1] * py + M[1, 2]
    return nx, ny


def _bbox_corners(
    x1: float, y1: float, x2: float, y2: float
) -> List[Tuple[float, float]]:
    """Return the 4 corners of an axis-aligned bounding box (TL, TR, BR, BL)."""
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _back_project_segment(
    seg: WallSegment,
    M_inv: np.ndarray,
) -> WallSegment:
    """
    Transform a ``WallSegment`` detected in the rotated frame back to the
    original image coordinate system.

    The four corners of the segment's axis-aligned bbox become an OBB in
    the original frame.  ``outline`` stores those four int corners;
    ``bbox`` is updated to the AABB of the OBB.

    The segment ``id`` is preserved; all other fields are kept as-is.
    """
    # Source corners in rotated frame
    rx1, ry1 = seg.bbox.x1, seg.bbox.y1
    rx2, ry2 = seg.bbox.x2, seg.bbox.y2
    rot_corners = _bbox_corners(rx1, ry1, rx2, ry2)

    # Back-project each corner to the original frame
    orig_corners = [_transform_point(px, py, M_inv) for px, py in rot_corners]

    xs = [p[0] for p in orig_corners]
    ys = [p[1] for p in orig_corners]

    new_bbox = BBox(
        x1=min(xs), y1=min(ys),
        x2=max(xs), y2=max(ys),
    )

    # Build the OBB outline (4 corners, integer)
    new_outline = [(int(round(p[0])), int(round(p[1]))) for p in orig_corners]

    # Rebuild the segment with updated geometry but same metadata.
    # Only the fields defined by WallSegment are set; any model-level extras
    # (e.g. door_type, window_style) must be copied manually if your
    # WallSegment dataclass/Pydantic model defines them.
    return WallSegment(
        id=seg.id,
        bbox=new_bbox,
        thickness=seg.thickness,
        is_structural=getattr(seg, "is_structural", True),
        is_diagonal=True,   # these are diagonal in original-frame terms
        outline=new_outline,
    )


# ── OBB overlap (for deduplication) ──────────────────────────────────────────

def _obb_aabb_overlap_fraction(a: WallSegment, b: WallSegment) -> float:
    """
    Quick approximation: fraction of the *smaller* segment's AABB that
    overlaps the *larger* segment's AABB.  Used for deduplication; exact OBB
    intersection is unnecessary given the typical size/spacing of features.
    """
    ax1, ay1 = a.bbox.x1, a.bbox.y1
    ax2, ay2 = a.bbox.x2, a.bbox.y2
    bx1, by1 = b.bbox.x1, b.bbox.y1
    bx2, by2 = b.bbox.x2, b.bbox.y2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    smaller_area = min(area_a, area_b)

    return inter / smaller_area if smaller_area > 0 else 0.0


def _deduplicate(
    segments: List[WallSegment],
    overlap_thresh: float = DEDUP_OBB_OVERLAP_THRESH,
) -> List[WallSegment]:
    """
    Remove duplicate segments that likely represent the same physical feature
    detected in two different rotated passes.

    Strategy: sort by bbox area descending; keep each segment only if it
    doesn't heavily overlap an already-kept segment.
    """
    by_area = sorted(
        segments,
        key=lambda s: (s.bbox.x2 - s.bbox.x1) * (s.bbox.y2 - s.bbox.y1),
        reverse=True,
    )
    kept: List[WallSegment] = []
    for cand in by_area:
        duplicate = any(
            _obb_aabb_overlap_fraction(cand, k) >= overlap_thresh
            for k in kept
        )
        if not duplicate:
            kept.append(cand)
    return kept


# ── Public API ────────────────────────────────────────────────────────────────

def extract_rotation_angles(
    diagonal_walls: List[WallSegment],
    cluster_tol_deg: float = ANGLE_CLUSTER_TOL_DEG,
) -> List[float]:
    """
    Derive the dominant rotation angles from a list of diagonal walls.

    Each diagonal wall contributes its direction angle (in [0°, 180°)).
    Angles are clustered; the circular mean of each cluster is returned as
    a rotation angle.

    The returned angles are in *ascending* order.

    Parameters
    ----------
    diagonal_walls:
        The list returned by ``detect_diagonal_walls``.
    cluster_tol_deg:
        Two walls are considered co-directional if their angles differ by
        less than this many degrees.

    Returns
    -------
    List of cluster-mean angles in [0°, 180°), one per dominant direction.
    """
    if not diagonal_walls:
        return []

    angles: List[float] = []
    for w in diagonal_walls:
        if w.outline and len(w.outline) >= 2:
            (x1, y1), (x2, y2) = w.outline[0], w.outline[-1]
        else:
            # Fall back to bbox diagonal
            x1, y1 = w.bbox.x1, w.bbox.y1
            x2, y2 = w.bbox.x2, w.bbox.y2
        angles.append(_canonical_angle_deg(x1, y1, x2, y2))

    clusters = _cluster_angles(angles, tol_deg=cluster_tol_deg)
    result = [_circular_mean_half(c) for c in clusters]
    result.sort()
    logger.debug(
        "extract_rotation_angles: %d diagonal walls → %d clusters: %s",
        len(diagonal_walls), len(result),
        [f"{a:.1f}°" for a in result],
    )
    return result


def detect_in_rotated_frames(
    img: np.ndarray,
    diagonal_walls: List[WallSegment],
    detector_fn: Callable[
        [np.ndarray, object, List[WallSegment], Optional[np.ndarray]],
        List[WallSegment],
    ],
    config: object,
    wall_mask: Optional[np.ndarray] = None,
    existing_walls: Optional[List[WallSegment]] = None,
    cluster_tol_deg: float = ANGLE_CLUSTER_TOL_DEG,
    dedup_overlap_thresh: float = DEDUP_OBB_OVERLAP_THRESH,
) -> List[WallSegment]:
    """
    For each dominant direction in *diagonal_walls*, rotate the image so
    that direction becomes horizontal, run *detector_fn*, then back-project
    all detected segments as oriented bounding boxes in the original frame.

    Parameters
    ----------
    img:
        Original BGR image (pre-processed, cropped, deskewed).
    diagonal_walls:
        Output of ``detect_diagonal_walls``.  Used only to derive rotation
        angles — these segments are NOT re-detected.
    detector_fn:
        Callable ``(rotated_img, config, prior_walls, rotated_mask)``
        → ``List[WallSegment]``.  Should run the standard axis-aligned
        wall / door / window detection **without** calling
        ``detect_diagonal_walls`` again.
    config:
        ``PipelineConfig`` passed unchanged to *detector_fn*.
    wall_mask:
        Optional binary mask rotated in sync with the image and forwarded
        to *detector_fn*.
    existing_walls:
        Axis-aligned walls already detected in Stage 1.  Passed as
        ``prior_walls`` to *detector_fn* so it can avoid duplicates.
    cluster_tol_deg:
        Angle tolerance for clustering diagonal-wall directions.
    dedup_overlap_thresh:
        Fraction of the smaller segment's bbox that must overlap another
        for it to be considered a duplicate.

    Returns
    -------
    List of ``WallSegment`` objects (``is_diagonal=True``) in the original
    coordinate system, with 4-corner OBB outlines.  Cross-cluster duplicates
    are removed before returning.
    """
    rotation_angles = extract_rotation_angles(diagonal_walls, cluster_tol_deg)
    if not rotation_angles:
        logger.info("detect_in_rotated_frames: no diagonal walls, nothing to do")
        return []

    prior_walls: List[WallSegment] = existing_walls or []
    all_back_projected: List[WallSegment] = []

    for angle_deg in rotation_angles:
        # ── 1. Rotate the image (and mask) ────────────────────────────────
        # We want the diagonal walls (at angle_deg) to become horizontal (0°).
        # A CCW rotation by −angle_deg achieves this.
        rot_angle = -angle_deg

        rot_img, rot_mask, M, rot_w, rot_h = _rotate_image_and_mask(
            img, rot_angle, mask=wall_mask
        )
        M_inv = _invert_affine(M)

        logger.info(
            "detect_in_rotated_frames: rotating by %.1f° "
            "(canvas %d×%d → %d×%d) for cluster angle %.1f°",
            rot_angle, img.shape[1], img.shape[0], rot_w, rot_h, angle_deg,
        )

        # ── 2. Forward-project prior walls into rotated frame ─────────────
        # Pass these to the detector so its internal overlap/dedup logic
        # operates in the correct (rotated) coordinate system.
        rot_prior = [_forward_project_segment(w, M) for w in prior_walls]

        # ── 3. Run standard detection on the rotated image ─────────────────
        try:
            detected: List[WallSegment] = detector_fn(
                rot_img, config, rot_prior, rot_mask
            )
        except Exception as exc:
            logger.warning(
                "detect_in_rotated_frames: detector_fn failed for angle %.1f°: %s",
                angle_deg, exc,
            )
            continue

        if not detected:
            logger.debug(
                "detect_in_rotated_frames: no segments detected at angle %.1f°",
                angle_deg,
            )
            continue

        # ── 4. Back-project detected segments to original frame ────────────
        back_projected: List[WallSegment] = []
        for seg in detected:
            try:
                bp = _back_project_segment(seg, M_inv)
                # Give each back-projected segment a fresh id to avoid
                # collisions if the same id appears in multiple clusters
                bp = WallSegment(
                    id=f"rotdet-{uuid4()}",
                    bbox=bp.bbox,
                    thickness=bp.thickness,
                    is_structural=getattr(bp, "is_structural", True),
                    is_diagonal=True,
                    outline=bp.outline,
                )
                back_projected.append(bp)
            except Exception as exc:
                logger.warning(
                    "detect_in_rotated_frames: back-projection failed "
                    "for segment %s: %s", seg.id, exc,
                )

        logger.info(
            "detect_in_rotated_frames: angle %.1f° → %d raw detections, "
            "%d back-projected",
            angle_deg, len(detected), len(back_projected),
        )
        all_back_projected.extend(back_projected)

    # ── 5. Deduplicate across clusters ────────────────────────────────────
    deduped = _deduplicate(all_back_projected, overlap_thresh=dedup_overlap_thresh)
    logger.info(
        "detect_in_rotated_frames: %d total back-projected → %d after dedup",
        len(all_back_projected), len(deduped),
    )
    return deduped


# ── Forward-projection helper (for transforming prior walls into rot. frame) ──

def _forward_project_segment(
    seg: WallSegment,
    M: np.ndarray,
) -> WallSegment:
    """
    Project a ``WallSegment`` from the original frame into the rotated frame
    using forward affine matrix *M*.

    Works the same way as ``_back_project_segment`` but uses M directly
    instead of its inverse.
    """
    rx1, ry1 = seg.bbox.x1, seg.bbox.y1
    rx2, ry2 = seg.bbox.x2, seg.bbox.y2
    corners = _bbox_corners(rx1, ry1, rx2, ry2)

    projected = [_transform_point(px, py, M) for px, py in corners]
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]

    return WallSegment(
        id=seg.id,
        bbox=BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
        thickness=seg.thickness,
        is_structural=getattr(seg, "is_structural", True),
        is_diagonal=getattr(seg, "is_diagonal", False),
        outline=[(int(round(p[0])), int(round(p[1]))) for p in projected],
    )