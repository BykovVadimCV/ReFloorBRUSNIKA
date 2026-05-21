"""Door swing arc detection via classical morphology (no deep learning)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("DoorArcDetector")

# ---------------------------------------------------------------------------
# IoU helper (used by filter_gaps_against_yolo_doors and DoorFusion)
# ---------------------------------------------------------------------------
def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0

# ---------------------------------------------------------------------------
# Try to import project-level Theme constants; fall back to sensible defaults
# ---------------------------------------------------------------------------
try:
    from core.theme import Theme  # type: ignore
    MORPH_KERNEL_SMALL = getattr(Theme, "MORPH_KERNEL_SMALL", 3)
    MORPH_KERNEL_MEDIUM = getattr(Theme, "MORPH_KERNEL_MEDIUM", 5)
    FURNITURE_COLOR = getattr(Theme, "FURNITURE_COLOR", (180, 180, 180))
    STRUCT_WALL_THK = getattr(Theme, "STRUCT_WALL_THICKNESS", 8)
except ImportError:
    MORPH_KERNEL_SMALL = 3
    MORPH_KERNEL_MEDIUM = 5
    FURNITURE_COLOR = (180, 180, 180)
    STRUCT_WALL_THK = 8


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

class SwingDirection(str, Enum):
    """Human-readable swing direction from the room interior's perspective."""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class ArcCandidate:
    """
    Internal working representation of a potential door-swing arc.
    """
    contour: np.ndarray
    # Geometry (filled in Stage 3)
    center: Optional[Tuple[float, float]] = None
    radius_px: float = 0.0
    radius_m: float = 0.0
    radial_deviation: float = 1.0
    angle_span_deg: float = 0.0
    arc_length_fraction: float = 0.0
    start_angle_deg: float = 0.0
    end_angle_deg: float = 0.0
    # Endpoints (filled in Stage 4)
    endpoint_a: Optional[Tuple[int, int]] = None
    endpoint_b: Optional[Tuple[int, int]] = None
    hinge_point: Optional[Tuple[int, int]] = None
    free_tip: Optional[Tuple[int, int]] = None
    direction: SwingDirection = SwingDirection.UNKNOWN
    opening_angle_deg: float = 90.0
    # Validation flags
    wall_attached: bool = False
    inside_gap: bool = False
    near_yolo: bool = False
    furniture_overlap: bool = False
    # Scoring
    confidence: float = 0.0
    rejection_reason: Optional[str] = None

    @property
    def rejected(self) -> bool:
        return self.rejection_reason is not None


@dataclass
class DetectedDoorArc:
    """
    Final public output — mirrors style of DetectedWindow.
    Serializable, renderer-friendly, SweetHome3D-exportable.
    """
    id: int
    hinge: Tuple[int, int]
    center: Tuple[int, int]
    radius_px: float
    radius_m: float
    angle_span_deg: float
    start_angle_deg: float
    end_angle_deg: float
    direction: str                  # "left" / "right" / "unknown"
    confidence: float               # 0.0 – 1.0
    matched_opening_idx: int = -1   # index into candidate_gaps list
    # Extra metadata for SH3D export
    wall_normal_deg: float = 0.0
    swing_clockwise: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for JSON / CSV serialization."""
        return {
            "id": self.id,
            "hinge_x": self.hinge[0],
            "hinge_y": self.hinge[1],
            "center_x": self.center[0],
            "center_y": self.center[1],
            "radius_px": round(self.radius_px, 1),
            "radius_m": round(self.radius_m, 3),
            "angle_span_deg": round(self.angle_span_deg, 1),
            "start_angle_deg": round(self.start_angle_deg, 1),
            "end_angle_deg": round(self.end_angle_deg, 1),
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "matched_opening_idx": self.matched_opening_idx,
            "wall_normal_deg": round(self.wall_normal_deg, 1),
            "swing_clockwise": self.swing_clockwise,
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class DoorArcDetector:
    """
    Pure-classical morphological door-swing-arc detector.

    Parameters
    ----------
    min_arc_radius_m : float
        Minimum realistic door-leaf radius in metres (default 0.6 m).
    max_arc_radius_m : float
        Maximum realistic door-leaf radius in metres (default 1.2 m).
    attachment_proximity_px : int
        Max distance (px) an arc endpoint can be from wall mask to count as
        "wall-attached" (default 12).
    enable_furniture_filter : bool
        Whether to suppress arcs that overlap furniture rectangles.
    yolo_expand_px : int
        How many pixels to expand each YOLO door bbox for ROI masking.
    gap_expand_px : int
        How many pixels to expand each gap rectangle for ROI masking.
    debug : bool
        If True, writes intermediate images to ``debug_output_dir``.
    debug_output_dir : str | Path
        Folder for debug images.
    """

    # ── Construction ──────────────────────────────────────────────────────
    def __init__(
        self,
        min_arc_radius_m: float = 0.6,
        max_arc_radius_m: float = 1.2,
        attachment_proximity_px: int = 12,
        enable_furniture_filter: bool = True,
        yolo_expand_px: int = 30,
        gap_expand_px: int = 30,
        debug: bool = False,
        debug_output_dir: Union[str, Path] = "output/debug",
    ):
        # Real-world radius bounds (converted to px at detect-time)
        self.min_arc_radius_m = min_arc_radius_m
        self.max_arc_radius_m = max_arc_radius_m

        # Proximity
        self.attachment_proximity_px = attachment_proximity_px
        self.yolo_expand_px = yolo_expand_px
        self.gap_expand_px = gap_expand_px

        # Feature toggles
        self.enable_furniture_filter = enable_furniture_filter
        self.debug = debug
        self.debug_output_dir = Path(debug_output_dir)

        # Maximum fraction of image area a YOLO door bbox may occupy.
        self.max_yolo_area_fraction: float = 0.05

        # ── Stage 0 thresholds ───────────────────────────────────────────
        self.adaptive_block_size: int = 11
        self.adaptive_c: int = 2
        self.dilation_kernel_size: int = 3
        self.dilation_iterations: int = 1

        # ── Stage 1 thresholds ───────────────────────────────────────────
        self.open_kernel_size: int = MORPH_KERNEL_SMALL
        self.open_iterations: int = 1
        self.close_kernel_size: int = MORPH_KERNEL_SMALL
        self.close_iterations: int = 2
        self.gradient_kernel_size: int = MORPH_KERNEL_MEDIUM
        self.curvature_threshold: int = 30
        self.gaussian_sigma: float = 1.0

        # ── Stage 2 thresholds ───────────────────────────────────────────
        self.min_contour_area: int = 80
        self.max_contour_area: int = 18_000
        self.min_perimeter: int = 30
        self.max_bbox_aspect: float = 3.0
        self.min_defect_depth: int = 8

        # ── Stage 3 thresholds ───────────────────────────────────────────
        self.max_radial_cv: float = 0.085
        self.min_angle_span_deg: float = 65.0
        self.max_angle_span_deg: float = 115.0
        self.min_arc_fraction: float = 0.13
        self.max_arc_fraction: float = 0.40
        self.circle_fit_sample_count: int = 200

        # ── Stage 4 thresholds ───────────────────────────────────────────
        self.hinge_wall_roi_px: int = 12
        self.hinge_hough_min_line: int = 8
        self.hinge_hough_max_gap: int = 5

        # ── Stage 5 thresholds ───────────────────────────────────────────
        self.gap_center_tolerance_px: int = 8
        self.yolo_center_tolerance_px: int = 30
        self.wall_hough_threshold: int = 10

        # ── Stage 6 thresholds ───────────────────────────────────────────
        self.radius_median_tolerance: float = 0.25

        # ── Stage 7 thresholds ───────────────────────────────────────────
        self.nms_hinge_iou_threshold: float = 0.4
        self.nms_neighborhood_px: int = 30

        # ── Runtime bookkeeping ──────────────────────────────────────────
        self._debug_images: Dict[str, np.ndarray] = {}
        self._timing: Dict[str, float] = {}

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
        """Universal safe mask normalizer – prevents all future 0xC0000005 crashes."""
        if mask is None or mask.size == 0:
            raise ValueError(f"{name} is empty")
        mask = mask.copy()
        if mask.ndim == 3:
            if mask.shape[2] == 4:   # RGBA
                mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
            else:                    # BGR/RGB
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask.astype(np.uint8)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════
    def detect(
            self,
            img: np.ndarray,
            wall_outline_mask: np.ndarray,
            candidate_gaps: Optional[List] = None,
            yolo_doors: Optional[List] = None,
            pixel_scale: float = 0.01,
            furniture_mask: Optional[np.ndarray] = None,
            ocr_boxes: Optional[List] = None,
            base_name: str = "unknown",
            roi_mask: Optional[np.ndarray] = None,  # already present in your code
    ) -> List[DetectedDoorArc]:
        """
        Run the full 8-stage pipeline and return detected door arcs.

        Parameters
        ----------
        img : np.ndarray
            Original BGR floor-plan image.
        wall_outline_mask : np.ndarray
            Binary mask (255 = wall) from BrusnikaWallDetector.
        candidate_gaps : list
            Gap/opening objects from GapDetector.
        yolo_doors : list
            YOLO detection objects.
        pixel_scale : float
            Metres per pixel (from RoomSizeAnalyzer).
        furniture_mask : np.ndarray, optional
            Binary mask (255 = furniture).
        ocr_boxes : list of (x, y, w, h), optional
            Bounding boxes of OCR room labels to exclude.
        base_name : str
            Used for debug-image filenames.
        roi_mask : np.ndarray, optional   ← NEW PARAMETER
            External ROI mask. If provided, used instead of building from
            YOLO+gaps in Stage 0. Enables DoorFusion Phase 0.

        Returns
        -------
        list of DetectedDoorArc
        """
        t_total = time.perf_counter()
        self._debug_images.clear()
        self._timing.clear()

        if img is None or img.size == 0:
            logger.error("Input image is empty")
            return []

        h, w = img.shape[:2]

        # === CRITICAL FIX: Normalize wall_outline_mask to single-channel binary ===
        if wall_outline_mask is None or wall_outline_mask.size == 0:
            logger.warning("wall_outline_mask is empty → using zero mask")
            wall_outline_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = wall_outline_mask.copy()
            if mask.ndim == 3:
                if mask.shape[2] == 4:  # RGBA
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
                else:  # BGR or RGB
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # Force binary
            _, wall_outline_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Shape safety (prevents segfault on resize mismatch)
        if wall_outline_mask.shape[:2] != (h, w):
            logger.warning("wall_outline_mask shape mismatch → resizing")
            wall_outline_mask = cv2.resize(
                wall_outline_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )

        # Normalise inputs
        gaps = self._normalise_gaps(candidate_gaps)
        doors = self._normalise_yolo(yolo_doors)
        furniture_rects = self._mask_to_rects(furniture_mask) if furniture_mask is not None else []
        ocr_rects = ocr_boxes or []

        # ── Oversized YOLO bbox gate ──────────────────────────────────────
        if doors:
            img_area = h * w
            doors = [
                d for d in doors
                if (d[2] - d[0]) * (d[3] - d[1]) <= self.max_yolo_area_fraction * img_area
            ]

        # Pixel-domain radius bounds
        min_r_px = self.min_arc_radius_m / pixel_scale if pixel_scale > 0 else 20
        max_r_px = self.max_arc_radius_m / pixel_scale if pixel_scale > 0 else 400

        # ── Stage 0 ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        clean_bw = self._stage0_preprocess(img, gaps, doors, h, w, roi_mask=roi_mask)
        self._timing["stage0"] = time.perf_counter() - t0

        # ── Stage 1 ──────────────────────────────────────────────────────
        t1 = time.perf_counter()
        enhanced = self._stage1_morphological_enhancement(clean_bw)
        self._timing["stage1"] = time.perf_counter() - t1

        # ── Stage 2 ──────────────────────────────────────────────────────
        t2 = time.perf_counter()
        candidates = self._stage2_contour_extraction(enhanced)
        self._timing["stage2"] = time.perf_counter() - t2
        logger.debug("Stage 2: %d contours survived initial pruning", len(candidates))

        # ── Stage 3 ──────────────────────────────────────────────────────
        t3 = time.perf_counter()
        self._stage3_circle_fitting(candidates, min_r_px, max_r_px, pixel_scale)
        survivors = [c for c in candidates if not c.rejected]
        self._timing["stage3"] = time.perf_counter() - t3
        logger.debug("Stage 3: %d arcs passed circle fitting", len(survivors))

        # ── Stage 4 ──────────────────────────────────────────────────────
        t4 = time.perf_counter()
        self._stage4_hinge_recovery(survivors, wall_outline_mask)
        self._timing["stage4"] = time.perf_counter() - t4

        # ── Stage 5 ──────────────────────────────────────────────────────
        t5 = time.perf_counter()
        self._stage5_wall_attachment(survivors, wall_outline_mask, gaps, doors)
        survivors = [c for c in survivors if not c.rejected]
        self._timing["stage5"] = time.perf_counter() - t5
        logger.debug("Stage 5: %d arcs passed attachment validation", len(survivors))

        # ── Stage 6 ──────────────────────────────────────────────────────
        t6 = time.perf_counter()
        self._stage6_false_positive_suppression(
            survivors, furniture_rects, ocr_rects, gaps, pixel_scale
        )
        survivors = [c for c in survivors if not c.rejected]
        self._timing["stage6"] = time.perf_counter() - t6
        logger.debug("Stage 6: %d arcs after FP suppression", len(survivors))

        # ── Stage 7 ──────────────────────────────────────────────────────
        t7 = time.perf_counter()
        survivors = self._stage7_nms(survivors)
        self._timing["stage7"] = time.perf_counter() - t7
        logger.debug("Stage 7: %d arcs after NMS", len(survivors))

        # ── Stage 8 ──────────────────────────────────────────────────────
        t8 = time.perf_counter()
        results = self._stage8_output(survivors, gaps, pixel_scale)
        self._timing["stage8"] = time.perf_counter() - t8

        self._timing["total"] = time.perf_counter() - t_total
        logger.info(
            "DoorArcDetector: %d arcs detected in %.1f ms",
            len(results),
            self._timing["total"] * 1000,
        )

        # Debug output
        if self.debug:
            self._save_debug_images(img, candidates, survivors, results, base_name)

        return results

    # ══════════════════════════════════════════════════════════════════════
    # STAGE IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════

    # ── Stage 0: Context-Aware Preprocessing ─────────────────────────────
    def _stage0_preprocess(
        self,
        img: np.ndarray,
        gaps: List[Tuple[int, int, int, int]],
        doors: List[Tuple[int, int, int, int]],
        h: int,
        w: int,
        roi_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a clean binary image restricted to regions of interest.

        If roi_mask is provided externally (e.g. from DoorFusion Phase 0),
        use it directly instead of building from YOLO + gaps.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

        # Adaptive threshold (inverted: arcs → white)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c,
        )

        # === MODIFIED: Use external roi_mask if provided ===
        if roi_mask is not None:
            # Ensure roi_mask matches image dimensions
            if roi_mask.shape[:2] != (h, w):
                roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            final_roi = roi_mask
        else:
            # Build ROI mask from YOLO + gaps (original behavior)
            final_roi = np.zeros((h, w), dtype=np.uint8)
            has_any_roi = False

            for (x1, y1, x2, y2) in doors:
                ex = self.yolo_expand_px
                rx1 = max(0, x1 - ex)
                ry1 = max(0, y1 - ex)
                rx2 = min(w, x2 + ex)
                ry2 = min(h, y2 + ex)
                final_roi[ry1:ry2, rx1:rx2] = 255
                has_any_roi = True

            for (gx, gy, gw, gh) in gaps:
                ex = self.gap_expand_px
                rx1 = max(0, gx - ex)
                ry1 = max(0, gy - ex)
                rx2 = min(w, gx + gw + ex)
                ry2 = min(h, gy + gh + ex)
                final_roi[ry1:ry2, rx1:rx2] = 255
                has_any_roi = True

            # No YOLO doors or gaps → nothing to scan; leave final_roi all-zero
            # so that subsequent stages find zero contours and return immediately.
            # (Previously this fell back to scanning the full image, which caused
            # OOM on large architectural drawings with thousands of contours.)
            if not has_any_roi:
                logger.debug("No YOLO doors or gaps provided – skipping arc scan")

        # Apply ROI mask
        clean_bw = cv2.bitwise_and(binary, final_roi)

        # Light dilation to connect scan-line breaks
        kern = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.dilation_kernel_size, self.dilation_kernel_size),
        )
        clean_bw = cv2.dilate(clean_bw, kern, iterations=self.dilation_iterations)

        if self.debug:
            self._debug_images["stage0_clean_bw"] = clean_bw.copy()

        return clean_bw

    # ── Stage 1: Multi-Scale Morphological Enhancement ───────────────────
    def _stage1_morphological_enhancement(self, clean_bw: np.ndarray) -> np.ndarray:
        # Opening: remove tiny noise
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.open_kernel_size, self.open_kernel_size),
        )
        opened = cv2.morphologyEx(clean_bw, cv2.MORPH_OPEN, k_open, iterations=self.open_iterations)

        # Closing: connect micro-gaps
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.close_kernel_size, self.close_kernel_size),
        )
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=self.close_iterations)

        # Morphological gradient → curvature proxy
        k_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.gradient_kernel_size, self.gradient_kernel_size),
        )
        dilated = cv2.dilate(closed, k_grad)
        eroded = cv2.erode(closed, k_grad)
        gradient = cv2.subtract(dilated, eroded)

        # Optional blur to suppress aliasing
        if self.gaussian_sigma > 0:
            gradient = cv2.GaussianBlur(
                gradient,
                (0, 0),
                sigmaX=self.gaussian_sigma,
            )

        # Curvature map
        curvature_map = cv2.subtract(closed.astype(np.int16), gradient.astype(np.int16))
        curvature_map = np.clip(curvature_map, 0, 255).astype(np.uint8)

        # Threshold curvature map
        _, curv_thresh = cv2.threshold(
            curvature_map, self.curvature_threshold, 255, cv2.THRESH_BINARY
        )

        # Combine
        enhanced = cv2.bitwise_or(closed, curv_thresh)

        if self.debug:
            self._debug_images["stage1_curvature_map"] = curvature_map.copy()
            self._debug_images["stage1_enhanced"] = enhanced.copy()

        return enhanced

    # ── Stage 2: Contour Extraction & Initial Pruning ────────────────────
    def _stage2_contour_extraction(self, enhanced: np.ndarray) -> List[ArcCandidate]:
        contours, _ = cv2.findContours(
            enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        candidates: List[ArcCandidate] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            if area > self.max_contour_area:
                continue
            peri = cv2.arcLength(cnt, closed=False)
            if peri < self.min_perimeter:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh == 0 or bw == 0:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > self.max_bbox_aspect:
                continue
            if len(cnt) < 5:
                continue
            hull_idx = cv2.convexHull(cnt, returnPoints=False)
            if hull_idx is None or len(hull_idx) < 4:
                continue
            try:
                defects = cv2.convexityDefects(cnt, hull_idx)
            except cv2.error:
                continue
            if defects is None:
                continue
            max_depth = 0.0
            for d in defects:
                depth = d[0][3] / 256.0
                if depth > max_depth:
                    max_depth = depth
            if max_depth < self.min_defect_depth:
                continue
            candidates.append(ArcCandidate(contour=cnt))
        return candidates

    # ── Stage 3: Robust Partial-Circle Fitting ───────────────────────────
    def _stage3_circle_fitting(
        self,
        candidates: List[ArcCandidate],
        min_r_px: float,
        max_r_px: float,
        pixel_scale: float,
    ) -> None:
        for cand in candidates:
            if cand.rejected:
                continue
            cnt = cand.contour
            pts = cnt.reshape(-1, 2).astype(np.float64)

            (cx, cy), enc_r = cv2.minEnclosingCircle(cnt)
            if enc_r < 1.0:
                cand.rejection_reason = "enc_radius_zero"
                continue
            if enc_r < min_r_px * 0.7 or enc_r > max_r_px * 1.3:
                cand.rejection_reason = f"radius_out_of_range ({enc_r:.1f} px)"
                continue

            n = len(pts)
            if n > self.circle_fit_sample_count:
                idx = np.linspace(0, n - 1, self.circle_fit_sample_count, dtype=int)
                sampled = pts[idx]
            else:
                sampled = pts

            dx = sampled[:, 0] - cx
            dy = sampled[:, 1] - cy
            dists = np.sqrt(dx * dx + dy * dy)
            mean_r = np.mean(dists)
            if mean_r < 1.0:
                cand.rejection_reason = "mean_radius_zero"
                continue
            std_r = np.std(dists)
            cv_r = std_r / mean_r
            if cv_r > self.max_radial_cv:
                cand.rejection_reason = f"radial_cv={cv_r:.4f} > {self.max_radial_cv}"
                continue

            angles = np.arctan2(dy, dx)
            angles_deg = np.degrees(angles)
            angles_sorted = np.sort(angles_deg)

            if len(angles_sorted) < 3:
                cand.rejection_reason = "too_few_angle_samples"
                continue

            gaps_between = np.diff(angles_sorted)
            max_gap = np.max(gaps_between) if len(gaps_between) > 0 else 360.0
            wrap_gap = 360.0 - (angles_sorted[-1] - angles_sorted[0])
            largest_gap = max(max_gap, wrap_gap)
            angular_span = 360.0 - largest_gap

            if angular_span < self.min_angle_span_deg:
                cand.rejection_reason = f"angle_span={angular_span:.1f}° < {self.min_angle_span_deg}°"
                continue
            if angular_span > self.max_angle_span_deg:
                cand.rejection_reason = f"angle_span={angular_span:.1f}° > {self.max_angle_span_deg}°"
                continue

            peri = cv2.arcLength(cnt, closed=False)
            full_circ = 2.0 * math.pi * mean_r
            arc_frac = peri / full_circ if full_circ > 0 else 0

            if arc_frac < self.min_arc_fraction:
                cand.rejection_reason = f"arc_frac={arc_frac:.3f} < {self.min_arc_fraction}"
                continue
            if arc_frac > self.max_arc_fraction:
                cand.rejection_reason = f"arc_frac={arc_frac:.3f} > {self.max_arc_fraction}"
                continue

            if wrap_gap >= max_gap:
                start_deg = angles_sorted[0]
                end_deg = angles_sorted[-1]
            else:
                gap_start_idx = np.argmax(gaps_between)
                start_deg = angles_sorted[gap_start_idx + 1]
                end_deg = angles_sorted[gap_start_idx]

            cand.center = (float(cx), float(cy))
            cand.radius_px = float(mean_r)
            cand.radius_m = float(mean_r * pixel_scale) if pixel_scale > 0 else 0.0
            cand.radial_deviation = float(cv_r)
            cand.angle_span_deg = float(angular_span)
            cand.arc_length_fraction = float(arc_frac)
            cand.start_angle_deg = float(start_deg)
            cand.end_angle_deg = float(end_deg)

    # ── Stage 4: Hinge & Swing Direction Recovery ────────────────────────
    def _stage4_hinge_recovery(
        self,
        candidates: List[ArcCandidate],
        wall_mask: np.ndarray,
    ) -> None:
        h_img, w_img = wall_mask.shape[:2]
        for cand in candidates:
            if cand.rejected:
                continue
            pts = cand.contour.reshape(-1, 2)
            if len(pts) < 2:
                cand.rejection_reason = "contour_too_short_for_endpoints"
                continue
            ep_a = tuple(pts[0])
            ep_b = tuple(pts[-1])
            cand.endpoint_a = ep_a
            cand.endpoint_b = ep_b

            score_a = self._wall_proximity_score(ep_a, wall_mask, h_img, w_img)
            score_b = self._wall_proximity_score(ep_b, wall_mask, h_img, w_img)
            if score_a >= score_b:
                cand.hinge_point = ep_a
                cand.free_tip = ep_b
            else:
                cand.hinge_point = ep_b
                cand.free_tip = ep_a

            if cand.center is not None and cand.hinge_point is not None and cand.free_tip is not None:
                cx, cy = cand.center
                hx, hy = cand.hinge_point
                fx, fy = cand.free_tip
                v1 = (cx - hx, cy - hy)
                v2 = (fx - hx, fy - hy)
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross > 0:
                    cand.direction = SwingDirection.LEFT
                    cand.swing_clockwise = False
                elif cross < 0:
                    cand.direction = SwingDirection.RIGHT
                    cand.swing_clockwise = True
                else:
                    cand.direction = SwingDirection.UNKNOWN
                    cand.swing_clockwise = True

    def _wall_proximity_score(
        self,
        point: Tuple[int, int],
        wall_mask: np.ndarray,
        h: int,
        w: int,
    ) -> float:
        px, py = int(point[0]), int(point[1])
        r = self.hinge_wall_roi_px
        x1 = max(0, px - r)
        y1 = max(0, py - r)
        x2 = min(w, px + r)
        y2 = min(h, py + r)
        roi = wall_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        density = np.count_nonzero(roi) / roi.size
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=self.wall_hough_threshold,
            minLineLength=self.hinge_hough_min_line,
            maxLineGap=self.hinge_hough_max_gap,
        )
        line_score = 1.0 if lines is not None and len(lines) > 0 else 0.0
        return density * 0.6 + line_score * 0.4

    # ── Stage 5: Wall-Attachment & Opening Validation ────────────────────
    def _stage5_wall_attachment(
        self,
        candidates: List[ArcCandidate],
        wall_mask: np.ndarray,
        gaps: List[Tuple[int, int, int, int]],
        doors: List[Tuple[int, int, int, int]],
    ) -> None:
        h_img, w_img = wall_mask.shape[:2]
        for cand in candidates:
            if cand.rejected:
                continue
            if cand.endpoint_a is not None and cand.endpoint_b is not None:
                score_a = self._wall_proximity_score(cand.endpoint_a, wall_mask, h_img, w_img)
                score_b = self._wall_proximity_score(cand.endpoint_b, wall_mask, h_img, w_img)
                cand.wall_attached = (score_a > 0.15) and (score_b > 0.15)
            else:
                cand.wall_attached = False
            if cand.center is not None:
                cx, cy = cand.center
                cand.inside_gap = any(
                    self._point_near_rect(cx, cy, g, self.gap_center_tolerance_px)
                    for g in gaps
                )
                if doors:
                    cand.near_yolo = any(
                        self._point_near_rect_xyxy(cx, cy, d, self.yolo_center_tolerance_px)
                        for d in doors
                    )
                else:
                    cand.near_yolo = True
            else:
                cand.inside_gap = False
                cand.near_yolo = False
            if not cand.wall_attached:
                cand.rejection_reason = "not_wall_attached"
            elif not cand.inside_gap:
                cand.rejection_reason = "not_inside_gap"
            elif not cand.near_yolo:
                cand.rejection_reason = "not_near_yolo_door"

    # ── Stage 6: False-Positive Suppression ──────────────────────────────
    def _stage6_false_positive_suppression(
        self,
        candidates: List[ArcCandidate],
        furniture_rects: List[Tuple[int, int, int, int]],
        ocr_rects: List[Tuple[int, int, int, int]],
        gaps: List[Tuple[int, int, int, int]],
        pixel_scale: float,
    ) -> None:
        if gaps:
            widths = [max(g[2], g[3]) for g in gaps]
            median_width = float(np.median(widths))
        else:
            median_width = 0.0

        for cand in candidates:
            if cand.rejected:
                continue
            cx, cy = cand.center if cand.center else (0, 0)
            if self.enable_furniture_filter and furniture_rects:
                for fr in furniture_rects:
                    if self._point_inside_rect(cx, cy, fr):
                        cand.rejection_reason = "inside_furniture"
                        cand.furniture_overlap = True
                        break
            if cand.rejected:
                continue
            for ocr in ocr_rects:
                if self._point_inside_rect(cx, cy, ocr):
                    cand.rejection_reason = "inside_ocr_label"
                    break
            if cand.rejected:
                continue
            if median_width > 0 and cand.radius_px > 0:
                deviation = abs(cand.radius_px - median_width) / median_width
                if deviation > self.radius_median_tolerance:
                    cand.rejection_reason = (
                        f"radius_deviation={deviation:.2f} > {self.radius_median_tolerance}"
                    )
                    continue

    # ── Stage 7: Non-Maximum Suppression & Merging ───────────────────────
    def _stage7_nms(self, candidates: List[ArcCandidate]) -> List[ArcCandidate]:
        if len(candidates) <= 1:
            return [c for c in candidates if not c.rejected]
        active = sorted(
            [c for c in candidates if not c.rejected],
            key=lambda c: c.radial_deviation,
        )
        kept: List[ArcCandidate] = []
        suppressed: set = set()
        for i, ci in enumerate(active):
            if i in suppressed:
                continue
            kept.append(ci)
            for j in range(i + 1, len(active)):
                if j in suppressed:
                    continue
                cj = active[j]
                if self._hinge_iou(ci, cj) > self.nms_hinge_iou_threshold:
                    suppressed.add(j)
                    logger.debug(
                        "NMS: suppressed arc (radial_cv=%.4f) in favour of (radial_cv=%.4f)",
                        cj.radial_deviation,
                        ci.radial_deviation,
                    )
        return kept

    def _hinge_iou(self, a: ArcCandidate, b: ArcCandidate) -> float:
        if a.hinge_point is None or b.hinge_point is None:
            return 0.0
        ax, ay = a.hinge_point
        bx, by = b.hinge_point
        dist = math.hypot(ax - bx, ay - by)
        r = self.nms_neighborhood_px
        if dist >= 2 * r:
            return 0.0
        overlap = 2 * r - dist
        union = 2 * r + dist
        return overlap / union if union > 0 else 0.0

    # ── Stage 8: Output Enrichment ───────────────────────────────────────
    def _stage8_output(
        self,
        candidates: List[ArcCandidate],
        gaps: List[Tuple[int, int, int, int]],
        pixel_scale: float,
    ) -> List[DetectedDoorArc]:
        results: List[DetectedDoorArc] = []
        for idx, cand in enumerate(candidates):
            if cand.rejected:
                continue
            conf = self._compute_confidence(cand)
            cand.confidence = conf
            matched_gap = -1
            if cand.center is not None and gaps:
                cx, cy = cand.center
                best_dist = float("inf")
                for gi, g in enumerate(gaps):
                    gcx = g[0] + g[2] / 2
                    gcy = g[1] + g[3] / 2
                    d = math.hypot(cx - gcx, cy - gcy)
                    if d < best_dist:
                        best_dist = d
                        matched_gap = gi
            wall_normal = self._estimate_wall_normal(cand)
            arc = DetectedDoorArc(
                id=idx,
                hinge=cand.hinge_point or (0, 0),
                center=(int(cand.center[0]), int(cand.center[1])) if cand.center else (0, 0),
                radius_px=cand.radius_px,
                radius_m=cand.radius_m,
                angle_span_deg=cand.angle_span_deg,
                start_angle_deg=cand.start_angle_deg,
                end_angle_deg=cand.end_angle_deg,
                direction=cand.direction.value,
                confidence=conf,
                matched_opening_idx=matched_gap,
                wall_normal_deg=wall_normal,
                swing_clockwise=getattr(cand, "swing_clockwise", True),
            )
            results.append(arc)
        return results

    def _compute_confidence(self, cand: ArcCandidate) -> float:
        fit_score = max(0.0, 1.0 - (cand.radial_deviation / self.max_radial_cv))
        angle_diff = abs(cand.angle_span_deg - 90.0)
        angle_score = max(0.0, 1.0 - angle_diff / 25.0)
        frac_diff = abs(cand.arc_length_fraction - 0.25)
        frac_score = max(0.0, 1.0 - frac_diff / 0.12)
        wall_bonus = 0.1 if cand.wall_attached else 0.0
        yolo_bonus = 0.05 if cand.near_yolo else 0.0
        raw = (
            0.35 * fit_score
            + 0.30 * angle_score
            + 0.20 * frac_score
            + wall_bonus
            + yolo_bonus
        )
        return float(np.clip(raw, 0.0, 1.0))

    def _estimate_wall_normal(self, cand: ArcCandidate) -> float:
        if cand.hinge_point is None or cand.center is None:
            return 0.0
        hx, hy = cand.hinge_point
        cx, cy = cand.center
        angle = math.degrees(math.atan2(cy - hy, cx - hx))
        return (angle + 90.0) % 360.0

    # ══════════════════════════════════════════════════════════════════════
    # GEOMETRY HELPERS
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _point_near_rect(
        px: float, py: float, rect: Tuple[int, int, int, int], tol: int
    ) -> bool:
        x, y, w, h = rect
        return (x - tol <= px <= x + w + tol) and (y - tol <= py <= y + h + tol)

    @staticmethod
    def _point_near_rect_xyxy(
        px: float, py: float, rect: Tuple[int, int, int, int], tol: int
    ) -> bool:
        x1, y1, x2, y2 = rect
        return (x1 - tol <= px <= x2 + tol) and (y1 - tol <= py <= y2 + tol)

    @staticmethod
    def _point_inside_rect(
        px: float, py: float, rect: Tuple[int, int, int, int]
    ) -> bool:
        x, y, w, h = rect
        return (x <= px <= x + w) and (y <= py <= y + h)

    # ── Input normalisation ──────────────────────────────────────────────
    @staticmethod
    def _normalise_gaps(
        candidate_gaps: Optional[List[Any]],
    ) -> List[Tuple[int, int, int, int]]:
        if not candidate_gaps:
            return []
        result = []
        for g in candidate_gaps:
            if isinstance(g, (list, tuple)) and len(g) >= 4:
                result.append((int(g[0]), int(g[1]), int(g[2]), int(g[3])))
            elif hasattr(g, "x") and hasattr(g, "y") and hasattr(g, "w") and hasattr(g, "h"):
                result.append((int(g.x), int(g.y), int(g.w), int(g.h)))
            elif hasattr(g, "bbox"):
                b = g.bbox
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return result

    @staticmethod
    def _normalise_yolo(
        yolo_doors: Optional[List[Any]],
    ) -> List[Tuple[int, int, int, int]]:
        if not yolo_doors:
            return []
        result = []
        for d in yolo_doors:
            if isinstance(d, (list, tuple)) and len(d) >= 4:
                result.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))
            elif hasattr(d, "x1"):
                result.append((int(d.x1), int(d.y1), int(d.x2), int(d.y2)))
            elif hasattr(d, "bbox"):
                b = d.bbox
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
            elif hasattr(d, "xyxy"):
                b = d.xyxy
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return result

    @staticmethod
    def _mask_to_rects(
        mask: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        if mask is None or mask.size == 0:
            return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 5:
                rects.append((x, y, w, h))
        return rects

    # ══════════════════════════════════════════════════════════════════════
    # ENHANCED ROI BUILDER (for DoorFusion Phase 0)
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def build_enhanced_roi_mask(
        h: int,
        w: int,
        yolo_doors: Optional[List[Any]] = None,
        gaps: Optional[List[Any]] = None,
        wall_outline_mask: Optional[np.ndarray] = None,
        yolo_expand_px: int = 50,
        gap_expand_px: int = 40,
    ) -> np.ndarray:
        """
        Build an enhanced ROI mask for Phase 0 of DoorFusion.

        roi = dilated(YOLO ±50 px) |
              dilated(gaps ±40 px) |
              wall-centerline buffer (dilate outline_mask 5×5, erode 9×9)

        Returns binary mask (255 = ROI).
        """
        roi = np.zeros((h, w), dtype=np.uint8)

        # YOLO regions
        norm_yolo = DoorArcDetector._normalise_yolo(yolo_doors)
        for (x1, y1, x2, y2) in norm_yolo:
            rx1 = max(0, x1 - yolo_expand_px)
            ry1 = max(0, y1 - yolo_expand_px)
            rx2 = min(w, x2 + yolo_expand_px)
            ry2 = min(h, y2 + yolo_expand_px)
            roi[ry1:ry2, rx1:rx2] = 255

        # Gap regions
        norm_gaps = DoorArcDetector._normalise_gaps(gaps)
        for (gx, gy, gw, gh) in norm_gaps:
            rx1 = max(0, gx - gap_expand_px)
            ry1 = max(0, gy - gap_expand_px)
            rx2 = min(w, gx + gw + gap_expand_px)
            ry2 = min(h, gy + gh + gap_expand_px)
            roi[ry1:ry2, rx1:rx2] = 255

        # Wall-centerline buffer
        if wall_outline_mask is not None:
            wm = wall_outline_mask
            if wm.shape[:2] != (h, w):
                wm = cv2.resize(wm, (w, h), interpolation=cv2.INTER_NEAREST)
            k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            wall_buffer = cv2.dilate(wm, k_dilate)
            wall_buffer = cv2.erode(wall_buffer, k_erode)
            roi = cv2.bitwise_or(roi, wall_buffer)

        # Fallback: if empty, use entire image
        if np.count_nonzero(roi) == 0:
            roi[:] = 255

        return roi

    # ══════════════════════════════════════════════════════════════════════
    # DEBUG VISUALISATION
    # ══════════════════════════════════════════════════════════════════════
    def _save_debug_images(
        self,
        img: np.ndarray,
        all_candidates: List[ArcCandidate],
        survivors: List[ArcCandidate],
        results: List[DetectedDoorArc],
        base_name: str,
    ) -> None:
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Arcs overlay on original image
        overlay = img.copy()
        for cand in all_candidates:
            if cand.center is None:
                continue
            color = (0, 255, 0) if not cand.rejected else (0, 0, 255)
            cx, cy = int(cand.center[0]), int(cand.center[1])
            r = int(cand.radius_px)
            cv2.circle(overlay, (cx, cy), r, color, 1)
            cv2.drawContours(overlay, [cand.contour], -1, color, 2)
            label = f"r={r}"
            if cand.rejected:
                label += f" X:{cand.rejection_reason}"
            cv2.putText(
                overlay, label, (cx - 40, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA
            )

        for arc in results:
            cx, cy = arc.center
            r = int(arc.radius_px)
            hx, hy = arc.hinge
            cv2.ellipse(
                overlay,
                (cx, cy),
                (r, r),
                0,
                arc.start_angle_deg,
                arc.end_angle_deg,
                (255, 0, 255),
                2,
            )
            cv2.drawMarker(
                overlay, (hx, hy), (0, 255, 255),
                cv2.MARKER_DIAMOND, 12, 2
            )
            cv2.putText(
                overlay,
                f"#{arc.id} {arc.direction} c={arc.confidence:.2f}",
                (cx + 5, cy + r + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA,
            )
        path = self.debug_output_dir / f"{base_name}_9_arcs_overlay.png"
        cv2.imwrite(str(path), overlay)
        logger.debug("Debug image saved: %s", path)

        if "stage1_curvature_map" in self._debug_images:
            path2 = self.debug_output_dir / f"{base_name}_9_curvature_map.png"
            cv2.imwrite(str(path2), self._debug_images["stage1_curvature_map"])

        if survivors:
            att_overlay = img.copy()
            for cand in survivors:
                if cand.endpoint_a:
                    ax, ay = cand.endpoint_a
                    r = self.hinge_wall_roi_px
                    cv2.rectangle(
                        att_overlay, (ax - r, ay - r), (ax + r, ay + r),
                        (255, 200, 0), 1,
                    )
                if cand.endpoint_b:
                    bx, by = cand.endpoint_b
                    r = self.hinge_wall_roi_px
                    cv2.rectangle(
                        att_overlay, (bx - r, by - r), (bx + r, by + r),
                        (0, 200, 255), 1,
                    )
            path3 = self.debug_output_dir / f"{base_name}_9_attachment_rois.png"
            cv2.imwrite(str(path3), att_overlay)

        if "stage0_clean_bw" in self._debug_images:
            path4 = self.debug_output_dir / f"{base_name}_9_stage0_bw.png"
            cv2.imwrite(str(path4), self._debug_images["stage0_clean_bw"])

        logger.debug("Timing breakdown: %s", {
            k: f"{v * 1000:.1f}ms" for k, v in self._timing.items()
        })

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC UTILITY: gap-vs-door overlap filter
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def filter_gaps_against_yolo_doors(
        gaps: List[Any],
        yolo_doors: List[Any],
        iou_thresh: float = 0.30,
    ) -> List[Any]:
        if not gaps or not yolo_doors:
            return gaps
        door_boxes: List[Tuple[float, float, float, float]] = []
        for d in yolo_doors:
            if hasattr(d, 'bbox'):
                b = d.bbox
                door_boxes.append((float(b[0]), float(b[1]),
                                   float(b[2]), float(b[3])))
            elif hasattr(d, 'x1'):
                door_boxes.append((float(d.x1), float(d.y1),
                                   float(d.x2), float(d.y2)))
            elif isinstance(d, (list, tuple)) and len(d) >= 4:
                door_boxes.append((float(d[0]), float(d[1]),
                                   float(d[2]), float(d[3])))
        if not door_boxes:
            return gaps
        kept = []
        for g in gaps:
            if hasattr(g, 'x1') and hasattr(g, 'y1'):
                gb = (float(g.x1), float(g.y1),
                      float(g.x2), float(g.y2))
            elif hasattr(g, 'x') and hasattr(g, 'w'):
                gb = (float(g.x), float(g.y),
                      float(g.x + g.w), float(g.y + g.h))
            elif isinstance(g, (list, tuple)) and len(g) == 4:
                gb = (float(g[0]), float(g[1]),
                      float(g[0] + g[2]), float(g[1] + g[3]))
            else:
                kept.append(g)
                continue
            if all(_iou_xyxy(gb, db) < iou_thresh for db in door_boxes):
                kept.append(g)
        return kept

    # ══════════════════════════════════════════════════════════════════════
    # CONVENIENCE: RENDERING HELPERS (for external renderers)
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def draw_arc_on_image(
        img: np.ndarray,
        arc: DetectedDoorArc,
        color: Tuple[int, int, int] = (255, 0, 255),
        thickness: int = 2,
        draw_hinge: bool = True,
        draw_label: bool = True,
    ) -> np.ndarray:
        cx, cy = arc.center
        r = int(arc.radius_px)
        cv2.ellipse(
            img,
            (cx, cy),
            (r, r),
            0,
            arc.start_angle_deg,
            arc.end_angle_deg,
            color,
            thickness,
        )
        if draw_hinge:
            hx, hy = arc.hinge
            cv2.drawMarker(img, (hx, hy), color, cv2.MARKER_DIAMOND, 10, 2)
        if draw_label:
            label = f"Door {arc.id} ({arc.direction})"
            cv2.putText(
                img, label, (cx + 5, cy - int(r) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
        return img


    @staticmethod
    def to_sweethome3d_xml(arc: DetectedDoorArc) -> str:
        return (
            f'<door id="{arc.id}" '
            f'hingeX="{arc.hinge[0]}" hingeY="{arc.hinge[1]}" '
            f'radius="{arc.radius_m:.3f}" '
            f'swingAngle="{arc.angle_span_deg:.1f}" '
            f'direction="{arc.direction}" '
            f'wallNormal="{arc.wall_normal_deg:.1f}" '
            f'clockwise="{str(arc.swing_clockwise).lower()}" '
            f'confidence="{arc.confidence:.3f}" />'
        )


