"""
Unified opening detection pipeline for the ReFloorBRUSNIKA pipeline.

Merges and orchestrates:
- GeometricOpeningDetector (gap.py) — ray casting
- AlgorithmicWindowDetector (windowdetector.py) — parallel lines
- AlgorithmicDoorArcDetector + DoorFusionEngine (doordetector.py) — door arcs
- YOLODoorWindowDetector (furnituredetector.py) — YOLO inference

All outputs are converted to unified core.models.Opening objects.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

from core.config import PipelineConfig
from core.geometry import iou_xyxy
from core.models import BBox, FusedDoor, Opening, OpeningType, SwingDirection, WallSegment

logger = logging.getLogger(__name__)


# ============================================================
# CONVERSION HELPERS
# ============================================================

def gap_opening_to_unified(gap_op) -> Opening:
    """
    Convert a gap.Opening (from gap.py) to unified core.models.Opening.

    Args:
        gap_op: gap.Opening dataclass instance

    Returns:
        Unified Opening object
    """
    bbox = BBox(
        float(gap_op.x1),
        float(gap_op.y1),
        float(gap_op.x2),
        float(gap_op.y2),
    )
    opening_type = OpeningType.GAP
    if hasattr(gap_op, 'opening_type'):
        t = gap_op.opening_type
        if t == 'door':
            opening_type = OpeningType.DOOR
        elif t == 'window':
            opening_type = OpeningType.WINDOW

    return Opening(
        bbox=bbox,
        opening_type=opening_type,
        confidence=float(gap_op.confidence) if hasattr(gap_op, 'confidence') else 1.0,
        source="geometric",
    )


def window_to_unified(win) -> Opening:
    """
    Convert a DetectedWindow (from windowdetector.py) to unified Opening.

    Args:
        win: DetectedWindow dataclass instance

    Returns:
        Unified Opening object
    """
    if hasattr(win, 'bbox'):
        bbox = win.bbox  # Already a BBox
    else:
        bbox = BBox.from_xywh(float(win.x), float(win.y), float(win.w), float(win.h))
    return Opening(
        bbox=bbox,
        opening_type=OpeningType.WINDOW,
        confidence=1.0,
        source="algorithmic",
    )


def fused_door_to_unified(fd: FusedDoor) -> Opening:
    """
    Convert a FusedDoor (from doordetector.py) to unified Opening.

    Args:
        fd: FusedDoor object from DoorFusionEngine

    Returns:
        Unified Opening object
    """
    cx, cy = fd.center
    half = fd.width_px / 2
    bbox = BBox(cx - half, cy - half, cx + half, cy + half)

    direction = SwingDirection.UNKNOWN
    if fd.direction == "left":
        direction = SwingDirection.LEFT
    elif fd.direction == "right":
        direction = SwingDirection.RIGHT

    return Opening(
        bbox=bbox,
        opening_type=OpeningType.DOOR,
        confidence=fd.confidence,
        hinge_point=fd.hinge,
        swing_direction=direction,
        swing_clockwise=fd.swing_clockwise,
        source=fd.source,
    )


def yolo_detection_to_unified(det, opening_type: OpeningType) -> Opening:
    """
    Convert a YOLODetection to unified Opening.

    Args:
        det: YOLODetection from furnituredetector.py
        opening_type: OpeningType to assign

    Returns:
        Unified Opening object
    """
    x1, y1, x2, y2 = det.bbox
    return Opening(
        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
        opening_type=opening_type,
        confidence=float(det.confidence),
        source="yolo",
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_openings(
    openings: List[Opening],
    iou_threshold: float = 0.30,
) -> List[Opening]:
    """
    Remove duplicate openings using IoU-based non-maximum suppression.

    Priority order: doors > windows > gaps
    Within same type: higher confidence wins.

    Args:
        openings: List of Opening objects to deduplicate
        iou_threshold: IoU threshold for suppression

    Returns:
        Deduplicated list of Opening objects
    """
    if len(openings) <= 1:
        return openings

    # Priority: DOOR=3, WINDOW=2, GAP=1
    priority = {OpeningType.DOOR: 3, OpeningType.WINDOW: 2, OpeningType.GAP: 1}

    # Sort by priority desc, then confidence desc
    sorted_ops = sorted(
        openings,
        key=lambda o: (priority.get(o.opening_type, 0), o.confidence),
        reverse=True,
    )

    kept: List[Opening] = []
    for candidate in sorted_ops:
        cb = candidate.bbox.to_xyxy()
        duplicate = False
        for existing in kept:
            eb = existing.bbox.to_xyxy()
            if iou_xyxy(cb, eb) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


# ============================================================
# UNIFIED OPENING DETECTION PIPELINE
# ============================================================

class OpeningDetectionPipeline:
    """
    Single orchestrator for all opening detection.

    Runs all detectors once (no redundant calls), fuses results,
    deduplicates, and returns unified Opening objects.

    Detection order:
    1. YOLO detection (doors + windows)
    2. Geometric gap detection (ray casting)
    3. Algorithmic window detection (parallel lines)
    4. Door arc detection (morphological)
    5. DoorFusionEngine fusion
    6. Deduplication
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._gap_detector = None
        self._window_detector = None
        self._arc_detector = None
        self._fusion_engine = None
        self._yolo_detector = None

    def initialize(self, yolo_model=None) -> None:
        """
        Initialize all sub-detectors.

        Args:
            yolo_model: Optional pre-loaded YOLODoorWindowDetector instance.
                        If None, uses the one from the YOLO adapter.
        """
        from detection.gap import GeometricOpeningDetector
        from detection.windowdetector import AlgorithmicWindowDetector
        from detection.doordetector import AlgorithmicDoorArcDetector, DoorFusionEngine

        cfg = self.config
        pixel_scale = cfg.pixels_to_m  # meters per pixel

        self._gap_detector = GeometricOpeningDetector(
            min_gap_width_m=cfg.min_opening_width_m,
            max_gap_width_m=cfg.max_opening_width_m,
            max_extension_length=300,
            merge_distance=8,
            yolo_iou_threshold=0.0,
            min_opening_aspect_ratio=4.0,   # was 5.0 — accept slightly squarer gaps
            max_non_white_ratio=0.45,        # was 0.35 — tolerate furniture overlapping gap
        )
        self._window_detector = AlgorithmicWindowDetector()
        self._arc_detector = AlgorithmicDoorArcDetector(
            min_arc_radius_m=0.6,
            max_arc_radius_m=1.2,
            debug=cfg.debug_arcs,
        )
        self._fusion_engine = DoorFusionEngine()
        self._yolo_detector = yolo_model

    def detect(
        self,
        image: np.ndarray,
        walls: List[WallSegment],
        wall_mask: np.ndarray,
        wall_rectangles: Optional[List] = None,
        yolo_results=None,
        ocr_bboxes: Optional[List] = None,
        pixel_scale: Optional[float] = None,
        base_name: str = "unknown",
    ) -> List[Opening]:
        """
        Run complete opening detection pipeline.

        Args:
            image: Input image in BGR format
            walls: Detected wall segments (unified WallSegment objects)
            wall_mask: Binary mask of wall pixels
            wall_rectangles: Raw WallRectangle list (for gap detector compatibility)
            yolo_results: Optional pre-computed YOLOResults (avoids duplicate inference)
            ocr_bboxes: OCR text bounding boxes to exclude
            pixel_scale: Meters per pixel (for size filtering)
            base_name: Image name for debug output

        Returns:
            List of unified Opening objects, deduplicated
        """
        cfg = self.config
        ps = pixel_scale or cfg.pixels_to_m

        all_openings: List[Opening] = []

        # ── Step 1: YOLO detection ─────────────────────────────────────
        yolo_doors: List[Opening] = []
        yolo_windows: List[Opening] = []

        if yolo_results is not None:
            # Use pre-computed results (avoids redundant YOLO inference)
            if hasattr(yolo_results, 'doors'):
                yolo_doors = [
                    yolo_detection_to_unified(d, OpeningType.DOOR)
                    for d in yolo_results.doors
                ]
            if hasattr(yolo_results, 'windows'):
                yolo_windows = [
                    yolo_detection_to_unified(w, OpeningType.WINDOW)
                    for w in yolo_results.windows
                ]
        elif self._yolo_detector is not None:
            raw = self._yolo_detector.detect(image)
            if hasattr(raw, 'doors') and hasattr(raw, 'windows'):
                yolo_doors = [
                    yolo_detection_to_unified(d, OpeningType.DOOR)
                    for d in raw.doors
                ]
                yolo_windows = [
                    yolo_detection_to_unified(w, OpeningType.WINDOW)
                    for w in raw.windows
                ]

        all_openings.extend(yolo_doors)
        all_openings.extend(yolo_windows)

        # ── Step 2: Geometric gap detection ───────────────────────────
        geo_gaps: List[Opening] = []
        rects = wall_rectangles or [_segment_to_wall_rect(s) for s in walls]

        if self._gap_detector is not None and rects:
            try:
                raw_gaps = self._gap_detector.detect_openings(
                    rectangles=rects,
                    yolo_results=yolo_results,
                    pixel_scale=ps,
                    image=image,
                )
                geo_gaps = [gap_opening_to_unified(g) for g in raw_gaps]
            except Exception:
                logger.warning("Geometric gap detection failed", exc_info=True)

        all_openings.extend(geo_gaps)

        # ── Step 3: Algorithmic window detection ───────────────────────
        algo_windows: List[Opening] = []

        if self._window_detector is not None:
            try:
                # Exclude YOLO door areas from window search
                door_exclude = [d.bbox.to_xyxy() for d in yolo_doors]
                raw_windows = self._window_detector.detect(
                    img=image,
                    wall_outline_mask=wall_mask,
                    exclude_bboxes=door_exclude,
                    ocr_bboxes=ocr_bboxes,
                    known_windows=[w.bbox.to_xyxy() for w in yolo_windows],
                )
                algo_windows = [window_to_unified(w) for w in raw_windows]
            except Exception:
                logger.warning("Algorithmic window detection failed", exc_info=True)

        all_openings.extend(algo_windows)

        # ── Step 4: Door arc detection + fusion ────────────────────────
        fused_doors: List[Opening] = []

        if cfg.enable_door_arcs and self._arc_detector is not None:
            try:
                # Convert raw gap detections back to gap.Opening format for arc detector
                from detection.gap import Opening as GapOpening
                raw_gaps_for_arc = [_unified_to_raw_gap(g, i) for i, g in enumerate(geo_gaps)]

                # Raw YOLO doors for arc detector
                raw_yolo_doors = (
                    yolo_results.doors
                    if yolo_results is not None and hasattr(yolo_results, 'doors')
                    else []
                )

                arcs = self._arc_detector.detect(
                    img=image,
                    wall_outline_mask=wall_mask,
                    candidate_gaps=raw_gaps_for_arc,
                    yolo_doors=raw_yolo_doors,
                    pixel_scale=ps,
                    ocr_boxes=ocr_bboxes,
                    base_name=base_name,
                )

                if self._fusion_engine is not None and arcs:
                    from detection.doordetector import FusedDoor as LegacyFusedDoor
                    fused = self._fusion_engine.fuse(
                        yolo_doors=raw_yolo_doors,
                        gaps=raw_gaps_for_arc,
                        arcs=arcs,
                        wall_segments=walls,
                    )
                    fused_doors = [_legacy_fused_door_to_unified(fd) for fd in fused]

            except Exception:
                logger.warning("Door arc detection failed", exc_info=True)

        if fused_doors:
            # Replace geo_gaps and yolo_doors with fused doors
            # (fusion already incorporated them)
            non_door_openings = [
                o for o in all_openings
                if o.opening_type != OpeningType.DOOR
            ]
            all_openings = non_door_openings + fused_doors
        # If fusion failed, keep existing doors from steps 1-2

        # ── Step 5: Final deduplication ───────────────────────────────
        return deduplicate_openings(all_openings, iou_threshold=cfg.iou_threshold)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _segment_to_wall_rect(seg: WallSegment):
    """
    Create a minimal duck-type WallRectangle from a WallSegment.

    The gap detector expects objects with .id, .x1, .y1, .x2, .y2.
    """
    class _FakeRect:
        pass

    r = _FakeRect()
    try:
        r.id = int(seg.id)
    except (ValueError, TypeError):
        r.id = abs(hash(seg.id)) % 100000
    r.x1 = int(seg.bbox.x1)
    r.y1 = int(seg.bbox.y1)
    r.x2 = int(seg.bbox.x2)
    r.y2 = int(seg.bbox.y2)
    r.width = int(seg.bbox.width)
    r.height = int(seg.bbox.height)
    r.area = int(seg.bbox.area)
    r.outline = seg.outline or []
    return r


def _unified_to_raw_gap(op: Opening, idx: int):
    """
    Convert unified Opening back to gap.Opening-compatible format.

    Used when passing gap results to the arc detector.
    """
    from detection.gap import Opening as GapOpening
    return GapOpening(
        id=idx,
        x1=int(op.bbox.x1),
        y1=int(op.bbox.y1),
        x2=int(op.bbox.x2),
        y2=int(op.bbox.y2),
        width=int(op.bbox.width),
        height=int(op.bbox.height),
        area=int(op.bbox.area),
        opening_type=op.opening_type.value,
        confidence=op.confidence,
        source_rects=[],
        yolo_matches=[],
    )


def _legacy_fused_door_to_unified(fd) -> Opening:
    """
    Convert a legacy FusedDoor (from doordetector.py) to unified Opening.

    Handles both the new core.models.FusedDoor and the legacy
    doordetector.FusedDoor (which has different field names).
    """
    # Legacy FusedDoor fields: hinge, center, width_px, direction, swing_clockwise, confidence, source
    cx, cy = fd.center
    half = fd.width_px / 2
    bbox = BBox(cx - half, cy - half, cx + half, cy + half)

    direction = SwingDirection.UNKNOWN
    d = getattr(fd, 'direction', 'unknown')
    if d == 'left':
        direction = SwingDirection.LEFT
    elif d == 'right':
        direction = SwingDirection.RIGHT

    return Opening(
        bbox=bbox,
        opening_type=OpeningType.DOOR,
        confidence=float(getattr(fd, 'confidence', 1.0)),
        hinge_point=fd.hinge,
        swing_direction=direction,
        swing_clockwise=bool(getattr(fd, 'swing_clockwise', True)),
        source=getattr(fd, 'source', 'arc'),
    )
