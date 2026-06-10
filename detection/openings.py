"""Unified opening detection: orchestrates gap, window, door-arc, and YOLO detectors."""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import numpy as np

from core.config import PipelineConfig
from core.geometry import iou_xyxy
from core.models import BBox, FusedDoor, Opening, OpeningType, SwingDirection, WallSegment

logger = logging.getLogger(__name__)


# ============================================================
# CONVERSION HELPERS
# ============================================================

def _ocr_bboxes_4(ocr_bboxes: Optional[List]) -> List[Tuple[int, int, int, int]]:
    """
    Normalise OCR bbox list to plain (x1, y1, x2, y2) 4-tuples.

    Accepts both the old format (x1, y1, x2, y2) and the new format
    (text, x1, y1, x2, y2).  The text token is silently dropped so legacy
    detectors that unpack exactly four values do not crash.
    """
    if not ocr_bboxes:
        return []
    out: List[Tuple[int, int, int, int]] = []
    for item in ocr_bboxes:
        if len(item) == 4:
            out.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        elif len(item) >= 5:
            out.append((int(item[1]), int(item[2]), int(item[3]), int(item[4])))
    return out


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


def _fused_door_bbox(fd) -> BBox:
    """
    Wall-aligned bbox for a fused door: the long side is the door width along
    the wall, the short side a nominal quarter-width (the true wall thickness
    is unknown at this point; the exporter re-derives it from the parent wall).
    """
    cx, cy = fd.center
    half_w = fd.width_px / 2.0
    half_t = max(2.0, fd.width_px / 8.0)
    normal = getattr(fd, "wall_normal_deg", None) or 0.0
    horizontal_wall = abs(math.sin(math.radians(normal))) > 0.707
    if horizontal_wall:
        return BBox(cx - half_w, cy - half_t, cx + half_w, cy + half_t)
    return BBox(cx - half_t, cy - half_w, cx + half_t, cy + half_w)


def fused_door_to_unified(fd: FusedDoor) -> Opening:
    """
    Convert a FusedDoor (from doordetector.py) to unified Opening.

    Args:
        fd: FusedDoor object from DoorFusion

    Returns:
        Unified Opening object
    """
    direction = SwingDirection.UNKNOWN
    if fd.direction == "left":
        direction = SwingDirection.LEFT
    elif fd.direction == "right":
        direction = SwingDirection.RIGHT

    return Opening(
        bbox=_fused_door_bbox(fd),
        opening_type=OpeningType.DOOR,
        confidence=fd.confidence,
        hinge_point=fd.hinge,
        swing_direction=direction,
        swing_clockwise=fd.swing_clockwise,
        wall_normal_deg=fd.wall_normal_deg,
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

    # Priority order (higher = survives deduplication):
    #   geometric/arc GAP  > YOLO/U-Net DOOR > WINDOW > unconfirmed GAP
    # Gap-sourced detections are the most geometrically precise; they beat
    # YOLO and U-Net doors that overlap them.
    def _priority(o: Opening) -> float:
        base = {OpeningType.DOOR: 2.0, OpeningType.WINDOW: 1.0, OpeningType.GAP: 3.0}
        # U-Net doors rank just below traditional YOLO/gap doors so that
        # geometric gaps and YOLO resolutions beat them when co-located.
        src = getattr(o, 'source', '') or ''
        if src.startswith('unet'):
            return 1.8 + o.confidence * 0.1
        return base.get(o.opening_type, 0.0) + o.confidence * 0.1

    # Sort by priority desc, then confidence desc
    sorted_ops = sorted(
        openings,
        key=_priority,
        reverse=True,
    )

    kept: List[Opening] = []
    for candidate in sorted_ops:
        cb = candidate.bbox.to_xyxy()
        dup_of: Optional[Opening] = None
        for existing in kept:
            eb = existing.bbox.to_xyxy()
            if iou_xyxy(cb, eb) > iou_threshold:
                dup_of = existing
                break
        if dup_of is None:
            kept.append(candidate)
        elif (dup_of.opening_type == OpeningType.GAP
              and candidate.opening_type == OpeningType.DOOR):
            # A door suppressed by a co-located gap: keep the gap's precise
            # wall-aligned geometry, but promote it to a DOOR carrying the
            # door's swing metadata — otherwise the doorway exports as a
            # bare passage and the detected door is lost.
            dup_of.opening_type = OpeningType.DOOR
            if dup_of.hinge_point is None:
                dup_of.hinge_point = candidate.hinge_point
            if dup_of.swing_direction == SwingDirection.UNKNOWN:
                dup_of.swing_direction = candidate.swing_direction
                dup_of.swing_clockwise = candidate.swing_clockwise
            if dup_of.wall_normal_deg is None:
                dup_of.wall_normal_deg = candidate.wall_normal_deg
            dup_of.confidence = max(dup_of.confidence, candidate.confidence)

    return kept


# ============================================================
# UNIFIED OPENING DETECTION PIPELINE
# ============================================================

class OpeningPipeline:
    """
    Single orchestrator for all opening detection.

    Runs all detectors once (no redundant calls), fuses results,
    deduplicates, and returns unified Opening objects.

    Detection order:
    1. YOLO detection (doors + windows)
    2. Geometric gap detection (ray casting)
    3. Algorithmic window detection (parallel lines)
    4. Door arc detection (morphological)
    5. DoorFusion fusion
    6. Deduplication
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._gap_detector = None
        self._window_detector = None
        self._arc_detector = None
        self._fusion_engine = None
        self._yolo_detector = None
        self._unet_door_detector = None
        # Populated during detect() for external debug consumers
        self._debug_yolo_doors_raw: List[Opening] = []
        self._debug_unet_door_mask: Optional[np.ndarray] = None
        self._debug_door_wall_segments: List = []   # WallSegment objects from U-Net door rects

    def initialize(self, yolo_model=None) -> None:
        """
        Initialize all sub-detectors.

        Args:
            yolo_model: Optional pre-loaded YOLODoorWindowDetector instance.
                        If None, uses the one from the YOLO adapter.
        """
        from detection.gap import GapDetector
        from detection.windowdetector import WindowDetector
        from detection.door_arc import DoorArcDetector
        from detection.door_fusion import DoorFusion

        cfg = self.config
        pixel_scale = cfg.pixels_to_m  # meters per pixel

        self._gap_detector = GapDetector(
            min_gap_width_m=cfg.min_opening_width_m,
            max_gap_width_m=cfg.max_opening_width_m,
            max_extension_length=300,
            merge_distance=8,
            yolo_iou_threshold=0.0,
            min_opening_aspect_ratio=4.0,   # was 5.0 — accept slightly squarer gaps
            max_non_white_ratio=0.45,        # was 0.35 — tolerate furniture overlapping gap
        )
        self._window_detector = WindowDetector()
        self._arc_detector = DoorArcDetector(
            min_arc_radius_m=0.6,
            max_arc_radius_m=1.2,
            debug=cfg.debug_arcs,
        )
        self._fusion_engine = DoorFusion()
        self._yolo_detector = yolo_model

        # U-Net 4-class door detector (epoch_040.pth) — optional
        if cfg.enable_unet:
            from detection.unet_doors import UNetDoorDetector
            self._unet_door_detector = UNetDoorDetector(
                ckpt_path=cfg.unet_checkpoint_path,
                img_size=cfg.unet_img_size,
                confidence=cfg.unet_confidence,
                min_door_area=cfg.unet_min_door_area,
                match_margin=getattr(cfg, "unet_door_match_margin", 40),
                device=cfg.unet_device,
                # Rect decomposition
                use_rect_decompose=getattr(cfg, "enable_door_rect_decompose", True),
                door_rect_coverage_stop=getattr(cfg, "door_rect_coverage_stop", 0.70),
                door_rect_max_spill=getattr(cfg, "door_rect_max_spill", 0.05),
                door_rect_max_overlap=getattr(cfg, "door_rect_max_overlap", 0.50),
                door_rect_bleed_weight=getattr(cfg, "door_rect_bleed_weight", 15.0),
                door_rect_initial_bleed_weight=getattr(cfg, "door_rect_initial_bleed_weight", 15.0),
                door_rect_bleed_decay=getattr(cfg, "door_rect_bleed_decay", 0.0),
                door_rect_penalty=getattr(cfg, "door_rect_penalty", 0.02),
                door_rect_max_grid_dim=getattr(cfg, "door_rect_max_grid_dim", 200),
                door_rect_axis_only=getattr(cfg, "door_rect_axis_only", True),
                # Wall-split fallback
                wall_band_ratio=getattr(cfg, "unet_door_wall_band_ratio", 0.50),
                min_wall_band_px=getattr(cfg, "unet_door_min_wall_band_px", 5),
                # Post-decompose validity
                min_piece_area=getattr(cfg, "unet_door_min_piece_area", 200),
                min_piece_thickness_px=getattr(cfg, "unet_door_min_piece_thickness_px", 12),
            )
            ok = self._unet_door_detector.initialize()
            if not ok:
                logger.warning(
                    "OpeningPipeline: UNetDoorDetector failed to load — "
                    "U-Net door detection disabled"
                )
                self._unet_door_detector = None

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
        self._debug_yolo_doors_raw = []   # reset before each run

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

        all_openings.extend(yolo_windows)
        # Snapshot raw YOLO doors before any resolution / deduplication
        self._debug_yolo_doors_raw = list(yolo_doors)

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

        # Keep original gaps for arc detector (Step 4) and for U-Net snap targets.
        all_geo_gaps = list(geo_gaps)

        # ── Step 2.5: U-Net 4-class door detection ────────────────────
        # Uses epoch_040.pth (bg/wall/window/door) to extract a door-class
        # mask, then snaps each detected blob to the nearest geometric gap.
        self._debug_unet_door_mask = None
        self._debug_door_wall_segments = []
        if self._unet_door_detector is not None:
            try:
                unet_doors, unet_door_mask = self._unet_door_detector.detect(
                    image, all_geo_gaps,
                    wall_mask=wall_mask,
                    ocr_bboxes=ocr_bboxes,
                    walls=walls,
                )
                self._debug_unet_door_mask = unet_door_mask
                self._debug_door_wall_segments = list(
                    getattr(self._unet_door_detector, '_debug_door_wall_segments', [])
                )
                if unet_doors:
                    all_openings.extend(unet_doors)
                    logger.info(
                        "OpeningPipeline: U-Net added %d door(s), "
                        "%d door-wall segments",
                        len(unet_doors), len(self._debug_door_wall_segments),
                    )
            except Exception:
                logger.warning("U-Net door detection failed", exc_info=True)

        # ── Step 2b: Resolve YOLO doors to gap bboxes ─────────────────
        # YOLO door bboxes include the door leaf (swinging part), which
        # offsets the centre from the actual wall gap.  When a geometric
        # gap overlaps a YOLO door, adopt the gap's bbox for precise
        # wall-gap positioning while keeping the DOOR type and confidence.
        if yolo_doors and geo_gaps:
            yolo_doors, geo_gaps = _resolve_doors_to_gaps(
                yolo_doors, geo_gaps, match_margin=25)

        all_openings.extend(yolo_doors)
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
                    ocr_bboxes=_ocr_bboxes_4(ocr_bboxes),
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
                # Use all_geo_gaps (pre-resolution) so arcs near claimed gaps are found
                from detection.gap import Opening as GapOpening
                raw_gaps_for_arc = [_unified_to_raw_gap(g, i) for i, g in enumerate(all_geo_gaps)]

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
                    ocr_boxes=_ocr_bboxes_4(ocr_bboxes),
                    base_name=base_name,
                )

                if self._fusion_engine is not None and arcs:
                    from detection.door_fusion import FusedDoor as LegacyFusedDoor
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

        # ── Step 4.5: Cap door/gap widths by detected door benchmark ──
        # Gap detection is geometric and can over-extend into adjacent
        # corridors, producing implausibly wide "openings". Use the
        # median detected door width on this image as a benchmark and
        # drop door/gap openings whose long side exceeds ~1.3× it.
        # Windows are exempted — they can legitimately be wider.
        all_openings = _cap_openings_by_door_benchmark(
            all_openings,
            cfg=cfg,
            pixel_scale=ps,
        )

        # ── Step 5: Final deduplication ───────────────────────────────
        return deduplicate_openings(all_openings, iou_threshold=cfg.iou_threshold)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _cap_openings_by_door_benchmark(
    openings: List[Opening],
    cfg: PipelineConfig,
    pixel_scale: float,
) -> List[Opening]:
    """Drop door/gap openings wider than ~1.3× the median detected door width.

    The benchmark comes from doors that survived ``_resolve_doors_to_gaps`` —
    those have wall-gap bboxes (not the inflated YOLO leaf-bbox) and so their
    long side is the true door width. If no door benchmark is available the
    function returns *openings* unchanged.
    """
    door_long_sides = [
        max(o.bbox.width, o.bbox.height)
        for o in openings
        if o.opening_type == OpeningType.DOOR
    ]
    if not door_long_sides:
        return openings

    benchmark_px = float(np.median(door_long_sides))
    multiplier = getattr(cfg, "opening_max_door_multiplier", 1.3)
    max_allowed_px = benchmark_px * multiplier

    # Hard floor so the cap never falls below a real door's physical width:
    # if YOLO+gap fusion misses the wider doorways in a plan, we still allow
    # openings up to min_plausible_opening_width_cm.
    floor_cm = getattr(cfg, "min_plausible_opening_width_cm", 40.0)
    if pixel_scale > 0:
        floor_px = (floor_cm / 100.0) / pixel_scale
        max_allowed_px = max(max_allowed_px, floor_px)

    kept: List[Opening] = []
    dropped = 0
    for op in openings:
        if op.opening_type == OpeningType.WINDOW:
            kept.append(op)
            continue
        long_side = max(op.bbox.width, op.bbox.height)
        if long_side > max_allowed_px:
            dropped += 1
            continue
        kept.append(op)

    if dropped:
        logger.info(
            "OpeningPipeline: capped %d opening(s) by door benchmark "
            "(median door=%.1fpx, max=%.1fpx)",
            dropped, benchmark_px, max_allowed_px,
        )
    return kept


def _resolve_doors_to_gaps(
    yolo_doors: List[Opening],
    geo_gaps: List[Opening],
    match_margin: int = 25,
) -> Tuple[List[Opening], List[Opening]]:
    """Replace YOLO door bboxes with matching gap bboxes for precise placement.

    When a geometric gap's centre falls inside a YOLO door bbox (expanded by
    *match_margin*), the YOLO door adopts the gap's bbox.  Each gap can be
    claimed by at most one YOLO door (closest centre wins).

    Returns (updated_yolo_doors, unclaimed_gaps).
    """
    import math

    claimed_gap_indices: set = set()
    updated_doors: List[Opening] = []

    for door in yolo_doors:
        db = door.bbox
        # Expand the YOLO bbox by the margin for matching
        ex1 = db.x1 - match_margin
        ey1 = db.y1 - match_margin
        ex2 = db.x2 + match_margin
        ey2 = db.y2 + match_margin
        dcx = db.center_x
        dcy = db.center_y

        candidates = []
        for gi, gap in enumerate(geo_gaps):
            if gi in claimed_gap_indices:
                continue
            gcx = gap.bbox.center_x
            gcy = gap.bbox.center_y
            if ex1 <= gcx <= ex2 and ey1 <= gcy <= ey2:
                dist = math.hypot(gcx - dcx, gcy - dcy)
                candidates.append((dist, gi, gap))

        if candidates:
            candidates.sort(key=lambda t: t[0])
            _, best_gi, best_gap = candidates[0]
            claimed_gap_indices.add(best_gi)
            # Adopt the gap's bbox but keep the door's type and confidence
            resolved = Opening(
                bbox=best_gap.bbox,
                opening_type=OpeningType.DOOR,
                confidence=door.confidence,
                source=door.source,
                hinge_point=door.hinge_point,
                swing_direction=door.swing_direction,
                swing_clockwise=door.swing_clockwise,
            )
            updated_doors.append(resolved)
            logger.debug(
                "Resolved YOLO door to gap: (%.0f,%.0f)→(%.0f,%.0f)",
                db.center_x, db.center_y,
                best_gap.bbox.center_x, best_gap.bbox.center_y,
            )
        else:
            # No matching gap — keep YOLO door as-is
            updated_doors.append(door)

    unclaimed_gaps = [
        g for i, g in enumerate(geo_gaps) if i not in claimed_gap_indices
    ]
    return updated_doors, unclaimed_gaps


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
    direction = SwingDirection.UNKNOWN
    d = getattr(fd, 'direction', 'unknown')
    if d == 'left':
        direction = SwingDirection.LEFT
    elif d == 'right':
        direction = SwingDirection.RIGHT

    return Opening(
        bbox=_fused_door_bbox(fd),
        opening_type=OpeningType.DOOR,
        confidence=float(getattr(fd, 'confidence', 1.0)),
        hinge_point=fd.hinge,
        swing_direction=direction,
        swing_clockwise=bool(getattr(fd, 'swing_clockwise', True)),
        wall_normal_deg=getattr(fd, 'wall_normal_deg', None),
        source=getattr(fd, 'source', 'arc'),
    )
