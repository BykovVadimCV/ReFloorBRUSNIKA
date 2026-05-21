"""SH3D export adapter — wraps the legacy SweetHome3D exporter, accepting unified types."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import PipelineConfig
from core.models import Opening, OpeningType, Room, WallSegment

logger = logging.getLogger(__name__)


def _wall_centerline(w: WallSegment) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return centerline endpoints in pixel coords for any WallSegment."""
    if w.outline and len(w.outline) >= 2:
        return (
            (float(w.outline[0][0]), float(w.outline[0][1])),
            (float(w.outline[1][0]), float(w.outline[1][1])),
        )
    p1, p2 = w.centerline
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def _seg_dist(px: float, py: float,
              a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def window_opening_to_overlay_wall(
    window: Opening, walls: List[WallSegment], pixels_to_cm: float,
):
    """
    Build a LegacyWall placed on top of the window's parent wall.

    Behaviour:
        • parent = nearest non-diagonal wall to the window's center
        • window endpoints projected onto the parent's centerline direction
        • new wall has parent's thickness, length spans the window's projected extent
        • returns None when no suitable parent wall exists

    The returned wall is appended to the SH3D export's
    ``extra_structural_walls`` list (with ``is_structural=False``).
    """
    from floorplan_export.legacy import Wall as _LegacyWall, Point as _LegacyPoint

    cx, cy = window.bbox.center

    best = None
    best_d = float('inf')
    for w in walls:
        if w.is_diagonal:
            continue
        p1, p2 = _wall_centerline(w)
        d = _seg_dist(cx, cy, p1, p2)
        if d < best_d:
            best_d = d
            best = (w, p1, p2)

    if best is None:
        return None
    parent, p1, p2 = best

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return None
    ux, uy = dx / L, dy / L

    # Project window bbox corners onto the parent direction
    corners = [
        (window.bbox.x1, window.bbox.y1),
        (window.bbox.x2, window.bbox.y1),
        (window.bbox.x2, window.bbox.y2),
        (window.bbox.x1, window.bbox.y2),
    ]
    projs = [(cx_ - p1[0]) * ux + (cy_ - p1[1]) * uy for cx_, cy_ in corners]
    t_min, t_max = min(projs), max(projs)

    # Window-wall endpoints, snapped to the parent's centerline
    wp1 = (p1[0] + t_min * ux, p1[1] + t_min * uy)
    wp2 = (p1[0] + t_max * ux, p1[1] + t_max * uy)

    return _LegacyWall(
        start=_LegacyPoint(wp1[0] * pixels_to_cm, wp1[1] * pixels_to_cm),
        end=_LegacyPoint(wp2[0] * pixels_to_cm, wp2[1] * pixels_to_cm),
        thickness=max(parent.thickness * pixels_to_cm, 2.0),
        is_structural=False,
    )


def wall_segment_to_rect_dict(seg: WallSegment) -> Dict[str, float]:
    """
    Convert a WallSegment to the dict format expected by SweetHome3DExporter.

    SweetHome3DExporter.export_to_sh3d() expects wall_rectangles as
    a list of dicts with 'x1', 'y1', 'x2', 'y2' keys.

    Args:
        seg: Unified WallSegment

    Returns:
        Dict with x1, y1, x2, y2 keys
    """
    return {
        'x1': seg.bbox.x1,
        'y1': seg.bbox.y1,
        'x2': seg.bbox.x2,
        'y2': seg.bbox.y2,
    }


def opening_to_rect_dict(op: Opening) -> Dict[str, float]:
    """
    Convert a unified Opening to a dict for the exporter.

    Args:
        op: Unified Opening

    Returns:
        Dict with x1, y1, x2, y2 keys
    """
    return {
        'x1': op.bbox.x1,
        'y1': op.bbox.y1,
        'x2': op.bbox.x2,
        'y2': op.bbox.y2,
    }


def opening_to_fused_door(op: Opening):
    """
    Convert a unified door Opening to a FusedDoor-compatible object.

    The SweetHome3DExporter.OpeningProcessor._process_fused_door() expects
    objects with .center, .width_px, .wall_normal_deg, .hinge, .swing_clockwise.

    This creates a duck-typed object matching that interface.

    Args:
        op: Unified Opening of type DOOR

    Returns:
        Duck-typed object compatible with FusedDoor
    """
    class _FusedDoorCompat:
        pass

    fd = _FusedDoorCompat()
    fd.center = (int(op.bbox.center_x), int(op.bbox.center_y))
    fd.width_px = int(max(op.bbox.width, op.bbox.height))
    fd.wall_normal_deg = 0.0  # Will be computed from parent wall
    fd.hinge = op.hinge_point
    fd.swing_clockwise = op.swing_clockwise
    fd.direction = op.swing_direction.value if op.swing_direction else "unknown"
    fd.confidence = op.confidence
    fd.source = op.source
    return fd


class SH3DExporter:
    """
    Clean SH3D export adapter.

    Accepts unified core.models types and delegates to the existing
    SweetHome3DExporter for the actual XML/ZIP generation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._inner = None
        self._init_inner()

    def _init_inner(self) -> None:
        """Initialize the underlying SweetHome3DExporter."""
        from floorplan_export.legacy import SweetHome3DExporter as LegacyExporter
        cfg = self.config
        self._inner = LegacyExporter(
            pixels_to_cm=cfg.pixels_to_cm,
            wall_height_cm=cfg.wall_height_cm,
            scale_factor=cfg.sh3d_scale_factor,
        )

    def update_scale(self, pixels_to_cm: float) -> None:
        """
        Update the pixel-to-cm conversion factor (e.g., after OCR calibration).

        Args:
            pixels_to_cm: New conversion factor
        """
        self.config = self.config.copy_with(pixels_to_cm=pixels_to_cm)
        if self._inner is not None:
            self._inner.update_scale(pixels_to_cm)

    def export(
        self,
        walls: List[WallSegment],
        openings: List[Opening],
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        ocr_labels: Optional[List] = None,
        debug_image_path: Optional[str] = None,
        wall_mask: Optional[np.ndarray] = None,
        enclosed_labels: Optional[np.ndarray] = None,
    ) -> List[Room]:
        """
        Export a floorplan to SH3D format.

        Args:
            walls: Detected wall segments
            openings: Detected openings (doors, windows, gaps)
            output_path: Output .sh3d file path
            original_image: Original BGR image for debug visualization
            ocr_labels: OCR text labels for debug overlay
            debug_image_path: Path to save debug image
            wall_mask: Wall binary mask for debug overlay

        Returns:
            List of detected Room objects for visualization
        """
        # Partition walls into three groups:
        #   regular   — axis-aligned solid walls  → rect-dict path
        #   diagonal  — rotated solid walls        → pre-built LegacyWall (is_structural=True)
        #   door_wall — door-opening segments      → pre-built LegacyWall (is_structural=False)
        #
        # Door walls are passed as non-structural extra walls so that:
        #   • ParentWallFinder finds them (it searches ALL walls)
        #   • ThicknessNormalizer ignores them (only uses is_structural=True walls)
        # This guarantees every U-Net door has a pre-built parent wall and the
        # exporter never needs to synthesise or guess wall geometry for doors.
        from floorplan_export.legacy import Wall as _LegacyWall, Point as _LegacyPoint

        regular_walls = [w for w in walls if not w.is_diagonal and not w.is_door_wall]
        diag_walls    = [w for w in walls if w.is_diagonal]
        door_walls    = [w for w in walls if w.is_door_wall]

        # Convert regular axis-aligned walls to rect dicts (existing path)
        wall_rects = [wall_segment_to_rect_dict(w) for w in regular_walls]

        ptcm = self.config.pixels_to_cm
        extra_structural: List = []

        # Diagonal walls (structural)
        for dw in diag_walls:
            if dw.outline and len(dw.outline) >= 2:
                (ox1, oy1), (ox2, oy2) = dw.outline[0], dw.outline[1]
                extra_structural.append(_LegacyWall(
                    start=_LegacyPoint(ox1 * ptcm, oy1 * ptcm),
                    end=_LegacyPoint(ox2 * ptcm, oy2 * ptcm),
                    thickness=max(dw.thickness * ptcm, 2.0),
                    is_structural=True,
                ))
        if diag_walls:
            logger.debug("SH3DExporter: %d diagonal walls → %d LegacyWall objects",
                         len(diag_walls), len(extra_structural))

        # Door walls (non-structural — parent for each U-Net door opening)
        for dw in door_walls:
            p1, p2 = _wall_centerline(dw)
            extra_structural.append(_LegacyWall(
                start=_LegacyPoint(p1[0] * ptcm, p1[1] * ptcm),
                end=_LegacyPoint(p2[0] * ptcm, p2[1] * ptcm),
                thickness=max(dw.thickness * ptcm, 2.0),
                is_structural=False,
            ))
        if door_walls:
            logger.info("SH3DExporter: %d door-wall segments → extra walls (non-structural)",
                        len(door_walls))

        # Separate openings by type
        doors = [o for o in openings if o.opening_type == OpeningType.DOOR]
        windows = [o for o in openings if o.opening_type == OpeningType.WINDOW]
        gaps = [o for o in openings if o.opening_type == OpeningType.GAP]

        # Doors with hinge data go through fused_doors path
        fused_doors = [opening_to_fused_door(d) for d in doors if d.hinge_point is not None]
        plain_doors = [opening_to_rect_dict(d) for d in doors if d.hinge_point is None]

        # ── Windows-as-overlay-walls ─────────────────────────────────────
        # When the rect-wall pipeline is active, render each window as a
        # separate wall placed on top of its parent wall (same thickness,
        # length = window-mask projection). The legacy window-rect path is
        # skipped so windows don't get cut into the parent walls.
        window_rects: List[Dict] = []
        if getattr(self.config, "use_rect_walls", False) and windows:
            for win in windows:
                lw = window_opening_to_overlay_wall(win, walls, ptcm)
                if lw is not None:
                    extra_structural.append(lw)
            logger.info(
                "SH3DExporter: %d windows → overlay walls on parent walls",
                len(windows),
            )
        else:
            window_rects = [opening_to_rect_dict(w) for w in windows]

        gap_rects = [opening_to_rect_dict(g) for g in gaps]

        try:
            legacy_rooms = self._inner.export_to_sh3d(
                wall_rectangles=wall_rects,
                output_path=output_path,
                door_rectangles=plain_doors if plain_doors else None,
                window_rectangles=window_rects if window_rects else None,
                gap_rectangles=gap_rects if gap_rects else None,
                original_image=original_image,
                ocr_labels=ocr_labels,
                debug_image_path=debug_image_path,
                wall_mask=wall_mask,
                fused_doors=fused_doors if fused_doors else None,
                enclosed_labels=enclosed_labels,
                extra_structural_walls=extra_structural if extra_structural else None,
            )
        except Exception as e:
            logger.error("SH3D export failed: %s", e, exc_info=True)
            return []

        # Convert legacy Room objects to core.models.Room
        rooms = _convert_legacy_rooms(legacy_rooms)
        return rooms

    def export_from_dicts(
        self,
        wall_rectangles: List[Dict],
        output_path: str,
        door_rectangles: Optional[List[Dict]] = None,
        window_rectangles: Optional[List[Dict]] = None,
        gap_rectangles: Optional[List[Dict]] = None,
        original_image: Optional[np.ndarray] = None,
        ocr_labels: Optional[List] = None,
        debug_image_path: Optional[str] = None,
        wall_mask: Optional[np.ndarray] = None,
        fused_doors: Optional[List] = None,
    ) -> List:
        """
        Export using raw dict format (backward compatibility).

        Delegates directly to the legacy exporter without conversion.
        Used by the old pipeline code during transition.

        Returns:
            Legacy Room list
        """
        return self._inner.export_to_sh3d(
            wall_rectangles=wall_rectangles,
            output_path=output_path,
            door_rectangles=door_rectangles,
            window_rectangles=window_rectangles,
            gap_rectangles=gap_rectangles,
            original_image=original_image,
            ocr_labels=ocr_labels,
            debug_image_path=debug_image_path,
            wall_mask=wall_mask,
            fused_doors=fused_doors,
        )


def _convert_legacy_rooms(legacy_rooms: List) -> List[Room]:
    """Convert legacy Room objects to core.models.Room."""
    from core.models import Point, Room

    result = []
    for lr in legacy_rooms:
        try:
            points = [Point(p.x, p.y) for p in lr.points]
            room = Room(
                points=points,
                room_id=getattr(lr, 'room_id', ''),
                name=getattr(lr, 'name', 'Room'),
            )
            result.append(room)
        except Exception:
            pass
    return result
