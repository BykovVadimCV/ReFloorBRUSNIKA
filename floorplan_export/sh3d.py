"""
SH3D export adapter for the ReFloorBRUSNIKA pipeline.

Wraps SweetHome3DExporter from floorplanexporter.py, accepting
unified core.models types and converting them as needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import PipelineConfig
from core.models import Opening, OpeningType, Room, WallSegment

logger = logging.getLogger(__name__)


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
        from floorplanexporter import SweetHome3DExporter as LegacyExporter
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
        # Convert wall segments to rect dicts
        wall_rects = [wall_segment_to_rect_dict(w) for w in walls]

        # Separate openings by type
        doors = [o for o in openings if o.opening_type == OpeningType.DOOR]
        windows = [o for o in openings if o.opening_type == OpeningType.WINDOW]
        gaps = [o for o in openings if o.opening_type == OpeningType.GAP]

        # Doors with hinge data go through fused_doors path
        fused_doors = [opening_to_fused_door(d) for d in doors if d.hinge_point is not None]
        plain_doors = [opening_to_rect_dict(d) for d in doors if d.hinge_point is None]

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
    """
    Convert legacy Room objects from floorplanexporter.py to core.models.Room.

    Args:
        legacy_rooms: List of floorplanexporter.Room objects

    Returns:
        List of core.models.Room objects
    """
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
