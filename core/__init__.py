"""
Core module for ReFloorBRUSNIKA pipeline.

This module contains shared utilities, data models, and configuration
used across all detection and export modules.
"""

from core.models import (
    BBox,
    Opening,
    OpeningType,
    WallSegment,
    DetectionResult,
    FusedDoor,
    DetectedWindow,
    DetectedDoorArc,
    Room,
    Point,
    Wall,
    SwingDirection,
)
from core.geometry import (
    iou,
    intersection_area,
    point_in_bbox,
    point_in_polygon,
    polygon_area,
    are_parallel_and_close,
    snap_to_grid,
    pixels_to_cm,
    cm_to_pixels,
    distance,
)
from core.config import PipelineConfig

__all__ = [
    # Models
    "BBox",
    "Opening",
    "OpeningType",
    "WallSegment",
    "DetectionResult",
    "FusedDoor",
    "DetectedWindow",
    "DetectedDoorArc",
    "Room",
    "Point",
    "Wall",
    "SwingDirection",
    # Geometry
    "iou",
    "intersection_area",
    "point_in_bbox",
    "point_in_polygon",
    "polygon_area",
    "are_parallel_and_close",
    "snap_to_grid",
    "pixels_to_cm",
    "cm_to_pixels",
    "distance",
    # Config
    "PipelineConfig",
]
