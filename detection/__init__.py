"""
Detection module for ReFloorBRUSNIKA pipeline.

Contains wall detection, opening detection (doors/windows/gaps),
and YOLO inference adapters.
"""

from detection.walls import ColorWallDetector
from detection.yolo import YOLODetector
from detection.openings import OpeningDetectionPipeline

__all__ = [
    "ColorWallDetector",
    "YOLODetector",
    "OpeningDetectionPipeline",
]
