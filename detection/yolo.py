"""
YOLO-based door/window detector adapter for the ReFloorBRUSNIKA pipeline.

Wraps YOLODoorWindowDetector from furnituredetector.py,
converting its output to unified core.models types.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from core.config import PipelineConfig
from core.models import BBox, Opening, OpeningType


def yolo_detection_to_opening(det, opening_type: OpeningType) -> Opening:
    """
    Convert a YOLODetection to a unified Opening.

    Args:
        det: YOLODetection from furnituredetector.py
        opening_type: OpeningType.DOOR or OpeningType.WINDOW

    Returns:
        Unified Opening object
    """
    x1, y1, x2, y2 = det.bbox
    return Opening(
        bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
        opening_type=opening_type,
        confidence=det.confidence,
        source="yolo",
    )


class YOLODetector:
    """
    YOLO door/window detector adapter.

    Wraps YOLODoorWindowDetector, returns unified Opening lists.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._inner = None

    def initialize(self) -> None:
        """Load the YOLO model. Must be called before detect()."""
        from detection.furnituredetector import YOLODoorWindowDetector

        weights_path = self.config.yolo_weights_path
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}"
            )

        self._inner = YOLODoorWindowDetector(
            model_path=weights_path,
            conf_threshold=self.config.yolo_confidence,
            iou_threshold=self.config.yolo_iou_threshold,
            device=self.config.yolo_device,
            max_door_area_fraction=self.config.yolo_max_area_fraction,
        )

    @property
    def is_loaded(self) -> bool:
        """True if the YOLO model has been loaded."""
        return self._inner is not None

    def detect(self, image: np.ndarray) -> tuple[List[Opening], List[Opening]]:
        """
        Run YOLO detection on an image.

        Args:
            image: Input image in BGR format

        Returns:
            (doors, windows) as lists of unified Opening objects
        """
        if self._inner is None:
            return [], []

        results = self._inner.detect(image)

        doors = [
            yolo_detection_to_opening(d, OpeningType.DOOR)
            for d in results.doors
        ]
        windows = [
            yolo_detection_to_opening(w, OpeningType.WINDOW)
            for w in results.windows
        ]

        return doors, windows

    def get_combined_masks(self, image: np.ndarray):
        """
        Get combined door/window masks for room analysis.

        Returns raw YOLOResults for compatibility with existing code
        that uses the combined_door_mask / combined_window_mask.
        """
        if self._inner is None:
            h, w = image.shape[:2]
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty

        results = self._inner.detect(image)
        return results.combined_door_mask, results.combined_window_mask

    def get_raw_results(self, image: np.ndarray):
        """
        Get raw YOLOResults for backward compatibility.

        Used by code that directly accesses YOLOResults fields.
        """
        if self._inner is None:
            return None
        return self._inner.detect(image)
