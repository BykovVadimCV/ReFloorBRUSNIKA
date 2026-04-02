"""
Detector protocol / base classes for the ReFloorBRUSNIKA pipeline.

Defines the abstract interface that all detectors must implement,
enabling polymorphic use and future extensibility (e.g., segmentation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.config import PipelineConfig
from core.models import DetectionResult, Opening, WallSegment


class Detector(ABC):
    """
    Base protocol for all detectors.

    Every detector in the pipeline must implement this interface,
    ensuring consistent input/output contracts.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, config: PipelineConfig) -> DetectionResult:
        """
        Run detection on an image and return unified results.

        Args:
            image: Input image in BGR format (OpenCV convention)
            config: Pipeline configuration

        Returns:
            DetectionResult containing walls, openings, and masks
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this detector.

        Used for logging, source tracking, and debugging.
        """
        ...


class WallDetector(Detector):
    """
    Specialization for wall-only detection.

    Wall detectors focus on finding structural wall segments
    and generating wall masks.
    """
    pass


class OpeningDetector(ABC):
    """
    Specialization for door/window/gap detection.

    Opening detectors require wall context since openings
    are defined relative to walls.
    """

    @abstractmethod
    def detect_openings(
        self,
        image: np.ndarray,
        walls: List[WallSegment],
        wall_mask: np.ndarray,
        config: PipelineConfig,
    ) -> List[Opening]:
        """
        Detect openings using wall context.

        Args:
            image: Input image in BGR format
            walls: Detected wall segments
            wall_mask: Binary mask of wall pixels
            config: Pipeline configuration

        Returns:
            List of detected Opening objects
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this detector."""
        ...
