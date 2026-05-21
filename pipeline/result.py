"""PipelineResult: aggregated output of a pipeline run on one image."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.models import Opening, OpeningType, Room, WallSegment


@dataclass
class PipelineResult:
    """
    Full result from a pipeline run on one image.

    Contains all detected geometry plus paths to output files.
    """
    # Detected geometry
    walls: List[WallSegment] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    door_arcs: List[Any] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)

    # Masks
    wall_mask: Optional[np.ndarray] = None
    outline_mask: Optional[np.ndarray] = None
    raw_wall_mask: Optional[np.ndarray] = None   # U-Net output before cap/refinement

    # Raw rect_decompose polygon corners (4×2 float32 each), captured before
    # endpoint snapping.  Used for the rect overlay debug image.
    wall_rect_polygons: List[Any] = field(default_factory=list)

    # Scale calibration
    pixel_scale: Optional[float] = None  # meters per pixel
    pixels_to_cm: float = 2.54

    # OCR labels
    ocr_labels: List[Any] = field(default_factory=list)

    # Output paths
    sh3d_path: Optional[str] = None
    summary_path: Optional[str] = None
    rooms_ocr_path: Optional[str] = None

    # Room area validation reports
    room_area_reports: List[Any] = field(default_factory=list)

    # Openings dropped because their nearest wall was diagonal; exposed so
    # callers can investigate without digging into logs.
    dropped_diagonal_openings: List[Opening] = field(default_factory=list)

    # Legacy raw results (for backward compat)
    raw_measurements: Optional[Any] = None
    raw_yolo_results: Optional[Any] = None

    @property
    def n_walls(self) -> int:
        return len(self.walls)

    @property
    def n_doors(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.DOOR)

    @property
    def n_windows(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.WINDOW)

    @property
    def n_gaps(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.GAP)

    def to_api_dict(self) -> Dict:
        """Serialize for API response."""
        return {
            'n_walls': self.n_walls,
            'n_doors': self.n_doors,
            'n_windows': self.n_windows,
            'n_gaps': self.n_gaps,
            'n_rooms': len(self.rooms),
            'n_door_arcs': len(self.door_arcs),
            'pixel_scale': self.pixel_scale,
            'pixels_to_cm': self.pixels_to_cm,
            'sh3d_path': self.sh3d_path,
        }
