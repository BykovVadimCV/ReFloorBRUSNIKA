"""
Unified data models for the ReFloorBRUSNIKA pipeline.

This module defines all shared data structures used across detection,
export, and visualization modules. Using unified models eliminates
redundant class definitions and ensures consistent data representation.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# ENUMERATIONS
# ============================================================

class OpeningType(Enum):
    """Type classification for wall openings."""
    DOOR = "door"
    WINDOW = "window"
    GAP = "gap"


class SwingDirection(str, Enum):
    """Door swing direction from the room interior's perspective."""
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass
class BBox:
    """
    Unified axis-aligned bounding box representation.

    Supports multiple coordinate formats and provides common
    geometric properties and conversions.
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return abs(self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def center_x(self) -> float:
        """X coordinate of center."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Y coordinate of center."""
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height), or inf if height is 0."""
        if self.height == 0:
            return float('inf')
        return self.width / self.height

    @property
    def orientation(self) -> str:
        """Returns 'horizontal' if wider than tall, else 'vertical'."""
        return "horizontal" if self.width >= self.height else "vertical"

    @property
    def is_horizontal(self) -> bool:
        """True if the box is wider than tall."""
        return self.width >= self.height

    @property
    def is_vertical(self) -> bool:
        """True if the box is taller than wide."""
        return self.height > self.width

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BBox:
        """Create BBox from (x, y, width, height) format."""
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_center(cls, cx: float, cy: float, w: float, h: float) -> BBox:
        """Create BBox from center point and dimensions."""
        return cls(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BBox:
        """
        Create BBox from dictionary with various formats.

        Supports:
            - {x1, y1, x2, y2}
            - {x, y, w, h}
            - {x, y, width, height}
        """
        if all(k in d for k in ('x1', 'y1', 'x2', 'y2')):
            return cls(d['x1'], d['y1'], d['x2'], d['y2'])
        if all(k in d for k in ('x', 'y', 'w', 'h')):
            return cls.from_xywh(d['x'], d['y'], d['w'], d['h'])
        if all(k in d for k in ('x', 'y', 'width', 'height')):
            return cls.from_xywh(d['x'], d['y'], d['width'], d['height'])
        raise ValueError(f"Unknown rectangle format: {list(d.keys())}")

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x1, self.y1, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to integer (x1, y1, x2, y2) tuple."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'width': self.width,
            'height': self.height,
        }

    def scale(self, factor: float) -> BBox:
        """Return a new BBox scaled by the given factor."""
        return BBox(
            self.x1 * factor,
            self.y1 * factor,
            self.x2 * factor,
            self.y2 * factor,
        )

    def expand(self, margin: float) -> BBox:
        """Return a new BBox expanded by the given margin."""
        return BBox(
            self.x1 - margin,
            self.y1 - margin,
            self.x2 + margin,
            self.y2 + margin,
        )

    def intersects(self, other: BBox) -> bool:
        """Check if this box intersects with another."""
        return not (
            self.x2 <= other.x1 or
            other.x2 <= self.x1 or
            self.y2 <= other.y1 or
            other.y2 <= self.y1
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


# ============================================================
# POINT
# ============================================================

@dataclass
class Point:
    """2D point with distance and equality methods."""
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def copy(self) -> Point:
        """Create a copy of this point."""
        return Point(self.x, self.y)

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (x, y) tuple."""
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer (x, y) tuple."""
        return (int(self.x), int(self.y))

    def __hash__(self) -> int:
        return hash((round(self.x, 2), round(self.y, 2)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01


# ============================================================
# OPENING (UNIFIED)
# ============================================================

@dataclass
class Opening:
    """
    Unified representation of a wall opening (door, window, or gap).

    This class consolidates the various opening representations from
    different detection modules into a single, canonical format.
    """
    bbox: BBox
    opening_type: OpeningType
    confidence: float = 1.0

    # Door-specific attributes (optional)
    hinge_point: Optional[Tuple[int, int]] = None
    swing_direction: SwingDirection = SwingDirection.UNKNOWN
    swing_clockwise: bool = True
    arc_radius_px: Optional[float] = None

    # Wall association
    parent_wall_id: Optional[str] = None

    # Source tracking
    source: str = "unknown"  # "yolo", "geometric", "segmentation", "algorithmic"

    # Unique identifier
    id: int = field(default_factory=lambda: id(object()))

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the opening."""
        return self.bbox.center

    @property
    def width(self) -> float:
        """Width of the opening."""
        return self.bbox.width

    @property
    def height(self) -> float:
        """Height of the opening."""
        return self.bbox.height

    @property
    def is_door(self) -> bool:
        """True if this opening is a door."""
        return self.opening_type == OpeningType.DOOR

    @property
    def is_window(self) -> bool:
        """True if this opening is a window."""
        return self.opening_type == OpeningType.WINDOW

    @property
    def is_gap(self) -> bool:
        """True if this opening is a gap."""
        return self.opening_type == OpeningType.GAP

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'x1': self.bbox.x1,
            'y1': self.bbox.y1,
            'x2': self.bbox.x2,
            'y2': self.bbox.y2,
            'width': self.width,
            'height': self.height,
            'type': self.opening_type.value,
            'confidence': self.confidence,
            'source': self.source,
            'hinge_point': self.hinge_point,
            'swing_direction': self.swing_direction.value if self.swing_direction else None,
            'swing_clockwise': self.swing_clockwise,
            'parent_wall_id': self.parent_wall_id,
        }


# ============================================================
# WALL SEGMENT
# ============================================================

@dataclass
class WallSegment:
    """
    Unified wall segment representation.

    Represents a detected wall section with bounding box,
    thickness, and structural properties.
    """
    id: str
    bbox: BBox
    thickness: float
    is_structural: bool = False
    is_outer: bool = False

    # Optional outline for complex shapes
    outline: Optional[List[Tuple[int, int]]] = None

    @property
    def orientation(self) -> str:
        """Returns 'horizontal' or 'vertical' based on bbox dimensions."""
        return self.bbox.orientation

    @property
    def is_horizontal(self) -> bool:
        """True if the wall is horizontal."""
        return self.bbox.is_horizontal

    @property
    def is_vertical(self) -> bool:
        """True if the wall is vertical."""
        return self.bbox.is_vertical

    @property
    def centerline(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns (start_point, end_point) along the wall center.

        For horizontal walls: left-center to right-center
        For vertical walls: top-center to bottom-center
        """
        if self.is_horizontal:
            cy = self.bbox.center_y
            return ((self.bbox.x1, cy), (self.bbox.x2, cy))
        else:
            cx = self.bbox.center_x
            return ((cx, self.bbox.y1), (cx, self.bbox.y2))

    @property
    def length(self) -> float:
        """Length of the wall (along its major axis)."""
        return max(self.bbox.width, self.bbox.height)

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the wall."""
        return self.bbox.center

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'x1': self.bbox.x1,
            'y1': self.bbox.y1,
            'x2': self.bbox.x2,
            'y2': self.bbox.y2,
            'width': self.bbox.width,
            'height': self.bbox.height,
            'thickness': self.thickness,
            'is_structural': self.is_structural,
            'is_outer': self.is_outer,
            'orientation': self.orientation,
        }


# ============================================================
# WALL (FOR EXPORT)
# ============================================================

@dataclass
class Wall:
    """
    Wall representation for SH3D export.

    Uses start/end points and centerline representation
    suitable for SweetHome3D format.
    """
    start: Point
    end: Point
    thickness: float
    height: float = 243.84  # Default wall height in cm
    wall_id: str = field(default_factory=lambda: f'wall-{uuid.uuid4()}')
    wall_at_start: Optional[str] = None
    wall_at_end: Optional[str] = None
    pattern: str = 'hatchUp'
    is_structural: bool = True

    @property
    def length(self) -> float:
        """Length of the wall."""
        return self.start.distance_to(self.end)

    @property
    def midpoint(self) -> Point:
        """Midpoint of the wall."""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
        )

    @property
    def angle(self) -> float:
        """Angle of the wall in radians."""
        return math.atan2(
            self.end.y - self.start.y,
            self.end.x - self.start.x,
        )

    @property
    def is_horizontal(self) -> bool:
        """True if the wall is predominantly horizontal."""
        return abs(self.end.y - self.start.y) < abs(self.end.x - self.start.x)

    def perpendicular_distance(self, point: Point) -> float:
        """Calculate perpendicular distance from a point to this wall."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = self.length
        if length < 0.001:
            return point.distance_to(self.start)
        return abs(
            dy * point.x - dx * point.y
            + self.end.x * self.start.y
            - self.end.y * self.start.x
        ) / length

    def project_point(self, point: Point) -> Tuple[float, Point]:
        """Project a point onto this wall, returning (t, projected_point)."""
        length_sq = self.length ** 2
        if length_sq < 0.001:
            return 0.0, self.start.copy()
        t = (
            (point.x - self.start.x) * (self.end.x - self.start.x)
            + (point.y - self.start.y) * (self.end.y - self.start.y)
        ) / length_sq
        return t, Point(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y),
        )

    def get_min_x(self) -> float:
        return min(self.start.x, self.end.x)

    def get_max_x(self) -> float:
        return max(self.start.x, self.end.x)

    def get_min_y(self) -> float:
        return min(self.start.y, self.end.y)

    def get_max_y(self) -> float:
        return max(self.start.y, self.end.y)


# ============================================================
# ROOM
# ============================================================

@dataclass
class Room:
    """Room polygon for SH3D export."""
    points: List[Point]
    room_id: str = field(default_factory=lambda: f'room-{uuid.uuid4()}')
    name: str = 'Room'
    area_m2: Optional[float] = None

    @property
    def centroid(self) -> Point:
        """Calculate the centroid of the room polygon."""
        if not self.points:
            return Point(0, 0)
        cx = sum(p.x for p in self.points) / len(self.points)
        cy = sum(p.y for p in self.points) / len(self.points)
        return Point(cx, cy)


# ============================================================
# FUSED DOOR
# ============================================================

@dataclass
class FusedDoor:
    """
    Canonical door representation produced by DoorFusionEngine.

    Combines information from YOLO detection, gap detection,
    and door arc detection into a single authoritative record.
    """
    id: int
    hinge: Tuple[int, int]
    center: Tuple[int, int]
    width_px: int
    wall_normal_deg: float
    direction: str  # "left" | "right"
    swing_clockwise: bool
    confidence: float  # 0.0 - 1.0
    matched_gap_idx: int = -1
    source: str = "arc"  # "arc" | "yolo_gap" | "yolo_only"

    def to_opening(self, pixels_to_cm: float = 1.0) -> Opening:
        """Convert to unified Opening format."""
        half_width = self.width_px / 2
        cx, cy = self.center
        bbox = BBox(
            cx - half_width,
            cy - half_width,
            cx + half_width,
            cy + half_width,
        )
        return Opening(
            bbox=bbox,
            opening_type=OpeningType.DOOR,
            confidence=self.confidence,
            hinge_point=self.hinge,
            swing_direction=SwingDirection(self.direction) if self.direction in ('left', 'right') else SwingDirection.UNKNOWN,
            swing_clockwise=self.swing_clockwise,
            source=self.source,
        )


# ============================================================
# DETECTED WINDOW
# ============================================================

@dataclass
class DetectedWindow:
    """
    Detected window from algorithmic detection.

    Lightweight representation for the window detector output.
    """
    id: int
    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> BBox:
        """Get as BBox."""
        return BBox.from_xywh(self.x, self.y, self.w, self.h)

    def to_opening(self) -> Opening:
        """Convert to unified Opening format."""
        return Opening(
            bbox=self.bbox,
            opening_type=OpeningType.WINDOW,
            confidence=1.0,
            source="algorithmic",
        )


# ============================================================
# DETECTED DOOR ARC
# ============================================================

@dataclass
class DetectedDoorArc:
    """
    Final public output from door arc detection.

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
    direction: str  # "left" | "right" | "unknown"
    confidence: float  # 0.0 - 1.0
    matched_opening_idx: int = -1
    wall_normal_deg: float = 0.0
    swing_clockwise: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for JSON/CSV serialization."""
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


# ============================================================
# DETECTION RESULT
# ============================================================

@dataclass
class DetectionResult:
    """
    Unified output from any detector.

    Provides a consistent interface for detection results
    regardless of the detection method used.
    """
    walls: List[WallSegment] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    wall_mask: Optional[np.ndarray] = None
    outline_mask: Optional[np.ndarray] = None

    # Metadata
    pixel_scale: Optional[float] = None  # meters per pixel
    source: str = "unknown"

    @property
    def doors(self) -> List[Opening]:
        """Get all door openings."""
        return [o for o in self.openings if o.opening_type == OpeningType.DOOR]

    @property
    def windows(self) -> List[Opening]:
        """Get all window openings."""
        return [o for o in self.openings if o.opening_type == OpeningType.WINDOW]

    @property
    def gaps(self) -> List[Opening]:
        """Get all gap openings."""
        return [o for o in self.openings if o.opening_type == OpeningType.GAP]

    def merge_with(self, other: DetectionResult) -> DetectionResult:
        """Merge with another detection result."""
        return DetectionResult(
            walls=self.walls + other.walls,
            openings=self.openings + other.openings,
            wall_mask=self.wall_mask if self.wall_mask is not None else other.wall_mask,
            outline_mask=self.outline_mask if self.outline_mask is not None else other.outline_mask,
            pixel_scale=self.pixel_scale or other.pixel_scale,
            source=f"{self.source}+{other.source}",
        )
