"""
Shared geometry utilities for the ReFloorBRUSNIKA pipeline.

This module provides common geometric operations used across
detection and export modules, eliminating code duplication.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Union

from core.models import BBox


# ============================================================
# INTERSECTION & OVERLAP
# ============================================================

def iou(a: BBox, b: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        a: First bounding box
        b: Second bounding box

    Returns:
        IoU value between 0.0 and 1.0
    """
    inter = intersection_area(a, b)
    if inter == 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """
    Calculate IoU for two (x1, y1, x2, y2) tuples.

    Provided for compatibility with existing code that uses raw tuples.

    Args:
        a: First box as (x1, y1, x2, y2)
        b: Second box as (x1, y1, x2, y2)

    Returns:
        IoU value between 0.0 and 1.0
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def intersection_area(a: BBox, b: BBox) -> float:
    """
    Calculate the intersection area of two bounding boxes.

    Args:
        a: First bounding box
        b: Second bounding box

    Returns:
        Intersection area (0 if no intersection)
    """
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def intersection_area_xyxy(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> int:
    """
    Calculate intersection area for two (x1, y1, x2, y2) integer tuples.

    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)

    Returns:
        Intersection area as integer
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    return max(0, xi2 - xi1) * max(0, yi2 - yi1)


def boxes_intersect(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> bool:
    """
    Check if two boxes intersect.

    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)

    Returns:
        True if boxes intersect, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)


# ============================================================
# POINT OPERATIONS
# ============================================================

def point_in_bbox(px: float, py: float, bbox: BBox) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        px: X coordinate
        py: Y coordinate
        bbox: Bounding box to check

    Returns:
        True if point is inside the bbox
    """
    return bbox.x1 <= px <= bbox.x2 and bbox.y1 <= py <= bbox.y2


def point_in_box_xyxy(
    px: int,
    py: int,
    box: Tuple[int, int, int, int],
) -> bool:
    """
    Check if a point is inside a box (tuple format).

    Args:
        px: X coordinate
        py: Y coordinate
        box: Box as (x1, y1, x2, y2)

    Returns:
        True if point is inside the box
    """
    return box[0] <= px <= box[2] and box[1] <= py <= box[3]


def point_in_polygon(
    px: float,
    py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Check if a point is inside a polygon using ray casting.

    Args:
        px: X coordinate
        py: Y coordinate
        polygon: List of (x, y) vertices

    Returns:
        True if point is inside the polygon
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside

        j = i

    return inside


def distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1, y1: First point
        x2, y2: Second point

    Returns:
        Euclidean distance
    """
    return math.hypot(x2 - x1, y2 - y1)


def distance_point_to_line(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Calculate perpendicular distance from a point to a line segment.

    Args:
        px, py: Point coordinates
        x1, y1: Line start
        x2, y2: Line end

    Returns:
        Perpendicular distance
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return distance(px, py, x1, y1)
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / length


# ============================================================
# POLYGON OPERATIONS
# ============================================================

def polygon_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculate polygon area using the Shoelace formula.

    Args:
        points: List of (x, y) vertices in order

    Returns:
        Absolute area of the polygon
    """
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    j = n - 1

    for i in range(n):
        area += (points[j][0] + points[i][0]) * (points[j][1] - points[i][1])
        j = i

    return abs(area) / 2.0


def polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate the centroid of a polygon.

    Args:
        points: List of (x, y) vertices

    Returns:
        (cx, cy) centroid coordinates
    """
    if not points:
        return (0.0, 0.0)

    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    return (cx, cy)


# ============================================================
# PARALLEL & PROXIMITY CHECKS
# ============================================================

def are_parallel_and_close(
    a: BBox,
    b: BBox,
    gap_px: float = 6.0,
) -> bool:
    """
    Check if two boxes are parallel (same orientation) and close together.

    Used for deduplication of overlapping/parallel detections.

    Args:
        a: First bounding box
        b: Second bounding box
        gap_px: Maximum gap in pixels to consider boxes as "close"

    Returns:
        True if boxes are parallel and within gap_px of each other
    """
    # Must have same orientation
    if a.orientation != b.orientation:
        return False

    if a.orientation == "horizontal":
        # Check vertical proximity
        vertical_dist = min(abs(a.y1 - b.y2), abs(b.y1 - a.y2))
        if vertical_dist > gap_px:
            return False
        # Check horizontal overlap
        overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
        return overlap > 0
    else:
        # Vertical walls - check horizontal proximity
        horiz_dist = min(abs(a.x1 - b.x2), abs(b.x1 - a.x2))
        if horiz_dist > gap_px:
            return False
        # Check vertical overlap
        overlap = min(a.y2, b.y2) - max(a.y1, b.y1)
        return overlap > 0


def are_rects_connected(
    rect1_bbox: Tuple[int, int, int, int],
    rect2_bbox: Tuple[int, int, int, int],
    tolerance: int = 2,
) -> bool:
    """
    Check if two rectangles are connected (touching or overlapping).

    Args:
        rect1_bbox: First rect as (x1, y1, x2, y2)
        rect2_bbox: Second rect as (x1, y1, x2, y2)
        tolerance: Pixel tolerance for connection

    Returns:
        True if rectangles are connected
    """
    x1 = max(0, rect1_bbox[0] - tolerance)
    y1 = max(0, rect1_bbox[1] - tolerance)
    x2 = rect1_bbox[2] + tolerance
    y2 = rect1_bbox[3] + tolerance

    ix1 = max(x1, rect2_bbox[0])
    iy1 = max(y1, rect2_bbox[1])
    ix2 = min(x2, rect2_bbox[2])
    iy2 = min(y2, rect2_bbox[3])

    return ix2 > ix1 and iy2 > iy1


# ============================================================
# COORDINATE TRANSFORMS
# ============================================================

def pixels_to_cm(px: float, scale: float) -> float:
    """
    Convert pixels to centimeters.

    Args:
        px: Value in pixels
        scale: Scale factor (cm per pixel)

    Returns:
        Value in centimeters
    """
    return px * scale


def cm_to_pixels(cm: float, scale: float) -> float:
    """
    Convert centimeters to pixels.

    Args:
        cm: Value in centimeters
        scale: Scale factor (cm per pixel)

    Returns:
        Value in pixels
    """
    if scale == 0:
        return 0.0
    return cm / scale


def pixels_to_meters(px: float, scale: float) -> float:
    """
    Convert pixels to meters.

    Args:
        px: Value in pixels
        scale: Scale factor (meters per pixel)

    Returns:
        Value in meters
    """
    return px * scale


def meters_to_pixels(m: float, scale: float) -> float:
    """
    Convert meters to pixels.

    Args:
        m: Value in meters
        scale: Scale factor (meters per pixel)

    Returns:
        Value in pixels
    """
    if scale == 0:
        return 0.0
    return m / scale


# ============================================================
# SNAPPING & ALIGNMENT
# ============================================================

def snap_to_grid(value: float, grid_size: float) -> float:
    """
    Snap a value to the nearest grid point.

    Args:
        value: Value to snap
        grid_size: Grid spacing

    Returns:
        Snapped value
    """
    if grid_size <= 0:
        return value
    return round(value / grid_size) * grid_size


def snap_to_axis(
    x: float,
    y: float,
    axis_x: Optional[float],
    axis_y: Optional[float],
    tolerance: float,
) -> Tuple[float, float]:
    """
    Snap coordinates to horizontal/vertical axes if within tolerance.

    Args:
        x, y: Coordinates to snap
        axis_x: X coordinate of vertical axis (or None)
        axis_y: Y coordinate of horizontal axis (or None)
        tolerance: Snap tolerance

    Returns:
        (x, y) possibly snapped to axes
    """
    result_x, result_y = x, y
    if axis_x is not None and abs(x - axis_x) <= tolerance:
        result_x = axis_x
    if axis_y is not None and abs(y - axis_y) <= tolerance:
        result_y = axis_y
    return (result_x, result_y)


# ============================================================
# ANGLE OPERATIONS
# ============================================================

def normalize_angle(angle: float) -> float:
    """
    Normalize angle to range [0, 360) degrees.

    Args:
        angle: Angle in degrees

    Returns:
        Normalized angle in [0, 360)
    """
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle


def angle_between_points(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Calculate angle between two points in degrees.

    Args:
        x1, y1: First point
        x2, y2: Second point

    Returns:
        Angle in degrees [0, 360)
    """
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return normalize_angle(angle)


def angle_difference(a1: float, a2: float) -> float:
    """
    Calculate the absolute angular difference between two angles.

    Args:
        a1: First angle in degrees
        a2: Second angle in degrees

    Returns:
        Absolute difference in [0, 180]
    """
    diff = abs(normalize_angle(a1) - normalize_angle(a2))
    if diff > 180:
        diff = 360 - diff
    return diff


# ============================================================
# LINE OPERATIONS
# ============================================================

def line_intersection(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two line segments.

    Args:
        x1, y1, x2, y2: First line segment
        x3, y3, x4, y4: Second line segment

    Returns:
        (x, y) intersection point or None if no intersection
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)

    return None


def project_point_to_line(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[float, Tuple[float, float]]:
    """
    Project a point onto a line segment.

    Args:
        px, py: Point to project
        x1, y1: Line start
        x2, y2: Line end

    Returns:
        (t, (proj_x, proj_y)) where t is the parameter along the line
    """
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-10:
        return (0.0, (x1, y1))

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return (t, (proj_x, proj_y))
