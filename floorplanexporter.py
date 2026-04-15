"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: Экспорт в SweetHome3D (floorplanexporter.py)
Версия: 6.0 — Flood-Fill Room Detection + Correct Opening Sizing
Автор: Быков Вадим Олегович
Дата: 2 марта 2026 г.
================================================================================
"""
import math
import uuid
import zipfile
import base64
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import cv2
import numpy as np

# ============================================================
# Встроенные 3D-ресурсы (base64 / OBJ)
# ============================================================
DUMMY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAA"
    "BJRU5ErkJggg=="
)

DOOR_OBJ_CONTENT = """# door.obj
g sweethome3d_hinge_1_3
v -0.40393 -0.84 0.02143
v -0.401 -0.84 0.0285
v -0.411 -0.84 0.0285
v -0.411 -0.84 0.0185
v -0.41807 -0.84 0.02143
v -0.421 -0.84 0.0285
v -0.41807 -0.84 0.03557
v -0.411 -0.84 0.0385
v -0.40393 -0.84 0.03557
v -0.401 -0.76 0.0285
v -0.40393 -0.76 0.02143
v -0.411 -0.76 0.0185
v -0.41807 -0.76 0.02143
v -0.421 -0.76 0.0285
v -0.41807 -0.76 0.03557
v -0.411 -0.76 0.0385
v -0.40393 -0.76 0.03557
v -0.411 -0.76 0.0285
usemtl white
f 1 2 3
f 4 1 3
f 5 4 3
f 6 5 3
f 7 6 3
f 8 7 3
f 9 8 3
f 2 9 3
f 18 10 11
f 18 11 12
f 18 12 13
f 18 13 14
f 18 14 15
f 18 15 16
f 18 16 17
f 18 17 10
f 10 2 1
f 11 1 4
f 12 4 5
f 13 5 6
f 14 6 7
f 15 7 8
f 16 8 9
f 17 9 2
f 10 1 11
f 11 4 12
f 12 5 13
f 13 6 14
f 14 7 15
f 15 8 16
f 16 9 17
f 17 2 10
g frame
v 0.4575 -1.02 0.0195
v 0.4575 -1.02 -0.0755
v 0.4575 1.065 -0.0755
v 0.4575 1.065 0.0195
v -0.4575 -1.02 -0.0755
v -0.4575 -1.02 0.0195
v -0.4575 1.065 0.0195
v -0.4575 1.065 -0.0755
v -0.4025 1.01 -0.0755
v -0.4025 -1.02 -0.0755
v 0.4025 -1.02 -0.0755
v 0.4025 -1.02 -0.02
v 0.4025 1.01 -0.02
v 0.4025 1.01 -0.0755
v -0.4135 -1.02 -0.02
v 0.4135 1.021 0.0195
v 0.4135 1.021 -0.02
v -0.4135 -1.02 0.0195
v -0.4135 1.021 0.0195
v 0.4135 -1.02 0.0195
v -0.4025 1.01 -0.02
v -0.4025 -1.02 -0.02
v -0.4135 1.021 -0.02
v 0.4135 -1.02 -0.02
usemtl white
s off
f 22 19 20 21
f 29 30 31 32
f 26 25 22 21
f 41 37 36 33
f 25 26 23 24
f 37 41 35 34
f 39 40 28 27
f 24 36 37 34 38 19 22 25
f 38 34 35 42
f 23 26 21 20 29 32 27 28
f 27 32 31 39
f 33 36 24 23
f 20 19 38 42
f 23 28 40 33
f 41 33 40 39
f 30 42 31
f 41 39 31 35
f 31 42 35
f 29 20 42 30
"""

WINDOW_OBJ_CONTENT = """# window85x163.obj
g sweethome3d_hinge_1_1
v -0.41601 2.01433 -0.00407
v -0.41308 2.01433 0.003
v -0.42308 2.01433 0.003
v -0.42308 2.01433 -0.007
v -0.43015 2.01433 -0.00407
v -0.43308 2.01433 0.003
v -0.43015 2.01433 0.01007
v -0.42308 2.01433 0.013
v -0.41601 2.01433 0.01007
v -0.42308 2.05433 0.003
v -0.41308 2.05433 0.003
v -0.41601 2.05433 -0.00407
v -0.42308 2.05433 -0.007
v -0.43015 2.05433 -0.00407
v -0.43308 2.05433 0.003
v -0.43015 2.05433 0.01007
v -0.42308 2.05433 0.013
v -0.41601 2.05433 0.01007
usemtl white
f 1 2 3
f 4 1 3
f 5 4 3
f 6 5 3
f 7 6 3
f 8 7 3
f 9 8 3
f 2 9 3
f 10 11 12
f 10 12 13
f 10 13 14
f 10 14 15
f 10 15 16
f 10 16 17
f 10 17 18
f 10 18 11
f 7 16 15 6
f 1 12 11 2
f 2 11 18 9
f 9 18 17 8
f 6 15 14 5
f 8 17 16 7
f 5 14 13 4
f 4 13 12 1
g sweethome3d_opening_on_hinge_1
v 0.415 0.53367 0.0125
v 0.34067 0.61867 0.0125
v -0.33933 0.61867 0.0125
v -0.415 0.53367 0.0125
v -0.33933 0.69667 0.0125
v -0.33933 2.07867 0.0125
v -0.415 2.16367 0.0125
v 0.34067 2.07867 0.0125
v 0.415 2.16367 0.0125
v 0.34067 0.91067 0.0125
v -0.33933 2.07867 -0.0475
v 0.34067 2.07867 -0.0475
v 0.398 2.16367 -0.0475
v -0.398 2.16367 -0.0475
v 0.398 0.53367 -0.0475
v 0.34067 0.61867 -0.0475
v 0.34067 1.78667 -0.0475
v -0.398 0.53367 -0.0475
v -0.33933 2.00067 -0.0475
v -0.33933 0.61867 -0.0475
v -0.398 0.53367 -0.00275
v -0.415 0.53367 -0.00275
v -0.398 2.16367 -0.00275
v -0.415 2.16367 -0.00275
v 0.398 2.16367 -0.00275
v 0.415 2.16367 -0.00275
v 0.398 0.53367 -0.00275
v 0.415 0.53367 -0.00275
usemtl white
s off
f 19 20 21 22
f 22 21 23 24 25
f 25 24 26 27
f 28 20 19 27 26
f 24 29 30 26
f 27 44 43 31 32 41 42 25
f 30 31 33 34 35
f 32 36 39 41
f 33 31 43 45
f 29 37 38 36 32
f 21 20 34 38
f 36 33 45 46 19 22 40 39
f 28 26 30 35 34 20
f 24 23 21 38 37 29
f 30 29 32 31
f 38 34 33 36
f 41 39 40 42
f 42 40 22 25
f 45 43 44 46
f 46 44 27 19
g frame
v -0.455 0.475 -0.00421
v -0.455 0.475 -0.27421
v 0.455 0.475 -0.27421
v 0.455 0.475 -0.00421
v -0.455 2.215 -0.00421
v -0.455 2.215 -0.27421
v 0.455 2.215 -0.00421
v 0.455 2.215 -0.27421
v -0.4 2.1637 -0.00421
v -0.4 2.1637 -0.27421
v 0.4 2.1637 -0.27421
v 0.4 2.1637 -0.00421
v -0.4 0.5337 -0.00421
v -0.4 0.5337 -0.27421
v 0.4 0.5337 -0.27421
v 0.4 0.5337 -0.00421
usemtl white
s off
f 47 48 49 50
f 51 52 48 47
f 53 54 52 51
f 55 56 57 58
f 57 56 52 54
f 55 59 60 56
f 53 50 49 54
f 49 48 60 61
f 61 60 59 62
f 57 61 62 58
f 55 58 53 51
f 58 62 50 53
f 55 51 47 59
f 54 49 61 57
f 47 50 62 59
f 48 52 56 60
g sweethome3d_window_pane_on_hinge_1
v -0.34433 2.08367 -0.0125
v -0.34433 0.61367 -0.0125
v 0.34567 0.61367 -0.0125
v 0.34567 2.08367 -0.0125
v -0.34433 2.08367 -0.0225
v -0.34433 0.61367 -0.0225
v 0.34567 0.61367 -0.0225
v 0.34567 2.08367 -0.0225
usemtl flltgrey
s off
f 63 64 65 66
f 67 68 64 63
f 68 69 65 64
f 70 67 63 66
f 70 69 68 67
f 69 70 66 65
"""

# ============================================================
# Конфигурация — все константы в одном месте
# ============================================================
SCALE_FACTOR: float = 2.2
DEFAULT_WALL_HEIGHT: float = 243.84
DEFAULT_WALL_THICKNESS: float = 15.0
DEFAULT_DOOR_HEIGHT: float = 208.5975
DEFAULT_WINDOW_HEIGHT: float = 173.98999
DEFAULT_WINDOW_ELEVATION: float = 46.989998
WALL_EXTENSION_FACTOR: float = 0.05
WALL_OVERLAP: float = 2.0
SNAP_TOLERANCE: float = 40.0
AXIS_SNAP_TOLERANCE: float = 15.0
T_JUNCTION_SNAP_TOLERANCE: float = 30.0
PARENT_WALL_SEARCH_TOLERANCE: float = 50.0
OPENING_AXIS_SNAP_TOLERANCE: float = 30.0
OPENING_EXTENSION_TOLERANCE: float = 30.0
DOOR_BBOX_EXTEND_RATIO: float = 0.05
MIN_GAP_CM: float = 5.0
GAP_DARK_THRESHOLD: int = 100
GAP_DARK_RATIO_LIMIT: float = 0.30

# --- REMOVED: these constants are no longer used ---
# DEFAULT_DOOR_WIDTH / DEFAULT_WINDOW_WIDTH — opening width now comes
# from the detected rectangle, not a hard-coded default.
# OPENING_TO_WALL_RATIO — was used to stretch openings to parent wall
# length; removed entirely.


# ============================================================
# Перечисление типов проёмов
# ============================================================
class OpeningType(Enum):
    DOOR = 'door'
    WINDOW = 'window'
    GAP = 'gap'


# ============================================================
# Структуры данных
# ============================================================
@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def __hash__(self) -> int:
        return hash((round(self.x, 2), round(self.y, 2)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01


@dataclass
class Wall:
    start: Point
    end: Point
    thickness: float
    height: float = DEFAULT_WALL_HEIGHT
    wall_id: str = field(default_factory=lambda: f'wall-{uuid.uuid4()}')
    wall_at_start: Optional[str] = None
    wall_at_end: Optional[str] = None
    pattern: str = 'hatchUp'
    is_structural: bool = True
    # Original detection bounding box in cm (x1, y1, x2, y2) — full rectangle as
    # returned by the detector, before the wall is narrowed to a centreline.
    # Used by _snap_doors_to_wall_gaps for proximity search.
    detection_bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def midpoint(self) -> Point:
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
        )

    @property
    def angle(self) -> float:
        return math.atan2(
            self.end.y - self.start.y,
            self.end.x - self.start.x,
        )

    @property
    def is_horizontal(self) -> bool:
        return abs(self.end.y - self.start.y) < abs(self.end.x - self.start.x)

    def perpendicular_distance(self, point: Point) -> float:
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

    def get_min_x(self) -> float: return min(self.start.x, self.end.x)
    def get_max_x(self) -> float: return max(self.start.x, self.end.x)
    def get_min_y(self) -> float: return min(self.start.y, self.end.y)
    def get_max_y(self) -> float: return max(self.start.y, self.end.y)
    def get_centerline_y(self) -> float: return (self.start.y + self.end.y) / 2.0
    def get_centerline_x(self) -> float: return (self.start.x + self.end.x) / 2.0


@dataclass
class Opening:
    opening_type: OpeningType
    center: Point
    width: float          # FIX: now set from detected rectangle, not wall length
    height: float
    depth: float
    angle: float
    elevation: float = 0.0
    parent_wall_id: Optional[str] = None
    opening_id: str = field(default_factory=lambda: f'opening-{uuid.uuid4()}')
    wall_thickness: float = 0.5
    wall_distance: float = 0.0
    model: str = ''
    icon: str = ''
    hinge: Optional[Tuple[int, int]] = None
    swing_clockwise: bool = True
    wall_normal_deg: float = 0.0


@dataclass
class Room:
    points: List[Point]
    room_id: str = field(default_factory=lambda: f'room-{uuid.uuid4()}')
    name: str = 'Room'


@dataclass
class WallSegment:
    wall_id: str
    start: Point
    end: Point
    original_wall_id: str


# ============================================================
# Утилиты
# ============================================================
def parse_rectangle(rect: Dict) -> Tuple[float, float, float, float]:
    if all(k in rect for k in ('x1', 'y1', 'x2', 'y2')):
        return rect['x1'], rect['y1'], rect['x2'], rect['y2']
    if all(k in rect for k in ('x', 'y', 'width', 'height')):
        return (rect['x'], rect['y'],
                rect['x'] + rect['width'], rect['y'] + rect['height'])
    if all(k in rect for k in ('x', 'y', 'w', 'h')):
        return rect['x'], rect['y'], rect['x'] + rect['w'], rect['y'] + rect['h']
    raise ValueError(f'Unknown rectangle format: {list(rect.keys())}')


# ============================================================
# DebugVisualizer — CV2-визуализация для отладки экспорта
# ============================================================
class DebugVisualizer:
    ROOM_COLORS: List[Tuple[int, int, int]] = [
        (255, 180, 180), (180, 255, 180), (180, 180, 255),
        (255, 255, 180), (255, 180, 255), (180, 255, 255),
        (220, 200, 160), (160, 220, 200), (200, 160, 220),
    ]

    def __init__(self, pixels_to_cm: float) -> None:
        self.pixels_to_cm = pixels_to_cm

    def _cm_to_px(self, val: float) -> int:
        return int(val / self.pixels_to_cm)

    def _point_to_px(self, p: Point) -> Tuple[int, int]:
        return self._cm_to_px(p.x), self._cm_to_px(p.y)

    def create_debug_image(
        self,
        original_image: np.ndarray,
        wall_rectangles: List[Dict],
        walls: List[Wall],
        rooms: List[Room],
        openings: List[Opening],
        ocr_labels: Optional[List] = None,
        wall_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        canvas = original_image.copy()
        if wall_mask is not None:
            self._draw_wall_mask(canvas, wall_mask)
        if rooms:
            room_overlay = canvas.copy()
            for i, room in enumerate(rooms):
                color = self.ROOM_COLORS[i % len(self.ROOM_COLORS)]
                pts = np.array(
                    [self._point_to_px(p) for p in room.points], dtype=np.int32
                )
                cv2.fillPoly(room_overlay, [pts], color)
            cv2.addWeighted(room_overlay, 0.25, canvas, 0.75, 0, canvas)
        for rect in wall_rectangles:
            x1, y1, x2, y2 = parse_rectangle(rect)
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          (90, 90, 90), 1)
        for wall in walls:
            pt1 = self._point_to_px(wall.start)
            pt2 = self._point_to_px(wall.end)
            color = (0, 0, 220) if wall.is_structural else (220, 120, 0)
            cv2.line(canvas, pt1, pt2, color, 2 if wall.is_structural else 1)
        for i, room in enumerate(rooms):
            outline = tuple(max(0, c - 100) for c in self.ROOM_COLORS[i % len(self.ROOM_COLORS)])
            px_pts = [self._point_to_px(p) for p in room.points]
            cv2.polylines(canvas, [np.array(px_pts, dtype=np.int32)], True, outline, 2)
            cx = int(sum(p[0] for p in px_pts) / len(px_pts))
            cy = int(sum(p[1] for p in px_pts) / len(px_pts))
            cv2.putText(canvas, room.name, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline, 1, cv2.LINE_AA)
        for opening in openings:
            cx, cy = self._point_to_px(opening.center)
            if opening.opening_type == OpeningType.DOOR:
                color, label = (255, 80, 0), 'D'
            elif opening.opening_type == OpeningType.WINDOW:
                color, label = (0, 200, 255), 'W'
            else:
                color, label = (0, 140, 255), 'G'
            cv2.circle(canvas, (cx, cy), 6, color, -1)
            cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), 1)
            cv2.putText(canvas, label, (cx + 9, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            if opening.hinge is not None:
                hx, hy = opening.hinge
                cv2.drawMarker(canvas, (hx, hy), (0, 255, 255),
                               cv2.MARKER_DIAMOND, 8, 2)
        if ocr_labels:
            self._draw_ocr_labels(canvas, ocr_labels)
        return canvas

    def _draw_wall_mask(
        self, canvas: np.ndarray, wall_mask: np.ndarray
    ) -> None:
        h, w = canvas.shape[:2]
        mh, mw = wall_mask.shape[:2]
        mask = (
            cv2.resize(wall_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if (mh != h or mw != w) else wall_mask
        )
        mask_bool = mask > 0
        fill = canvas.copy()
        fill[mask_bool] = (180, 100, 200)
        cv2.addWeighted(fill, 0.25, canvas, 0.75, 0, canvas)
        contours, _ = cv2.findContours(
            (mask_bool.astype(np.uint8)) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, (200, 50, 200), 1)

    def _draw_ocr_labels(
        self, canvas: np.ndarray, ocr_labels: List
    ) -> None:
        for text, x1, y1, x2, y2 in ocr_labels:
            is_num = self._is_numeric(text)
            bc = (0, 200, 0) if is_num else (160, 160, 0)
            tc = (0, 180, 0) if is_num else (140, 140, 0)
            th = 2 if is_num else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bc, th)
            (tw, tth), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            ly = max(y1 - 4, tth + 2)
            cv2.rectangle(canvas, (x1, ly - tth - 2), (x1 + tw + 4, ly + 2),
                          (255, 255, 255), -1)
            cv2.putText(canvas, text, (x1 + 2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, tc, 1, cv2.LINE_AA)

    @staticmethod
    def _is_numeric(text: str) -> bool:
        cleaned = text.strip().replace(',', '.').replace(' ', '')
        if not cleaned:
            return False
        try:
            float(cleaned)
            return True
        except ValueError:
            return False


# ============================================================
# StructuralWallConverter
# ============================================================
class StructuralWallConverter:
    def __init__(
        self, pixels_to_cm: float = 2.54, min_thickness: float = 2.0
    ) -> None:
        self.pixels_to_cm = pixels_to_cm
        self.min_thickness = min_thickness

    def convert(self, rectangles: List[Dict]) -> List[Wall]:
        walls: List[Wall] = []
        for rect in rectangles:
            x1, y1, x2, y2 = parse_rectangle(rect)
            x1 *= self.pixels_to_cm; y1 *= self.pixels_to_cm
            x2 *= self.pixels_to_cm; y2 *= self.pixels_to_cm
            width, height = abs(x2 - x1), abs(y2 - y1)
            if width >= height:
                thickness = max(height, self.min_thickness)
                cy = (y1 + y2) / 2
                start, end = Point(min(x1, x2), cy), Point(max(x1, x2), cy)
            else:
                thickness = max(width, self.min_thickness)
                cx = (x1 + x2) / 2
                start, end = Point(cx, min(y1, y2)), Point(cx, max(y1, y2))
            walls.append(Wall(start=start, end=end, thickness=thickness,
                              is_structural=True))
        return walls


# ============================================================
# ThicknessNormalizer
# ============================================================
class ThicknessNormalizer:
    def __init__(
        self, structural_walls: List[Wall], search_radius: float = 80.0
    ) -> None:
        self.structural_walls = [w for w in structural_walls if w.is_structural]
        self.search_radius = search_radius

    def get_thickness_for_segment(
        self, start: Point, end: Point
    ) -> float:
        midpoint = Point((start.x + end.x) / 2, (start.y + end.y) / 2)
        is_horizontal = abs(end.x - start.x) > abs(end.y - start.y)
        candidates: List[Tuple[int, float, float]] = []
        for wall in self.structural_walls:
            if wall.perpendicular_distance(midpoint) > self.search_radius:
                continue
            t, _ = wall.project_point(midpoint)
            if t < -0.5 or t > 1.5:
                continue
            priority = 0 if (wall.is_horizontal == is_horizontal) else 1
            candidates.append((
                priority,
                wall.perpendicular_distance(midpoint),
                wall.thickness,
            ))
        if not candidates:
            return DEFAULT_WALL_THICKNESS
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]


# ============================================================
# ParentWallFinder
# ============================================================
class ParentWallFinder:
    def __init__(
        self,
        walls: List[Wall],
        tolerance: float = PARENT_WALL_SEARCH_TOLERANCE,
    ) -> None:
        self.walls = walls
        self.tolerance = tolerance

    def find_parent(
        self, center: Point, opening_width: float, is_horizontal: bool
    ) -> Optional[Wall]:
        best_wall: Optional[Wall] = None
        best_distance = float('inf')
        for wall in self.walls:
            if wall.is_horizontal != is_horizontal:
                continue
            dx = wall.end.x - wall.start.x
            dy = wall.end.y - wall.start.y
            length_sq = dx * dx + dy * dy
            if length_sq < 0.001:
                continue
            wall_length = math.sqrt(length_sq)
            t = (
                (center.x - wall.start.x) * dx
                + (center.y - wall.start.y) * dy
            ) / length_sq
            # Allow the opening centre to project slightly beyond the
            # wall endpoints — doors in a gap between two walls have
            # their centre between the endpoints of adjacent segments.
            margin = (opening_width / 2) / wall_length if wall_length > 0 else 0.5
            if t < -margin or t > 1 + margin:
                continue
            proj = Point(
                wall.start.x + max(0.0, min(1.0, t)) * dx,
                wall.start.y + max(0.0, min(1.0, t)) * dy,
            )
            perp_dist = center.distance_to(proj)
            if perp_dist < self.tolerance and perp_dist < best_distance:
                best_distance = perp_dist
                best_wall = wall
        return best_wall


# ============================================================
# OpeningProcessor
# ============================================================
class OpeningProcessor:
    def __init__(self, pixels_to_cm: float = 2.54) -> None:
        self.pixels_to_cm = pixels_to_cm

    def process(
        self,
        structural_walls: List[Wall],
        door_rects: Optional[List[Dict]] = None,
        window_rects: Optional[List[Dict]] = None,
        gap_rects: Optional[List[Dict]] = None,
        fused_doors: Optional[List] = None,
    ) -> Tuple[List[Wall], List[Opening]]:
        parent_finder = ParentWallFinder(structural_walls)
        thickness_normalizer = ThicknessNormalizer(structural_walls)
        new_walls: List[Wall] = []
        openings: List[Opening] = []
        processed_centers: List[Point] = []

        if fused_doors:
            for fd in fused_doors:
                result = self._process_fused_door(
                    fd, parent_finder, thickness_normalizer
                )
                if result is None:
                    continue
                opening, wall = result
                if any(opening.center.distance_to(pc) < 20.0 for pc in processed_centers):
                    continue
                processed_centers.append(opening.center)
                openings.append(opening)
                if wall:
                    new_walls.append(wall)
            door_rects = None

        all_rects: List[Tuple[Dict, OpeningType]] = []
        if window_rects:
            all_rects.extend((r, OpeningType.WINDOW) for r in window_rects)
        if gap_rects:
            all_rects.extend((r, OpeningType.GAP) for r in gap_rects)
        if door_rects:
            all_rects.extend((r, OpeningType.DOOR) for r in door_rects)

        for rect, opening_type in all_rects:
            result = self._process_single(
                rect, opening_type, parent_finder, thickness_normalizer
            )
            if result is None:
                continue
            opening, wall = result
            if any(opening.center.distance_to(pc) < 20.0 for pc in processed_centers):
                continue
            processed_centers.append(opening.center)
            openings.append(opening)
            if wall:
                new_walls.append(wall)
        return new_walls, openings

    def _process_fused_door(
        self,
        fused_door,
        parent_finder: ParentWallFinder,
        thickness_normalizer: ThicknessNormalizer,
    ) -> Optional[Tuple[Opening, Optional[Wall]]]:
        cx, cy = fused_door.center
        cx_cm = cx * self.pixels_to_cm
        cy_cm = cy * self.pixels_to_cm
        center = Point(cx_cm, cy_cm)

        # FIX: derive opening width from the fused door's detected pixel width
        width_cm = fused_door.width_px * self.pixels_to_cm
        is_horizontal = abs(math.cos(math.radians(fused_door.wall_normal_deg))) > 0.707

        ow = width_cm                       # ← actual detected gap width
        oh = DEFAULT_DOOR_HEIGHT
        elev = 0.0
        wt, wd = 0.51724136, 0.06896552
        model, icon = '1/door/door.obj', '0'

        parent = parent_finder.find_parent(center, width_cm, is_horizontal)
        new_wall: Optional[Wall] = None

        if parent:
            pid, depth, angle = parent.wall_id, parent.thickness, parent.angle
        else:
            thickness = thickness_normalizer.get_thickness_for_segment(
                Point(cx_cm - width_cm / 2, cy_cm) if is_horizontal else Point(cx_cm, cy_cm - width_cm / 2),
                Point(cx_cm + width_cm / 2, cy_cm) if is_horizontal else Point(cx_cm, cy_cm + width_cm / 2),
            )
            if is_horizontal:
                half = width_cm * (1 + WALL_EXTENSION_FACTOR * 2) / 2
                start = Point(center.x - half, center.y)
                end = Point(center.x + half, center.y)
                angle = 0.0
            else:
                half = width_cm * (1 + WALL_EXTENSION_FACTOR * 2) / 2
                start = Point(center.x, center.y - half)
                end = Point(center.x, center.y + half)
                angle = math.pi / 2
            new_wall = Wall(start=start, end=end, thickness=thickness, is_structural=False)
            # Store the full detection bbox so _snap_doors_to_wall_gaps can use it
            half_perp = max(thickness, width_cm) / 2
            if is_horizontal:
                new_wall.detection_bbox = (cx_cm - half, cy_cm - half_perp,
                                           cx_cm + half, cy_cm + half_perp)
            else:
                new_wall.detection_bbox = (cx_cm - half_perp, cy_cm - half,
                                           cx_cm + half_perp, cy_cm + half)
            pid, depth = new_wall.wall_id, thickness

        fused_angle = math.radians(fused_door.wall_normal_deg) if hasattr(fused_door, 'wall_normal_deg') else angle
        hinge_px = fused_door.hinge if fused_door.hinge else (cx, cy)

        opening = Opening(
            opening_type=OpeningType.DOOR,
            center=center,
            width=ow, height=oh, depth=depth,
            angle=fused_angle, elevation=elev,
            parent_wall_id=pid,
            wall_thickness=wt, wall_distance=wd,
            model=model, icon=icon,
            hinge=hinge_px,
            swing_clockwise=fused_door.swing_clockwise,
            wall_normal_deg=fused_door.wall_normal_deg,
        )
        return opening, new_wall

    def _process_single(
        self,
        rect: Dict,
        opening_type: OpeningType,
        parent_finder: ParentWallFinder,
        thickness_normalizer: ThicknessNormalizer,
    ) -> Optional[Tuple[Opening, Optional[Wall]]]:
        x1, y1, x2, y2 = parse_rectangle(rect)
        x1 *= self.pixels_to_cm; y1 *= self.pixels_to_cm
        x2 *= self.pixels_to_cm; y2 *= self.pixels_to_cm
        bbox_w, bbox_h = abs(x2 - x1), abs(y2 - y1)
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        is_horizontal = bbox_w >= bbox_h

        # ──────────────────────────────────────────────────
        # FIX: opening width = dimension parallel to the wall
        # (the detected gap size), NOT a hard-coded default.
        # ──────────────────────────────────────────────────
        gap_width = bbox_w if is_horizontal else bbox_h

        if opening_type == OpeningType.WINDOW:
            ow = gap_width
            oh = DEFAULT_WINDOW_HEIGHT
            elev = DEFAULT_WINDOW_ELEVATION
            wt, wd = 0.7352941, 0.0
            model, icon = '3/window85x163/window85x163.obj', '2'
        elif opening_type == OpeningType.GAP:
            ow = gap_width
            oh = DEFAULT_WINDOW_HEIGHT
            elev = DEFAULT_WINDOW_ELEVATION
            wt, wd = 0.51724136, 0.06896552
            model, icon = '3/window85x163/window85x163.obj', '2'
        else:
            ow = gap_width
            oh = DEFAULT_DOOR_HEIGHT
            elev = 0.0
            wt, wd = 0.51724136, 0.06896552
            model, icon = '1/door/door.obj', '0'

        parent = parent_finder.find_parent(center, max(bbox_w, bbox_h), is_horizontal)
        new_wall: Optional[Wall] = None

        if parent:
            pid, depth, angle = parent.wall_id, parent.thickness, parent.angle
        else:
            thickness = thickness_normalizer.get_thickness_for_segment(
                Point(x1, (y1 + y2) / 2) if is_horizontal else Point((x1 + x2) / 2, y1),
                Point(x2, (y1 + y2) / 2) if is_horizontal else Point((x1 + x2) / 2, y2),
            )
            if is_horizontal:
                half = bbox_w * (1 + WALL_EXTENSION_FACTOR * 2) / 2
                start = Point(center.x - half, center.y)
                end = Point(center.x + half, center.y)
                angle = 0.0
            else:
                half = bbox_h * (1 + WALL_EXTENSION_FACTOR * 2) / 2
                start = Point(center.x, center.y - half)
                end = Point(center.x, center.y + half)
                angle = math.pi / 2
            new_wall = Wall(start=start, end=end, thickness=thickness, is_structural=False)
            # Store the full detection bbox (original rectangle in cm)
            new_wall.detection_bbox = (x1, y1, x2, y2)
            pid, depth = new_wall.wall_id, thickness

        opening = Opening(
            opening_type=opening_type,
            center=center,
            width=ow, height=oh, depth=depth,
            angle=angle, elevation=elev,
            parent_wall_id=pid,
            wall_thickness=wt, wall_distance=wd,
            model=model, icon=icon,
        )
        return opening, new_wall


# ============================================================
# RoomDetector  — REWRITTEN: raster flood-fill approach
# ============================================================
class RoomDetector:
    """
    Detect rooms by rasterising all wall geometry onto a binary mask,
    flood-filling the exterior from the image border, then finding
    connected components of remaining interior (non-wall, non-exterior)
    space.  Each connected component = one room polygon.

    After polygon extraction vertices are snapped to the nearest wall
    face so room boundaries exactly follow wall edges rather than
    approximating them.
    """

    def __init__(
        self,
        resolution: float = 0.5,        # finer raster → sharper corners
        min_room_area_cm2: float = 2000.0,
        wall_dilation_px: int = 2,
        contour_simplify_ratio: float = 0.003,   # less aggressive simplification
        snap_tolerance_cm: float = 18.0,          # max snap distance
    ) -> None:
        self.resolution = resolution
        self.min_room_area_cm2 = min_room_area_cm2
        self.wall_dilation_px = wall_dilation_px
        self.contour_simplify_ratio = contour_simplify_ratio
        self.snap_tolerance_cm = snap_tolerance_cm

    # ---- public API ----
    def detect(
        self,
        walls: List[Wall],
        enclosed_labels: Optional[np.ndarray] = None,
    ) -> List[Room]:
        if len(walls) < 3:
            return []

        # Pre-compute wall corner geometry once for snapping
        self._wall_geom = self._compute_wall_geom(walls)

        mask, origin = self._rasterize_walls(walls)
        interior = self._extract_interior(mask)
        rooms = self._components_to_rooms(interior, origin, enclosed_labels, mask)
        return rooms

    # ---- wall geometry for snap ----
    @staticmethod
    def _compute_wall_geom(walls: List[Wall]) -> List[Tuple]:
        """Return list of (4 corners as (x,y) tuples, 4 edge segments) per wall."""
        geom = []
        for w in walls:
            dx = w.end.x - w.start.x
            dy = w.end.y - w.start.y
            length = math.hypot(dx, dy)
            if length < 0.001:
                continue
            half_t = w.thickness / 2.0
            nx, ny = -dy / length * half_t, dx / length * half_t
            corners = [
                (w.start.x + nx, w.start.y + ny),
                (w.end.x   + nx, w.end.y   + ny),
                (w.end.x   - nx, w.end.y   - ny),
                (w.start.x - nx, w.start.y - ny),
            ]
            edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
            geom.append(edges)
        return geom

    def _snap_to_walls(self, x_cm: float, y_cm: float) -> Tuple[float, float]:
        """Project a point onto the nearest wall edge face within snap_tolerance_cm."""
        best_dist = self.snap_tolerance_cm
        best_x, best_y = x_cm, y_cm
        for edges in self._wall_geom:
            for (ax, ay), (bx, by) in edges:
                edx, edy = bx - ax, by - ay
                len_sq = edx * edx + edy * edy
                if len_sq < 1e-6:
                    continue
                t = ((x_cm - ax) * edx + (y_cm - ay) * edy) / len_sq
                t = max(0.0, min(1.0, t))
                px = ax + t * edx
                py = ay + t * edy
                d = math.hypot(x_cm - px, y_cm - py)
                if d < best_dist:
                    best_dist = d
                    best_x, best_y = px, py
        return best_x, best_y

    # ---- rasterise walls as filled rectangles ----
    def _rasterize_walls(
        self, walls: List[Wall]
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        padding = 60.0  # cm around the bounding box
        all_x, all_y = [], []
        for w in walls:
            half_t = w.thickness / 2.0
            dx = w.end.x - w.start.x
            dy = w.end.y - w.start.y
            length = math.hypot(dx, dy)
            if length < 0.001:
                continue
            nx, ny = -dy / length, dx / length
            for p in (w.start, w.end):
                all_x.extend([p.x + nx * half_t, p.x - nx * half_t])
                all_y.extend([p.y + ny * half_t, p.y - ny * half_t])

        if not all_x:
            return np.zeros((10, 10), np.uint8), (0.0, 0.0)

        min_x = min(all_x) - padding
        min_y = min(all_y) - padding
        max_x = max(all_x) + padding
        max_y = max(all_y) + padding

        res = self.resolution
        w_px = int(math.ceil((max_x - min_x) / res)) + 1
        h_px = int(math.ceil((max_y - min_y) / res)) + 1
        mask = np.zeros((h_px, w_px), dtype=np.uint8)

        for wall in walls:
            half_t = wall.thickness / 2.0
            dx = wall.end.x - wall.start.x
            dy = wall.end.y - wall.start.y
            length = math.hypot(dx, dy)
            if length < 0.001:
                continue
            nx, ny = -dy / length, dx / length
            corners_cm = [
                (wall.start.x + nx * half_t, wall.start.y + ny * half_t),
                (wall.end.x   + nx * half_t, wall.end.y   + ny * half_t),
                (wall.end.x   - nx * half_t, wall.end.y   - ny * half_t),
                (wall.start.x - nx * half_t, wall.start.y - ny * half_t),
            ]
            pts = np.array(
                [
                    [int((x - min_x) / res), int((y - min_y) / res)]
                    for x, y in corners_cm
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(mask, pts, 255)

        # Dilate slightly to close sub-pixel gaps between abutting walls
        if self.wall_dilation_px > 0:
            kernel = np.ones(
                (2 * self.wall_dilation_px + 1, 2 * self.wall_dilation_px + 1),
                dtype=np.uint8,
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, (min_x, min_y)

    # ---- flood-fill exterior, then interior = rooms ----
    def _extract_interior(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        flood = mask.copy()
        # cv2.floodFill needs a mask that is h+2 × w+2
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Fill from (0, 0) — guaranteed to be exterior because of padding
        cv2.floodFill(flood, ff_mask, (0, 0), 128)
        # flood now: 255 = wall, 128 = exterior, 0 = interior
        interior = (flood == 0).astype(np.uint8) * 255
        return interior

    # ---- connected components → Room objects ----
    def _components_to_rooms(
        self,
        interior: np.ndarray,
        origin: Tuple[float, float],
        enclosed_labels: Optional[np.ndarray] = None,
        wall_mask: Optional[np.ndarray] = None,
    ) -> List[Room]:
        min_x, min_y = origin
        res = self.resolution
        num_labels, labels = cv2.connectedComponents(interior)

        # Build a pixel-scale enclosed-region validity mask from U-Net labels.
        # enclosed_labels is in original image pixels; interior is in cm-raster
        # pixels. We'll check overlap by projecting room component bboxes back
        # to image space via the cm-to-pixel scale (self.pixels_to_cm stored on
        # call site). Since we don't have pixels_to_cm here, we use a simpler
        # approach: scale the enclosed_labels mask to match the raster size.
        enc_resized: Optional[np.ndarray] = None
        if enclosed_labels is not None:
            # Any pixel with label > 0 is inside an enclosed region.
            enc_bool = (enclosed_labels > 0).astype(np.uint8) * 255
            h_raster, w_raster = interior.shape[:2]
            enc_resized = cv2.resize(
                enc_bool, (w_raster, h_raster), interpolation=cv2.INTER_NEAREST
            )

        rooms: List[Room] = []
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            area_px = int(np.sum(component > 0))
            area_cm2 = area_px * (res * res)
            if area_cm2 < self.min_room_area_cm2:
                continue

            # U-Net filter: at least 20% of the room's pixels must overlap
            # with an enclosed region detected by U-Net.
            if enc_resized is not None:
                overlap_px = int(np.sum((component > 0) & (enc_resized > 0)))
                if overlap_px / max(area_px, 1) < 0.20:
                    continue

            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)

            # Simplify to a reasonable polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.contour_simplify_ratio * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue

            points: List[Point] = []
            for pt in approx:
                x_cm = pt[0][0] * res + min_x
                y_cm = pt[0][1] * res + min_y
                # Snap vertex to nearest wall face for precise room boundaries
                x_cm, y_cm = self._snap_to_walls(x_cm, y_cm)
                points.append(Point(x_cm, y_cm))

            rooms.append(Room(
                points=points,
                name=f'Room {len(rooms) + 1}',
            ))

        return rooms


# ============================================================
# RoomOverlapRemover  (kept for compatibility but much simpler
# now — the flood-fill detector rarely produces overlaps)
# ============================================================
class RoomOverlapRemover:
    def __init__(
        self,
        min_area_threshold: float = 50.0,
        overlap_threshold: float = 0.50,
        **_kwargs,
    ) -> None:
        self.min_area_threshold = min_area_threshold
        self.overlap_threshold = overlap_threshold

    @staticmethod
    def _polygon_area(points: List[Point]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i].x * points[j].y - points[j].x * points[i].y
        return abs(area) / 2.0

    def remove_overlaps(self, rooms: List[Room]) -> List[Room]:
        # With flood-fill rooms overlap should not happen; keep as safety net
        if not rooms:
            return []
        return [
            r for r in rooms
            if self._polygon_area(r.points) >= self.min_area_threshold
        ]


# ============================================================
# SH3DXMLGenerator
# ============================================================
class SH3DXMLGenerator:
    def __init__(
        self,
        wall_height: float = DEFAULT_WALL_HEIGHT,
        scale_factor: float = SCALE_FACTOR,
    ) -> None:
        self.wall_height = wall_height
        self.scale_factor = scale_factor

    def generate(
        self,
        walls: List[Wall],
        openings: List[Opening],
        rooms: List[Room],
        name: str = 'floor_plan',
    ) -> str:
        sf = self.scale_factor
        root = ET.Element('home')
        root.set('version', '7400')
        root.set('name', name)
        root.set('camera', 'topCamera')
        root.set('wallHeight', f'{self.wall_height / sf:.4f}')
        prop = ET.SubElement(root, 'property')
        prop.set('name', 'com.eteks.sweethome3d.SweetHome3D.PlanScale')
        prop.set('value', '0.02')
        env = ET.SubElement(root, 'environment')
        env.set('groundColor', '00A8A8A8')
        env.set('skyColor', '00CCE4FC')
        env.set('lightColor', '00D0D0D0')
        cam = ET.SubElement(root, 'camera')
        cam.set('attribute', 'topCamera')
        cam.set('lens', 'PINHOLE')
        cam.set('x', f'{500 / sf:.4f}')
        cam.set('y', f'{500 / sf:.4f}')
        cam.set('z', f'{1500 / sf:.4f}')
        cam.set('yaw', '0.0')
        cam.set('pitch', '1.4')
        cam.set('fieldOfView', '1.0')
        for room in rooms:
            self._add_room(root, room)
        for opening in openings:
            self._add_opening(root, opening)
        for wall in walls:
            self._add_wall(root, wall)
        return ET.tostring(root, encoding='unicode')

    def _add_wall(self, root: ET.Element, w: Wall) -> None:
        sf = self.scale_factor
        e = ET.SubElement(root, 'wall')
        e.set('id', w.wall_id)
        e.set('xStart', f'{w.start.x / sf:.4f}')
        e.set('yStart', f'{w.start.y / sf:.4f}')
        e.set('xEnd', f'{w.end.x / sf:.4f}')
        e.set('yEnd', f'{w.end.y / sf:.4f}')
        e.set('height', f'{w.height / sf:.4f}')
        e.set('thickness', f'{w.thickness / sf:.4f}')
        e.set('pattern', w.pattern)
        if w.wall_at_start:
            e.set('wallAtStart', w.wall_at_start)
        if w.wall_at_end:
            e.set('wallAtEnd', w.wall_at_end)

    def _add_opening(self, root: ET.Element, o: Opening) -> None:
        sf = self.scale_factor
        e = ET.SubElement(root, 'doorOrWindow')
        e.set('id', o.opening_id)
        if o.opening_type in (OpeningType.DOOR, OpeningType.GAP):
            e.set('catalogId', 'eteks-door')
            e.set('name', 'Door')
        else:
            e.set('catalogId', 'eteks-window85x163')
            e.set('name', 'Window')
        e.set('creator', 'eTeks')
        e.set('model', o.model)
        e.set('icon', o.icon)
        e.set('x', f'{o.center.x / sf:.4f}')
        e.set('y', f'{o.center.y / sf:.4f}')
        if o.elevation > 0:
            e.set('elevation', f'{o.elevation / sf:.4f}')
        e.set('angle', str(o.angle))
        e.set('width', f'{o.width / sf:.4f}')
        e.set('depth', f'{o.depth / sf:.4f}')
        e.set('height', f'{o.height / sf:.4f}')
        e.set('movable', 'false')
        e.set('wallThickness', str(o.wall_thickness))
        e.set('wallDistance', str(o.wall_distance))
        e.set('cutOutShape', 'M0,0 v1 h1 v-1 z')
        e.set('wallCutOutOnBothSides', 'true')

        # ──────────────────────────────────────────────────────────
        # FIX: NO <sash> element → SweetHome3D will not draw a
        # swing arc for doors or windows.
        # ──────────────────────────────────────────────────────────

    def _add_room(self, root: ET.Element, room: Room) -> None:
        sf = self.scale_factor
        e = ET.SubElement(root, 'room')
        e.set('id', room.room_id)
        e.set('name', room.name)
        e.set('areaVisible', 'true')
        e.set('floorColor', '-3355444')
        for p in room.points:
            pt = ET.SubElement(e, 'point')
            pt.set('x', f'{p.x / sf:.4f}')
            pt.set('y', f'{p.y / sf:.4f}')


# ============================================================
# SweetHome3DExporter — фасад всего пайплайна экспорта
# ============================================================
class SweetHome3DExporter:
    def __init__(
        self,
        min_wall_thickness: float = 2.0,
        pixels_to_cm: float = 2.54,
        wall_height_cm: float = DEFAULT_WALL_HEIGHT,
        snap_tolerance: float = SNAP_TOLERANCE,
        wall_overlap: float = WALL_OVERLAP,
        scale_factor: float = SCALE_FACTOR,
    ) -> None:
        self.pixels_to_cm = pixels_to_cm
        self.wall_height = wall_height_cm
        self.snap_tolerance = snap_tolerance
        self.wall_overlap = wall_overlap
        self.wall_converter = StructuralWallConverter(pixels_to_cm, min_wall_thickness)
        self.opening_processor = OpeningProcessor(pixels_to_cm)
        self.xml_generator = SH3DXMLGenerator(wall_height_cm, scale_factor)

    def update_scale(self, pixels_to_cm: float) -> None:
        self.pixels_to_cm = pixels_to_cm
        self.wall_converter.pixels_to_cm = pixels_to_cm
        self.opening_processor.pixels_to_cm = pixels_to_cm

    def export_to_sh3d(
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
        enclosed_labels: Optional[np.ndarray] = None,
    ) -> List[Room]:
        structural_walls = self.wall_converter.convert(wall_rectangles)

        opening_walls, openings = self.opening_processor.process(
            structural_walls,
            door_rects=door_rectangles,
            window_rects=window_rectangles,
            gap_rects=gap_rectangles,
            fused_doors=fused_doors,
        )

        self._snap_doors_to_wall_gaps(
            opening_walls, openings, structural_walls, original_image
        )
        all_walls = structural_walls + opening_walls
        if original_image is not None and debug_image_path is not None:
            door_dbg = self._draw_door_placement_debug(
                original_image, structural_walls,
            )
            door_dbg_path = debug_image_path.replace(
                '_debug.png', '_door_debug.png',
            )
            cv2.imwrite(door_dbg_path, door_dbg)
        self._process_wall_connections(all_walls, structural_walls, opening_walls)
        self._update_openings_from_walls(openings, all_walls)

        # ──────────────────────────────────────────────────────
        # FIX: use the new flood-fill RoomDetector
        # ──────────────────────────────────────────────────────
        rooms = RoomDetector(
            resolution=0.5,
            min_room_area_cm2=2000.0,
            wall_dilation_px=3,
            contour_simplify_ratio=0.003,
            snap_tolerance_cm=18.0,
        ).detect(all_walls, enclosed_labels=enclosed_labels)

        rooms = RoomOverlapRemover(
            min_area_threshold=50.0, overlap_threshold=0.50
        ).remove_overlaps(rooms)

        xml = self.xml_generator.generate(
            walls=all_walls, openings=openings, rooms=rooms, name=output_path
        )
        self._write_sh3d_file(output_path, xml)

        if original_image is not None and debug_image_path is not None:
            viz = DebugVisualizer(self.wall_converter.pixels_to_cm)
            debug_img = viz.create_debug_image(
                original_image, wall_rectangles, all_walls,
                rooms, openings, ocr_labels, wall_mask,
            )
            cv2.imwrite(debug_image_path, debug_img)
        return rooms

    # ----- bbox-overlap gap bridging for door placement -----
    def _snap_doors_to_wall_gaps(
        self,
        opening_walls: List[Wall],
        openings: List[Opening],
        structural_walls: List[Wall],
        image: Optional[np.ndarray],
    ) -> None:
        """Place each opening wall in the gap between two structural walls.

        Algorithm per door:
        1. Compute door bbox (including thickness), extend by 5%.
        2. Find structural walls whose bbox overlaps the extended door bbox.
        3. Classify overlapping walls by side (L/R for horizontal, T/B for vertical).
        4. For each opposite pair compute the gap between inner edges.
        5. Validate gap with a colour check (gap area should be empty space).
        6. Pick the smallest valid gap and set opening wall endpoints to bridge it.
        7. Remove doors with no valid wall pair.
        """
        gray: Optional[np.ndarray] = None
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        s = self.pixels_to_cm  # cm per pixel

        def _wall_bbox(w: Wall) -> Tuple[float, float, float, float]:
            if w.is_horizontal:
                return (w.get_min_x(),
                        w.get_centerline_y() - w.thickness / 2,
                        w.get_max_x(),
                        w.get_centerline_y() + w.thickness / 2)
            else:
                return (w.get_centerline_x() - w.thickness / 2,
                        w.get_min_y(),
                        w.get_centerline_x() + w.thickness / 2,
                        w.get_max_y())

        def _overlap(a: Tuple[float, float, float, float],
                     b: Tuple[float, float, float, float]) -> bool:
            return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

        def _gap_is_clear(x1_cm: float, y1_cm: float,
                          x2_cm: float, y2_cm: float) -> bool:
            """Sample pixels along the line from (x1,y1) to (x2,y2) in cm.
            Return True if fewer than GAP_DARK_RATIO_LIMIT of samples are dark."""
            if gray is None:
                return True
            img_h, img_w = gray.shape[:2]
            n_samples = max(int(max(abs(x2_cm - x1_cm), abs(y2_cm - y1_cm)) / s), 5)
            dark = 0
            for i in range(n_samples + 1):
                t = i / max(n_samples, 1)
                px = int((x1_cm + (x2_cm - x1_cm) * t) / s)
                py = int((y1_cm + (y2_cm - y1_cm) * t) / s)
                if 0 <= px < img_w and 0 <= py < img_h:
                    if gray[py, px] < GAP_DARK_THRESHOLD:
                        dark += 1
            return dark / max(n_samples + 1, 1) < GAP_DARK_RATIO_LIMIT

        def _inner_edge_h(sw: Wall, side: str) -> float:
            """Inner edge X for a wall on the LEFT or RIGHT of a horizontal door."""
            if side == 'L':
                return (sw.get_centerline_x() + sw.thickness / 2) if not sw.is_horizontal else sw.get_max_x()
            else:
                return (sw.get_centerline_x() - sw.thickness / 2) if not sw.is_horizontal else sw.get_min_x()

        def _inner_edge_v(sw: Wall, side: str) -> float:
            """Inner edge Y for a wall on the TOP or BOTTOM of a vertical door."""
            if side == 'T':
                return (sw.get_centerline_y() + sw.thickness / 2) if sw.is_horizontal else sw.get_max_y()
            else:
                return (sw.get_centerline_y() - sw.thickness / 2) if sw.is_horizontal else sw.get_min_y()

        sw_bboxes = [(sw, _wall_bbox(sw)) for sw in structural_walls]
        self._door_placement_debug: List[dict] = []
        to_remove: set = set()

        # Proximity limit: 50 image-pixels converted to cm
        PROXIMITY_CM = 50.0 * s

        for ow in opening_walls:
            if ow.length < 0.1:
                continue
            _orig = (ow.start.x, ow.start.y, ow.end.x, ow.end.y)

            # Use the full original detection bbox for wall search (as seen in
            # the summary image).  Fall back to the narrow wall bbox only when no
            # detection bbox was stored (e.g. when a parent wall was found).
            if ow.detection_bbox is not None:
                door_bb = ow.detection_bbox
            else:
                door_bb = _wall_bbox(ow)
            dw = door_bb[2] - door_bb[0]
            dh = door_bb[3] - door_bb[1]
            ext_bb = (door_bb[0] - dw * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[1] - dh * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[2] + dw * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[3] + dh * DOOR_BBOX_EXTEND_RATIO)
            # Detection center — used to place the opening wall perpendicularly
            det_cx = (door_bb[0] + door_bb[2]) / 2
            det_cy = (door_bb[1] + door_bb[3]) / 2

            best_gap = float('inf')
            best_pair = None  # (edge_a, edge_b, sw_a, sw_b)

            if ow.is_horizontal:
                door_cx = (door_bb[0] + door_bb[2]) / 2
                left_seen: set = set()
                right_seen: set = set()
                left_candidates: List[Tuple[Wall, Tuple, float]] = []
                right_candidates: List[Tuple[Wall, Tuple, float]] = []
                for sw, swbb in sw_bboxes:
                    y_overlap = swbb[3] > door_bb[1] and swbb[1] < door_bb[3]
                    overlaps_ext = _overlap(ext_bb, swbb)
                    sw_cx = (swbb[0] + swbb[2]) / 2

                    le = _inner_edge_h(sw, 'L')   # right-facing inner edge
                    re = _inner_edge_h(sw, 'R')   # left-facing inner edge

                    # LEFT candidate: passes overlap test OR proximity test
                    if sw.wall_id not in left_seen:
                        prox_l = y_overlap and 0 <= door_bb[0] - le <= PROXIMITY_CM
                        ovlp_l = overlaps_ext and sw_cx < door_cx
                        if prox_l or ovlp_l:
                            dist_l = max(0.0, door_bb[0] - le)
                            left_candidates.append((sw, swbb, dist_l))
                            left_seen.add(sw.wall_id)

                    # RIGHT candidate: passes overlap test OR proximity test
                    if sw.wall_id not in right_seen:
                        prox_r = y_overlap and 0 <= re - door_bb[2] <= PROXIMITY_CM
                        ovlp_r = overlaps_ext and sw_cx >= door_cx
                        if prox_r or ovlp_r:
                            dist_r = max(0.0, re - door_bb[2])
                            right_candidates.append((sw, swbb, dist_r))
                            right_seen.add(sw.wall_id)

                # Keep closest wall on each side
                left_candidates.sort(key=lambda t: t[2])
                right_candidates.sort(key=lambda t: t[2])
                left_walls  = [(sw, swbb) for sw, swbb, _ in left_candidates[:1]]
                right_walls = [(sw, swbb) for sw, swbb, _ in right_candidates[:1]]

                # Find best pair
                for lsw, _ in left_walls:
                    le = _inner_edge_h(lsw, 'L')
                    for rsw, _ in right_walls:
                        re = _inner_edge_h(rsw, 'R')
                        gap = re - le
                        if gap <= MIN_GAP_CM or gap >= best_gap:
                            continue
                        if _gap_is_clear(le, det_cy, re, det_cy):
                            best_gap = gap
                            best_pair = (le, re, lsw, rsw)

                if best_pair is not None:
                    ow.start.x, ow.end.x = best_pair[0], best_pair[1]
                    # Snap perpendicular coordinate to detection bbox centre
                    ow.start.y = ow.end.y = det_cy
                    for o in openings:
                        if o.parent_wall_id == ow.wall_id:
                            o.width = best_gap
                            o.center = Point(det_cx, det_cy)
                else:
                    to_remove.add(ow.wall_id)

                self._door_placement_debug.append({
                    'h': True,
                    'orig_bbox': door_bb,
                    'ext_bbox': ext_bb,
                    'overlaps': [('L', sw, swbb) for sw, swbb in left_walls]
                              + [('R', sw, swbb) for sw, swbb in right_walls],
                    'selected': best_pair,
                    'final': (ow.start.x, ow.start.y, ow.end.x, ow.end.y),
                    'removed': best_pair is None,
                })
            else:
                # Vertical door — search TOP and BOTTOM narrow sides
                door_cy = (door_bb[1] + door_bb[3]) / 2
                top_seen: set = set()
                bot_seen: set = set()
                top_candidates: List[Tuple[Wall, Tuple, float]] = []
                bot_candidates: List[Tuple[Wall, Tuple, float]] = []
                for sw, swbb in sw_bboxes:
                    x_overlap = swbb[2] > door_bb[0] and swbb[0] < door_bb[2]
                    overlaps_ext = _overlap(ext_bb, swbb)
                    sw_cy = (swbb[1] + swbb[3]) / 2

                    te = _inner_edge_v(sw, 'T')    # bottom-facing inner edge
                    be = _inner_edge_v(sw, 'B')    # top-facing inner edge

                    # TOP candidate
                    if sw.wall_id not in top_seen:
                        prox_t = x_overlap and 0 <= door_bb[1] - te <= PROXIMITY_CM
                        ovlp_t = overlaps_ext and sw_cy < door_cy
                        if prox_t or ovlp_t:
                            dist_t = max(0.0, door_bb[1] - te)
                            top_candidates.append((sw, swbb, dist_t))
                            top_seen.add(sw.wall_id)

                    # BOTTOM candidate
                    if sw.wall_id not in bot_seen:
                        prox_b = x_overlap and 0 <= be - door_bb[3] <= PROXIMITY_CM
                        ovlp_b = overlaps_ext and sw_cy >= door_cy
                        if prox_b or ovlp_b:
                            dist_b = max(0.0, be - door_bb[3])
                            bot_candidates.append((sw, swbb, dist_b))
                            bot_seen.add(sw.wall_id)

                top_candidates.sort(key=lambda t: t[2])
                bot_candidates.sort(key=lambda t: t[2])
                top_walls = [(sw, swbb) for sw, swbb, _ in top_candidates[:1]]
                bot_walls = [(sw, swbb) for sw, swbb, _ in bot_candidates[:1]]

                for tsw, _ in top_walls:
                    te = _inner_edge_v(tsw, 'T')
                    for bsw, _ in bot_walls:
                        be = _inner_edge_v(bsw, 'B')
                        gap = be - te
                        if gap <= MIN_GAP_CM or gap >= best_gap:
                            continue
                        if _gap_is_clear(det_cx, te, det_cx, be):
                            best_gap = gap
                            best_pair = (te, be, tsw, bsw)

                if best_pair is not None:
                    if ow.start.y < ow.end.y:
                        ow.start.y, ow.end.y = best_pair[0], best_pair[1]
                    else:
                        ow.start.y, ow.end.y = best_pair[1], best_pair[0]
                    # Snap perpendicular coordinate to detection bbox centre
                    ow.start.x = ow.end.x = det_cx
                    for o in openings:
                        if o.parent_wall_id == ow.wall_id:
                            o.width = best_gap
                            o.center = Point(det_cx, det_cy)
                else:
                    to_remove.add(ow.wall_id)

                self._door_placement_debug.append({
                    'h': False,
                    'orig_bbox': door_bb,
                    'ext_bbox': ext_bb,
                    'overlaps': [('T', sw, swbb) for sw, swbb in top_walls]
                              + [('B', sw, swbb) for sw, swbb in bot_walls],
                    'selected': best_pair,
                    'final': (ow.start.x, ow.start.y, ow.end.x, ow.end.y),
                    'removed': best_pair is None,
                })

        if to_remove:
            opening_walls[:] = [w for w in opening_walls if w.wall_id not in to_remove]
            openings[:] = [o for o in openings if o.parent_wall_id not in to_remove]

    def _draw_door_placement_debug(
        self,
        original_image: 'np.ndarray',
        structural_walls: List[Wall],
    ) -> 'np.ndarray':
        """Draw debug overlay showing bbox-overlap door placement decisions."""
        canvas = original_image.copy()
        s = self.pixels_to_cm

        def c(v: float) -> int:
            return int(round(v / s))

        # Structural wall bounding boxes — thin dark-blue
        for sw in structural_walls:
            if sw.is_horizontal:
                p1 = (c(sw.get_min_x()), c(sw.get_centerline_y() - sw.thickness / 2))
                p2 = (c(sw.get_max_x()), c(sw.get_centerline_y() + sw.thickness / 2))
            else:
                p1 = (c(sw.get_centerline_x() - sw.thickness / 2), c(sw.get_min_y()))
                p2 = (c(sw.get_centerline_x() + sw.thickness / 2), c(sw.get_max_y()))
            cv2.rectangle(canvas, p1, p2, (180, 130, 70), 1)

        for entry in self._door_placement_debug:
            is_h = entry['h']
            f = entry['final']
            removed = entry.get('removed', False)

            # Original door bbox — thin red rectangle
            obb = entry['orig_bbox']
            cv2.rectangle(canvas, (c(obb[0]), c(obb[1])), (c(obb[2]), c(obb[3])),
                          (0, 0, 255), 1)

            # Extended bbox — dashed red rectangle (drawn as dotted lines)
            ebb = entry['ext_bbox']
            ep1, ep2 = (c(ebb[0]), c(ebb[1])), (c(ebb[2]), c(ebb[3]))
            for i in range(0, abs(ep2[0] - ep1[0]), 6):
                cv2.line(canvas, (ep1[0] + i, ep1[1]), (min(ep1[0] + i + 3, ep2[0]), ep1[1]), (0, 0, 200), 1)
                cv2.line(canvas, (ep1[0] + i, ep2[1]), (min(ep1[0] + i + 3, ep2[0]), ep2[1]), (0, 0, 200), 1)
            for i in range(0, abs(ep2[1] - ep1[1]), 6):
                cv2.line(canvas, (ep1[0], ep1[1] + i), (ep1[0], min(ep1[1] + i + 3, ep2[1])), (0, 0, 200), 1)
                cv2.line(canvas, (ep2[0], ep1[1] + i), (ep2[0], min(ep1[1] + i + 3, ep2[1])), (0, 0, 200), 1)

            # Overlapping wall bboxes — orange with side labels
            for side, sw, swbb in entry['overlaps']:
                sp1 = (c(swbb[0]), c(swbb[1]))
                sp2 = (c(swbb[2]), c(swbb[3]))
                cv2.rectangle(canvas, sp1, sp2, (0, 165, 255), 2)
                lx = sp1[0] if side in ('L', 'T') else sp2[0] - 10
                ly = sp1[1] - 4
                cv2.putText(canvas, side, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

            # Selected pair inner edges + gap — green
            sel = entry.get('selected')
            if sel is not None:
                if is_h:
                    fy = c(f[1])
                    cv2.line(canvas, (c(sel[0]), fy - 20), (c(sel[0]), fy + 20),
                             (0, 255, 0), 2)
                    cv2.line(canvas, (c(sel[1]), fy - 20), (c(sel[1]), fy + 20),
                             (0, 255, 0), 2)
                    # Gap line — yellow
                    cv2.line(canvas, (c(sel[0]), fy), (c(sel[1]), fy),
                             (0, 255, 255), 1)
                    gap = abs(sel[1] - sel[0])
                    mid = c((sel[0] + sel[1]) / 2)
                    cv2.putText(canvas, f'{gap:.0f}cm', (mid - 15, fy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    fx = c(f[0])
                    cv2.line(canvas, (fx - 20, c(sel[0])), (fx + 20, c(sel[0])),
                             (0, 255, 0), 2)
                    cv2.line(canvas, (fx - 20, c(sel[1])), (fx + 20, c(sel[1])),
                             (0, 255, 0), 2)
                    cv2.line(canvas, (fx, c(sel[0])), (fx, c(sel[1])),
                             (0, 255, 255), 1)
                    gap = abs(sel[1] - sel[0])
                    mid = c((sel[0] + sel[1]) / 2)
                    cv2.putText(canvas, f'{gap:.0f}cm', (fx + 25, mid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            if removed:
                # Red X through original bbox
                cv2.line(canvas, (c(obb[0]), c(obb[1])), (c(obb[2]), c(obb[3])),
                         (0, 0, 255), 2)
                cv2.line(canvas, (c(obb[2]), c(obb[1])), (c(obb[0]), c(obb[3])),
                         (0, 0, 255), 2)
            else:
                # Final door wall — thick green
                cv2.line(canvas, (c(f[0]), c(f[1])), (c(f[2]), c(f[3])),
                         (0, 255, 0), 3)

        return canvas

    def _process_wall_connections(
        self,
        all_walls: List[Wall],
        structural_walls: List[Wall],
        opening_walls: List[Wall],
    ) -> None:
        self._align_walls_to_shared_axes(structural_walls, AXIS_SNAP_TOLERANCE)
        self._extend_walls_along_axis(
            structural_walls, T_JUNCTION_SNAP_TOLERANCE, iterations=2
        )
        self._align_opening_walls_to_structural(
            opening_walls, structural_walls, OPENING_AXIS_SNAP_TOLERANCE
        )
        self._extend_walls_along_axis_against(
            opening_walls, all_walls, OPENING_EXTENSION_TOLERANCE, iterations=2
        )
        self._snap_endpoints_to_corners(all_walls, snap_distance=5.0)
        self._connect_walls(all_walls)

    def _align_walls_to_shared_axes(
        self, walls: List[Wall], tolerance: float
    ) -> None:
        h_walls = [w for w in walls if w.is_horizontal]
        v_walls = [w for w in walls if not w.is_horizontal]
        for w in h_walls:
            avg = (w.start.y + w.end.y) / 2
            w.start.y = w.end.y = avg
        for w in v_walls:
            avg = (w.start.x + w.end.x) / 2
            w.start.x = w.end.x = avg
        for wall_list, attr, cfn in (
            (h_walls, 'y', lambda w: w.get_centerline_y()),
            (v_walls, 'x', lambda w: w.get_centerline_x()),
        ):
            groups: List[Dict] = []
            for w in wall_list:
                val = cfn(w)
                placed = False
                for g in groups:
                    if abs(val - g['val']) < tolerance:
                        g['walls'].append(w)
                        total = sum(ww.length for ww in g['walls'])
                        if total > 0:
                            g['val'] = (
                                sum(cfn(ww) * ww.length for ww in g['walls'])
                                / total
                            )
                        placed = True
                        break
                if not placed:
                    groups.append({'walls': [w], 'val': val})
            for g in groups:
                v = g['val']
                for w in g['walls']:
                    if attr == 'y':
                        w.start.y = w.end.y = v
                    else:
                        w.start.x = w.end.x = v

    def _align_opening_walls_to_structural(
        self,
        opening_walls: List[Wall],
        structural_walls: List[Wall],
        tolerance: float,
    ) -> None:
        h_struct = [
            (w.get_centerline_y(), w.length)
            for w in structural_walls if w.is_horizontal
        ]
        v_struct = [
            (w.get_centerline_x(), w.length)
            for w in structural_walls if not w.is_horizontal
        ]
        for ow in opening_walls:
            if ow.is_horizontal:
                avg = (ow.start.y + ow.end.y) / 2
                best_y, best_d = None, tolerance
                for sy, _ in h_struct:
                    d = abs(avg - sy)
                    if d < best_d:
                        best_d, best_y = d, sy
                ow.start.y = ow.end.y = best_y if best_y is not None else avg
            else:
                avg = (ow.start.x + ow.end.x) / 2
                best_x, best_d = None, tolerance
                for sx, _ in v_struct:
                    d = abs(avg - sx)
                    if d < best_d:
                        best_d, best_x = d, sx
                ow.start.x = ow.end.x = best_x if best_x is not None else avg

    def _extend_walls_along_axis(
        self, walls: List[Wall], max_gap: float, iterations: int = 2
    ) -> None:
        for _ in range(iterations):
            for wall in walls:
                if wall.length < 0.1:
                    continue
                self._try_extend_endpoint(wall, 'start', walls, max_gap)
                self._try_extend_endpoint(wall, 'end', walls, max_gap)

    def _extend_walls_along_axis_against(
        self,
        walls_to_extend: List[Wall],
        target_walls: List[Wall],
        max_gap: float,
        iterations: int = 2,
    ) -> None:
        for _ in range(iterations):
            for wall in walls_to_extend:
                if wall.length < 0.1:
                    continue
                self._try_extend_endpoint(wall, 'start', target_walls, max_gap)
                self._try_extend_endpoint(wall, 'end', target_walls, max_gap)

    def _try_extend_endpoint(
        self,
        wall: Wall,
        endpoint: str,
        all_walls: List[Wall],
        max_gap: float,
    ) -> None:
        point = wall.start if endpoint == 'start' else wall.end
        best_ext, best_gap = None, max_gap
        for other in all_walls:
            if other.wall_id == wall.wall_id:
                continue
            info = self._calc_gap(wall, point, other, wall.is_horizontal)
            if info is None:
                continue
            gap, target, _ = info
            if 0 <= gap < best_gap:
                best_gap, best_ext = gap, target
        if best_ext is not None:
            if wall.is_horizontal:
                if endpoint == 'start':
                    wall.start.x = best_ext
                else:
                    wall.end.x = best_ext
            else:
                if endpoint == 'start':
                    wall.start.y = best_ext
                else:
                    wall.end.y = best_ext

    def _calc_gap(
        self,
        wall: Wall,
        point: Point,
        other: Wall,
        is_horizontal: bool,
    ) -> Optional[Tuple[float, float, bool]]:
        half = (wall.thickness + other.thickness) / 2
        if is_horizontal:
            wy = point.y
            if other.is_horizontal:
                if abs(wy - other.get_centerline_y()) > half:
                    return None
                if point.x < other.get_min_x():
                    return (
                        other.get_min_x() - point.x - half,
                        other.get_min_x() - self.wall_overlap,
                        False,
                    )
                if point.x > other.get_max_x():
                    return (
                        point.x - other.get_max_x() - half,
                        other.get_max_x() + self.wall_overlap,
                        False,
                    )
            else:
                ox = other.get_centerline_x()
                if not (other.get_min_y() - half <= wy <= other.get_max_y() + half):
                    return None
                if point.x < ox:
                    edge = ox - other.thickness / 2
                else:
                    edge = ox + other.thickness / 2
                return (max(0, abs(point.x - edge)), edge, True)
        else:
            wx = point.x
            if not other.is_horizontal:
                if abs(wx - other.get_centerline_x()) > half:
                    return None
                if point.y < other.get_min_y():
                    return (
                        other.get_min_y() - point.y - half,
                        other.get_min_y() - self.wall_overlap,
                        False,
                    )
                if point.y > other.get_max_y():
                    return (
                        point.y - other.get_max_y() - half,
                        other.get_max_y() + self.wall_overlap,
                        False,
                    )
            else:
                oy = other.get_centerline_y()
                if not (other.get_min_x() - half <= wx <= other.get_max_x() + half):
                    return None
                if point.y < oy:
                    edge = oy - other.thickness / 2
                else:
                    edge = oy + other.thickness / 2
                return (max(0, abs(point.y - edge)), edge, True)
        return None

    def _snap_endpoints_to_corners(
        self, walls: List[Wall], snap_distance: float = 5.0
    ) -> None:
        endpoints = [
            (w, e, w.start if e == 'start' else w.end)
            for w in walls for e in ('start', 'end')
        ]
        for i, (w1, e1, p1) in enumerate(endpoints):
            for w2, e2, p2 in endpoints[i + 1:]:
                if w1.wall_id == w2.wall_id or p1.distance_to(p2) > snap_distance:
                    continue
                if w1.is_structural and not w2.is_structural:
                    target = p1
                elif w2.is_structural and not w1.is_structural:
                    target = p2
                elif w1.length > w2.length:
                    target = p1
                else:
                    target = p2
                for w, e in ((w1, e1), (w2, e2)):
                    pt = w.start if e == 'start' else w.end
                    if w.is_horizontal:
                        pt.x = target.x
                    else:
                        pt.y = target.y

    def _connect_walls(self, walls: List[Wall]) -> None:
        eps = self.wall_overlap + 2.0
        for i, w1 in enumerate(walls):
            for w2 in walls[i + 1:]:
                if w1.end.distance_to(w2.start) < eps:
                    w1.wall_at_end = w2.wall_id
                    w2.wall_at_start = w1.wall_id
                elif w1.start.distance_to(w2.end) < eps:
                    w1.wall_at_start = w2.wall_id
                    w2.wall_at_end = w1.wall_id
                elif w1.end.distance_to(w2.end) < eps:
                    w1.wall_at_end = w2.wall_id
                    w2.wall_at_end = w1.wall_id
                elif w1.start.distance_to(w2.start) < eps:
                    w1.wall_at_start = w2.wall_id
                    w2.wall_at_start = w1.wall_id

    # ──────────────────────────────────────────────────────────────
    # FIX: _update_openings_from_walls
    #   OLD behaviour:
    #     o.width = parent.length * OPENING_TO_WALL_RATIO   ← BUG
    #     o.center = parent.midpoint  (for non-structural)  ← BUG
    #   NEW behaviour:
    #     o.width is KEPT as-is (set from detected rectangle
    #     in OpeningProcessor).
    #     o.center is PROJECTED onto the parent wall
    #     centerline — never replaced with the wall midpoint.
    # ──────────────────────────────────────────────────────────────
    def _update_openings_from_walls(
        self, openings: List[Opening], walls: List[Wall]
    ) -> None:
        wall_by_id = {w.wall_id: w for w in walls}
        for o in openings:
            parent = wall_by_id.get(o.parent_wall_id)
            if not parent:
                continue
            # Align angle to parent wall (unless fused door set a precise angle)
            if o.hinge is None:
                o.angle = parent.angle
            # Project opening centre onto the parent wall centerline
            # but do NOT replace it with the wall midpoint and do NOT
            # resize to the wall length.
            t, projected_pt = parent.project_point(o.center)
            # Clamp t so the opening sits within the wall bounds
            t_clamped = max(0.0, min(1.0, t))
            o.center = Point(
                parent.start.x + t_clamped * (parent.end.x - parent.start.x),
                parent.start.y + t_clamped * (parent.end.y - parent.start.y),
            )
            # Depth = thinnest wall the opening is connected to.
            # For doors in a gap between two structural walls, this is
            # min(wall_at_start.thickness, wall_at_end.thickness).
            connected_thicknesses = [parent.thickness]
            for wid in (parent.wall_at_start, parent.wall_at_end):
                neighbor = wall_by_id.get(wid)
                if neighbor:
                    connected_thicknesses.append(neighbor.thickness)
            o.depth = min(connected_thicknesses)

    def _write_sh3d_file(self, output_path: str, xml_content: str) -> None:
        if not output_path.endswith('.sh3d'):
            output_path += '.sh3d'
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('Home.xml', xml_content)
            zf.writestr('1/door/door.obj', DOOR_OBJ_CONTENT)
            zf.writestr('3/window85x163/window85x163.obj', WINDOW_OBJ_CONTENT)
            zf.writestr('0', base64.b64decode(DUMMY_PNG_B64))
            zf.writestr('2', base64.b64decode(DUMMY_PNG_B64))


FixedSweetHome3DExporter = SweetHome3DExporter