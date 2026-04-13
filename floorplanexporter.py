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
CENTERLINE_SEARCH_RADIUS: float = 150.0
OPENING_AXIS_SNAP_TOLERANCE: float = 30.0
OPENING_EXTENSION_TOLERANCE: float = 30.0
WALL_FACING_TOLERANCE: float = 40.0

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

    This guarantees:
      • every enclosed space is detected,
      • no room polygon contains walls inside it,
      • adjacent rooms connected only by a doorway are kept separate
        (because opening walls seal the gap in the raster).
    """

    def __init__(
        self,
        resolution: float = 1.0,
        min_room_area_cm2: float = 2000.0,
        wall_dilation_px: int = 2,
        contour_simplify_ratio: float = 0.012,
    ) -> None:
        self.resolution = resolution              # cm per raster pixel
        self.min_room_area_cm2 = min_room_area_cm2
        self.wall_dilation_px = wall_dilation_px  # close sub-pixel gaps
        self.contour_simplify_ratio = contour_simplify_ratio

    # ---- public API (same signature the exporter expects) ----
    def detect(self, walls: List[Wall]) -> List[Room]:
        if len(walls) < 3:
            return []

        mask, origin = self._rasterize_walls(walls)
        interior = self._extract_interior(mask)
        rooms = self._components_to_rooms(interior, origin)
        return rooms

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
        self, interior: np.ndarray, origin: Tuple[float, float]
    ) -> List[Room]:
        min_x, min_y = origin
        res = self.resolution
        num_labels, labels = cv2.connectedComponents(interior)

        rooms: List[Room] = []
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8) * 255
            area_px = int(np.sum(component > 0))
            area_cm2 = area_px * (res * res)
            if area_cm2 < self.min_room_area_cm2:
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
    ) -> List[Room]:
        structural_walls = self.wall_converter.convert(wall_rectangles)

        opening_walls, openings = self.opening_processor.process(
            structural_walls,
            door_rects=door_rectangles,
            window_rects=window_rectangles,
            gap_rects=gap_rectangles,
            fused_doors=fused_doors,
        )

        all_walls = structural_walls + opening_walls
        self._move_opening_walls_to_structural_centerlines(
            opening_walls, structural_walls
        )
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
            resolution=1.0,
            min_room_area_cm2=2000.0,
            wall_dilation_px=2,
            contour_simplify_ratio=0.012,
        ).detect(all_walls)

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

    # ----- opening-wall → structural-centerline snapping (unchanged) -----
    def _move_opening_walls_to_structural_centerlines(
        self,
        opening_walls: List[Wall],
        structural_walls: List[Wall],
    ) -> None:
        horiz_s = [w for w in structural_walls if w.is_horizontal]
        vert_s = [w for w in structural_walls if not w.is_horizontal]
        search_r = CENTERLINE_SEARCH_RADIUS
        self._door_placement_debug: List[dict] = []

        for ow in opening_walls:
            if ow.length < 0.1:
                continue
            _orig = (ow.start.x, ow.start.y, ow.end.x, ow.end.y)
            if ow.is_horizontal:
                ow_y = ow.get_centerline_y()
                ow_cx = (ow.get_min_x() + ow.get_max_x()) / 2
                best_y, best_yd = None, search_r
                for sw in horiz_s:
                    d = abs(sw.get_centerline_y() - ow_y)
                    if (
                        d < best_yd
                        and not (
                            ow.get_max_x() < sw.get_min_x() - 20
                            or ow.get_min_x() > sw.get_max_x() + 20
                        )
                    ):
                        best_y, best_yd = sw.get_centerline_y(), d
                if best_y is not None:
                    ow.start.y = ow.end.y = best_y
                best_gap, best_lx, best_rx = float('inf'), None, None
                left_c = [
                    (sw.get_centerline_x() + sw.thickness / 2, sw) for sw in vert_s
                    if (
                        sw.get_min_y() - WALL_FACING_TOLERANCE <= ow_y <= sw.get_max_y() + WALL_FACING_TOLERANCE
                        and sw.get_centerline_x() < ow_cx
                        and abs(sw.get_centerline_x() - ow_cx) <= search_r
                    )
                ]
                left_c.extend([
                    (sw.get_max_x(), sw) for sw in horiz_s
                    if (
                        abs(sw.get_centerline_y() - ow_y) <= WALL_FACING_TOLERANCE
                        and sw.get_max_x() < ow_cx
                        and abs(sw.get_max_x() - ow_cx) <= search_r
                    )
                ])
                right_c = [
                    (sw.get_centerline_x() - sw.thickness / 2, sw) for sw in vert_s
                    if (
                        sw.get_min_y() - WALL_FACING_TOLERANCE <= ow_y <= sw.get_max_y() + WALL_FACING_TOLERANCE
                        and sw.get_centerline_x() >= ow_cx
                        and abs(sw.get_centerline_x() - ow_cx) <= search_r
                    )
                ]
                right_c.extend([
                    (sw.get_min_x(), sw) for sw in horiz_s
                    if (
                        abs(sw.get_centerline_y() - ow_y) <= WALL_FACING_TOLERANCE
                        and sw.get_min_x() >= ow_cx
                        and abs(sw.get_min_x() - ow_cx) <= search_r
                    )
                ])
                for lx, lw in left_c:
                    for rx, rw in right_c:
                        if not (lw.get_min_y() - WALL_FACING_TOLERANCE <= ow_y <= lw.get_max_y() + WALL_FACING_TOLERANCE):
                            continue
                        if not (rw.get_min_y() - WALL_FACING_TOLERANCE <= ow_y <= rw.get_max_y() + WALL_FACING_TOLERANCE):
                            continue
                        if lw.is_horizontal and rw.is_horizontal:
                            y_ovl = min(
                                lw.get_centerline_y() + lw.thickness / 2,
                                rw.get_centerline_y() + rw.thickness / 2,
                            ) - max(
                                lw.get_centerline_y() - lw.thickness / 2,
                                rw.get_centerline_y() - rw.thickness / 2,
                            )
                            if y_ovl < -5.0:
                                continue
                        gap = rx - lx
                        if gap <= 5.0 or gap >= best_gap:
                            continue
                        occupied = any(
                            oow.wall_id != ow.wall_id
                            and oow.is_horizontal
                            and abs(oow.get_centerline_y() - ow_y) < 20.0
                            and oow.get_min_x() < rx
                            and oow.get_max_x() > lx
                            for oow in opening_walls
                        ) or any(
                            abs(sw.get_centerline_y() - ow.get_centerline_y()) < 20.0
                            and sw.get_min_x() <= lx + 5
                            and sw.get_max_x() >= rx - 5
                            for sw in horiz_s
                        )
                        if not occupied:
                            best_gap, best_lx, best_rx = gap, lx, rx
                if best_lx is not None:
                    ow.start.x, ow.end.x = best_lx, best_rx
                elif best_y is not None:
                    ow.start.y = ow.end.y = (_orig[1] + _orig[3]) / 2
                self._door_placement_debug.append({
                    'h': True, 'orig': _orig,
                    'lc': list(left_c), 'rc': list(right_c),
                    'sel': (best_lx, best_rx) if best_lx is not None else None,
                    'final': (ow.start.x, ow.start.y, ow.end.x, ow.end.y),
                })
            else:
                ow_x = ow.get_centerline_x()
                ow_cy = (ow.get_min_y() + ow.get_max_y()) / 2
                best_x, best_xd = None, search_r
                for sw in vert_s:
                    d = abs(sw.get_centerline_x() - ow_x)
                    if (
                        d < best_xd
                        and not (
                            ow.get_max_y() < sw.get_min_y() - 20
                            or ow.get_min_y() > sw.get_max_y() + 20
                        )
                    ):
                        best_x, best_xd = sw.get_centerline_x(), d
                if best_x is not None:
                    ow.start.x = ow.end.x = best_x
                best_gap, best_ty, best_by = float('inf'), None, None
                top_c = [
                    (sw.get_centerline_y() + sw.thickness / 2, sw) for sw in horiz_s
                    if (
                        sw.get_min_x() - WALL_FACING_TOLERANCE <= ow_x <= sw.get_max_x() + WALL_FACING_TOLERANCE
                        and sw.get_centerline_y() < ow_cy
                        and abs(sw.get_centerline_y() - ow_cy) <= search_r
                    )
                ]
                top_c.extend([
                    (sw.get_max_y(), sw) for sw in vert_s
                    if (
                        abs(sw.get_centerline_x() - ow_x) <= WALL_FACING_TOLERANCE
                        and sw.get_max_y() < ow_cy
                        and abs(sw.get_max_y() - ow_cy) <= search_r
                    )
                ])
                bot_c = [
                    (sw.get_centerline_y() - sw.thickness / 2, sw) for sw in horiz_s
                    if (
                        sw.get_min_x() - WALL_FACING_TOLERANCE <= ow_x <= sw.get_max_x() + WALL_FACING_TOLERANCE
                        and sw.get_centerline_y() >= ow_cy
                        and abs(sw.get_centerline_y() - ow_cy) <= search_r
                    )
                ]
                bot_c.extend([
                    (sw.get_min_y(), sw) for sw in vert_s
                    if (
                        abs(sw.get_centerline_x() - ow_x) <= WALL_FACING_TOLERANCE
                        and sw.get_min_y() >= ow_cy
                        and abs(sw.get_min_y() - ow_cy) <= search_r
                    )
                ])
                for ty, tw in top_c:
                    for by_, bw in bot_c:
                        if not (tw.get_min_x() - WALL_FACING_TOLERANCE <= ow_x <= tw.get_max_x() + WALL_FACING_TOLERANCE):
                            continue
                        if not (bw.get_min_x() - WALL_FACING_TOLERANCE <= ow_x <= bw.get_max_x() + WALL_FACING_TOLERANCE):
                            continue
                        if not tw.is_horizontal and not bw.is_horizontal:
                            x_ovl = min(
                                tw.get_centerline_x() + tw.thickness / 2,
                                bw.get_centerline_x() + bw.thickness / 2,
                            ) - max(
                                tw.get_centerline_x() - tw.thickness / 2,
                                bw.get_centerline_x() - bw.thickness / 2,
                            )
                            if x_ovl < -5.0:
                                continue
                        gap = by_ - ty
                        if gap <= 5.0 or gap >= best_gap:
                            continue
                        occupied = any(
                            oow.wall_id != ow.wall_id
                            and not oow.is_horizontal
                            and abs(oow.get_centerline_x() - ow_x) < 20.0
                            and oow.get_min_y() < by_
                            and oow.get_max_y() > ty
                            for oow in opening_walls
                        ) or any(
                            abs(sw.get_centerline_x() - ow.get_centerline_x()) < 20.0
                            and sw.get_min_y() <= ty + 5
                            and sw.get_max_y() >= by_ - 5
                            for sw in vert_s
                        )
                        if not occupied:
                            best_gap, best_ty, best_by = gap, ty, by_
                if best_ty is not None:
                    if ow.start.y < ow.end.y:
                        ow.start.y, ow.end.y = best_ty, best_by
                    else:
                        ow.start.y, ow.end.y = best_by, best_ty
                elif best_x is not None:
                    ow.start.x = ow.end.x = (_orig[0] + _orig[2]) / 2
                self._door_placement_debug.append({
                    'h': False, 'orig': _orig,
                    'tc': list(top_c), 'bc': list(bot_c),
                    'sel': (best_ty, best_by) if best_ty is not None else None,
                    'final': (ow.start.x, ow.start.y, ow.end.x, ow.end.y),
                })

    def _draw_door_placement_debug(
        self,
        original_image: 'np.ndarray',
        structural_walls: List[Wall],
    ) -> 'np.ndarray':
        """Draw debug overlay showing door placement decisions."""
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
            o = entry['orig']
            f = entry['final']

            # Original position — thin red
            cv2.line(canvas, (c(o[0]), c(o[1])), (c(o[2]), c(o[3])),
                     (0, 0, 255), 1)

            if is_h:
                fy = c(f[1])
                # Left candidates — orange edge lines
                for pos, sw in entry['lc']:
                    px = c(pos)
                    if not sw.is_horizontal:
                        y1, y2 = c(sw.get_min_y()), c(sw.get_max_y())
                    else:
                        y1 = c(sw.get_centerline_y() - sw.thickness / 2)
                        y2 = c(sw.get_centerline_y() + sw.thickness / 2)
                    cv2.line(canvas, (px, y1), (px, y2), (0, 165, 255), 1)
                    cv2.putText(canvas, 'L', (px - 8, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
                # Right candidates — cyan edge lines
                for pos, sw in entry['rc']:
                    px = c(pos)
                    if not sw.is_horizontal:
                        y1, y2 = c(sw.get_min_y()), c(sw.get_max_y())
                    else:
                        y1 = c(sw.get_centerline_y() - sw.thickness / 2)
                        y2 = c(sw.get_centerline_y() + sw.thickness / 2)
                    cv2.line(canvas, (px, y1), (px, y2), (255, 255, 0), 1)
                    cv2.putText(canvas, 'R', (px + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                # Selected boundaries — green
                sel = entry['sel']
                if sel:
                    cv2.line(canvas, (c(sel[0]), fy - 20), (c(sel[0]), fy + 20),
                             (0, 255, 0), 2)
                    cv2.line(canvas, (c(sel[1]), fy - 20), (c(sel[1]), fy + 20),
                             (0, 255, 0), 2)
                    gap = abs(sel[1] - sel[0])
                    mid = c((sel[0] + sel[1]) / 2)
                    cv2.putText(canvas, f'{gap:.0f}cm', (mid - 15, fy - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                fx = c(f[0])
                # Top candidates — orange
                for pos, sw in entry['tc']:
                    py = c(pos)
                    if sw.is_horizontal:
                        x1, x2 = c(sw.get_min_x()), c(sw.get_max_x())
                    else:
                        x1 = c(sw.get_centerline_x() - sw.thickness / 2)
                        x2 = c(sw.get_centerline_x() + sw.thickness / 2)
                    cv2.line(canvas, (x1, py), (x2, py), (0, 165, 255), 1)
                    cv2.putText(canvas, 'T', (x2 + 2, py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
                # Bottom candidates — cyan
                for pos, sw in entry['bc']:
                    py = c(pos)
                    if sw.is_horizontal:
                        x1, x2 = c(sw.get_min_x()), c(sw.get_max_x())
                    else:
                        x1 = c(sw.get_centerline_x() - sw.thickness / 2)
                        x2 = c(sw.get_centerline_x() + sw.thickness / 2)
                    cv2.line(canvas, (x1, py), (x2, py), (255, 255, 0), 1)
                    cv2.putText(canvas, 'B', (x2 + 2, py + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                # Selected boundaries — green
                sel = entry['sel']
                if sel:
                    cv2.line(canvas, (fx - 20, c(sel[0])), (fx + 20, c(sel[0])),
                             (0, 255, 0), 2)
                    cv2.line(canvas, (fx - 20, c(sel[1])), (fx + 20, c(sel[1])),
                             (0, 255, 0), 2)
                    gap = abs(sel[1] - sel[0])
                    mid = c((sel[0] + sel[1]) / 2)
                    cv2.putText(canvas, f'{gap:.0f}cm', (fx + 25, mid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

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