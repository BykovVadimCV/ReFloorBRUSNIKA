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
MAX_DOOR_DEPTH: float = 15.0
MAX_WINDOW_DEPTH: float = 15.0

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
    is_bridging: bool = False  # True = spans between two structural walls

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
            mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
            label = wall.wall_id.replace('wall-', '')[:8]
            cv2.putText(canvas, label, (mx + 2, my - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
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
        self, pixels_to_cm: float = 2.54, min_thickness: float = 2.0,
        max_expected_thickness: float = 30.0,
    ) -> None:
        self.pixels_to_cm = pixels_to_cm
        self.min_thickness = min_thickness
        self.max_expected_thickness = max_expected_thickness

    def convert(self, rectangles: List[Dict]) -> List[Wall]:
        walls: List[Wall] = []
        for rect in rectangles:
            x1, y1, x2, y2 = parse_rectangle(rect)
            x1 *= self.pixels_to_cm; y1 *= self.pixels_to_cm
            x2 *= self.pixels_to_cm; y2 *= self.pixels_to_cm
            width, height = abs(x2 - x1), abs(y2 - y1)
            if width >= height:
                thickness = max(height, self.min_thickness)
                if thickness > self.max_expected_thickness:
                    thickness = self.max_expected_thickness
                cy = (y1 + y2) / 2
                start, end = Point(min(x1, x2), cy), Point(max(x1, x2), cy)
            else:
                thickness = max(width, self.min_thickness)
                if thickness > self.max_expected_thickness:
                    thickness = self.max_expected_thickness
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
# PlausibilityValidator
# ============================================================
class PlausibilityValidator:
    STANDARD_THICKNESSES = (8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)

    def __init__(
        self,
        min_wall_thickness: float = 5.0,
        max_wall_thickness: float = 40.0,
        min_wall_length: float = 10.0,
        min_opening_width: float = 40.0,
        max_opening_width: float = 200.0,
    ) -> None:
        self.min_wall_thickness = min_wall_thickness
        self.max_wall_thickness = max_wall_thickness
        self.min_wall_length = min_wall_length
        self.min_opening_width = min_opening_width
        self.max_opening_width = max_opening_width

    def validate_walls(self, walls: List[Wall]) -> List[Wall]:
        result: List[Wall] = []
        for w in walls:
            if w.length < 0.5:
                continue
            if w.length < self.min_wall_length:
                has_neighbor = any(
                    other.wall_id != w.wall_id
                    and (
                        w.start.distance_to(other.start) < 20.0
                        or w.start.distance_to(other.end) < 20.0
                        or w.end.distance_to(other.start) < 20.0
                        or w.end.distance_to(other.end) < 20.0
                    )
                    for other in walls
                )
                if not has_neighbor:
                    continue
            w.thickness = max(self.min_wall_thickness,
                              min(w.thickness, self.max_wall_thickness))
            w.thickness = self._snap_to_standard(w.thickness)
            result.append(w)
        return result

    def validate_openings(self, openings: List[Opening]) -> List[Opening]:
        return [
            o for o in openings
            if self.min_opening_width <= o.width <= self.max_opening_width
        ]

    def _snap_to_standard(self, thickness: float) -> float:
        best = min(self.STANDARD_THICKNESSES, key=lambda s: abs(s - thickness))
        return best if abs(best - thickness) <= 5.0 else thickness


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
            t = (
                (center.x - wall.start.x) * dx
                + (center.y - wall.start.y) * dy
            ) / length_sq
            if t < 0.0 or t > 1.0:
                continue
            proj = Point(wall.start.x + t * dx, wall.start.y + t * dy)
            perp_dist = center.distance_to(proj)
            if perp_dist < self.tolerance and perp_dist < best_distance:
                best_distance = perp_dist
                best_wall = wall
        return best_wall

    def find_bridging_walls(
        self,
        center: Point,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        is_horizontal: bool,
        gap_tolerance: float = 30.0,
    ) -> Optional[Tuple[Wall, Wall]]:
        """Find two walls whose facing edges form a gap containing the bbox.

        For a horizontal bbox, search pairs of vertical walls whose
        inner faces create a gap that overlaps the bbox x-range.
        For a vertical bbox, search pairs of horizontal walls whose
        inner faces create a gap that overlaps the bbox y-range.

        Returns the tightest-fitting (wall_a, wall_b) pair, or *None*.
        """
        best_pair: Optional[Tuple[Wall, Wall]] = None
        best_gap = float('inf')

        if is_horizontal:
            # Horizontal bbox → look for vertical wall pairs (left/right)
            verts = [w for w in self.walls if not w.is_horizontal]
            # Each vertical wall must span the bbox y-range
            for i, wa in enumerate(verts):
                wa_min_y = min(wa.start.y, wa.end.y)
                wa_max_y = max(wa.start.y, wa.end.y)
                if wa_max_y < bbox_y1 - gap_tolerance or wa_min_y > bbox_y2 + gap_tolerance:
                    continue
                wa_cx = wa.get_centerline_x()
                # wa must be to the left of bbox center
                if wa_cx > center.x:
                    continue
                wa_right_edge = wa_cx + wa.thickness / 2

                for wb in verts[i + 1:]:
                    wb_min_y = min(wb.start.y, wb.end.y)
                    wb_max_y = max(wb.start.y, wb.end.y)
                    if wb_max_y < bbox_y1 - gap_tolerance or wb_min_y > bbox_y2 + gap_tolerance:
                        continue
                    wb_cx = wb.get_centerline_x()
                    # wb must be to the right of bbox center
                    if wb_cx < center.x:
                        continue
                    wb_left_edge = wb_cx - wb.thickness / 2

                    # Gap between facing edges
                    gap = wb_left_edge - wa_right_edge
                    if gap < 5.0:
                        continue  # walls overlap or too close
                    # Door bbox must fit within the gap (with tolerance)
                    if bbox_x1 < wa_right_edge - gap_tolerance:
                        continue
                    if bbox_x2 > wb_left_edge + gap_tolerance:
                        continue
                    if gap < best_gap:
                        best_gap = gap
                        best_pair = (wa, wb)
        else:
            # Vertical bbox → look for horizontal wall pairs (top/bottom)
            horizs = [w for w in self.walls if w.is_horizontal]
            for i, wa in enumerate(horizs):
                wa_min_x = min(wa.start.x, wa.end.x)
                wa_max_x = max(wa.start.x, wa.end.x)
                if wa_max_x < bbox_x1 - gap_tolerance or wa_min_x > bbox_x2 + gap_tolerance:
                    continue
                wa_cy = wa.get_centerline_y()
                if wa_cy > center.y:
                    continue
                wa_bottom_edge = wa_cy + wa.thickness / 2

                for wb in horizs[i + 1:]:
                    wb_min_x = min(wb.start.x, wb.end.x)
                    wb_max_x = max(wb.start.x, wb.end.x)
                    if wb_max_x < bbox_x1 - gap_tolerance or wb_min_x > bbox_x2 + gap_tolerance:
                        continue
                    wb_cy = wb.get_centerline_y()
                    if wb_cy < center.y:
                        continue
                    wb_top_edge = wb_cy - wb.thickness / 2

                    gap = wb_top_edge - wa_bottom_edge
                    if gap < 5.0:
                        continue
                    if bbox_y1 < wa_bottom_edge - gap_tolerance:
                        continue
                    if bbox_y2 > wb_top_edge + gap_tolerance:
                        continue
                    if gap < best_gap:
                        best_gap = gap
                        best_pair = (wa, wb)

        return best_pair

    def find_collinear_gap(
        self,
        center: Point,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        gap_tolerance: float = 30.0,
    ) -> Optional[Tuple[Wall, Wall, bool]]:
        """Find two collinear wall segments whose facing endpoints form a gap
        containing the door bbox.

        Unlike ``find_bridging_walls`` (walls on *opposite* sides of the bbox),
        this searches for walls at the *same* axis position whose *endpoints*
        create a gap.  Typical case: a horizontal wall is split into two
        segments with a doorway between them.

        Returns ``(wall_left, wall_right, gap_is_horizontal)`` or *None*.
        ``gap_is_horizontal`` indicates the wall orientation (True = horizontal
        walls with a gap along X; False = vertical walls with a gap along Y).
        """
        best_pair: Optional[Tuple[Wall, Wall, bool]] = None
        best_gap = float('inf')

        for search_horiz in (True, False):
            candidates = [w for w in self.walls if w.is_horizontal == search_horiz]
            for i, wa in enumerate(candidates):
                for wb in candidates[i + 1:]:
                    if search_horiz:
                        # Both horizontal — must share approximate Y
                        wa_cy = wa.get_centerline_y()
                        wb_cy = wb.get_centerline_y()
                        max_thickness = max(wa.thickness, wb.thickness)
                        if abs(wa_cy - wb_cy) > max_thickness:
                            continue
                        # Door center must be near the wall line
                        avg_y = (wa_cy + wb_cy) / 2
                        if abs(center.y - avg_y) > max_thickness / 2 + gap_tolerance:
                            continue
                        # Determine which wall is left, which is right
                        wa_max_x = max(wa.start.x, wa.end.x)
                        wb_min_x = min(wb.start.x, wb.end.x)
                        wa_min_x = min(wa.start.x, wa.end.x)
                        wb_max_x = max(wb.start.x, wb.end.x)
                        if wa_max_x < wb_min_x:
                            left_end, right_start = wa_max_x, wb_min_x
                            w_left, w_right = wa, wb
                        elif wb_max_x < wa_min_x:
                            left_end, right_start = wb_max_x, wa_min_x
                            w_left, w_right = wb, wa
                        else:
                            continue  # overlapping, not a gap
                        gap = right_start - left_end
                        if gap < 2.0 or gap >= best_gap:
                            continue
                        # Door bbox x-range must fit within the gap
                        if bbox_x1 < left_end - gap_tolerance:
                            continue
                        if bbox_x2 > right_start + gap_tolerance:
                            continue
                        # Door center must be inside the gap
                        if center.x < left_end - gap_tolerance or center.x > right_start + gap_tolerance:
                            continue
                        best_gap = gap
                        best_pair = (w_left, w_right, True)
                    else:
                        # Both vertical — must share approximate X
                        wa_cx = wa.get_centerline_x()
                        wb_cx = wb.get_centerline_x()
                        max_thickness = max(wa.thickness, wb.thickness)
                        if abs(wa_cx - wb_cx) > max_thickness:
                            continue
                        avg_x = (wa_cx + wb_cx) / 2
                        if abs(center.x - avg_x) > max_thickness / 2 + gap_tolerance:
                            continue
                        wa_max_y = max(wa.start.y, wa.end.y)
                        wb_min_y = min(wb.start.y, wb.end.y)
                        wa_min_y = min(wa.start.y, wa.end.y)
                        wb_max_y = max(wb.start.y, wb.end.y)
                        if wa_max_y < wb_min_y:
                            top_end, bot_start = wa_max_y, wb_min_y
                            w_top, w_bot = wa, wb
                        elif wb_max_y < wa_min_y:
                            top_end, bot_start = wb_max_y, wa_min_y
                            w_top, w_bot = wb, wa
                        else:
                            continue
                        gap = bot_start - top_end
                        if gap < 2.0 or gap >= best_gap:
                            continue
                        if bbox_y1 < top_end - gap_tolerance:
                            continue
                        if bbox_y2 > bot_start + gap_tolerance:
                            continue
                        if center.y < top_end - gap_tolerance or center.y > bot_start + gap_tolerance:
                            continue
                        best_gap = gap
                        best_pair = (w_top, w_bot, False)

        return best_pair


def _clamp_opening_depth(depth: float, opening_type) -> float:
    if opening_type in (OpeningType.WINDOW, OpeningType.GAP):
        return min(depth, MAX_WINDOW_DEPTH)
    return min(depth, MAX_DOOR_DEPTH)


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
            pid, depth, angle = parent.wall_id, _clamp_opening_depth(parent.thickness, OpeningType.DOOR), parent.angle
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
            pid, depth = new_wall.wall_id, _clamp_opening_depth(thickness, OpeningType.DOOR)

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
            pid, depth, angle = parent.wall_id, _clamp_opening_depth(parent.thickness, opening_type), parent.angle
        else:
            # No same-orientation parent → first check for a collinear gap
            # (two wall segments at the same axis with a doorway between them).
            collinear = parent_finder.find_collinear_gap(
                center, x1, y1, x2, y2,
            )
            if collinear is not None:
                w_a, w_b, gap_is_horizontal = collinear
                thickness = min(w_a.thickness, w_b.thickness)
                if gap_is_horizontal:
                    # Gap in a horizontal wall — bridging wall is horizontal
                    sx = max(w_a.start.x, w_a.end.x)   # left segment's right end
                    ex = min(w_b.start.x, w_b.end.x)   # right segment's left end
                    avg_y = (w_a.get_centerline_y() + w_b.get_centerline_y()) / 2
                    start = Point(sx, avg_y)
                    end = Point(ex, avg_y)
                    angle = 0.0
                    ow = ex - sx
                    is_horizontal = True  # flip orientation to match wall
                else:
                    # Gap in a vertical wall — bridging wall is vertical
                    sy = max(w_a.start.y, w_a.end.y)   # top segment's bottom end
                    ey = min(w_b.start.y, w_b.end.y)   # bottom segment's top end
                    avg_x = (w_a.get_centerline_x() + w_b.get_centerline_x()) / 2
                    start = Point(avg_x, sy)
                    end = Point(avg_x, ey)
                    angle = math.pi / 2
                    ow = ey - sy
                    is_horizontal = False
                new_wall = Wall(
                    start=start, end=end,
                    thickness=thickness, is_structural=False,
                    is_bridging=True,
                )
                pid, depth = new_wall.wall_id, _clamp_opening_depth(thickness, opening_type)
            else:
                # Try to bridge two walls on opposite ends of the bbox.
                bridge_pair = parent_finder.find_bridging_walls(
                    center, x1, y1, x2, y2, is_horizontal,
                )
                if bridge_pair is not None:
                    w_a, w_b = bridge_pair
                    thickness = min(w_a.thickness, w_b.thickness)
                    if is_horizontal:
                        # Horizontal bbox → bridging vertical walls left/right
                        # Wall endpoints at inner faces (not centerlines)
                        sx = w_a.get_centerline_x() + w_a.thickness / 2
                        ex = w_b.get_centerline_x() - w_b.thickness / 2
                        start = Point(sx, center.y)
                        end = Point(ex, center.y)
                        angle = 0.0
                        ow = ex - sx
                    else:
                        # Vertical bbox → bridging horizontal walls top/bottom
                        sy = w_a.get_centerline_y() + w_a.thickness / 2
                        ey = w_b.get_centerline_y() - w_b.thickness / 2
                        start = Point(center.x, sy)
                        end = Point(center.x, ey)
                        angle = math.pi / 2
                        ow = ey - sy
                    new_wall = Wall(
                        start=start, end=end,
                        thickness=thickness, is_structural=False,
                        is_bridging=True,
                    )
                    pid, depth = new_wall.wall_id, _clamp_opening_depth(thickness, opening_type)
                else:
                    # Fallback: synthetic opening wall from bbox
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
                    new_wall = Wall(
                        start=start, end=end,
                        thickness=thickness, is_structural=False,
                    )
                    pid, depth = new_wall.wall_id, _clamp_opening_depth(thickness, opening_type)

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
        mask = self._close_wall_gaps(mask, walls, origin)
        interior = self._extract_interior(mask)
        rooms = self._components_to_rooms(interior, origin)
        return rooms

    # ---- bridge small gaps between wall endpoints ----
    def _close_wall_gaps(
        self,
        mask: np.ndarray,
        walls: List[Wall],
        origin: Tuple[float, float],
        max_gap_cm: float = 15.0,
    ) -> np.ndarray:
        min_x, min_y = origin
        res = self.resolution
        endpoints: List[Tuple[float, float, str, float]] = []
        for w in walls:
            if w.length < 0.5:
                continue
            endpoints.append((w.start.x, w.start.y, w.wall_id, w.thickness))
            endpoints.append((w.end.x, w.end.y, w.wall_id, w.thickness))

        for i, (x1, y1, id1, t1) in enumerate(endpoints):
            for x2, y2, id2, t2 in endpoints[i + 1:]:
                if id1 == id2:
                    continue
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < 0.5 or dist > max_gap_cm:
                    continue
                px1 = int((x1 - min_x) / res)
                py1 = int((y1 - min_y) / res)
                px2 = int((x2 - min_x) / res)
                py2 = int((y2 - min_y) / res)
                line_t = max(2, int(min(t1, t2) / res / 2))
                cv2.line(mask, (px1, py1), (px2, py2), 255, thickness=line_t)
        return mask

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
# RoomPolygonSnapper  — post-process flood-fill room polygons
# to snap vertices onto actual wall inner edges
# ============================================================

@dataclass
class _WallEdgeLine:
    """One side of a wall, as an axis-aligned line segment."""
    orientation: str       # 'horizontal' or 'vertical'
    fixed_coord: float     # y for horizontal, x for vertical
    range_min: float       # min x (horiz) or min y (vert)
    range_max: float       # max x (horiz) or max y (vert)


@dataclass
class _MatchedEdge:
    """A room polygon edge matched to a wall edge line."""
    orientation: str       # 'horizontal' or 'vertical'
    fixed_coord: float     # the snapped coordinate
    range_min: float       # start of the room edge along axis
    range_max: float       # end of the room edge along axis


class RoomPolygonSnapper:
    """
    Post-processes room polygons from flood-fill detection,
    snapping vertices to actual wall inner edges to produce
    clean rectilinear room boundaries.

    Algorithm:
      1. Build a lookup of wall edge lines (each wall → two parallel edges).
      2. Classify each room polygon edge as horizontal, vertical, or diagonal.
      3. Match each H/V edge to the nearest wall edge within tolerance.
      4. Reconstruct the polygon from consecutive edge intersections.
      5. Validate area preservation; fall back to original on failure.
    """

    def __init__(
        self,
        snap_tolerance: float = 15.0,
        angle_tolerance_deg: float = 15.0,
        area_change_limit: float = 0.25,
    ) -> None:
        self.snap_tolerance = snap_tolerance
        self.angle_tolerance = math.radians(angle_tolerance_deg)
        self.area_change_limit = area_change_limit

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def snap_rooms(
        self, rooms: List[Room], walls: List[Wall],
    ) -> List[Room]:
        """Snap all room polygons to wall edges. Returns new list."""
        if not walls:
            return rooms
        wall_edges = self._build_wall_edges(walls)
        h_edges = [e for e in wall_edges if e.orientation == 'horizontal']
        v_edges = [e for e in wall_edges if e.orientation == 'vertical']
        result = []
        for room in rooms:
            snapped = self._snap_single_room(room, h_edges, v_edges)
            result.append(snapped)
        return result

    # ----------------------------------------------------------
    # Step 0: Build wall edge lookup
    # ----------------------------------------------------------

    @staticmethod
    def _build_wall_edges(walls: List[Wall]) -> List[_WallEdgeLine]:
        edges: List[_WallEdgeLine] = []
        for w in walls:
            ht = w.thickness / 2.0
            if w.is_horizontal:
                cy = (w.start.y + w.end.y) / 2.0
                xmin = min(w.start.x, w.end.x)
                xmax = max(w.start.x, w.end.x)
                edges.append(_WallEdgeLine('horizontal', cy - ht, xmin, xmax))
                edges.append(_WallEdgeLine('horizontal', cy + ht, xmin, xmax))
            else:
                cx = (w.start.x + w.end.x) / 2.0
                ymin = min(w.start.y, w.end.y)
                ymax = max(w.start.y, w.end.y)
                edges.append(_WallEdgeLine('vertical', cx - ht, ymin, ymax))
                edges.append(_WallEdgeLine('vertical', cx + ht, ymin, ymax))
        return edges

    # ----------------------------------------------------------
    # Step 1: Classify edges
    # ----------------------------------------------------------

    def _classify_edge(self, p1: Point, p2: Point) -> str:
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        length = math.hypot(dx, dy)
        if length < 1.0:
            return 'skip'
        angle = math.atan2(dy, dx)
        if angle <= self.angle_tolerance:
            return 'horizontal'
        if angle >= (math.pi / 2.0 - self.angle_tolerance):
            return 'vertical'
        return 'diagonal'

    # ----------------------------------------------------------
    # Step 2: Match room edge to wall edge
    # ----------------------------------------------------------

    def _find_best_wall_edge(
        self,
        p1: Point, p2: Point,
        orientation: str,
        edges: List[_WallEdgeLine],
        centroid: Point,
    ) -> Optional[float]:
        """Return the fixed_coord of the best matching wall edge, or None."""
        if orientation == 'horizontal':
            room_coord = (p1.y + p2.y) / 2.0
            room_min = min(p1.x, p2.x)
            room_max = max(p1.x, p2.x)
            room_span = room_max - room_min
        else:
            room_coord = (p1.x + p2.x) / 2.0
            room_min = min(p1.y, p2.y)
            room_max = max(p1.y, p2.y)
            room_span = room_max - room_min

        best_coord: Optional[float] = None
        best_dist = self.snap_tolerance + 1.0

        for edge in edges:
            dist = abs(edge.fixed_coord - room_coord)
            if dist > self.snap_tolerance:
                continue

            # Check range overlap
            overlap_min = max(edge.range_min, room_min)
            overlap_max = min(edge.range_max, room_max)
            overlap = overlap_max - overlap_min
            if overlap < room_span * 0.3:
                continue

            # Prefer the edge closest to the room centroid (inner edge)
            if orientation == 'horizontal':
                centroid_dist = abs(edge.fixed_coord - centroid.y)
            else:
                centroid_dist = abs(edge.fixed_coord - centroid.x)

            # Score: primary = proximity to room edge, secondary = closer to centroid
            score = dist + centroid_dist * 0.01

            if score < best_dist:
                best_dist = score
                best_coord = edge.fixed_coord

        return best_coord

    # ----------------------------------------------------------
    # Step 3: Reconstruct polygon
    # ----------------------------------------------------------

    def _snap_single_room(
        self,
        room: Room,
        h_edges: List[_WallEdgeLine],
        v_edges: List[_WallEdgeLine],
    ) -> Room:
        pts = room.points
        n = len(pts)
        if n < 3:
            return room

        # Compute centroid
        cx = sum(p.x for p in pts) / n
        cy = sum(p.y for p in pts) / n
        centroid = Point(cx, cy)

        # Classify and match each edge
        matched: List[Optional[_MatchedEdge]] = []
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            orient = self._classify_edge(p1, p2)

            if orient in ('diagonal', 'skip'):
                matched.append(None)
                continue

            edges_pool = h_edges if orient == 'horizontal' else v_edges
            snapped_coord = self._find_best_wall_edge(
                p1, p2, orient, edges_pool, centroid,
            )

            if snapped_coord is None:
                # No wall found — use average of the two endpoints
                if orient == 'horizontal':
                    snapped_coord = (p1.y + p2.y) / 2.0
                else:
                    snapped_coord = (p1.x + p2.x) / 2.0

            if orient == 'horizontal':
                rmin = min(p1.x, p2.x)
                rmax = max(p1.x, p2.x)
            else:
                rmin = min(p1.y, p2.y)
                rmax = max(p1.y, p2.y)

            matched.append(_MatchedEdge(orient, snapped_coord, rmin, rmax))

        # Compact: remove None entries, merging adjacent non-None edges
        compact = [m for m in matched if m is not None]
        if len(compact) < 2:
            return room

        # Reconstruct polygon from consecutive edge intersections
        new_points: List[Point] = []
        m = len(compact)
        for i in range(m):
            e_curr = compact[i]
            e_next = compact[(i + 1) % m]

            if e_curr.orientation != e_next.orientation:
                # Perpendicular edges → single intersection point
                if e_curr.orientation == 'horizontal':
                    new_points.append(Point(e_next.fixed_coord, e_curr.fixed_coord))
                else:
                    new_points.append(Point(e_curr.fixed_coord, e_next.fixed_coord))
            else:
                # Same orientation → L-shaped jog, insert connector
                if e_curr.orientation == 'horizontal':
                    # Need a vertical connector
                    # Use the range boundary where the two horizontal edges meet
                    conn_x = self._find_connector_coord(
                        e_curr, e_next, v_edges, centroid, 'vertical',
                    )
                    new_points.append(Point(conn_x, e_curr.fixed_coord))
                    new_points.append(Point(conn_x, e_next.fixed_coord))
                else:
                    conn_y = self._find_connector_coord(
                        e_curr, e_next, h_edges, centroid, 'horizontal',
                    )
                    new_points.append(Point(e_curr.fixed_coord, conn_y))
                    new_points.append(Point(e_next.fixed_coord, conn_y))

        if len(new_points) < 3:
            return room

        # Deduplicate consecutive equal points
        deduped: List[Point] = [new_points[0]]
        for p in new_points[1:]:
            if p.distance_to(deduped[-1]) > 0.5:
                deduped.append(p)
        if len(deduped) > 1 and deduped[-1].distance_to(deduped[0]) < 0.5:
            deduped.pop()

        if len(deduped) < 3:
            return room

        # Validate area preservation
        if not self._validate_polygon(pts, deduped):
            return room

        return Room(
            points=deduped,
            room_id=room.room_id,
            name=room.name,
        )

    def _find_connector_coord(
        self,
        e_curr: _MatchedEdge,
        e_next: _MatchedEdge,
        cross_edges: List[_WallEdgeLine],
        centroid: Point,
        cross_orient: str,
    ) -> float:
        """Find the coordinate for a perpendicular connector between
        two same-orientation edges (L-shaped jog)."""
        # The connector should be at the boundary of the two edges' ranges
        if e_curr.orientation == 'horizontal':
            # Connector is vertical. Look for a vertical wall edge
            # between the two horizontal edges' y-values.
            y_lo = min(e_curr.fixed_coord, e_next.fixed_coord)
            y_hi = max(e_curr.fixed_coord, e_next.fixed_coord)
            range_overlap_min = max(e_curr.range_min, e_next.range_min)
            range_overlap_max = min(e_curr.range_max, e_next.range_max)

            best_x: Optional[float] = None
            best_dist = float('inf')
            for edge in cross_edges:
                if edge.range_min > y_hi or edge.range_max < y_lo:
                    continue
                if not (range_overlap_min - 10 <= edge.fixed_coord
                        <= range_overlap_max + 10):
                    # Must be in the x-range where both edges exist
                    if not (e_curr.range_min - 10 <= edge.fixed_coord
                            <= e_curr.range_max + 10
                            or e_next.range_min - 10 <= edge.fixed_coord
                            <= e_next.range_max + 10):
                        continue
                dist = abs(edge.fixed_coord - centroid.x)
                if dist < best_dist:
                    best_dist = dist
                    best_x = edge.fixed_coord
            if best_x is not None:
                return best_x

            # Fallback: use the x-coordinate where the two ranges meet
            if e_curr.range_max < e_next.range_max:
                return e_curr.range_max
            return e_next.range_max
        else:
            # Connector is horizontal
            x_lo = min(e_curr.fixed_coord, e_next.fixed_coord)
            x_hi = max(e_curr.fixed_coord, e_next.fixed_coord)
            range_overlap_min = max(e_curr.range_min, e_next.range_min)
            range_overlap_max = min(e_curr.range_max, e_next.range_max)

            best_y: Optional[float] = None
            best_dist = float('inf')
            for edge in cross_edges:
                if edge.range_min > x_hi or edge.range_max < x_lo:
                    continue
                if not (range_overlap_min - 10 <= edge.fixed_coord
                        <= range_overlap_max + 10):
                    if not (e_curr.range_min - 10 <= edge.fixed_coord
                            <= e_curr.range_max + 10
                            or e_next.range_min - 10 <= edge.fixed_coord
                            <= e_next.range_max + 10):
                        continue
                dist = abs(edge.fixed_coord - centroid.y)
                if dist < best_dist:
                    best_dist = dist
                    best_y = edge.fixed_coord
            if best_y is not None:
                return best_y

            if e_curr.range_max < e_next.range_max:
                return e_curr.range_max
            return e_next.range_max

    @staticmethod
    def _polygon_area(points: List[Point]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i].x * points[j].y - points[j].x * points[i].y
        return abs(area) / 2.0

    def _validate_polygon(
        self, original: List[Point], snapped: List[Point],
    ) -> bool:
        orig_area = self._polygon_area(original)
        snap_area = self._polygon_area(snapped)
        if orig_area < 1.0:
            return True
        ratio = abs(snap_area - orig_area) / orig_area
        return ratio <= self.area_change_limit


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
        root.set('wallHeight', f'{self.wall_height:.4f}')
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
        e.set('height', f'{w.height:.4f}')
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
            e.set('elevation', f'{o.elevation:.4f}')
        e.set('angle', str(o.angle))
        e.set('width', f'{o.width / sf:.4f}')
        e.set('depth', f'{o.depth / sf:.4f}')
        e.set('height', f'{o.height:.4f}')
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

        validator = PlausibilityValidator()
        structural_walls = validator.validate_walls(structural_walls)

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
        self._process_wall_connections(all_walls, structural_walls, opening_walls)
        self._update_openings_from_walls(openings, all_walls)
        self._clip_openings_to_walls(openings, all_walls)

        # ──────────────────────────────────────────────────────
        # FIX: use the new flood-fill RoomDetector
        # ──────────────────────────────────────────────────────
        rooms = RoomDetector(
            resolution=1.0,
            min_room_area_cm2=2000.0,
            wall_dilation_px=6,
            contour_simplify_ratio=0.012,
        ).detect(all_walls)

        rooms = RoomPolygonSnapper(
            snap_tolerance=15.0,
            angle_tolerance_deg=15.0,
            area_change_limit=0.25,
        ).snap_rooms(rooms, all_walls)

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

        for ow in opening_walls:
            if ow.length < 0.1:
                continue
            # Bridging walls already span between structural wall
            # centerlines — do not reposition them.
            if getattr(ow, 'is_bridging', False):
                continue
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
                # Use inner faces: left wall's right face, right wall's left face
                left_c = [
                    (sw.get_centerline_x() + sw.thickness / 2, sw) for sw in vert_s
                    if (
                        sw.get_min_y() - search_r <= ow_y <= sw.get_max_y() + search_r
                        and sw.get_centerline_x() < ow_cx
                        and abs(sw.get_centerline_x() - ow_cx) <= search_r
                    )
                ]
                right_c = [
                    (sw.get_centerline_x() - sw.thickness / 2, sw) for sw in vert_s
                    if (
                        sw.get_min_y() - search_r <= ow_y <= sw.get_max_y() + search_r
                        and sw.get_centerline_x() >= ow_cx
                        and abs(sw.get_centerline_x() - ow_cx) <= search_r
                    )
                ]
                for lx, _ in left_c:
                    for rx, _ in right_c:
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
                            abs(sw.get_centerline_y() - ow_y) < 20.0
                            and sw.get_min_x() <= lx + 5
                            and sw.get_max_x() >= rx - 5
                            for sw in horiz_s
                        )
                        if not occupied:
                            best_gap, best_lx, best_rx = gap, lx, rx
                if best_lx is not None:
                    ow.start.x, ow.end.x = best_lx, best_rx
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
                        sw.get_min_x() - search_r <= ow_x <= sw.get_max_x() + search_r
                        and sw.get_centerline_y() < ow_cy
                        and abs(sw.get_centerline_y() - ow_cy) <= search_r
                    )
                ]
                bot_c = [
                    (sw.get_centerline_y() - sw.thickness / 2, sw) for sw in horiz_s
                    if (
                        sw.get_min_x() - search_r <= ow_x <= sw.get_max_x() + search_r
                        and sw.get_centerline_y() >= ow_cy
                        and abs(sw.get_centerline_y() - ow_cy) <= search_r
                    )
                ]
                for ty, _ in top_c:
                    for by_, _ in bot_c:
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
                            abs(sw.get_centerline_x() - ow_x) < 20.0
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
        self._snap_endpoints_to_corners(all_walls, snap_distance=15.0)
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

    def _extension_crosses_wall(
        self,
        wall: Wall,
        endpoint: str,
        target_coord: float,
        all_walls: List[Wall],
    ) -> bool:
        point = wall.start if endpoint == 'start' else wall.end
        for other in all_walls:
            if other.wall_id == wall.wall_id:
                continue
            if other.is_horizontal == wall.is_horizontal:
                continue
            half_t = (wall.thickness + other.thickness) / 2
            if wall.is_horizontal:
                other_x = other.get_centerline_x()
                lo = min(point.x, target_coord)
                hi = max(point.x, target_coord)
                if lo < other_x < hi:
                    wall_y = (wall.start.y + wall.end.y) / 2
                    if other.get_min_y() - half_t <= wall_y <= other.get_max_y() + half_t:
                        return True
            else:
                other_y = other.get_centerline_y()
                lo = min(point.y, target_coord)
                hi = max(point.y, target_coord)
                if lo < other_y < hi:
                    wall_x = (wall.start.x + wall.end.x) / 2
                    if other.get_min_x() - half_t <= wall_x <= other.get_max_x() + half_t:
                        return True
        return False

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
                if not self._extension_crosses_wall(wall, endpoint, target, all_walls):
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
                return (abs(point.x - ox) - half, ox, True)
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
                return (abs(point.y - oy) - half, oy, True)
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
        eps = max(self.wall_overlap + 2.0, 15.0)
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
            # Depth = wall thickness (already set at creation, but refresh
            # in case the wall was adjusted during connection processing)
            o.depth = _clamp_opening_depth(parent.thickness, o.opening_type)

    def _clip_openings_to_walls(
        self,
        openings: List[Opening],
        all_walls: List[Wall],
    ) -> None:
        """Clip each opening's width so it ends at the first structural
        wall it encounters on either side along its parent wall axis.
        Also clips the parent (opening) wall endpoints to match."""
        wall_by_id = {w.wall_id: w for w in all_walls}
        structural = [w for w in all_walls if w.is_structural]

        for o in openings:
            parent = wall_by_id.get(o.parent_wall_id)
            if not parent or o.width <= 0:
                continue
            half = o.width / 2
            cx, cy = o.center.x, o.center.y

            if parent.is_horizontal:
                # Door extends left/right along X
                left_bound = cx - half
                right_bound = cx + half
                for sw in structural:
                    if sw.is_horizontal:
                        continue  # only perpendicular walls clip
                    sw_min_y = min(sw.start.y, sw.end.y)
                    sw_max_y = max(sw.start.y, sw.end.y)
                    if not (sw_min_y - 5 <= cy <= sw_max_y + 5):
                        continue
                    sw_cx = sw.get_centerline_x()
                    sw_inner_right = sw_cx + sw.thickness / 2
                    sw_inner_left = sw_cx - sw.thickness / 2
                    # Wall to the left whose right face intrudes
                    if sw_cx < cx and sw_inner_right > left_bound:
                        left_bound = max(left_bound, sw_inner_right)
                    # Wall to the right whose left face intrudes
                    if sw_cx > cx and sw_inner_left < right_bound:
                        right_bound = min(right_bound, sw_inner_left)
                new_w = max(right_bound - left_bound, 1.0)
                new_cx = (left_bound + right_bound) / 2
                o.width = new_w
                o.center = Point(new_cx, cy)
                # Clip the parent opening wall to match
                if not parent.is_structural:
                    if parent.start.x < parent.end.x:
                        parent.start.x = left_bound
                        parent.end.x = right_bound
                    else:
                        parent.start.x = right_bound
                        parent.end.x = left_bound
            else:
                # Door extends up/down along Y
                top_bound = cy - half
                bottom_bound = cy + half
                for sw in structural:
                    if not sw.is_horizontal:
                        continue  # only perpendicular walls clip
                    sw_min_x = min(sw.start.x, sw.end.x)
                    sw_max_x = max(sw.start.x, sw.end.x)
                    if not (sw_min_x - 5 <= cx <= sw_max_x + 5):
                        continue
                    sw_cy = sw.get_centerline_y()
                    sw_inner_bottom = sw_cy + sw.thickness / 2
                    sw_inner_top = sw_cy - sw.thickness / 2
                    if sw_cy < cy and sw_inner_bottom > top_bound:
                        top_bound = max(top_bound, sw_inner_bottom)
                    if sw_cy > cy and sw_inner_top < bottom_bound:
                        bottom_bound = min(bottom_bound, sw_inner_top)
                new_w = max(bottom_bound - top_bound, 1.0)
                new_cy = (top_bound + bottom_bound) / 2
                o.width = new_w
                o.center = Point(cx, new_cy)
                # Clip the parent opening wall to match
                if not parent.is_structural:
                    if parent.start.y < parent.end.y:
                        parent.start.y = top_bound
                        parent.end.y = bottom_bound
                    else:
                        parent.start.y = bottom_bound
                        parent.end.y = top_bound

    def _filter_phantom_openings(
        self,
        openings: List[Opening],
        structural_walls: List[Wall],
        tolerance: float = 5.0,
    ) -> List[Opening]:
        """Remove openings that sit entirely inside a continuous structural
        wall (no gap at an endpoint).  Openings on non-structural or bridging
        walls are always kept."""
        structural_by_id = {w.wall_id: w for w in structural_walls}
        filtered: List[Opening] = []
        for o in openings:
            parent = structural_by_id.get(o.parent_wall_id)
            if parent is None:
                filtered.append(o)
                continue
            half_w = o.width / 2
            if parent.is_horizontal:
                if (o.center.x - half_w > parent.get_min_x() + tolerance
                        and o.center.x + half_w < parent.get_max_x() - tolerance):
                    continue
            else:
                if (o.center.y - half_w > parent.get_min_y() + tolerance
                        and o.center.y + half_w < parent.get_max_y() - tolerance):
                    continue
            filtered.append(o)
        return filtered

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