"""
Экспортёр планировок Sweet Home 3D - Полная переработка
Версия: 6.6
Автор: Расширение для системы анализа планировок
Дата: 02.06.2026

Изменения:
1. Стены с проёмами РЕАЛЬНО ПЕРЕМЕЩАЮТСЯ для выравнивания с соседними структурными стенами
2. Простой прямой подход: поиск ближайших стен, перемещение к их центральным линиям
3. Финальный проход расширяет стены для проникновения в ближайшие стеновые блоки
4. Корректное хранилище .sh3d файлов со встроенными OBJ моделями и иконками
"""

import math
import uuid
import zipfile
import base64
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from xml.etree import ElementTree as ET
from xml.dom import minidom


# ============================================================
# РЕСУРСЫ И КОНСТАНТЫ
# ============================================================


# ============================================================
# ASSETS & CONSTANTS
# ============================================================

DUMMY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="

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
# CONFIGURATION
# ============================================================

DEFAULT_WALL_HEIGHT = 243.84
DEFAULT_WALL_THICKNESS = 15.0
DEFAULT_DOOR_WIDTH = 91.44
DEFAULT_DOOR_HEIGHT = 208.5975
DEFAULT_WINDOW_WIDTH = 91.1225
DEFAULT_WINDOW_HEIGHT = 173.98999
DEFAULT_WINDOW_ELEVATION = 46.989998

SNAP_TOLERANCE = 15.0
PARENT_WALL_SEARCH_TOLERANCE = 50.0
WALL_EXTENSION_FACTOR = 0.05
WALL_OVERLAP = 2.0
BOX_PROXIMITY_TOLERANCE = 20.0
BOX_EXTENSION_AMOUNT = 5.0
CENTERLINE_SEARCH_RADIUS = 150.0
OPENING_TO_WALL_RATIO = 0.85


class OpeningType(Enum):
    DOOR = "door"
    WINDOW = "window"
    GAP = "gap"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Point:
    """2D точка на планировке"""
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __hash__(self):
        return hash((round(self.x, 2), round(self.y, 2)))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 0.01 and abs(self.y - other.y) < 0.01

    def copy(self) -> 'Point':
        return Point(self.x, self.y)

    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"


@dataclass
class Wall:
    """
    Представление стены для экспорта Sweet Home 3D.
    Содержит геометрические параметры, связи с другими стенами и метаданные.
    """
    start: Point
    end: Point
    thickness: float
    height: float = DEFAULT_WALL_HEIGHT
    wall_id: str = field(default_factory=lambda: f"wall-{uuid.uuid4()}")
    wall_at_start: Optional[str] = None
    wall_at_end: Optional[str] = None
    pattern: str = "hatchUp"
    is_structural: bool = True

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def midpoint(self) -> Point:
        return Point((self.start.x + self.end.x) / 2, (self.start.y + self.end.y) / 2)

    @property
    def angle(self) -> float:
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)

    @property
    def is_horizontal(self) -> bool:
        return abs(self.end.y - self.start.y) < abs(self.end.x - self.start.x)

    def direction_vector(self) -> Tuple[float, float]:
        length = self.length
        if length < 0.001:
            return (1.0, 0.0)
        return ((self.end.x - self.start.x) / length, (self.end.y - self.start.y) / length)

    def perpendicular_distance(self, point: Point) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = self.length
        if length < 0.001:
            return point.distance_to(self.start)
        return abs(dy * point.x - dx * point.y + self.end.x * self.start.y - self.end.y * self.start.x) / length

    def project_point(self, point: Point) -> Tuple[float, Point]:
        length_sq = self.length ** 2
        if length_sq < 0.001:
            return (0.0, self.start.copy())

        t = ((point.x - self.start.x) * (self.end.x - self.start.x) +
             (point.y - self.start.y) * (self.end.y - self.start.y)) / length_sq

        proj = Point(
            self.start.x + t * (self.end.x - self.start.x),
            self.start.y + t * (self.end.y - self.start.y)
        )
        return (t, proj)

    def get_min_x(self) -> float:
        return min(self.start.x, self.end.x)

    def get_max_x(self) -> float:
        return max(self.start.x, self.end.x)

    def get_min_y(self) -> float:
        return min(self.start.y, self.end.y)

    def get_max_y(self) -> float:
        return max(self.start.y, self.end.y)

    def get_centerline_y(self) -> float:
        return (self.start.y + self.end.y) / 2.0

    def get_centerline_x(self) -> float:
        return (self.start.x + self.end.x) / 2.0


@dataclass
class Opening:
    """
    Представление проёма (дверь или окно) в Sweet Home 3D.
    Включает позицию, размеры, тип и связь с родительской стеной.
    """
    opening_type: OpeningType
    center: Point
    width: float
    height: float
    depth: float
    angle: float
    elevation: float = 0.0
    parent_wall_id: Optional[str] = None
    opening_id: str = field(default_factory=lambda: f"opening-{uuid.uuid4()}")
    wall_thickness: float = 0.5
    wall_distance: float = 0.0
    model: str = ""
    icon: str = ""


@dataclass
class Room:
    points: List[Point]
    room_id: str = field(default_factory=lambda: f"room-{uuid.uuid4()}")
    name: str = "Room"


# ============================================================
# UTILITIES
# ============================================================

def parse_rectangle(rect: Dict) -> Tuple[float, float, float, float]:
    if all(k in rect for k in ['x1', 'y1', 'x2', 'y2']):
        return rect['x1'], rect['y1'], rect['x2'], rect['y2']
    elif all(k in rect for k in ['x', 'y', 'width', 'height']):
        return rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']
    elif all(k in rect for k in ['x', 'y', 'w', 'h']):
        return rect['x'], rect['y'], rect['x'] + rect['w'], rect['y'] + rect['h']
    else:
        raise ValueError(f"Unknown rectangle format: {rect.keys()}")


# ============================================================
# STRUCTURAL WALL CONVERTER
# ============================================================

class StructuralWallConverter:
    def __init__(self, pixels_to_cm: float = 2.54, min_thickness: float = 5.0):
        self.pixels_to_cm = pixels_to_cm
        self.min_thickness = min_thickness

    def convert(self, rectangles: List[Dict]) -> List[Wall]:
        walls = []

        for rect in rectangles:
            x1, y1, x2, y2 = parse_rectangle(rect)
            x1 *= self.pixels_to_cm
            y1 *= self.pixels_to_cm
            x2 *= self.pixels_to_cm
            y2 *= self.pixels_to_cm

            width = abs(x2 - x1)
            height = abs(y2 - y1)

            if width >= height:
                thickness = max(height, self.min_thickness)
                center_y = (y1 + y2) / 2
                start = Point(min(x1, x2), center_y)
                end = Point(max(x1, x2), center_y)
            else:
                thickness = max(width, self.min_thickness)
                center_x = (x1 + x2) / 2
                start = Point(center_x, min(y1, y2))
                end = Point(center_x, max(y1, y2))

            wall = Wall(
                start=start,
                end=end,
                thickness=thickness,
                is_structural=True
            )
            walls.append(wall)

        return walls


# ============================================================
# THICKNESS NORMALIZER
# ============================================================

class ThicknessNormalizer:
    def __init__(self, structural_walls: List[Wall], search_radius: float = 100.0):
        self.structural_walls = [w for w in structural_walls if w.is_structural]
        self.search_radius = search_radius

    def get_thickness_for_segment(self, start: Point, end: Point) -> float:
        midpoint = Point((start.x + end.x) / 2, (start.y + end.y) / 2)
        is_horizontal = abs(end.x - start.x) > abs(end.y - start.y)

        candidates = []
        for wall in self.structural_walls:
            perp_dist = wall.perpendicular_distance(midpoint)
            if perp_dist > self.search_radius:
                continue

            t, _ = wall.project_point(midpoint)
            if t < -0.5 or t > 1.5:
                continue

            orientation_match = (wall.is_horizontal == is_horizontal)
            priority = 0 if orientation_match else 1
            candidates.append((priority, perp_dist, wall.thickness))

        if not candidates:
            return DEFAULT_WALL_THICKNESS

        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][2]


# ============================================================
# PARENT WALL FINDER
# ============================================================

class ParentWallFinder:
    def __init__(self, walls: List[Wall], tolerance: float = PARENT_WALL_SEARCH_TOLERANCE):
        self.walls = walls
        self.tolerance = tolerance

    def find_parent(self, center: Point, opening_width: float, is_horizontal: bool) -> Optional[Wall]:
        best_wall = None
        best_distance = float('inf')

        for wall in self.walls:
            if wall.is_horizontal != is_horizontal:
                continue

            margin = (opening_width / 2) / wall.length if wall.length > 0 else 0.5
            t, proj = wall.project_point(center)

            if t < -margin or t > 1 + margin:
                continue

            perp_dist = center.distance_to(proj)

            if perp_dist < self.tolerance and perp_dist < best_distance:
                best_distance = perp_dist
                best_wall = wall

        return best_wall


# ============================================================
# OPENING PROCESSOR
# ============================================================

class OpeningProcessor:
    def __init__(self, pixels_to_cm: float = 2.54):
        self.pixels_to_cm = pixels_to_cm

    def process(
        self,
        structural_walls: List[Wall],
        door_rects: Optional[List[Dict]] = None,
        window_rects: Optional[List[Dict]] = None,
        gap_rects: Optional[List[Dict]] = None
    ) -> Tuple[List[Wall], List[Opening]]:
        parent_finder = ParentWallFinder(structural_walls)
        thickness_normalizer = ThicknessNormalizer(structural_walls)

        new_walls = []
        openings = []
        processed_centers: List[Point] = []

        all_rects = []
        if window_rects:
            all_rects.extend([(r, OpeningType.WINDOW) for r in window_rects])
        if gap_rects:
            all_rects.extend([(r, OpeningType.GAP) for r in gap_rects])
        if door_rects:
            all_rects.extend([(r, OpeningType.DOOR) for r in door_rects])

        for rect, opening_type in all_rects:
            result = self._process_single(
                rect, opening_type, parent_finder, thickness_normalizer
            )

            if result is None:
                continue

            opening, wall = result

            is_duplicate = False
            for prev_center in processed_centers:
                if opening.center.distance_to(prev_center) < 20.0:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            processed_centers.append(opening.center)
            openings.append(opening)
            if wall:
                new_walls.append(wall)

        return new_walls, openings

    def _process_single(
        self,
        rect: Dict,
        opening_type: OpeningType,
        parent_finder: ParentWallFinder,
        thickness_normalizer: ThicknessNormalizer
    ) -> Optional[Tuple[Opening, Optional[Wall]]]:

        x1, y1, x2, y2 = parse_rectangle(rect)
        x1 *= self.pixels_to_cm
        y1 *= self.pixels_to_cm
        x2 *= self.pixels_to_cm
        y2 *= self.pixels_to_cm

        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        is_horizontal = bbox_width >= bbox_height

        if opening_type == OpeningType.WINDOW:
            opening_width = DEFAULT_WINDOW_WIDTH
            opening_height = DEFAULT_WINDOW_HEIGHT
            elevation = DEFAULT_WINDOW_ELEVATION
            wall_thickness_param = 0.7352941
            wall_distance_param = 0.0
            model = "3/window85x163/window85x163.obj"
            icon = "2"
        elif opening_type == OpeningType.GAP:
            # GAP (геометрически обнаруженные окна) используют параметры дверей для стен
            opening_width = DEFAULT_WINDOW_WIDTH
            opening_height = DEFAULT_WINDOW_HEIGHT
            elevation = DEFAULT_WINDOW_ELEVATION
            wall_thickness_param = 0.51724136  # Как у дверей
            wall_distance_param = 0.06896552   # Как у дверей
            model = "3/window85x163/window85x163.obj"
            icon = "2"
        else:  # DOOR
            opening_width = DEFAULT_DOOR_WIDTH
            opening_height = DEFAULT_DOOR_HEIGHT
            elevation = 0.0
            wall_thickness_param = 0.51724136
            wall_distance_param = 0.06896552
            model = "1/door/door.obj"
            icon = "0"

        parent_wall = parent_finder.find_parent(center, max(bbox_width, bbox_height), is_horizontal)

        new_wall = None
        if parent_wall:
            parent_wall_id = parent_wall.wall_id
            depth = parent_wall.thickness
            angle = parent_wall.angle
        else:
            thickness = thickness_normalizer.get_thickness_for_segment(
                Point(x1, (y1 + y2) / 2) if is_horizontal else Point((x1 + x2) / 2, y1),
                Point(x2, (y1 + y2) / 2) if is_horizontal else Point((x1 + x2) / 2, y2)
            )

            if is_horizontal:
                extent = bbox_width * (1 + WALL_EXTENSION_FACTOR * 2)
                half_extent = extent / 2
                start = Point(center.x - half_extent, center.y)
                end = Point(center.x + half_extent, center.y)
                angle = 0.0
            else:
                extent = bbox_height * (1 + WALL_EXTENSION_FACTOR * 2)
                half_extent = extent / 2
                start = Point(center.x, center.y - half_extent)
                end = Point(center.x, center.y + half_extent)
                angle = math.pi / 2

            new_wall = Wall(
                start=start,
                end=end,
                thickness=thickness,
                is_structural=False
            )
            parent_wall_id = new_wall.wall_id
            depth = thickness

        opening = Opening(
            opening_type=opening_type,
            center=center,
            width=opening_width,
            height=opening_height,
            depth=depth,
            angle=angle,
            elevation=elevation,
            parent_wall_id=parent_wall_id,
            wall_thickness=wall_thickness_param,
            wall_distance=wall_distance_param,
            model=model,
            icon=icon
        )

        return (opening, new_wall)


# ============================================================
# ROOM DETECTOR
# ============================================================

class RoomDetector:
    def __init__(self, tolerance: float = 1.0):
        self.tolerance = tolerance

    def detect(self, walls: List[Wall]) -> List[Room]:
        if len(walls) < 3:
            return []

        graph = self._build_graph(walls)
        if not graph:
            return []

        cycles = self._find_minimal_cycles(graph)

        rooms = []
        for i, cycle in enumerate(cycles):
            if len(cycle) >= 3:
                points = [Point(x, y) for x, y in cycle]
                rooms.append(Room(points=points, name=f"Room {i + 1}"))

        return rooms

    def _build_graph(self, walls: List[Wall]) -> Dict[Tuple[float, float], Set[Tuple[float, float]]]:
        graph = defaultdict(set)
        for wall in walls:
            start_key = (round(wall.start.x, 1), round(wall.start.y, 1))
            end_key = (round(wall.end.x, 1), round(wall.end.y, 1))
            if start_key != end_key:
                graph[start_key].add(end_key)
                graph[end_key].add(start_key)
        return graph

    def _find_minimal_cycles(self, graph: Dict) -> List[List[Tuple[float, float]]]:
        cycles = []

        def dfs(start, current, path, depth):
            if depth > 12:
                return
            for neighbor in graph[current]:
                edge = tuple(sorted([current, neighbor]))
                if neighbor == start and len(path) >= 3:
                    cycles.append(path[:])
                elif neighbor not in path:
                    path.append(neighbor)
                    dfs(start, neighbor, path, depth + 1)
                    path.pop()

        for start_node in graph:
            dfs(start_node, start_node, [start_node], 0)

        return self._remove_duplicate_cycles(cycles)

    def _remove_duplicate_cycles(self, cycles: List[List]) -> List[List]:
        unique = []
        seen = set()
        for cycle in cycles:
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            reversed_cycle = list(reversed(cycle))
            min_idx_rev = reversed_cycle.index(min(reversed_cycle))
            normalized_rev = tuple(reversed_cycle[min_idx_rev:] + reversed_cycle[:min_idx_rev])
            key = min(normalized, normalized_rev)
            if key not in seen:
                seen.add(key)
                unique.append(cycle)
        return unique


# ============================================================
# XML GENERATOR
# ============================================================

class SH3DXMLGenerator:
    def __init__(self, wall_height: float = DEFAULT_WALL_HEIGHT):
        self.wall_height = wall_height

    def generate(
        self,
        walls: List[Wall],
        openings: List[Opening],
        rooms: List[Room],
        name: str = "floor_plan"
    ) -> str:
        root = ET.Element('home')
        root.set('version', '7400')
        root.set('name', name)
        root.set('camera', 'topCamera')
        root.set('wallHeight', str(self.wall_height))

        # Properties
        prop = ET.SubElement(root, 'property')
        prop.set('name', 'com.eteks.sweethome3d.SweetHome3D.PlanScale')
        prop.set('value', '0.02')

        env = ET.SubElement(root, 'environment')
        env.set('groundColor', '00A8A8A8')
        env.set('skyColor', '00CCE4FC')
        env.set('lightColor', '00D0D0D0')

        camera = ET.SubElement(root, 'camera')
        camera.set('attribute', 'topCamera')
        camera.set('lens', 'PINHOLE')
        camera.set('x', '500.0')
        camera.set('y', '500.0')
        camera.set('z', '1500.0')
        camera.set('yaw', '0.0')
        camera.set('pitch', '1.4')
        camera.set('fieldOfView', '1.0')

        for room in rooms:
            self._add_room(root, room)

        for opening in openings:
            self._add_opening(root, opening)

        for wall in walls:
            self._add_wall(root, wall)

        xml_str = ET.tostring(root, encoding='unicode')
        return xml_str

    def _add_wall(self, root: ET.Element, wall: Wall):
        elem = ET.SubElement(root, 'wall')
        elem.set('id', wall.wall_id)
        elem.set('xStart', f"{wall.start.x:.4f}")
        elem.set('yStart', f"{wall.start.y:.4f}")
        elem.set('xEnd', f"{wall.end.x:.4f}")
        elem.set('yEnd', f"{wall.end.y:.4f}")
        elem.set('height', str(wall.height))
        elem.set('thickness', f"{wall.thickness:.4f}")
        elem.set('pattern', wall.pattern)

        if wall.wall_at_start:
            elem.set('wallAtStart', wall.wall_at_start)
        if wall.wall_at_end:
            elem.set('wallAtEnd', wall.wall_at_end)

    def _add_opening(self, root: ET.Element, opening: Opening):
        elem = ET.SubElement(root, 'doorOrWindow')
        elem.set('id', opening.opening_id)

        if opening.opening_type in (OpeningType.DOOR, OpeningType.GAP):
            elem.set('catalogId', 'eteks-door')
            elem.set('name', 'Door')
        else:
            elem.set('catalogId', 'eteks-window85x163')
            elem.set('name', 'Window')

        elem.set('creator', 'eTeks')
        elem.set('model', opening.model)
        elem.set('icon', opening.icon)
        elem.set('x', f"{opening.center.x:.4f}")
        elem.set('y', f"{opening.center.y:.4f}")

        if opening.elevation > 0:
            elem.set('elevation', str(opening.elevation))

        elem.set('angle', str(opening.angle))
        elem.set('width', str(opening.width))
        elem.set('depth', f"{opening.depth:.4f}")
        elem.set('height', str(opening.height))
        elem.set('movable', 'false')

        elem.set('wallThickness', str(opening.wall_thickness))
        elem.set('wallDistance', str(opening.wall_distance))
        elem.set('cutOutShape', 'M0,0 v1 h1 v-1 z')
        elem.set('wallCutOutOnBothSides', 'true')

        sash = ET.SubElement(elem, 'sash')
        if opening.opening_type in (OpeningType.DOOR, OpeningType.GAP):
            sash.set('xAxis', '0.05464481')
            sash.set('yAxis', '0.5862069')
            sash.set('width', '0.89071035')
        else:
            sash.set('xAxis', '0.021978023')
            sash.set('yAxis', '0.7352941')
            sash.set('width', '0.94505495')
        sash.set('startAngle', '0.0')
        sash.set('endAngle', '-1.5707964')

    def _add_room(self, root: ET.Element, room: Room):
        elem = ET.SubElement(root, 'room')
        elem.set('id', room.room_id)
        elem.set('name', room.name)
        elem.set('areaVisible', 'true')
        elem.set('floorColor', '-3355444')

        for point in room.points:
            pt = ET.SubElement(elem, 'point')
            pt.set('x', f"{point.x:.4f}")
            pt.set('y', f"{point.y:.4f}")


# ============================================================
# MAIN EXPORTER
# ============================================================

class SweetHome3DExporter:
    """
    Основной класс экспортёра в формат Sweet Home 3D.

    Функциональность:
    - Преобразование сегментов стен в геометрию Sweet Home 3D
    - Создание и позиционирование проёмов (дверей/окон)
    - Выравнивание стен с проёмами относительно структурных стен
    - Автоматическое соединение стен в углах
    - Генерация XML для .sh3d файлов
    - Упаковка всех ресурсов в итоговый .sh3d архив
    """
    """
    Sweet Home 3D Floor Plan Exporter - Version 6.6
    """

    def __init__(
        self,
        min_wall_thickness: float = 5.0,
        pixels_to_cm: float = 2.54,
        wall_height_cm: float = DEFAULT_WALL_HEIGHT,
        snap_tolerance: float = SNAP_TOLERANCE,
        wall_overlap: float = WALL_OVERLAP
    ):
        self.pixels_to_cm = pixels_to_cm
        self.wall_height = wall_height_cm
        self.snap_tolerance = snap_tolerance
        self.wall_overlap = wall_overlap

        self.wall_converter = StructuralWallConverter(pixels_to_cm, min_wall_thickness)
        self.opening_processor = OpeningProcessor(pixels_to_cm)
        self.xml_generator = SH3DXMLGenerator(wall_height_cm)

    def export_to_sh3d(
        self,
        wall_rectangles: List[Dict],
        output_path: str,
        door_rectangles: Optional[List[Dict]] = None,
        window_rectangles: Optional[List[Dict]] = None,
        gap_rectangles: Optional[List[Dict]] = None,
        **kwargs
    ):
        # Step 1: Convert structural walls
        structural_walls = self.wall_converter.convert(wall_rectangles)

        # Step 2: Process openings - creates opening walls
        opening_walls, openings = self.opening_processor.process(
            structural_walls,
            door_rects=door_rectangles,
            window_rects=window_rectangles,
            gap_rects=gap_rectangles
        )

        # Step 3: Combine all walls
        all_walls = structural_walls + opening_walls

        # Step 4: MOVE opening walls to align with adjacent structural walls
        self._move_opening_walls_to_structural_centerlines(opening_walls, structural_walls)

        # Step 5: Extend all walls to connect with overlaps
        self._extend_walls_to_connect(all_walls)

        # Step 6: Final pass - extend walls close to other wall boxes
        self._extend_walls_into_nearby_boxes(all_walls)

        # Step 7: Connect walls
        self._connect_walls(all_walls)

        # Step 8: Update openings to match moved walls
        self._update_openings_from_walls(openings, all_walls)

        # Step 9: Detect rooms
        room_detector = RoomDetector()
        rooms = room_detector.detect(all_walls)

        # Step 10: Generate XML
        xml_content = self.xml_generator.generate(
            walls=all_walls,
            openings=openings,
            rooms=rooms,
            name=output_path
        )

        # Step 11: Write file
        self._write_sh3d_file(output_path, xml_content)

    def _move_opening_walls_to_structural_centerlines(
            self,
            opening_walls: List[Wall],
            structural_walls: List[Wall]
    ):
        """
        Move each opening wall to snap between two structural walls that form
        the smallest non-zero gap that isn't already occupied by another wall.
        """
        horiz_structural = [w for w in structural_walls if w.is_horizontal]
        vert_structural = [w for w in structural_walls if not w.is_horizontal]

        for ow in opening_walls:
            if ow.length < 0.1:
                continue

            old_start = ow.start.copy()
            old_end = ow.end.copy()

            if ow.is_horizontal:
                # === HORIZONTAL OPENING WALL ===
                # Opening runs left-right, need to find two VERTICAL walls to snap between
                ow_y = ow.get_centerline_y()
                ow_center_x = (ow.get_min_x() + ow.get_max_x()) / 2

                # Find the best pair of vertical walls (left and right) with smallest gap
                best_gap = float('inf')
                best_left_x = None
                best_right_x = None

                # Get all vertical walls that could be candidates (near our Y level)
                left_candidates = []  # walls to the left of center
                right_candidates = []  # walls to the right of center

                for sw in vert_structural:
                    sw_x = sw.get_centerline_x()
                    sw_y_min = sw.get_min_y()
                    sw_y_max = sw.get_max_y()

                    # Check if this wall spans our Y level
                    if not (sw_y_min - CENTERLINE_SEARCH_RADIUS <= ow_y <= sw_y_max + CENTERLINE_SEARCH_RADIUS):
                        continue

                    dist_from_center = abs(sw_x - ow_center_x)
                    if dist_from_center > CENTERLINE_SEARCH_RADIUS:
                        continue

                    if sw_x < ow_center_x:
                        left_candidates.append((sw_x, sw))
                    else:
                        right_candidates.append((sw_x, sw))

                # Try all pairs of (left_wall, right_wall) to find smallest non-zero gap
                for left_x, left_wall in left_candidates:
                    for right_x, right_wall in right_candidates:
                        gap = right_x - left_x

                        # Must be positive (right is actually to the right of left)
                        if gap <= 0:
                            continue

                        # Skip if gap is too small (walls touching)
                        if gap < 5.0:
                            continue

                        # Check if this gap is already occupied by another wall
                        is_occupied = False
                        for other_ow in opening_walls:
                            if other_ow.wall_id == ow.wall_id:
                                continue
                            if not other_ow.is_horizontal:
                                continue
                            # Check if other opening wall is in this gap
                            other_y = other_ow.get_centerline_y()
                            if abs(other_y - ow_y) < 20.0:  # Same Y level
                                other_min_x = other_ow.get_min_x()
                                other_max_x = other_ow.get_max_x()
                                # Check overlap with gap
                                if other_min_x < right_x and other_max_x > left_x:
                                    is_occupied = True
                                    break

                        # Also check structural walls in the gap
                        if not is_occupied:
                            for sw in horiz_structural:
                                sw_y = sw.get_centerline_y()
                                if abs(sw_y - ow_y) < 20.0:
                                    sw_min_x = sw.get_min_x()
                                    sw_max_x = sw.get_max_x()
                                    # Check if structural wall fills this gap
                                    if sw_min_x <= left_x + 5 and sw_max_x >= right_x - 5:
                                        is_occupied = True
                                        break

                        if is_occupied:
                            continue

                        # This gap is valid - check if it's the smallest
                        if gap < best_gap:
                            best_gap = gap
                            best_left_x = left_x
                            best_right_x = right_x

                # Apply the best gap found
                if best_left_x is not None and best_right_x is not None:
                    ow.start.x = best_left_x
                    ow.end.x = best_right_x

                # Now find the best Y position (snap to nearest horizontal structural wall)
                best_y = None
                best_y_dist = CENTERLINE_SEARCH_RADIUS

                for sw in horiz_structural:
                    sw_y = sw.get_centerline_y()
                    dist = abs(sw_y - ow_y)

                    sw_x_min = sw.get_min_x()
                    sw_x_max = sw.get_max_x()

                    # Check if wall overlaps with our X range
                    has_overlap = not (ow.get_max_x() < sw_x_min - 20 or ow.get_min_x() > sw_x_max + 20)

                    if has_overlap and dist < best_y_dist:
                        best_y = sw_y
                        best_y_dist = dist

                if best_y is not None:
                    ow.start.y = best_y
                    ow.end.y = best_y

            else:
                # === VERTICAL OPENING WALL ===
                # Opening runs top-bottom, need to find two HORIZONTAL walls to snap between
                ow_x = ow.get_centerline_x()
                ow_center_y = (ow.get_min_y() + ow.get_max_y()) / 2

                # Find the best pair of horizontal walls (top and bottom) with smallest gap
                best_gap = float('inf')
                best_top_y = None
                best_bottom_y = None

                # Get all horizontal walls that could be candidates (near our X level)
                top_candidates = []  # walls above center (smaller Y)
                bottom_candidates = []  # walls below center (larger Y)

                for sw in horiz_structural:
                    sw_y = sw.get_centerline_y()
                    sw_x_min = sw.get_min_x()
                    sw_x_max = sw.get_max_x()

                    # Check if this wall spans our X level
                    if not (sw_x_min - CENTERLINE_SEARCH_RADIUS <= ow_x <= sw_x_max + CENTERLINE_SEARCH_RADIUS):
                        continue

                    dist_from_center = abs(sw_y - ow_center_y)
                    if dist_from_center > CENTERLINE_SEARCH_RADIUS:
                        continue

                    if sw_y < ow_center_y:
                        top_candidates.append((sw_y, sw))
                    else:
                        bottom_candidates.append((sw_y, sw))

                # Try all pairs of (top_wall, bottom_wall) to find smallest non-zero gap
                for top_y, top_wall in top_candidates:
                    for bottom_y, bottom_wall in bottom_candidates:
                        gap = bottom_y - top_y

                        # Must be positive
                        if gap <= 0:
                            continue

                        # Skip if gap is too small
                        if gap < 5.0:
                            continue

                        # Check if this gap is already occupied
                        is_occupied = False
                        for other_ow in opening_walls:
                            if other_ow.wall_id == ow.wall_id:
                                continue
                            if other_ow.is_horizontal:
                                continue
                            # Check if other opening wall is in this gap
                            other_x = other_ow.get_centerline_x()
                            if abs(other_x - ow_x) < 20.0:  # Same X level
                                other_min_y = other_ow.get_min_y()
                                other_max_y = other_ow.get_max_y()
                                if other_min_y < bottom_y and other_max_y > top_y:
                                    is_occupied = True
                                    break

                        # Also check structural walls in the gap
                        if not is_occupied:
                            for sw in vert_structural:
                                sw_x = sw.get_centerline_x()
                                if abs(sw_x - ow_x) < 20.0:
                                    sw_min_y = sw.get_min_y()
                                    sw_max_y = sw.get_max_y()
                                    if sw_min_y <= top_y + 5 and sw_max_y >= bottom_y - 5:
                                        is_occupied = True
                                        break

                        if is_occupied:
                            continue

                        # This gap is valid - check if it's the smallest
                        if gap < best_gap:
                            best_gap = gap
                            best_top_y = top_y
                            best_bottom_y = bottom_y

                # Apply the best gap found
                if best_top_y is not None and best_bottom_y is not None:
                    # Preserve start/end direction
                    if ow.start.y < ow.end.y:
                        ow.start.y = best_top_y
                        ow.end.y = best_bottom_y
                    else:
                        ow.start.y = best_bottom_y
                        ow.end.y = best_top_y

                # Now find the best X position (snap to nearest vertical structural wall)
                best_x = None
                best_x_dist = CENTERLINE_SEARCH_RADIUS

                for sw in vert_structural:
                    sw_x = sw.get_centerline_x()
                    dist = abs(sw_x - ow_x)

                    sw_y_min = sw.get_min_y()
                    sw_y_max = sw.get_max_y()

                    # Check if wall overlaps with our Y range
                    has_overlap = not (ow.get_max_y() < sw_y_min - 20 or ow.get_min_y() > sw_y_max + 20)

                    if has_overlap and dist < best_x_dist:
                        best_x = sw_x
                        best_x_dist = dist

                if best_x is not None:
                    ow.start.x = best_x
                    ow.end.x = best_x

    def _extend_walls_to_connect(
    self, walls: List[Wall]):
        """
        Расширение стен вдоль их направления для соединения с близлежащими стенами.

        Выполняется в несколько проходов для постепенного достижения соединений.
        Проверяет каждый конец стены и расширяет его при обнаружении близкой цели.

        Параметры:
            walls: Список стен для обработки
        """
        for _ in range(3):
            for wall in walls:
                if wall.length < 0.1:
                    continue

                dx, dy = wall.direction_vector()

                for endpoint_name in ('start', 'end'):
                    p = getattr(wall, endpoint_name)

                    if endpoint_name == 'start':
                        ext_dir = (-dx, -dy)
                    else:
                        ext_dir = (dx, dy)

                    best_target = None
                    best_dist = self.snap_tolerance

                    for other in walls:
                        if other.wall_id == wall.wall_id:
                            continue

                        result = self._calc_extension_to_wall(p, ext_dir, other)
                        if result is not None:
                            dist, target_pt = result
                            if dist < best_dist:
                                best_dist = dist
                                best_target = target_pt

                    if best_target is not None:
                        new_x = best_target.x + ext_dir[0] * self.wall_overlap
                        new_y = best_target.y + ext_dir[1] * self.wall_overlap

                        if endpoint_name == 'start':
                            wall.start = Point(new_x, new_y)
                        else:
                            wall.end = Point(new_x, new_y)

    def _calc_extension_to_wall(
        self,
        p: Point,
        direction: Tuple[float, float],
        target: Wall
    ) -> Optional[Tuple[float, Point]]:
        """Calculate extension from point p in direction to reach target wall."""
        dx, dy = direction

        t1x, t1y = target.start.x, target.start.y
        t2x, t2y = target.end.x, target.end.y
        tdx, tdy = t2x - t1x, t2y - t1y

        denom = dx * tdy - dy * tdx

        if abs(denom) < 1e-6:
            perp = target.perpendicular_distance(p)
            if perp < 5.0:
                d1 = (t1x - p.x) * dx + (t1y - p.y) * dy
                d2 = (t2x - p.x) * dx + (t2y - p.y) * dy
                candidates = []
                if d1 > 0.01:
                    candidates.append((d1, Point(t1x, t1y)))
                if d2 > 0.01:
                    candidates.append((d2, Point(t2x, t2y)))
                if candidates:
                    return min(candidates, key=lambda x: x[0])
            return None

        diff_x = t1x - p.x
        diff_y = t1y - p.y

        t = (diff_x * tdy - diff_y * tdx) / denom
        s = (diff_x * dy - diff_y * dx) / denom

        if t > 0.01 and -0.2 <= s <= 1.2:
            ix = p.x + dx * t
            iy = p.y + dy * t
            return (t, Point(ix, iy))

        return None

    def _extend_walls_into_nearby_boxes(self, walls: List[Wall]):
        """
        Final pass: if wall endpoint is close to another wall's thickness box,
        extend the wall to penetrate into that box.
        """
        for wall in walls:
            if wall.length < 0.1:
                continue

            dx, dy = wall.direction_vector()

            for endpoint_name in ('start', 'end'):
                p = getattr(wall, endpoint_name)

                if endpoint_name == 'start':
                    ext_dx, ext_dy = -dx, -dy
                else:
                    ext_dx, ext_dy = dx, dy

                for other in walls:
                    if other.wall_id == wall.wall_id:
                        continue

                    half_t = other.thickness / 2.0

                    if other.is_horizontal:
                        box_x_min = other.get_min_x()
                        box_x_max = other.get_max_x()
                        box_y_min = other.get_centerline_y() - half_t
                        box_y_max = other.get_centerline_y() + half_t
                    else:
                        box_x_min = other.get_centerline_x() - half_t
                        box_x_max = other.get_centerline_x() + half_t
                        box_y_min = other.get_min_y()
                        box_y_max = other.get_max_y()

                    dist_to_box = self._point_to_box_distance(p, box_x_min, box_y_min, box_x_max, box_y_max)

                    if 0.5 < dist_to_box < BOX_PROXIMITY_TOLERANCE:
                        extension = dist_to_box + BOX_EXTENSION_AMOUNT

                        new_x = p.x + ext_dx * extension
                        new_y = p.y + ext_dy * extension

                        if endpoint_name == 'start':
                            wall.start = Point(new_x, new_y)
                        else:
                            wall.end = Point(new_x, new_y)
                        break

    def _point_to_box_distance(
        self,
        p: Point,
        x_min: float, y_min: float,
        x_max: float, y_max: float
    ) -> float:
        """Calculate distance from point to axis-aligned box."""
        if x_min <= p.x <= x_max and y_min <= p.y <= y_max:
            return 0.0

        dx = max(x_min - p.x, 0, p.x - x_max)
        dy = max(y_min - p.y, 0, p.y - y_max)

        return math.hypot(dx, dy)

    def _connect_walls(
        self, walls: List[Wall]):
        """
        Установка связей между стенами (wall-to-wall connections).

        Определяет какие стены соединены в углах и обновляет атрибуты
        wall_at_start и wall_at_end для корректного рендеринга соединений.

        Параметры:
            walls: Список стен
        """
        eps = self.wall_overlap + 2.0

        for i, wall1 in enumerate(walls):
            for j, wall2 in enumerate(walls):
                if i >= j:
                    continue

                if wall1.end.distance_to(wall2.start) < eps:
                    wall1.wall_at_end = wall2.wall_id
                    wall2.wall_at_start = wall1.wall_id
                elif wall1.start.distance_to(wall2.end) < eps:
                    wall1.wall_at_start = wall2.wall_id
                    wall2.wall_at_end = wall1.wall_id
                elif wall1.end.distance_to(wall2.end) < eps:
                    wall1.wall_at_end = wall2.wall_id
                    wall2.wall_at_end = wall1.wall_id
                elif wall1.start.distance_to(wall2.start) < eps:
                    wall1.wall_at_start = wall2.wall_id
                    wall2.wall_at_start = wall1.wall_id

    def _update_openings_from_walls(self, openings: List[Opening], walls: List[Wall]):
        """Update opening positions based on final wall positions."""
        wall_by_id = {w.wall_id: w for w in walls}

        for opening in openings:
            parent = wall_by_id.get(opening.parent_wall_id)
            if not parent:
                continue

            # Move opening center to wall midpoint
            opening.center = parent.midpoint
            opening.angle = parent.angle

            # For non-structural walls (opening walls), make the opening object
            # slightly shorter than the wall
            if not parent.is_structural:
                opening.width = parent.length * OPENING_TO_WALL_RATIO

    def _write_sh3d_file(self, output_path: str, xml_content: str):
        if not output_path.endswith('.sh3d'):
            output_path += '.sh3d'

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('Home.xml', xml_content)
            zf.writestr('1/door/door.obj', DOOR_OBJ_CONTENT)
            zf.writestr('3/window85x163/window85x163.obj', WINDOW_OBJ_CONTENT)
            zf.writestr('0', base64.b64decode(DUMMY_PNG_B64))
            zf.writestr('2', base64.b64decode(DUMMY_PNG_B64))


# Backward compatibility
FixedSweetHome3DExporter = SweetHome3DExporter