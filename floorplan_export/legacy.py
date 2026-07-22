"""SweetHome3D exporter: flood-fill room detection, opening sizing, and SH3D ZIP packaging."""
import logging
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

logger = logging.getLogger(__name__)


def _point_in_polygon(x: float, y: float, pts: List["Point"]) -> bool:
    """Ray-casting point-in-polygon test. ``pts`` are Point objects (cm)."""
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i].x, pts[i].y
        xj, yj = pts[j].x, pts[j].y
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_area(pts: List["Point"]) -> float:
    """Shoelace area of a polygon given as Point objects (cm²)."""
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i].x * pts[j].y - pts[j].x * pts[i].y
    return abs(a) / 2.0


def _assign_room_names_from_ocr(
    rooms: List["Room"],
    ocr_labels: Optional[List],
    pixels_to_cm: float,
) -> int:
    """
    Name each detected room from the OCR label(s) falling inside it.

    ``ocr_labels`` are ``(text, x1, y1, x2, y2)`` tuples in the same
    (deskewed) image-pixel space as the walls; room polygons are in cm, so
    label centres are scaled by ``pixels_to_cm`` before the inside test.

    Preference: a non-numeric label (a real room name such as "Кухня") wins;
    otherwise the largest numeric value inside the room is taken as its area
    and formatted as "<area> м²".  Rooms with no enclosed label keep their
    default "Room N" name.  Returns the number of rooms (re)named.

    An apartment-total label (the printed sum of all room areas, e.g. a
    header "137,8 м²") is excluded up front rather than let it attach to
    whichever room happens to contain its centre: without this a room whose
    polygon overlaps the total's position picks it up as "the largest
    numeric value inside the room" and ships labeled with the whole
    apartment's size instead of its own. Detected the same way as the
    scale-calibration apartment-total defense (orchestrator.py
    ``_calibrate_scale_from_rooms``): the largest numeric value across the
    whole plan, when it is within ~12% of the sum of all the others.
    """
    if not rooms or not ocr_labels or not pixels_to_cm or pixels_to_cm <= 0:
        return 0
    try:
        from core.ocr_utils import parse_area_m2, is_numeric_ocr_label
    except Exception:
        return 0

    # Pre-pass: find the apartment-total label (if any) so it's never used
    # to size an individual room below.
    all_areas: List[float] = []
    for item in ocr_labels:
        if len(item) < 5:
            continue
        text = str(item[0]).strip()
        if text and is_numeric_ocr_label(text):
            v = parse_area_m2(text)
            if v is not None:
                all_areas.append(v)
    total_area: Optional[float] = None
    if len(all_areas) >= 3:
        vmax = max(all_areas)
        rest = sum(all_areas) - vmax
        if rest > 0 and abs(vmax - rest) <= 0.12 * max(vmax, rest):
            total_area = vmax

    assigned = 0
    for room in rooms:
        if len(room.points) < 3:
            continue
        name_text: Optional[str] = None
        best_area: Optional[float] = None
        for item in ocr_labels:
            if len(item) < 5:
                continue
            text = str(item[0]).strip()
            if not text:
                continue
            cx = ((float(item[1]) + float(item[3])) / 2.0) * pixels_to_cm
            cy = ((float(item[2]) + float(item[4])) / 2.0) * pixels_to_cm
            if not _point_in_polygon(cx, cy, room.points):
                continue
            if not is_numeric_ocr_label(text):
                # A real room name ("Кухня", "Спальня"), not OCR shrapnel:
                # watermark fragments and stray glyphs ('+', '€', 'Ba', 'L')
                # otherwise win the priority race and ship as room names.
                letters = sum(1 for ch in text if ch.isalpha())
                if letters >= 3 or text.lower() in ("wc", "су", "с/у", "c/y"):
                    name_text = text      # real room name — highest priority
                    break
                continue
            area_val = parse_area_m2(text)
            if area_val is None:
                continue
            if total_area is not None and abs(area_val - total_area) < 1e-6:
                continue  # apartment total — filtered, never a room's size
            if best_area is None or area_val > best_area:
                best_area = area_val
        if name_text is None and best_area is not None:
            name_text = f"{best_area:g} м²"
        if name_text:
            room.name = name_text
            assigned += 1
    logger.info("Room naming: assigned OCR names to %d/%d rooms",
                assigned, len(rooms))
    return assigned


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
    # True for rotated (non-axis-aligned) walls. The post-processing passes
    # (_align_walls_to_shared_axes, _extend_walls_along_axis,
    # _snap_endpoints_to_corners, ThicknessNormalizer, ParentWallFinder)
    # all assume axis-aligned geometry classified by is_horizontal; applying
    # them to a diagonal wall flattens it onto a single x or y row, which
    # SH3D then renders as a single massive wall. Diagonals are passed
    # through these passes unchanged.
    is_diagonal: bool = False
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
    source: str = ''


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
        try:
            from core.ocr_utils import is_numeric_ocr_label
            return is_numeric_ocr_label(text)
        except ImportError:
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
        # Skip diagonals: they're classified by is_horizontal off their AABB
        # and would distort thickness candidates for axis-aligned openings.
        self.structural_walls = [
            w for w in structural_walls
            if w.is_structural and not w.is_diagonal
        ]
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
            # Diagonals are never parents of an axis-aligned opening here:
            # is_horizontal is bbox-derived for them so this check would
            # spuriously match.
            if wall.is_diagonal:
                continue
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
        # wall_normal_deg points PERPENDICULAR to the wall: a horizontal wall
        # (door runs along x) has a vertical normal, so |sin| ≈ 1.
        is_horizontal = abs(math.sin(math.radians(fused_door.wall_normal_deg))) > 0.707

        ow = width_cm                       # ← actual detected gap width
        oh = DEFAULT_DOOR_HEIGHT
        elev = 0.0
        # wallThickness is a FRACTION of piece depth in SH3D semantics.
        # depth is set to the parent wall thickness below, so the wall
        # occupies the full piece depth (1.0) with no front offset (0.0).
        wt, wd = 1.0, 0.0
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

        # The piece angle follows the WALL direction (parent.angle when a
        # parent was found, axis angle otherwise) — never the wall normal,
        # which is perpendicular and would rotate the door 90° out of the wall.
        hinge_px = fused_door.hinge if fused_door.hinge else (cx, cy)

        opening = Opening(
            opening_type=OpeningType.DOOR,
            center=center,
            width=ow, height=oh, depth=depth,
            angle=angle, elevation=elev,
            parent_wall_id=pid,
            wall_thickness=wt, wall_distance=wd,
            model=model, icon=icon,
            hinge=hinge_px,
            swing_clockwise=fused_door.swing_clockwise,
            wall_normal_deg=fused_door.wall_normal_deg,
            source=getattr(fused_door, 'source', ''),
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

        # wallThickness is a FRACTION of piece depth in SH3D semantics; depth
        # is set to the parent wall thickness, so 1.0/0.0 makes the opening
        # cut the full wall with no front offset.
        if opening_type == OpeningType.WINDOW:
            ow = gap_width
            oh = DEFAULT_WINDOW_HEIGHT
            elev = DEFAULT_WINDOW_ELEVATION
            wt, wd = 1.0, 0.0
            model, icon = '3/window85x163/window85x163.obj', '2'
        elif opening_type == OpeningType.GAP:
            # A gap is a doorless passage: full-height cut, floor-level.
            ow = gap_width
            oh = DEFAULT_DOOR_HEIGHT
            elev = 0.0
            wt, wd = 1.0, 0.0
            model, icon = '1/door/door.obj', '0'
        else:
            ow = gap_width
            oh = DEFAULT_DOOR_HEIGHT
            elev = 0.0
            wt, wd = 1.0, 0.0
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
            source=str(rect.get('source', '')) if isinstance(rect, dict) else '',
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

    After polygon extraction the contour is snapped onto the walls
    edge-wise rather than vertex-wise: every polygon edge is matched to the
    wall face it runs along, the edge is replaced by that face's supporting
    line, and each corner is rebuilt as the intersection of its two
    neighbouring lines.  Room boundaries therefore sit exactly on the inner
    wall faces (airtight) at any wall angle, and corners shaved off by
    contour simplification are restored.
    """

    # Hard ceiling on raster area (px). 64M ≈ 8000×8000 keeps peak memory
    # ~1 GB through the distance-transform/watershed passes; normal plans
    # use <10M. Exceeding it coarsens self.resolution (see _rasterize_walls).
    MAX_RASTER_PX = 64_000_000

    def __init__(
        self,
        resolution: float = 0.5,        # finer raster → sharper corners
        min_room_area_cm2: float = 2000.0,
        wall_dilation_px: int = 4,
        simplify_epsilon_cm: float = 2.0,         # absolute, NOT % of perimeter
        snap_tolerance_cm: float = 18.0,          # max snap distance
        split_doorways: bool = True,
        max_doorway_cm: float = 110.0,
    ) -> None:
        self.resolution = resolution
        self.min_room_area_cm2 = min_room_area_cm2
        self.wall_dilation_px = wall_dilation_px
        self.simplify_epsilon_cm = simplify_epsilon_cm
        self.snap_tolerance_cm = snap_tolerance_cm
        self.split_doorways = split_doorways
        self.max_doorway_cm = max_doorway_cm

    # ---- public API ----
    def detect(
        self,
        walls: List[Wall],
        enclosed_labels: Optional[np.ndarray] = None,
        openings: Optional[List[Opening]] = None,
    ) -> List[Room]:
        if len(walls) < 3:
            return []

        # Pre-compute wall corner geometry once for snapping.  Edge snapping
        # matches against individual faces, so keep a flat list of them.
        self._wall_geom = self._compute_wall_geom(walls)
        self._wall_faces = [face for edges in self._wall_geom for face in edges]
        # Corner notches scale with wall thickness (see _snap_polygon_to_walls).
        self._max_wall_thickness = max((w.thickness for w in walls), default=0.0)

        mask, origin = self._rasterize_walls(walls)
        # Seal every DETECTED opening (door/window/gap) into the wall raster.
        # Opening-walls normally seal detected passages already, but a thin or
        # slightly-misplaced opening-wall can leave a 1-px leak; painting the
        # full opening footprint here guarantees a crisp cut.
        if openings:
            self._seal_openings(mask, openings, origin)
        # Separate rooms joined through UNDETECTED doorways — including
        # doorways to the OUTSIDE (entrance, stairs), which is why the neck
        # cut must run on the full non-wall space before interior extraction
        # rather than as an interior-only post-pass.
        if self.split_doorways:
            # Derive the doorway width from the detected doors themselves:
            # both door widths and room geometry come from px × pixels_to_cm,
            # so the ratio survives a broken scale calibration where the
            # fixed max_doorway_cm (real-world cm) does not.
            door_widths = [
                float(o.width) for o in (openings or [])
                if o.opening_type in (OpeningType.DOOR, OpeningType.GAP)
                and o.width > 0
            ]
            if len(door_widths) >= 2:
                # Never go BELOW the config value: junk gap detections can be
                # narrow slits (plan 1: 26 cm median → 39 cm neck would make
                # real ~90 cm doorways unsplittable). The derived value only
                # WIDENS the neck when detected doors are larger than the
                # config assumes (the broken-large-scale case).
                derived = max(self.max_doorway_cm,
                              1.5 * float(np.median(door_widths)))
                logger.info(
                    "RoomDetector: max_doorway derived from %d door(s): "
                    "%.0f cm (config %.0f cm)",
                    len(door_widths), derived, self.max_doorway_cm,
                )
                self.max_doorway_cm = derived
            interior = self._extract_interior_with_neck_cut(mask)
        else:
            interior = self._extract_interior(mask)
        rooms = self._components_to_rooms(interior, origin, enclosed_labels, mask)
        logger.info(
            "RoomDetector: %d room(s) from flood-fill (split_doorways=%s, "
            "max_doorway_cm=%.0f, %d sealed opening(s))",
            len(rooms), self.split_doorways, self.max_doorway_cm,
            len(openings or []),
        )
        return rooms

    # ---- seal detected openings into the wall raster ----
    def _seal_openings(
        self,
        mask: np.ndarray,
        openings: List[Opening],
        origin: Tuple[float, float],
    ) -> None:
        """Paint each opening's footprint as wall so its passage is sealed.

        The opening is a rectangle centred at ``opening.center``, ``width`` long
        along the wall direction (``angle``) and ``depth`` (parent wall
        thickness) across it — exactly the doorway hole in the wall.
        """
        min_x, min_y = origin
        res = self.resolution
        h_px, w_px = mask.shape[:2]
        for op in openings:
            try:
                cx, cy = op.center.x, op.center.y
                length = float(op.width)
                thick = float(op.depth) if op.depth else 0.0
                if length <= 0:
                    continue
                # A doorway must be blocked across the full wall thickness; use a
                # generous floor so a thin/under-measured depth still seals.
                thick = max(thick, 8.0)
                ang = float(op.angle)
                dx, dy = math.cos(ang), math.sin(ang)      # along wall
                nx, ny = -dy, dx                            # across wall
                hl, ht = length / 2.0, thick / 2.0
                corners_cm = [
                    (cx + dx * hl + nx * ht, cy + dy * hl + ny * ht),
                    (cx + dx * hl - nx * ht, cy + dy * hl - ny * ht),
                    (cx - dx * hl - nx * ht, cy - dy * hl - ny * ht),
                    (cx - dx * hl + nx * ht, cy - dy * hl + ny * ht),
                ]
                pts = np.array(
                    [[int((x - min_x) / res), int((y - min_y) / res)]
                     for x, y in corners_cm],
                    dtype=np.int32,
                )
                if pts[:, 0].max() < 0 or pts[:, 1].max() < 0:
                    continue
                if pts[:, 0].min() >= w_px or pts[:, 1].min() >= h_px:
                    continue
                cv2.fillConvexPoly(mask, pts, 255)
            except Exception:
                continue

    # ---- split rooms merged through undetected doorways ----
    def _split_at_necks(self, interior: np.ndarray) -> np.ndarray:
        """Cut narrow doorway "necks" so merged rooms become separate blobs.

        A doorway with no detected door leaves the two rooms it joins connected
        by a channel roughly one door-width wide.  Eroding the interior by half
        the max doorway width detaches every such channel while leaving the much
        larger room bodies as separate "cores"; a watershed then re-grows the
        cores back to the wall faces and the ridge lines between neighbouring
        cores become the cut.  Channels wider than ``max_doorway_cm`` (genuine
        open-plan openings) survive erosion joined, so they are NOT split.
        """
        res = self.resolution
        neck_px = (self.max_doorway_cm / res) / 2.0
        if neck_px < 1:
            return interior

        # "Erode by neck_px" == keep pixels farther than neck_px from any wall.
        # The distance transform gives this in O(n), vastly faster than a
        # morphological erode with a ~neck_px-radius kernel on a large raster.
        dist = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
        cores = (dist > neck_px).astype(np.uint8) * 255

        n_cores, core_lbls = cv2.connectedComponents(cores)
        # n_cores counts the background label; need >=2 real cores to split.
        if n_cores <= 2:
            return interior

        # Watershed markers: 1..n for cores, a separate label for exterior,
        # 0 for the unknown band (the eroded-away channels + room rims).
        markers = core_lbls.astype(np.int32).copy()
        bg_label = n_cores
        markers[interior == 0] = bg_label

        img3 = cv2.cvtColor(interior, cv2.COLOR_GRAY2BGR)
        cv2.watershed(img3, markers)

        # Watershed marks the basin boundary as a 1-px ridge (-1).  A single-pixel
        # gap still leaks under 8-connectivity, so thicken the ridge to guarantee
        # the two rooms become separate connected components.
        ridge = (markers == -1).astype(np.uint8)
        ridge = cv2.dilate(ridge, np.ones((3, 3), np.uint8), iterations=2)
        cut = interior.copy()
        cut[ridge > 0] = 0
        return cut

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

    # An edge whose direction is within this of a wall face counts as running
    # along it.  The raster staircase plus approxPolyDP tilt edges by a few
    # degrees even when the underlying wall is perfectly straight.
    _PARALLEL_TOL_DEG = 20.0
    # Edges this short are raster noise — a single staircase step.  Their
    # direction says nothing, so they inherit a face from their neighbours
    # rather than voting for one of their own.
    _NOISE_EDGE_CM = 4.0

    def _match_face(self, p: Point, q: Point) -> Optional[int]:
        """Index of the wall face that edge ``p→q`` runs along, or None.

        A face qualifies when it is near-parallel to the edge, both edge
        endpoints lie within ``snap_tolerance_cm`` of the face's supporting
        line, and the edge overlaps the face's extent.  Among the qualifiers
        the closest (and most parallel) wins, which is what keeps an interior
        edge on the wall's INNER face: the outer face of the same wall is a
        further whole wall-thickness away.
        """
        ex, ey = q.x - p.x, q.y - p.y
        elen = math.hypot(ex, ey)
        if elen < self._NOISE_EDGE_CM:
            return None
        ux, uy = ex / elen, ey / elen
        mx, my = (p.x + q.x) / 2.0, (p.y + q.y) / 2.0

        tol = self.snap_tolerance_cm
        sin_tol = math.sin(math.radians(self._PARALLEL_TOL_DEG))
        best_score, best_idx = float('inf'), None
        for idx, ((ax, ay), (bx, by)) in enumerate(self._wall_faces):
            fx, fy = bx - ax, by - ay
            flen = math.hypot(fx, fy)
            if flen < 1e-6:
                continue
            vx, vy = fx / flen, fy / flen
            # Undirected parallelism: a face may run either way along the edge.
            sin_a = abs(ux * vy - uy * vx)
            if sin_a > sin_tol:
                continue
            # Perpendicular distance of both endpoints to the face's line.
            d_p = abs((p.x - ax) * vy - (p.y - ay) * vx)
            d_q = abs((q.x - ax) * vy - (q.y - ay) * vx)
            if max(d_p, d_q) > tol:
                continue
            # The edge must actually sit alongside the face, not past its end.
            t = (mx - ax) * vx + (my - ay) * vy
            if t < -tol or t > flen + tol:
                continue
            score = 0.5 * (d_p + d_q) + sin_a * tol
            if score < best_score:
                best_score, best_idx = score, idx
        return best_idx

    def _face_line(self, face_id: int) -> Optional[Tuple[float, float, float, float]]:
        (ax, ay), (bx, by) = self._wall_faces[face_id]
        return self._line_through(Point(ax, ay), Point(bx, by))

    def _cap_line(self, face_id: int, near: Point):
        """Supporting line of the end cap of ``face_id``'s wall nearest ``near``.

        ``_compute_wall_geom`` emits exactly four faces per wall in a fixed
        order — 0 and 2 are the long sides, 1 and 3 the end caps — so the wall
        and its caps are recoverable from a face index alone.
        """
        if face_id % 4 not in (0, 2):
            return None                        # neighbour is itself a cap
        edges = self._wall_geom[face_id // 4]
        best, best_d = None, float('inf')
        for cap in (edges[1], edges[3]):
            (ax, ay), (bx, by) = cap
            d = math.hypot((ax + bx) / 2.0 - near.x, (ay + by) / 2.0 - near.y)
            if d < best_d:
                best, best_d = cap, d
        (ax, ay), (bx, by) = best
        return self._line_through(Point(ax, ay), Point(bx, by))

    # Two parallel face lines closer than this are the same line — a wall seen
    # from both sides of a doorway, rather than a step between two faces.
    _COLLINEAR_TOL_CM = 2.0

    def _chain_ok(self, seq, mid: Point, sin_corner: float, max_shift: float) -> bool:
        """Every consecutive pair in ``seq`` must meet at a corner near ``mid``."""
        for u, v in zip(seq, seq[1:]):
            if u is None or v is None:
                return False
            if abs(u[2] * v[3] - u[3] * v[2]) < sin_corner:
                return False
            pt = self._intersect(u, v)
            if pt is None or mid.distance_to(pt) > max_shift:
                return False
        return True

    def _classify_free_run(
        self,
        runs: List[Tuple[Optional[int], List[int]]],
        k: int,
        points: List[Point],
        n: int,
        notch_max: float,
        sin_corner: float,
        max_shift: float,
    ):
        """Classify free run ``k`` as 'convex', 'bridge' or 'cut'.

        Returns ``(kind, lines)``; ``lines`` are the end-cap lines to emit in
        place of a 'bridge' run, and is empty otherwise.
        """
        prev_face = runs[k - 1][0]
        next_face = runs[(k + 1) % len(runs)][0]
        if prev_face is None or next_face is None:
            return 'cut', []

        idxs = runs[k][1]
        a = points[idxs[0]]
        b = points[(idxs[-1] + 1) % n]
        if a.distance_to(b) > notch_max:
            return 'cut', []          # too long to be a notch (e.g. a doorway)

        l0, l1 = self._face_line(prev_face), self._face_line(next_face)
        if l0 is None or l1 is None:
            return 'cut', []
        mid = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

        if abs(l0[2] * l1[3] - l0[3] * l1[2]) < sin_corner:
            # Parallel faces.  Collinear ones are the same wall either side of a
            # doorway — the watershed cut between them must stay put.  Offset
            # ones are a step between two faces and need a cap to bridge them.
            offset = abs((l1[0] - l0[0]) * l0[3] - (l1[1] - l0[1]) * l0[2])
            if offset < self._COLLINEAR_TOL_CM:
                return 'cut', []
        else:
            corner = self._intersect(l0, l1)
            # A mitre point OUTSIDE the room means the faces cross beyond the
            # notch, so mitring adds it to the floor.  INSIDE means they cross
            # within the room and mitring would slice the notch off — that is a
            # reflex corner, and the boundary must instead detour along the
            # walls' end caps, which the bridge search below reconstructs.
            if (corner is not None
                    and mid.distance_to(corner) <= max_shift
                    and not _point_in_polygon(corner.x, corner.y, points)):
                return 'convex', []

        # Bridge the two faces with the end cap(s) of their walls: both caps for
        # a reflex corner, a single cap for a step onto an already-matched cap.
        cap_p = self._cap_line(prev_face, mid)
        cap_n = self._cap_line(next_face, mid)
        for bridge in ([cap_p, cap_n], [cap_p], [cap_n]):
            if any(c is None for c in bridge):
                continue
            if self._chain_ok([l0] + bridge + [l1], mid, sin_corner, max_shift):
                return 'bridge', bridge
        return 'cut', []

    @staticmethod
    def _line_through(p: Point, q: Point) -> Optional[Tuple[float, float, float, float]]:
        """Line as (point_x, point_y, dir_x, dir_y) with a unit direction."""
        dx, dy = q.x - p.x, q.y - p.y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            return None
        return (p.x, p.y, dx / d, dy / d)

    @staticmethod
    def _intersect(l0, l1) -> Optional[Point]:
        """Intersection of two lines, or None when they are near-parallel."""
        if l0 is None or l1 is None:
            return None
        x0, y0, dx0, dy0 = l0
        x1, y1, dx1, dy1 = l1
        denom = dx0 * dy1 - dy0 * dx1
        if abs(denom) < 1e-6:          # parallel: no stable corner
            return None
        t = ((x1 - x0) * dy1 - (y1 - y0) * dx1) / denom
        return Point(x0 + t * dx0, y0 + t * dy0)

    def _snap_polygon_to_walls(self, points: List[Point]) -> List[Point]:
        """Snap a room contour onto the wall faces edge-wise.

        Vertex-wise snapping cannot make a corner airtight: a corner vertex has
        two wall faces meeting at it, and projecting the vertex onto whichever
        face happens to be nearest leaves it inset from the other one.  Instead
        each EDGE is matched to the face it runs along and replaced by that
        face's supporting line; each corner is then rebuilt as the intersection
        of its two neighbouring lines.  The corner lands exactly where the two
        wall faces meet, at any wall angle, even if contour simplification had
        shaved the corner off entirely.

        Edges matching no face (e.g. the watershed cut across a doorway) keep
        their own chord as their line, so they are left where they were.
        """
        n = len(points)
        if n < 3 or not self._wall_faces:
            return points

        # 1. Assign each edge i (points[i] → points[i+1]) the face it lies on.
        face_of: List[Optional[int]] = [
            self._match_face(points[i], points[(i + 1) % n]) for i in range(n)
        ]

        # 2. A noise-length edge wedged between two edges of the same face is
        #    a staircase step on that face — absorb it so it can't break the run.
        for _ in range(2):
            for i in range(n):
                if face_of[i] is not None:
                    continue
                p, q = points[i], points[(i + 1) % n]
                if math.hypot(q.x - p.x, q.y - p.y) >= self._NOISE_EDGE_CM:
                    continue
                prv, nxt = face_of[i - 1], face_of[(i + 1) % n]
                if prv is not None and prv == nxt:
                    face_of[i] = prv

        # 3. Group consecutive edges sharing a face into runs (all consecutive
        #    unmatched edges collapse into one free run).
        start = None
        for i in range(n):
            if face_of[i] != face_of[i - 1]:
                start = i
                break
        if start is None:
            return points          # every edge on one face — degenerate

        runs: List[Tuple[Optional[int], List[int]]] = []
        cur_face, cur = face_of[start], [start]
        for k in range(1, n):
            i = (start + k) % n
            if face_of[i] == cur_face:
                cur.append(i)
            else:
                runs.append((cur_face, cur))
                cur_face, cur = face_of[i], [i]
        runs.append((cur_face, cur))
        if len(runs) < 3:
            return points          # can't rebuild a polygon from <3 lines

        # 4. Turn every run into one or more supporting lines.  A face run gives
        #    its face's line.  A free run is resolved by what it actually is:
        #      convex notch — walls are not mitred, so at a corner their inner
        #        faces stop short of one another.  Drop the run and let the two
        #        face lines intersect: the floor runs into the notch.
        #      reflex notch / step — there mitring would slice the notch off (the
        #        face lines cross inside the room), or the faces are parallel but
        #        offset.  Either way the boundary detours around the missing wall
        #        corner along the walls' end caps, so emit those cap lines.
        #      doorway cut — the watershed cut across an opening.  It is flanked
        #        by COLLINEAR faces (one wall, either side of the door) forming
        #        no corner, so it keeps its own chord and stays put.
        max_shift = 2.0 * self.snap_tolerance_cm
        notch_max = min(1.6 * self._max_wall_thickness, 0.5 * self.max_doorway_cm)
        sin_corner = math.sin(math.radians(10.0))

        lines = []
        anchors: List[Point] = []      # a nearby raster vertex per line, for fallback
        for k, (face_id, idxs) in enumerate(runs):
            if face_id is not None:
                lines.append(self._face_line(face_id))
                anchors.append(points[idxs[0]])
                continue
            a = points[idxs[0]]
            b = points[(idxs[-1] + 1) % n]
            mid = Point((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)
            kind, bridge = self._classify_free_run(
                runs, k, points, n, notch_max, sin_corner, max_shift
            )
            if kind == 'convex':
                continue                       # drop → neighbours mitre
            if kind == 'bridge':
                lines.extend(bridge)           # end cap(s) around the wall corner
                anchors.extend([mid] * len(bridge))
                continue
            lines.append(self._line_through(a, b))
            anchors.append(a)
        if len(lines) < 3:
            return points

        # 5. Each corner = intersection of the two lines meeting there.
        snapped: List[Point] = []
        for k in range(len(lines)):
            original = anchors[k]
            corner = self._intersect(lines[k - 1], lines[k])
            # Near-parallel lines give a corner far away (or none): keep the
            # original vertex rather than flinging it across the plan.
            if corner is None or original.distance_to(corner) > max_shift:
                corner = original
            snapped.append(corner)

        # Drop corners that collapsed onto each other.
        deduped: List[Point] = []
        for p in snapped:
            if not deduped or p.distance_to(deduped[-1]) > 0.5:
                deduped.append(p)
        if len(deduped) > 1 and deduped[0].distance_to(deduped[-1]) <= 0.5:
            deduped.pop()
        if len(deduped) < 3:
            return points

        # Safety net: snapping corrects centimetres, never reshapes a room.
        # If the area moved wildly, some face match was wrong — keep the raster.
        a_old, a_new = _polygon_area(points), _polygon_area(deduped)
        if a_old > 0 and abs(a_new - a_old) / a_old > 0.25:
            return points
        return deduped

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
        # A broken scale calibration can inflate the plan to tens of
        # thousands of cm; at 0.5 cm/px the float32 distance transform and
        # int32 watershed labels over that grid OOM-kill the whole batch
        # (case 11520: ~42,000×17,000 px, >10 GB). Coarsen the resolution so
        # the raster stays bounded — a degraded plan beats a dead batch.
        # self.resolution must follow: every later cm↔px conversion
        # (neck_px, simplify epsilon, room polygons) divides by it.
        if w_px * h_px > self.MAX_RASTER_PX:
            factor = math.sqrt((w_px * h_px) / self.MAX_RASTER_PX)
            self.resolution = res = res * factor
            w_px = int(math.ceil((max_x - min_x) / res)) + 1
            h_px = int(math.ceil((max_y - min_y) / res)) + 1
            logger.warning(
                "RoomDetector: raster would exceed %d px — coarsened "
                "resolution ×%.1f to %.2f cm/px (%dx%d px). Scale "
                "calibration is likely broken for this plan.",
                self.MAX_RASTER_PX, factor, res, w_px, h_px)
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
    @staticmethod
    def _close_wall_gaps(mask: np.ndarray) -> np.ndarray:
        """Close wall-raster gaps so exterior doesn't leak through cracks.

        Two passes: first closes single-pixel cracks, second closes gaps up
        to ~20 raster px wide (≈10 cm at resolution=0.5 cm/px). Union with
        the original preserves existing wall pixels. min_room_area_cm2 in
        _components_to_rooms filters false rooms from over-aggressive
        closing.
        """
        k5  = np.ones((5,  5),  dtype=np.uint8)
        k21 = np.ones((21, 21), dtype=np.uint8)
        sealed = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, k5,  iterations=1)
        sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, k21, iterations=2)
        return cv2.bitwise_or(sealed, mask)

    @staticmethod
    def _flood_interior(sealed: np.ndarray) -> np.ndarray:
        h, w = sealed.shape
        flood = sealed.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 128)
        # flood: 255 = wall, 128 = exterior, 0 = interior
        return (flood == 0).astype(np.uint8) * 255

    def _extract_interior(self, mask: np.ndarray) -> np.ndarray:
        return self._flood_interior(self._close_wall_gaps(mask))

    def _extract_interior_with_neck_cut(self, mask: np.ndarray) -> np.ndarray:
        """Extract interior after cutting door-width necks in the ENTIRE
        non-wall space, exterior included.

        A room joined to the OUTSIDE through an undetected doorway (entrance
        door, stair passage) floods as exterior during interior extraction
        and is lost before _split_at_necks can see it — on post_training
        plan 3 that killed every room in the left half of the plan. Running
        the same erode-core/watershed cut with the exterior as one more
        basin severs those passages first, so the flood-fill stays out.
        """
        sealed = self._close_wall_gaps(mask)
        res = self.resolution
        neck_px = (self.max_doorway_cm / res) / 2.0
        if neck_px >= 1:
            non_wall = (sealed == 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(non_wall, cv2.DIST_L2, 3)
            cores = (dist > neck_px).astype(np.uint8) * 255
            # The raster border is padding (60 cm beyond every wall), so it
            # is exterior by construction. Force it to be a core: with a
            # large derived neck the erosion could otherwise consume the
            # whole exterior margin and leave the outside without a basin.
            ring = np.zeros_like(cores)
            ring[0, :] = ring[-1, :] = 255
            ring[:, 0] = ring[:, -1] = 255
            cores = cv2.bitwise_or(cores, cv2.bitwise_and(ring, non_wall))
            n_cores, core_lbls = cv2.connectedComponents(cores)
            if n_cores > 2:
                markers = core_lbls.astype(np.int32)
                markers[non_wall == 0] = n_cores
                img3 = cv2.cvtColor(non_wall, cv2.COLOR_GRAY2BGR)
                cv2.watershed(img3, markers)
                ridge = (markers == -1).astype(np.uint8)
                ridge = cv2.dilate(ridge, np.ones((3, 3), np.uint8),
                                   iterations=2)
                sealed = sealed.copy()
                sealed[ridge > 0] = 255
        return self._flood_interior(sealed)

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

            # Simplify with an ABSOLUTE tolerance.  A tolerance proportional to
            # the perimeter would scale with room size (a 20x12 m room gets a
            # ~19 cm budget) and flatten away the short wall end-cap edges that
            # bound a reflex corner's notch, which the snap step needs to see.
            epsilon = self.simplify_epsilon_cm / self.resolution
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue

            points = [
                Point(pt[0][0] * res + min_x, pt[0][1] * res + min_y)
                for pt in approx
            ]
            # Snap the contour onto the wall faces so the floor is airtight.
            points = self._snap_polygon_to_walls(points)
            if len(points) < 3:
                continue

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
        # Vertical dimensions are absolute cm; only plan-space XY values are
        # divided by scale_factor (see sh3d-scale-rule).
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
        e.set('height', f'{w.height:.4f}')  # absolute cm, never scaled
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
            e.set('elevation', f'{o.elevation:.4f}')  # absolute cm
        e.set('angle', str(o.angle))
        e.set('width', f'{o.width / sf:.4f}')
        e.set('depth', f'{o.depth / sf:.4f}')
        e.set('height', f'{o.height:.4f}')  # absolute cm, never scaled
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
        preserve_structural_geometry: bool = False,
        split_doorways: bool = True,
        max_doorway_cm: float = 110.0,
    ) -> None:
        self.pixels_to_cm = pixels_to_cm
        self.wall_height = wall_height_cm
        self.snap_tolerance = snap_tolerance
        self.wall_overlap = wall_overlap
        # Room-detector doorway-splitting knobs (threaded to RoomDetector below).
        self.split_doorways = split_doorways
        self.max_doorway_cm = max_doorway_cm
        # When True, structural wall endpoints/axes are NEVER modified after
        # conversion from rect dicts. Use this with the rect_decompose pipeline
        # so the exporter keeps every wall exactly where the decomposer put it.
        self.preserve_structural_geometry = preserve_structural_geometry
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
        extra_structural_walls: Optional[List] = None,
        seal_only_openings: Optional[List[Opening]] = None,
    ) -> List[Room]:
        structural_walls = self.wall_converter.convert(wall_rectangles)

        # Prepend pre-built Wall objects (e.g. diagonal walls) that bypass
        # the rect-dict → Wall conversion in StructuralWallConverter.
        if extra_structural_walls:
            structural_walls = extra_structural_walls + structural_walls

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
        # Seal-only openings (e.g. door proposals dropped from export because
        # they sit on diagonal walls) still mark real doorways: they close the
        # passage for room flood-fill but are never written to the XML.
        room_openings = list(openings)
        if seal_only_openings:
            room_openings.extend(seal_only_openings)
            logger.info(
                "RoomDetector: sealing %d extra seal-only opening(s) "
                "(not exported)", len(seal_only_openings),
            )
        rooms = RoomDetector(
            resolution=0.5,
            min_room_area_cm2=2000.0,
            wall_dilation_px=3,
            simplify_epsilon_cm=2.0,
            snap_tolerance_cm=18.0,
            split_doorways=self.split_doorways,
            max_doorway_cm=self.max_doorway_cm,
        ).detect(
            all_walls, enclosed_labels=enclosed_labels, openings=room_openings,
        )

        _n_before_overlap = len(rooms)
        rooms = RoomOverlapRemover(
            min_area_threshold=50.0, overlap_threshold=0.50
        ).remove_overlaps(rooms)
        if len(rooms) != _n_before_overlap:
            logger.info("RoomOverlapRemover: %d -> %d room(s)",
                        _n_before_overlap, len(rooms))

        # Name rooms from OCR labels (room names / areas) that fall inside each
        # room polygon, so the SH3D carries the real room identity instead of a
        # generic "Room N".  Falls back silently when no labels are available.
        _assign_room_names_from_ocr(
            rooms, ocr_labels, self.wall_converter.pixels_to_cm,
        )

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

        # Diagonals have no meaningful axis-aligned bbox for this snap — skip them.
        sw_bboxes = [(sw, _wall_bbox(sw)) for sw in structural_walls if not sw.is_diagonal]
        self._door_placement_debug: List[dict] = []
        to_remove: set = set()

        # Sources that were already precisely snapped upstream — skip re-snapping
        # to avoid fighting with the gap-based placement from UNetDoorDetector
        # or DoorArcDetector.
        _confident_sources: frozenset = frozenset({"unet_door", "arc"})

        # Build a quick lookup: opening_wall id → source string
        _ow_sources: dict = {}
        for op in openings:
            if op.parent_wall_id:
                _ow_sources.setdefault(op.parent_wall_id, op.source or "")

        # Proximity limit: 50 image-pixels converted to cm
        PROXIMITY_CM = 50.0 * s

        for ow in opening_walls:
            # Skip walls whose opening was already placed by a confident source
            if _ow_sources.get(ow.wall_id, "") in _confident_sources:
                continue
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
                    # The opening centre is the midpoint of the bridged gap —
                    # using the detection centre would decenter the door
                    # whenever the gap is asymmetric around the detection.
                    gap_mid_x = (best_pair[0] + best_pair[1]) / 2
                    for o in openings:
                        if o.parent_wall_id == ow.wall_id:
                            o.width = best_gap
                            o.center = Point(gap_mid_x, det_cy)
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
                    # Centre the opening on the bridged gap (see horizontal
                    # branch above).
                    gap_mid_y = (best_pair[0] + best_pair[1]) / 2
                    for o in openings:
                        if o.parent_wall_id == ow.wall_id:
                            o.width = best_gap
                            o.center = Point(det_cx, gap_mid_y)
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

        # ──────────────────────────────────────────────────────────────
        # Second pass — fill the FULL flanking gap for openings that were
        # NOT snapped above.  Doors that matched a parent wall (fused arc /
        # U-Net / YOLO doors, and gap doors that found a co-linear host) keep
        # their raw detection width and centre.  For swing-leaf or
        # half-detected doors that leaves the piece ending in the MIDDLE of
        # the doorway — "half open" — instead of spanning it.  Here we
        # re-measure the clear gap between the two flanking STRUCTURAL walls
        # and resize + recentre the opening to bridge it edge-to-edge.
        #
        # The result is bounded by a ratio window so a door sitting on a
        # genuinely solid wall (no door-sized gap nearby) is left untouched —
        # only doors whose measured doorway is plausibly the same opening get
        # reworked.
        # ──────────────────────────────────────────────────────────────
        GAP_FILL_MIN_RATIO = 0.5    # ignore measured gaps below half the door
        GAP_FILL_MAX_RATIO = 2.5    # ...or above 2.5× it (that's a room, not a gap)
        wall_by_id_all = {w.wall_id: w for w in structural_walls}
        synth_ids = {ow.wall_id for ow in opening_walls}

        def _closest_flank_gap(door_bb, is_horizontal):
            """Clear inner-edge gap between the nearest flanking structural
            walls on either side of the door bbox, or None."""
            dw = door_bb[2] - door_bb[0]
            dh = door_bb[3] - door_bb[1]
            ext_bb = (door_bb[0] - dw * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[1] - dh * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[2] + dw * DOOR_BBOX_EXTEND_RATIO,
                      door_bb[3] + dh * DOOR_BBOX_EXTEND_RATIO)
            det_cx = (door_bb[0] + door_bb[2]) / 2
            det_cy = (door_bb[1] + door_bb[3]) / 2
            lo_list: List[Tuple[float, float]] = []
            hi_list: List[Tuple[float, float]] = []
            for sw, swbb in sw_bboxes:
                # Only real structural walls flank a doorway — never other
                # door/window opening walls (which are non-structural).
                if not getattr(sw, 'is_structural', True):
                    continue
                ovl = _overlap(ext_bb, swbb)
                if is_horizontal:
                    perp = swbb[3] > door_bb[1] and swbb[1] < door_bb[3]
                    sw_c = (swbb[0] + swbb[2]) / 2
                    le = _inner_edge_h(sw, 'L')
                    re = _inner_edge_h(sw, 'R')
                    if (perp and 0 <= door_bb[0] - le <= PROXIMITY_CM) or (ovl and sw_c < det_cx):
                        lo_list.append((max(0.0, door_bb[0] - le), le))
                    if (perp and 0 <= re - door_bb[2] <= PROXIMITY_CM) or (ovl and sw_c >= det_cx):
                        hi_list.append((max(0.0, re - door_bb[2]), re))
                else:
                    perp = swbb[2] > door_bb[0] and swbb[0] < door_bb[2]
                    sw_c = (swbb[1] + swbb[3]) / 2
                    te = _inner_edge_v(sw, 'T')
                    be = _inner_edge_v(sw, 'B')
                    if (perp and 0 <= door_bb[1] - te <= PROXIMITY_CM) or (ovl and sw_c < det_cy):
                        lo_list.append((max(0.0, door_bb[1] - te), te))
                    if (perp and 0 <= be - door_bb[3] <= PROXIMITY_CM) or (ovl and sw_c >= det_cy):
                        hi_list.append((max(0.0, be - door_bb[3]), be))
            if not lo_list or not hi_list:
                return None
            lo_list.sort(key=lambda t: t[0])
            hi_list.sort(key=lambda t: t[0])
            lo_edge = lo_list[0][1]
            hi_edge = hi_list[0][1]
            gap = hi_edge - lo_edge
            if gap <= MIN_GAP_CM:
                return None
            if is_horizontal:
                clear = _gap_is_clear(lo_edge, det_cy, hi_edge, det_cy)
            else:
                clear = _gap_is_clear(det_cx, lo_edge, det_cx, hi_edge)
            if not clear:
                return None
            return gap, lo_edge, hi_edge

        for o in openings:
            if o.opening_type not in (OpeningType.DOOR, OpeningType.GAP):
                continue
            if o.parent_wall_id in synth_ids or o.parent_wall_id in to_remove:
                continue
            if o.width is None or o.width <= 0:
                continue
            parent = wall_by_id_all.get(o.parent_wall_id)
            if parent is None or parent.is_diagonal:
                continue
            is_h = parent.is_horizontal
            half_w = o.width / 2.0
            half_t = max(parent.thickness, 2.0) / 2.0
            cx, cy = o.center.x, o.center.y
            if is_h:
                door_bb = (cx - half_w, cy - half_t, cx + half_w, cy + half_t)
            else:
                door_bb = (cx - half_t, cy - half_w, cx + half_t, cy + half_w)
            res = _closest_flank_gap(door_bb, is_h)
            if res is None:
                continue
            gap, lo_edge, hi_edge = res
            ratio = gap / o.width
            if ratio < GAP_FILL_MIN_RATIO or ratio > GAP_FILL_MAX_RATIO:
                continue
            mid = (lo_edge + hi_edge) / 2.0
            o.width = gap
            if is_h:
                o.center = Point(mid, parent.get_centerline_y())
            else:
                o.center = Point(parent.get_centerline_x(), mid)

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
        if not self.preserve_structural_geometry:
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
        if self.preserve_structural_geometry:
            # Snap only the opening walls; structural walls remain at the
            # endpoints rect_decompose produced.
            self._snap_endpoints_to_corners(opening_walls, snap_distance=5.0)
        else:
            self._snap_endpoints_to_corners(all_walls, snap_distance=5.0)
        self._connect_walls(all_walls)

    def _align_walls_to_shared_axes(
        self, walls: List[Wall], tolerance: float
    ) -> None:
        # Diagonals must be excluded: is_horizontal classifies them by AABB
        # and the snap below would flatten their endpoints onto one axis.
        axis_walls = [w for w in walls if not w.is_diagonal]
        h_walls = [w for w in axis_walls if w.is_horizontal]
        v_walls = [w for w in axis_walls if not w.is_horizontal]
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
        # Diagonals are not valid axis snap targets.
        h_struct = [
            (w.get_centerline_y(), w.length)
            for w in structural_walls if not w.is_diagonal and w.is_horizontal
        ]
        v_struct = [
            (w.get_centerline_x(), w.length)
            for w in structural_walls if not w.is_diagonal and not w.is_horizontal
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
        # Diagonals are skipped — _try_extend_endpoint moves only x or y.
        for _ in range(iterations):
            for wall in walls:
                if wall.length < 0.1 or wall.is_diagonal:
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
                if wall.length < 0.1 or wall.is_diagonal:
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
            # Diagonals are not valid axis-aligned targets — their
            # is_horizontal is bbox-derived and would mis-guide _calc_gap.
            if other.is_diagonal:
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
        # Diagonals are not eligible as snap participants — snapping only x
        # or only y would slide the endpoint off the diagonal's line.
        endpoints = [
            (w, e, w.start if e == 'start' else w.end)
            for w in walls if not w.is_diagonal for e in ('start', 'end')
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
            # Align angle to the parent wall for ALL openings. Hinged (arc)
            # doors used to be exempt to preserve "precise" arc orientation,
            # but openings only sit on axis-aligned walls, so the parent's
            # angle is always the correct one.
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
            # Keep the gap wall the opening spawns on as thick as the opening
            # itself.  A door/passage hosted by a synthesised non-structural
            # "gap wall" gets its depth from the thinnest connected wall
            # (above); leaving the gap wall at its original (apartment-average)
            # thickness makes the wall read fatter than its own door.  Apply
            # the door-thickness result back to the gap wall so the two stay
            # consistent.  Structural walls are never touched.
            if (not parent.is_structural
                    and o.opening_type in (OpeningType.DOOR, OpeningType.GAP)):
                parent.thickness = o.depth

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