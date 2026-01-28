"""
Система анализа планировок помещений - Экспортёр планировок - формат Blueprint3D
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Форматирование и сохранение планировок в формате .blueprint3D.
"""

import uuid
import json
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import IntEnum, Enum


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

class ItemType(IntEnum):
    """Типы объектов в Blueprint3D"""
    FLOOR = 1
    WALL = 2
    IN_WALL = 3      # Окна
    ROOF = 4
    WALL_FLOOR = 7   # Двери
    ON_FLOOR = 8


# Текстура стен по умолчанию
WALL_TEXTURE = {
    "url": "rooms/textures/wallmap.png",
    "stretch": True,
    "scale": 0
}

# 3D модели объектов
MODELS = {
    "window": "models/js/whitewindow.js",
    "open_door": "models/js/open_door.js",
}

# Константа преобразования дюймов в сантиметры
INCHES_TO_CM = 2.54


# ============================================================
# ГЕОМЕТРИЧЕСКИЕ ПРИМИТИВЫ (только целые числа)
# ============================================================

@dataclass(frozen=True, order=True)
class Pt:
    """Неизменяемая точка с целочисленными координатами."""
    x: int
    y: int

    def dist(self, other: 'Pt') -> float:
        """Расстояние до другой точки."""
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class Rect:
    """Прямоугольник стены с целочисленными координатами."""
    x1: int
    y1: int
    x2: int
    y2: int
    rect_id: int = 0

    def __post_init__(self):
        """Нормализация координат (x1 < x2, y1 < y2)."""
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> int:
        """Ширина прямоугольника."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Высота прямоугольника."""
        return self.y2 - self.y1

    def contains(self, x: int, y: int) -> bool:
        """Проверка, находится ли точка строго внутри (не на границе)."""
        return self.x1 < x < self.x2 and self.y1 < y < self.y2

    def get_edges(self) -> List[Tuple[Pt, Pt, str]]:
        """Получить 4 грани как (точка1, точка2, сторона)."""
        return [
            (Pt(self.x1, self.y1), Pt(self.x2, self.y1), 'top'),
            (Pt(self.x1, self.y2), Pt(self.x2, self.y2), 'bottom'),
            (Pt(self.x1, self.y1), Pt(self.x1, self.y2), 'left'),
            (Pt(self.x2, self.y1), Pt(self.x2, self.y2), 'right'),
        ]


@dataclass
class Opening:
    """Проём двери/окна."""
    x1: int
    y1: int
    x2: int
    y2: int
    opening_type: str = 'unknown'

    @property
    def is_horizontal(self) -> bool:
        """Проверка горизонтальной ориентации."""
        return abs(self.x2 - self.x1) >= abs(self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        """Центр проёма."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def size(self) -> int:
        """Размер проёма (максимальное измерение)."""
        return max(abs(self.x2 - self.x1), abs(self.y2 - self.y1))


# ============================================================
# КООРДИНАТНАЯ СЕТКА
# ============================================================

class Grid:
    """
    Целочисленная координатная сетка.
    Выполняет привязку координат и отслеживает все валидные X/Y линии.
    """

    def __init__(self, snap_dist: int = 5):
        """
        Инициализация сетки.

        Параметры:
            snap_dist: Максимальное расстояние для привязки координат
        """
        self.snap_dist = snap_dist
        self.xs: List[int] = []
        self.ys: List[int] = []

    def build(self, rects: List[Rect]):
        """Построение сетки из углов прямоугольников."""
        x_set = set()
        y_set = set()

        for r in rects:
            x_set.add(r.x1)
            x_set.add(r.x2)
            y_set.add(r.y1)
            y_set.add(r.y2)

        self.xs = self._merge(sorted(x_set))
        self.ys = self._merge(sorted(y_set))

    def _merge(self, coords: List[int]) -> List[int]:
        """Объединение координат в пределах snap_dist."""
        if not coords:
            return []
        result = [coords[0]]
        for c in coords[1:]:
            if c - result[-1] <= self.snap_dist:
                # Объединяем близкие координаты
                result[-1] = (result[-1] + c) // 2
            else:
                result.append(c)
        return result

    def snap(self, val: int, coords: List[int]) -> int:
        """Привязка к ближайшей координате."""
        if not coords:
            return val
        best = val
        best_d = self.snap_dist + 1
        for c in coords:
            d = abs(c - val)
            if d < best_d:
                best_d = d
                best = c
        return best if best_d <= self.snap_dist else val

    def snap_x(self, x: int) -> int:
        """Привязка X координаты."""
        return self.snap(x, self.xs)

    def snap_y(self, y: int) -> int:
        """Привязка Y координаты."""
        return self.snap(y, self.ys)

    def snap_pt(self, p: Pt) -> Pt:
        """Привязка точки к сетке."""
        return Pt(self.snap_x(p.x), self.snap_y(p.y))

    def snap_rect(self, r: Rect) -> Rect:
        """Привязка прямоугольника к сетке."""
        return Rect(
            self.snap_x(r.x1), self.snap_y(r.y1),
            self.snap_x(r.x2), self.snap_y(r.y2),
            r.rect_id
        )


# ============================================================
# КОЛЛЕКЦИЯ СЕГМЕНТОВ (с объединением и пересечением)
# ============================================================

class SegmentCollection:
    """
    Коллекция выровненных по осям сегментов с функциями:
    - Автоматическое объединение перекрывающихся сегментов
    - Детекция пересечений между горизонтальными и вертикальными линиями
    - Детекция проёмов
    """

    def __init__(self):
        # Горизонтальные: y -> список (x_начало, x_конец)
        self.h_segs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        # Вертикальные: x -> список (y_начало, y_конец)
        self.v_segs: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    def add_h(self, y: int, x1: int, x2: int):
        """Добавление горизонтального сегмента."""
        if x1 > x2:
            x1, x2 = x2, x1
        if x1 == x2:
            return
        self.h_segs[y].append((x1, x2))

    def add_v(self, x: int, y1: int, y2: int):
        """Добавление вертикального сегмента."""
        if y1 > y2:
            y1, y2 = y2, y1
        if y1 == y2:
            return
        self.v_segs[x].append((y1, y2))

    def merge_all(self):
        """Объединение перекрывающихся сегментов на каждой линии."""
        for y in self.h_segs:
            self.h_segs[y] = self._merge_segments(self.h_segs[y])
        for x in self.v_segs:
            self.v_segs[x] = self._merge_segments(self.v_segs[x])

    def _merge_segments(self, segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Объединение перекрывающихся/смежных сегментов."""
        if len(segs) <= 1:
            return segs
        segs = sorted(segs)
        result = [segs[0]]
        for s, e in segs[1:]:
            ls, le = result[-1]
            if s <= le + 1:  # Перекрытие или смежность
                result[-1] = (ls, max(le, e))
            else:
                result.append((s, e))
        return result

    def find_intersections(self) -> Set[Pt]:
        """Поиск всех точек пересечения горизонтальных и вертикальных сегментов."""
        intersections = set()

        for y, h_list in self.h_segs.items():
            for x, v_list in self.v_segs.items():
                # Проверка пересечения любого H сегмента на y с x
                for hx1, hx2 in h_list:
                    if hx1 <= x <= hx2:
                        # Проверка пересечения любого V сегмента на x с y
                        for vy1, vy2 in v_list:
                            if vy1 <= y <= vy2:
                                intersections.add(Pt(x, y))

        return intersections

    def extend_to_intersections(self, intersections: Set[Pt], max_extend: int = 10):
        """
        Расширение концов сегментов до ближайших пересечений.
        Исправляет стены, которые почти достигают, но не соединяются.
        """
        # Расширение горизонтальных сегментов
        for y in list(self.h_segs.keys()):
            new_segs = []
            for x1, x2 in self.h_segs[y]:
                # Проверка пересечений около x1 и x2
                for pt in intersections:
                    if pt.y == y:
                        if 0 < x1 - pt.x <= max_extend:
                            x1 = pt.x
                        if 0 < pt.x - x2 <= max_extend:
                            x2 = pt.x
                new_segs.append((x1, x2))
            self.h_segs[y] = self._merge_segments(new_segs)

        # Расширение вертикальных сегментов
        for x in list(self.v_segs.keys()):
            new_segs = []
            for y1, y2 in self.v_segs[x]:
                for pt in intersections:
                    if pt.x == x:
                        if 0 < y1 - pt.y <= max_extend:
                            y1 = pt.y
                        if 0 < pt.y - y2 <= max_extend:
                            y2 = pt.y
                new_segs.append((y1, y2))
            self.v_segs[x] = self._merge_segments(new_segs)

    def split_at_intersections(self, intersections: Set[Pt]):
        """
        Разделение сегментов в точках пересечения.
        Создаёт правильные угловые вершины.
        """
        # Разделение горизонтальных
        for y in list(self.h_segs.keys()):
            x_splits = sorted([pt.x for pt in intersections if pt.y == y])
            new_segs = []
            for x1, x2 in self.h_segs[y]:
                # Поиск точек разделения внутри сегмента
                splits = [x for x in x_splits if x1 < x < x2]
                if not splits:
                    new_segs.append((x1, x2))
                else:
                    # Разделение сегмента
                    prev = x1
                    for sx in splits:
                        if prev < sx:
                            new_segs.append((prev, sx))
                        prev = sx
                    if prev < x2:
                        new_segs.append((prev, x2))
            self.h_segs[y] = new_segs

        # Разделение вертикальных
        for x in list(self.v_segs.keys()):
            y_splits = sorted([pt.y for pt in intersections if pt.x == x])
            new_segs = []
            for y1, y2 in self.v_segs[x]:
                splits = [y for y in y_splits if y1 < y < y2]
                if not splits:
                    new_segs.append((y1, y2))
                else:
                    prev = y1
                    for sy in splits:
                        if prev < sy:
                            new_segs.append((prev, sy))
                        prev = sy
                    if prev < y2:
                        new_segs.append((prev, y2))
            self.v_segs[x] = new_segs

    def get_gaps(self, min_gap: int = 15, max_gap: int = 300) -> List[Opening]:
        """Поиск проёмов в линиях стен."""
        gaps = []

        # Проёмы в горизонтальных линиях
        for y, segs in self.h_segs.items():
            for i in range(len(segs) - 1):
                gap_start = segs[i][1]
                gap_end = segs[i + 1][0]
                if min_gap <= gap_end - gap_start <= max_gap:
                    gaps.append(Opening(gap_start, y, gap_end, y))

        # Проёмы в вертикальных линиях
        for x, segs in self.v_segs.items():
            for i in range(len(segs) - 1):
                gap_start = segs[i][1]
                gap_end = segs[i + 1][0]
                if min_gap <= gap_end - gap_start <= max_gap:
                    gaps.append(Opening(x, gap_start, x, gap_end))

        return gaps

    def get_all_corners(self) -> Set[Pt]:
        """Получение всех конечных точек сегментов как углов."""
        corners = set()
        for y, segs in self.h_segs.items():
            for x1, x2 in segs:
                corners.add(Pt(x1, y))
                corners.add(Pt(x2, y))
        for x, segs in self.v_segs.items():
            for y1, y2 in segs:
                corners.add(Pt(x, y1))
                corners.add(Pt(x, y2))
        return corners

    def stats(self) -> Tuple[int, int]:
        """Возвращает (кол-во горизонтальных, кол-во вертикальных) сегментов."""
        nh = sum(len(segs) for segs in self.h_segs.values())
        nv = sum(len(segs) for segs in self.v_segs.values())
        return nh, nv


# ============================================================
# ЭКСТРАКТОР ВНЕШНИХ СТЕН
# ============================================================

class ExternalWallExtractor:
    """
    Извлечение внешних (обращённых к комнате) стен.
    Устраняет внутренние стены, скрытые в массе стен.
    """

    def __init__(self, grid: Grid, test_offset: int = 2):
        """
        Инициализация экстрактора.

        Параметры:
            grid: Координатная сетка
            test_offset: Смещение для тестовой точки
        """
        self.grid = grid
        self.test_offset = test_offset
        self.rects: List[Rect] = []
        self.segments = SegmentCollection()

    def add_rect(self, r: Rect):
        """Добавление прямоугольника (привязанного к сетке)."""
        self.rects.append(self.grid.snap_rect(r))

    def _is_external(self, p1: Pt, p2: Pt, side: str, source: Rect) -> bool:
        """Проверка, обращена ли грань наружу."""
        mx = (p1.x + p2.x) // 2
        my = (p1.y + p2.y) // 2

        # Тестовая точка снаружи грани
        if side == 'top':
            tx, ty = mx, my - self.test_offset
        elif side == 'bottom':
            tx, ty = mx, my + self.test_offset
        elif side == 'left':
            tx, ty = mx - self.test_offset, my
        else:  # right
            tx, ty = mx + self.test_offset, my

        # Проверка, находится ли точка внутри другого прямоугольника
        for r in self.rects:
            if r.rect_id == source.rect_id:
                continue
            if r.contains(tx, ty):
                return False
        return True

    def extract(self) -> SegmentCollection:
        """Извлечение внешних стен."""
        ext_count = 0
        int_count = 0

        for rect in self.rects:
            for p1, p2, side in rect.get_edges():
                if self._is_external(p1, p2, side, rect):
                    if side in ('top', 'bottom'):
                        self.segments.add_h(p1.y, p1.x, p2.x)
                    else:
                        self.segments.add_v(p1.x, p1.y, p2.y)
                    ext_count += 1
                else:
                    int_count += 1

        # Объединение перекрывающихся сегментов
        self.segments.merge_all()

        return self.segments


# ============================================================
# ГРАФ УГЛОВ
# ============================================================

class CornerGraph:
    """Граф углов и стен."""

    def __init__(self):
        self.corners: Dict[Pt, int] = {}  # Точка -> id
        self.corner_pts: Dict[int, Pt] = {}  # id -> Точка
        self._next_id = 0

        self.walls: List[Tuple[int, int]] = []
        self._wall_set: Set[Tuple[int, int]] = set()

    def get_corner(self, p: Pt) -> int:
        """Получение или создание угла."""
        if p in self.corners:
            return self.corners[p]
        cid = self._next_id
        self._next_id += 1
        self.corners[p] = cid
        self.corner_pts[cid] = p
        return cid

    def add_wall(self, p1: Pt, p2: Pt) -> bool:
        """Добавление стены между точками."""
        c1 = self.get_corner(p1)
        c2 = self.get_corner(p2)
        if c1 == c2:
            return False
        key = (min(c1, c2), max(c1, c2))
        if key in self._wall_set:
            return False
        self._wall_set.add(key)
        self.walls.append((c1, c2))
        return True

    def build_from_segments(self, segs: SegmentCollection):
        """Построение графа из коллекции сегментов."""
        for y, seg_list in segs.h_segs.items():
            for x1, x2 in seg_list:
                self.add_wall(Pt(x1, y), Pt(x2, y))

        for x, seg_list in segs.v_segs.items():
            for y1, y2 in seg_list:
                self.add_wall(Pt(x, y1), Pt(x, y2))


# ============================================================
# КЛАССИФИКАТОР ПРОЁМОВ
# ============================================================

class OpeningClassifier:
    """Классификация проёмов на двери и окна."""

    @staticmethod
    def classify(gaps: List[Opening],
                 yolo_doors: List = None,
                 yolo_windows: List = None) -> List[Opening]:
        """
        Классификация проёмов на основе YOLO детекций.

        Параметры:
            gaps: Список обнаруженных проёмов
            yolo_doors: Список дверей от YOLO
            yolo_windows: Список окон от YOLO
        """

        def iou(b1, b2):
            """Расчёт Intersection over Union для двух bbox."""
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            return inter / (a1 + a2 - inter)

        for gap in gaps:
            # Расширение для сопоставления
            box = (gap.x1 - 20, gap.y1 - 20, gap.x2 + 20, gap.y2 + 20)

            max_door = 0.0
            max_win = 0.0

            if yolo_doors:
                for d in yolo_doors:
                    bbox = d if isinstance(d, (list, tuple)) else d[:4]
                    max_door = max(max_door, iou(box, tuple(bbox)))

            if yolo_windows:
                for w in yolo_windows:
                    bbox = w if isinstance(w, (list, tuple)) else w[:4]
                    max_win = max(max_win, iou(box, tuple(bbox)))

            if max_door > 0.05 or max_door > max_win:
                gap.opening_type = 'door'
            elif max_win > 0.05:
                gap.opening_type = 'window'

        return gaps


# ============================================================
# ЭКСПОРТЁР BLUEPRINT3D
# ============================================================

class Blueprint3DExporter:
    """Экспорт в формат Blueprint3D."""

    def __init__(self,
                 px_to_cm: float = 1.0,
                 wall_height: float = 300.0,
                 window_width_in: float = 48.0,
                 door_width_in: float = 65.0):
        """
        Инициализация экспортёра.

        Параметры:
            px_to_cm: Коэффициент преобразования пикселей в сантиметры
            wall_height: Высота стен в сантиметрах
            window_width_in: Ширина окна по умолчанию в дюймах
            door_width_in: Ширина двери по умолчанию в дюймах
        """
        self.px_to_cm = px_to_cm
        self.wall_height = wall_height
        self.window_width = window_width_in
        self.door_width = door_width_in
        self.graph: Optional[CornerGraph] = None
        self.items: List[Dict] = []

    def set_graph(self, g: CornerGraph):
        """Установка графа углов и стен."""
        self.graph = g

    def _cm(self, x: int, y: int) -> Tuple[float, float]:
        """Преобразование координат в сантиметры."""
        return (x * self.px_to_cm, y * self.px_to_cm)

    def add_window(self, op: Opening):
        """Добавление окна."""
        cx, cy = op.center
        cx_cm, cy_cm = self._cm(int(cx), int(cy))

        rotation = 0.0 if op.is_horizontal else math.pi / 2

        gap_cm = op.size * self.px_to_cm
        model_cm = self.window_width * INCHES_TO_CM
        scale = max(0.4, min(2.5, gap_cm / model_cm)) if model_cm > 0 else 1.0

        self.items.append({
            "item_name": "Window",
            "item_type": int(ItemType.IN_WALL),
            "model_url": MODELS["window"],
            "xpos": round(cx_cm, 2),
            "ypos": round(self.wall_height * 0.55, 2),
            "zpos": round(cy_cm, 2),
            "rotation": round(rotation, 4),
            "scale_x": round(scale, 3),
            "scale_y": 1.0,
            "scale_z": 1.0,
            "fixed": False
        })

    def add_door(self, op: Opening):
        """Добавление двери."""
        cx, cy = op.center
        cx_cm, cy_cm = self._cm(int(cx), int(cy))

        # Дверь: поворот на 90° от окна (перпендикулярно стене)
        rotation = math.pi / 2 if op.is_horizontal else 0.0

        gap_cm = op.size * self.px_to_cm
        model_cm = self.door_width * INCHES_TO_CM
        scale = max(0.4, min(2.5, gap_cm / model_cm)) if model_cm > 0 else 1.0

        self.items.append({
            "item_name": "Open Door",
            "item_type": int(ItemType.WALL_FLOOR),
            "model_url": MODELS["open_door"],
            "xpos": round(cx_cm, 2),
            "ypos": 0.0,
            "zpos": round(cy_cm, 2),
            "rotation": round(rotation, 4),
            "scale_x": round(scale, 3),
            "scale_y": 1.0,
            "scale_z": 1.0,
            "fixed": False
        })

    def export(self) -> Dict:
        """Экспорт в формат Blueprint3D."""
        if not self.graph:
            raise ValueError("Граф не установлен")

        # Генерация UUID для углов
        uuids = {cid: str(uuid.uuid4()) for cid in self.graph.corner_pts}

        # Углы
        corners = {}
        for cid, pt in self.graph.corner_pts.items():
            x_cm, y_cm = self._cm(pt.x, pt.y)
            corners[uuids[cid]] = {
                "x": round(x_cm, 2),
                "y": round(y_cm, 2),
                "elevation": 0.0
            }

        # Стены
        walls = []
        for c1, c2 in self.graph.walls:
            walls.append({
                "corner1": uuids[c1],
                "corner2": uuids[c2],
                "frontTexture": dict(WALL_TEXTURE),
                "backTexture": dict(WALL_TEXTURE)
            })

        return {
            "floorplan": {
                "corners": corners,
                "walls": walls,
                "rooms": {},
                "wallTextures": [],
                "floorTextures": {},
                "newFloorTextures": {}
            },
            "items": self.items
        }

    def stats(self) -> Dict:
        """Статистика экспорта."""
        wins = sum(1 for i in self.items if i['item_type'] == ItemType.IN_WALL)
        doors = sum(1 for i in self.items if i['item_type'] == ItemType.WALL_FLOOR)
        return {
            'corners': len(self.graph.corner_pts) if self.graph else 0,
            'walls': len(self.graph.walls) if self.graph else 0,
            'windows': wins,
            'doors': doors
        }


# ============================================================
# ПОСТРОИТЕЛЬ ПЛАНИРОВКИ
# ============================================================

class FloorplanBuilder:
    """
    Главный построитель с правильной обработкой пересечений.

    Алгоритм:
    1. Построение сетки из прямоугольников
    2. Привязка прямоугольников
    3. Извлечение внешних стен
    4. Объединение перекрывающихся сегментов
    5. Поиск всех H/V пересечений
    6. Расширение стен до пересечений
    7. Разделение стен в пересечениях (создание углов)
    8. Детекция проёмов
    9. Классификация проёмов
    10. Построение графа
    11. Соединение проёмов стенами
    12. Экспорт
    """

    def __init__(self,
                 wall_height_cm: float = 300.0,
                 snap_distance_px: int = 5,
                 extend_distance_px: int = 15,
                 default_window_width_inches: float = 48.0,
                 default_door_width_inches: float = 65.0,
                 min_gap_px: int = 15,
                 max_gap_px: int = 300,
                 # Совместимость с предыдущими версиями
                 corner_snap_tolerance_px: float = None,
                 line_merge_tolerance_px: float = None,
                 min_wall_length_cm: float = None,
                 skip_bbox_check: bool = True):
        """
        Инициализация построителя.

        Параметры:
            wall_height_cm: Высота стен в сантиметрах
            snap_distance_px: Расстояние привязки в пикселях
            extend_distance_px: Расстояние расширения в пикселях
            default_window_width_inches: Ширина окна по умолчанию в дюймах
            default_door_width_inches: Ширина двери по умолчанию в дюймах
            min_gap_px: Минимальный размер проёма в пикселях
            max_gap_px: Максимальный размер проёма в пикселях
        """
        self.wall_height = wall_height_cm
        self.snap_dist = snap_distance_px
        self.extend_dist = extend_distance_px
        self.window_width = default_window_width_inches
        self.door_width = default_door_width_inches
        self.min_gap = min_gap_px
        self.max_gap = max_gap_px

        self.grid: Optional[Grid] = None
        self.segments: Optional[SegmentCollection] = None
        self.graph: Optional[CornerGraph] = None
        self.exporter: Optional[Blueprint3DExporter] = None

    def build(self,
              rectangles: List,
              pixels_to_cm: float = 1.0,
              algo_windows: List = None,
              yolo_doors: List = None,
              yolo_windows: List = None,
              pre_detected_openings: List = None) -> Dict:
        """
        Построение планировки и экспорт в Blueprint3D.

        Параметры:
            rectangles: Список прямоугольников стен
            pixels_to_cm: Масштаб в см/пиксель
            algo_windows: Окна от алгоритмической детекции
            yolo_doors: Двери от YOLO
            yolo_windows: Окна от YOLO
            pre_detected_openings: Предварительно обнаруженные проёмы
        """
        # Преобразование входных данных
        rects = self._convert_rects(rectangles)

        # Шаг 1: Построение координатной сетки
        self.grid = Grid(self.snap_dist)
        self.grid.build(rects)

        # Шаг 2: Извлечение внешних стен
        extractor = ExternalWallExtractor(self.grid)
        for r in rects:
            extractor.add_rect(r)
        self.segments = extractor.extract()

        # Шаг 3: Поиск пересечений
        intersections = self.segments.find_intersections()

        # Шаг 4: Расширение стен до пересечений
        self.segments.extend_to_intersections(intersections, self.extend_dist)

        # Шаг 5: Повторное объединение после расширения
        self.segments.merge_all()

        # Шаг 6: Повторный поиск пересечений после расширения
        intersections = self.segments.find_intersections()

        # Шаг 7: Разделение в пересечениях
        self.segments.split_at_intersections(intersections)

        # Шаг 8: Детекция проёмов
        if pre_detected_openings:
            gaps = self._convert_openings(pre_detected_openings)
        else:
            gaps = self.segments.get_gaps(self.min_gap, self.max_gap)

        # Шаг 9: Классификация проёмов
        gaps = OpeningClassifier.classify(gaps, yolo_doors, yolo_windows)

        # Шаг 10: Соединение проёмов (добавление сегментов в проёмах)
        for gap in gaps:
            if gap.is_horizontal:
                y = (gap.y1 + gap.y2) // 2
                self.segments.add_h(y, gap.x1, gap.x2)
            else:
                x = (gap.x1 + gap.x2) // 2
                self.segments.add_v(x, gap.y1, gap.y2)

        # Шаг 11: Построение графа
        self.graph = CornerGraph()
        self.graph.build_from_segments(self.segments)

        # Шаг 12: Настройка экспортёра
        self.exporter = Blueprint3DExporter(
            px_to_cm=pixels_to_cm,
            wall_height=self.wall_height,
            window_width_in=self.window_width,
            door_width_in=self.door_width
        )
        self.exporter.set_graph(self.graph)

        # Шаг 13: Добавление объектов (двери/окна)
        for gap in gaps:
            if gap.opening_type == 'door':
                self.exporter.add_door(gap)
            elif gap.opening_type == 'window':
                self.exporter.add_window(gap)

        # Добавление алгоритмических окон
        if algo_windows:
            self._add_algo_windows(algo_windows, gaps)

        # Добавление YOLO дверей
        if yolo_doors:
            self._add_yolo_doors(yolo_doors, gaps)

        return self.exporter.export()

    def _convert_rects(self, rectangles: List) -> List[Rect]:
        """Преобразование входных прямоугольников в внутренний формат."""
        result = []
        for i, r in enumerate(rectangles):
            if hasattr(r, 'x1'):
                result.append(Rect(int(r.x1), int(r.y1), int(r.x2), int(r.y2), i))
            elif isinstance(r, dict):
                x1 = int(r.get('x1', r.get('x', 0)))
                y1 = int(r.get('y1', r.get('y', 0)))
                x2 = int(r.get('x2', x1 + r.get('width', 0)))
                y2 = int(r.get('y2', y1 + r.get('height', 0)))
                result.append(Rect(x1, y1, x2, y2, i))
        return result

    def _convert_openings(self, openings: List) -> List[Opening]:
        """Преобразование входных проёмов во внутренний формат."""
        result = []
        for op in openings:
            if hasattr(op, 'x1'):
                result.append(Opening(
                    int(op.x1), int(op.y1), int(op.x2), int(op.y2),
                    getattr(op, 'type', getattr(op, 'opening_type', 'unknown'))
                ))
            elif isinstance(op, dict):
                result.append(Opening(
                    int(op.get('x1', 0)), int(op.get('y1', 0)),
                    int(op.get('x2', 0)), int(op.get('y2', 0)),
                    op.get('type', op.get('opening_type', 'unknown'))
                ))
        return result

    def _add_algo_windows(self, algo_windows: List, existing: List[Opening]):
        """Добавление алгоритмических окон, избегая дублирования."""
        for w in algo_windows:
            if hasattr(w, 'x'):
                x, y, ww, h = w.x, w.y, w.w, w.h
            elif isinstance(w, dict):
                x = w.get('x', 0)
                y = w.get('y', 0)
                ww = w.get('w', w.get('width', 0))
                h = w.get('h', w.get('height', 0))
            else:
                continue

            cx, cy = x + ww/2, y + h/2
            covered = any(
                g.opening_type == 'window' and
                math.hypot(cx - g.center[0], cy - g.center[1]) < 30
                for g in existing
            )
            if not covered:
                op = Opening(int(x), int(y), int(x+ww), int(y+h), 'window')
                # Добавление сегмента
                if op.is_horizontal:
                    self.segments.add_h((op.y1+op.y2)//2, op.x1, op.x2)
                else:
                    self.segments.add_v((op.x1+op.x2)//2, op.y1, op.y2)
                self.exporter.add_window(op)

    def _add_yolo_doors(self, yolo_doors: List, existing: List[Opening]):
        """Добавление YOLO дверей, избегая дублирования."""
        for d in yolo_doors:
            bbox = d if isinstance(d, (list, tuple)) else d[:4]
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1+x2)/2, (y1+y2)/2
            covered = any(
                g.opening_type == 'door' and
                math.hypot(cx - g.center[0], cy - g.center[1]) < 30
                for g in existing
            )
            if not covered:
                op = Opening(x1, y1, x2, y2, 'door')
                if op.is_horizontal:
                    self.segments.add_h((op.y1+op.y2)//2, op.x1, op.x2)
                else:
                    self.segments.add_v((op.x1+op.x2)//2, op.y1, op.y2)
                self.exporter.add_door(op)

    def save(self, path: str):
        """Сохранение результата в файл."""
        if self.exporter:
            data = self.exporter.export()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)


# ============================================================
# УДОБНЫЕ ФУНКЦИИ
# ============================================================

def export_floorplan_to_blueprint3d(
    rectangles: List,
    output_path: str,
    pixels_to_cm: float = 1.0,
    algo_windows: List = None,
    yolo_doors: List = None,
    yolo_windows: List = None,
    wall_height_cm: float = 300.0,
    snap_distance_px: int = 5,
    extend_distance_px: int = 15,
    **kwargs
) -> Dict:
    """
    Экспорт планировки в формат Blueprint3D.

    Параметры:
        rectangles: Список прямоугольников стен
        output_path: Путь для сохранения файла
        pixels_to_cm: Масштаб в см/пиксель
        algo_windows: Окна от алгоритмической детекции
        yolo_doors: Двери от YOLO
        yolo_windows: Окна от YOLO
        wall_height_cm: Высота стен в см
        snap_distance_px: Расстояние привязки в пикселях
        extend_distance_px: Расстояние расширения в пикселях
    """
    builder = FloorplanBuilder(
        wall_height_cm=wall_height_cm,
        snap_distance_px=snap_distance_px,
        extend_distance_px=extend_distance_px,
        **kwargs
    )

    result = builder.build(
        rectangles=rectangles,
        pixels_to_cm=pixels_to_cm,
        algo_windows=algo_windows,
        yolo_doors=yolo_doors,
        yolo_windows=yolo_windows
    )

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Экспорт планировки в формат Blueprint3D"
    )
    parser.add_argument('--rectangles', '-r', required=True,
                        help='JSON файл с прямоугольниками стен')
    parser.add_argument('--windows', '-w',
                        help='JSON файл с окнами')
    parser.add_argument('--doors', '-d',
                        help='JSON файл с дверями')
    parser.add_argument('--output', '-o', default='floorplan.blueprint3d',
                        help='Выходной файл')
    parser.add_argument('--pixels-to-cm', type=float, default=1.0,
                        help='Масштаб в см/пиксель')
    parser.add_argument('--wall-height', type=float, default=300.0,
                        help='Высота стен в см')
    parser.add_argument('--snap-distance', type=int, default=5,
                        help='Расстояние привязки в пикселях')
    parser.add_argument('--extend-distance', type=int, default=15,
                        help='Расстояние расширения в пикселях')

    args = parser.parse_args()

    # Загрузка данных
    with open(args.rectangles) as f:
        rects = json.load(f)

    wins = None
    if args.windows:
        with open(args.windows) as f:
            wins = json.load(f)

    doors = None
    if args.doors:
        with open(args.doors) as f:
            doors = json.load(f)

    # Экспорт
    export_floorplan_to_blueprint3d(
        rects, args.output,
        pixels_to_cm=args.pixels_to_cm,
        algo_windows=wins,
        yolo_doors=doors,
        wall_height_cm=args.wall_height,
        snap_distance_px=args.snap_distance,
        extend_distance_px=args.extend_distance
    )