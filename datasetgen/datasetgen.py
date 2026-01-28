"""
Система анализа планировок помещений - Геренатор датасетов
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Архитектурный подход к генерации: концептуально-ориентированная генерация планировок,
зонная пространственная организация
"""

import os
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum, auto
from abc import ABC, abstractmethod

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# КОНСТАНТЫ И СТИЛИЗАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

WALL_COLOR = (128, 135, 142)
FURNITURE_COLOR = (221, 223, 226)
WINDOW_COLOR = WALL_COLOR
BG_COLOR = (255, 255, 255)
BALCONY_STRIPE_COLOR = (0, 0, 255)
LINE_TYPE = cv2.LINE_AA

# Архитектурные константы
GOLDEN_RATIO = 1.618
MIN_ROOM_ASPECT = 0.4
MAX_ROOM_ASPECT = 2.5


# ═══════════════════════════════════════════════════════════════════════════════
# ПЕРЕЧИСЛЕНИЯ И СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

class Zone(Enum):
    """Функциональные зоны в соответствии с архитектурными принципами"""
    PUBLIC = auto()  # Гостиная, Столовая - гостевые зоны
    PRIVATE = auto()  # Спальни - приватные зоны
    SERVICE = auto()  # Кухня, Ванная - служебные помещения
    CIRCULATION = auto()  # Коридоры, Холлы - зоны движения


class RoomPriority(Enum):
    """Приоритет выделения пространства"""
    PRIMARY = 3  # Основные жилые зоны
    SECONDARY = 2  # Спальни
    TERTIARY = 1  # Служебные помещения


class Orientation(Enum):
    """Предпочтительная ориентация окон для естественного освещения"""
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


@dataclass
class Wall:
    """Представление стены"""
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int
    is_external: bool
    is_guardrail: bool = False
    is_balcony_interface: bool = False


@dataclass
class Room:
    """Представление комнаты"""
    x1: int
    y1: int
    x2: int
    y2: int
    room_type: str
    zone: Zone = Zone.PUBLIC
    priority: RoomPriority = RoomPriority.SECONDARY
    needs_window: bool = True
    id: int = field(default_factory=lambda: random.randint(0, 100000))

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class Balcony:
    """Представление балкона"""
    x1: int
    y1: int
    x2: int
    y2: int
    side: str
    archetype: str
    interface_wall: Optional[Wall] = None


@dataclass
class OccupiedRegion:
    """Занятая область"""
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str


@dataclass
class ApartmentBrief:
    """Высокоуровневая концепция квартиры"""
    style: str  # 'compact', 'spacious', 'linear', 'square'
    num_rooms: int  # 1-4
    has_balcony: bool
    circulation_type: str  # 'central', 'linear', 'radial'
    light_priority: str  # 'living', 'bedroom', 'balanced'


# ═══════════════════════════════════════════════════════════════════════════════
# ДВИЖОК ПРОСТРАНСТВЕННОГО ПЛАНИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialPlanner:
    """
    Основная логика планирования с использованием архитектурных принципов.
    Реализует зонную организацию с подходом приоритета циркуляции.
    """

    # Матрица смежности: положительные значения = должны быть рядом, отрицательные = не должны
    ADJACENCY_MATRIX = {
        ('living', 'kitchen'): 0.9,
        ('living', 'corridor'): 0.7,
        ('living', 'bedroom'): 0.3,
        ('living', 'bathroom'): -0.5,
        ('kitchen', 'bathroom'): 0.4,  # Общая стена с коммуникациями
        ('bedroom', 'bathroom'): 0.6,
        ('bedroom', 'corridor'): 0.5,
        ('corridor', 'bathroom'): 0.8,
        ('corridor', 'kitchen'): 0.6,
    }

    # Конфигурации типов комнат
    ROOM_CONFIGS = {
        'living': {
            'zone': Zone.PUBLIC,
            'priority': RoomPriority.PRIMARY,
            'min_area_ratio': 0.25,
            'max_area_ratio': 0.40,
            'aspect_range': (0.7, 1.5),
            'needs_window': True,
            'margin_factor': 1.2,
        },
        'kitchen': {
            'zone': Zone.SERVICE,
            'priority': RoomPriority.SECONDARY,
            'min_area_ratio': 0.12,
            'max_area_ratio': 0.22,
            'aspect_range': (0.6, 1.8),
            'needs_window': True,
            'margin_factor': 1.0,
        },
        'bedroom': {
            'zone': Zone.PRIVATE,
            'priority': RoomPriority.PRIMARY,
            'min_area_ratio': 0.15,
            'max_area_ratio': 0.28,
            'aspect_range': (0.7, 1.4),
            'needs_window': True,
            'margin_factor': 1.1,
        },
        'bathroom': {
            'zone': Zone.SERVICE,
            'priority': RoomPriority.TERTIARY,
            'min_area_ratio': 0.06,
            'max_area_ratio': 0.12,
            'aspect_range': (0.5, 2.0),
            'needs_window': False,
            'margin_factor': 0.8,
        },
        'corridor': {
            'zone': Zone.CIRCULATION,
            'priority': RoomPriority.TERTIARY,
            'min_area_ratio': 0.08,
            'max_area_ratio': 0.18,
            'aspect_range': (0.2, 5.0),
            'needs_window': False,
            'margin_factor': 0.6,
        },
    }

    def __init__(self, rng: random.Random, size: int):
        self.rng = rng
        self.size = size

    def generate_brief(self) -> ApartmentBrief:
        """Создание концепции квартиры на основе взвешенного случайного выбора"""
        styles = ['compact', 'spacious', 'linear', 'square']
        style_weights = [0.3, 0.25, 0.25, 0.2]

        circulation_types = ['central', 'linear', 'radial']
        circ_weights = [0.4, 0.4, 0.2]

        return ApartmentBrief(
            style=self.rng.choices(styles, style_weights)[0],
            num_rooms=self.rng.choices([1, 2, 3, 4], [0.2, 0.35, 0.3, 0.15])[0],
            has_balcony=self.rng.random() < 0.7,
            circulation_type=self.rng.choices(circulation_types, circ_weights)[0],
            light_priority=self.rng.choice(['living', 'bedroom', 'balanced']),
        )

    def calculate_zone_allocation(self, brief: ApartmentBrief, total_area: int) -> Dict[Zone, float]:
        """Распределение процента общей площади по зонам"""
        base_allocation = {
            Zone.PUBLIC: 0.35,
            Zone.PRIVATE: 0.30,
            Zone.SERVICE: 0.20,
            Zone.CIRCULATION: 0.15,
        }

        # Корректировка на основе стиля
        if brief.style == 'compact':
            base_allocation[Zone.CIRCULATION] *= 0.7
            base_allocation[Zone.PUBLIC] *= 0.9
        elif brief.style == 'spacious':
            base_allocation[Zone.CIRCULATION] *= 1.3
            base_allocation[Zone.PUBLIC] *= 1.1

        # Корректировка на основе количества комнат
        if brief.num_rooms >= 3:
            base_allocation[Zone.PRIVATE] *= 1.2
            base_allocation[Zone.PUBLIC] *= 0.85

        # Нормализация
        total = sum(base_allocation.values())
        return {z: v / total for z, v in base_allocation.items()}

    def get_adjacency_score(self, room1: str, room2: str) -> float:
        """Получение оценки смежности между двумя типами комнат"""
        key = (room1, room2) if (room1, room2) in self.ADJACENCY_MATRIX else (room2, room1)
        return self.ADJACENCY_MATRIX.get(key, 0.0)

    def optimize_proportions(self, width: int, height: int, room_type: str) -> Tuple[int, int]:
        """Оптимизация размеров для лучших пропорций с использованием золотого сечения"""
        config = self.ROOM_CONFIGS.get(room_type, self.ROOM_CONFIGS['living'])
        min_asp, max_asp = config['aspect_range']

        current_aspect = width / max(1, height)

        if current_aspect < min_asp:
            # Слишком узкая - уменьшаем высоту
            target_aspect = min_asp + (max_asp - min_asp) * 0.3
            height = int(width / target_aspect)
        elif current_aspect > max_asp:
            # Слишком широкая - уменьшаем ширину
            target_aspect = max_asp - (max_asp - min_asp) * 0.3
            width = int(height * target_aspect)

        return width, height


# ═══════════════════════════════════════════════════════════════════════════════
# СТРАТЕГИИ ПЛАНИРОВКИ
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutStrategy(ABC):
    """Абстрактная база для различных подходов к генерации планировки"""

    @abstractmethod
    def generate(self, bounds: Tuple[int, int, int, int],
                 ext_th: int, int_th: int,
                 planner: SpatialPlanner,
                 brief: ApartmentBrief) -> Tuple[List[Wall], List[Room]]:
        pass


class CentralCorridorStrategy(LayoutStrategy):
    """
    Создает планировки с центральной осью циркуляции.
    Комнаты ответвляются от коридора с оптимизацией смежностей.
    """

    def generate(self, bounds: Tuple[int, int, int, int],
                 ext_th: int, int_th: int,
                 planner: SpatialPlanner,
                 brief: ApartmentBrief) -> Tuple[List[Wall], List[Room]]:

        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = planner.rng

        walls: List[Wall] = []
        rooms: List[Room] = []

        # Внешние стены
        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        # Определяем ориентацию коридора на основе соотношения сторон
        is_horizontal_corridor = width > height * 1.2

        if is_horizontal_corridor:
            # Горизонтальный коридор, разделяющий верхнюю/нижнюю зоны
            corridor_height = int(height * rng.uniform(0.15, 0.22))
            corridor_y = top + int(height * rng.uniform(0.35, 0.50))

            walls.append(Wall(left, corridor_y, right, corridor_y, int_th, False))
            walls.append(Wall(left, corridor_y + corridor_height, right,
                              corridor_y + corridor_height, int_th, False))

            rooms.append(Room(left, corridor_y, right, corridor_y + corridor_height,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

            # Верхняя зона - общественные помещения (гостиная, кухня у окон)
            top_zone_height = corridor_y - top
            kitchen_width = int(width * rng.uniform(0.35, 0.45))

            walls.append(Wall(left + kitchen_width, top, left + kitchen_width, corridor_y, int_th, False))

            rooms.append(Room(left, top, left + kitchen_width, corridor_y,
                              "kitchen", Zone.SERVICE, RoomPriority.SECONDARY))
            rooms.append(Room(left + kitchen_width, top, right, corridor_y,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))

            # Нижняя зона - приватные/служебные помещения
            bottom_start = corridor_y + corridor_height
            bathroom_width = int(width * rng.uniform(0.20, 0.30))

            walls.append(Wall(left + bathroom_width, bottom_start,
                              left + bathroom_width, bottom, int_th, False))

            rooms.append(Room(left, bottom_start, left + bathroom_width, bottom,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))

            # Разделение оставшегося пространства на спальни
            remaining_width = width - bathroom_width
            if brief.num_rooms >= 2 and remaining_width > width * 0.4:
                bedroom_split = left + bathroom_width + int(remaining_width * 0.5)
                walls.append(Wall(bedroom_split, bottom_start, bedroom_split, bottom, int_th, False))

                rooms.append(Room(left + bathroom_width, bottom_start, bedroom_split, bottom,
                                  "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))
                rooms.append(Room(bedroom_split, bottom_start, right, bottom,
                                  "living", Zone.PUBLIC, RoomPriority.SECONDARY))
            else:
                rooms.append(Room(left + bathroom_width, bottom_start, right, bottom,
                                  "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))

        else:
            # Вертикальный коридор для высоких квартир
            corridor_width = int(width * rng.uniform(0.15, 0.22))
            corridor_x = left + int(width * rng.uniform(0.35, 0.50))

            walls.append(Wall(corridor_x, top, corridor_x, bottom, int_th, False))
            walls.append(Wall(corridor_x + corridor_width, top,
                              corridor_x + corridor_width, bottom, int_th, False))

            rooms.append(Room(corridor_x, top, corridor_x + corridor_width, bottom,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

            # Левая зона
            bathroom_height = int(height * rng.uniform(0.22, 0.32))
            walls.append(Wall(left, top + bathroom_height, corridor_x, top + bathroom_height, int_th, False))

            rooms.append(Room(left, top, corridor_x, top + bathroom_height,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))
            rooms.append(Room(left, top + bathroom_height, corridor_x, bottom,
                              "kitchen", Zone.SERVICE, RoomPriority.SECONDARY))

            # Правая зона
            right_start = corridor_x + corridor_width
            living_height = int(height * rng.uniform(0.45, 0.60))

            walls.append(Wall(right_start, top + living_height, right, top + living_height, int_th, False))

            rooms.append(Room(right_start, top, right, top + living_height,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))
            rooms.append(Room(right_start, top + living_height, right, bottom,
                              "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))

        return walls, rooms


class LinearFlowStrategy(LayoutStrategy):
    """
    Создает планировки с комнатами, расположенными в линейной последовательности.
    Оптимально для узких форм квартир.
    """

    def generate(self, bounds: Tuple[int, int, int, int],
                 ext_th: int, int_th: int,
                 planner: SpatialPlanner,
                 brief: ApartmentBrief) -> Tuple[List[Wall], List[Room]]:

        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = planner.rng

        walls: List[Wall] = []
        rooms: List[Room] = []

        # Внешние стены
        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        # Направление потока на основе формы
        is_horizontal_flow = width > height

        if is_horizontal_flow:
            # Комнаты слева направо: Вход → Сервис → Гостиная → Приватная зона
            zone_splits = self._calculate_splits(width, rng, brief)
            current_x = left

            # Зона 1: Вход/Ванная
            z1_width = int(width * zone_splits[0])
            bathroom_height = int(height * rng.uniform(0.35, 0.50))

            walls.append(Wall(current_x + z1_width, top, current_x + z1_width, bottom, int_th, False))
            walls.append(Wall(current_x, top + bathroom_height,
                              current_x + z1_width, top + bathroom_height, int_th, False))

            rooms.append(Room(current_x, top, current_x + z1_width, top + bathroom_height,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))
            rooms.append(Room(current_x, top + bathroom_height, current_x + z1_width, bottom,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

            current_x += z1_width

            # Зона 2: Кухня
            z2_width = int(width * zone_splits[1])
            walls.append(Wall(current_x + z2_width, top, current_x + z2_width, bottom, int_th, False))

            rooms.append(Room(current_x, top, current_x + z2_width, bottom,
                              "kitchen", Zone.SERVICE, RoomPriority.SECONDARY))
            current_x += z2_width

            # Зона 3: Гостиная (наибольшая)
            z3_width = int(width * zone_splits[2])
            walls.append(Wall(current_x + z3_width, top, current_x + z3_width, bottom, int_th, False))

            rooms.append(Room(current_x, top, current_x + z3_width, bottom,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))
            current_x += z3_width

            # Зона 4: Спальни
            rooms.append(Room(current_x, top, right, bottom,
                              "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))

        else:
            # Вертикальный поток: сверху вниз
            zone_splits = self._calculate_splits(height, rng, brief)
            current_y = top

            # Зона 1: Гостиная (вверху - лучший свет)
            z1_height = int(height * zone_splits[2])  # Меняем порядок для приоритета освещения
            walls.append(Wall(left, current_y + z1_height, right, current_y + z1_height, int_th, False))

            rooms.append(Room(left, current_y, right, current_y + z1_height,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))
            current_y += z1_height

            # Зона 2: Кухня + Коридор
            z2_height = int(height * zone_splits[1])
            kitchen_width = int(width * rng.uniform(0.55, 0.70))

            walls.append(Wall(left, current_y + z2_height, right, current_y + z2_height, int_th, False))
            walls.append(Wall(left + kitchen_width, current_y,
                              left + kitchen_width, current_y + z2_height, int_th, False))

            rooms.append(Room(left, current_y, left + kitchen_width, current_y + z2_height,
                              "kitchen", Zone.SERVICE, RoomPriority.SECONDARY))
            rooms.append(Room(left + kitchen_width, current_y, right, current_y + z2_height,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))
            current_y += z2_height

            # Зона 3: Ванная + Спальня
            bathroom_width = int(width * rng.uniform(0.28, 0.38))
            walls.append(Wall(left + bathroom_width, current_y, left + bathroom_width, bottom, int_th, False))

            rooms.append(Room(left, current_y, left + bathroom_width, bottom,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))
            rooms.append(Room(left + bathroom_width, current_y, right, bottom,
                              "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))

        return walls, rooms

    def _calculate_splits(self, total: int, rng: random.Random, brief: ApartmentBrief) -> List[float]:
        """Расчет соотношений разделения зон на основе концепции"""
        if brief.style == 'compact':
            return [0.18, 0.22, 0.35, 0.25]
        elif brief.style == 'spacious':
            return [0.15, 0.20, 0.40, 0.25]
        else:
            base = [0.17, 0.21, 0.37, 0.25]
            # Добавление небольшой случайности
            return [v * rng.uniform(0.9, 1.1) for v in base]


class RadialHubStrategy(LayoutStrategy):
    """
    Создает планировки с центральным хабом (обычно гостиная) и комнатами вокруг.
    Лучше всего подходит для квадратных форм квартир.
    """

    def generate(self, bounds: Tuple[int, int, int, int],
                 ext_th: int, int_th: int,
                 planner: SpatialPlanner,
                 brief: ApartmentBrief) -> Tuple[List[Wall], List[Room]]:
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = planner.rng

        walls: List[Wall] = []
        rooms: List[Room] = []

        # Внешние стены
        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        # Создание центрального жилого пространства с окружающими комнатами
        margin_x = int(width * rng.uniform(0.25, 0.35))
        margin_y = int(height * rng.uniform(0.25, 0.35))

        # Центральная жилая зона
        living_left = left + margin_x
        living_right = right - margin_x
        living_top = top + margin_y
        living_bottom = bottom - margin_y

        # Создание L-образных или U-образных коридоров и периферийных комнат
        # Верхняя полоса
        walls.append(Wall(left, living_top, right, living_top, int_th, False))

        # Разделение верхней полосы
        top_split = left + int(width * rng.uniform(0.35, 0.50))
        walls.append(Wall(top_split, top, top_split, living_top, int_th, False))

        rooms.append(Room(left, top, top_split, living_top,
                          "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))
        rooms.append(Room(top_split, top, right, living_top,
                          "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

        # Левая полоса
        walls.append(Wall(living_left, living_top, living_left, bottom, int_th, False))

        left_split = living_top + int((bottom - living_top) * rng.uniform(0.40, 0.55))
        walls.append(Wall(left, left_split, living_left, left_split, int_th, False))

        rooms.append(Room(left, living_top, living_left, left_split,
                          "kitchen", Zone.SERVICE, RoomPriority.SECONDARY))
        rooms.append(Room(left, left_split, living_left, bottom,
                          "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))

        # Центральная/Правая зона - гостиная
        rooms.append(Room(living_left, living_top, right, bottom,
                          "living", Zone.PUBLIC, RoomPriority.PRIMARY))

        return walls, rooms


# ═══════════════════════════════════════════════════════════════════════════════
# УМНАЯ РАССТАНОВКА МЕБЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

class SmartFurniturePlacer:
    """
    Интеллектуальная расстановка мебели с учетом:
    - Потоков движения и зон прохода
    - Фокусных точек и функций комнат
    - Группировки мебели и взаимосвязей
    - Использования естественного освещения
    """

    # Минимальные зазоры вокруг мебели для прохода (как отношение к размеру комнаты)
    CLEARANCE_RATIOS = {
        'bathroom': 0.15,
        'kitchen': 0.12,
        'living': 0.10,
        'bedroom': 0.12,
        'corridor': 0.05,
    }

    # Плотность мебели по типам комнат
    DENSITY = {
        'bathroom': (2, 4),
        'kitchen': (1, 2),
        'living': (4, 8),
        'bedroom': (2, 5),
        'corridor': (0, 2),
    }

    def __init__(self, rng: random.Random, size: int):
        self.rng = rng
        self.size = size
        self.occupied: List[OccupiedRegion] = []

    def reset(self):
        self.occupied = []

    def add_occupied(self, x1: int, y1: int, x2: int, y2: int, region_type: str):
        self.occupied.append(OccupiedRegion(
            min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), region_type
        ))

    def check_collision(self, x1: int, y1: int, x2: int, y2: int, margin: int = 5) -> bool:
        for region in self.occupied:
            if not (x2 + margin < region.x1 or x1 - margin > region.x2 or
                    y2 + margin < region.y1 or y1 - margin > region.y2):
                return True
        return False

    def place_furniture(self, img: np.ndarray, rooms: List[Room], walls: List[Wall]):
        """Основная точка входа для расстановки мебели"""
        for room in rooms:
            rw, rh = room.width, room.height
            if rw < 80 or rh < 80:
                continue

            # Сначала размещаем специализированную мебель
            if room.room_type == 'kitchen':
                self._place_kitchen_suite(img, room)
            elif room.room_type == 'bathroom':
                self._place_bathroom_suite(img, room)

            # Добавляем общую мебель на основе плотности
            density = self.DENSITY.get(room.room_type, (1, 3))
            n_items = self.rng.randint(*density)
            clearance = self.CLEARANCE_RATIOS.get(room.room_type, 0.1)
            margin = int(min(rw, rh) * clearance)

            for _ in range(n_items):
                self._place_single_furniture(img, room, margin)

            # Добавляем декоративные элементы типа дверей
            if room.room_type in ('living', 'kitchen', 'corridor'):
                if self.rng.random() < 0.7:
                    for _ in range(self.rng.randint(1, 3)):
                        self._place_decorative_element(img, room, margin)

    def _place_single_furniture(self, img: np.ndarray, room: Room, margin: int):
        """Размещение одного предмета мебели с проверкой коллизий"""
        for _ in range(10):
            shape = self.rng.choice(['rectangle', 'ellipse', 'circle'])
            rw, rh = room.width, room.height

            if shape == 'rectangle':
                w = self.rng.randint(int(rw * 0.15), max(20, int(rw * 0.40)))
                h = self.rng.randint(int(rh * 0.15), max(20, int(rh * 0.40)))

                x_range = room.x2 - margin - w - (room.x1 + margin)
                y_range = room.y2 - margin - h - (room.y1 + margin)

                if x_range <= 0 or y_range <= 0:
                    continue

                x = room.x1 + margin + self.rng.randint(0, x_range)
                y = room.y1 + margin + self.rng.randint(0, y_range)

                if self.check_collision(x, y, x + w, y + h):
                    continue

                cv2.rectangle(img, (x, y), (x + w, y + h), FURNITURE_COLOR, -1, LINE_TYPE)
                cv2.rectangle(img, (x, y), (x + w, y + h), WALL_COLOR, 1, LINE_TYPE)
                self.add_occupied(x, y, x + w, y + h, "furniture")
                break

            elif shape == 'ellipse':
                a = self.rng.randint(int(rw * 0.10), max(15, int(rw * 0.25)))
                b = self.rng.randint(int(rh * 0.10), max(15, int(rh * 0.25)))

                cx_range = room.x2 - margin - a - (room.x1 + margin + a)
                cy_range = room.y2 - margin - b - (room.y1 + margin + b)

                if cx_range <= 0 or cy_range <= 0:
                    continue

                cx = room.x1 + margin + a + self.rng.randint(0, cx_range)
                cy = room.y1 + margin + b + self.rng.randint(0, cy_range)

                if self.check_collision(cx - a, cy - b, cx + a, cy + b):
                    continue

                cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, FURNITURE_COLOR, -1, LINE_TYPE)
                cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, WALL_COLOR, 1, LINE_TYPE)
                self.add_occupied(cx - a, cy - b, cx + a, cy + b, "furniture")
                break

            else:  # circle
                r = self.rng.randint(int(min(rw, rh) * 0.08), max(10, int(min(rw, rh) * 0.18)))

                cx_range = room.x2 - margin - r - (room.x1 + margin + r)
                cy_range = room.y2 - margin - r - (room.y1 + margin + r)

                if cx_range <= 0 or cy_range <= 0:
                    continue

                cx = room.x1 + margin + r + self.rng.randint(0, cx_range)
                cy = room.y1 + margin + r + self.rng.randint(0, cy_range)

                if self.check_collision(cx - r, cy - r, cx + r, cy + r):
                    continue

                cv2.circle(img, (cx, cy), r, FURNITURE_COLOR, -1, LINE_TYPE)
                cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, LINE_TYPE)
                self.add_occupied(cx - r, cy - r, cx + r, cy + r, "furniture")
                break

    def _place_kitchen_suite(self, img: np.ndarray, room: Room):
        """Размещение кухонного гарнитура вдоль одной стены с бытовыми приборами"""
        rw, rh = room.width, room.height
        if rw < 120 or rh < 120:
            return

        depth = max(24, int(min(rw, rh) * self.rng.uniform(0.08, 0.13)))
        side = self.rng.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            usable_width = rw - depth * 2
            if usable_width <= depth * 2:  # Недостаточно места
                return

            band_len = int(usable_width * self.rng.uniform(0.6, 0.95))
            if band_len < depth * 3:  # Слишком короткая столешница
                return

            x1 = self.rng.randint(room.x1 + depth, room.x2 - depth - band_len)
            x2 = x1 + band_len

            if side == "top":
                y1, y2 = room.y1 + 5, room.y1 + 5 + depth
            else:
                y2, y1 = room.y2 - 5, room.y2 - 5 - depth

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, LINE_TYPE)
            self.add_occupied(x1, y1, x2, y2, "furniture")

            # Добавление кругов плиты с проверкой границ
            stove_width = depth * 2
            if x2 - x1 >= stove_width + depth * 2:  # Достаточно места для плиты
                stove_x_min = x1 + depth
                stove_x_max = x2 - depth - int(depth * 0.6)  # Запас справа

                if stove_x_max > stove_x_min:
                    stove_x = self.rng.randint(stove_x_min, stove_x_max)
                    for i in range(2):
                        for j in range(2):
                            cx = stove_x + int(depth * 0.3 * (i - 0.5))
                            cy = (y1 + y2) // 2 + int(depth * 0.3 * (j - 0.5))
                            r = int(depth * 0.12)
                            if r > 0:
                                cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, LINE_TYPE)

        else:  # left or right
            usable_height = rh - depth * 2
            if usable_height <= depth * 2:  # Недостаточно места
                return

            band_len = int(usable_height * self.rng.uniform(0.6, 0.95))
            if band_len < depth * 3:  # Слишком короткая столешница
                return

            y1 = self.rng.randint(room.y1 + depth, room.y2 - depth - band_len)
            y2 = y1 + band_len

            if side == "left":
                x1, x2 = room.x1 + 5, room.x1 + 5 + depth
            else:
                x2, x1 = room.x2 - 5, room.x2 - 5 - depth

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, LINE_TYPE)
            self.add_occupied(x1, y1, x2, y2, "furniture")

            # Добавление кругов плиты с проверкой границ
            stove_height = depth * 2
            if y2 - y1 >= stove_height + depth * 2:  # Достаточно места для плиты
                stove_y_min = y1 + depth
                stove_y_max = y2 - depth - int(depth * 0.6)  # Запас снизу

                if stove_y_max > stove_y_min:
                    stove_y = self.rng.randint(stove_y_min, stove_y_max)
                    for i in range(2):
                        for j in range(2):
                            cx = (x1 + x2) // 2 + int(depth * 0.3 * (i - 0.5))
                            cy = stove_y + int(depth * 0.3 * (j - 0.5))
                            r = int(depth * 0.12)
                            if r > 0:
                                cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, LINE_TYPE)

    def _place_bathroom_suite(self, img: np.ndarray, room: Room):
        """Размещение ванны, раковины и унитаза"""
        rw, rh = room.width, room.height
        if rw < 80 or rh < 80:
            return

        # Ванна
        if rw >= rh:
            tub_len = int(rw * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rh * self.rng.uniform(0.35, 0.5))
            x1 = self.rng.randint(room.x1 + 8, room.x2 - 8 - tub_len)
            y1 = room.y1 + 5
            x2, y2 = x1 + tub_len, y1 + tub_depth
        else:
            tub_len = int(rh * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rw * self.rng.uniform(0.35, 0.5))
            x1 = room.x1 + 5
            y1 = self.rng.randint(room.y1 + 8, room.y2 - 8 - tub_len)
            x2, y2 = x1 + tub_depth, y1 + tub_len

        cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, LINE_TYPE)
        cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, LINE_TYPE)
        self.add_occupied(x1, y1, x2, y2, "furniture")

        # Внутренний контур ванны
        pad = max(3, int(min(x2 - x1, y2 - y1) * 0.08))
        cv2.rectangle(img, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad),
                      BG_COLOR, -1, LINE_TYPE)
        cv2.rectangle(img, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad),
                      WALL_COLOR, 1, LINE_TYPE)

        # Раковина
        base = min(rw, rh)
        sink_w, sink_h = int(base * 0.18), int(base * 0.14)
        sx1 = room.x2 - sink_w - 8
        sy1 = room.y2 - sink_h - 8

        cv2.rectangle(img, (sx1, sy1), (sx1 + sink_w, sy1 + sink_h), FURNITURE_COLOR, -1, LINE_TYPE)
        cv2.rectangle(img, (sx1, sy1), (sx1 + sink_w, sy1 + sink_h), WALL_COLOR, 1, LINE_TYPE)
        self.add_occupied(sx1, sy1, sx1 + sink_w, sy1 + sink_h, "furniture")

        # Унитаз
        wc_r = int(base * 0.06)
        wc_cx, wc_cy = room.x1 + 12 + wc_r, room.y2 - 12 - wc_r
        cv2.circle(img, (wc_cx, wc_cy), wc_r, FURNITURE_COLOR, -1, LINE_TYPE)
        cv2.circle(img, (wc_cx, wc_cy), wc_r, WALL_COLOR, 1, LINE_TYPE)
        cv2.rectangle(img, (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
                      (wc_cx + wc_r // 2, wc_cy - wc_r), FURNITURE_COLOR, -1, LINE_TYPE)
        cv2.rectangle(img, (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
                      (wc_cx + wc_r // 2, wc_cy - wc_r), WALL_COLOR, 1, LINE_TYPE)
        self.add_occupied(wc_cx - wc_r, wc_cy - wc_r * 2, wc_cx + wc_r, wc_cy + wc_r, "furniture")

    def _place_decorative_element(self, img: np.ndarray, room: Room, margin: int):
        """Размещение декоративных элементов мебели типа дверей"""
        rw, rh = room.width, room.height

        # Определение размеров
        len_range = (0.02, 0.07)
        thick_ratio = (0.10, 1.20)

        elem_len = int(self.size * self.rng.uniform(*len_range))
        elem_thick = max(3, int(elem_len * self.rng.uniform(*thick_ratio)))

        orientation = self.rng.choice(["vertical", "horizontal"])
        if orientation == "vertical":
            rect_h, rect_w = elem_len, elem_thick
        else:
            rect_w, rect_h = elem_len, elem_thick

        # Масштабирование для вписывания
        scale = min((rw - 2 * margin) / rect_w, (rh - 2 * margin) / rect_h, 1.0)
        rect_w, rect_h = max(3, int(rect_w * scale)), max(3, int(rect_h * scale))

        for _ in range(10):
            x_max = room.x2 - margin - rect_w
            x_min = room.x1 + margin
            y_max = room.y2 - margin - rect_h
            y_min = room.y1 + margin

            if x_max <= x_min or y_max <= y_min:
                return

            x = self.rng.randint(x_min, x_max)
            y = self.rng.randint(y_min, y_max)

            if self.check_collision(x, y, x + rect_w, y + rect_h, margin=10):
                continue

            corner_r = int(min(rect_w, rect_h) * self.rng.uniform(0.15, 0.40))
            self._draw_rounded_rect(img, x, y, x + rect_w, y + rect_h,
                                    FURNITURE_COLOR, corner_r, WALL_COLOR, 1)
            self.add_occupied(x, y, x + rect_w, y + rect_h, "furniture")
            break

    def _draw_rounded_rect(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                           fill: Tuple[int, int, int], radius: int,
                           border: Optional[Tuple[int, int, int]] = None, border_th: int = 1):
        """Отрисовка прямоугольника с закругленными углами"""
        if x2 <= x1 or y2 <= y1:
            return

        w, h = x2 - x1, y2 - y1
        radius = min(radius, w // 2, h // 2)

        if radius <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, LINE_TYPE)
            if border:
                cv2.rectangle(img, (x1, y1), (x2, y2), border, border_th, LINE_TYPE)
            return

        # Заливка центральных прямоугольников
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), fill, -1, LINE_TYPE)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), fill, -1, LINE_TYPE)

        # Заливка углов
        cv2.circle(img, (x1 + radius, y1 + radius), radius, fill, -1, LINE_TYPE)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, fill, -1, LINE_TYPE)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, fill, -1, LINE_TYPE)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, fill, -1, LINE_TYPE)

        if border and border_th > 0:
            # Отрисовка линий границ
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border, border_th, LINE_TYPE)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border, border_th, LINE_TYPE)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border, border_th, LINE_TYPE)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border, border_th, LINE_TYPE)

            # Отрисовка угловых дуг
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius),
                        0, 180, 270, border, border_th, LINE_TYPE)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius),
                        0, 270, 360, border, border_th, LINE_TYPE)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),
                        0, 90, 180, border, border_th, LINE_TYPE)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),
                        0, 0, 90, border, border_th, LINE_TYPE)


# ═══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС ГЕНЕРАТОРА
# ═══════════════════════════════════════════════════════════════════════════════

class SmartFloorPlanGenerator:
    """
    Интеллектуальный генератор планировок с использованием архитектурных принципов.
    Координирует все компоненты планирования для создания целостного результата.
    """

    def __init__(self, img_size: int = 1024, seed: int = 0, super_sampling: int = 2):
        self.out_size = img_size
        self.super_sampling = super_sampling
        self.size = img_size * super_sampling
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        # Инициализация компонентов планирования
        self.planner = SpatialPlanner(self.rng, self.size)
        self.furniture_placer = SmartFurniturePlacer(self.rng, self.size)

        # Стратегии планировки
        self.strategies: Dict[str, LayoutStrategy] = {
            'central': CentralCorridorStrategy(),
            'linear': LinearFlowStrategy(),
            'radial': RadialHubStrategy(),
        }

        # Конфигурация
        self.window_length_ratios = (0.005, 0.010, 0.025, 0.035, 0.045, 0.05, 0.07, 0.08, 0.09)
        self.door_length_ratios = (0.04, 0.05, 0.06)
        self.bg_color = BG_COLOR

    def generate_dataset(self, n_images: int, out_dir: str,
                         wall_thickness_range: Tuple[int, int] = (7, 30)):
        """Генерация датасета с указанным количеством планировок"""
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n_images):
            base_ext_thickness = self.rng.randint(*wall_thickness_range)
            ext_wall_thickness = base_ext_thickness * self.super_sampling
            int_wall_thickness = int(ext_wall_thickness * self.rng.uniform(0.5, 0.7))

            hi_res_img, labels = self.generate_single_plan(ext_wall_thickness, int_wall_thickness)

            img = cv2.resize(hi_res_img, (self.out_size, self.out_size),
                             interpolation=cv2.INTER_AREA)

            if self.rng.random() < 0.15:
                img, labels = self._apply_rotation(img, labels)

            if self.rng.random() < 0.3:
                img = self._apply_augmentations(img)

            fname = f"{i:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), img)

            with open(os.path.join(lbl_dir, fname + ".txt"), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def generate_single_plan(self, ext_wall_thickness: int, int_wall_thickness: int
                             ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        """Генерация одной планировки с использованием умного планирования"""

        # Сброс состояния
        self.furniture_placer.reset()

        # Инициализация холста
        base_bg = self.rng.randint(248, 255)
        self.bg_color = (base_bg, base_bg, base_bg)
        img = np.full((self.size, self.size, 3), self.bg_color, dtype=np.uint8)

        # Расчет границ с умными отступами
        margin = int(self.size * self.rng.uniform(0.15, 0.25))
        bounds = (margin, margin, self.size - margin, self.size - margin)

        # Генерация концепции квартиры
        brief = self.planner.generate_brief()

        # Выбор и выполнение стратегии планировки
        strategy = self.strategies.get(brief.circulation_type, self.strategies['central'])
        walls, rooms = strategy.generate(bounds, ext_wall_thickness, int_wall_thickness,
                                         self.planner, brief)

        # Интеграция балконов при необходимости
        if brief.has_balcony:
            walls, rooms, balconies = self._integrate_balconies(walls, rooms, ext_wall_thickness)
        else:
            balconies = []

        # Отрисовка полов балконов в первую очередь
        self._draw_balcony_floor(img, balconies)

        # Отрисовка стен
        self._draw_walls(img, walls)

        # Расстановка мебели
        self.furniture_placer.place_furniture(img, rooms, walls)

        # Сбор меток
        labels: List[Tuple[int, float, float, float, float]] = []

        # Отрисовка элементов балкона
        balcony_labels = self._draw_balcony_features(img, balconies, int_wall_thickness)
        labels.extend(balcony_labels)

        # Размещение окон на внешних стенах (исключая ограждения)
        external_walls = [w for w in walls if w.is_external and not w.is_guardrail]
        for w in external_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            n_windows = self.rng.randint(1, max(2, int(length / 200)))
            for _ in range(n_windows):
                if self.rng.random() < 0.7:
                    box = self._place_window(img, w)
                    if box:
                        labels.append((1, *box))

        # Размещение дверей на внутренних стенах (исключая интерфейсы балконов)
        internal_walls = [w for w in walls if not w.is_external and not w.is_balcony_interface]
        door_spans: Dict[int, List[Tuple[int, int]]] = {}

        for w in internal_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            if length < int_wall_thickness * 3:
                continue
            n_doors = 1 if length < 250 * self.super_sampling else self.rng.randint(1, 2)
            spans = door_spans.setdefault(id(w), [])
            for _ in range(n_doors):
                if self.rng.random() < 0.8:
                    box = self._place_door(img, w, spans)
                    if box:
                        labels.append((0, *box))

        # Добавление меток комнат
        self._add_annotations(img, rooms)

        return img, labels

    # ═══════════════════════════════════════════════════════════════════════════
    # ИНТЕГРАЦИЯ БАЛКОНОВ
    # ═══════════════════════════════════════════════════════════════════════════

    def _integrate_balconies(self, walls: List[Wall], rooms: List[Room],
                             ext_th: int) -> Tuple[List[Wall], List[Room], List[Balcony]]:
        """Интеграция балконов путем вырезания пространства из подходящих комнат"""
        balconies = []
        new_walls = list(walls)
        candidates = [r for r in rooms if r.room_type in ["living", "kitchen", "bedroom"]]
        if not candidates:
            return new_walls, rooms, balconies
        selected = self.rng.sample(candidates, k=min(len(candidates), self.rng.randint(1, 2)))
        for room in selected:
            wall, side = self._find_external_wall(room, new_walls)
            if not wall:
                continue
            # Определение типа и глубины балкона
            dim = room.height if side in ['left', 'right'] else room.width
            if dim > self.size * 0.35:
                archetype = 'large_linear'
                depth = int(self.size * 0.08)
            else:
                archetype = 'compact_box'
                depth = int(self.size * 0.12)
            # Создание геометрии балкона
            balcony, interface_wall = self._carve_balcony(room, side, depth, ext_th)
            if balcony:
                new_walls.append(interface_wall)
                balconies.append(balcony)
                # Преобразование внешней стены в ограждение
                new_walls = self._convert_to_guardrail(new_walls, wall, balcony, side)
        return new_walls, rooms, balconies

    def _find_external_wall(self, room: Room, walls: List[Wall]) -> Tuple[Optional[Wall], Optional[str]]:
        """Поиск внешней стены, примыкающей к комнате"""
        for side in ['left', 'right', 'top', 'bottom']:
            for w in walls:
                if not w.is_external or w.is_guardrail:
                    continue
                if self._wall_matches_side(w, room, side):
                    return w, side
        return None, None

    def _wall_matches_side(self, wall: Wall, room: Room, side: str) -> bool:
        """Проверка соответствия стены указанной стороне комнаты"""
        tolerance = 20
        if side == 'left' and wall.x1 == wall.x2 and abs(wall.x1 - room.x1) < tolerance:
            return max(wall.y1, wall.y2) >= room.y1 and min(wall.y1, wall.y2) <= room.y2
        if side == 'right' and wall.x1 == wall.x2 and abs(wall.x1 - room.x2) < tolerance:
            return max(wall.y1, wall.y2) >= room.y1 and min(wall.y1, wall.y2) <= room.y2
        if side == 'top' and wall.y1 == wall.y2 and abs(wall.y1 - room.y1) < tolerance:
            return max(wall.x1, wall.x2) >= room.x1 and min(wall.x1, wall.x2) <= room.x2
        if side == 'bottom' and wall.y1 == wall.y2 and abs(wall.y1 - room.y2) < tolerance:
            return max(wall.x1, wall.x2) >= room.x1 and min(wall.x1, wall.x2) <= room.x2
        return False

    def _carve_balcony(self, room: Room, side: str, depth: int,
                       ext_th: int) -> Tuple[Optional[Balcony], Optional[Wall]]:
        """Вырезание пространства балкона из комнаты, возврат балкона и стены-интерфейса"""
        if side == 'left':
            old_x1 = room.x1
            room.x1 += depth
            bx1, by1, bx2, by2 = old_x1, room.y1, room.x1, room.y2
            interface = Wall(room.x1, room.y1, room.x1, room.y2, ext_th, False, is_balcony_interface=True)
        elif side == 'right':
            old_x2 = room.x2
            room.x2 -= depth
            bx1, by1, bx2, by2 = room.x2, room.y1, old_x2, room.y2
            interface = Wall(room.x2, room.y1, room.x2, room.y2, ext_th, False, is_balcony_interface=True)
        elif side == 'top':
            old_y1 = room.y1
            room.y1 += depth
            bx1, by1, bx2, by2 = room.x1, old_y1, room.x2, room.y1
            interface = Wall(room.x1, room.y1, room.x2, room.y1, ext_th, False, is_balcony_interface=True)
        elif side == 'bottom':
            old_y2 = room.y2
            room.y2 -= depth
            bx1, by1, bx2, by2 = room.x1, room.y2, room.x2, old_y2
            interface = Wall(room.x1, room.y2, room.x2, room.y2, ext_th, False, is_balcony_interface=True)
        else:
            return None, None
        archetype = 'large_linear' if max(bx2 - bx1, by2 - by1) > self.size * 0.3 else 'compact_box'
        return Balcony(bx1, by1, bx2, by2, side, archetype, interface), interface

    def _convert_to_guardrail(self, walls: List[Wall], target: Wall,
                              balcony: Balcony, side: str) -> List[Wall]:
        """Преобразование сегмента стены в тонкое ограждение с сохранением концевых точек"""
        if target in walls:
            walls.remove(target)
        if side in ['left', 'right']:
            w_start, w_end = min(target.y1, target.y2), max(target.y1, target.y2)
            b_start, b_end = min(balcony.y1, balcony.y2), max(balcony.y1, balcony.y2)
            if w_start < b_start:
                walls.append(Wall(target.x1, w_start, target.x2, b_start, target.thickness, True))
            walls.append(Wall(target.x1, b_start, target.x2, b_end, 2, True, is_guardrail=True))
            if w_end > b_end:
                walls.append(Wall(target.x1, b_end, target.x2, w_end, target.thickness, True))
        else:
            w_start, w_end = min(target.x1, target.x2), max(target.x1, target.x2)
            b_start, b_end = min(balcony.x1, balcony.x2), max(balcony.x1, balcony.x2)
            if w_start < b_start:
                walls.append(Wall(w_start, target.y1, b_start, target.y2, target.thickness, True))
            walls.append(Wall(b_start, target.y1, b_end, target.y2, 2, True, is_guardrail=True))
            if w_end > b_end:
                walls.append(Wall(b_end, target.y1, w_end, target.y2, target.thickness, True))
        return walls

    def _draw_balcony_floor(self, img: np.ndarray, balconies: List[Balcony]):
        """Отрисовка красной полосатой текстуры для полов балконов"""
        for b in balconies:
            x1, x2 = min(b.x1, b.x2), max(b.x1, b.x2)
            y1, y2 = min(b.y1, b.y2), max(b.y1, b.y2)
            step = int(self.size * 0.005)
            for y in range(y1, y2, step):
                cv2.line(img, (x1, y), (x2, y), BALCONY_STRIPE_COLOR, 1, LINE_TYPE)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "balcony")

    def _draw_walls(self, img: np.ndarray, walls: List[Wall]):
        """Отрисовка всех стен с соответствующей толщиной"""
        for w in walls:
            if w.is_guardrail:
                cv2.line(img, (w.x1, w.y1), (w.x2, w.y2), WALL_COLOR, 2, LINE_TYPE)
            else:
                half = w.thickness // 2
                x1, x2 = min(w.x1, w.x2), max(w.x1, w.x2)
                y1, y2 = min(w.y1, w.y2), max(w.y1, w.y2)
                if w.x1 == w.x2:  # Вертикальная
                    pt1, pt2 = (x1 - half, y1), (x1 + half, y2)
                else:  # Горизонтальная
                    pt1, pt2 = (x1, y1 - half), (x2, y1 + half)
                cv2.rectangle(img, pt1, pt2, WALL_COLOR, -1, LINE_TYPE)

    def _draw_balcony_features(self, img: np.ndarray, balconies: List[Balcony],
                               int_th: int) -> List[Tuple]:
        """Отрисовка дверей/окон на стенах-интерфейсах балконов"""
        labels = []
        for b in balconies:
            w = b.interface_wall
            if not w:
                continue
            is_vert = abs(w.x1 - w.x2) < abs(w.y1 - w.y2)
            start, end = (min(w.y1, w.y2), max(w.y1, w.y2)) if is_vert else (min(w.x1, w.x2), max(w.x1, w.x2))
            pad = int_th * 2
            start, end = start + pad, end - pad
            length = end - start
            if length <= 0:
                continue
            elements = []
            if b.archetype == 'large_linear':
                d_len = int(length * 0.2)
                w_len = int(length * 0.4)
                elements.append(('door', start, start + d_len))
                mid = start + d_len + int(length * 0.1)
                elements.append(('window', mid, mid + w_len))
                elements.append(('door', end - d_len, end))
            else:
                w_len = int(length * 0.5)
                d_len = int(length * 0.3)
                if self.rng.random() < 0.5:
                    elements.append(('window', start, start + w_len))
                    elements.append(('door', start + w_len + int(length * 0.05),
                                     start + w_len + int(length * 0.05) + d_len))
                else:
                    elements.append(('door', start, start + d_len))
                    elements.append(('window', start + d_len + int(length * 0.05),
                                     start + d_len + int(length * 0.05) + w_len))
            for el_type, s, e in elements:
                box = self._draw_interface_element(img, w, s, e, el_type, b.side)
                if box:
                    labels.append((0 if el_type == 'door' else 1, *box))
        return labels

    def _draw_interface_element(self, img: np.ndarray, wall: Wall, start: int, end: int,
                                el_type: str, side: str) -> Optional[Tuple[float, float, float, float]]:
        """Отрисовка двери или окна на стене-интерфейсе"""
        th = wall.thickness
        half_th = th // 2
        is_vert = abs(wall.x1 - wall.x2) < abs(wall.y1 - wall.y2)
        # Очистка проёма
        if is_vert:
            cx = wall.x1
            p1, p2 = (cx - half_th, start), (cx + half_th, end)
        else:
            cy = wall.y1
            p1, p2 = (start, cy - half_th), (end, cy + half_th)
        cv2.rectangle(img, p1, p2, self.bg_color, -1)
        # Отрисовка параллельных линий рамы
        line_th = self.rng.randint(1, 2)
        off = max(1, half_th // 8)
        if is_vert:
            cv2.line(img, (cx - off, start), (cx - off, end), WALL_COLOR, line_th, LINE_TYPE)
            cv2.line(img, (cx + off, start), (cx + off, end), WALL_COLOR, line_th, LINE_TYPE)
        else:
            cv2.line(img, (start, cy - off), (end, cy - off), WALL_COLOR, line_th, LINE_TYPE)
            cv2.line(img, (start, cy + off), (end, cy + off), WALL_COLOR, line_th, LINE_TYPE)
        # Добавление дуги открывания для дверей
        if el_type == 'door':
            swing_len = self.rng.randint((end - start) // 3, (end - start) // 2)
            if is_vert:
                swing_dir = 1 if side == 'left' else -1
                pt1 = (cx + off * swing_dir, start)
                pt2 = (cx + int(swing_len * swing_dir), start + swing_len)
            else:
                swing_dir = 1 if side == 'top' else -1
                pt1 = (start, cy + off * swing_dir)
                pt2 = (start + swing_len, cy + int(swing_len * swing_dir))
            cv2.line(img, pt1, pt2, WALL_COLOR, line_th, LINE_TYPE)
        return self._points_to_yolo([p1, p2])

    # ═══════════════════════════════════════════════════════════════════════════
    # РАЗМЕЩЕНИЕ ОКОН И ДВЕРЕЙ
    # ═══════════════════════════════════════════════════════════════════════════
    def _place_window(self, img: np.ndarray, w: Wall) -> Optional[Tuple[float, float, float, float]]:
        """Размещение окна с различными стилями"""
        pad = max(20, int(self.size * 0.01))
        is_horiz = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)
        window_type = self.rng.choices(
            ['standard', 'wide', 'narrow'],
            weights=[0.65, 0.20, 0.15]
        )[0]
        if window_type == 'wide':
            length_ratios = (0.07, 0.09, 0.11)
        elif window_type == 'narrow':
            length_ratios = (0.02, 0.03, 0.04)
        else:
            length_ratios = self.window_length_ratios
        typical_lengths = [int(self.size * r) for r in length_ratios]
        line_th = self.rng.randint(1, 2)
        gap_w = self.rng.randint(1, 8)
        if is_horiz:
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            center_y = (w.y1 + w.y2) // 2
            half = w.thickness // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None
            candidates = [L for L in typical_lengths if w.thickness * 3 <= L <= avail * 0.9]
            win_len = self.rng.choice(candidates) if candidates else self.rng.randint(
                max(int(w.thickness * 3), int(avail * 0.3)), int(avail * 0.9))
            offset_range = avail - win_len
            if offset_range < 0:
                return None
            for _ in range(10):
                start = axis_start + pad + self.rng.randint(0, offset_range)
                end = start + win_len
                offset = (gap_w + line_th) // 2
                line1_y = center_y - offset
                line2_y = center_y + offset
                y1 = center_y - half
                y2 = center_y + half
                if self.furniture_placer.check_collision(start, y1, end, y2, 15):
                    continue
                cv2.rectangle(img, (start, y1), (end, y2), self.bg_color, -1, LINE_TYPE)
                cv2.line(img, (start, line1_y), (end, line1_y), WALL_COLOR, line_th, LINE_TYPE)
                cv2.line(img, (start, line2_y), (end, line2_y), WALL_COLOR, line_th, LINE_TYPE)
                self.furniture_placer.add_occupied(start, y1, end, y2, "window")
                self._add_radiator(img, True, start, end, center_y, w.thickness)
                return self._points_to_yolo([(start, y1), (end, y2)])
        else:
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            center_x = (w.x1 + w.x2) // 2
            half = w.thickness // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None
            candidates = [L for L in typical_lengths if w.thickness * 3 <= L <= avail * 0.9]
            win_len = self.rng.choice(candidates) if candidates else self.rng.randint(
                max(int(w.thickness * 3), int(avail * 0.3)), int(avail * 0.9))
            offset_range = avail - win_len
            if offset_range < 0:
                return None
            for _ in range(10):
                start = axis_start + pad + self.rng.randint(0, offset_range)
                end = start + win_len
                offset = (gap_w + line_th) // 2
                line1_x = center_x - offset
                line2_x = center_x + offset
                x1 = center_x - half
                x2 = center_x + half
                if self.furniture_placer.check_collision(x1, start, x2, end, 15):
                    continue
                cv2.rectangle(img, (x1, start), (x2, end), self.bg_color, -1, LINE_TYPE)
                cv2.line(img, (line1_x, start), (line1_x, end), WALL_COLOR, line_th, LINE_TYPE)
                cv2.line(img, (line2_x, start), (line2_x, end), WALL_COLOR, line_th, LINE_TYPE)
                self.furniture_placer.add_occupied(x1, start, x2, end, "window")
                self._add_radiator(img, False, start, end, center_x, w.thickness)
                return self._points_to_yolo([(x1, start), (x2, end)])
        return None

    def _place_door(self, img: np.ndarray, w: Wall,
                    occupied_spans: List[Tuple[int, int]]) -> Optional[Tuple[float, float, float, float]]:
        """Размещение двери с дугой открывания"""
        door_th = self.rng.randint(2, 7)
        pad = max(40, int(w.thickness * 2.5))
        is_horiz = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)
        if is_horiz:
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            center_y = (w.y1 + w.y2) // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None
            door_len = int(self.size * self.rng.choice(self.door_length_ratios))
            if door_len > avail * 0.9:
                return None
            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                gap_start = axis_start + pad + self.rng.randint(0, usable)
                gap_end = gap_start + door_len
                if any(not (gap_end <= s or gap_start >= e) for s, e in occupied_spans):
                    continue
                half = w.thickness // 2
                y1, y2 = center_y - half, center_y + half
                hinge_left = self.rng.choice([True, False])
                open_up = self.rng.choice([True, False])
                swing_angle = self.rng.randint(70, 105)
                if hinge_left:
                    hinge_pos = (gap_start, center_y)
                    closed_angle = 0
                    open_angle = -swing_angle if open_up else swing_angle
                else:
                    hinge_pos = (gap_end, center_y)
                    closed_angle = 180
                    open_angle = 180 + swing_angle if open_up else 180 - swing_angle
                open_rad = np.deg2rad(open_angle)
                open_pos = (hinge_pos[0] + int(door_len * np.cos(open_rad)),
                            hinge_pos[1] + int(door_len * np.sin(open_rad)))
                perp = open_rad + np.pi / 2
                dx, dy = int(door_th / 2 * np.cos(perp)), int(door_th / 2 * np.sin(perp))
                corners = np.array([
                    [hinge_pos[0] - dx, hinge_pos[1] - dy],
                    [hinge_pos[0] + dx, hinge_pos[1] + dy],
                    [open_pos[0] + dx, open_pos[1] + dy],
                    [open_pos[0] - dx, open_pos[1] - dy]
                ], dtype=np.int32)
                all_x = [p[0] for p in corners] + [gap_start, gap_end]
                all_y = [p[1] for p in corners] + [y1, y2]
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
                if self.furniture_placer.check_collision(*bbox, 20):
                    continue
                occupied_spans.append((gap_start, gap_end))
                cv2.rectangle(img, (gap_start, y1), (gap_end, y2), self.bg_color, -1, LINE_TYPE)
                cv2.fillPoly(img, [corners], self.bg_color, LINE_TYPE)
                cv2.polylines(img, [corners], True, WALL_COLOR, 1, LINE_TYPE)
                cv2.ellipse(img, hinge_pos, (door_len, door_len), 0,
                            min(closed_angle, open_angle), max(closed_angle, open_angle),
                            WALL_COLOR, 1, LINE_TYPE)
                self.furniture_placer.add_occupied(*bbox, "door")
                return self._points_to_yolo([(gap_start, y1), (gap_end, y2), open_pos])
        else:
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            center_x = (w.x1 + w.x2) // 2
            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None
            door_len = int(self.size * self.rng.choice(self.door_length_ratios))
            if door_len > avail * 0.9:
                return None
            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                gap_start = axis_start + pad + self.rng.randint(0, usable)
                gap_end = gap_start + door_len
                if any(not (gap_end <= s or gap_start >= e) for s, e in occupied_spans):
                    continue
                half = w.thickness // 2
                x1, x2 = center_x - half, center_x + half
                hinge_top = self.rng.choice([True, False])
                open_right = self.rng.choice([True, False])
                swing_angle = self.rng.randint(70, 105)
                if hinge_top:
                    hinge_pos = (center_x, gap_start)
                    closed_angle = 90
                    open_angle = 90 - swing_angle if open_right else 90 + swing_angle
                else:
                    hinge_pos = (center_x, gap_end)
                    closed_angle = 270
                    open_angle = 270 + swing_angle if open_right else 270 - swing_angle
                open_rad = np.deg2rad(open_angle)
                open_pos = (hinge_pos[0] + int(door_len * np.cos(open_rad)),
                            hinge_pos[1] + int(door_len * np.sin(open_rad)))
                perp = open_rad + np.pi / 2
                dx, dy = int(door_th / 2 * np.cos(perp)), int(door_th / 2 * np.sin(perp))
                corners = np.array([
                    [hinge_pos[0] - dx, hinge_pos[1] - dy],
                    [hinge_pos[0] + dx, hinge_pos[1] + dy],
                    [open_pos[0] + dx, open_pos[1] + dy],
                    [open_pos[0] - dx, open_pos[1] - dy]
                ], dtype=np.int32)
                all_x = [p[0] for p in corners] + [x1, x2]
                all_y = [p[1] for p in corners] + [gap_start, gap_end]
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
                if self.furniture_placer.check_collision(*bbox, 20):
                    continue
                occupied_spans.append((gap_start, gap_end))
                cv2.rectangle(img, (x1, gap_start), (x2, gap_end), self.bg_color, -1, LINE_TYPE)
                cv2.fillPoly(img, [corners], self.bg_color, LINE_TYPE)
                cv2.polylines(img, [corners], True, WALL_COLOR, 1, LINE_TYPE)
                cv2.ellipse(img, hinge_pos, (door_len, door_len), 0,
                            min(closed_angle, open_angle), max(closed_angle, open_angle),
                            WALL_COLOR, 1, LINE_TYPE)
                self.furniture_placer.add_occupied(*bbox, "door")
                return self._points_to_yolo([(x1, gap_start), (x2, gap_end), open_pos])
        return None

    def _add_radiator(self, img: np.ndarray, horizontal: bool, axis_start: int,
                      axis_end: int, center_coord: int, wall_th: int):
        """Добавление радиатора под окнами"""
        length = axis_end - axis_start
        if length < wall_th * 2 or wall_th < 4:
            return
        rad_len = int(length * self.rng.uniform(0.7, 0.9))
        rad_height = max(3, int(wall_th * self.rng.uniform(0.6, 0.9)))
        gap = max(3, int(wall_th * self.rng.uniform(0.2, 0.6)))
        if horizontal:
            interior_dir = 1 if center_coord < self.size // 2 else -1
            cx = (axis_start + axis_end) // 2
            x1, x2 = max(0, cx - rad_len // 2), min(self.size - 1, cx + rad_len // 2)
            if interior_dir == 1:
                y1 = center_coord + wall_th // 2 + gap
                y2 = y1 + rad_height
            else:
                y2 = center_coord - wall_th // 2 - gap
                y1 = y2 - rad_height
            y1, y2 = max(0, y1), min(self.size - 1, y2)
            if y2 - y1 < 3 or x2 - x1 < 5:
                return
            if self.furniture_placer.check_collision(x1, y1, x2, y2):
                return
            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, LINE_TYPE)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "radiator")
            n_fins = self.rng.randint(4, 10)
            step = (x2 - x1) / (n_fins + 1)
            for i in range(1, n_fins + 1):
                fx = int(x1 + step * i)
                cv2.line(img, (fx, y1 + 1), (fx, y2 - 1), WALL_COLOR, 1, LINE_TYPE)
        else:
            interior_dir = 1 if center_coord < self.size // 2 else -1
            cy = (axis_start + axis_end) // 2
            y1, y2 = max(0, cy - rad_len // 2), min(self.size - 1, cy + rad_len // 2)
            if interior_dir == 1:
                x1 = center_coord + wall_th // 2 + gap
                x2 = x1 + rad_height
            else:
                x2 = center_coord - wall_th // 2 - gap
                x1 = x2 - rad_height
            x1, x2 = max(0, x1), min(self.size - 1, x2)
            if x2 - x1 < 3 or y2 - y1 < 5:
                return
            if self.furniture_placer.check_collision(x1, y1, x2, y2):
                return
            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, LINE_TYPE)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "radiator")
            n_fins = self.rng.randint(4, 10)
            step = (y2 - y1) / (n_fins + 1)
            for i in range(1, n_fins + 1):
                fy = int(y1 + step * i)
                cv2.line(img, (x1 + 1, fy), (x2 - 1, fy), WALL_COLOR, 1, LINE_TYPE)

    # ═══════════════════════════════════════════════════════════════════════════
    # УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════════════════════
    def _points_to_yolo(self, pts: List[Tuple[int, int]],
                        pad: int = 0) -> Optional[Tuple[float, float, float, float]]:
        """Преобразование пиксельных координат в формат YOLO"""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min = max(0, min(xs) - pad)
        x_max = min(self.size - 1, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(self.size - 1, max(ys) + pad)
        if x_max <= x_min or y_max <= y_min:
            return None
        cx = (x_min + x_max) / 2.0 / self.size
        cy = (y_min + y_max) / 2.0 / self.size
        w = (x_max - x_min) / self.size
        h = (y_max - y_min) / self.size
        return (
            min(max(cx, 0.0), 1.0),
            min(max(cy, 0.0), 1.0),
            min(max(w, 0.0), 1.0),
            min(max(h, 0.0), 1.0),
        )

    def _apply_rotation(self, img: np.ndarray, labels: List) -> Tuple[np.ndarray, List]:
        """Применение аугментации случайного поворота"""
        size = img.shape[0]
        angle = self.rng.uniform(-15, 15)
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (size, size), borderValue=self.bg_color)
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        new_labels = []
        for cls_id, cx, cy, w, h in labels:
            cx_px, cy_px = cx * size, cy * size
            x_rel, y_rel = cx_px - center[0], cy_px - center[1]
            x_rot = x_rel * cos_a - y_rel * sin_a + center[0]
            y_rot = x_rel * sin_a + y_rel * cos_a + center[1]
            new_labels.append((cls_id, x_rot / size, y_rot / size, w, h))
        return rotated, new_labels

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Применение аугментаций изображения"""
        if self.rng.random() < 0.5:
            ksize = self.rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), self.rng.uniform(0.3, 0.7))
        if self.rng.random() < 0.3:
            noise = self.nprng.normal(0, self.rng.uniform(2, 8), img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    def _add_annotations(self, img: np.ndarray, rooms: List[Room]):
        """Добавление меток площади комнат"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.0007 * self.size
        thickness = max(1, int(self.size * 0.0012))
        for room in rooms:
            if room.room_type == 'balcony':
                continue
            cx, cy = room.center
            txt = f"{self.rng.uniform(4.0, 50.0):.1f}".replace(".", ",")
            cv2.putText(img, txt, (cx, cy), font, base_scale, WALL_COLOR, thickness, LINE_TYPE)


# ═══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Умный генератор планировок v3.0")
    parser.add_argument("--output", type=str, default="smart_dataset", help="Выходная директория")
    parser.add_argument("--count", type=int, default=10, help="Количество изображений")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument("--size", type=int, default=1024, help="Размер изображения")
    args = parser.parse_args()
    gen = SmartFloorPlanGenerator(img_size=args.size, seed=args.seed)
    gen.generate_dataset(args.count, args.output)
    print(f"Сгенерировано {args.count} умных планировок в {args.output}")
