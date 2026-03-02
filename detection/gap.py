"""
Система анализа планировок помещений - Модуль геометрической детекции проёмов
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Ray casting для поиска разрывов между стенам, обработка пересекающихся стен
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional


@dataclass
class Opening:
    """
    Структура данных для описания проёма.

    Атрибуты:
        id: Уникальный идентификатор проёма
        x1, y1, x2, y2: Координаты ограничивающего прямоугольника
        width, height: Размеры проёма
        area: Площадь проёма
        opening_type: Тип проёма (дверь/окно/неизвестно)
        confidence: Уверенность детекции
        source_rects: Список ID исходных прямоугольников
        yolo_matches: Список совпадений с YOLO детекциями
        is_valid: Флаг валидности проёма
        status: Статус валидации
    """
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    opening_type: str
    confidence: float
    source_rects: List[int]
    yolo_matches: List[int]
    is_valid: bool = True
    status: str = "valid"


@dataclass
class RayCastResult:
    """
    Результат трассировки луча между прямоугольниками стен.

    Атрибуты:
        source_rect_id: ID исходного прямоугольника
        ray_x1, ray_y1, ray_x2, ray_y2: Координаты луча
        target_rect_id: ID целевого прямоугольника
        ray_length: Длина луча
        overlap_ratio: Коэффициент перекрытия
        occlusion_ratio: Коэффициент окклюзии
    """
    source_rect_id: int
    ray_x1: int
    ray_y1: int
    ray_x2: int
    ray_y2: int
    target_rect_id: int
    ray_length: float
    overlap_ratio: float
    occlusion_ratio: float


class GeometricOpeningDetector:
    """
    Детектор геометрических проёмов в стенах.

    Использует метод ray casting для обнаружения разрывов между
    прямоугольниками стен и применяет множественные критерии валидации
    для фильтрации ложных детекций.
    """

    def __init__(self,
                 max_extension_length: int = 300,
                 min_extension_length: int = 10,
                 min_overlap_ratio: float = 0.5,
                 max_occlusion_ratio: float = 0.8,
                 small_rect_aspect_threshold: float = 3.0,
                 small_rect_min_side: int = 20,
                 merge_distance: int = 0,
                 merge_min_overlap_ratio: float = 0.1,
                 merge_parallel_overlap_ratio: float = 0.1,
                 yolo_iou_threshold: float = 0.3,
                 adjacency_max_gap: int = 3,
                 adjacency_min_ratio: float = 0.6,
                 max_opening_thickness_factor: float = 8.0,
                 min_opening_length_factor: float = 1.0,
                 max_opening_length_factor: float = 8.0,
                 max_opening_aspect_ratio: float = 8.0,
                 min_opening_aspect_ratio: float = 5.0,
                 connectivity_tolerance: int = 2,
                 min_gap_width_m: float = 0.30,
                 max_gap_width_m: float = 1.4,
                 max_wall_touch_length_m: float = 0.45,
                 max_non_white_ratio: float = 0.35,
                 white_threshold: int = 240,
                 third_wall_check_tolerance: int = 3):
        """
        Инициализация детектора проёмов.

        Параметры ray casting:
            max_extension_length: Максимальная длина луча в пикселях
            min_extension_length: Минимальная длина луча в пикселях
            min_overlap_ratio: Минимальный коэффициент перекрытия
            max_occlusion_ratio: Максимальный коэффициент окклюзии

        Параметры геометрии:
            small_rect_aspect_threshold: Порог соотношения сторон для малых прямоугольников
            small_rect_min_side: Минимальная сторона малого прямоугольника
            connectivity_tolerance: Допуск для проверки соединённости

        Параметры объединения:
            merge_distance: Расстояние для объединения проёмов
            merge_min_overlap_ratio: Минимальное перекрытие для объединения
            merge_parallel_overlap_ratio: Перекрытие для параллельных проёмов

        Параметры валидации в реальных единицах:
            min_gap_width_m: Минимальная ширина проёма в метрах (0.6м = 60см)
            max_gap_width_m: Максимальная ширина проёма в метрах (0.9м = 90см)
            max_wall_touch_length_m: Макс. толщина касания стены в метрах (0.35м = 35см)

        Параметры валидации содержимого:
            max_non_white_ratio: Максимальная доля не-белых пикселей (25%)
            white_threshold: Порог для определения белого пикселя (240)
            third_wall_check_tolerance: Допуск для проверки третьей стены
        """
        # Параметры ray casting
        self.max_extension_length = int(max_extension_length)
        self.min_extension_length = int(min_extension_length)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.max_occlusion_ratio = float(max_occlusion_ratio)

        # Параметры геометрии
        self.small_rect_aspect_threshold = float(small_rect_aspect_threshold)
        self.small_rect_min_side = int(small_rect_min_side)
        self.connectivity_tolerance = int(connectivity_tolerance)

        # Параметры объединения
        self.merge_distance = int(merge_distance)
        self.merge_min_overlap_ratio = float(merge_min_overlap_ratio)
        self.merge_parallel_overlap_ratio = float(merge_parallel_overlap_ratio)

        # Параметры YOLO
        self.yolo_iou_threshold = float(yolo_iou_threshold)

        # Параметры смежности
        self.adjacency_max_gap = int(adjacency_max_gap)
        self.adjacency_min_ratio = float(adjacency_min_ratio)

        # Факторы проёмов
        self.max_opening_thickness_factor = float(max_opening_thickness_factor)
        self.min_opening_length_factor = float(min_opening_length_factor)
        self.max_opening_length_factor = float(max_opening_length_factor)
        self.max_opening_aspect_ratio = float(max_opening_aspect_ratio)
        self.min_opening_aspect_ratio = float(min_opening_aspect_ratio)

        # Ограничения реального мира
        self.min_gap_width_m = float(min_gap_width_m)
        self.max_gap_width_m = float(max_gap_width_m)

        # Параметры валидации
        self.max_wall_touch_length_m = float(max_wall_touch_length_m)
        self.max_non_white_ratio = float(max_non_white_ratio)
        self.white_threshold = int(white_threshold)
        self.third_wall_check_tolerance = int(third_wall_check_tolerance)

        # Отслеживание удалённых проёмов для визуализации
        self.last_deleted_openings: List[Opening] = []

    # ============================================================
    # ГЕОМЕТРИЧЕСКИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # ============================================================

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int]) -> float:
        """Вычисление Intersection over Union для двух прямоугольников."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0.0

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return float(inter_area) / float(union_area) if union_area > 0 else 0.0

    @staticmethod
    def _rect_intersection_area(box1: Tuple[int, int, int, int],
                                box2: Tuple[int, int, int, int]) -> int:
        """Вычисление площади пересечения двух прямоугольников."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        return max(0, xi2 - xi1) * max(0, yi2 - yi1)

    @staticmethod
    def _boxes_intersect(box1: Tuple[int, int, int, int],
                         box2: Tuple[int, int, int, int]) -> bool:
        """Проверка пересечения двух прямоугольников."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    @staticmethod
    def _box_area(box: Tuple[int, int, int, int]) -> int:
        """Вычисление площади прямоугольника."""
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def _box_orientation(box: Tuple[int, int, int, int]) -> str:
        """Определение ориентации прямоугольника."""
        w = max(0, box[2] - box[0])
        h = max(0, box[3] - box[1])
        return 'vertical' if h >= w else 'horizontal'

    @staticmethod
    def _point_in_box(px: int, py: int, box: Tuple[int, int, int, int]) -> bool:
        """Проверка нахождения точки внутри прямоугольника."""
        return box[0] <= px <= box[2] and box[1] <= py <= box[3]

    def _is_small_rectangle(self, rect) -> bool:
        """Определение, является ли прямоугольник малым."""
        width = rect.width
        height = rect.height
        aspect = max(width, height) / (min(width, height) + 1e-6)
        min_side = min(width, height)
        return aspect < self.small_rect_aspect_threshold or min_side < self.small_rect_min_side

    def _are_rects_connected(self, rect1, rect2) -> bool:
        """Проверка соединённости двух прямоугольников."""
        tol = self.connectivity_tolerance
        x1 = max(0, rect1.x1 - tol)
        y1 = max(0, rect1.y1 - tol)
        x2 = rect1.x2 + tol
        y2 = rect1.y2 + tol

        ix1 = max(x1, rect2.x1)
        iy1 = max(y1, rect2.y1)
        ix2 = min(x2, rect2.x2)
        iy2 = min(y2, rect2.y2)
        return ix2 > ix1 and iy2 > iy1

    def _get_edge_touch_length(self, opening: Opening, rectangles: List,
                               edge: str, tolerance: int) -> float:
        """Вычисление длины касания края проёма со стенами."""
        total_touch = 0.0

        for rect in rectangles:
            if edge == 'top':
                if rect.y2 >= opening.y1 - tolerance and rect.y2 <= opening.y1 + tolerance:
                    if rect.y1 <= opening.y1 and rect.y2 >= opening.y1 - tolerance:
                        overlap = max(0, min(opening.x2, rect.x2) - max(opening.x1, rect.x1))
                        total_touch += overlap

            elif edge == 'bottom':
                if rect.y1 >= opening.y2 - tolerance and rect.y1 <= opening.y2 + tolerance:
                    if rect.y2 >= opening.y2 and rect.y1 <= opening.y2 + tolerance:
                        overlap = max(0, min(opening.x2, rect.x2) - max(opening.x1, rect.x1))
                        total_touch += overlap

            elif edge == 'left':
                if rect.x2 >= opening.x1 - tolerance and rect.x2 <= opening.x1 + tolerance:
                    if rect.x1 <= opening.x1 and rect.x2 >= opening.x1 - tolerance:
                        overlap = max(0, min(opening.y2, rect.y2) - max(opening.y1, rect.y1))
                        total_touch += overlap

            elif edge == 'right':
                if rect.x1 >= opening.x2 - tolerance and rect.x1 <= opening.x2 + tolerance:
                    if rect.x2 >= opening.x2 and rect.x1 <= opening.x2 + tolerance:
                        overlap = max(0, min(opening.y2, rect.y2) - max(opening.y1, rect.y1))
                        total_touch += overlap

        return total_touch

    # ============================================================
    # ВАЛИДАЦИЯ ЦВЕТОВОГО СОДЕРЖИМОГО ПРОЁМА
    # ============================================================

    def _validate_gap_color_content(self, opening: Opening,
                                    image: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Валидация содержимого проёма по цвету.
        Проверяет, что область проёма преимущественно белая (пустая).

        Параметры:
            opening: Проверяемый проём
            image: Изображение для анализа

        Возвращает:
            (is_valid, trimmed_coords) - валидность и обрезанные координаты
        """
        if image is None:
            return True, None

        if len(image.shape) == 3:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            h, w = image.shape
            gray = image

        # Ограничение координат размерами изображения
        x1 = max(0, opening.x1)
        y1 = max(0, opening.y1)
        x2 = min(w, opening.x2)
        y2 = min(h, opening.y2)

        if x2 <= x1 or y2 <= y1:
            return False, None

        roi = gray[y1:y2, x1:x2]
        total_pixels = roi.size

        if total_pixels == 0:
            return False, None

        # Подсчёт не-белых пикселей
        non_white_mask = roi < self.white_threshold
        non_white_pixels = np.sum(non_white_mask)
        non_white_ratio = non_white_pixels / total_pixels

        if non_white_ratio > self.max_non_white_ratio:
            return False, None

        # Обрезка краёв с не-белым содержимым
        gap_width = x2 - x1
        gap_height = y2 - y1
        is_horizontal = gap_width > gap_height

        trimmed_x1, trimmed_y1 = x1, y1
        trimmed_x2, trimmed_y2 = x2, y2

        min_white_ratio = 0.80

        if is_horizontal:
            # Обрезка слева
            for col_offset in range(gap_width):
                if col_offset >= gap_width:
                    break
                col = roi[:, col_offset]
                white_ratio = np.sum(col >= self.white_threshold) / len(col)
                if white_ratio >= min_white_ratio:
                    trimmed_x1 = x1 + col_offset
                    break
            else:
                return False, None

            # Обрезка справа
            for col_offset in range(gap_width - 1, -1, -1):
                if col_offset < 0:
                    break
                col = roi[:, col_offset]
                white_ratio = np.sum(col >= self.white_threshold) / len(col)
                if white_ratio >= min_white_ratio:
                    trimmed_x2 = x1 + col_offset + 1
                    break
            else:
                return False, None
        else:
            # Обрезка сверху
            for row_offset in range(gap_height):
                if row_offset >= gap_height:
                    break
                row = roi[row_offset, :]
                white_ratio = np.sum(row >= self.white_threshold) / len(row)
                if white_ratio >= min_white_ratio:
                    trimmed_y1 = y1 + row_offset
                    break
            else:
                return False, None

            # Обрезка снизу
            for row_offset in range(gap_height - 1, -1, -1):
                if row_offset < 0:
                    break
                row = roi[row_offset, :]
                white_ratio = np.sum(row >= self.white_threshold) / len(row)
                if white_ratio >= min_white_ratio:
                    trimmed_y2 = y1 + row_offset + 1
                    break
            else:
                return False, None

        if trimmed_x2 <= trimmed_x1 or trimmed_y2 <= trimmed_y1:
            return False, None

        new_width = trimmed_x2 - trimmed_x1
        new_height = trimmed_y2 - trimmed_y1
        if new_width < 3 or new_height < 3:
            return False, None

        if trimmed_x1 != x1 or trimmed_y1 != y1 or trimmed_x2 != x2 or trimmed_y2 != y2:
            return True, (trimmed_x1, trimmed_y1, trimmed_x2, trimmed_y2)

        return True, None

    # ============================================================
    # ПРОВЕРКИ ПЕРЕКРЫТИЙ
    # ============================================================

    def _check_door_window_overlap(self, opening: Opening,
                                   door_window_boxes: List) -> bool:
        """Проверка перекрытия проёма с дверями/окнами."""
        if not door_window_boxes:
            return False

        op_box = (opening.x1, opening.y1, opening.x2, opening.y2)
        op_center_x = (opening.x1 + opening.x2) // 2
        op_center_y = (opening.y1 + opening.y2) // 2

        for idx, bbox in enumerate(door_window_boxes):
            if hasattr(bbox, 'bbox'):
                box = tuple(map(int, bbox.bbox))
            elif hasattr(bbox, 'x1'):
                box = (int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2))
            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                box = tuple(map(int, bbox[:4]))
            else:
                continue

            # Расширение bbox для более мягкой проверки
            expand = 3
            expanded_box = (box[0] - expand, box[1] - expand,
                            box[2] + expand, box[3] + expand)

            # Проверка пересечения
            if self._boxes_intersect(op_box, expanded_box):
                inter_area = self._rect_intersection_area(op_box, expanded_box)
                if inter_area > 0:
                    return True

            # Проверка центров
            if self._point_in_box(op_center_x, op_center_y, expanded_box):
                return True

            dw_center_x = (box[0] + box[2]) // 2
            dw_center_y = (box[1] + box[3]) // 2
            if self._point_in_box(dw_center_x, dw_center_y, op_box):
                return True

            # Проверка IoU
            iou = self._compute_iou(op_box, box)
            if iou > 0.05:
                return True

        return False

    def _check_wall_touch_length(self, opening: Opening,
                                 rectangles: List,
                                 pixel_scale: float) -> bool:
        """
        Проверка толщины касания стены (максимум 35см).

        Параметры:
            opening: Проверяемый проём
            rectangles: Список прямоугольников стен
            pixel_scale: Масштаб в метрах на пиксель
        """
        if pixel_scale <= 0:
            return False

        max_touch_px = self.max_wall_touch_length_m / pixel_scale

        width = opening.x2 - opening.x1
        height = opening.y2 - opening.y1

        thickness_px = min(width, height)

        if thickness_px > max_touch_px:
            return True

        return False

    def _check_third_wall_dead_end(self, opening: Opening, rectangles: List) -> bool:
        """
        Детекция тупиковых проёмов.
        Отклоняет проём, если он касается стен с 3 или более сторон.
        """
        width = opening.x2 - opening.x1
        height = opening.y2 - opening.y1
        tolerance = self.third_wall_check_tolerance

        # Получение длины касания для всех 4 сторон
        top_touch = self._get_edge_touch_length(opening, rectangles, 'top', tolerance)
        bottom_touch = self._get_edge_touch_length(opening, rectangles, 'bottom', tolerance)
        left_touch = self._get_edge_touch_length(opening, rectangles, 'left', tolerance)
        right_touch = self._get_edge_touch_length(opening, rectangles, 'right', tolerance)

        # Расчёт коэффициента покрытия для каждой стороны
        top_ratio = top_touch / (width + 1e-6)
        bottom_ratio = bottom_touch / (width + 1e-6)
        left_ratio = left_touch / (height + 1e-6)
        right_ratio = right_touch / (height + 1e-6)

        # Порог для определения блокированной стороны
        block_threshold = 0.3

        sides_blocked_count = 0
        if top_ratio > block_threshold: sides_blocked_count += 1
        if bottom_ratio > block_threshold: sides_blocked_count += 1
        if left_ratio > block_threshold: sides_blocked_count += 1
        if right_ratio > block_threshold: sides_blocked_count += 1

        # Отклонение только если 3 или более сторон заблокированы
        if sides_blocked_count >= 3:
            return True

        return False

    def _check_min_aspect_ratio(self, opening: Opening) -> bool:
        """
        Проверка минимального соотношения сторон проёма (5:1).

        Проёмы в стенах должны быть вытянутыми — одна сторона (длина проёма)
        значительно больше другой (толщина стены). Слишком «квадратные»
        детекции обычно являются ложными срабатываниями.

        Параметры:
            opening: Проверяемый проём

        Возвращает:
            True если проём слишком квадратный (невалидный), False если OK
        """
        w = opening.width
        h = opening.height

        if w <= 0 or h <= 0:
            return True  # Невалидный — нулевые размеры

        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        if aspect_ratio < self.min_opening_aspect_ratio:
            return True  # Невалидный — слишком квадратный

        return False

    # ============================================================
    # RAY CASTING
    # ============================================================

    def _cast_ray_from_side(self, rect, side: str, all_rects: List, rect_lookup: Dict[int, object]) -> List[
        RayCastResult]:
        """
        Трассировка луча от стороны прямоугольника.

        Параметры:
            rect: Исходный прямоугольник
            side: Сторона ('top', 'bottom', 'left', 'right')
            all_rects: Все прямоугольники
            rect_lookup: Словарь прямоугольников по ID
        """
        results: List[RayCastResult] = []

        # Определение параметров луча
        if side == 'top':
            ray_x1, ray_y1 = rect.x1, rect.y1 - self.max_extension_length
            ray_x2, ray_y2 = rect.x2, rect.y1
            direction = 'vertical'
            ray_breadth = rect.width
        elif side == 'bottom':
            ray_x1, ray_y1 = rect.x1, rect.y2
            ray_x2, ray_y2 = rect.x2, rect.y2 + self.max_extension_length
            direction = 'vertical'
            ray_breadth = rect.width
        elif side == 'left':
            ray_x1, ray_y1 = rect.x1 - self.max_extension_length, rect.y1
            ray_x2, ray_y2 = rect.x1, rect.y2
            direction = 'horizontal'
            ray_breadth = rect.height
        else:
            ray_x1, ray_y1 = rect.x2, rect.y1
            ray_x2, ray_y2 = rect.x2 + self.max_extension_length, rect.y2
            direction = 'horizontal'
            ray_breadth = rect.height

        # Ограничение координат
        ray_x1, ray_y1 = max(0, ray_x1), max(0, ray_y1)
        ray_x2, ray_y2 = max(ray_x1, ray_x2), max(ray_y1, ray_y2)

        ray_box = (min(ray_x1, ray_x2), min(ray_y1, ray_y2), max(ray_x1, ray_x2), max(ray_y1, ray_y2))

        # Поиск пересечений с другими прямоугольниками
        for target_rect in all_rects:
            if target_rect.id == rect.id:
                continue
            if self._are_rects_connected(rect, target_rect):
                continue

            target_box = (target_rect.x1, target_rect.y1, target_rect.x2, target_rect.y2)
            inter_area = self._rect_intersection_area(ray_box, target_box)
            if inter_area == 0:
                continue

            # Расчёт коэффициента перекрытия
            if direction == 'vertical':
                overlap_width = min(ray_box[2], target_box[2]) - max(ray_box[0], target_box[0])
                overlap_ratio = float(overlap_width) / float(ray_breadth + 1e-6)
            else:
                overlap_height = min(ray_box[3], target_box[3]) - max(ray_box[1], target_box[1])
                overlap_ratio = float(overlap_height) / float(ray_breadth + 1e-6)

            if overlap_ratio < self.min_overlap_ratio:
                continue

            # Расчёт фактической длины луча
            if direction == 'vertical':
                if side == 'top':
                    actual_ray_length = abs(rect.y1 - target_rect.y2)
                    actual_ray_x1 = max(ray_box[0], target_box[0])
                    actual_ray_x2 = min(ray_box[2], target_box[2])
                    actual_ray_y1, actual_ray_y2 = target_rect.y2, rect.y1
                else:
                    actual_ray_length = abs(target_rect.y1 - rect.y2)
                    actual_ray_x1 = max(ray_box[0], target_box[0])
                    actual_ray_x2 = min(ray_box[2], target_box[2])
                    actual_ray_y1, actual_ray_y2 = rect.y2, target_rect.y1
            else:
                if side == 'left':
                    actual_ray_length = abs(rect.x1 - target_rect.x2)
                    actual_ray_y1 = max(ray_box[1], target_box[1])
                    actual_ray_y2 = min(ray_box[3], target_box[3])
                    actual_ray_x1, actual_ray_x2 = target_rect.x2, rect.x1
                else:
                    actual_ray_length = abs(target_rect.x1 - rect.x2)
                    actual_ray_y1 = max(ray_box[1], target_box[1])
                    actual_ray_y2 = min(ray_box[3], target_box[3])
                    actual_ray_x1, actual_ray_x2 = rect.x2, target_rect.x1

            if actual_ray_length < self.min_extension_length:
                continue

            actual_ray_box = (min(actual_ray_x1, actual_ray_x2), min(actual_ray_y1, actual_ray_y2),
                              max(actual_ray_x1, actual_ray_x2), max(actual_ray_y1, actual_ray_y2))

            ray_area = (actual_ray_box[2] - actual_ray_box[0]) * (actual_ray_box[3] - actual_ray_box[1])
            if ray_area <= 0:
                continue

            # Расчёт окклюзии
            occlusion_area = 0
            for other_rect in all_rects:
                if other_rect.id in (rect.id, target_rect.id):
                    continue
                other_box = (other_rect.x1, other_rect.y1, other_rect.x2, other_rect.y2)
                occlusion_area += self._rect_intersection_area(actual_ray_box, other_box)

            occlusion_ratio = float(occlusion_area) / float(ray_area + 1e-6)
            if occlusion_ratio > self.max_occlusion_ratio:
                continue

            results.append(RayCastResult(
                source_rect_id=rect.id,
                ray_x1=actual_ray_box[0],
                ray_y1=actual_ray_box[1],
                ray_x2=actual_ray_box[2],
                ray_y2=actual_ray_box[3],
                target_rect_id=target_rect.id,
                ray_length=actual_ray_length,
                overlap_ratio=overlap_ratio,
                occlusion_ratio=occlusion_ratio
            ))

        return results

    def _cast_rays_from_rectangle(self, rect, all_rects, rect_lookup):
        """Трассировка лучей от всех подходящих сторон прямоугольника."""
        is_small = self._is_small_rectangle(rect)
        if is_small:
            sides = ['top', 'bottom', 'left', 'right']
        else:
            sides = ['left', 'right'] if rect.width >= rect.height else ['top', 'bottom']
        all_results = []
        for side in sides:
            all_results.extend(self._cast_ray_from_side(rect, side, all_rects, rect_lookup))
        return all_results

    # ============================================================
    # ОБЪЕДИНЕНИЕ ПРОЁМОВ
    # ============================================================

    def _boxes_should_merge(self, box_a, box_b):
        """Определение необходимости объединения двух прямоугольников."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        if inter_area > 0:
            area_a = self._box_area(box_a)
            area_b = self._box_area(box_b)
            overlap_ratio = inter_area / float(min(area_a, area_b) + 1e-6)
            return overlap_ratio >= self.merge_min_overlap_ratio

        ori_a, ori_b = self._box_orientation(box_a), self._box_orientation(box_b)
        if ori_a != ori_b:
            return False

        if ori_a == 'vertical':
            gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
            if gap_x > self.merge_distance:
                return False
            overlap_y = min(ay2, by2) - max(ay1, by1)
            return (overlap_y / float(min(ay2 - ay1, by2 - by1) + 1e-6)) >= self.merge_parallel_overlap_ratio
        else:
            gap_y = max(0, max(ay1, by1) - min(ay2, by2))
            if gap_y > self.merge_distance:
                return False
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            return (overlap_x / float(min(ax2 - ax1, bx2 - bx1) + 1e-6)) >= self.merge_parallel_overlap_ratio

    def _merge_adjacent_openings(self, openings: List[Opening]) -> List[Opening]:
        """Объединение смежных проёмов."""
        if not openings:
            return []
        merged: List[Opening] = []
        used: Set[int] = set()

        for i, opening_i in enumerate(openings):
            if i in used:
                continue
            merged_box = [opening_i.x1, opening_i.y1, opening_i.x2, opening_i.y2]
            merged_sources = set(opening_i.source_rects)
            group_idxs = [i]

            changed = True
            while changed:
                changed = False
                cur_box = tuple(merged_box)
                for j, opening_j in enumerate(openings):
                    if j in used or j in group_idxs:
                        continue
                    box_j = (opening_j.x1, opening_j.y1, opening_j.x2, opening_j.y2)
                    if not self._boxes_should_merge(cur_box, box_j):
                        continue
                    merged_box[0] = min(merged_box[0], box_j[0])
                    merged_box[1] = min(merged_box[1], box_j[1])
                    merged_box[2] = max(merged_box[2], box_j[2])
                    merged_box[3] = max(merged_box[3], box_j[3])
                    merged_sources.update(opening_j.source_rects)
                    group_idxs.append(j)
                    used.add(j)
                    changed = True

            used.update(group_idxs)
            width = merged_box[2] - merged_box[0]
            height = merged_box[3] - merged_box[1]
            merged.append(Opening(
                id=len(merged),
                x1=merged_box[0],
                y1=merged_box[1],
                x2=merged_box[2],
                y2=merged_box[3],
                width=width,
                height=height,
                area=width * height,
                opening_type='unknown',
                confidence=0.0,
                source_rects=sorted(list(merged_sources)),
                yolo_matches=[]
            ))
        return merged

    # ============================================================
    # ОБРАБОТКА ПЕРЕСЕКАЮЩИХСЯ СТЕН
    # ============================================================

    def _handle_crossing_walls(self, openings: List[Opening], rectangles: List) -> List[Opening]:
        """Обработка проёмов, пересекающихся со стенами."""
        processed_openings = []

        for op in openings:
            if not op.is_valid:
                processed_openings.append(op)
                continue

            op_box = (op.x1, op.y1, op.x2, op.y2)
            op_width = op.x2 - op.x1
            op_height = op.y2 - op.y1

            is_horizontal_gap = op_width > op_height
            crossing_rects = []

            # Поиск пересекающихся прямоугольников
            for rect in rectangles:
                rect_box = (rect.x1, rect.y1, rect.x2, rect.y2)
                inter_area = self._rect_intersection_area(op_box, rect_box)

                if inter_area > 0:
                    overlap_ratio = inter_area / (op.area + 1e-6)
                    if overlap_ratio > 0.95:
                        op.is_valid = False
                        op.status = "obstructed_full"
                        break

                    if is_horizontal_gap:
                        if rect.x1 > op.x1 and rect.x2 < op.x2:
                            crossing_rects.append(rect)
                        elif (rect.x1 < op.x1 and rect.x2 > op.x1) or (rect.x1 < op.x2 and rect.x2 > op.x2):
                            crossing_rects.append(rect)
                    else:
                        if rect.y1 > op.y1 and rect.y2 < op.y2:
                            crossing_rects.append(rect)
                        elif (rect.y1 < op.y1 and rect.y2 > op.y1) or (rect.y1 < op.y2 and rect.y2 > op.y2):
                            crossing_rects.append(rect)

            if not op.is_valid:
                processed_openings.append(op)
                continue

            if not crossing_rects:
                processed_openings.append(op)
                continue

            # Обработка пересечений
            if is_horizontal_gap:
                cuts = []
                for cr in crossing_rects:
                    c_start = max(op.x1, cr.x1)
                    c_end = min(op.x2, cr.x2)
                    if c_end > c_start:
                        cuts.append((c_start, c_end))

                cuts.sort()
                free_segments = []
                current_x = op.x1

                for c_start, c_end in cuts:
                    if c_start > current_x:
                        free_segments.append((current_x, c_start))
                    current_x = max(current_x, c_end)

                if current_x < op.x2:
                    free_segments.append((current_x, op.x2))

                if not free_segments:
                    op.is_valid = False
                    op.status = "fully_blocked"
                    processed_openings.append(op)
                else:
                    # Выбор наибольшего сегмента
                    best_seg = max(free_segments, key=lambda s: s[1] - s[0])
                    op.x1, op.x2 = int(best_seg[0]), int(best_seg[1])
                    op.width = op.x2 - op.x1
                    op.area = op.width * op.height
                    processed_openings.append(op)
            else:
                cuts = []
                for cr in crossing_rects:
                    c_start = max(op.y1, cr.y1)
                    c_end = min(op.y2, cr.y2)
                    if c_end > c_start:
                        cuts.append((c_start, c_end))

                cuts.sort()
                free_segments = []
                current_y = op.y1

                for c_start, c_end in cuts:
                    if c_start > current_y:
                        free_segments.append((current_y, c_start))
                    current_y = max(current_y, c_end)

                if current_y < op.y2:
                    free_segments.append((current_y, op.y2))

                if not free_segments:
                    op.is_valid = False
                    op.status = "fully_blocked"
                    processed_openings.append(op)
                else:
                    best_seg = max(free_segments, key=lambda s: s[1] - s[0])
                    op.y1, op.y2 = int(best_seg[0]), int(best_seg[1])
                    op.height = op.y2 - op.y1
                    op.area = op.width * op.height
                    processed_openings.append(op)

        return processed_openings

    def _estimate_wall_thickness(self, rectangles):
        """Оценка средней толщины стены."""
        if not rectangles:
            return 0.0
        return float(np.median([min(r.width, r.height) for r in rectangles]))

    def _filter_by_real_world_size_strict(self, openings: List[Opening],
                                          pixel_scale: float,
                                          wall_thickness: float) -> List[Opening]:
        """Фильтрация проёмов по реальным размерам."""
        if not openings or pixel_scale <= 0:
            return openings

        min_gap_px = self.min_gap_width_m / pixel_scale
        max_gap_px = self.max_gap_width_m / pixel_scale

        for op in openings:
            if not op.is_valid:
                continue

            w = op.width
            h = op.height

            # Определение длины проёма на основе толщины стены
            dist_w = abs(w - wall_thickness)
            dist_h = abs(h - wall_thickness)

            if dist_w < dist_h:
                gap_length_px = h
            else:
                gap_length_px = w

            if gap_length_px < min_gap_px:
                op.is_valid = False
                op.status = "too_narrow"
            elif gap_length_px > max_gap_px:
                op.is_valid = False
                op.status = "too_long"

        return openings

    def _check_wall_overlap(self, opening, rectangles, threshold=0.5):
        """Проверка перекрытия проёма со стенами."""
        opening_box = (opening.x1, opening.y1, opening.x2, opening.y2)
        total_overlap_area = 0
        for rect in rectangles:
            rect_box = (rect.x1, rect.y1, rect.x2, rect.y2)
            total_overlap_area += self._rect_intersection_area(opening_box, rect_box)
        overlap_ratio = total_overlap_area / (opening.area + 1e-6)
        if overlap_ratio > threshold:
            return True
        return False

    # ============================================================
    # ОСНОВНОЙ МЕТОД ДЕТЕКЦИИ
    # ============================================================

    def detect_openings(self,
                        rectangles: List,
                        yolo_results=None,
                        pixel_scale: Optional[float] = None,
                        image: Optional[np.ndarray] = None,
                        door_window_boxes: Optional[List] = None,
                        visualize: bool = False,
                        vis_output_path: Optional[str] = None,
                        deleted_openings: Optional[List] = None) -> List[Opening]:
        """
        Детекция проёмов со строгой валидацией перед объединением.
        Невалидные проёмы фильтруются и не объединяются.

        Параметры:
            rectangles: Список прямоугольников стен
            yolo_results: Результаты YOLO детекции
            pixel_scale: Масштаб в метрах на пиксель
            image: Изображение для анализа цвета
            door_window_boxes: Список bbox дверей/окон
            visualize: Флаг визуализации
            vis_output_path: Путь для сохранения визуализации
            deleted_openings: Список для сохранения удалённых проёмов

        Возвращает:
            Список валидных объединённых проёмов
        """
        self.last_deleted_openings = []

        if not rectangles:
            return []

        rect_lookup = {r.id: r for r in rectangles}
        all_ray_results: List[RayCastResult] = []

        # Шаг 1: Трассировка лучей
        for rect in rectangles:
            all_ray_results.extend(self._cast_rays_from_rectangle(rect, rectangles, rect_lookup))

        if not all_ray_results:
            return []

        # Шаг 2: Создание исходных проёмов из лучей
        raw_openings: List[Opening] = []
        for idx, ray in enumerate(all_ray_results):
            raw_openings.append(Opening(
                id=idx,
                x1=ray.ray_x1,
                y1=ray.ray_y1,
                x2=ray.ray_x2,
                y2=ray.ray_y2,
                width=ray.ray_x2 - ray.ray_x1,
                height=ray.ray_y2 - ray.ray_y1,
                area=(ray.ray_x2 - ray.ray_x1) * (ray.ray_y2 - ray.ray_y1),
                opening_type='unknown',
                confidence=0.0,
                source_rects=[ray.source_rect_id, ray.target_rect_id],
                yolo_matches=[],
                is_valid=True,
                status="valid"
            ))

        # Шаг 3: Обработка пересекающихся стен
        geom_corrected_openings = self._handle_crossing_walls(raw_openings, rectangles)

        wall_thickness = self._estimate_wall_thickness(rectangles)

        # Шаг 4: Фильтрация по размеру
        if pixel_scale is not None and pixel_scale > 0:
            size_filtered_openings = self._filter_by_real_world_size_strict(
                geom_corrected_openings, pixel_scale, wall_thickness
            )
        else:
            size_filtered_openings = geom_corrected_openings

        # Сбор всех bbox дверей/окон
        all_door_window_boxes = []
        if door_window_boxes:
            all_door_window_boxes.extend(door_window_boxes)
        if yolo_results is not None:
            if hasattr(yolo_results, 'doors') and yolo_results.doors:
                all_door_window_boxes.extend(yolo_results.doors)
            if hasattr(yolo_results, 'windows') and yolo_results.windows:
                all_door_window_boxes.extend(yolo_results.windows)

        # Шаг 5: Применение строгих проверок валидации
        validated_candidates = []

        for opening in size_filtered_openings:
            # Пропуск уже невалидных
            if not opening.is_valid:
                self.last_deleted_openings.append(opening)
                continue

            # Проверка 1: Перекрытие с дверями/окнами
            if self._check_door_window_overlap(opening, all_door_window_boxes):
                opening.is_valid = False
                opening.status = "door_window_overlap"
                self.last_deleted_openings.append(opening)
                continue

            # Проверка 2: Цветовое содержимое проёма
            if image is not None:
                is_valid, trimmed_coords = self._validate_gap_color_content(opening, image)
                if not is_valid:
                    opening.is_valid = False
                    opening.status = "non_white_content"
                    self.last_deleted_openings.append(opening)
                    continue
                elif trimmed_coords is not None:
                    # Обновление координат после обрезки
                    opening.x1, opening.y1, opening.x2, opening.y2 = trimmed_coords
                    opening.width = opening.x2 - opening.x1
                    opening.height = opening.y2 - opening.y1
                    opening.area = opening.width * opening.height

            # Проверка 3: Толщина касания стены
            if pixel_scale is not None and pixel_scale > 0:
                if self._check_wall_touch_length(opening, rectangles, pixel_scale):
                    opening.is_valid = False
                    opening.status = "wall_touch_exceeded"
                    self.last_deleted_openings.append(opening)
                    continue

            # Проверка 4: Детекция тупиков
            if self._check_third_wall_dead_end(opening, rectangles):
                opening.is_valid = False
                opening.status = "dead_end"
                self.last_deleted_openings.append(opening)
                continue

            # Проверка 5: Перекрытие со стенами
            if self._check_wall_overlap(opening, rectangles, threshold=0.5):
                opening.is_valid = False
                opening.status = "wall_overlap"
                self.last_deleted_openings.append(opening)
                continue

            # Проверка 6: Минимальное соотношение сторон (5:1)
            if self._check_min_aspect_ratio(opening):
                opening.is_valid = False
                opening.status = "aspect_ratio_too_low"
                self.last_deleted_openings.append(opening)
                continue

            # Проём прошёл все проверки
            validated_candidates.append(opening)

        # Шаг 6: Объединение ТОЛЬКО валидных кандидатов
        final_merged_openings = self._merge_adjacent_openings(validated_candidates)

        # Сохранение визуализации при необходимости
        if visualize and vis_output_path and image is not None:
            vis_img = self.visualize_openings(
                img=image,
                openings=final_merged_openings,
                rectangles=rectangles,
                show_rectangles=True,
                show_labels=True
            )
            cv2.imwrite(vis_output_path, vis_img)

        if deleted_openings is not None:
            deleted_openings.clear()
            deleted_openings.extend(self.last_deleted_openings)

        # Возврат ТОЛЬКО валидных объединённых проёмов
        return final_merged_openings

    # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================

    def visualize_openings(self, img, openings, rectangles=None,
                           show_rectangles=True, show_labels=True):
        """Визуализация обнаруженных проёмов на изображении."""
        vis_img = img.copy()

        if show_rectangles and rectangles:
            for r in rectangles:
                cv2.rectangle(vis_img, (r.x1, r.y1), (r.x2, r.y2), (180, 180, 180), 1)

        valid_color = (0, 200, 0)

        # Карта цветов для различных статусов
        status_colors = {
            'non_white_content': (0, 128, 255),
            'door_window_overlap': (255, 0, 255),
            'wall_touch_exceeded': (0, 0, 255),
            'dead_end': (128, 0, 128),
            'wall_overlap': (50, 50, 200),
            'too_narrow': (255, 200, 0),
            'too_long': (255, 0, 128),
            'obstructed_full': (50, 50, 50),
            'fully_blocked': (80, 80, 80),
        }

        for op in openings:
            overlay = vis_img.copy()
            if op.is_valid:
                cv2.rectangle(overlay, (op.x1, op.y1), (op.x2, op.y2), valid_color, -1)
                cv2.addWeighted(vis_img, 0.5, overlay, 0.5, 0, vis_img)
                cv2.rectangle(vis_img, (op.x1, op.y1), (op.x2, op.y2), (0, 150, 0), 2)
                if show_labels:
                    cv2.putText(vis_img, f"OK", (op.x1 + 2, op.y1 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
            else:
                color = status_colors.get(op.status, (100, 100, 100))
                cv2.rectangle(overlay, (op.x1, op.y1), (op.x2, op.y2), color, -1)
                cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0, vis_img)
                cv2.rectangle(vis_img, (op.x1, op.y1), (op.x2, op.y2), color, 2)
                cv2.line(vis_img, (op.x1, op.y1), (op.x2, op.y2), color, 1)
                cv2.line(vis_img, (op.x1, op.y2), (op.x2, op.y1), color, 1)
                if show_labels:
                    label = op.status[:12]
                    cv2.putText(vis_img, label, (op.x1 + 2, op.y1 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        return vis_img

    def create_wall_outline_with_gaps(self, rectangles, openings, img_shape):
        """Создание контура стен с учётом проёмов."""
        h, w = img_shape[:2]
        wall_mask = np.zeros((h, w), dtype=np.uint8)

        for rect in rectangles:
            cv2.rectangle(wall_mask, (rect.x1, rect.y1), (rect.x2, rect.y2), 255, -1)

        for opening in openings:
            if opening.is_valid:
                cv2.rectangle(wall_mask, (opening.x1, opening.y1),
                              (opening.x2, opening.y2), 0, -1)

        return wall_mask

    def create_wall_outline_image(self, rectangles, openings, img_shape,
                                  line_thickness=2):
        """Создание изображения контура стен."""
        h, w = img_shape[:2]
        wall_mask = self.create_wall_outline_with_gaps(rectangles, openings, img_shape)
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        outline_img = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(outline_img, contours, -1, 255, line_thickness)
        return outline_img