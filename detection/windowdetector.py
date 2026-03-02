"""
Система анализа планировок помещений - Модуль алгоритмической детекции окон
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Алгоритм детекции окон на основе параллельных тонких линий.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectedWindow:
    """
    Структура данных для обнаруженного окна.

    Атрибуты:
        id: Уникальный идентификатор окна
        x, y: Координаты левого верхнего угла
        w, h: Ширина и высота окна
    """
    id: int
    x: int
    y: int
    w: int
    h: int


# Константы цветов
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class AlgorithmicWindowDetector:
    """
    Детектор окон на основе алгоритма параллельных линий.

    Использует маску контура стен и исключает области дверей
    для точной детекции оконных проёмов.
    """

    def __init__(self,
                 min_window_len: int = 15,
                 ocr_model: Optional[object] = None,
                 wall_overlap_threshold: float = 0.4,
                 wall_connectivity_proximity: int = 5,
                 min_wall_endpoint_contact: int = 1,
                 min_thickness_ratio: float = 0.35):
        """
        Инициализация детектора окон.

        Параметры:
            min_window_len: Минимальная длина линии для детекции окна (пиксели)
            ocr_model: Предварительно инициализированная модель docTR (опционально)
            wall_overlap_threshold: Максимальная доля окна, которая может перекрываться со стеной
            wall_connectivity_proximity: Максимальное расстояние для проверки связности со стеной
            min_wall_endpoint_contact: Минимальное количество концов, касающихся стен
            min_thickness_ratio: Minimum ratio of window narrow-side to estimated wall
                thickness.  Windows thinner than ``min_thickness_ratio * wall_thickness``
                are rejected as spurious thin-line artefacts (default 0.35).
        """
        self.min_window_len = min_window_len
        self.ocr_model = ocr_model
        self.wall_overlap_threshold = wall_overlap_threshold
        self.wall_connectivity_proximity = wall_connectivity_proximity
        self.min_wall_endpoint_contact = min_wall_endpoint_contact
        self.min_thickness_ratio = min_thickness_ratio

    def _threshold_by_rgb_variance(
            self,
            img: Image.Image,
            diff_threshold: int = 30,
            white_threshold: int = 225,
            gb_close_threshold: int = 20,
            red_dominance_threshold: int = 5,
    ) -> Image.Image:
        """
        Пороговая обработка изображения для выделения чёрных линий на белом фоне.

        Параметры:
            img: Входное изображение PIL
            diff_threshold: Порог разницы RGB для низкой вариации
            white_threshold: Порог для определения белого цвета
            gb_close_threshold: Порог близости зелёного и синего каналов
            red_dominance_threshold: Порог доминирования красного канала

        Возвращает:
            Бинарное изображение (чёрное/белое)
        """
        img = img.convert("RGB")
        out = img.copy()
        p = out.load()
        w, h = out.size

        for y in range(h):
            for x in range(w):
                r, g, b = p[x, y]

                # Проверка на белый цвет
                is_white = r >= white_threshold and g >= white_threshold and b >= white_threshold

                # Проверка вариации RGB
                rgb_diff = max(r, g, b) - min(r, g, b)
                low_variance = rgb_diff < diff_threshold

                # Проверка доминирования красного
                red_dominant = (
                        r > g + red_dominance_threshold and
                        r > b + red_dominance_threshold and
                        abs(g - b) < gb_close_threshold
                )

                if (low_variance or not red_dominant) and not is_white:
                    p[x, y] = BLACK
                else:
                    p[x, y] = WHITE

        return out

    def _detect_thin_lines(
            self,
            img: Image.Image,
            max_thickness: int = 3,
            min_thickness: int = 3,
            white_ratio_threshold: float = 0.75,
            side_check_px: int = 2,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Детекция тонких горизонтальных и вертикальных линий.

        Параметры:
            img: Бинарное изображение PIL
            max_thickness: Максимальная толщина линии
            min_thickness: Минимальная толщина линии; сегменты тоньше отбрасываются
            white_ratio_threshold: Минимальная доля белых пикселей по краям
            side_check_px: Расстояние проверки по сторонам линии

        Возвращает:
            Кортеж (горизонтальные_сегменты, вертикальные_сегменты)
        """
        p = img.load()
        w, h = img.size

        horiz_segments: List[dict] = []
        vert_segments: List[dict] = []

        def is_black(x: int, y: int) -> bool:
            """Проверка, является ли пиксель чёрным."""
            return 0 <= x < w and 0 <= y < h and p[x, y] == BLACK

        def is_white(x: int, y: int) -> bool:
            """Проверка, является ли пиксель белым."""
            return 0 <= x < w and 0 <= y < h and p[x, y] == WHITE

        # Детекция горизонтальных линий
        for y in range(h):
            x = 0
            while x < w:
                if is_black(x, y):
                    xs = x
                    t = 1
                    # Определение толщины линии
                    while t < max_thickness and y + t < h and is_black(x, y + t):
                        t += 1
                    # Продолжение по горизонтали
                    while x < w and is_black(x, y):
                        x += 1

                    length = x - xs
                    if length >= self.min_window_len:
                        # Проверка белого пространства по краям
                        wc = tc = 0
                        for offset in range(1, side_check_px + 1):
                            # Верхняя сторона
                            upper_y = y - offset
                            if upper_y >= 0:
                                for xx in range(xs, x):
                                    tc += 1
                                    wc += is_white(xx, upper_y)
                            # Нижняя сторона
                            lower_y = y + t - 1 + offset
                            if lower_y < h:
                                for xx in range(xs, x):
                                    tc += 1
                                    wc += is_white(xx, lower_y)
                        if tc > 0 and wc / tc >= white_ratio_threshold and t >= min_thickness:
                            horiz_segments.append({
                                "x1": xs,
                                "x2": x - 1,
                                "y1": y,
                                "y2": y + t - 1,
                            })
                else:
                    x += 1

        # Детекция вертикальных линий
        for x in range(w):
            y = 0
            while y < h:
                if is_black(x, y):
                    ys = y
                    t = 1
                    # Определение толщины линии
                    while t < max_thickness and x + t < w and is_black(x + t, y):
                        t += 1
                    # Продолжение по вертикали
                    while y < h and is_black(x, y):
                        y += 1

                    length = y - ys
                    if length >= self.min_window_len:
                        # Проверка белого пространства по краям
                        wc = tc = 0
                        for offset in range(1, side_check_px + 1):
                            # Левая сторона
                            left_x = x - offset
                            if left_x >= 0:
                                for yy in range(ys, y):
                                    tc += 1
                                    wc += is_white(left_x, yy)
                            # Правая сторона
                            right_x = x + t - 1 + offset
                            if right_x < w:
                                for yy in range(ys, y):
                                    tc += 1
                                    wc += is_white(right_x, yy)
                        if tc > 0 and wc / tc >= white_ratio_threshold and t >= min_thickness:
                            vert_segments.append({
                                "x1": x,
                                "x2": x + t - 1,
                                "y1": ys,
                                "y2": y - 1,
                            })
                else:
                    y += 1

        return horiz_segments, vert_segments

    def _merge_intervals(self, intervals: List[Tuple[int, int]], max_gap: int = 0) -> List[Tuple[int, int]]:
        """
        Объединение перекрывающихся или близких интервалов.

        Параметры:
            intervals: Список интервалов (начало, конец)
            max_gap: Максимальный зазор для объединения

        Возвращает:
            Объединённые интервалы
        """
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda i: i[0])
        merged = []
        curr_start, curr_end = sorted_intervals[0]
        for start, end in sorted_intervals[1:]:
            if start <= curr_end + max_gap + 1:
                curr_end = max(curr_end, end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = start, end
        merged.append((curr_start, curr_end))
        return merged

    def _group_segments(
            self,
            segments: List[dict],
            orientation: str,
            max_gap_perp: int = 15,
            max_gap_length: int = 10,
    ) -> List[dict]:
        """
        Группировка параллельных сегментов линий в ограничивающие рамки окон.

        Параметры:
            segments: Список сегментов линий
            orientation: Ориентация ('h' для горизонтальной, 'v' для вертикальной)
            max_gap_perp: Максимальный зазор перпендикулярно линиям
            max_gap_length: Максимальный зазор вдоль линий

        Возвращает:
            Список ограничивающих рамок
        """
        if not segments:
            return []

        is_horiz = orientation == "h"
        perp_min = "y1" if is_horiz else "x1"
        perp_max = "y2" if is_horiz else "x2"
        len_min = "x1" if is_horiz else "y1"
        len_max = "x2" if is_horiz else "y2"

        segments = sorted(segments, key=lambda s: s[perp_min])

        bboxes = []
        curr_segments: List[dict] = []
        curr_min_perp = segments[0][perp_min]
        curr_max_perp = segments[0][perp_max]

        for seg in segments:
            if seg[perp_min] > curr_max_perp + max_gap_perp + 1:
                # Завершение текущей группы (только если есть 2+ сегмента)
                if len(curr_segments) >= 2:
                    length_ints = [(s[len_min], s[len_max]) for s in curr_segments]
                    merged_length = self._merge_intervals(length_ints, max_gap=max_gap_length)
                    for l_start, l_end in merged_length:
                        bboxes.append({
                            "x1": l_start if is_horiz else curr_min_perp,
                            "x2": l_end if is_horiz else curr_max_perp,
                            "y1": curr_min_perp if is_horiz else l_start,
                            "y2": curr_max_perp if is_horiz else l_end,
                        })
                # Новая группа
                curr_segments = [seg]
                curr_min_perp = seg[perp_min]
                curr_max_perp = seg[perp_max]
            else:
                curr_segments.append(seg)
                curr_min_perp = min(curr_min_perp, seg[perp_min])
                curr_max_perp = max(curr_max_perp, seg[perp_max])

        # Последняя группа (только если есть 2+ сегмента)
        if len(curr_segments) >= 2:
            length_ints = [(s[len_min], s[len_max]) for s in curr_segments]
            merged_length = self._merge_intervals(length_ints, max_gap=max_gap_length)
            for l_start, l_end in merged_length:
                bboxes.append({
                    "x1": l_start if is_horiz else curr_min_perp,
                    "x2": l_end if is_horiz else curr_max_perp,
                    "y1": curr_min_perp if is_horiz else l_start,
                    "y2": curr_max_perp if is_horiz else l_end,
                })

        return bboxes

    def _erase_mask(
            self,
            img: Image.Image,
            mask: np.ndarray,
            dilate_px: int = 2,
    ) -> Image.Image:
        """
        Удаление областей из изображения на основе маски.

        Параметры:
            img: Входное изображение PIL
            mask: Бинарная маска областей для удаления
            dilate_px: Размер расширения маски

        Возвращает:
            Изображение с удалёнными областями
        """
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        out = img.copy()
        p = out.load()
        h, w = mask.shape

        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    p[x, y] = WHITE

        return out

    def _bbox_overlaps(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int],
                       threshold: float = 0.3) -> bool:
        """
        Проверка значительного перекрытия двух ограничивающих рамок.

        Параметры:
            bbox1, bbox2: Ограничивающие рамки (x1, y1, x2, y2)
            threshold: Порог перекрытия

        Возвращает:
            True, если рамки значительно перекрываются
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Расчёт пересечения
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap

        if intersection == 0:
            return False

        # Расчёт площадей
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # Проверка значительности пересечения
        iou_1 = intersection / area1 if area1 > 0 else 0
        iou_2 = intersection / area2 if area2 > 0 else 0

        return max(iou_1, iou_2) > threshold

    def _window_inside_wall_region(
            self,
            window_bbox: dict,
            wall_mask: np.ndarray,
            max_interior_coverage: float = 0.4,
            edge_margin: int = 2
    ) -> bool:
        """
        Проверка, находится ли окно внутри области стены (ложное срабатывание).

        Валидное окно должно иметь в основном пустое внутреннее пространство
        (белая область между параллельными линиями). Если внутренняя область
        сильно покрыта маской стен, это вероятно ложная детекция.

        Параметры:
            window_bbox: Ограничивающая рамка окна
            wall_mask: Бинарная маска контуров стен
            max_interior_coverage: Максимальное покрытие стенами внутренней области
            edge_margin: Отступ от краёв при проверке внутренней области

        Возвращает:
            True, если окно следует отклонить (находится внутри стены)
        """
        x1, y1, x2, y2 = window_bbox["x1"], window_bbox["y1"], window_bbox["x2"], window_bbox["y2"]
        h, w = wall_mask.shape

        win_w = x2 - x1 + 1
        win_h = y2 - y1 + 1

        # Пропуск проверки для очень тонких окон (нет реальной внутренней области)
        if win_w < (2 * edge_margin + 3) or win_h < (2 * edge_margin + 3):
            return False

        # Определение внутренней области (с отступом от краёв)
        int_x1 = max(0, min(w - 1, x1 + edge_margin))
        int_x2 = max(0, min(w, x2 - edge_margin + 1))
        int_y1 = max(0, min(h - 1, y1 + edge_margin))
        int_y2 = max(0, min(h, y2 - edge_margin + 1))

        if int_x2 <= int_x1 or int_y2 <= int_y1:
            return False

        # Получение внутренней области из маски стен
        interior = wall_mask[int_y1:int_y2, int_x1:int_x2]

        # Расчёт покрытия внутренней области стенами
        total_pixels = interior.size
        wall_pixels = np.count_nonzero(interior)

        coverage = wall_pixels / total_pixels if total_pixels > 0 else 0

        return coverage > max_interior_coverage

    def _check_wall_connectivity(
            self,
            window_bbox: dict,
            wall_mask: np.ndarray,
            proximity_px: int = 5,
            min_endpoint_contact: int = 1,
            contact_search_width: int = 3
    ) -> bool:
        """
        Проверка правильной связности окна со стенами на концах.

        Валидные окна должны быть соединены со стенами на концах
        (где окно встречается с перпендикулярными сегментами стен).

        Параметры:
            window_bbox: Ограничивающая рамка окна
            wall_mask: Бинарная маска контуров стен
            proximity_px: Максимальное расстояние поиска контакта со стеной
            min_endpoint_contact: Минимальное количество концов, касающихся стен
            contact_search_width: Ширина области поиска перпендикулярно окну

        Возвращает:
            True, если окно имеет достаточную связность со стенами
        """
        x1, y1, x2, y2 = window_bbox["x1"], window_bbox["y1"], window_bbox["x2"], window_bbox["y2"]
        h, w = wall_mask.shape

        win_w = x2 - x1 + 1
        win_h = y2 - y1 + 1
        is_horizontal = win_w >= win_h

        endpoint_contacts = 0

        if is_horizontal:
            # Горизонтальное окно: проверка левого и правого концов
            y_center = (y1 + y2) // 2
            y_search_min = max(0, y_center - contact_search_width)
            y_search_max = min(h, y_center + contact_search_width + 1)

            # Проверка левого конца
            left_found = False
            for dx in range(proximity_px + 1):
                check_x = x1 - dx
                if 0 <= check_x < w:
                    for check_y in range(y_search_min, y_search_max):
                        if wall_mask[check_y, check_x] > 0:
                            left_found = True
                            break
                if left_found:
                    break
            if left_found:
                endpoint_contacts += 1

            # Проверка правого конца
            right_found = False
            for dx in range(proximity_px + 1):
                check_x = x2 + dx
                if 0 <= check_x < w:
                    for check_y in range(y_search_min, y_search_max):
                        if wall_mask[check_y, check_x] > 0:
                            right_found = True
                            break
                if right_found:
                    break
            if right_found:
                endpoint_contacts += 1

        else:
            # Вертикальное окно: проверка верхнего и нижнего концов
            x_center = (x1 + x2) // 2
            x_search_min = max(0, x_center - contact_search_width)
            x_search_max = min(w, x_center + contact_search_width + 1)

            # Проверка верхнего конца
            top_found = False
            for dy in range(proximity_px + 1):
                check_y = y1 - dy
                if 0 <= check_y < h:
                    for check_x in range(x_search_min, x_search_max):
                        if wall_mask[check_y, check_x] > 0:
                            top_found = True
                            break
                if top_found:
                    break
            if top_found:
                endpoint_contacts += 1

            # Проверка нижнего конца
            bottom_found = False
            for dy in range(proximity_px + 1):
                check_y = y2 + dy
                if 0 <= check_y < h:
                    for check_x in range(x_search_min, x_search_max):
                        if wall_mask[check_y, check_x] > 0:
                            bottom_found = True
                            break
                if bottom_found:
                    break
            if bottom_found:
                endpoint_contacts += 1

        return endpoint_contacts >= min_endpoint_contact

    def _check_wall_edge_contact(
            self,
            window_bbox: dict,
            wall_mask: np.ndarray,
            proximity_px: int = 3,
            min_contact_ratio: float = 0.2
    ) -> bool:
        """
        Проверка контакта краёв окна с линиями стен.

        Окна должны находиться на стенах, то есть маска стен должна
        иметь пиксели рядом/вдоль длинных краёв окна (параллельных линий).

        Параметры:
            window_bbox: Ограничивающая рамка окна
            wall_mask: Бинарная маска контуров стен
            proximity_px: Расстояние проверки от краёв окна
            min_contact_ratio: Минимальная доля края, контактирующая со стеной

        Возвращает:
            True, если окно имеет достаточный контакт краёв со стенами
        """
        x1, y1, x2, y2 = window_bbox["x1"], window_bbox["y1"], window_bbox["x2"], window_bbox["y2"]
        h, w = wall_mask.shape

        win_w = x2 - x1 + 1
        win_h = y2 - y1 + 1
        is_horizontal = win_w >= win_h

        contact_points = 0
        total_points = 0

        if is_horizontal:
            # Проверка вдоль горизонтальной длины окна
            for x in range(max(0, x1), min(w, x2 + 1)):
                # Проверка около верхнего края
                found_top = False
                for dy in range(proximity_px + 1):
                    check_y = y1 - dy
                    if 0 <= check_y < h and wall_mask[check_y, x] > 0:
                        found_top = True
                        break

                # Проверка около нижнего края
                found_bottom = False
                for dy in range(proximity_px + 1):
                    check_y = y2 + dy
                    if 0 <= check_y < h and wall_mask[check_y, x] > 0:
                        found_bottom = True
                        break

                if found_top or found_bottom:
                    contact_points += 1
                total_points += 1
        else:
            # Проверка вдоль вертикальной длины окна
            for y in range(max(0, y1), min(h, y2 + 1)):
                # Проверка около левого края
                found_left = False
                for dx in range(proximity_px + 1):
                    check_x = x1 - dx
                    if 0 <= check_x < w and wall_mask[y, check_x] > 0:
                        found_left = True
                        break

                # Проверка около правого края
                found_right = False
                for dx in range(proximity_px + 1):
                    check_x = x2 + dx
                    if 0 <= check_x < w and wall_mask[y, check_x] > 0:
                        found_right = True
                        break

                if found_left or found_right:
                    contact_points += 1
                total_points += 1

        contact_ratio = contact_points / total_points if total_points > 0 else 0
        return contact_ratio >= min_contact_ratio

    def _classify_narrow_side_connection(
        self,
        win: 'DetectedWindow',
        side: str,
        wall_mask: np.ndarray,
        other_windows: List['DetectedWindow'],
        proximity_px: int = 8,
    ) -> str:
        """
        For one narrow side of ``win`` return what it is connected to:

          'wall'          – wall pixels present in the outward search zone
          'window_narrow' – the closest feature of an overlapping window is
                            that window's own narrow end (end-to-end or
                            perpendicular T-junction)
          'window_long'   – the only window contact is the long body of
                            another window (bad: A hangs off B's side)
          'none'          – nothing found within proximity_px

        ``side`` is one of 'left'/'right' for horizontal windows or
        'top'/'bottom' for vertical windows.
        """
        mh, mw = wall_mask.shape
        is_horiz = win.w >= win.h

        # ── 1. Build the outward search rectangle ──────────────────────────
        if is_horiz:
            # narrow sides are left/right vertical edges
            edge_x = win.x if side == 'left' else win.x + win.w
            sx1 = max(0, edge_x - proximity_px)
            sx2 = min(mw, edge_x + proximity_px + 1)
            sy1 = max(0, win.y)
            sy2 = min(mh, win.y + win.h)
        else:
            # narrow sides are top/bottom horizontal edges
            edge_y = win.y if side == 'top' else win.y + win.h
            sy1 = max(0, edge_y - proximity_px)
            sy2 = min(mh, edge_y + proximity_px + 1)
            sx1 = max(0, win.x)
            sx2 = min(mw, win.x + win.w)

        if sx2 <= sx1 or sy2 <= sy1:
            return 'none'

        # ── 2. Wall check ───────────────────────────────────────────────────
        if wall_mask[sy1:sy2, sx1:sx2].any():
            return 'wall'

        # ── 3. Window check ─────────────────────────────────────────────────
        for other in other_windows:
            ox1 = other.x
            oy1 = other.y
            ox2 = other.x + other.w
            oy2 = other.y + other.h

            # Does the other window bbox intersect the search zone?
            if ox2 < sx1 or ox1 > sx2 or oy2 < sy1 or oy1 > sy2:
                continue

            other_horiz = other.w >= other.h

            if is_horiz and other_horiz:
                # Both horizontal.
                # Narrow ends of other are at ox1 and ox2.
                # Valid if our edge_x is close to one of those ends.
                if abs(edge_x - ox1) <= proximity_px or abs(edge_x - ox2) <= proximity_px:
                    return 'window_narrow'
                else:
                    return 'window_long'   # edge_x is in the middle of other's span

            elif not is_horiz and not other_horiz:
                # Both vertical.
                # Narrow ends of other are at oy1 and oy2.
                if abs(edge_y - oy1) <= proximity_px or abs(edge_y - oy2) <= proximity_px:
                    return 'window_narrow'
                else:
                    return 'window_long'

            elif is_horiz and not other_horiz:
                # A is horizontal, B is vertical (perpendicular).
                # A's narrow side is a vertical segment at edge_x.
                # B's narrow sides are at oy1 and oy2 (horizontal edges).
                # B's long sides are at ox1 and ox2 (vertical edges).
                # Valid if A's edge_x is close to B's long side (it sits flush
                # against B's wall face) AND the y extents overlap —
                # BUT invalid if A's edge_x is well inside B's horizontal span
                # without aligning to B's end.
                on_b_long_side = (
                    abs(edge_x - ox1) <= proximity_px or
                    abs(edge_x - ox2) <= proximity_px
                )
                # Also valid if A's end aligns with B's narrow end y-position
                a_y_mid = win.y + win.h / 2
                on_b_narrow_end = (
                    abs(a_y_mid - oy1) <= proximity_px * 2 or
                    abs(a_y_mid - oy2) <= proximity_px * 2
                )
                if on_b_long_side or on_b_narrow_end:
                    return 'window_narrow'
                return 'window_long'

            else:
                # A is vertical, B is horizontal (perpendicular).
                on_b_long_side = (
                    abs(edge_y - oy1) <= proximity_px or
                    abs(edge_y - oy2) <= proximity_px
                )
                a_x_mid = win.x + win.w / 2
                on_b_narrow_end = (
                    abs(a_x_mid - ox1) <= proximity_px * 2 or
                    abs(a_x_mid - ox2) <= proximity_px * 2
                )
                if on_b_long_side or on_b_narrow_end:
                    return 'window_narrow'
                return 'window_long'

        return 'none'

    def _validate_narrow_connectivity(
        self,
        windows: List['DetectedWindow'],
        wall_mask: np.ndarray,
        proximity_px: int = 8,
    ) -> List['DetectedWindow']:
        """
        Keep only windows whose BOTH narrow sides have a valid connection:
        either a wall pixel or another window's narrow end.
        A window is dropped if either narrow side is unconnected ('none') or
        only touches the long body of another window ('window_long').
        """
        valid = []
        for win in windows:
            others = [w for w in windows if w is not win]
            is_horiz = win.w >= win.h
            sides = ('left', 'right') if is_horiz else ('top', 'bottom')
            ok = True
            for side in sides:
                result = self._classify_narrow_side_connection(
                    win, side, wall_mask, others, proximity_px
                )
                if result in ('none', 'window_long'):
                    ok = False
                    break
            if ok:
                valid.append(win)
        return valid

    def _estimate_wall_thickness(
            self,
            wall_mask: np.ndarray,
            sample_count: int = 200,
    ) -> float:
        """
        Estimate the typical wall thickness in pixels by sampling the wall mask.

        For each sampled wall pixel, measure how many consecutive wall pixels
        exist in both the horizontal and vertical directions. The median of
        these local thickness measurements is returned as the representative
        wall thickness.

        Parameters:
            wall_mask: Binary wall outline mask (255 = wall).
            sample_count: Number of wall pixels to sample.

        Returns:
            Estimated wall thickness in pixels, or 0.0 if mask is empty.
        """
        ys, xs = np.where(wall_mask > 0)
        if len(ys) == 0:
            return 0.0

        h, w = wall_mask.shape
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ys), size=min(sample_count, len(ys)), replace=False)

        thicknesses = []
        for idx in indices:
            cy, cx = int(ys[idx]), int(xs[idx])

            # Measure horizontal run length through this pixel
            lx = cx
            while lx > 0 and wall_mask[cy, lx - 1] > 0:
                lx -= 1
            rx = cx
            while rx < w - 1 and wall_mask[cy, rx + 1] > 0:
                rx += 1
            horiz = rx - lx + 1

            # Measure vertical run length through this pixel
            ty = cy
            while ty > 0 and wall_mask[ty - 1, cx] > 0:
                ty -= 1
            by = cy
            while by < h - 1 and wall_mask[by + 1, cx] > 0:
                by += 1
            vert = by - ty + 1

            # The minimum of the two directions is a better proxy for
            # local thickness (the other direction measures wall length)
            thicknesses.append(min(horiz, vert))

        return float(np.median(thicknesses)) if thicknesses else 0.0

    def _filter_by_thickness(
            self,
            bboxes: List[dict],
            wall_mask: np.ndarray,
            min_thickness_ratio: float = 0.35,
    ) -> List[dict]:
        """
        Discard candidate windows whose narrow dimension is significantly
        thinner than the typical wall thickness.

        A real window occupies the full depth of a wall, so its narrow side
        should be comparable to wall thickness.  Spurious thin-line artefacts
        are typically far narrower.

        Parameters:
            bboxes: List of bbox dicts with x1/y1/x2/y2 keys.
            wall_mask: Binary wall outline mask used to estimate wall thickness.
            min_thickness_ratio: A window is rejected if its narrow side is
                less than ``min_thickness_ratio * wall_thickness``.

        Returns:
            Filtered list of bboxes.
        """
        wall_thickness = self._estimate_wall_thickness(wall_mask)
        if wall_thickness <= 0:
            # Cannot estimate — skip filter entirely
            return bboxes

        min_narrow = wall_thickness * min_thickness_ratio
        kept = []
        for bbox in bboxes:
            win_w = bbox["x2"] - bbox["x1"] + 1
            win_h = bbox["y2"] - bbox["y1"] + 1
            narrow = min(win_w, win_h)
            if narrow >= min_narrow:
                kept.append(bbox)
            else:
                logger.debug(
                    "Thickness filter: dropped bbox %s "
                    "(narrow=%d < %.1f = %.2f * wall_thickness=%.1f)",
                    bbox, narrow, min_narrow, min_thickness_ratio, wall_thickness,
                )
        return kept

    def detect(self,
               img: np.ndarray,
               wall_outline_mask: np.ndarray,
               exclude_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
               ocr_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
               known_windows: Optional[List[Tuple[float, float, float, float]]] = None,
               ) -> List[DetectedWindow]:
        """
        Детекция окон на изображении.

        Параметры:
            img: Входное изображение в формате BGR
            wall_outline_mask: Бинарная маска контуров стен
            exclude_bboxes: Список исключаемых областей (например, двери) как (x1, y1, x2, y2)
            ocr_bboxes: Список областей текста OCR для исключения как (x1, y1, x2, y2)
            known_windows: Confirmed window boxes from another source (e.g. YOLO) as
                (x1, y1, x2, y2).  Any algo window whose IoU with one of these
                exceeds 0.30 is dropped — the caller's detection is authoritative.

        Возвращает:
            Список обнаруженных окон
        """
        # Преобразование в PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Очистка изображения путём удаления стен
        img_clean = self._erase_mask(img_pil, wall_outline_mask, dilate_px=4)

        # Удаление областей дверей, если предоставлены
        if exclude_bboxes:
            # Создание маски исключения из ограничивающих рамок
            h, w = wall_outline_mask.shape
            exclusion_mask = np.zeros((h, w), dtype=np.uint8)

            for bbox in exclude_bboxes:
                x1, y1, x2, y2 = bbox
                x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
                y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))

                # Ограничение границами изображения
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h, y2))

                if x2 > x1 and y2 > y1:
                    exclusion_mask[y1:y2, x1:x2] = 255

            # Удаление исключённых областей (с большим расширением для дверей)
            img_clean = self._erase_mask(img_clean, exclusion_mask, dilate_px=4)

        # Удаление областей текста OCR, если предоставлены
        if ocr_bboxes:
            h, w = wall_outline_mask.shape
            ocr_mask = np.zeros((h, w), dtype=np.uint8)

            for x1, y1, x2, y2 in ocr_bboxes:
                # Уже в пиксельных координатах от walldetector
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h, y2))

                if x2 > x1 and y2 > y1:
                    ocr_mask[y1:y2, x1:x2] = 255

            # Удаление OCR областей (умеренное расширение)
            img_clean = self._erase_mask(img_clean, ocr_mask, dilate_px=3)

        # Пороговая обработка до чёрно-белого
        bw = self._threshold_by_rgb_variance(img_clean)

        # Детекция тонких линейных сегментов
        horiz_segments, vert_segments = self._detect_thin_lines(
            bw,
            max_thickness=3,
            min_thickness=3,
            white_ratio_threshold=0.75,
            side_check_px=2,
        )

        # Группировка параллельных сегментов в ограничивающие рамки окон
        horiz_bboxes = self._group_segments(horiz_segments, "h", max_gap_perp=15, max_gap_length=10)
        vert_bboxes = self._group_segments(vert_segments, "v", max_gap_perp=15, max_gap_length=10)
        all_bboxes = horiz_bboxes + vert_bboxes

        # ── Thickness filter ──────────────────────────────────────────────────
        # Reject windows whose narrow side is significantly thinner than the
        # walls on this image.  This eliminates thin-line artefacts that pass
        # the parallel-lines heuristic but are nowhere near wall depth.
        all_bboxes = self._filter_by_thickness(all_bboxes, wall_outline_mask,
                                               min_thickness_ratio=self.min_thickness_ratio)

        # Фильтрация окон, перекрывающихся с исключёнными областями
        filtered_bboxes = []
        all_exclusions = []
        if exclude_bboxes:
            all_exclusions.extend(exclude_bboxes)
        if ocr_bboxes:
            all_exclusions.extend(ocr_bboxes)

        if all_exclusions:
            for bbox in all_bboxes:
                window_bbox = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
                overlaps_exclusion = False

                for excl_bbox in all_exclusions:
                    if self._bbox_overlaps(window_bbox, excl_bbox, threshold=0.3):
                        overlaps_exclusion = True
                        break

                if not overlaps_exclusion:
                    filtered_bboxes.append(bbox)
        else:
            filtered_bboxes = all_bboxes

        # Фильтрация окон, находящихся внутри областей стен
        wall_filtered_bboxes = []
        for bbox in filtered_bboxes:
            if self._window_inside_wall_region(bbox, wall_outline_mask,
                                                max_interior_coverage=self.wall_overlap_threshold):
                # Окно внутри области стены - отклонить
                continue
            wall_filtered_bboxes.append(bbox)

        filtered_bboxes = wall_filtered_bboxes

        # Строгая проверка связности со стенами
        connectivity_filtered_bboxes = []
        for bbox in filtered_bboxes:
            # Проверка связности концов (концы окна должны касаться стен)
            has_endpoint_contact = self._check_wall_connectivity(
                bbox,
                wall_outline_mask,
                proximity_px=self.wall_connectivity_proximity,
                min_endpoint_contact=self.min_wall_endpoint_contact,
                contact_search_width=5
            )

            # Проверка контакта краёв (окно должно быть вдоль линии стены)
            has_edge_contact = self._check_wall_edge_contact(
                bbox,
                wall_outline_mask,
                proximity_px=4,
                min_contact_ratio=0.15
            )

            # Окно должно пройти хотя бы одну проверку связности
            if has_endpoint_contact or has_edge_contact:
                connectivity_filtered_bboxes.append(bbox)

        filtered_bboxes = connectivity_filtered_bboxes

        # Преобразование в объекты DetectedWindow
        windows = []
        for i, bbox in enumerate(filtered_bboxes):
            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            windows.append(DetectedWindow(
                id=i,
                x=x1,
                y=y1,
                w=w,
                h=h
            ))

        # Narrow-side connectivity check: both ends must touch a wall or
        # another window's narrow end — not just its long body.
        windows = self._validate_narrow_connectivity(windows, wall_outline_mask)

        # Suppress algo windows that overlap a confirmed external detection
        if known_windows and windows:
            iou_thresh = 0.30
            windows = [
                w for w in windows
                if all(
                    _iou_xyxy(
                        (float(w.x), float(w.y),
                         float(w.x + w.w), float(w.y + w.h)),
                        tuple(float(v) for v in kw),
                    ) < iou_thresh
                    for kw in known_windows
                )
            ]
            # Re-assign sequential ids after filtering
            for i, w in enumerate(windows):
                w.id = i

        return windows

    def get_mask(self, img_shape: Tuple[int, int, int], windows: List[DetectedWindow]) -> np.ndarray:
        """
        Создание бинарной маски из обнаруженных окон.

        Параметры:
            img_shape: Форма оригинального изображения (h, w, c)
            windows: Список обнаруженных окон

        Возвращает:
            Бинарную маску с заполненными окнами
        """
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for win in windows:
            x1, y1 = win.x, win.y
            x2, y2 = win.x + win.w, win.y + win.h

            # Ограничение границами изображения
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        return mask