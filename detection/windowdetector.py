"""
Система анализа планировок помещений - Модуль алгоритмической детекции окон
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Алгоритм детекции окон на основе параллельных тонких линий.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image


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
                 min_wall_endpoint_contact: int = 1):
        """
        Инициализация детектора окон.

        Параметры:
            min_window_len: Минимальная длина линии для детекции окна (пиксели)
            ocr_model: Предварительно инициализированная модель docTR (опционально)
            wall_overlap_threshold: Максимальная доля окна, которая может перекрываться со стеной
            wall_connectivity_proximity: Максимальное расстояние для проверки связности со стеной
            min_wall_endpoint_contact: Минимальное количество концов, касающихся стен
        """
        self.min_window_len = min_window_len
        self.ocr_model = ocr_model
        self.wall_overlap_threshold = wall_overlap_threshold
        self.wall_connectivity_proximity = wall_connectivity_proximity
        self.min_wall_endpoint_contact = min_wall_endpoint_contact

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
            white_ratio_threshold: float = 0.75,
            side_check_px: int = 2,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Детекция тонких горизонтальных и вертикальных линий.

        Параметры:
            img: Бинарное изображение PIL
            max_thickness: Максимальная толщина линии
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
                        if tc > 0 and wc / tc >= white_ratio_threshold:
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
                        if tc > 0 and wc / tc >= white_ratio_threshold:
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

    def detect(self,
               img: np.ndarray,
               wall_outline_mask: np.ndarray,
               exclude_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
               ocr_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[DetectedWindow]:
        """
        Детекция окон на изображении.

        Параметры:
            img: Входное изображение в формате BGR
            wall_outline_mask: Бинарная маска контуров стен
            exclude_bboxes: Список исключаемых областей (например, двери) как (x1, y1, x2, y2)
            ocr_bboxes: Список областей текста OCR для исключения как (x1, y1, x2, y2)

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
            white_ratio_threshold=0.75,
            side_check_px=2,
        )

        # Группировка параллельных сегментов в ограничивающие рамки окон
        horiz_bboxes = self._group_segments(horiz_segments, "h", max_gap_perp=15, max_gap_length=10)
        vert_bboxes = self._group_segments(vert_segments, "v", max_gap_perp=15, max_gap_length=10)
        all_bboxes = horiz_bboxes + vert_bboxes

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