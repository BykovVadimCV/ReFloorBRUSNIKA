"""
Система анализа планировок помещений - Модуль анализа размеров помещений
Версия: 4.0
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Детекция контура квартиры через raytracing по оригинальному изображению, определение
внешних стен (outer crust) и формы квартиры, OCR (docTR) с объединением боксов и стратегией "largest area",
расчёт пиксельного масштаба (включая стены), измерение длин стен в метрах и классификация внешних стен
"""

import cv2
import numpy as np
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import ndimage


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class WallSegment:
    """
    Структура данных для сегмента стены.

    Атрибуты:
        x1, y1, x2, y2: Координаты начала и конца сегмента
        thickness_mean: Средняя толщина стены
        thickness_std: Стандартное отклонение толщины
    """
    x1: int
    y1: int
    x2: int
    y2: int
    thickness_mean: float = 0.0
    thickness_std: float = 0.0


@dataclass
class RoomMeasurements:
    """
    Структура данных для результатов измерений.

    Атрибуты:
        area_m2: Общая площадь в квадратных метрах
        pixel_scale: Масштаб в метрах на пиксель
        apartment_area_pixels: Количество пикселей всей квартиры (включая стены)
        living_space_pixels: Количество пикселей жилой площади (без стен)
        walls: Список измеренных стен
        outer_crust: Маска внешней корки стен
        apartment_boundary: Маска границы квартиры
        room_areas: Список обнаруженных площадей комнат
        interior_mask: Маска внутреннего пространства
        exterior_mask: Маска внешнего пространства
    """
    area_m2: float
    pixel_scale: float
    apartment_area_pixels: int
    living_space_pixels: int
    walls: List[Dict]
    outer_crust: np.ndarray
    apartment_boundary: np.ndarray
    room_areas: List[Dict] = field(default_factory=list)
    interior_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    exterior_mask: np.ndarray = field(default_factory=lambda: np.array([]))


# ============================================================
# АНАЛИЗАТОР РАЗМЕРОВ ПОМЕЩЕНИЙ
# ============================================================

class RoomSizeAnalyzer:
    """
    Анализатор размеров помещений с OCR и измерениями стен.

    Использует трассировку лучей для определения границ квартиры,
    OCR для распознавания площади и вычисляет масштаб для измерения стен.
    """

    def __init__(self,
                 edge_distance_ratio: float = 0.15,
                 white_threshold: int = 240,
                 closing_kernel_ratio: float = 0.025,
                 debug_ocr: bool = False):
        """
        Инициализация анализатора.

        Параметры:
            edge_distance_ratio: Доля от края для определения внешних стен
            white_threshold: Порог яркости для определения белого фона
            closing_kernel_ratio: Относительный размер ядра морфологии
            debug_ocr: Флаг детализированного вывода OCR
        """
        self.edge_distance_ratio = edge_distance_ratio
        self.white_threshold = white_threshold
        self.closing_kernel_ratio = closing_kernel_ratio
        self.debug_ocr = debug_ocr
        self._init_ocr()

    def _init_ocr(self) -> None:
        """Инициализация модели OCR (docTR)."""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(pretrained=True)
        except ImportError:
            self.ocr_model = None

    # ============================================================
    # ДЕТЕКЦИЯ КОНТУРА ЧЕРЕЗ RAYTRACING
    # ============================================================

    def _connect_outline_gaps(self,
                              mask: np.ndarray,
                              max_component_iters: int = 10,
                              max_endpoint_iters: int = 5) -> np.ndarray:
        """
        Соединение разрывов в контуре квартиры.

        Алгоритм:
        1. Соединяет отдельные компоненты (кратчайшая связь между контурами)
        2. Соединяет концевые точки на скелете

        Параметры:
            mask: Бинарная маска контура
            max_component_iters: Максимум итераций для соединения компонент
            max_endpoint_iters: Максимум итераций для соединения концов

        Возвращает:
            Маску с замкнутым контуром
        """
        out = mask.copy()
        h, w = out.shape

        # Этап 1: Соединение отдельных компонент
        for _ in range(max_component_iters):
            contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) <= 1:
                break

            approx_list = []
            for c in contours:
                if len(c) >= 5:
                    eps = 0.01 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, eps, True)
                else:
                    approx = c
                approx_list.append(approx)

            best_p = None
            best_q = None
            min_d2: Optional[float] = None

            # Поиск ближайших точек между контурами
            for i in range(len(approx_list)):
                for j in range(i + 1, len(approx_list)):
                    ci = approx_list[i]
                    cj = approx_list[j]
                    for pi in ci:
                        x1, y1 = pi[0]
                        for pj in cj:
                            x2, y2 = pj[0]
                            dx = x1 - x2
                            dy = y1 - y2
                            d2 = dx * dx + dy * dy
                            if min_d2 is None or d2 < min_d2:
                                min_d2 = d2
                                best_p = (int(x1), int(y1))
                                best_q = (int(x2), int(y2))

            if best_p is None or best_q is None:
                break

            cv2.line(out, best_p, best_q, 255, 2)

        # Этап 2: Соединение открытых концов на скелете
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize(out > 0).astype(np.uint8) * 255
        except ImportError:
            skel = out.copy()

        max_gap_len = max(h, w) * 0.25  # Допустимая длина замыкания

        for _ in range(max_endpoint_iters):
            ys, xs = np.where(skel > 0)
            endpoints = []

            # Поиск концевых точек
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - 1), min(h, y + 2)
                x0, x1 = max(0, x - 1), min(w, x + 2)
                nb = np.count_nonzero(skel[y0:y1, x0:x1]) - 1
                if nb <= 1:
                    endpoints.append((x, y))

            if len(endpoints) < 2:
                break

            used = [False] * len(endpoints)

            # Соединение ближайших концов
            for i in range(len(endpoints)):
                if used[i]:
                    continue
                x1, y1 = endpoints[i]
                best_j = None
                best_d2: Optional[float] = None

                for j in range(len(endpoints)):
                    if i == j or used[j]:
                        continue
                    x2, y2 = endpoints[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_j = j

                if best_j is None:
                    continue

                d = float(np.sqrt(best_d2))
                if d <= max_gap_len:
                    x2, y2 = endpoints[best_j]
                    cv2.line(out, (x1, y1), (x2, y2), 255, 2)
                    used[i] = True
                    used[best_j] = True

            try:
                skel = skeletonize(out > 0).astype(np.uint8) * 255
            except ImportError:
                break

        return out

    def detect_apartment_outline(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция контура квартиры через трассировку лучей по оригинальному изображению.

        Алгоритм:
        1. Трассировка лучей с краёв к центру, поиск первых тёмных пикселей
        2. Морфологическая обработка и замыкание разрывов
        3. Выбор самого большого контура как квартиры
        4. Построение маски квартиры и внешней корки (outer crust)

        Параметры:
            image: Исходное изображение в формате BGR

        Возвращает:
            outer_crust: Маска внешних стен (тонкий слой)
            apartment_boundary: Заполненная маска квартиры
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        outline = np.zeros((h, w), dtype=np.uint8)
        thr = self.white_threshold

        # Вертикальные лучи сверху и снизу
        for x in range(w):
            # Сверху вниз
            for y in range(h):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break
            # Снизу вверх
            for y in range(h - 1, -1, -1):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break

        # Горизонтальные лучи слева и справа
        for y in range(h):
            # Слева направо
            for x in range(w):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break
            # Справа налево
            for x in range(w - 1, -1, -1):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break

        hit_count = int(np.sum(outline > 0))
        if hit_count == 0:
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty

        # Морфологическая обработка
        k = max(3, int(min(h, w) * 0.005))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        outline = cv2.dilate(outline, np.ones((3, 3), np.uint8), 1)
        outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Замыкание разрывов
        outline = self._connect_outline_gaps(outline)

        # Поиск самого большого контура
        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = h * w * 0.005

        if area < min_area:
            pass  # Контур слишком мал, но продолжаем

        # Создание маски квартиры
        apartment = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(apartment, [largest], -1, 255, -1)

        # Выделение внешней корки
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(apartment, erode_k, iterations=1)
        outer_crust = cv2.subtract(apartment, eroded)

        return outer_crust, apartment

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ МАСОК
    # ============================================================

    def detect_exterior_region(self,
                               wall_mask: np.ndarray,
                               closing_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Определение внешней и внутренней области по маске стен.

        Параметры:
            wall_mask: Бинарная маска стен
            closing_iterations: Число итераций морфологического замыкания

        Возвращает:
            exterior_mask: Маска внешнего пространства
            interior_mask: Маска внутреннего пространства
        """
        h, w = wall_mask.shape
        wall_binary = (wall_mask > 0).astype(np.uint8)

        # Морфологическое замыкание для получения сплошной формы
        base_size = max(15, int(min(h, w) * self.closing_kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size, base_size))

        apartment_solid = wall_binary * 255
        for _ in range(closing_iterations):
            apartment_solid = cv2.morphologyEx(apartment_solid, cv2.MORPH_CLOSE, kernel)

        # Поиск самого большого контура
        contours, _ = cv2.findContours(
            apartment_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        apartment_filled = np.zeros((h, w), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(apartment_filled, [largest], -1, 255, -1)
        else:
            apartment_filled = apartment_solid

        work = apartment_filled.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Точки по периметру для flood fill
        border_points = set()
        step = max(1, min(w, h) // 50)
        for x in range(0, w, step):
            border_points.add((x, 0))
            border_points.add((x, h - 1))
        for y in range(0, h, step):
            border_points.add((0, y))
            border_points.add((w - 1, y))
        border_points.update([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

        # Заливка внешней области
        fill_value = 128
        for (x, y) in border_points:
            if 0 <= x < w and 0 <= y < h:
                if work[y, x] == 0:
                    cv2.floodFill(work, flood_mask, (x, y), fill_value)

        exterior_mask = (work == fill_value).astype(np.uint8) * 255
        interior_mask = ((apartment_filled > 0) & (wall_binary == 0)).astype(np.uint8) * 255
        unfilled_interior = ((work != fill_value) & (apartment_filled == 0)).astype(np.uint8) * 255
        interior_mask = cv2.bitwise_or(interior_mask, unfilled_interior)

        return exterior_mask, interior_mask

    def detect_outer_crust(self,
                           wall_mask: np.ndarray,
                           image: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Вычисление внешней корки по маске стен (альтернативный метод).

        Параметры:
            wall_mask: Маска стен
            image: Исходное изображение (опционально)

        Возвращает:
            final_crust: Маска внешней корки
            exterior_mask: Внешняя область
            interior_mask: Внутренняя область
        """
        h, w = wall_mask.shape
        wall_binary = (wall_mask > 0).astype(np.uint8)

        if not np.any(wall_binary):
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty, empty

        wall_count = np.sum(wall_binary)

        # Определение внешней и внутренней областей
        exterior_mask, interior_mask = self.detect_exterior_region(wall_mask)

        # Расширение внешней области для поиска смежных стен
        exterior_adjacent = cv2.dilate(
            exterior_mask,
            np.ones((3, 3), np.uint8),
            iterations=1
        )
        exterior_adjacent_8 = cv2.dilate(
            exterior_mask,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            iterations=1
        )
        exterior_adjacent = cv2.bitwise_or(exterior_adjacent, exterior_adjacent_8)

        # Внешняя корка - стены, смежные с внешней областью
        outer_crust = (wall_binary > 0) & (exterior_adjacent > 0)
        outer_crust = outer_crust.astype(np.uint8) * 255

        # Очистка от шума
        outer_crust_clean = cv2.morphologyEx(
            outer_crust, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )

        # Фильтрация малых компонент
        labeled, num_components = ndimage.label(outer_crust_clean > 0)
        component_sizes = ndimage.sum(
            outer_crust_clean > 0, labeled, range(1, num_components + 1)
        )

        min_component_size = max(3, wall_count * 0.0005)
        final_crust = np.zeros((h, w), dtype=np.uint8)

        for i, size in enumerate(component_sizes, 1):
            if size >= min_component_size:
                final_crust[labeled == i] = 255

        return final_crust, exterior_mask, interior_mask

    # ============================================================
    # ДЕТЕКЦИЯ УГЛОВ И РАСЧЁТ ПЛОЩАДЕЙ
    # ============================================================

    def detect_inner_corners(self,
                             apartment_boundary: np.ndarray,
                             wall_mask: np.ndarray) -> np.ndarray:
        """
        Детекция внутренних углов помещения по форме квартиры.

        Использует выпуклую оболочку и дефекты выпуклости для поиска внутренних углов.

        Параметры:
            apartment_boundary: Маска границы квартиры
            wall_mask: Маска стен

        Возвращает:
            Маску внутренних углов
        """
        h, w = apartment_boundary.shape

        contours, _ = cv2.findContours(
            apartment_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return np.zeros((h, w), dtype=np.uint8)

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) < 5:
            return np.zeros((h, w), dtype=np.uint8)

        hull = cv2.convexHull(approx, returnPoints=False)
        hull = np.sort(hull, axis=0)

        try:
            defects = cv2.convexityDefects(approx, hull)
        except cv2.error:
            return np.zeros((h, w), dtype=np.uint8)

        inner_corner_mask = np.zeros((h, w), dtype=np.uint8)

        if defects is not None:
            significant_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0
                min_depth = min(h, w) * 0.02
                if depth > min_depth:
                    far = tuple(approx[f][0])
                    significant_defects.append({
                        'point': far,
                        'depth': depth,
                        'start': tuple(approx[s][0]),
                        'end': tuple(approx[e][0])
                    })
                    radius = int(max(10, depth / 2))
                    cv2.circle(inner_corner_mask, far, radius, 255, -1)

        return inner_corner_mask

    def form_apartment_boundary(self,
                                outer_crust: np.ndarray,
                                wall_mask: np.ndarray = None) -> np.ndarray:
        """
        Формирование сплошной формы квартиры по внешней корке (альтернативный метод).

        Параметры:
            outer_crust: Маска внешней корки
            wall_mask: Маска стен (опционально)

        Возвращает:
            Маску формы квартиры
        """
        if np.sum(outer_crust) == 0:
            return np.zeros_like(outer_crust)

        h, w = outer_crust.shape
        base_size = max(3, int(min(h, w) * 0.005))

        kernel_dilate = np.ones((base_size, base_size), dtype=np.uint8)
        kernel_close = np.ones((base_size * 3, base_size * 3), dtype=np.uint8)

        # Соединение корки в сплошную форму
        connected = cv2.dilate(outer_crust, kernel_dilate, iterations=5)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # Заливка внутренней области
        flood_mask = connected.copy()
        h_pad, w_pad = h + 2, w + 2
        mask = np.zeros((h_pad, w_pad), dtype=np.uint8)

        for start_point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(flood_mask, mask, start_point, 128)

        interior = (flood_mask != 128) | (connected > 0)

        # Выбор самого большого контура
        contours, _ = cv2.findContours(
            interior.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return connected

        largest = max(contours, key=cv2.contourArea)

        apartment_shape = np.zeros_like(outer_crust)
        cv2.drawContours(apartment_shape, [largest], -1, 255, -1)

        return apartment_shape

    def calculate_apartment_area_pixels(self, apartment_boundary: np.ndarray) -> int:
        """
        Расчёт общей площади квартиры в пикселях (включая стены).

        Параметры:
            apartment_boundary: Маска границы квартиры

        Возвращает:
            Количество пикселей всей квартиры
        """
        return int(np.sum(apartment_boundary > 0))

    def calculate_living_space(self,
                               apartment_shape: np.ndarray,
                               wall_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Расчёт жилой площади (квартира минус стены).

        Параметры:
            apartment_shape: Маска квартиры
            wall_mask: Маска стен

        Возвращает:
            pixels: Количество пикселей жилой площади
            living_mask: Бинарная маска жилого пространства
        """
        apartment_binary = apartment_shape > 0
        wall_binary = wall_mask > 0

        # Жилая площадь = квартира - стены
        living_space = apartment_binary & ~wall_binary
        pixels = int(np.sum(living_space))

        return pixels, (living_space.astype(np.uint8) * 255)

    # ============================================================
    # OCR И ОПРЕДЕЛЕНИЕ МАСШТАБА
    # ============================================================

    def detect_room_areas_ocr(self,
                              image: np.ndarray,
                              img_path: str) -> List[Dict]:
        """
        Определение общей площади квартиры через OCR.

        Особенности:
        - Использует docTR для распознавания текста
        - Объединяет соседние слова в одну строку
        - Ищет шаблон чисел вида '137,8 м²', '52.0 m2' и т.п.
        - Выбирает только наибольшую площадь (стратегия "largest area")

        Параметры:
            image: Изображение для анализа
            img_path: Путь к изображению

        Возвращает:
            Список с единственным элементом - общей площадью квартиры
        """
        if self.ocr_model is None:
            return []

        from doctr.io import DocumentFile

        doc = DocumentFile.from_images([img_path])
        h, w = image.shape[:2]
        result = self.ocr_model(doc)

        # Шаблон для поиска площади
        area_pattern = re.compile(
            r'^(\d{1,3}[.,]\d{1,2})\s*(?:m²|м²|m2|м2|кв\.?\s*м\.?)?$',
            re.IGNORECASE
        )

        merged_boxes: List[Dict] = []
        individual_words: List[Dict] = []

        # Сбор слов и формирование объединённых боксов по строкам
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = line.words
                    if not words:
                        continue

                    # Сохранение индивидуальных слов
                    for word in words:
                        geom = word.geometry
                        x1 = int(geom[0][0] * w)
                        y1 = int(geom[0][1] * h)
                        x2 = int(geom[1][0] * w)
                        y2 = int(geom[1][1] * h)
                        individual_words.append({
                            'text': word.value.strip(),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': word.confidence
                        })

                    # Объединение слов в строке
                    run_texts: List[str] = []
                    run_confs: List[float] = []
                    xs1: List[int] = []
                    ys1: List[int] = []
                    xs2: List[int] = []
                    ys2: List[int] = []

                    prev_x2 = None
                    prev_h = None

                    for idx, word in enumerate(words):
                        geom = word.geometry
                        x1 = int(geom[0][0] * w)
                        y1 = int(geom[0][1] * h)
                        x2 = int(geom[1][0] * w)
                        y2 = int(geom[1][1] * h)
                        box_h = y2 - y1

                        if idx == 0:
                            prev_x2 = x2
                            prev_h = box_h
                            run_texts = [word.value.strip()]
                            run_confs = [word.confidence]
                            xs1 = [x1]
                            ys1 = [y1]
                            xs2 = [x2]
                            ys2 = [y2]
                            continue

                        gap = x1 - prev_x2
                        avg_h = (prev_h + box_h) / 2.0 if prev_h is not None else box_h
                        gap_thresh = max(5, avg_h * 1.2)

                        if gap <= gap_thresh:
                            # Слова рядом - объединяем
                            run_texts.append(word.value.strip())
                            run_confs.append(word.confidence)
                            xs1.append(x1)
                            ys1.append(y1)
                            xs2.append(x2)
                            ys2.append(y2)
                        else:
                            # Большой разрыв - сохраняем текущую группу
                            if run_texts:
                                mx1, my1 = min(xs1), min(ys1)
                                mx2, my2 = max(xs2), max(ys2)
                                text = " ".join(t for t in run_texts if t)
                                conf = float(np.mean(run_confs)) if run_confs else 0.0
                                merged_boxes.append({
                                    'text': text,
                                    'bbox': (mx1, my1, mx2, my2),
                                    'confidence': conf
                                })
                            # Начинаем новую группу
                            run_texts = [word.value.strip()]
                            run_confs = [word.confidence]
                            xs1 = [x1]
                            ys1 = [y1]
                            xs2 = [x2]
                            ys2 = [y2]

                        prev_x2 = x2
                        prev_h = box_h

                    # Сохранение последней группы
                    if run_texts:
                        mx1, my1 = min(xs1), min(ys1)
                        mx2, my2 = max(xs2), max(ys2)
                        text = " ".join(t for t in run_texts if t)
                        conf = float(np.mean(run_confs)) if run_confs else 0.0
                        merged_boxes.append({
                            'text': text,
                            'bbox': (mx1, my1, mx2, my2),
                            'confidence': conf
                        })

        detected: List[Dict] = []
        seen_centers: List[Tuple[int, int]] = []

        # Обработка объединённых боксов
        for box in merged_boxes:
            text = box['text'].strip()

            if self.debug_ocr:
                print(f"[OCR] Text: '{text}' conf={box['confidence']:.2f}")

            match = area_pattern.match(text)
            if match:
                try:
                    area_str = match.group(1).replace(',', '.')
                    area = float(area_str)
                except ValueError:
                    continue

                # Фильтрация по реалистичным значениям площади
                if not (1.0 <= area <= 1000.0):
                    continue

                x1, y1, x2, y2 = box['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Фильтр дубликатов
                if any(abs(center[0] - c[0]) < 20 and abs(center[1] - c[1]) < 20
                       for c in seen_centers):
                    continue

                seen_centers.append(center)
                detected.append({
                    'area_m2': area,
                    'text': text,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'confidence': box['confidence']
                })

        # Обработка индивидуальных слов (fallback)
        for word in individual_words:
            text = word['text'].strip()

            match = area_pattern.match(text)
            if match:
                try:
                    area_str = match.group(1).replace(',', '.')
                    area = float(area_str)
                except ValueError:
                    continue

                if not (1.0 <= area <= 1000.0):
                    continue

                x1, y1, x2, y2 = word['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if any(abs(center[0] - c[0]) < 20 and abs(center[1] - c[1]) < 20
                       for c in seen_centers):
                    continue

                seen_centers.append(center)
                detected.append({
                    'area_m2': area,
                    'text': text,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'confidence': word['confidence']
                })

        if detected:
            # Сортировка по площади и выбор наибольшей
            detected.sort(key=lambda x: x['area_m2'], reverse=True)
            largest = detected[0]
            # Возвращаем список из одного элемента - общей площади квартиры
            return [largest]
        else:
            return []

    def calculate_pixel_scale(self,
                              apartment_area_pixels: int,
                              total_area_m2: float) -> float:
        """
        Расчёт пиксельного масштаба (метров на пиксель).

        Масштаб рассчитывается на основе полной площади квартиры,
        включая стены, что соответствует стандартному измерению
        общей площади на архитектурных планах.

        Параметры:
            apartment_area_pixels: Количество пикселей всей квартиры (включая стены)
            total_area_m2: Общая площадь квартиры в м²

        Возвращает:
            Масштаб в метрах на пиксель
        """
        if apartment_area_pixels <= 0 or total_area_m2 <= 0:
            return 0.0

        pixel_scale = np.sqrt(total_area_m2 / apartment_area_pixels)
        return pixel_scale

    def measure_walls(self,
                      wall_segments: List[Dict],
                      pixel_scale: float,
                      outer_crust: np.ndarray) -> List[Dict]:
        """
        Измерение длин стен и определение внешних стен.

        Параметры:
            wall_segments: Список сегментов стен
            pixel_scale: Масштаб в метрах на пиксель
            outer_crust: Маска внешней корки

        Возвращает:
            Список сегментов с длиной и флагом внешней стены
        """
        measured: List[Dict] = []
        h, w = outer_crust.shape if outer_crust is not None else (0, 0)

        for seg in wall_segments:
            # Расчёт длины в пикселях
            length_px = seg.get('length_pixels', 0)
            if length_px == 0:
                x1, y1 = seg['x1'], seg['y1']
                x2, y2 = seg['x2'], seg['y2']
                length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Перевод в метры
            length_m = length_px * pixel_scale

            # Определение внешней стены
            is_outer = False
            if outer_crust is not None and h > 0:
                x1, y1 = int(seg['x1']), int(seg['y1'])
                x2, y2 = int(seg['x2']), int(seg['y2'])

                thickness = float(seg.get('thickness', 0.0))
                radius = int(max(3, thickness / 2.0 + 2)) if thickness > 0 else 6

                # Проверка точек вдоль сегмента
                num_samples = max(10, int(length_px / 3))
                for t in np.linspace(0, 1, num_samples):
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))

                    # Проверка окрестности точки
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if outer_crust[ny, nx] > 0:
                                    is_outer = True
                                    break
                        if is_outer:
                            break
                    if is_outer:
                        break

            measured.append({
                'id': seg.get('id', len(measured)),
                'x1': seg['x1'], 'y1': seg['y1'],
                'x2': seg['x2'], 'y2': seg['y2'],
                'length_pixels': length_px,
                'length_meters': length_m,
                'is_outer': is_outer
            })

        return measured

    # ============================================================
    # ОСНОВНОЙ МЕТОД ОБРАБОТКИ
    # ============================================================

    def process(self,
                img_path: str,
                wall_segments: List[Dict],
                output_dir: str,
                wall_mask: np.ndarray = None,
                original_image: np.ndarray = None,
                known_area_m2: float = None) -> Optional[RoomMeasurements]:
        """
        Полный пайплайн анализа размеров помещения.

        Этапы:
        1. Трассировка лучей по оригинальному изображению для контура
        2. Детекция внутренних углов
        3. Расчёт площади квартиры (включая стены) и жилой площади
        4. OCR для определения общей площади
        5. Расчёт пиксельного масштаба (на основе полной площади)
        6. Измерение стен
        7. Сохранение визуализаций для отладки

        Параметры:
            img_path: Путь к изображению
            wall_segments: Список сегментов стен
            output_dir: Директория для вывода
            wall_mask: Маска стен
            original_image: Оригинальное изображение
            known_area_m2: Известная площадь (если есть)

        Возвращает:
            RoomMeasurements или None при ошибке
        """
        # Загрузка изображения
        if original_image is not None:
            image = original_image
        else:
            image = cv2.imread(img_path)
        if image is None:
            return None

        h, w = image.shape[:2]

        if wall_mask is None:
            return None

        # Этап 1: Контур квартиры через трассировку лучей
        outer_crust, apartment_boundary = self.detect_apartment_outline(image)

        if np.sum(outer_crust) == 0 or np.sum(apartment_boundary) == 0:
            return None

        exterior_mask = (apartment_boundary == 0).astype(np.uint8) * 255
        interior_mask = (apartment_boundary > 0).astype(np.uint8) * 255

        # Этап 2: Детекция внутренних углов
        inner_corners = self.detect_inner_corners(apartment_boundary, wall_mask)

        # Этап 3: Расчёт площадей
        # Полная площадь квартиры (включая стены) - используется для масштаба
        apartment_area_pixels = self.calculate_apartment_area_pixels(apartment_boundary)

        # Жилая площадь (без стен) - информационно
        living_space_pixels, living_mask = self.calculate_living_space(
            apartment_boundary, wall_mask
        )

        if apartment_area_pixels == 0:
            return None

        # Этап 4: Определение общей площади
        room_areas: List[Dict] = []
        if known_area_m2 is not None:
            total_area_m2 = known_area_m2
        else:
            room_areas = self.detect_room_areas_ocr(image, img_path)
            if not room_areas:
                return None
            total_area_m2 = float(sum(r['area_m2'] for r in room_areas))

        # Этап 5: Расчёт масштаба (на основе полной площади включая стены)
        pixel_scale = self.calculate_pixel_scale(apartment_area_pixels, total_area_m2)

        if pixel_scale == 0:
            return None

        # Этап 6: Измерение стен
        measured_walls = self.measure_walls(wall_segments, pixel_scale, outer_crust)

        # Этап 7: Сохранение визуализаций
        self._save_debug_images(
            output_dir, img_path, image,
            outer_crust, apartment_boundary, wall_mask,
            living_mask, room_areas, measured_walls, pixel_scale,
            exterior_mask, interior_mask, inner_corners,
            apartment_area_pixels, living_space_pixels
        )

        return RoomMeasurements(
            area_m2=total_area_m2,
            pixel_scale=pixel_scale,
            apartment_area_pixels=apartment_area_pixels,
            living_space_pixels=living_space_pixels,
            walls=measured_walls,
            outer_crust=outer_crust,
            apartment_boundary=apartment_boundary,
            room_areas=room_areas,
            interior_mask=interior_mask,
            exterior_mask=exterior_mask
        )

    def _save_debug_images(self,
                           output_dir: str,
                           img_path: str,
                           image: np.ndarray,
                           outer_crust: np.ndarray,
                           apartment_boundary: np.ndarray,
                           wall_mask: np.ndarray,
                           living_mask: np.ndarray,
                           room_areas: List[Dict],
                           walls: List[Dict],
                           pixel_scale: float,
                           exterior_mask: np.ndarray,
                           interior_mask: np.ndarray,
                           inner_corners: np.ndarray,
                           apartment_area_pixels: int,
                           living_space_pixels: int) -> None:
        """
        Сохранение отладочных визуализаций.

        Создаёт набор изображений для анализа результатов:
        - Отдельные маски (корка, граница, жилое пространство)
        - Композитное изображение
        - Наложение на оригинал
        """
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        base = Path(img_path).stem
        h, w = wall_mask.shape

        # Сохранение масок
        cv2.imwrite(os.path.join(debug_dir, f"{base}_1_outer_crust.png"), outer_crust)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_2_apartment_boundary.png"), apartment_boundary)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_3_living_space.png"), living_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_4_exterior_mask.png"), exterior_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_5_interior_mask.png"), interior_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_6_inner_corners.png"), inner_corners)

        # Композитное изображение
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[exterior_mask > 0] = [50, 50, 50]
        composite[interior_mask > 0] = [255, 220, 180]
        composite[wall_mask > 0] = [100, 100, 100]
        composite[outer_crust > 0] = [0, 255, 0]
        composite[inner_corners > 0] = [255, 255, 0]

        cv2.imwrite(os.path.join(debug_dir, f"{base}_7_composite.png"), composite)

        # Наложение на оригинал
        overlay = image.copy()
        living_overlay = np.zeros_like(overlay)
        living_overlay[living_mask > 0] = [255, 200, 150]
        overlay = cv2.addWeighted(overlay, 0.7, living_overlay, 0.3, 0)

        overlay[outer_crust > 0] = [0, 255, 0]

        # Добавление текстовой информации
        total_area = float(sum(r['area_m2'] for r in room_areas)) if room_areas else 0.0

        # Расчёт площади стен
        wall_area_pixels = apartment_area_pixels - living_space_pixels
        wall_area_m2 = wall_area_pixels * (pixel_scale ** 2)
        living_area_m2 = living_space_pixels * (pixel_scale ** 2)

        info1 = f"Общая площадь: {total_area:.1f} м² | Масштаб: {pixel_scale * 100:.3f} см/пкс"
        info2 = f"Жилая: {living_area_m2:.1f} м² | Стены: {wall_area_m2:.1f} м²"

        cv2.putText(overlay, info1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, info1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(overlay, info2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(overlay, info2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(debug_dir, f"{base}_8_overlay.png"), overlay)