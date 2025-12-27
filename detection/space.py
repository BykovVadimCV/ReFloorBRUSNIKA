"""
Быков Вадим Олегович - 19.12.2025

Модуль анализа размеров помещений - v1.4

ПАЙПЛАЙН:
- Детекция контура квартиры через raytracing по ОРИГИНАЛЬНОМУ изображению
- Определение внешних стен (outer crust) и формы квартиры
- Обнаружение разрывов в стенах (двери/окна)
- OCR (docTR) с объединением боксов и стратегией "largest area"
- Расчёт пиксельного масштаба
- Измерение длин стен в метрах и классификация внешних стен

СТЕК:
- OpenCV: обработка изображений, морфология
- docTR: OCR для распознавания текста
- NumPy: векторные операции
- scipy: обработка сигналов

Последнее изменение - 22.12.25 - Largest area OCR strategy + подробный лог
"""

import cv2
import numpy as np
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import ndimage


# ------------------------------------------------------------------ #
# DATA STRUCTURES
# ------------------------------------------------------------------ #

@dataclass
class WallSegment:
    """Структура данных для сегмента стены"""
    x1: int
    y1: int
    x2: int
    y2: int
    thickness_mean: float = 0.0
    thickness_std: float = 0.0


@dataclass
class GapInfo:
    """Структура данных для разрыва в стене"""
    id: int
    center: Tuple[int, int]
    size: int
    bbox: Tuple[int, int, int, int]
    width: int
    height: int
    gap_type: str


@dataclass
class RoomMeasurements:
    """Структура данных для результатов измерений"""
    area_m2: float
    pixel_scale: float
    living_space_pixels: int
    walls: List[Dict]
    outer_crust: np.ndarray
    apartment_boundary: np.ndarray
    room_areas: List[Dict] = field(default_factory=list)
    gap_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    gaps: List[GapInfo] = field(default_factory=list)
    interior_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    exterior_mask: np.ndarray = field(default_factory=lambda: np.array([]))


# ------------------------------------------------------------------ #
# ROOM SIZE ANALYZER
# ------------------------------------------------------------------ #

class RoomSizeAnalyzer:
    """Анализатор размеров помещений с OCR и измерениями стен"""

    def __init__(self,
                 edge_distance_ratio: float = 0.15,
                 white_threshold: int = 240,
                 max_gap_size: int = 50,
                 min_gap_size: int = 5,
                 closing_kernel_ratio: float = 0.025,
                 debug_ocr: bool = False):
        """
        Инициализация анализатора

        Параметры:
            edge_distance_ratio: доля от края для определения внешних стен (исторический параметр)
            white_threshold: порог яркости для определения "белого" фона
            max_gap_size: максимальный размер разрыва в пикселях
            min_gap_size: минимальный размер разрыва в пикселях
            closing_kernel_ratio: относительный размер ядра морфологии
            debug_ocr: флаг детализированного вывода OCR
        """
        self.edge_distance_ratio = edge_distance_ratio
        self.white_threshold = white_threshold
        self.max_gap_size = max_gap_size
        self.min_gap_size = min_gap_size
        self.closing_kernel_ratio = closing_kernel_ratio
        self.debug_ocr = debug_ocr
        self._init_ocr()

    def _init_ocr(self) -> None:
        """Инициализация модели OCR (docTR)"""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(pretrained=True)
            print("[OCR] docTR model initialized")
        except ImportError:
            print("[WARN] docTR not available, OCR disabled")
            self.ocr_model = None

    # ------------------------------------------------------------------ #
    # OUTER OUTLINE VIA RAYTRACING ON ORIGINAL IMAGE
    # ------------------------------------------------------------------ #

    def _connect_outline_gaps(self,
                              mask: np.ndarray,
                              max_component_iters: int = 10,
                              max_endpoint_iters: int = 5) -> np.ndarray:
        """
        Соединение разрывов в контуре квартиры

        1) Соединяем отдельные компоненты (кратчайшая связь между контурами)
        2) Соединяем "торчащие" концы на скелете

        Параметры:
            mask: бинарная маска (контур)
            max_component_iters: максимум итераций для соединения компонент
            max_endpoint_iters: максимум итераций для соединения концов

        Возвращает:
            Маску с замкнутым контуром
        """
        out = mask.copy()
        h, w = out.shape

        # 1) Соединение отдельных компонент
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

        # 2) Соединение открытых концов на скелете
        try:
            from skimage.morphology import skeletonize
            skel = skeletonize(out > 0).astype(np.uint8) * 255
        except ImportError:
            skel = out.copy()

        max_gap_len = max(h, w) * 0.25  # допускаем достаточно длинные замыкания

        for _ in range(max_endpoint_iters):
            ys, xs = np.where(skel > 0)
            endpoints = []
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - 1), min(h, y + 2)
                x0, x1 = max(0, x - 1), min(w, x + 2)
                nb = np.count_nonzero(skel[y0:y1, x0:x1]) - 1
                if nb <= 1:
                    endpoints.append((x, y))

            if len(endpoints) < 2:
                break

            used = [False] * len(endpoints)
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

        contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[RAY] After gap-connecting: {len(contours)} contour(s)")
        return out

    def detect_apartment_outline(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция контура квартиры через raytracing по ОРИГИНАЛЬНОМУ изображению.

        1) Кидаем лучи с краёв к центру, ищем первые тёмные пиксели
        2) Делаем морфологию + замыкание разрывов
        3) Выбираем самый большой контур как квартиру
        4) Строим маску квартиры и внешнюю "корку" (outer crust)

        Параметры:
            image: исходное изображение (BGR)

        Возвращает:
            outer_crust: маска внешних стен (тонкий слой)
            apartment_boundary: заполненная маска квартиры
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        outline = np.zeros((h, w), dtype=np.uint8)
        thr = self.white_threshold

        # Вертикальные лучи
        for x in range(w):
            for y in range(h):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break
            for y in range(h - 1, -1, -1):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break

        # Горизонтальные лучи
        for y in range(h):
            for x in range(w):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break
            for x in range(w - 1, -1, -1):
                if gray[y, x] < thr:
                    outline[y, x] = 255
                    break

        hit_count = int(np.sum(outline > 0))
        print(f"[RAY] Outline hits: {hit_count}")
        if hit_count == 0:
            print("[RAY] No non-white pixels hit from edges")
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty

        # Немного расширяем и закрываем контур
        k = max(3, int(min(h, w) * 0.01))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        outline = cv2.dilate(outline, np.ones((3, 3), np.uint8), 1)
        outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Замыкаем оставшиеся разрывы
        outline = self._connect_outline_gaps(outline)

        contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[RAY] No contour from outline")
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = h * w * 0.01
        print(f"[RAY] Largest outline contour: {area:.0f} px²")
        if area < min_area:
            print("[RAY] Largest contour too small, probably noise")

        apartment = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(apartment, [largest], -1, 255, -1)

        # Выделяем внешнюю "корку" квартиры
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(apartment, erode_k, iterations=1)
        outer_crust = cv2.subtract(apartment, eroded)
        crust_pixels = np.sum(outer_crust > 0)
        print(f"[RAY] Outer crust pixels: {crust_pixels:,}")

        return outer_crust, apartment

    # ------------------------------------------------------------------ #
    # LEGACY MASK-BASED HELPERS (kept for debugging, not for scale)
    # ------------------------------------------------------------------ #

    def detect_exterior_region(self,
                               wall_mask: np.ndarray,
                               closing_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Исторический метод: определение внешней/внутренней области только по маске стен.

        Параметры:
            wall_mask: бинарная маска стен
            closing_iterations: число итераций морфологического замыкания

        Возвращает:
            exterior_mask: маска внешнего пространства
            interior_mask: маска внутреннего пространства
        """
        h, w = wall_mask.shape
        wall_binary = (wall_mask > 0).astype(np.uint8)

        print(f"[EXTERIOR] Image size: {w}x{h}")

        base_size = max(15, int(min(h, w) * self.closing_kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_size, base_size))

        print(f"[EXTERIOR] Closing kernel size: {base_size}x{base_size}")

        apartment_solid = wall_binary * 255
        for _ in range(closing_iterations):
            apartment_solid = cv2.morphologyEx(apartment_solid, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            apartment_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        apartment_filled = np.zeros((h, w), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(apartment_filled, [largest], -1, 255, -1)
            print(f"[EXTERIOR] Largest contour area: {cv2.contourArea(largest)} px²")
        else:
            apartment_filled = apartment_solid
            print("[EXTERIOR] No contours found, using closed mask")

        work = apartment_filled.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Точки по периметру для floodFill
        border_points = set()
        step = max(1, min(w, h) // 50)
        for x in range(0, w, step):
            border_points.add((x, 0))
            border_points.add((x, h - 1))
        for y in range(0, h, step):
            border_points.add((0, y))
            border_points.add((w - 1, y))
        border_points.update([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

        fill_value = 128
        for (x, y) in border_points:
            if 0 <= x < w and 0 <= y < h:
                if work[y, x] == 0:
                    cv2.floodFill(work, flood_mask, (x, y), fill_value)

        exterior_mask = (work == fill_value).astype(np.uint8) * 255
        interior_mask = ((apartment_filled > 0) & (wall_binary == 0)).astype(np.uint8) * 255
        unfilled_interior = ((work != fill_value) & (apartment_filled == 0)).astype(np.uint8) * 255
        interior_mask = cv2.bitwise_or(interior_mask, unfilled_interior)

        exterior_area = np.sum(exterior_mask > 0)
        interior_area = np.sum(interior_mask > 0)
        wall_area = np.sum(wall_binary > 0)

        print(f"[EXTERIOR] Exterior: {exterior_area:,} px")
        print(f"[EXTERIOR] Interior: {interior_area:,} px")
        print(f"[EXTERIOR] Walls: {wall_area:,} px")
        print(f"[EXTERIOR] Sum check: {exterior_area + interior_area + wall_area:,} / {h * w:,}")

        return exterior_mask, interior_mask

    def detect_outer_crust(self,
                           wall_mask: np.ndarray,
                           image: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Исторический метод: вычисление outer crust по маске стен.

        Сейчас основной метод outer_crust строится по raytracing на оригинале,
        но этот оставлен для отладки.

        Параметры:
            wall_mask: маска стен
            image: исходное изображение (опционально, не используется)

        Возвращает:
            final_crust: маска внешней корки
            exterior_mask: внешняя область
            interior_mask: внутренняя область
        """
        h, w = wall_mask.shape
        wall_binary = (wall_mask > 0).astype(np.uint8)

        if not np.any(wall_binary):
            print("[CRUST] No wall pixels found")
            empty = np.zeros((h, w), dtype=np.uint8)
            return empty, empty, empty

        wall_count = np.sum(wall_binary)
        print(f"[CRUST] Total wall pixels: {wall_count:,}")

        exterior_mask, interior_mask = self.detect_exterior_region(wall_mask)

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

        outer_crust = (wall_binary > 0) & (exterior_adjacent > 0)
        outer_crust = outer_crust.astype(np.uint8) * 255

        outer_crust_clean = cv2.morphologyEx(
            outer_crust, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )

        labeled, num_components = ndimage.label(outer_crust_clean > 0)
        component_sizes = ndimage.sum(
            outer_crust_clean > 0, labeled, range(1, num_components + 1)
        )

        min_component_size = max(10, wall_count * 0.001)
        final_crust = np.zeros((h, w), dtype=np.uint8)

        for i, size in enumerate(component_sizes, 1):
            if size >= min_component_size:
                final_crust[labeled == i] = 255

        crust_count = np.sum(final_crust > 0)
        print(f"[CRUST] Outer crust pixels: {crust_count:,} ({crust_count / max(1, wall_count) * 100:.1f}% of walls)")

        interior_adjacent = cv2.dilate(interior_mask, np.ones((3, 3), np.uint8))
        interior_only = (wall_binary > 0) & (interior_adjacent > 0) & (exterior_adjacent == 0)
        interior_wall_count = np.sum(interior_only)
        print(f"[CRUST] Interior-only wall pixels: {interior_wall_count:,} (correctly excluded)")

        return final_crust, exterior_mask, interior_mask

    # ------------------------------------------------------------------ #
    # GAPS / CORNERS / LIVING SPACE / OCR / SCALE
    # ------------------------------------------------------------------ #

    def detect_wall_gaps(self,
                         wall_mask: np.ndarray,
                         exterior_mask: np.ndarray = None,
                         interior_mask: np.ndarray = None) -> Tuple[np.ndarray, List[GapInfo]]:
        """
        Детекция разрывов в стенах (двери, окна и пр.).

        Комбинирует:
        - морфологическое закрытие с разными ядрами,
        - анализ distance transform,
        - (опционально) скелетизацию и поиск "концов" стен.

        Параметры:
            wall_mask: маска стен
            exterior_mask: маска внешнего пространства (опционально)
            interior_mask: маска внутреннего пространства (опционально)

        Возвращает:
            gap_mask: объединённая маска разрывов
            gaps: список структур GapInfo
        """
        h, w = wall_mask.shape
        wall_binary = (wall_mask > 0).astype(np.uint8)

        print(f"[GAPS] Analyzing gaps (max_size={self.max_gap_size}, min_size={self.min_gap_size})")

        gap_sizes = [self.max_gap_size, self.max_gap_size // 2, self.max_gap_size // 4]
        all_gaps = np.zeros((h, w), dtype=np.uint8)

        # Морфологическое закрытие с разными размерами
        for gap_size in gap_sizes:
            if gap_size < self.min_gap_size:
                continue
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (gap_size, gap_size)
            )
            closed = cv2.morphologyEx(wall_binary * 255, cv2.MORPH_CLOSE, kernel)
            added = cv2.subtract(closed, wall_binary * 255)
            all_gaps = cv2.bitwise_or(all_gaps, added)

        # Distance transform по "не-стенам"
        non_wall = (1 - wall_binary).astype(np.uint8)
        dist_transform = cv2.distanceTransform(non_wall, cv2.DIST_L2, 5)
        kernel_max = np.ones((5, 5), dtype=np.uint8)
        local_max = cv2.dilate(dist_transform, kernel_max)
        _ = (dist_transform == local_max) & (dist_transform > self.min_gap_size / 2)

        # Попытка учесть "концы" стен через скелет
        try:
            from skimage.morphology import skeletonize
            wall_skeleton = skeletonize(wall_binary > 0).astype(np.uint8) * 255
            endpoint_kernel = np.array(
                [
                    [1, 1, 1],
                    [1, 10, 1],
                    [1, 1, 1]
                ],
                dtype=np.uint8
            )
            convolved = cv2.filter2D(wall_skeleton // 255, -1, endpoint_kernel)
            endpoints = (convolved == 11) & (wall_skeleton > 0)
            endpoint_regions = cv2.dilate(
                endpoints.astype(np.uint8) * 255,
                np.ones((self.max_gap_size // 2, self.max_gap_size // 2), np.uint8)
            )
            all_gaps = cv2.bitwise_or(all_gaps, endpoint_regions)
        except ImportError:
            print("[GAPS] skimage not available, using morphological method only")

        # Ограничиваемся зоной квартиры, если есть interior/exterior
        if interior_mask is not None and exterior_mask is not None:
            apartment_zone = cv2.bitwise_or(interior_mask, wall_binary * 255)
            apartment_zone = cv2.dilate(apartment_zone, np.ones((10, 10), np.uint8))
            all_gaps = cv2.bitwise_and(all_gaps, apartment_zone)

        # Очистка мелкого шума
        all_gaps = cv2.morphologyEx(
            all_gaps, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )

        # Компонентный анализ
        labeled_gaps, num_gaps = ndimage.label(all_gaps > 0)

        gap_info_list: List[GapInfo] = []
        for i in range(1, num_gaps + 1):
            gap_pixels = np.where(labeled_gaps == i)
            if len(gap_pixels[0]) < self.min_gap_size:
                continue

            y_coords, x_coords = gap_pixels
            size = len(x_coords)

            min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
            min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            aspect_ratio = max(width, height) / max(1, min(width, height))

            if aspect_ratio > 3 and size < 500:
                gap_type = 'doorway'
            elif aspect_ratio > 2 and size < 300:
                gap_type = 'window'
            elif size > 1000:
                gap_type = 'structural'
            else:
                gap_type = 'unknown'

            gap_info_list.append(GapInfo(
                id=i,
                center=(int(np.mean(x_coords)), int(np.mean(y_coords))),
                size=size,
                bbox=(min_x, min_y, max_x, max_y),
                width=width,
                height=height,
                gap_type=gap_type
            ))

        print(f"[GAPS] Found {len(gap_info_list)} significant gaps")
        for gap in gap_info_list[:5]:
            print(f"[GAPS]   Gap {gap.id}: {gap.size}px at {gap.center}, type={gap.gap_type}")
        if len(gap_info_list) > 5:
            print(f"[GAPS]   ... and {len(gap_info_list) - 5} more")

        return all_gaps, gap_info_list

    def detect_inner_corners(self,
                             apartment_boundary: np.ndarray,
                             wall_mask: np.ndarray) -> np.ndarray:
        """
        Детекция внутренних углов помещения по форме квартиры.

        Использует выпуклую оболочку и дефекты выпуклости для поиска "внутренних" углов.
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

            print(f"[CORNERS] Found {len(significant_defects)} inner corners")
            for corner in significant_defects:
                print(f"[CORNERS]   At {corner['point']}, depth={corner['depth']:.1f}px")

        return inner_corner_mask

    def form_apartment_boundary(self,
                                outer_crust: np.ndarray,
                                wall_mask: np.ndarray = None) -> np.ndarray:
        """
        Формирование сплошной формы квартиры по outer_crust (альтернативный путь).

        Сейчас основной контур квартиры приходит напрямую из detect_apartment_outline,
        но этот метод сохранён для отладки.
        """
        if np.sum(outer_crust) == 0:
            print("[BOUNDARY] ERROR: No crust pixels")
            return np.zeros_like(outer_crust)

        h, w = outer_crust.shape
        base_size = max(5, int(min(h, w) * 0.01))

        kernel_dilate = np.ones((base_size, base_size), dtype=np.uint8)
        kernel_close = np.ones((base_size * 3, base_size * 3), dtype=np.uint8)

        connected = cv2.dilate(outer_crust, kernel_dilate, iterations=5)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        flood_mask = connected.copy()
        h_pad, w_pad = h + 2, w + 2
        mask = np.zeros((h_pad, w_pad), dtype=np.uint8)

        for start_point in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(flood_mask, mask, start_point, 128)

        interior = (flood_mask != 128) | (connected > 0)

        contours, _ = cv2.findContours(
            interior.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("[BOUNDARY] No contours found")
            return connected

        largest = max(contours, key=cv2.contourArea)

        apartment_shape = np.zeros_like(outer_crust)
        cv2.drawContours(apartment_shape, [largest], -1, 255, -1)

        print(f"[BOUNDARY] Apartment area: {cv2.contourArea(largest):,} px²")

        return apartment_shape

    def calculate_living_space(self,
                               apartment_shape: np.ndarray,
                               wall_mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Расчёт жилой площади: квартира минус стены.

        Параметры:
            apartment_shape: маска квартиры (внутренняя площадь)
            wall_mask: маска стен

        Возвращает:
            pixels: число пикселей жилой площади
            living_mask: бинарная маска жилого пространства
        """
        apartment_binary = apartment_shape > 0
        wall_binary = wall_mask > 0

        living_space = apartment_binary & ~wall_binary
        pixels = int(np.sum(living_space))

        total_apartment = int(np.sum(apartment_binary))
        walls_inside = int(np.sum(apartment_binary & wall_binary))

        print(f"[SPACE] Apartment total: {total_apartment:,} px²")
        print(f"[SPACE] Walls inside: {walls_inside:,} px² ({walls_inside / max(1, total_apartment) * 100:.1f}%)")
        print(f"[SPACE] Living space: {pixels:,} px² ({pixels / max(1, total_apartment) * 100:.1f}%)")

        return pixels, (living_space.astype(np.uint8) * 255)

    # ------------------------------------------------------------------ #
    # OCR WITH MERGED BOUNDING BOXES + LARGEST AREA STRATEGY
    # ------------------------------------------------------------------ #

    def detect_room_areas_ocr(self,
                              image: np.ndarray,
                              img_path: str) -> List[Dict]:
        """
        Определение общей площади квартиры через OCR.

        Особенности:
        - Используется docTR (ocr_predictor + DocumentFile)
        - Объединяются соседние слова в одну строку (merged boxes)
        - Ищется шаблон чисел вида '137,8 м²', '52.0 m2', '44,5' и т.п.
        - Выбирается ТОЛЬКО крупнейшая площадь (стратегия "largest area")
        """
        if self.ocr_model is None:
            print("[OCR] Model not available")
            return []

        from doctr.io import DocumentFile

        doc = DocumentFile.from_images([img_path])
        h, w = image.shape[:2]
        result = self.ocr_model(doc)

        # Шаблон: m² опционально, чтобы ловить и просто числа "137,8"
        area_pattern = re.compile(
            r'^(\d{1,3}[.,]\d{1,2})\s*(?:m²|м²|m2|м2|кв\.?\s*м\.?)?$',
            re.IGNORECASE
        )

        merged_boxes: List[Dict] = []
        individual_words: List[Dict] = []

        # Сбор слов и формирование "объединённых" боксов по строкам
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    words = line.words
                    if not words:
                        continue

                    # Индивидуальные слова (на всякий случай)
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
                        # Чуть увеличенный порог, чтобы "137,8" и "м²" склеивались
                        gap_thresh = max(5, avg_h * 1.2)

                        if gap <= gap_thresh:
                            run_texts.append(word.value.strip())
                            run_confs.append(word.confidence)
                            xs1.append(x1)
                            ys1.append(y1)
                            xs2.append(x2)
                            ys2.append(y2)
                        else:
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
                            run_texts = [word.value.strip()]
                            run_confs = [word.confidence]
                            xs1 = [x1]
                            ys1 = [y1]
                            xs2 = [x2]
                            ys2 = [y2]

                        prev_x2 = x2
                        prev_h = box_h

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

        # Сначала обрабатываем объединённые боксы
        for box in merged_boxes:
            text = box['text'].strip()
            if self.debug_ocr:
                print(f"[OCR-MERGED] '{text}' conf={box['confidence']:.2f} bbox={box['bbox']}")

            match = area_pattern.match(text)
            if match:
                try:
                    area_str = match.group(1).replace(',', '.')
                    area = float(area_str)
                except ValueError:
                    continue

                # Допускаем 1–1000 м² (квартиры могут быть крупными)
                if not (1.0 <= area <= 1000.0):
                    continue

                x1, y1, x2, y2 = box['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Фильтр дубликатов (рядом лежащие числа)
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
                print(f"[OCR] ✓ Area: {area:.2f} m² ('{text}') conf={box['confidence']:.2f} [MERGED]")

        # Fallback: обрабатываем отдельные слова
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
                print(f"[OCR] ✓ Area: {area:.2f} m² ('{text}') conf={word['confidence']:.2f} [INDIVIDUAL]")

        if detected:
            # Сортируем по площади и берём только самую большую
            detected.sort(key=lambda x: x['area_m2'], reverse=True)
            largest = detected[0]
            print(f"[OCR] Detected {len(detected)} area label(s)")
            print(f"[OCR] ★ LARGEST: {largest['area_m2']:.2f} m² - using as apartment total")

            # Возвращаем список из одного элемента — общей площади квартиры
            return [largest]
        else:
            print(f"[OCR] Detected 0 room areas")
            return []

    def calculate_pixel_scale(self,
                              living_space_pixels: int,
                              total_area_m2: float) -> float:
        """
        Расчёт пиксельного масштаба (метров на пиксель).

        Параметры:
            living_space_pixels: количество пикселей жилой площади
            total_area_m2: общая площадь квартиры (м²)

        Возвращает:
            pixel_scale: масштаб в м/пиксель
        """
        if living_space_pixels <= 0 or total_area_m2 <= 0:
            print("[SCALE] ERROR: Invalid input")
            return 0.0

        pixel_scale = np.sqrt(total_area_m2 / living_space_pixels)

        print(f"[SCALE] sqrt({total_area_m2:.2f} / {living_space_pixels:,})")
        print(f"[SCALE] = {pixel_scale:.6f} m/px = {pixel_scale * 100:.4f} cm/px")
        print(f"[SCALE] 1 meter = {1 / pixel_scale:.1f} pixels")

        return pixel_scale

    def measure_walls(self,
                      wall_segments: List[Dict],
                      pixel_scale: float,
                      outer_crust: np.ndarray) -> List[Dict]:
        """
        Измерение длин стен и определение, какие из них внешние.

        Параметры:
            wall_segments: список сегментов стен (словарей)
            pixel_scale: масштаб (м/пиксель)
            outer_crust: маска внешней корки по контуру квартиры

        Возвращает:
            Список сегментов с длиной в пикселях и метрах и флагом is_outer.
        """
        measured: List[Dict] = []
        h, w = outer_crust.shape if outer_crust is not None else (0, 0)

        for seg in wall_segments:
            length_px = seg.get('length_pixels', 0)
            if length_px == 0:
                x1, y1 = seg['x1'], seg['y1']
                x2, y2 = seg['x2'], seg['y2']
                length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            length_m = length_px * pixel_scale

            is_outer = False
            if outer_crust is not None and h > 0:
                x1, y1 = int(seg['x1']), int(seg['y1'])
                x2, y2 = int(seg['x2']), int(seg['y2'])

                thickness = float(seg.get('thickness', 0.0))
                radius = int(max(3, thickness / 2.0 + 2)) if thickness > 0 else 6

                num_samples = max(10, int(length_px / 3))
                for t in np.linspace(0, 1, num_samples):
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))

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

        outer_count = sum(1 for w_ in measured if w_['is_outer'])
        total_outer_length = sum(w_['length_meters'] for w_ in measured if w_['is_outer'])

        print(f"[WALLS] Total: {len(measured)}")
        print(f"[WALLS] Outer: {outer_count} ({total_outer_length:.2f} m)")

        return measured

    # ------------------------------------------------------------------ #
    # FULL PIPELINE
    # ------------------------------------------------------------------ #

    def process(self,
                img_path: str,
                wall_segments: List[Dict],
                output_dir: str,
                wall_mask: np.ndarray = None,
                original_image: np.ndarray = None,
                known_area_m2: float = None) -> Optional[RoomMeasurements]:
        """
        Полный пайплайн анализа размеров помещения.

        Шаги:
            1) Raytracing по ОРИГИНАЛЬНОМУ изображению -> outer crust + apartment boundary
            2) Детекция разрывов в стенах
            3) Детекция внутренних углов
            4) Расчёт жилой площади
            5) OCR площади квартиры (или использование known_area_m2)
            6) Расчёт пиксельного масштаба
            7) Измерение стен
            8) Сохранение отладочных визуализаций

        Возвращает:
            RoomMeasurements или None при ошибке.
        """
        print("\n" + "=" * 60)
        print("ROOM SIZE ANALYSIS v4.1 - Largest Area Strategy")
        print("=" * 60)

        if original_image is not None:
            image = original_image
        else:
            image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Cannot load: {img_path}")
            return None

        h, w = image.shape[:2]
        print(f"[INPUT] Image: {w}x{h}")
        print(f"[INPUT] Wall segments: {len(wall_segments)}")

        if wall_mask is None:
            print("[ERROR] Wall mask required")
            return None

        # --- STEP 1: Apartment outline via raytracing on ORIGINAL image ---
        print("\n--- STEP 1: Apartment outline via raytracing on ORIGINAL image ---")
        outer_crust, apartment_boundary = self.detect_apartment_outline(image)

        if np.sum(outer_crust) == 0 or np.sum(apartment_boundary) == 0:
            print("[ERROR] Raytracing outline failed (no valid crust/boundary)")
            return None

        exterior_mask = (apartment_boundary == 0).astype(np.uint8) * 255
        interior_mask = (apartment_boundary > 0).astype(np.uint8) * 255

        # --- STEP 2: Detect Wall Gaps ---
        print("\n--- STEP 2: Detect Wall Gaps ---")
        gap_mask, gaps = self.detect_wall_gaps(wall_mask, exterior_mask, interior_mask)

        # --- STEP 3: Detect Inner Corners ---
        print("\n--- STEP 3: Detect Inner Corners ---")
        inner_corners = self.detect_inner_corners(apartment_boundary, wall_mask)

        # --- STEP 4: Calculate Living Space (apartment - walls) ---
        print("\n--- STEP 4: Calculate Living Space (apartment - walls) ---")
        living_space_pixels, living_mask = self.calculate_living_space(
            apartment_boundary, wall_mask
        )

        if living_space_pixels == 0:
            print("[ERROR] No living space calculated")
            return None

        # --- STEP 5: Determine Total Area (OCR) ---
        print("\n--- STEP 5: Determine Total Area (OCR) ---")
        room_areas: List[Dict] = []
        if known_area_m2 is not None:
            total_area_m2 = known_area_m2
            print(f"[AREA] Using provided: {total_area_m2:.2f} m²")
        else:
            room_areas = self.detect_room_areas_ocr(image, img_path)
            if not room_areas:
                print("[ERROR] No room areas detected by OCR")
                return None
            total_area_m2 = float(sum(r['area_m2'] for r in room_areas))
            print(f"[AREA] OCR total: {total_area_m2:.2f} m²")

        # --- STEP 6: Calculate Pixel Scale ---
        print("\n--- STEP 6: Calculate Pixel Scale ---")
        pixel_scale = self.calculate_pixel_scale(living_space_pixels, total_area_m2)

        if pixel_scale == 0:
            print("[ERROR] Could not calculate pixel scale")
            return None

        # --- STEP 7: Measure Walls ---
        print("\n--- STEP 7: Measure Walls ---")
        measured_walls = self.measure_walls(wall_segments, pixel_scale, outer_crust)

        # --- STEP 8: Debug visualizations ---
        self._save_debug_images(
            output_dir, img_path, image,
            outer_crust, apartment_boundary, wall_mask,
            living_mask, room_areas, measured_walls, pixel_scale,
            gap_mask, gaps, exterior_mask, interior_mask, inner_corners
        )

        # Итоговый лог
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"  Total area: {total_area_m2:.2f} m²")
        print(f"  Living space: {living_space_pixels:,} pixels")
        print(f"  Pixel scale: {pixel_scale * 100:.4f} cm/px")
        print(f"  1 meter = {1 / pixel_scale:.1f} pixels")
        print(f"  Gaps detected: {len(gaps)}")
        outer_walls = [w_ for w_ in measured_walls if w_['is_outer']]
        print(f"  Outer walls: {len(outer_walls)}/{len(measured_walls)}")
        if outer_walls:
            total_outer = sum(w_['length_meters'] for w_ in outer_walls)
            print(f"  Outer perimeter: {total_outer:.2f} m")
        print("=" * 60 + "\n")

        return RoomMeasurements(
            area_m2=total_area_m2,
            pixel_scale=pixel_scale,
            living_space_pixels=living_space_pixels,
            walls=measured_walls,
            outer_crust=outer_crust,
            apartment_boundary=apartment_boundary,
            room_areas=room_areas,
            gap_mask=gap_mask,
            gaps=gaps,
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
                           gap_mask: np.ndarray,
                           gaps: List[GapInfo],
                           exterior_mask: np.ndarray,
                           interior_mask: np.ndarray,
                           inner_corners: np.ndarray) -> None:
        """
        Сохранение отладочных визуализаций:
        - отдельные маски (корка, граница, жилое, экстерьер/интерьер, разрывы, углы)
        - композитное изображение
        - оверлей на оригинале
        """
        debug_dir = os.path.join(output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        base = Path(img_path).stem
        h, w = wall_mask.shape

        # Маски
        cv2.imwrite(os.path.join(debug_dir, f"{base}_1_outer_crust.png"), outer_crust)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_2_apartment_boundary.png"), apartment_boundary)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_3_living_space.png"), living_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_4_exterior_mask.png"), exterior_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_5_interior_mask.png"), interior_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_6_gap_mask.png"), gap_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base}_7_inner_corners.png"), inner_corners)

        # Композит
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[exterior_mask > 0] = [50, 50, 50]
        composite[interior_mask > 0] = [255, 220, 180]
        composite[wall_mask > 0] = [100, 100, 100]
        composite[outer_crust > 0] = [0, 255, 0]
        composite[gap_mask > 0] = [255, 0, 0]
        composite[inner_corners > 0] = [255, 255, 0]

        for gap in gaps:
            cx, cy = gap.center
            cv2.circle(composite, (cx, cy), 5, (255, 0, 255), -1)
            cv2.putText(composite, f"{gap.gap_type}", (cx + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(debug_dir, f"{base}_8_composite.png"), composite)

        # Оверлей на исходном изображении
        overlay = image.copy()
        living_overlay = np.zeros_like(overlay)
        living_overlay[living_mask > 0] = [255, 200, 150]
        overlay = cv2.addWeighted(overlay, 0.7, living_overlay, 0.3, 0)

        overlay[outer_crust > 0] = [0, 255, 0]
        overlay[gap_mask > 0] = [0, 0, 255]

        total_area = float(sum(r['area_m2'] for r in room_areas)) if room_areas else 0.0
        info = f"Area: {total_area:.1f}m² | Scale: {pixel_scale * 100:.3f}cm/px | Gaps: {len(gaps)}"
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(overlay, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(debug_dir, f"{base}_9_overlay.png"), overlay)

        print(f"[DEBUG] Saved visualizations to {debug_dir}/")