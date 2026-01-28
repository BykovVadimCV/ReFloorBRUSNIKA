"""
Система анализа планировок помещений - Модуль детекции стен на основе цветовой фильтрации
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Цветовая фильтрация стен с комбинированным RGB + HSV подходом
+ формирование единой формы стен с внешними контурами

"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class WallRectangle:
    """
    Осе-ориентированный прямоугольник, аппроксимирующий часть маски стен.

    Атрибуты:
        id: Уникальный идентификатор прямоугольника
        x1, y1: Левый верхний угол (включительно)
        x2, y2: Правый нижний угол (включительно)
        width, height: Размеры прямоугольника
        area: Площадь прямоугольника
        outline: 4 точки контура прямоугольника
    """
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    outline: List[Tuple[int, int]]


# ============================================================
# МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ
# ============================================================

def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """
    Морфологическая очистка маски стен.

    Выполняет закрытие дыр, удаление мелких шумов и фильтрацию
    по минимальной площади компонент.

    Параметры:
        mask: Входная бинарная маска

    Возвращает:
        Очищенную маску
    """
    kernel_small = np.ones((3, 3), np.uint8)

    # Закрытие дыр и удаление шума
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Анализ связных компонент
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Фильтрация малых компонент
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:  # Минимальная площадь компоненты
            cleaned[labels == i] = 255

    return cleaned


# ============================================================
# ДЕТЕКТОР СТЕН
# ============================================================

class ColorBasedWallDetector:
    """
    Детектор стен на основе цветовой фильтрации и прямоугольной декомпозиции.

    Особенности:
    - Прямоугольники используются как внутреннее представление формы стен
    - При визуализации строится общий контур формы без внутренних разделов
    - Поддержка автоматического определения цвета стен
    - Интеграция с внешней OCR моделью
    """

    def __init__(self,
                 wall_color_hex: str = "8E8780",
                 color_tolerance: int = 3,
                 auto_wall_color: bool = False,
                 min_rect_area_ratio: float = 0.0002,
                 rect_visual_gap_px: int = 3,
                 ocr_model: Optional[object] = None,
                 edge_expansion_tolerance: int = 8,
                 max_edge_expansion_px: int = 10):
        """
        Инициализация детектора стен.

        Параметры:
            wall_color_hex: Цвет стен в формате HEX
            color_tolerance: Допуск по RGB каналам
            auto_wall_color: Автоматическое определение цвета стен
            min_rect_area_ratio: Минимальная относительная площадь прямоугольника
            rect_visual_gap_px: Максимальный зазор для объединения при визуализации
            ocr_model: Предварительно инициализированная модель OCR
            edge_expansion_tolerance: Допуск цвета для расширения краёв
            max_edge_expansion_px: Максимальное расширение краёв в пикселях
        """
        self.auto_wall_color = auto_wall_color

        if not self.auto_wall_color:
            self.wall_rgb = self._hex_to_rgb(wall_color_hex)
        else:
            self.wall_rgb = None

        self.color_tolerance = color_tolerance
        self.edge_expansion_tolerance = edge_expansion_tolerance
        self.max_edge_expansion_px = max_edge_expansion_px
        self.min_rect_area_ratio = float(min_rect_area_ratio)
        self.rect_visual_gap_px = int(rect_visual_gap_px)

        # Использование предоставленной OCR модели или инициализация новой
        self.ocr_model = ocr_model
        if self.ocr_model is None:
            self._init_ocr()

    # ============================================================
    # OCR ФУНКЦИОНАЛ
    # ============================================================

    def _init_ocr(self) -> None:
        """Инициализация модели OCR (docTR) для удаления текстовых меток."""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(pretrained=True)
        except Exception:
            self.ocr_model = None

    def get_ocr_bboxes(self,
                      img: np.ndarray,
                      bbox_expansion_px: int = 2) -> List[Tuple[int, int, int, int]]:
        """
        Получение ограничивающих рамок текста через OCR.

        Параметры:
            img: Входное изображение
            bbox_expansion_px: Расширение рамок в пикселях

        Возвращает:
            Список рамок в формате (x1, y1, x2, y2)
        """
        if self.ocr_model is None:
            return []

        h, w = img.shape[:2]
        bboxes = []

        try:
            result = self.ocr_model([img])
        except Exception:
            return []

        try:
            for page in getattr(result, "pages", []):
                for block in getattr(page, "blocks", []):
                    for line in getattr(block, "lines", []):
                        for word in getattr(line, "words", []):
                            geom = getattr(word, "geometry", None)
                            if geom is None:
                                continue

                            x_min_rel = y_min_rel = x_max_rel = y_max_rel = None
                            try:
                                (x_min_rel, y_min_rel), (x_max_rel, y_max_rel) = geom
                            except Exception:
                                try:
                                    if len(geom) == 2 and len(geom[0]) == 2:
                                        x_min_rel, y_min_rel = geom[0]
                                        x_max_rel, y_max_rel = geom[1]
                                    elif len(geom) == 4:
                                        x_min_rel, y_min_rel, x_max_rel, y_max_rel = geom
                                    else:
                                        continue
                                except Exception:
                                    continue

                            if x_min_rel is None or x_max_rel is None:
                                continue

                            # Преобразование относительных координат в абсолютные
                            x_min = int(np.floor(float(x_min_rel) * w))
                            y_min = int(np.floor(float(y_min_rel) * h))
                            x_max = int(np.ceil(float(x_max_rel) * w))
                            y_max = int(np.ceil(float(y_max_rel) * h))

                            # Расширение рамок
                            x_min = max(0, x_min - bbox_expansion_px)
                            y_min = max(0, y_min - bbox_expansion_px)
                            x_max = min(w - 1, x_max + bbox_expansion_px)
                            y_max = min(h - 1, y_max + bbox_expansion_px)

                            if x_max <= x_min or y_max <= y_min:
                                continue

                            bboxes.append((x_min, y_min, x_max, y_max))
        except Exception:
            return []

        return bboxes

    def remove_text_labels_ocr(self,
                               img: np.ndarray,
                               inpaint_radius: int = 3,
                               bbox_expansion_px: int = 2) -> np.ndarray:
        """
        Удаление текстовых меток с изображения через OCR и инпэйнтинг.

        Параметры:
            img: Входное изображение
            inpaint_radius: Радиус инпэйнтинга
            bbox_expansion_px: Расширение рамок текста

        Возвращает:
            Изображение с удалённым текстом
        """
        if self.ocr_model is None:
            return img

        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)

        # Получение рамок текста
        bboxes = self.get_ocr_bboxes(img, bbox_expansion_px)

        # Отрисовка рамок на маске
        for x_min, y_min, x_max, y_max in bboxes:
            cv2.rectangle(text_mask, (x_min, y_min), (x_max, y_max), 255, -1)

        if np.count_nonzero(text_mask) == 0:
            return img

        # Дополнительное расширение маски
        if bbox_expansion_px > 1:
            k = max(1, bbox_expansion_px)
            kernel = np.ones((k, k), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # Инпэйнтинг для удаления текста
        try:
            cleaned = cv2.inpaint(img, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
        except Exception:
            cleaned = img

        return cleaned

    # ============================================================
    # ЦВЕТОВАЯ ДЕТЕКЦИЯ
    # ============================================================

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Конвертация цвета из HEX в RGB формат."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _detect_wall_color_auto(
        self,
        img: np.ndarray,
        min_fraction: float = 0.03,
        k: int = 6,
        max_samples: int = 2000000
    ) -> Tuple[int, int, int]:
        """
        Автоматическое определение цвета стен через кластеризацию.

        Использует K-means для поиска доминирующих цветов и выбирает
        самый тёмный из часто встречающихся.

        Параметры:
            img: Входное изображение
            min_fraction: Минимальная доля для рассмотрения кластера
            k: Количество кластеров
            max_samples: Максимальное количество пикселей для анализа

        Возвращает:
            Цвет стен в формате (R, G, B)
        """
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3).astype(np.float32)

        # Сэмплирование пикселей для ускорения
        if pixels.shape[0] > max_samples:
            idx = np.random.choice(pixels.shape[0], max_samples, replace=False)
            sample = pixels[idx]
        else:
            sample = pixels

        if sample.shape[0] == 0:
            mean_bgr = img.mean(axis=(0, 1)).astype(np.uint8)
            return int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])

        K = min(k, sample.shape[0])
        if K < 2:
            mean_bgr = sample.mean(axis=0).astype(np.uint8)
            return int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])

        # K-means кластеризация
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0
        )
        _, labels, centers = cv2.kmeans(
            sample,
            K,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        centers = centers.astype(np.uint8)

        # Анализ кластеров
        counts = np.bincount(labels, minlength=K).astype(np.float32)
        fractions = counts / float(labels.size)

        # Преобразование в LAB для оценки яркости
        centers_bgr = centers.reshape(K, 1, 3)
        centers_lab = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2LAB).reshape(K, 3)
        L = centers_lab[:, 0]

        # Выбор самого тёмного из достаточно больших кластеров
        candidate_idx = np.where(fractions >= min_fraction)[0]
        if candidate_idx.size == 0:
            candidate_idx = np.arange(K)

        darkest_idx = candidate_idx[np.argmin(L[candidate_idx])]
        chosen_bgr = centers[darkest_idx]

        r = int(chosen_bgr[2])
        g = int(chosen_bgr[1])
        b = int(chosen_bgr[0])
        return r, g, b

    def filter_walls_by_color(self, img: np.ndarray) -> np.ndarray:
        """
        Цветовая фильтрация стен с комбинированным RGB + HSV подходом.

        Параметры:
            img: Входное изображение в формате BGR

        Возвращает:
            Бинарную маску стен (0/255)
        """
        if self.auto_wall_color:
            wall_rgb = self._detect_wall_color_auto(img)
            self.wall_rgb = wall_rgb
        else:
            wall_rgb = self.wall_rgb

        target_r, target_g, target_b = wall_rgb
        tol = self.color_tolerance

        # RGB фильтрация
        img_16 = img.astype(np.int16)
        r_match = np.abs(img_16[:, :, 2] - target_r) <= tol
        g_match = np.abs(img_16[:, :, 1] - target_g) <= tol
        b_match = np.abs(img_16[:, :, 0] - target_b) <= tol
        rgb_mask = (r_match & g_match & b_match).astype(np.uint8) * 255

        # HSV фильтрация
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        target_hsv = cv2.cvtColor(
            np.uint8([[wall_rgb[::-1]]]),
            cv2.COLOR_BGR2HSV
        )[0, 0]

        h_tol = 4
        s_tol = 15
        v_tol = 12

        h = hsv[:, :, 0].astype(np.int16)
        s = hsv[:, :, 1].astype(np.int16)
        v = hsv[:, :, 2].astype(np.int16)

        target_h = int(target_hsv[0])
        target_s = int(target_hsv[1])
        target_v = int(target_hsv[2])

        # Обработка циклической природы Hue
        h_diff = np.abs(h - target_h)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        h_match = h_diff <= h_tol

        s_match = np.abs(s - target_s) <= s_tol
        v_match = np.abs(v - target_v) <= v_tol
        hsv_mask = (h_match & s_match & v_match).astype(np.uint8) * 255

        # Объединение масок
        combined_mask = cv2.bitwise_and(rgb_mask, hsv_mask)

        # Морфологическое закрытие малых разрывов
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return combined_mask

    # ============================================================
    # ОБРАБОТКА СПЕЦИАЛЬНЫХ МЕТОК
    # ============================================================

    def remove_circular_labels(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        Удаление наиболее выраженной круглой метки комнаты.

        Использует преобразование Хафа для поиска кругов.

        Параметры:
            wall_mask: Маска стен

        Возвращает:
            Маску с удалённой круглой меткой
        """
        cleaned = wall_mask.copy()
        blurred = cv2.GaussianBlur(wall_mask, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,
            minRadius=25,
            maxRadius=100
        )

        if circles is not None:
            circles = np.around(circles[0]).astype(np.uint16)
            if len(circles) > 0:
                # Удаление первого (наиболее выраженного) круга
                cx, cy, r = circles[0]
                cv2.circle(cleaned, (cx, cy), r + 5, 0, -1)

        return cleaned

    # ============================================================
    # ПРЯМОУГОЛЬНАЯ ДЕКОМПОЗИЦИЯ
    # ============================================================

    @staticmethod
    def _largest_rectangle_in_mask(mask_bool: np.ndarray) -> Optional[Tuple[int, int, int, int, int]]:
        """
        Поиск наибольшего прямоугольника единиц в бинарной матрице.

        Использует алгоритм на основе стека для эффективного поиска.

        Параметры:
            mask_bool: Бинарная маска (True/False)

        Возвращает:
            (x1, y1, x2, y2, area) или None, если единиц нет
        """
        h, w = mask_bool.shape
        heights = np.zeros(w, dtype=np.int32)

        best_area = 0
        best_coords: Optional[Tuple[int, int, int, int]] = None

        for i in range(h):
            row = mask_bool[i]
            heights = heights + row.astype(np.int32)
            heights[~row] = 0

            stack: List[int] = []
            j = 0
            while j <= w:
                cur_h = heights[j] if j < w else 0
                if not stack or cur_h >= heights[stack[-1]]:
                    stack.append(j)
                    j += 1
                else:
                    top = stack.pop()
                    height = heights[top]
                    if height == 0:
                        continue
                    width = j if not stack else j - stack[-1] - 1
                    area = int(height * width)
                    if area > best_area:
                        best_area = area
                        y2 = i
                        y1 = i - height + 1
                        x2 = j - 1
                        x1 = (stack[-1] + 1) if stack else 0
                        best_coords = (x1, y1, x2, y2)

        if best_coords is None or best_area <= 0:
            return None

        x1, y1, x2, y2 = best_coords
        return x1, y1, x2, y2, best_area

    def _expand_rectangle_edges(self,
                                wall_mask: np.ndarray,
                                img: np.ndarray,
                                rect: WallRectangle) -> WallRectangle:
        """
        Расширение краёв прямоугольника в областях с близким цветом стен.

        Параметры:
            wall_mask: Текущая маска стен
            img: Исходное изображение (BGR)
            rect: Прямоугольник для расширения

        Возвращает:
            Расширенный прямоугольник
        """
        if self.wall_rgb is None:
            return rect

        h, w = wall_mask.shape
        target_r, target_g, target_b = self.wall_rgb
        tol = self.edge_expansion_tolerance

        x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2

        def color_matches(y_coord: int, x_coord: int) -> bool:
            """Проверка близости цвета пикселя к цвету стены."""
            if y_coord < 0 or y_coord >= h or x_coord < 0 or x_coord >= w:
                return False
            b, g, r = img[y_coord, x_coord]
            return (abs(int(r) - target_r) <= tol and
                    abs(int(g) - target_g) <= tol and
                    abs(int(b) - target_b) <= tol)

        # Расширение влево
        for expansion in range(1, self.max_edge_expansion_px + 1):
            new_x1 = x1 - expansion
            if new_x1 < 0:
                break
            matches = sum(color_matches(y, new_x1) for y in range(y1, y2 + 1))
            if matches / (y2 - y1 + 1) < 0.5:  # Минимум 50% совпадения
                break
            x1 = new_x1

        # Расширение вправо
        for expansion in range(1, self.max_edge_expansion_px + 1):
            new_x2 = x2 + expansion
            if new_x2 >= w:
                break
            matches = sum(color_matches(y, new_x2) for y in range(y1, y2 + 1))
            if matches / (y2 - y1 + 1) < 0.5:
                break
            x2 = new_x2

        # Расширение вверх
        for expansion in range(1, self.max_edge_expansion_px + 1):
            new_y1 = y1 - expansion
            if new_y1 < 0:
                break
            matches = sum(color_matches(new_y1, x) for x in range(x1, x2 + 1))
            if matches / (x2 - x1 + 1) < 0.5:
                break
            y1 = new_y1

        # Расширение вниз
        for expansion in range(1, self.max_edge_expansion_px + 1):
            new_y2 = y2 + expansion
            if new_y2 >= h:
                break
            matches = sum(color_matches(new_y2, x) for x in range(x1, x2 + 1))
            if matches / (x2 - x1 + 1) < 0.5:
                break
            y2 = new_y2

        # Создание расширенного прямоугольника
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        area = width * height

        outline = [
            (int(x1), int(y1)),
            (int(x2), int(y1)),
            (int(x2), int(y2)),
            (int(x1), int(y2)),
        ]

        return WallRectangle(
            id=rect.id,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            width=int(width),
            height=int(height),
            area=int(area),
            outline=outline
        )

    def _decompose_to_rectangles(self, wall_mask: np.ndarray, img: Optional[np.ndarray] = None) -> List[WallRectangle]:
        """
        Разложение бинарной маски стен на набор осевых прямоугольников.

        Алгоритм:
        - На каждом шаге ищется максимальный прямоугольник единиц
        - Он записывается и вычитается из маски
        - Края прямоугольников расширяются в областях с близким цветом
        - Процесс продолжается до исчерпания маски

        Параметры:
            wall_mask: Маска стен
            img: Изображение для расширения краёв (опционально)

        Возвращает:
            Список прямоугольников
        """
        h, w = wall_mask.shape
        image_area = h * w
        min_area = int(max(1, self.min_rect_area_ratio * image_area))

        work = (wall_mask > 0).copy()
        rectangles: List[WallRectangle] = []
        rect_id = 0

        while True:
            res = self._largest_rectangle_in_mask(work)
            if res is None:
                break

            x1, y1, x2, y2, area = res
            if area < min_area:
                break

            width = x2 - x1 + 1
            height = y2 - y1 + 1

            outline = [
                (int(x1), int(y1)),
                (int(x2), int(y1)),
                (int(x2), int(y2)),
                (int(x1), int(y2)),
            ]

            rect = WallRectangle(
                id=rect_id,
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
                width=int(width),
                height=int(height),
                area=int(area),
                outline=outline
            )

            # Расширение краёв при наличии изображения
            if img is not None:
                rect = self._expand_rectangle_edges(wall_mask, img, rect)

            rectangles.append(rect)
            rect_id += 1

            # Вычитание исходного прямоугольника из рабочей маски
            work[y1:y2 + 1, x1:x2 + 1] = False

            if not work.any():
                break

        return rectangles

    # ============================================================
    # ОСНОВНОЙ МЕТОД ОБРАБОТКИ
    # ============================================================

    def process(self, img: np.ndarray) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
        """
        Полный пайплайн обработки изображения.

        Этапы:
        1. OCR-очистка текстовых меток
        2. Цветовая фильтрация стен
        3. Удаление круглых меток
        4. Морфологическая очистка
        5. Прямоугольная декомпозиция
        6. Формирование объединённой формы
        7. Построение внешних контуров

        Параметры:
            img: Входное изображение в формате BGR

        Возвращает:
            wall_mask: Бинарная маска стен
            rectangles: Список прямоугольников
            outline_mask: Маска с контурами объединённой формы
        """
        # Этап 1: OCR-очистка текстовых меток
        img_no_text = self.remove_text_labels_ocr(img)

        # Этап 2: Цветовая фильтрация
        wall_mask = self.filter_walls_by_color(img_no_text)

        # Этап 3: Удаление круглой метки
        wall_mask = self.remove_circular_labels(wall_mask)

        # Этап 4: Морфологическая очистка
        wall_mask = morphological_cleanup(wall_mask)

        # Этап 5: Прямоугольная декомпозиция
        rectangles = self._decompose_to_rectangles(wall_mask, img=img_no_text)

        # Этап 6: Формирование объединённой формы стен
        union_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        for rect in rectangles:
            cv2.rectangle(
                union_mask,
                (rect.x1, rect.y1),
                (rect.x2, rect.y2),
                255,
                -1  # Заливка
            )

        # Морфологическое замыкание зазоров между прямоугольниками
        if self.rect_visual_gap_px > 0:
            k = 2 * self.rect_visual_gap_px + 1
            kernel = np.ones((k, k), np.uint8)
            union_mask = cv2.morphologyEx(union_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Этап 7: Построение только внешних контуров
        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(outline_mask, contours, -1, 255, 1)

        return wall_mask, rectangles, outline_mask

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================

    @staticmethod
    def rectangles_to_dict_list(rectangles: List[WallRectangle]) -> List[Dict]:
        """
        Преобразование списка прямоугольников в словари для сериализации.

        Параметры:
            rectangles: Список прямоугольников

        Возвращает:
            Список словарей для JSON/CSV экспорта
        """
        out = []
        for r in rectangles:
            out.append({
                "id": r.id,
                "x1": r.x1,
                "y1": r.y1,
                "x2": r.x2,
                "y2": r.y2,
                "width": r.width,
                "height": r.height,
                "area": r.area,
                "outline": r.outline,
            })
        return out