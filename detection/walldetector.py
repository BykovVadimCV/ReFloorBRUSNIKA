"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: Детекция стен — цветовая + структурная фильтрация (walldetector.py)
Версия: 4.1 — Furniture Filtering Edition
Автор: Быков Вадим Олегович (оригинал), расширение — Grok
Дата:  23 февраля 2026 г.

Описание модуля:
  Универсальная детекция стен на планировках квартир.
  Поддерживает:
    • Залитые цветом стены (оригинальное поведение)
    • Контурные / линейные стены (CAD, сканы, PDF-экспорт)
    • Штрихованные / толстые / двойные линии
    • Смешанные обозначения на одном изображении
    • Автоматический выбор стратегии детекции
    • Фильтрация мебели и внутренних объектов (НОВОЕ в v4.1)

Архитектура:
  Theme                      — единая дизайн-система: цвета, пороги, константы
  WallRectangle              — структура данных прямоугольника стены
  morphological_cleanup      — морфологическая очистка маски
  ColorBasedWallDetector     — базовый детектор (цветовая фильтрация)
  VersatileWallDetector      — расширенный детектор (цвет + структура + гибрид)
                               + фильтрация мебели (v4.1)

Изменения v4.1 vs v4.0:
  - Theme: добавлены STRUCT_MIN_LINE_RATIO, FURNITURE_MIN_ASPECT_RATIO,
    MAX_FURNITURE_AREA_RATIO, ANGLE_TOLERANCE_DEG, PARALLEL_* параметры,
    ROOM_FLOOD_*, STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK
  - VersatileWallDetector: добавлены методы фильтрации мебели:
    * _filter_lines_by_geometry — длина + угловая фильтрация линий
    * _filter_components_by_shape — aspect ratio + fill ratio компонент
    * _remove_internal_objects — flood-fill удаление внутренней мебели
    * _filter_furniture_rects — фильтрация прямоугольников по форме/связности
    * _detect_parallel_wall_pairs — детекция параллельных пар (стены)
    * _should_fallback_to_color — эвристика фоллбэка при избытке компонент
  - process() дополнен шагами фильтрации между маской и декомпозицией.

Принципы:
  - Single Responsibility: каждый класс/функция решает одну задачу
  - Open/Closed: расширение через наследование, без изменения базового класса
  - Сбой одного этапа не прерывает обработку
  - «Магических» чисел нет — все константы вынесены в Theme
  - Полная обратная совместимость: выходной формат НЕИЗМЕНЕН:
      (wall_mask: np.ndarray, rectangles: List[WallRectangle], outline_mask: np.ndarray)
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Theme — единая дизайн-система детектора
# ============================================================

class Theme:
    """Централизованное хранилище всех констант, порогов и параметров детектора.

    Следует принципу единственной ответственности: изменение любого
    числового или цветового параметра требует правки только этого класса.
    """

    # --- Морфологические операции ---
    MORPH_KERNEL_SIZE: int = 3
    MORPH_CLOSE_ITERATIONS: int = 2
    MORPH_OPEN_ITERATIONS: int = 1
    MORPH_COMPONENT_MIN_AREA: int = 5  # ↓ LOWERED from 20: Keep very small components (pillars/short stubs)

    # --- Цветовые допуски (HSV) ---
    HSV_H_TOLERANCE: int = 4
    HSV_S_TOLERANCE: int = 15
    HSV_V_TOLERANCE: int = 12

    # --- Расширение краёв прямоугольников ---
    EDGE_MATCH_MIN_FRACTION: float = 0.5

    # --- Прямоугольная декомпозиция ---
    RECT_VISUAL_GAP_DEFAULT: int = 3

    # --- Автодетекция цвета стен (K-means) ---
    KMEANS_K: int = 6
    KMEANS_MAX_SAMPLES: int = 2_000_000
    KMEANS_MIN_FRACTION: float = 0.03
    KMEANS_ITERATIONS: int = 10
    KMEANS_EPSILON: float = 1.0

    # --- OCR-очистка ---
    OCR_INPAINT_RADIUS_DEFAULT: int = 3
    OCR_BBOX_EXPANSION_DEFAULT: int = 2
    OCR_DILATE_MIN_KERNEL: int = 1

    # --- Удаление круглых меток (преобразование Хафа) ---
    HOUGH_DP: int = 1
    HOUGH_MIN_DIST: int = 20
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 20
    HOUGH_MIN_RADIUS: int = 25
    HOUGH_MAX_RADIUS: int = 100
    HOUGH_CIRCLE_EXPANSION: int = 5
    HOUGH_BLUR_KERNEL: Tuple[int, int] = (5, 5)

    # --- Геометрические ограничения ---
    RECT_STACK_SENTINEL: int = 0

    # --- Структурная детекция (v4.0) ---
    STRUCT_MIN_LINE_LENGTH: int = 5  # ↓ LOWERED from 30/10: Detect very short lines (narrow turns/stubs)
    STRUCT_HOUGH_THRESHOLD: int = 30  # ↓ LOWERED from 50: Detect more/fainter lines (small features)
    STRUCT_HOUGH_MAX_GAP: int = 20  # ↑ INCREASED from 10/15: Connect broken small lines (turns/pillars)
    STRUCT_CANNY_LOW: int = 20  # ↓ LOWERED from 50/30: More sensitive to faint/short edges
    STRUCT_CANNY_HIGH: int = 100  # ↓ LOWERED from 150/120: Avoid missing weak/small edges
    STRUCT_CANNY_EDGES_LOW: int = 10  # ↓ LOWERED from 30/20: Even more edge sensitivity
    STRUCT_CANNY_EDGES_HIGH: int = 60  # ↓ LOWERED from 100/80
    STRUCT_DILATE_KERNEL: int = 3  # Keep small: Less aggressive for narrow details
    STRUCT_DILATE_ITERATIONS: int = 1  # ↓ LOWERED from 2: Preserve narrow/short features
    STRUCT_MIN_WALL_AREA_RATIO: float = 0.0001  # ↓ LOWERED from 0.001/0.0003: Keep tiny wall areas (pillars)
    ADAPTIVE_BLOCK_SIZE: int = 11
    ADAPTIVE_C: int = 2

    # --- Гибридная / авто-детекция (v4.0) ---
    HYBRID_COLOR_COVERAGE_THRESHOLD: float = 0.02
    AUTO_COLOR_COVERAGE_THRESHOLD: float = 0.015
    AUTO_FUSION_CLOSE_KERNEL: int = 3

    # --- Фильтрация мебели (НОВОЕ в v4.1) ---
    STRUCT_MIN_LINE_RATIO: float = 0.005  # ↓ LOWERED from 0.015/0.008: Keep shorter lines as % of image diagonal (turns)
    FURNITURE_MIN_ASPECT_RATIO: float = 1.2  # ↓ LOWERED from 2.5/1.5: Keep squarer shapes

    # Максимальная площадь компоненты как доля от изображения,
    # ниже которой компактные (низкий aspect) объекты удаляются.
    # Крупные компоненты сохраняются даже при низком aspect ratio.
    MAX_FURNITURE_AREA_RATIO: float = 0.005

    # Допуск угла (градусы) для классификации линии как
    # горизонтальной (0°/180°) или вертикальной (90°).
    # Линии за пределами этого допуска — диагональные, отбрасываются.
    ANGLE_TOLERANCE_DEG: float = 5.0

    # Максимальная доля заполнения bounding box контуром.
    # Стены: заполняют bbox на < 70% (вытянутые). Мебель: > 70% (компактные).
    FURNITURE_MAX_FILL_RATIO: float = 0.70

    # Минимальный aspect ratio для прямоугольника декомпозиции.
    # Квадратные прямоугольники (aspect < 1.5) — вероятная мебель.
    RECT_MIN_ASPECT_RATIO: float = 1.5

    # Минимальная площадь прямоугольника как доля изображения.
    # Мелкие rect'ы ниже этого порога удаляются.
    RECT_MIN_AREA_RATIO: float = 0.0003

    # Максимальное расстояние (px) для проверки связности прямоугольника
    # с другими прямоугольниками или краем изображения.
    RECT_CONNECTIVITY_MARGIN_PX: int = 15

    # --- Параллельные стены (v4.1) ---
    # Допуск расстояния между параллельными линиями для
    # идентификации пары как «двойная стена».
    PARALLEL_DISTANCE_MIN_PX: int = 4
    PARALLEL_DISTANCE_MAX_PX: int = 30
    PARALLEL_ANGLE_TOLERANCE_DEG: float = 3.0

    # --- Удаление внутренних объектов через flood-fill (v4.1) ---
    # Минимальная площадь замкнутого контура (доля изображения),
    # чтобы считаться «комнатой» для flood-fill.
    ROOM_FLOOD_MIN_AREA_RATIO: float = 0.005

    # Размер эрозии при удалении внутренних объектов (px).
    ROOM_FLOOD_ERODE_SIZE: int = 3

    # --- Гибридный фоллбэк (v4.1) ---
    # Если структурная детекция порождает слишком много мелких компонент,
    # переключаемся на цветовой режим + морфологию.
    STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK: int = 200


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class WallRectangle:
    """Осе-ориентированный прямоугольник, аппроксимирующий часть маски стен.

    Атрибуты:
        id:             уникальный идентификатор прямоугольника.
        x1, y1:         левый верхний угол (включительно).
        x2, y2:         правый нижний угол (включительно).
        width, height:  размеры прямоугольника.
        area:           площадь прямоугольника.
        outline:        4 точки контура прямоугольника.
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
    """Морфологическая очистка маски стен.

    Выполняет закрытие дыр, удаление мелких шумов и фильтрацию
    по минимальной площади связных компонент.

    Параметры:
        mask: входная бинарная маска.

    Возвращает:
        Очищенную бинарную маску.
    """
    kernel_small = np.ones(
        (Theme.MORPH_KERNEL_SIZE, Theme.MORPH_KERNEL_SIZE), np.uint8
    )

    # Закрытие дыр внутри стен
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_small,
        iterations=Theme.MORPH_CLOSE_ITERATIONS,
    )
    # Удаление мелкого шума
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_small,
        iterations=Theme.MORPH_OPEN_ITERATIONS,
    )

    # Анализ связных компонент и фильтрация слишком маленьких
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > Theme.MORPH_COMPONENT_MIN_AREA:
            cleaned[labels == i] = 255

    return cleaned


# ============================================================
# БАЗОВЫЙ ДЕТЕКТОР СТЕН (ЦВЕТОВАЯ ФИЛЬТРАЦИЯ)
# ============================================================

class ColorBasedWallDetector:
    """Детектор стен на основе цветовой фильтрации и прямоугольной декомпозиции.

    Особенности:
      - Прямоугольники — внутреннее представление формы стен.
      - При визуализации строится общий контур формы без внутренних разделов.
      - Поддержка автоматического определения цвета стен через K-means.
      - Интеграция с внешней OCR-моделью для удаления текстовых меток.
    """

    def __init__(
        self,
        wall_color_hex: str = "8E8780",
        color_tolerance: int = 3,
        auto_wall_color: bool = False,
        min_rect_area_ratio: float = 0.0002,
        rect_visual_gap_px: int = Theme.RECT_VISUAL_GAP_DEFAULT,
        ocr_model: Optional[object] = None,
        edge_expansion_tolerance: int = 8,
        max_edge_expansion_px: int = 10,
    ) -> None:
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

        self.ocr_model = ocr_model
        if self.ocr_model is None:
            self._init_ocr()

    # ============================================================
    # OCR-ФУНКЦИОНАЛ
    # ============================================================

    def _init_ocr(self) -> None:
        """Инициализация модели OCR (docTR) для удаления текстовых меток."""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(pretrained=True)
        except Exception:
            self.ocr_model = None

    def get_ocr_bboxes(
        self,
        img: np.ndarray,
        bbox_expansion_px: int = Theme.OCR_BBOX_EXPANSION_DEFAULT,
    ) -> List[Tuple[int, int, int, int]]:
        """Возвращает ограничивающие рамки текста, найденного OCR."""
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

                            coords = self._parse_ocr_geometry(geom)
                            if coords is None:
                                continue

                            x_min_rel, y_min_rel, x_max_rel, y_max_rel = coords

                            x_min = int(np.floor(float(x_min_rel) * w))
                            y_min = int(np.floor(float(y_min_rel) * h))
                            x_max = int(np.ceil(float(x_max_rel) * w))
                            y_max = int(np.ceil(float(y_max_rel) * h))

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

    def get_ocr_texts_with_coords(
        self,
        img: np.ndarray,
    ) -> List[Tuple[str, int, int, int, int]]:
        """Возвращает распознанный текст вместе с пиксельными координатами."""
        if self.ocr_model is None:
            return []

        h, w = img.shape[:2]
        results = []

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
                            value = getattr(word, "value", None)
                            if geom is None or value is None:
                                continue

                            coords = self._parse_ocr_geometry(geom)
                            if coords is None:
                                continue

                            x_min_rel, y_min_rel, x_max_rel, y_max_rel = coords

                            x_min = max(
                                0, int(np.floor(float(x_min_rel) * w))
                            )
                            y_min = max(
                                0, int(np.floor(float(y_min_rel) * h))
                            )
                            x_max = min(
                                w - 1, int(np.ceil(float(x_max_rel) * w))
                            )
                            y_max = min(
                                h - 1, int(np.ceil(float(y_max_rel) * h))
                            )

                            if x_max <= x_min or y_max <= y_min:
                                continue

                            results.append(
                                (str(value), x_min, y_min, x_max, y_max)
                            )
        except Exception:
            return []

        return results

    def remove_text_labels_ocr(
        self,
        img: np.ndarray,
        inpaint_radius: int = Theme.OCR_INPAINT_RADIUS_DEFAULT,
        bbox_expansion_px: int = Theme.OCR_BBOX_EXPANSION_DEFAULT,
    ) -> np.ndarray:
        """Удаляет текстовые метки с изображения через OCR и инпэйнтинг."""
        if self.ocr_model is None:
            return img

        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)

        bboxes = self.get_ocr_bboxes(img, bbox_expansion_px)
        for x_min, y_min, x_max, y_max in bboxes:
            cv2.rectangle(
                text_mask, (x_min, y_min), (x_max, y_max), 255, -1
            )

        if np.count_nonzero(text_mask) == 0:
            return img

        if bbox_expansion_px > 1:
            k = max(Theme.OCR_DILATE_MIN_KERNEL, bbox_expansion_px)
            kernel = np.ones((k, k), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        try:
            cleaned = cv2.inpaint(
                img, text_mask, inpaint_radius, cv2.INPAINT_TELEA
            )
        except Exception:
            cleaned = img

        return cleaned

    # ============================================================
    # ЦВЕТОВАЯ ДЕТЕКЦИЯ
    # ============================================================

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Конвертирует цвет из формата HEX в кортеж (R, G, B)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _parse_ocr_geometry(
        geom: object,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Разбирает геометрию слова OCR в нормализованные координаты."""
        try:
            (x_min, y_min), (x_max, y_max) = geom
            return float(x_min), float(y_min), float(x_max), float(y_max)
        except Exception:
            pass
        try:
            if len(geom) == 2 and len(geom[0]) == 2:
                x_min, y_min = geom[0]
                x_max, y_max = geom[1]
                return (
                    float(x_min), float(y_min),
                    float(x_max), float(y_max),
                )
            if len(geom) == 4:
                x_min, y_min, x_max, y_max = geom
                return (
                    float(x_min), float(y_min),
                    float(x_max), float(y_max),
                )
        except Exception:
            pass
        return None

    def _detect_wall_color_auto(
        self,
        img: np.ndarray,
        min_fraction: float = Theme.KMEANS_MIN_FRACTION,
        k: int = Theme.KMEANS_K,
        max_samples: int = Theme.KMEANS_MAX_SAMPLES,
    ) -> Tuple[int, int, int]:
        """Автоматически определяет цвет стен через K-means кластеризацию."""
        pixels = img.reshape(-1, 3).astype(np.float32)

        if pixels.shape[0] > max_samples:
            idx = np.random.choice(
                pixels.shape[0], max_samples, replace=False
            )
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

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            Theme.KMEANS_ITERATIONS,
            Theme.KMEANS_EPSILON,
        )
        _, labels, centers = cv2.kmeans(
            sample, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        centers = centers.astype(np.uint8)

        counts = np.bincount(labels, minlength=K).astype(np.float32)
        fractions = counts / float(labels.size)

        centers_bgr = centers.reshape(K, 1, 3)
        centers_lab = cv2.cvtColor(
            centers_bgr, cv2.COLOR_BGR2LAB
        ).reshape(K, 3)
        L = centers_lab[:, 0]

        candidate_idx = np.where(fractions >= min_fraction)[0]
        if candidate_idx.size == 0:
            candidate_idx = np.arange(K)

        darkest_idx = candidate_idx[np.argmin(L[candidate_idx])]
        chosen_bgr = centers[darkest_idx]

        return int(chosen_bgr[2]), int(chosen_bgr[1]), int(chosen_bgr[0])

    def filter_walls_by_color(self, img: np.ndarray) -> np.ndarray:
        """Создаёт бинарную маску стен комбинированным RGB + HSV методом."""
        if self.auto_wall_color:
            wall_rgb = self._detect_wall_color_auto(img)
            self.wall_rgb = wall_rgb
        else:
            wall_rgb = self.wall_rgb

        target_r, target_g, target_b = wall_rgb
        tol = self.color_tolerance

        # --- RGB-фильтрация ---
        img_16 = img.astype(np.int16)
        r_match = np.abs(img_16[:, :, 2] - target_r) <= tol
        g_match = np.abs(img_16[:, :, 1] - target_g) <= tol
        b_match = np.abs(img_16[:, :, 0] - target_b) <= tol
        rgb_mask = (r_match & g_match & b_match).astype(np.uint8) * 255

        # --- HSV-фильтрация ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        target_hsv = cv2.cvtColor(
            np.uint8([[wall_rgb[::-1]]]),
            cv2.COLOR_BGR2HSV,
        )[0, 0]

        h = hsv[:, :, 0].astype(np.int16)
        s = hsv[:, :, 1].astype(np.int16)
        v = hsv[:, :, 2].astype(np.int16)

        target_h = int(target_hsv[0])
        target_s = int(target_hsv[1])
        target_v = int(target_hsv[2])

        h_diff = np.abs(h - target_h)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        h_match = h_diff <= Theme.HSV_H_TOLERANCE

        s_match = np.abs(s - target_s) <= Theme.HSV_S_TOLERANCE
        v_match = np.abs(v - target_v) <= Theme.HSV_V_TOLERANCE
        hsv_mask = (h_match & s_match & v_match).astype(np.uint8) * 255

        combined_mask = cv2.bitwise_and(rgb_mask, hsv_mask)

        kernel = np.ones(
            (Theme.MORPH_KERNEL_SIZE, Theme.MORPH_KERNEL_SIZE), np.uint8
        )
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        return combined_mask

    # ============================================================
    # ОБРАБОТКА СПЕЦИАЛЬНЫХ МЕТОК
    # ============================================================

    def remove_circular_labels(self, wall_mask: np.ndarray) -> np.ndarray:
        """Удаляет наиболее выраженную круглую метку комнаты из маски."""
        cleaned = wall_mask.copy()
        blurred = cv2.GaussianBlur(wall_mask, Theme.HOUGH_BLUR_KERNEL, 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=Theme.HOUGH_DP,
            minDist=Theme.HOUGH_MIN_DIST,
            param1=Theme.HOUGH_PARAM1,
            param2=Theme.HOUGH_PARAM2,
            minRadius=Theme.HOUGH_MIN_RADIUS,
            maxRadius=Theme.HOUGH_MAX_RADIUS,
        )

        if circles is not None:
            circles = np.around(circles[0]).astype(np.uint16)
            if len(circles) > 0:
                cx, cy, r = circles[0]
                cv2.circle(
                    cleaned, (cx, cy),
                    r + Theme.HOUGH_CIRCLE_EXPANSION,
                    0, -1,
                )

        return cleaned

    # ============================================================
    # ПРЯМОУГОЛЬНАЯ ДЕКОМПОЗИЦИЯ
    # ============================================================

    @staticmethod
    def _largest_rectangle_in_mask(
        mask_bool: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """Находит наибольший прямоугольник единиц в бинарной матрице."""
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
                cur_h = (
                    heights[j] if j < w else Theme.RECT_STACK_SENTINEL
                )
                if not stack or cur_h >= heights[stack[-1]]:
                    stack.append(j)
                    j += 1
                else:
                    top = stack.pop()
                    height = heights[top]
                    if height == 0:
                        continue
                    width_val = j if not stack else j - stack[-1] - 1
                    area = int(height * width_val)
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

    def _expand_rectangle_edges(
        self,
        wall_mask: np.ndarray,
        img: np.ndarray,
        rect: WallRectangle,
    ) -> WallRectangle:
        """Расширяет края прямоугольника в пикселях с близким цветом стены."""
        if self.wall_rgb is None:
            return rect

        h, w = wall_mask.shape
        target_r, target_g, target_b = self.wall_rgb
        tol = self.edge_expansion_tolerance

        x1, y1, x2, y2 = rect.x1, rect.y1, rect.x2, rect.y2

        def color_matches(y_coord: int, x_coord: int) -> bool:
            if (
                y_coord < 0 or y_coord >= h
                or x_coord < 0 or x_coord >= w
            ):
                return False
            b, g, r = img[y_coord, x_coord]
            return (
                abs(int(r) - target_r) <= tol
                and abs(int(g) - target_g) <= tol
                and abs(int(b) - target_b) <= tol
            )

        def try_expand(
            coord: int, delta: int, is_x: bool,
            fixed_min: int, fixed_max: int,
        ) -> int:
            new_coord = coord + delta
            if is_x:
                if new_coord < 0 or new_coord >= w:
                    return coord
                matches = sum(
                    color_matches(y, new_coord)
                    for y in range(fixed_min, fixed_max + 1)
                )
                span = fixed_max - fixed_min + 1
            else:
                if new_coord < 0 or new_coord >= h:
                    return coord
                matches = sum(
                    color_matches(new_coord, x)
                    for x in range(fixed_min, fixed_max + 1)
                )
                span = fixed_max - fixed_min + 1
            return (
                new_coord
                if matches / span >= Theme.EDGE_MATCH_MIN_FRACTION
                else coord
            )

        for _ in range(self.max_edge_expansion_px):
            new_x1 = try_expand(
                x1, -1, is_x=True, fixed_min=y1, fixed_max=y2
            )
            if new_x1 == x1:
                break
            x1 = new_x1

        for _ in range(self.max_edge_expansion_px):
            new_x2 = try_expand(
                x2, +1, is_x=True, fixed_min=y1, fixed_max=y2
            )
            if new_x2 == x2:
                break
            x2 = new_x2

        for _ in range(self.max_edge_expansion_px):
            new_y1 = try_expand(
                y1, -1, is_x=False, fixed_min=x1, fixed_max=x2
            )
            if new_y1 == y1:
                break
            y1 = new_y1

        for _ in range(self.max_edge_expansion_px):
            new_y2 = try_expand(
                y2, +1, is_x=False, fixed_min=x1, fixed_max=x2
            )
            if new_y2 == y2:
                break
            y2 = new_y2

        width_val = x2 - x1 + 1
        height_val = y2 - y1 + 1
        area = width_val * height_val
        outline = [
            (int(x1), int(y1)),
            (int(x2), int(y1)),
            (int(x2), int(y2)),
            (int(x1), int(y2)),
        ]

        return WallRectangle(
            id=rect.id,
            x1=int(x1), y1=int(y1),
            x2=int(x2), y2=int(y2),
            width=int(width_val), height=int(height_val),
            area=int(area), outline=outline,
        )

    def _decompose_to_rectangles(
        self,
        wall_mask: np.ndarray,
        img: Optional[np.ndarray] = None,
    ) -> List[WallRectangle]:
        """Разлагает бинарную маску стен на набор осевых прямоугольников."""
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

            width_val = x2 - x1 + 1
            height_val = y2 - y1 + 1
            outline = [
                (int(x1), int(y1)),
                (int(x2), int(y1)),
                (int(x2), int(y2)),
                (int(x1), int(y2)),
            ]

            rect = WallRectangle(
                id=rect_id,
                x1=int(x1), y1=int(y1),
                x2=int(x2), y2=int(y2),
                width=int(width_val), height=int(height_val),
                area=int(area), outline=outline,
            )

            if img is not None:
                rect = self._expand_rectangle_edges(wall_mask, img, rect)

            rectangles.append(rect)
            rect_id += 1

            work[y1:y2 + 1, x1:x2 + 1] = False

            if not work.any():
                break

        return rectangles

    # ============================================================
    # ОСНОВНОЙ МЕТОД ОБРАБОТКИ
    # ============================================================

    def process(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
        """Запускает полный пайплайн обработки изображения.

        Этапы:
          1. OCR-очистка текстовых меток.
          2. Цветовая фильтрация стен.
          3. Удаление круглых меток.
          4. Морфологическая очистка.
          5. Прямоугольная декомпозиция.
          6. Формирование объединённой формы стен.
          7. Построение только внешних контуров.

        Возвращает:
            wall_mask, rectangles, outline_mask
        """
        img_no_text = self.remove_text_labels_ocr(img)
        wall_mask = self.filter_walls_by_color(img_no_text)
        wall_mask = self.remove_circular_labels(wall_mask)
        wall_mask = morphological_cleanup(wall_mask)

        rectangles = self._decompose_to_rectangles(
            wall_mask, img=img_no_text
        )

        union_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        for rect in rectangles:
            cv2.rectangle(
                union_mask,
                (rect.x1, rect.y1),
                (rect.x2, rect.y2),
                255, -1,
            )

        if self.rect_visual_gap_px > 0:
            k = 2 * self.rect_visual_gap_px + 1
            kernel = np.ones((k, k), np.uint8)
            union_mask = cv2.morphologyEx(
                union_mask, cv2.MORPH_CLOSE, kernel, iterations=1
            )

        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outline_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(outline_mask, contours, -1, 255, 1)

        return wall_mask, rectangles, outline_mask

    # ============================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================

    @staticmethod
    def rectangles_to_dict_list(
        rectangles: List[WallRectangle],
    ) -> List[Dict]:
        """Преобразует список прямоугольников в словари для сериализации."""
        return [
            {
                "id": r.id,
                "x1": r.x1, "y1": r.y1,
                "x2": r.x2, "y2": r.y2,
                "width": r.width, "height": r.height,
                "area": r.area, "outline": r.outline,
            }
            for r in rectangles
        ]


# ============================================================
# РАСШИРЕННЫЙ ДЕТЕКТОР СТЕН (ЦВЕТ + СТРУКТУРА + ГИБРИД)
# С ФИЛЬТРАЦИЕЙ МЕБЕЛИ (v4.1)
# ============================================================

class VersatileWallDetector(ColorBasedWallDetector):
    """Универсальный детектор стен: наследует всю цветовую логику
    и добавляет структурную детекцию для линейных/контурных планировок.

    v4.1: Добавлена многоуровневая фильтрация мебели.

    Режимы работы (wall_detection_mode):
      "color"     — только цветовая фильтрация (оригинальное поведение).
      "structure" — только структурная детекция (линии, контуры, пороги).
      "hybrid"    — сначала цвет; если покрытие < 2%, → структура.
      "auto"      — цвет → при низком покрытии → объединение цвета и структуры.

    Метод структурной детекции (structure_method):
      "lines"    — LSD / HoughLinesP с дилатацией (быстрый, рекомендуемый).
      "edges"    — Canny + дилатация (простой, для толстых стен).
      "contours" — адаптивный порог + контурный анализ.

    Выходной формат идентичен ColorBasedWallDetector.process().
    """

    def __init__(
        self,
        # --- Параметры базового класса ---
        wall_color_hex: str = "8E8780",
        color_tolerance: int = 3,
        auto_wall_color: bool = False,
        min_rect_area_ratio: float = 0.0002,
        rect_visual_gap_px: int = Theme.RECT_VISUAL_GAP_DEFAULT,
        ocr_model: Optional[object] = None,
        edge_expansion_tolerance: int = 8,
        max_edge_expansion_px: int = 10,
        # --- Параметры структурной детекции (v4.0) ---
        wall_detection_mode: str = "hybrid",
        structure_method: str = "lines",
        line_thickness_estimate_px: int = 8,
        min_wall_length_ratio: float = 0.02,
        use_adaptive_threshold: bool = True,
        # --- Параметры фильтрации мебели (НОВОЕ v4.1) ---
        enable_furniture_filter: bool = True,
        enable_internal_object_removal: bool = True,
        enable_parallel_wall_detection: bool = False,
        enable_fallback_heuristic: bool = True,
    ) -> None:
        """Инициализация универсального детектора стен.

        Параметры:
            wall_detection_mode:         режим детекции.
            structure_method:            метод структурной детекции.
            line_thickness_estimate_px:  толщина линий при отрисовке.
            min_wall_length_ratio:       мин. длина стены (доля диагонали).
            use_adaptive_threshold:      адаптивная (True) или Otsu (False).
            enable_furniture_filter:     включить фильтрацию мебели (v4.1).
            enable_internal_object_removal: flood-fill удаление внутренних объектов.
            enable_parallel_wall_detection: детекция параллельных пар.
            enable_fallback_heuristic:   фоллбэк на цвет при избытке компонент.
        """
        super().__init__(
            wall_color_hex=wall_color_hex,
            color_tolerance=color_tolerance,
            auto_wall_color=auto_wall_color,
            min_rect_area_ratio=min_rect_area_ratio,
            rect_visual_gap_px=rect_visual_gap_px,
            ocr_model=ocr_model,
            edge_expansion_tolerance=edge_expansion_tolerance,
            max_edge_expansion_px=max_edge_expansion_px,
        )

        # Валидация режима
        valid_modes = ("color", "structure", "hybrid", "auto")
        if wall_detection_mode not in valid_modes:
            raise ValueError(
                f"wall_detection_mode must be one of {valid_modes}, "
                f"got '{wall_detection_mode}'"
            )

        valid_methods = ("lines", "edges", "contours")
        if structure_method not in valid_methods:
            raise ValueError(
                f"structure_method must be one of {valid_methods}, "
                f"got '{structure_method}'"
            )

        self.wall_detection_mode = wall_detection_mode
        self.structure_method = structure_method
        self.line_thickness_estimate_px = int(line_thickness_estimate_px)
        self.min_wall_length_ratio = float(min_wall_length_ratio)
        self.use_adaptive_threshold = bool(use_adaptive_threshold)

        # v4.1 feature flags
        self.enable_furniture_filter = bool(enable_furniture_filter)
        self.enable_internal_object_removal = bool(
            enable_internal_object_removal
        )
        self.enable_parallel_wall_detection = bool(
            enable_parallel_wall_detection
        )
        self.enable_fallback_heuristic = bool(enable_fallback_heuristic)

    # ============================================================
    # ЕДИНАЯ ТОЧКА ВХОДА ДЛЯ СОЗДАНИЯ МАСКИ
    # ============================================================

    def create_wall_mask(self, img: np.ndarray) -> np.ndarray:
        """Создаёт бинарную маску стен в соответствии с выбранным режимом.

        Параметры:
            img: входное изображение в формате BGR (уже без текстовых меток).

        Возвращает:
            Бинарную маску стен (0 / 255).
        """
        if self.wall_detection_mode == "color":
            return self.filter_walls_by_color(img)

        elif self.wall_detection_mode == "structure":
            return self._structure_with_furniture_filter(img)

        elif self.wall_detection_mode == "hybrid":
            color_mask = self.filter_walls_by_color(img)
            coverage = np.count_nonzero(color_mask) / color_mask.size
            if coverage > Theme.HYBRID_COLOR_COVERAGE_THRESHOLD:
                return color_mask
            return self._structure_with_furniture_filter(img)

        else:  # "auto"
            return self._auto_select_detection(img)

    # ============================================================
    # СТРУКТУРНАЯ ДЕТЕКЦИЯ С ФИЛЬТРАЦИЕЙ МЕБЕЛИ (v4.1)
    # ============================================================

    def _structure_with_furniture_filter(
        self, img: np.ndarray,
    ) -> np.ndarray:
        """Структурная детекция с последовательной фильтрацией мебели.

        Пайплайн:
          1. Базовая структурная детекция (filter_walls_by_structure).
          2. Проверка фоллбэка: если слишком много компонент → цвет + морфология.
          3. Фильтрация компонент по форме (aspect ratio, fill ratio).
          4. Удаление внутренних объектов через flood-fill.
          5. Опциональная детекция параллельных пар стен.

        Параметры:
            img: входное изображение BGR.

        Возвращает:
            Очищенную бинарную маску стен.
        """
        mask = self.filter_walls_by_structure(img)

        # --- Шаг 2: фоллбэк при избытке компонент ---
        if self.enable_fallback_heuristic and self._should_fallback_to_color(
            mask
        ):
            color_mask = self.filter_walls_by_color(img)
            color_mask = morphological_cleanup(color_mask)
            # Если цветовая маска тоже пуста, вернём структурную
            if np.count_nonzero(color_mask) > 0:
                return color_mask

        # --- Шаг 3: фильтрация компонент по геометрии ---
        if self.enable_furniture_filter:
            mask = self._filter_components_by_shape(mask)

        # --- Шаг 4: удаление внутренних объектов ---
        if self.enable_internal_object_removal:
            mask = self._remove_internal_objects(mask)

        # --- Шаг 5: параллельные пары ---
        if self.enable_parallel_wall_detection:
            parallel_mask = self._detect_parallel_wall_pairs(img)
            if parallel_mask is not None:
                mask = cv2.bitwise_or(mask, parallel_mask)

        return mask

    # ============================================================
    # СТРУКТУРНАЯ ДЕТЕКЦИЯ СТЕН (v4.0, без изменений)
    # ============================================================

    def filter_walls_by_structure(self, img: np.ndarray) -> np.ndarray:
        """Детектирует стены, определённые контурами/линиями.

        Параметры:
            img: входное изображение в формате BGR.

        Возвращает:
            Бинарную маску стен (0 / 255).
        """
        # --- 1. Подготовка ---
        if (
            hasattr(self, 'remove_text_labels_ocr')
            and self.ocr_model is not None
        ):
            img_clean = self.remove_text_labels_ocr(img)
        else:
            img_clean = img

        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

        # --- 2. Бинаризация ---
        if self.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                Theme.ADAPTIVE_BLOCK_SIZE,
                Theme.ADAPTIVE_C,
            )
        else:
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )

        # --- 3. Морфологическая очистка базовой бинаризации ---
        binary = morphological_cleanup(binary)

        # --- 4. Детекция линий ---
        if self.structure_method == "lines":
            binary = self._detect_lines_lsd_or_hough(gray, binary)
        elif self.structure_method == "edges":
            binary = self._detect_edges_canny(gray, binary)
        # "contours" — только бинаризация + морфология

        # --- 5. Финальная очистка ---
        binary = morphological_cleanup(binary)

        # --- 6. Удаление мелких компонент ---
        #binary = self._remove_small_components(binary)

        return binary

    def _detect_lines_lsd_or_hough(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
    ) -> np.ndarray:
        """Детектирует линейные сегменты через LSD или HoughLinesP.

        v4.1: добавлена фильтрация по длине и углу через
        _filter_lines_by_geometry перед отрисовкой.

        Параметры:
            gray:   изображение в градациях серого.
            binary: текущая бинарная маска.

        Возвращает:
            Обновлённую бинарную маску.
        """
        thickness = self.line_thickness_estimate_px
        h_img, w_img = gray.shape[:2]
        diagonal = np.hypot(w_img, h_img)

        # Минимальная длина: max из фиксированного порога и доли диагонали
        min_length_fixed = Theme.STRUCT_MIN_LINE_LENGTH
        min_length_ratio = Theme.STRUCT_MIN_LINE_RATIO * diagonal
        min_length = max(min_length_fixed, min_length_ratio)

        line_mask = np.zeros_like(gray, dtype=np.uint8)
        raw_lines: List[Tuple[int, int, int, int]] = []
        lines_found = False

        # --- Попытка 1: LSD ---
        try:
            lsd = cv2.createLineSegmentDetector(0)
            lines_result = lsd.detect(gray)

            if lines_result is not None:
                lines = lines_result[0]
                if lines is not None and len(lines) > 0:
                    for line in lines:
                        x1, y1, x2, y2 = map(int, line[0])
                        raw_lines.append((x1, y1, x2, y2))
                    lines_found = True
        except (AttributeError, cv2.error):
            pass

        # --- Попытка 2: HoughLinesP ---
        if not lines_found:
            edges = cv2.Canny(
                gray,
                Theme.STRUCT_CANNY_LOW,
                Theme.STRUCT_CANNY_HIGH,
                apertureSize=3,
            )
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=Theme.STRUCT_HOUGH_THRESHOLD,
                minLineLength=int(min_length_fixed),
                maxLineGap=Theme.STRUCT_HOUGH_MAX_GAP,
            )
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    raw_lines.append((x1, y1, x2, y2))

        # --- v4.1: Фильтрация линий по геометрии ---
        if self.enable_furniture_filter:
            filtered_lines = self._filter_lines_by_geometry(
                raw_lines, min_length, diagonal
            )
        else:
            filtered_lines = [
                seg for seg in raw_lines
                if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= min_length
            ]

        # Отрисовка отфильтрованных линий
        for x1, y1, x2, y2 in filtered_lines:
            cv2.line(
                line_mask, (x1, y1), (x2, y2),
                255, thickness=thickness,
            )

        binary = cv2.bitwise_or(binary, line_mask)
        return binary

    def _detect_edges_canny(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
    ) -> np.ndarray:
        """Детектирует стены через Canny + морфологическую дилатацию.

        Параметры:
            gray:   изображение в градациях серого.
            binary: текущая бинарная маска.

        Возвращает:
            Обновлённую бинарную маску.
        """
        edges = cv2.Canny(
            gray,
            Theme.STRUCT_CANNY_EDGES_LOW,
            Theme.STRUCT_CANNY_EDGES_HIGH,
        )
        kernel = np.ones(
            (Theme.STRUCT_DILATE_KERNEL, Theme.STRUCT_DILATE_KERNEL),
            np.uint8,
        )
        dilated = cv2.dilate(
            edges, kernel,
            iterations=Theme.STRUCT_DILATE_ITERATIONS,
        )

        binary = cv2.bitwise_or(binary, dilated)
        return binary

    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        """Удаляет связные компоненты ниже порога площади.

        Параметры:
            binary: бинарная маска.

        Возвращает:
            Маску с удалёнными мелкими компонентами.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        min_area = int(Theme.STRUCT_MIN_WALL_AREA_RATIO * binary.size)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0

        return binary

    # ============================================================
    # АВТОМАТИЧЕСКИЙ ВЫБОР СТРАТЕГИИ
    # ============================================================

    def _auto_select_detection(self, img: np.ndarray) -> np.ndarray:
        """Автоматически выбирает стратегию на основе эвристик.

        Параметры:
            img: входное изображение BGR.

        Возвращает:
            Бинарную маску стен.
        """
        color_mask = self.filter_walls_by_color(img)
        coverage = np.count_nonzero(color_mask) / color_mask.size

        if coverage > Theme.AUTO_COLOR_COVERAGE_THRESHOLD:
            return color_mask

        # Структурная детекция с фильтрацией мебели
        struct_mask = self._structure_with_furniture_filter(img)

        # Объединение: OR + морфологическое закрытие
        fused = cv2.bitwise_or(color_mask, struct_mask)

        k = Theme.AUTO_FUSION_CLOSE_KERNEL
        kernel = np.ones((k, k), np.uint8)
        fused = cv2.morphologyEx(
            fused, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        return fused

    # ============================================================
    # SOLUTION 1: ФИЛЬТРАЦИЯ ЛИНИЙ ПО ГЕОМЕТРИИ (v4.1)
    # ============================================================

    @staticmethod
    def _filter_lines_by_geometry(
        lines: List[Tuple[int, int, int, int]],
        min_length: float,
        diagonal: float,
    ) -> List[Tuple[int, int, int, int]]:
        """Фильтрует линейные сегменты по длине и ориентации.

        Критерии фильтрации:
          - Длина >= min_length (порог = max(фиксированный, доля диагонали)).
          - Ориентация: горизонтальная (±ANGLE_TOLERANCE_DEG от 0°/180°)
            или вертикальная (±ANGLE_TOLERANCE_DEG от 90°).
          - Диагональные линии отбрасываются: стены на планировках
            почти всегда ортогональны.

        Параметры:
            lines:      список сегментов (x1, y1, x2, y2).
            min_length:  минимальная длина в пикселях.
            diagonal:    диагональ изображения (для контекста).

        Возвращает:
            Отфильтрованный список сегментов.
        """
        angle_tol = Theme.ANGLE_TOLERANCE_DEG
        filtered: List[Tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            # Фильтр по длине
            if length < min_length:
                continue

            # Фильтр по ориентации: вычисляем угол к горизонтали
            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

            # Нормализация в [0, 180)
            angle_deg = angle_deg % 180.0

            is_horizontal = (
                angle_deg <= angle_tol
                or angle_deg >= (180.0 - angle_tol)
            )
            is_vertical = abs(angle_deg - 90.0) <= angle_tol

            if is_horizontal or is_vertical:
                filtered.append((x1, y1, x2, y2))

        return filtered

    # ============================================================
    # SOLUTION 2: ФИЛЬТРАЦИЯ КОМПОНЕНТ ПО ФОРМЕ (v4.1)
    # ============================================================

    @staticmethod
    def _filter_components_by_shape(mask: np.ndarray) -> np.ndarray:
        """Фильтрует связные компоненты по aspect ratio и fill ratio.

        Логика:
          Для каждой связной компоненты вычисляются:
            - aspect_ratio = max(w, h) / max(1, min(w, h))
            - fill_ratio = component_area / bounding_box_area
            - area_ratio = component_area / total_image_pixels

          Компонента СОХРАНЯЕТСЯ, если:
            (a) aspect_ratio >= FURNITURE_MIN_ASPECT_RATIO (вытянутая = стена)
            ИЛИ
            (b) area_ratio >= MAX_FURNITURE_AREA_RATIO (крупная, даже не вытянутая)
            ИЛИ
            (c) fill_ratio < FURNITURE_MAX_FILL_RATIO (не плотно заполненная)

          Компактные, мелкие, плотно заполненные компоненты —
          вероятная мебель — удаляются.

        Параметры:
            mask: бинарная маска (0/255).

        Возвращает:
            Очищенную маску.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        total_pixels = mask.size
        cleaned = np.zeros_like(mask, dtype=np.uint8)

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]

            bbox_area = max(1, comp_w * comp_h)
            dim_max = max(comp_w, comp_h)
            dim_min = max(1, min(comp_w, comp_h))

            aspect_ratio = dim_max / dim_min
            fill_ratio = comp_area / bbox_area
            area_ratio = comp_area / total_pixels

            # Правило сохранения
            is_elongated = (
                aspect_ratio >= Theme.FURNITURE_MIN_ASPECT_RATIO
            )
            is_large = area_ratio >= Theme.MAX_FURNITURE_AREA_RATIO
            is_sparse = fill_ratio < Theme.FURNITURE_MAX_FILL_RATIO

            if is_elongated or is_large or is_sparse:
                cleaned[labels == i] = 255

        return cleaned

    # ============================================================
    # SOLUTION 3: УДАЛЕНИЕ ВНУТРЕННИХ ОБЪЕКТОВ (v4.1)
    # ============================================================

    @staticmethod
    def _remove_internal_objects(wall_mask: np.ndarray) -> np.ndarray:
        """Удаляет внутренние объекты (мебель) через контурный анализ.

        Алгоритм:
          1. Находим все контуры с иерархией (RETR_TREE).
          2. Контуры верхнего уровня (без родителя) — внешние границы
             стен или крупных структур.
          3. Дочерние контуры (уровень 1+) внутри крупных контуров —
             потенциальная мебель / внутренние объекты.
          4. Если дочерний контур:
             - имеет площадь < ROOM_FLOOD_MIN_AREA_RATIO * img_size
               (не «комната»)
             - и его bounding box компактен (aspect < FURNITURE_MIN_ASPECT_RATIO)
             → заполняем его нулями (удаляем из маски).
          5. Это эффективно убирает столы, диваны, лестницы и т.п.,
             «вложенные» внутри контура стен.

        Параметры:
            wall_mask: бинарная маска стен (0/255).

        Возвращает:
            Очищенную маску.
        """
        cleaned = wall_mask.copy()
        total_pixels = wall_mask.size

        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None or len(contours) == 0:
            return cleaned

        hierarchy = hierarchy[0]  # shape: (N, 4) — next, prev, child, parent

        min_room_area = Theme.ROOM_FLOOD_MIN_AREA_RATIO * total_pixels

        for idx in range(len(contours)):
            parent_idx = hierarchy[idx][3]

            # Обрабатываем только дочерние контуры (есть родитель)
            if parent_idx < 0:
                continue

            contour = contours[idx]
            area = cv2.contourArea(contour)

            # Пропускаем крупные внутренние контуры (комнаты, большие полости)
            if area >= min_room_area:
                continue

            # Проверяем компактность: мебель обычно компактна
            x, y, w, h = cv2.boundingRect(contour)
            dim_max = max(w, h)
            dim_min = max(1, min(w, h))
            aspect = dim_max / dim_min

            if aspect < Theme.FURNITURE_MIN_ASPECT_RATIO:
                # Компактный внутренний контур → вероятная мебель → удаляем
                cv2.drawContours(
                    cleaned, [contour], -1, 0, thickness=cv2.FILLED,
                )

        # Лёгкая эрозия для удаления остаточных артефактов на границах
        erode_size = Theme.ROOM_FLOOD_ERODE_SIZE
        if erode_size > 0:
            kernel = np.ones((erode_size, erode_size), np.uint8)
            # Эрозия + обратная дилатация (открытие) — удаляет мелкие выступы
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_OPEN, kernel, iterations=1,
            )

        return cleaned

    # ============================================================
    # SOLUTION 4: ФИЛЬТРАЦИЯ ПРЯМОУГОЛЬНИКОВ (v4.1)
    # ============================================================

    def _filter_furniture_rects(
        self,
        rectangles: List[WallRectangle],
        img_shape: Tuple[int, int],
    ) -> List[WallRectangle]:
        """Фильтрует прямоугольники, вероятно являющиеся мебелью.

        Критерии удаления прямоугольника:
          1. Площадь < RECT_MIN_AREA_RATIO × image_area.
          2. Aspect ratio < RECT_MIN_ASPECT_RATIO (квадратный → мебель).
          3. Не связан с другими прямоугольниками и не касается
             границы изображения (изолированный → мебель).

        Связность проверяется как расстояние между ближайшими
        краями двух прямоугольников <= RECT_CONNECTIVITY_MARGIN_PX.

        Параметры:
            rectangles: список прямоугольников из декомпозиции.
            img_shape:  (height, width) изображения.

        Возвращает:
            Отфильтрованный список прямоугольников (с пересчитанными id).
        """
        if not rectangles:
            return rectangles

        img_h, img_w = img_shape
        image_area = img_h * img_w
        min_area = Theme.RECT_MIN_AREA_RATIO * image_area
        min_aspect = Theme.RECT_MIN_ASPECT_RATIO
        margin = Theme.RECT_CONNECTIVITY_MARGIN_PX

        def touches_border(r: WallRectangle) -> bool:
            """Проверяет, касается ли прямоугольник границы изображения."""
            return (
                r.x1 <= margin
                or r.y1 <= margin
                or r.x2 >= img_w - 1 - margin
                or r.y2 >= img_h - 1 - margin
            )

        def rects_connected(
            a: WallRectangle, b: WallRectangle,
        ) -> bool:
            """Проверяет, находятся ли два прямоугольника рядом."""
            # Расстояние между ближайшими краями по X и Y
            gap_x = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
            gap_y = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))
            return gap_x <= margin and gap_y <= margin

        # Шаг 1: базовая фильтрация по площади и aspect ratio
        candidates: List[Tuple[int, WallRectangle, bool]] = []
        for idx, rect in enumerate(rectangles):
            dim_max = max(rect.width, rect.height)
            dim_min = max(1, min(rect.width, rect.height))
            aspect = dim_max / dim_min

            # Автоматически сохраняем: крупные ИЛИ вытянутые
            auto_keep = rect.area >= min_area and aspect >= min_aspect
            # Также сохраняем крупные даже при низком aspect
            force_keep = rect.area >= min_area * 5  # очень крупные
            candidates.append((idx, rect, auto_keep or force_keep))

        # Шаг 2: проверка связности для «сомнительных» прямоугольников
        kept_rects: List[WallRectangle] = []
        for idx, rect, auto_kept in candidates:
            if auto_kept:
                kept_rects.append(rect)
                continue

            # Проверяем: касается границы?
            if touches_border(rect):
                kept_rects.append(rect)
                continue

            # Проверяем: связан с каким-либо «хорошим» прямоугольником?
            connected_to_good = any(
                rects_connected(rect, other_rect)
                for other_idx, other_rect, other_kept in candidates
                if other_kept and other_idx != idx
            )
            if connected_to_good:
                kept_rects.append(rect)
                continue

            # Иначе — отбрасываем как вероятную мебель
            # (можно добавить логирование здесь при отладке)

        # Пересчёт id
        result: List[WallRectangle] = []
        for new_id, rect in enumerate(kept_rects):
            result.append(WallRectangle(
                id=new_id,
                x1=rect.x1, y1=rect.y1,
                x2=rect.x2, y2=rect.y2,
                width=rect.width, height=rect.height,
                area=rect.area, outline=rect.outline,
            ))

        return result

    # ============================================================
    # SOLUTION 5: ДЕТЕКЦИЯ ПАРАЛЛЕЛЬНЫХ ПАР СТЕН (v4.1)
    # ============================================================

    def _detect_parallel_wall_pairs(
        self, img: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Находит пары параллельных линий на заданном расстоянии
        и заполняет область между ними (двойные стены).

        Алгоритм:
          1. Извлекаем все линии через LSD/Hough.
          2. Группируем по ориентации (горизонтальные / вертикальные).
          3. Для каждой пары проверяем:
             - Параллельность (разница углов < PARALLEL_ANGLE_TOLERANCE_DEG).
             - Расстояние между ними в диапазоне
               [PARALLEL_DISTANCE_MIN_PX, PARALLEL_DISTANCE_MAX_PX].
             - Перекрытие проекций > 50%.
          4. Рисуем заполненную область между такими парами.

        Параметры:
            img: входное изображение BGR.

        Возвращает:
            Маску параллельных стен или None если ничего не найдено.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]
        diagonal = np.hypot(w_img, h_img)
        min_length = max(
            Theme.STRUCT_MIN_LINE_LENGTH,
            Theme.STRUCT_MIN_LINE_RATIO * diagonal,
        )

        # Извлечение линий
        raw_lines: List[Tuple[int, int, int, int]] = []
        try:
            lsd = cv2.createLineSegmentDetector(0)
            result = lsd.detect(gray)
            if result is not None and result[0] is not None:
                for line in result[0]:
                    x1, y1, x2, y2 = map(int, line[0])
                    raw_lines.append((x1, y1, x2, y2))
        except (AttributeError, cv2.error):
            edges = cv2.Canny(
                gray, Theme.STRUCT_CANNY_LOW, Theme.STRUCT_CANNY_HIGH,
            )
            hough = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                Theme.STRUCT_HOUGH_THRESHOLD,
                minLineLength=int(min_length),
                maxLineGap=Theme.STRUCT_HOUGH_MAX_GAP,
            )
            if hough is not None:
                for line in hough:
                    raw_lines.append(tuple(line[0]))

        # Фильтрация по длине и углу
        filtered = self._filter_lines_by_geometry(
            raw_lines, min_length, diagonal
        )

        if len(filtered) < 2:
            return None

        # Разделение на горизонтальные и вертикальные
        horizontal: List[Tuple[int, int, int, int, float]] = []
        vertical: List[Tuple[int, int, int, int, float]] = []
        angle_tol = Theme.ANGLE_TOLERANCE_DEG

        for x1, y1, x2, y2 in filtered:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            length = np.hypot(x2 - x1, y2 - y1)

            if angle <= angle_tol or angle >= (180.0 - angle_tol):
                horizontal.append((x1, y1, x2, y2, angle))
            elif abs(angle - 90.0) <= angle_tol:
                vertical.append((x1, y1, x2, y2, angle))

        parallel_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        found_any = False

        dist_min = Theme.PARALLEL_DISTANCE_MIN_PX
        dist_max = Theme.PARALLEL_DISTANCE_MAX_PX
        par_angle_tol = Theme.PARALLEL_ANGLE_TOLERANCE_DEG

        # Поиск горизонтальных пар (сортировка по Y)
        for group in [horizontal, vertical]:
            if len(group) < 2:
                continue

            is_horiz = (group is horizontal)

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    x1a, y1a, x2a, y2a, ang_a = group[i]
                    x1b, y1b, x2b, y2b, ang_b = group[j]

                    # Проверка параллельности
                    angle_diff = abs(ang_a - ang_b)
                    angle_diff = min(angle_diff, 180.0 - angle_diff)
                    if angle_diff > par_angle_tol:
                        continue

                    # Расстояние между линиями
                    if is_horiz:
                        # Для горизонтальных: расстояние по Y
                        mid_y_a = (y1a + y2a) / 2.0
                        mid_y_b = (y1b + y2b) / 2.0
                        dist = abs(mid_y_a - mid_y_b)

                        # Перекрытие по X
                        xa_min, xa_max = min(x1a, x2a), max(x1a, x2a)
                        xb_min, xb_max = min(x1b, x2b), max(x1b, x2b)
                        overlap_start = max(xa_min, xb_min)
                        overlap_end = min(xa_max, xb_max)
                        overlap = max(0, overlap_end - overlap_start)
                        span = max(
                            1,
                            min(xa_max - xa_min, xb_max - xb_min),
                        )
                    else:
                        # Для вертикальных: расстояние по X
                        mid_x_a = (x1a + x2a) / 2.0
                        mid_x_b = (x1b + x2b) / 2.0
                        dist = abs(mid_x_a - mid_x_b)

                        # Перекрытие по Y
                        ya_min, ya_max = min(y1a, y2a), max(y1a, y2a)
                        yb_min, yb_max = min(y1b, y2b), max(y1b, y2b)
                        overlap_start = max(ya_min, yb_min)
                        overlap_end = min(ya_max, yb_max)
                        overlap = max(0, overlap_end - overlap_start)
                        span = max(
                            1,
                            min(ya_max - ya_min, yb_max - yb_min),
                        )

                    # Проверки
                    if dist < dist_min or dist > dist_max:
                        continue
                    if overlap / span < 0.5:
                        continue

                    # Рисуем заполненный полигон между двумя линиями
                    pts = np.array([
                        [x1a, y1a], [x2a, y2a],
                        [x2b, y2b], [x1b, y1b],
                    ], dtype=np.int32)
                    cv2.fillConvexPoly(parallel_mask, pts, 255)
                    found_any = True

        return parallel_mask if found_any else None

    # ============================================================
    # SOLUTION 6: ЭВРИСТИКА ФОЛЛБЭКА (v4.1)
    # ============================================================

    @staticmethod
    def _should_fallback_to_color(mask: np.ndarray) -> bool:
        """Определяет, нужно ли переключиться на цветовой режим.

        Эвристика: если структурная детекция порождает слишком много
        мелких разрозненных компонент, это признак того, что
        она «подхватила» мебель, текстуры и шумы вместо стен.

        Параметры:
            mask: бинарная маска после структурной детекции.

        Возвращает:
            True если нужен фоллбэк, False иначе.
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        # Вычитаем 1 (фон)
        num_components = num_labels - 1

        if num_components > Theme.STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK:
            return True

        # Дополнительная проверка: если медианная площадь компонент
        # слишком мала — тоже признак шума/мебели
        if num_components > 0:
            areas = [
                stats[i, cv2.CC_STAT_AREA]
                for i in range(1, num_labels)
            ]
            median_area = float(np.median(areas))
            min_expected = Theme.STRUCT_MIN_WALL_AREA_RATIO * mask.size
            # Если медианная компонента меньше 2× минимального порога
            if median_area < 2.0 * min_expected:
                return True

        return False

    # ============================================================
    # ПЕРЕОПРЕДЕЛЕНИЕ process() — v4.1 С ФИЛЬТРАЦИЕЙ МЕБЕЛИ
    # ============================================================

    def process(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
        """Запускает полный пайплайн с фильтрацией мебели (v4.1).

        Этапы:
          1. OCR-очистка текстовых меток.
          2. Создание маски стен (цвет / структура / гибрид / авто)
             с фильтрацией мебели на уровне маски.
          3. Удаление круглых меток.
          4. Морфологическая очистка.
          5. Прямоугольная декомпозиция.
          6. Фильтрация прямоугольников-мебели (v4.1).
          7. Формирование объединённой формы стен.
          8. Построение только внешних контуров.

        Параметры:
            img: входное изображение BGR.

        Возвращает:
            wall_mask:    бинарная маска стен.
            rectangles:   список прямоугольников декомпозиции.
            outline_mask: маска с внешними контурами.
        """
        # 1. Удаление текстовых меток через OCR + инпэйнтинг
        img_no_text = self.remove_text_labels_ocr(img)

        # 2. Создание маски стен (ВКЛЮЧАЕТ фильтрацию мебели на
        #    уровне маски для structure/hybrid/auto режимов)
        wall_mask = self.create_wall_mask(img_no_text)

        # 3. Удаление наиболее выраженной круглой метки
        wall_mask = self.remove_circular_labels(wall_mask)

        # 4. Морфологическая очистка шума и мелких компонент
        wall_mask = morphological_cleanup(wall_mask)

        # 5. Жадная прямоугольная декомпозиция маски
        rectangles = self._decompose_to_rectangles(
            wall_mask, img=img_no_text
        )

        # 6. Фильтрация прямоугольников-мебели (v4.1)
        #    Применяется только для не-цветовых режимов,
        #    где структурная детекция может породить ложные rect'ы.
        if (
            self.enable_furniture_filter
            and self.wall_detection_mode != "color"
        ):
            rectangles = self._filter_furniture_rects(
                rectangles, wall_mask.shape,
            )

        # 7. Объединённая форма из всех прямоугольников
        union_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        for rect in rectangles:
            cv2.rectangle(
                union_mask,
                (rect.x1, rect.y1),
                (rect.x2, rect.y2),
                255, -1,
            )

        # Закрываем зазоры между соседними прямоугольниками
        if self.rect_visual_gap_px > 0:
            k = 2 * self.rect_visual_gap_px + 1
            kernel = np.ones((k, k), np.uint8)
            union_mask = cv2.morphologyEx(
                union_mask, cv2.MORPH_CLOSE, kernel, iterations=1,
            )

        # 8. Только внешние контуры объединённой формы
        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        outline_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(outline_mask, contours, -1, 255, 1)

        return wall_mask, rectangles, outline_mask