"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: Детекция стен — цветовая + структурная + контурная фильтрация
Версия: 5.0 — Outline / БТИ Wall Detection Edition
Автор: Быков Вадим Олегович (оригинал), расширение — Grok, Claude
Дата:  4 марта 2026 г.

Описание модуля:
  Универсальная детекция стен на планировках квартир.
  Поддерживает:
    • Залитые цветом стены (оригинальное поведение)
    • Контурные / линейные стены (CAD, сканы, PDF-экспорт)
    • Штрихованные / толстые / двойные линии
    • Стены-контуры в стиле БТИ (НОВОЕ в v5.0):
        — Стены изображены как чёрные контуры прямоугольников
        — Несколько прямоугольников «слиты» в единый контур
        — Фильтрация цифр, размерных линий, выносок
    • Смешанные обозначения на одном изображении
    • Автоматический выбор стратегии детекции
    • Фильтрация мебели и внутренних объектов

Архитектура (ПОРЯДОК ОПРЕДЕЛЕНИЯ = ПОРЯДОК ЗАВИСИМОСТЕЙ):
  1. Theme                      — единая дизайн-система: цвета, пороги, константы
  2. WallRectangle              — структура данных прямоугольника стены
  3. morphological_cleanup      — морфологическая очистка маски
  4. ColorBasedWallDetector     — базовый детектор (цветовая фильтрация)
  5. WallStyleClassifier        — классификатор типа изображения стен (v5.0)
  6. OutlineWallDetector        — детектор контурных стен БТИ (v5.0)
     └── наследует ColorBasedWallDetector
  7. VersatileWallDetector      — расширенный детектор
     └── наследует ColorBasedWallDetector, использует OutlineWallDetector

Выходной формат НЕИЗМЕНЕН:
    (wall_mask: np.ndarray, rectangles: List[WallRectangle], outline_mask: np.ndarray)
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 1. Theme — единая дизайн-система детектора
# ============================================================

class Theme:
    """Централизованное хранилище всех констант, порогов и параметров детектора."""

    # --- Морфологические операции ---
    MORPH_KERNEL_SIZE: int = 3
    MORPH_CLOSE_ITERATIONS: int = 2
    MORPH_OPEN_ITERATIONS: int = 1
    MORPH_COMPONENT_MIN_AREA: int = 5

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
    STRUCT_MIN_LINE_LENGTH: int = 5
    STRUCT_HOUGH_THRESHOLD: int = 30
    STRUCT_HOUGH_MAX_GAP: int = 20
    STRUCT_CANNY_LOW: int = 20
    STRUCT_CANNY_HIGH: int = 100
    STRUCT_CANNY_EDGES_LOW: int = 10
    STRUCT_CANNY_EDGES_HIGH: int = 60
    STRUCT_DILATE_KERNEL: int = 3
    STRUCT_DILATE_ITERATIONS: int = 1
    STRUCT_MIN_WALL_AREA_RATIO: float = 0.0001
    ADAPTIVE_BLOCK_SIZE: int = 11
    ADAPTIVE_C: int = 2

    # --- Гибридная / авто-детекция (v4.0) ---
    HYBRID_COLOR_COVERAGE_THRESHOLD: float = 0.02
    AUTO_COLOR_COVERAGE_THRESHOLD: float = 0.015
    AUTO_FUSION_CLOSE_KERNEL: int = 3

    # --- Фильтрация мебели (v4.1) ---
    STRUCT_MIN_LINE_RATIO: float = 0.005
    FURNITURE_MIN_ASPECT_RATIO: float = 1.2
    MAX_FURNITURE_AREA_RATIO: float = 0.005
    ANGLE_TOLERANCE_DEG: float = 5.0
    FURNITURE_MAX_FILL_RATIO: float = 0.70
    RECT_MIN_ASPECT_RATIO: float = 1.5
    RECT_MIN_AREA_RATIO: float = 0.0003
    RECT_CONNECTIVITY_MARGIN_PX: int = 15

    # --- Параллельные стены (v4.1) ---
    PARALLEL_DISTANCE_MIN_PX: int = 4
    PARALLEL_DISTANCE_MAX_PX: int = 30
    PARALLEL_ANGLE_TOLERANCE_DEG: float = 3.0

    # --- Удаление внутренних объектов (v4.1) ---
    ROOM_FLOOD_MIN_AREA_RATIO: float = 0.005
    ROOM_FLOOD_ERODE_SIZE: int = 3

    # --- Гибридный фоллбэк (v4.1) ---
    STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK: int = 200

    # ==========================================================
    # КОНТУРНЫЕ / БТИ СТЕНЫ (НОВОЕ v5.0)
    # ==========================================================

    # --- Классификатор стиля стен ---
    CLASSIFIER_BLACK_GRAY_THRESH: int = 80
    CLASSIFIER_OUTLINE_BLACK_RATIO_MIN: float = 0.005
    CLASSIFIER_OUTLINE_BLACK_RATIO_MAX: float = 0.18
    CLASSIFIER_FILLED_COVERAGE_MIN: float = 0.015
    CLASSIFIER_THIN_COMPONENT_RATIO_MIN: float = 0.25

    # --- Извлечение чёрных пикселей ---
    OUTLINE_BLACK_THRESH: int = 90
    OUTLINE_ADAPTIVE_BLOCK: int = 25
    OUTLINE_ADAPTIVE_C: int = 10
    OUTLINE_USE_ADAPTIVE: bool = True

    # --- Фильтрация текста / цифр ---
    TEXT_FILTER_MAX_AREA_RATIO: float = 0.0015
    TEXT_FILTER_MIN_HEIGHT_RATIO: float = 0.005
    TEXT_FILTER_MAX_HEIGHT_RATIO: float = 0.04
    TEXT_FILTER_MAX_ASPECT: float = 4.0
    TEXT_FILTER_MIN_FILL: float = 0.15
    TEXT_FILTER_MAX_FILL: float = 0.85
    TEXT_FILTER_CLUSTER_GAP_PX: int = 15
    TEXT_FILTER_MIN_CLUSTER_SIZE: int = 2

    # --- Фильтрация размерных линий ---
    DIMLINE_MAX_THIN_DIM_PX: int = 3
    DIMLINE_MIN_ASPECT: float = 8.0
    DIMLINE_MAX_AREA_RATIO: float = 0.002
    DIMLINE_CONNECTIVITY_RADIUS_PX: int = 5

    # --- Заполнение контурных стен ---
    OUTLINE_FILL_CLOSE_KERNEL: int = 9
    OUTLINE_FILL_DIRECTIONAL_KERNEL_LENGTH: int = 11
    OUTLINE_FILL_DIRECTIONAL_KERNEL_WIDTH: int = 3
    OUTLINE_FILL_CLOSE_ITERATIONS: int = 2
    OUTLINE_MIN_WALL_AREA_RATIO: float = 0.0003

    # --- Финальная очистка контурного детектора ---
    OUTLINE_FINAL_OPEN_KERNEL: int = 3
    OUTLINE_FINAL_OPEN_ITERATIONS: int = 1
    OUTLINE_MIN_FILLED_COMPONENT_RATIO: float = 0.0005
    OUTLINE_MIN_RECTANGULARITY: float = 0.3
    OUTLINE_ROOM_HOLE_MIN_RATIO: float = 0.003


# ============================================================
# 2. СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class WallRectangle:
    """Осе-ориентированный прямоугольник, аппроксимирующий часть маски стен."""

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
# 3. МОРФОЛОГИЧЕСКИЕ ОПЕРАЦИИ
# ============================================================

def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Морфологическая очистка маски стен."""
    kernel_small = np.ones(
        (Theme.MORPH_KERNEL_SIZE, Theme.MORPH_KERNEL_SIZE), np.uint8
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_small,
        iterations=Theme.MORPH_CLOSE_ITERATIONS,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_small,
        iterations=Theme.MORPH_OPEN_ITERATIONS,
    )
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
# 4. БАЗОВЫЙ ДЕТЕКТОР СТЕН (ЦВЕТОВАЯ ФИЛЬТРАЦИЯ)
#    ← Определяется ПЕРВЫМ, т.к. от него наследуют остальные
# ============================================================

class ColorBasedWallDetector:
    """Детектор стен на основе цветовой фильтрации и прямоугольной декомпозиции."""

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

    # ---- OCR ----

    def _init_ocr(self) -> None:
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

                            x_min = max(0, int(np.floor(float(x_min_rel) * w)))
                            y_min = max(0, int(np.floor(float(y_min_rel) * h)))
                            x_max = min(w - 1, int(np.ceil(float(x_max_rel) * w)))
                            y_max = min(h - 1, int(np.ceil(float(y_max_rel) * h)))

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

    # ---- ЦВЕТОВАЯ ДЕТЕКЦИЯ ----

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _parse_ocr_geometry(
        geom: object,
    ) -> Optional[Tuple[float, float, float, float]]:
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
        if self.auto_wall_color:
            wall_rgb = self._detect_wall_color_auto(img)
            self.wall_rgb = wall_rgb
        else:
            wall_rgb = self.wall_rgb

        target_r, target_g, target_b = wall_rgb
        tol = self.color_tolerance

        img_16 = img.astype(np.int16)
        r_match = np.abs(img_16[:, :, 2] - target_r) <= tol
        g_match = np.abs(img_16[:, :, 1] - target_g) <= tol
        b_match = np.abs(img_16[:, :, 0] - target_b) <= tol
        rgb_mask = (r_match & g_match & b_match).astype(np.uint8) * 255

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

    # ---- СПЕЦИАЛЬНЫЕ МЕТКИ ----

    def remove_circular_labels(self, wall_mask: np.ndarray) -> np.ndarray:
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

    # ---- ПРЯМОУГОЛЬНАЯ ДЕКОМПОЗИЦИЯ ----

    @staticmethod
    def _largest_rectangle_in_mask(
        mask_bool: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, int]]:
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
            new_x1 = try_expand(x1, -1, is_x=True, fixed_min=y1, fixed_max=y2)
            if new_x1 == x1:
                break
            x1 = new_x1

        for _ in range(self.max_edge_expansion_px):
            new_x2 = try_expand(x2, +1, is_x=True, fixed_min=y1, fixed_max=y2)
            if new_x2 == x2:
                break
            x2 = new_x2

        for _ in range(self.max_edge_expansion_px):
            new_y1 = try_expand(y1, -1, is_x=False, fixed_min=x1, fixed_max=x2)
            if new_y1 == y1:
                break
            y1 = new_y1

        for _ in range(self.max_edge_expansion_px):
            new_y2 = try_expand(y2, +1, is_x=False, fixed_min=x1, fixed_max=x2)
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

    # ---- ОСНОВНОЙ МЕТОД ОБРАБОТКИ ----

    def process(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
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

    @staticmethod
    def rectangles_to_dict_list(
        rectangles: List[WallRectangle],
    ) -> List[Dict]:
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
# 5. КЛАССИФИКАТОР ТИПА СТЕН (v5.0)
#    ← После ColorBasedWallDetector (использует его в classify)
# ============================================================

class WallStyleClassifier:
    """Определяет, каким способом изображены стены на планировке.

    Возвращаемые типы:
      "filled"  — стены залиты одним цветом.
      "outline" — стены показаны как чёрные контуры (БТИ-стиль).
      "mixed"   — смесь обоих подходов.
    """

    @staticmethod
    def classify(
        img: np.ndarray,
        color_detector: Optional[ColorBasedWallDetector] = None,
    ) -> str:
        """Классифицирует тип изображения стен.

        Метрики:
          1. Доля чёрных пикселей (для outline: 0.5–18%).
          2. Покрытие цветовой маской (для filled: >1.5%).
          3. Доля тонких CC (для outline: >25% вытянутых).
          4. Наличие параллельных пар линий (для outline).
        """
        h, w = img.shape[:2]
        total_pixels = h * w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Метрика 1: доля чёрных пикселей ---
        black_mask = (
            gray < Theme.CLASSIFIER_BLACK_GRAY_THRESH
        ).astype(np.uint8) * 255
        black_ratio = np.count_nonzero(black_mask) / total_pixels

        has_black_outlines = (
            Theme.CLASSIFIER_OUTLINE_BLACK_RATIO_MIN
            <= black_ratio
            <= Theme.CLASSIFIER_OUTLINE_BLACK_RATIO_MAX
        )

        # --- Метрика 2: покрытие цветовой маской ---
        color_coverage = 0.0
        if color_detector is not None:
            try:
                color_mask = color_detector.filter_walls_by_color(img)
                color_coverage = np.count_nonzero(color_mask) / total_pixels
            except Exception:
                pass

        has_filled_walls = (
            color_coverage >= Theme.CLASSIFIER_FILLED_COVERAGE_MIN
        )

        # --- Метрика 3: доля тонких CC ---
        thin_ratio = WallStyleClassifier._compute_thin_component_ratio(
            black_mask
        )
        has_thin_lines = (
            thin_ratio >= Theme.CLASSIFIER_THIN_COMPONENT_RATIO_MIN
        )

        # --- Метрика 4: параллельные пары ---
        has_parallel_pairs = WallStyleClassifier._check_parallel_pairs(gray)

        # --- Решение ---
        outline_score = sum([
            has_black_outlines,
            has_thin_lines,
            has_parallel_pairs,
        ])

        if has_filled_walls and outline_score >= 2:
            return "mixed"
        elif outline_score >= 2:
            return "outline"
        elif has_filled_walls:
            return "filled"
        elif has_black_outlines:
            return "outline"
        else:
            return "filled"

    @staticmethod
    def _compute_thin_component_ratio(black_mask: np.ndarray) -> float:
        """Доля CC с высоким aspect ratio (тонкие/линейные)."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            black_mask, connectivity=8
        )
        if num_labels <= 1:
            return 0.0

        thin_count = 0
        total_count = 0

        for i in range(1, num_labels):
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_area = stats[i, cv2.CC_STAT_AREA]

            if comp_area < 10:
                continue

            total_count += 1
            dim_max = max(comp_w, comp_h)
            dim_min = max(1, min(comp_w, comp_h))
            aspect = dim_max / dim_min

            if aspect >= 3.0:
                thin_count += 1

        return thin_count / max(1, total_count)

    @staticmethod
    def _check_parallel_pairs(
        gray: np.ndarray,
        sample_lines: int = 200,
    ) -> bool:
        """Быстрая проверка наличия параллельных пар линий."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=30,
            maxLineGap=10,
        )
        if lines is None or len(lines) < 4:
            return False

        if len(lines) > sample_lines:
            indices = np.random.choice(
                len(lines), sample_lines, replace=False
            )
            lines = lines[indices]

        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
            if angle <= 10.0 or angle >= 170.0:
                h_lines.append((x1, y1, x2, y2))
            elif abs(angle - 90.0) <= 10.0:
                v_lines.append((x1, y1, x2, y2))

        pair_count = 0
        dist_min = Theme.PARALLEL_DISTANCE_MIN_PX
        dist_max = Theme.PARALLEL_DISTANCE_MAX_PX

        for i in range(len(h_lines)):
            for j in range(i + 1, min(i + 20, len(h_lines))):
                mid_y_a = (h_lines[i][1] + h_lines[i][3]) / 2.0
                mid_y_b = (h_lines[j][1] + h_lines[j][3]) / 2.0
                dist = abs(mid_y_a - mid_y_b)
                if dist_min <= dist <= dist_max:
                    xa_min = min(h_lines[i][0], h_lines[i][2])
                    xa_max = max(h_lines[i][0], h_lines[i][2])
                    xb_min = min(h_lines[j][0], h_lines[j][2])
                    xb_max = max(h_lines[j][0], h_lines[j][2])
                    overlap = max(
                        0, min(xa_max, xb_max) - max(xa_min, xb_min)
                    )
                    span = max(1, min(xa_max - xa_min, xb_max - xb_min))
                    if overlap / span > 0.4:
                        pair_count += 1

        for i in range(len(v_lines)):
            for j in range(i + 1, min(i + 20, len(v_lines))):
                mid_x_a = (v_lines[i][0] + v_lines[i][2]) / 2.0
                mid_x_b = (v_lines[j][0] + v_lines[j][2]) / 2.0
                dist = abs(mid_x_a - mid_x_b)
                if dist_min <= dist <= dist_max:
                    ya_min = min(v_lines[i][1], v_lines[i][3])
                    ya_max = max(v_lines[i][1], v_lines[i][3])
                    yb_min = min(v_lines[j][1], v_lines[j][3])
                    yb_max = max(v_lines[j][1], v_lines[j][3])
                    overlap = max(
                        0, min(ya_max, yb_max) - max(ya_min, yb_min)
                    )
                    span = max(1, min(ya_max - ya_min, yb_max - yb_min))
                    if overlap / span > 0.4:
                        pair_count += 1

        return pair_count >= 3


# ============================================================
# 6. ДЕТЕКТОР КОНТУРНЫХ СТЕН БТИ (v5.0)
#    ← После ColorBasedWallDetector (наследует)
# ============================================================

class OutlineWallDetector(ColorBasedWallDetector):
    """Детектор стен для планировок в стиле БТИ.

    Стены = чёрные контуры прямоугольников, «склеенные» в единый контур.

    Пайплайн:
      1. _extract_black_mask      — извлечение чёрных пикселей
      2. _remove_text_components   — удаление текста/цифр по CC-геометрии
      3. _remove_standalone_lines  — удаление размерных линий по связности
      4. _fill_outline_walls       — заполнение между параллельными линиями
      5. Финальная очистка + стандартная прямоугольная декомпозиция
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
        # --- Параметры контурной детекции (v5.0) ---
        black_threshold: Optional[int] = None,
        use_adaptive: Optional[bool] = None,
        wall_thickness_range_px: Tuple[int, int] = (4, 30),
        enable_text_filter: bool = True,
        enable_dimline_filter: bool = True,
        enable_ocr_assist: bool = True,
    ) -> None:
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

        self.black_threshold = (
            black_threshold
            if black_threshold is not None
            else Theme.OUTLINE_BLACK_THRESH
        )
        self.use_adaptive = (
            use_adaptive
            if use_adaptive is not None
            else Theme.OUTLINE_USE_ADAPTIVE
        )
        self.wall_thickness_min = wall_thickness_range_px[0]
        self.wall_thickness_max = wall_thickness_range_px[1]
        self.enable_text_filter = enable_text_filter
        self.enable_dimline_filter = enable_dimline_filter
        self.enable_ocr_assist = enable_ocr_assist

    # ---- ШАГ 1: ИЗВЛЕЧЕНИЕ ЧЁРНЫХ ПИКСЕЛЕЙ ----

    def _extract_black_mask(self, img: np.ndarray) -> np.ndarray:
        """Двойной порог (глобальный AND адаптивный) для чёрных линий."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, global_mask = cv2.threshold(
            gray,
            self.black_threshold,
            255,
            cv2.THRESH_BINARY_INV,
        )

        if not self.use_adaptive:
            return global_mask

        adaptive_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            Theme.OUTLINE_ADAPTIVE_BLOCK,
            Theme.OUTLINE_ADAPTIVE_C,
        )

        combined = cv2.bitwise_and(global_mask, adaptive_mask)
        return combined

    # ---- ШАГ 2: УДАЛЕНИЕ ТЕКСТА / ЦИФР ----

    def _remove_text_components(
        self,
        mask: np.ndarray,
        img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Двухуровневая фильтрация: CC-геометрия + OCR bbox'ы."""
        cleaned = mask.copy()
        h_img, w_img = mask.shape[:2]
        total_pixels = h_img * w_img

        max_text_area = int(Theme.TEXT_FILTER_MAX_AREA_RATIO * total_pixels)
        min_text_h = int(Theme.TEXT_FILTER_MIN_HEIGHT_RATIO * h_img)
        max_text_h = int(Theme.TEXT_FILTER_MAX_HEIGHT_RATIO * h_img)

        # --- OCR bbox'ы ---
        ocr_bboxes: List[Tuple[int, int, int, int]] = []
        if (
            self.enable_ocr_assist
            and img is not None
            and self.ocr_model is not None
        ):
            try:
                ocr_bboxes = self.get_ocr_bboxes(
                    img,
                    bbox_expansion_px=Theme.OCR_BBOX_EXPANSION_DEFAULT + 2,
                )
            except Exception:
                pass

        if ocr_bboxes:
            ocr_erase_mask = np.zeros_like(mask, dtype=np.uint8)
            for x_min, y_min, x_max, y_max in ocr_bboxes:
                cv2.rectangle(
                    ocr_erase_mask,
                    (x_min, y_min), (x_max, y_max),
                    255, -1,
                )
            cleaned[ocr_erase_mask > 0] = 0

        # --- CC-геометрия ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8,
        )

        text_candidates: List[int] = []
        candidate_centroids: List[Tuple[float, float]] = []

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]

            if comp_area > max_text_area:
                continue

            char_height = comp_h
            if char_height < min_text_h or char_height > max_text_h:
                continue

            dim_max = max(comp_w, comp_h)
            dim_min = max(1, min(comp_w, comp_h))
            aspect = dim_max / dim_min
            if aspect > Theme.TEXT_FILTER_MAX_ASPECT:
                continue

            bbox_area = max(1, comp_w * comp_h)
            fill = comp_area / bbox_area
            if fill < Theme.TEXT_FILTER_MIN_FILL or fill > Theme.TEXT_FILTER_MAX_FILL:
                continue

            text_candidates.append(i)
            cx, cy = centroids[i]
            candidate_centroids.append((cx, cy))

        # Кластеризация
        gap = Theme.TEXT_FILTER_CLUSTER_GAP_PX
        confirmed_text: set = set()

        for idx_a in range(len(text_candidates)):
            for idx_b in range(idx_a + 1, len(text_candidates)):
                cx_a, cy_a = candidate_centroids[idx_a]
                cx_b, cy_b = candidate_centroids[idx_b]
                dist = np.hypot(cx_b - cx_a, cy_b - cy_a)
                if dist <= gap:
                    confirmed_text.add(text_candidates[idx_a])
                    confirmed_text.add(text_candidates[idx_b])

        # Одиночные в OCR bbox → тоже текст
        if ocr_bboxes:
            for idx_a in range(len(text_candidates)):
                cc_idx = text_candidates[idx_a]
                if cc_idx in confirmed_text:
                    continue
                cx, cy = candidate_centroids[idx_a]
                for bx1, by1, bx2, by2 in ocr_bboxes:
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        confirmed_text.add(cc_idx)
                        break

        for cc_idx in confirmed_text:
            cleaned[labels == cc_idx] = 0

        return cleaned

    # ---- ШАГ 3: УДАЛЕНИЕ РАЗМЕРНЫХ / ВЫНОСНЫХ ЛИНИЙ ----

    def _remove_standalone_lines(self, mask: np.ndarray) -> np.ndarray:
        """Удаляет тонкие изолированные линии через connectivity к основной CC."""
        cleaned = mask.copy()
        h_img, w_img = mask.shape[:2]
        total_pixels = h_img * w_img

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8,
        )

        if num_labels <= 2:
            return cleaned

        # Крупнейшая CC = основная стеновая структура
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        main_cc_label = np.argmax(areas) + 1

        # Дилатируем для проверки связности
        main_mask = (labels == main_cc_label).astype(np.uint8) * 255
        radius = Theme.DIMLINE_CONNECTIVITY_RADIUS_PX
        dilate_kernel = np.ones(
            (2 * radius + 1, 2 * radius + 1), np.uint8
        )
        main_dilated = cv2.dilate(main_mask, dilate_kernel, iterations=1)

        # Итеративно растим множество «связанных» CC
        connected_labels: set = {main_cc_label}
        changed = True
        while changed:
            changed = False
            for i in range(1, num_labels):
                if i in connected_labels:
                    continue
                comp_mask = (labels == i)
                overlap = np.count_nonzero(comp_mask & (main_dilated > 0))
                if overlap > 0:
                    connected_labels.add(i)
                    main_dilated = cv2.dilate(
                        (
                            np.isin(labels, list(connected_labels))
                        ).astype(np.uint8) * 255,
                        dilate_kernel,
                        iterations=1,
                    )
                    changed = True

        max_thin_dim = Theme.DIMLINE_MAX_THIN_DIM_PX
        min_dimline_aspect = Theme.DIMLINE_MIN_ASPECT
        max_dimline_area = Theme.DIMLINE_MAX_AREA_RATIO * total_pixels

        for i in range(1, num_labels):
            if i in connected_labels:
                continue

            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            dim_max = max(comp_w, comp_h)
            dim_min = max(1, min(comp_w, comp_h))
            aspect = dim_max / dim_min

            is_thin_line = (
                dim_min <= max_thin_dim
                and aspect >= min_dimline_aspect
                and comp_area <= max_dimline_area
            )

            is_small_isolated = comp_area < max_dimline_area * 0.5

            if is_thin_line or is_small_isolated:
                cleaned[labels == i] = 0

        return cleaned

    # ---- ШАГ 4: ЗАПОЛНЕНИЕ КОНТУРНЫХ СТЕН ----

    def _fill_outline_walls(self, mask: np.ndarray) -> np.ndarray:
        """Направленное морфологическое закрытие + AND для заполнения стен."""
        kl = Theme.OUTLINE_FILL_DIRECTIONAL_KERNEL_LENGTH
        kw = Theme.OUTLINE_FILL_DIRECTIONAL_KERNEL_WIDTH
        iterations = Theme.OUTLINE_FILL_CLOSE_ITERATIONS

        # Горизонтальное ядро
        kernel_h = np.zeros((kw, kl), dtype=np.uint8)
        kernel_h[kw // 2, :] = 1

        # Вертикальное ядро
        kernel_v = np.zeros((kl, kw), dtype=np.uint8)
        kernel_v[:, kw // 2] = 1

        closed_h = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel_h, iterations=iterations
        )
        closed_v = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel_v, iterations=iterations
        )

        # AND: только если оба направления подтвердили
        filled = cv2.bitwise_and(closed_h, closed_v)

        # Вернуть исходные пиксели
        filled = cv2.bitwise_or(filled, mask)

        # Изотропное закрытие
        iso_k = Theme.OUTLINE_FILL_CLOSE_KERNEL
        kernel_iso = np.ones((iso_k, iso_k), np.uint8)
        filled = cv2.morphologyEx(
            filled, cv2.MORPH_CLOSE, kernel_iso, iterations=1
        )

        # Удаление случайно заполненных комнат
        filled = self._remove_overfilled_regions(filled, mask)

        # Финальное открытие
        open_k = Theme.OUTLINE_FINAL_OPEN_KERNEL
        kernel_open = np.ones((open_k, open_k), np.uint8)
        filled = cv2.morphologyEx(
            filled, cv2.MORPH_OPEN, kernel_open,
            iterations=Theme.OUTLINE_FINAL_OPEN_ITERATIONS,
        )

        return filled

    def _remove_overfilled_regions(
        self,
        filled_mask: np.ndarray,
        original_outline_mask: np.ndarray,
    ) -> np.ndarray:
        """Удаляет области, случайно заполненные при закрытии (комнаты)."""
        cleaned = filled_mask.copy()
        total_pixels = filled_mask.size

        new_pixels = cv2.bitwise_and(
            filled_mask,
            cv2.bitwise_not(original_outline_mask),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            new_pixels, connectivity=8,
        )

        room_hole_min = Theme.OUTLINE_ROOM_HOLE_MIN_RATIO * total_pixels

        for i in range(1, num_labels):
            comp_area = stats[i, cv2.CC_STAT_AREA]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]

            if comp_area > room_hole_min:
                bbox_area = max(1, comp_w * comp_h)
                fill_ratio = comp_area / bbox_area
                if fill_ratio > 0.4:
                    cleaned[labels == i] = 0

        return cleaned

    # ---- ПОЛНЫЙ ПАЙПЛАЙН ----

    def detect_outline_walls(self, img: np.ndarray) -> np.ndarray:
        """Полный пайплайн детекции контурных стен."""
        # 1. Извлечение чёрных пикселей
        black_mask = self._extract_black_mask(img)

        # 2. Лёгкая очистка шума
        kernel_small = np.ones((2, 2), np.uint8)
        black_mask = cv2.morphologyEx(
            black_mask, cv2.MORPH_OPEN, kernel_small, iterations=1
        )

        # 3. Удаление текста / цифр
        if self.enable_text_filter:
            black_mask = self._remove_text_components(black_mask, img)

        # 4. Удаление размерных линий
        if self.enable_dimline_filter:
            black_mask = self._remove_standalone_lines(black_mask)

        # 5. Заполнение контурных стен
        wall_mask = self._fill_outline_walls(black_mask)

        # 6. Удаление мелких CC
        wall_mask = self._remove_small_filled_components(wall_mask)

        return wall_mask

    def _remove_small_filled_components(
        self, mask: np.ndarray,
    ) -> np.ndarray:
        total_pixels = mask.size
        min_area = int(
            Theme.OUTLINE_MIN_FILLED_COMPONENT_RATIO * total_pixels
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        cleaned = np.zeros_like(mask, dtype=np.uint8)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255

        return cleaned

    # ---- process() для контурного режима ----

    def process(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
        img_no_text = self.remove_text_labels_ocr(img)
        wall_mask = self.detect_outline_walls(img_no_text)
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
                union_mask, cv2.MORPH_CLOSE, kernel, iterations=1,
            )

        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outline_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(outline_mask, contours, -1, 255, 1)

        return wall_mask, rectangles, outline_mask


# ============================================================
# 7. РАСШИРЕННЫЙ ДЕТЕКТОР СТЕН (v5.0)
#    ← После всех зависимостей: ColorBasedWallDetector,
#      WallStyleClassifier, OutlineWallDetector
# ============================================================

class VersatileWallDetector(ColorBasedWallDetector):
    """Универсальный детектор стен v5.0.

    Режимы (wall_detection_mode):
      "color"     — только цветовая фильтрация.
      "structure" — структурная детекция (линии, контуры, пороги).
      "outline"   — контурная детекция для БТИ-планировок (v5.0).
      "hybrid"    — сначала цвет; при низком покрытии → структура.
      "auto"      — автоклассификация → выбор оптимального метода.
    """

    def __init__(
        self,
        # --- Базовый класс ---
        wall_color_hex: str = "8E8780",
        color_tolerance: int = 3,
        auto_wall_color: bool = False,
        min_rect_area_ratio: float = 0.0002,
        rect_visual_gap_px: int = Theme.RECT_VISUAL_GAP_DEFAULT,
        ocr_model: Optional[object] = None,
        edge_expansion_tolerance: int = 8,
        max_edge_expansion_px: int = 10,
        # --- Структурная детекция (v4.0) ---
        wall_detection_mode: str = "auto",
        structure_method: str = "lines",
        line_thickness_estimate_px: int = 8,
        min_wall_length_ratio: float = 0.02,
        use_adaptive_threshold: bool = True,
        # --- Фильтрация мебели (v4.1) ---
        enable_furniture_filter: bool = True,
        enable_internal_object_removal: bool = True,
        enable_parallel_wall_detection: bool = False,
        enable_fallback_heuristic: bool = True,
        # --- Контурная детекция (v5.0) ---
        outline_black_threshold: Optional[int] = None,
        outline_use_adaptive: Optional[bool] = None,
        outline_wall_thickness_range_px: Tuple[int, int] = (4, 30),
        outline_enable_text_filter: bool = True,
        outline_enable_dimline_filter: bool = True,
        outline_enable_ocr_assist: bool = True,
    ) -> None:
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

        valid_modes = ("color", "structure", "outline", "hybrid", "auto")
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

        # v4.1
        self.enable_furniture_filter = bool(enable_furniture_filter)
        self.enable_internal_object_removal = bool(enable_internal_object_removal)
        self.enable_parallel_wall_detection = bool(enable_parallel_wall_detection)
        self.enable_fallback_heuristic = bool(enable_fallback_heuristic)

        # v5.0: внутренний OutlineWallDetector (переиспользует OCR модель)
        self._outline_detector = OutlineWallDetector(
            wall_color_hex=wall_color_hex,
            color_tolerance=color_tolerance,
            auto_wall_color=auto_wall_color,
            min_rect_area_ratio=min_rect_area_ratio,
            rect_visual_gap_px=rect_visual_gap_px,
            ocr_model=self.ocr_model,
            edge_expansion_tolerance=edge_expansion_tolerance,
            max_edge_expansion_px=max_edge_expansion_px,
            black_threshold=outline_black_threshold,
            use_adaptive=outline_use_adaptive,
            wall_thickness_range_px=outline_wall_thickness_range_px,
            enable_text_filter=outline_enable_text_filter,
            enable_dimline_filter=outline_enable_dimline_filter,
            enable_ocr_assist=outline_enable_ocr_assist,
        )

        # Кэш последней классификации
        self._last_classified_style: Optional[str] = None

    @property
    def last_classified_style(self) -> Optional[str]:
        """Тип стен, определённый при последнем вызове process()."""
        return self._last_classified_style

    # ---- СОЗДАНИЕ МАСКИ ----

    def create_wall_mask(self, img: np.ndarray) -> np.ndarray:
        if self.wall_detection_mode == "color":
            self._last_classified_style = "filled"
            return self.filter_walls_by_color(img)

        elif self.wall_detection_mode == "outline":
            self._last_classified_style = "outline"
            return self._outline_detector.detect_outline_walls(img)

        elif self.wall_detection_mode == "structure":
            self._last_classified_style = "structure"
            return self._structure_with_furniture_filter(img)

        elif self.wall_detection_mode == "hybrid":
            color_mask = self.filter_walls_by_color(img)
            coverage = np.count_nonzero(color_mask) / color_mask.size
            if coverage > Theme.HYBRID_COLOR_COVERAGE_THRESHOLD:
                self._last_classified_style = "filled"
                return color_mask
            self._last_classified_style = "structure"
            return self._structure_with_furniture_filter(img)

        else:  # "auto"
            return self._auto_select_detection_v5(img)

    # ---- АВТО-ВЫБОР v5.0 ----

    def _auto_select_detection_v5(self, img: np.ndarray) -> np.ndarray:
        """Классификация → диспетчеризация → фоллбэк."""
        style = WallStyleClassifier.classify(img, color_detector=self)
        self._last_classified_style = style

        if style == "filled":
            color_mask = self.filter_walls_by_color(img)
            color_coverage = np.count_nonzero(color_mask) / color_mask.size

            if color_coverage >= Theme.AUTO_COLOR_COVERAGE_THRESHOLD:
                return color_mask

            # Фоллбэк → outline
            self._last_classified_style = "outline_fallback"
            outline_mask = self._outline_detector.detect_outline_walls(img)
            outline_coverage = np.count_nonzero(outline_mask) / outline_mask.size

            if outline_coverage > color_coverage:
                return outline_mask
            return color_mask

        elif style == "outline":
            outline_mask = self._outline_detector.detect_outline_walls(img)
            outline_coverage = np.count_nonzero(outline_mask) / outline_mask.size

            if outline_coverage >= Theme.OUTLINE_MIN_WALL_AREA_RATIO * 10:
                return outline_mask

            # Фоллбэк → structure
            self._last_classified_style = "structure_fallback"
            struct_mask = self._structure_with_furniture_filter(img)
            struct_coverage = np.count_nonzero(struct_mask) / struct_mask.size

            if struct_coverage > outline_coverage:
                return struct_mask
            return outline_mask

        else:  # "mixed"
            color_mask = self.filter_walls_by_color(img)
            outline_mask = self._outline_detector.detect_outline_walls(img)

            fused = cv2.bitwise_or(color_mask, outline_mask)
            k = Theme.AUTO_FUSION_CLOSE_KERNEL
            kernel = np.ones((k, k), np.uint8)
            fused = cv2.morphologyEx(
                fused, cv2.MORPH_CLOSE, kernel, iterations=1
            )
            return fused

    # ---- СТРУКТУРА + ФИЛЬТРАЦИЯ МЕБЕЛИ (v4.1) ----

    def _structure_with_furniture_filter(
        self, img: np.ndarray,
    ) -> np.ndarray:
        mask = self.filter_walls_by_structure(img)

        if self.enable_fallback_heuristic and self._should_fallback_to_color(mask):
            color_mask = self.filter_walls_by_color(img)
            color_mask = morphological_cleanup(color_mask)
            if np.count_nonzero(color_mask) > 0:
                return color_mask

        if self.enable_furniture_filter:
            mask = self._filter_components_by_shape(mask)

        if self.enable_internal_object_removal:
            mask = self._remove_internal_objects(mask)

        if self.enable_parallel_wall_detection:
            parallel_mask = self._detect_parallel_wall_pairs(img)
            if parallel_mask is not None:
                mask = cv2.bitwise_or(mask, parallel_mask)

        return mask

    # ---- СТРУКТУРНАЯ ДЕТЕКЦИЯ (v4.0) ----

    def filter_walls_by_structure(self, img: np.ndarray) -> np.ndarray:
        if self.ocr_model is not None:
            img_clean = self.remove_text_labels_ocr(img)
        else:
            img_clean = img

        gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

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

        binary = morphological_cleanup(binary)

        if self.structure_method == "lines":
            binary = self._detect_lines_lsd_or_hough(gray, binary)
        elif self.structure_method == "edges":
            binary = self._detect_edges_canny(gray, binary)

        binary = morphological_cleanup(binary)
        return binary

    def _detect_lines_lsd_or_hough(
        self,
        gray: np.ndarray,
        binary: np.ndarray,
    ) -> np.ndarray:
        thickness = self.line_thickness_estimate_px
        h_img, w_img = gray.shape[:2]
        diagonal = np.hypot(w_img, h_img)

        min_length_fixed = Theme.STRUCT_MIN_LINE_LENGTH
        min_length_ratio = Theme.STRUCT_MIN_LINE_RATIO * diagonal
        min_length = max(min_length_fixed, min_length_ratio)

        line_mask = np.zeros_like(gray, dtype=np.uint8)
        raw_lines: List[Tuple[int, int, int, int]] = []
        lines_found = False

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

        if self.enable_furniture_filter:
            filtered_lines = self._filter_lines_by_geometry(
                raw_lines, min_length, diagonal
            )
        else:
            filtered_lines = [
                seg for seg in raw_lines
                if np.hypot(seg[2] - seg[0], seg[3] - seg[1]) >= min_length
            ]

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

    # ---- ФИЛЬТРАЦИЯ ЛИНИЙ ПО ГЕОМЕТРИИ (v4.1) ----

    @staticmethod
    def _filter_lines_by_geometry(
        lines: List[Tuple[int, int, int, int]],
        min_length: float,
        diagonal: float,
    ) -> List[Tuple[int, int, int, int]]:
        angle_tol = Theme.ANGLE_TOLERANCE_DEG
        filtered: List[Tuple[int, int, int, int]] = []

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if length < min_length:
                continue

            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
            angle_deg = angle_deg % 180.0

            is_horizontal = (
                angle_deg <= angle_tol
                or angle_deg >= (180.0 - angle_tol)
            )
            is_vertical = abs(angle_deg - 90.0) <= angle_tol

            if is_horizontal or is_vertical:
                filtered.append((x1, y1, x2, y2))

        return filtered

    # ---- ФИЛЬТРАЦИЯ КОМПОНЕНТ (v4.1) ----

    @staticmethod
    def _filter_components_by_shape(mask: np.ndarray) -> np.ndarray:
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

            is_elongated = aspect_ratio >= Theme.FURNITURE_MIN_ASPECT_RATIO
            is_large = area_ratio >= Theme.MAX_FURNITURE_AREA_RATIO
            is_sparse = fill_ratio < Theme.FURNITURE_MAX_FILL_RATIO

            if is_elongated or is_large or is_sparse:
                cleaned[labels == i] = 255

        return cleaned

    # ---- УДАЛЕНИЕ ВНУТРЕННИХ ОБЪЕКТОВ (v4.1) ----

    @staticmethod
    def _remove_internal_objects(wall_mask: np.ndarray) -> np.ndarray:
        cleaned = wall_mask.copy()
        total_pixels = wall_mask.size

        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
        )

        if hierarchy is None or len(contours) == 0:
            return cleaned

        hierarchy = hierarchy[0]
        min_room_area = Theme.ROOM_FLOOD_MIN_AREA_RATIO * total_pixels

        for idx in range(len(contours)):
            parent_idx = hierarchy[idx][3]
            if parent_idx < 0:
                continue

            contour = contours[idx]
            area = cv2.contourArea(contour)

            if area >= min_room_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            dim_max = max(w, h)
            dim_min = max(1, min(w, h))
            aspect = dim_max / dim_min

            if aspect < Theme.FURNITURE_MIN_ASPECT_RATIO:
                cv2.drawContours(
                    cleaned, [contour], -1, 0, thickness=cv2.FILLED,
                )

        erode_size = Theme.ROOM_FLOOD_ERODE_SIZE
        if erode_size > 0:
            kernel = np.ones((erode_size, erode_size), np.uint8)
            cleaned = cv2.morphologyEx(
                cleaned, cv2.MORPH_OPEN, kernel, iterations=1,
            )

        return cleaned

    # ---- ФИЛЬТРАЦИЯ ПРЯМОУГОЛЬНИКОВ (v4.1) ----

    def _filter_furniture_rects(
        self,
        rectangles: List[WallRectangle],
        img_shape: Tuple[int, int],
    ) -> List[WallRectangle]:
        if not rectangles:
            return rectangles

        img_h, img_w = img_shape
        image_area = img_h * img_w
        min_area = Theme.RECT_MIN_AREA_RATIO * image_area
        min_aspect = Theme.RECT_MIN_ASPECT_RATIO
        margin = Theme.RECT_CONNECTIVITY_MARGIN_PX

        def touches_border(r: WallRectangle) -> bool:
            return (
                r.x1 <= margin
                or r.y1 <= margin
                or r.x2 >= img_w - 1 - margin
                or r.y2 >= img_h - 1 - margin
            )

        def rects_connected(a: WallRectangle, b: WallRectangle) -> bool:
            gap_x = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
            gap_y = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))
            return gap_x <= margin and gap_y <= margin

        candidates: List[Tuple[int, WallRectangle, bool]] = []
        for idx, rect in enumerate(rectangles):
            dim_max = max(rect.width, rect.height)
            dim_min = max(1, min(rect.width, rect.height))
            aspect = dim_max / dim_min

            auto_keep = rect.area >= min_area and aspect >= min_aspect
            force_keep = rect.area >= min_area * 5
            candidates.append((idx, rect, auto_keep or force_keep))

        kept_rects: List[WallRectangle] = []
        for idx, rect, auto_kept in candidates:
            if auto_kept:
                kept_rects.append(rect)
                continue
            if touches_border(rect):
                kept_rects.append(rect)
                continue
            connected_to_good = any(
                rects_connected(rect, other_rect)
                for other_idx, other_rect, other_kept in candidates
                if other_kept and other_idx != idx
            )
            if connected_to_good:
                kept_rects.append(rect)

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

    # ---- ПАРАЛЛЕЛЬНЫЕ ПАРЫ (v4.1) ----

    def _detect_parallel_wall_pairs(
        self, img: np.ndarray,
    ) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape[:2]
        diagonal = np.hypot(w_img, h_img)
        min_length = max(
            Theme.STRUCT_MIN_LINE_LENGTH,
            Theme.STRUCT_MIN_LINE_RATIO * diagonal,
        )

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

        filtered = self._filter_lines_by_geometry(
            raw_lines, min_length, diagonal
        )

        if len(filtered) < 2:
            return None

        horizontal: List[Tuple[int, int, int, int, float]] = []
        vertical: List[Tuple[int, int, int, int, float]] = []
        angle_tol = Theme.ANGLE_TOLERANCE_DEG

        for x1, y1, x2, y2 in filtered:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            if angle <= angle_tol or angle >= (180.0 - angle_tol):
                horizontal.append((x1, y1, x2, y2, angle))
            elif abs(angle - 90.0) <= angle_tol:
                vertical.append((x1, y1, x2, y2, angle))

        parallel_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        found_any = False

        dist_min = Theme.PARALLEL_DISTANCE_MIN_PX
        dist_max = Theme.PARALLEL_DISTANCE_MAX_PX
        par_angle_tol = Theme.PARALLEL_ANGLE_TOLERANCE_DEG

        for group in [horizontal, vertical]:
            if len(group) < 2:
                continue
            is_horiz = (group is horizontal)

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    x1a, y1a, x2a, y2a, ang_a = group[i]
                    x1b, y1b, x2b, y2b, ang_b = group[j]

                    angle_diff = abs(ang_a - ang_b)
                    angle_diff = min(angle_diff, 180.0 - angle_diff)
                    if angle_diff > par_angle_tol:
                        continue

                    if is_horiz:
                        mid_y_a = (y1a + y2a) / 2.0
                        mid_y_b = (y1b + y2b) / 2.0
                        dist = abs(mid_y_a - mid_y_b)
                        xa_min, xa_max = min(x1a, x2a), max(x1a, x2a)
                        xb_min, xb_max = min(x1b, x2b), max(x1b, x2b)
                        overlap = max(
                            0, min(xa_max, xb_max) - max(xa_min, xb_min)
                        )
                        span = max(1, min(xa_max - xa_min, xb_max - xb_min))
                    else:
                        mid_x_a = (x1a + x2a) / 2.0
                        mid_x_b = (x1b + x2b) / 2.0
                        dist = abs(mid_x_a - mid_x_b)
                        ya_min, ya_max = min(y1a, y2a), max(y1a, y2a)
                        yb_min, yb_max = min(y1b, y2b), max(y1b, y2b)
                        overlap = max(
                            0, min(ya_max, yb_max) - max(ya_min, yb_min)
                        )
                        span = max(1, min(ya_max - ya_min, yb_max - yb_min))

                    if dist < dist_min or dist > dist_max:
                        continue
                    if overlap / span < 0.5:
                        continue

                    pts = np.array([
                        [x1a, y1a], [x2a, y2a],
                        [x2b, y2b], [x1b, y1b],
                    ], dtype=np.int32)
                    cv2.fillConvexPoly(parallel_mask, pts, 255)
                    found_any = True

        return parallel_mask if found_any else None

    # ---- ФОЛЛБЭК (v4.1) ----

    @staticmethod
    def _should_fallback_to_color(mask: np.ndarray) -> bool:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        num_components = num_labels - 1

        if num_components > Theme.STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK:
            return True

        if num_components > 0:
            areas = [
                stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)
            ]
            median_area = float(np.median(areas))
            min_expected = Theme.STRUCT_MIN_WALL_AREA_RATIO * mask.size
            if median_area < 2.0 * min_expected:
                return True

        return False

    # ---- process() v5.0 ----

    def process(
        self,
        img: np.ndarray,
    ) -> Tuple[np.ndarray, List[WallRectangle], np.ndarray]:
        """Полный пайплайн v5.0 с авто-классификацией стен."""
        # 1. OCR-очистка
        img_no_text = self.remove_text_labels_ocr(img)

        # 2. Маска стен (автоклассификация внутри create_wall_mask)
        wall_mask = self.create_wall_mask(img_no_text)

        # 3. Круглые метки
        wall_mask = self.remove_circular_labels(wall_mask)

        # 4. Морфологическая очистка
        wall_mask = morphological_cleanup(wall_mask)

        # 5. Прямоугольная декомпозиция
        rectangles = self._decompose_to_rectangles(
            wall_mask, img=img_no_text
        )

        # 6. Фильтрация мебельных rect'ов (не для чистого «filled»)
        if (
            self.enable_furniture_filter
            and self._last_classified_style not in ("filled",)
        ):
            rectangles = self._filter_furniture_rects(
                rectangles, wall_mask.shape,
            )

        # 7. Объединённая форма
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
                union_mask, cv2.MORPH_CLOSE, kernel, iterations=1,
            )

        # 8. Внешние контуры
        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        outline_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(outline_mask, contours, -1, 255, 1)

        return wall_mask, rectangles, outline_mask