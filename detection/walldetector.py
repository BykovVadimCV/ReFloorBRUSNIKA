"""
Быков Вадим Олегович - 16.12.2025

Модуль детекции стен на основе цветовой фильтрации - v2.2

ПАЙПЛАЙН:
- OCR-очистка текстовых меток (docTR)
- Цветовая фильтрация стен с комбинированным RGB + HSV подходом
- Удаление наиболее выраженной круглой метки комнаты
- Морфологическая очистка и скелетизация
- Детекция структуры стен (Probabilistic Hough Line Transform)
- Distance Transform для измерения толщины
- Трассировка и фильтрация по толщине
- Детекция и анализ соединений стен

СТЕК:
- OpenCV: цветовая фильтрация, Hough Transform, distance transform
- docTR: OCR для удаления текстовых меток
- scikit-image: прецизионная скелетизация
- NumPy: векторные операции для измерений
- scipy: интерполяция и кластеризация

Последнее изменение - 22.12.25 - Silent operation + OCR text removal
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


# ------------------------------------------------------------------ #
# DATA STRUCTURES
# ------------------------------------------------------------------ #

@dataclass
class WallSegment:
    """Структура данных для сегмента стены"""
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    length: float
    thickness_mean: float
    thickness_std: float
    thickness_min: float
    thickness_max: float
    thickness_profile: List[float]
    sample_points: List[Tuple[int, int]]


@dataclass
class WallJunction:
    """Структура данных для соединения стен"""
    id: int
    x: int
    y: int
    connected_walls: List[int]
    junction_type: str
    local_thickness: float


# ------------------------------------------------------------------ #
# MORPHOLOGICAL OPERATIONS
# ------------------------------------------------------------------ #

def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Морфологическая очистка маски стен"""
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:
            cleaned[labels == i] = 255

    return cleaned


def skeletonize_walls(mask: np.ndarray) -> np.ndarray:
    """Скелетизация стен для последующей детекции линий"""
    skeleton = skeletonize(mask > 0).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

    return skeleton


# ------------------------------------------------------------------ #
# WALL DETECTOR
# ------------------------------------------------------------------ #

class ColorBasedWallDetector:
    """Детектор стен на основе цветовой фильтрации"""

    def __init__(self,
                 wall_color_hex: str = "8E8780",
                 color_tolerance: int = 3,
                 min_wall_length: int = 10,
                 hough_threshold: int = 10,
                 auto_wall_color: bool = False):
        """
        Инициализация детектора стен

        Параметры:
            wall_color_hex: цвет стен в формате HEX
            color_tolerance: толерантность цвета RGB
            min_wall_length: минимальная длина стены в пикселях
            hough_threshold: порог для Hough Line Transform
            auto_wall_color: автоматическое определение цвета стен
        """
        self.auto_wall_color = auto_wall_color

        if not self.auto_wall_color:
            self.wall_rgb = self._hex_to_rgb(wall_color_hex)
        else:
            self.wall_rgb = None

        self.color_tolerance = color_tolerance
        self.min_wall_length = min_wall_length
        self.hough_threshold = hough_threshold

        # Параметры фильтрации сегментов
        self.min_valid_thickness_ratio = 0.6
        self.max_rel_thickness_std = 0.45
        self.max_angle_deviation_deg = 12.0

        # OCR модель для удаления текстовых меток
        self.ocr_model = None
        self._init_ocr()

    # ------------------------------------------------------------------ #
    # OCR ИНИЦИАЛИЗАЦИЯ / ОЧИСТКА ТЕКСТА
    # ------------------------------------------------------------------ #

    def _init_ocr(self) -> None:
        """Инициализация модели OCR (docTR) для удаления текстовых меток"""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(pretrained=True)
        except Exception:
            # Без docTR работаем без OCR-очистки, в silent-режиме
            self.ocr_model = None

    def remove_text_labels_ocr(self,
                               img: np.ndarray,
                               inpaint_radius: int = 3,
                               bbox_expansion_px: int = 2) -> np.ndarray:
        """
        Удаление текстовых меток с чертежа с помощью OCR (docTR)

        Параметры:
            img: входное изображение (BGR)
            inpaint_radius: радиус для cv2.inpaint
            bbox_expansion_px: расширение прямоугольников вокруг текста в пикселях

        Возвращает:
            Изображение без текстовых надписей (после инпейнтинга)
        """
        if self.ocr_model is None:
            return img

        h, w = img.shape[:2]
        text_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            # docTR ожидает список изображений [H, W, C] (np.uint8)
            result = self.ocr_model([img])
        except Exception:
            return img

        # Обход всех распознанных слов и построение маски текста
        try:
            for page in getattr(result, "pages", []):
                for block in getattr(page, "blocks", []):
                    for line in getattr(block, "lines", []):
                        for word in getattr(line, "words", []):
                            geom = getattr(word, "geometry", None)
                            if geom is None:
                                continue

                            # geometry в docTR обычно: ((x_min, y_min), (x_max, y_max)) в относительных координатах
                            x_min_rel = y_min_rel = x_max_rel = y_max_rel = None
                            try:
                                # Попытка распаковки как ((x_min, y_min), (x_max, y_max))
                                (x_min_rel, y_min_rel), (x_max_rel, y_max_rel) = geom  # type: ignore[misc]
                            except Exception:
                                # Возможные альтернативные форматы экспорта
                                try:
                                    # [[x_min, y_min], [x_max, y_max]]
                                    if len(geom) == 2 and len(geom[0]) == 2:
                                        x_min_rel, y_min_rel = geom[0]
                                        x_max_rel, y_max_rel = geom[1]
                                    # [x_min, y_min, x_max, y_max]
                                    elif len(geom) == 4:
                                        x_min_rel, y_min_rel, x_max_rel, y_max_rel = geom
                                    else:
                                        continue
                                except Exception:
                                    continue

                            if x_min_rel is None or x_max_rel is None:
                                continue

                            # Перевод относительных координат (0..1) в пиксели
                            x_min = int(np.floor(float(x_min_rel) * w))
                            y_min = int(np.floor(float(y_min_rel) * h))
                            x_max = int(np.ceil(float(x_max_rel) * w))
                            y_max = int(np.ceil(float(y_max_rel) * h))

                            # Небольшое расширение bbox, чтобы захватить контур текста
                            x_min = max(0, x_min - bbox_expansion_px)
                            y_min = max(0, y_min - bbox_expansion_px)
                            x_max = min(w - 1, x_max + bbox_expansion_px)
                            y_max = min(h - 1, y_max + bbox_expansion_px)

                            if x_max <= x_min or y_max <= y_min:
                                continue

                            cv2.rectangle(text_mask, (x_min, y_min), (x_max, y_max), 255, -1)
        except Exception:
            # Любая ошибка OCR не должна ломать основной пайплайн
            return img

        if np.count_nonzero(text_mask) == 0:
            return img

        # Небольшое расширение маски, чтобы убрать ореолы вокруг текста
        if bbox_expansion_px > 1:
            k = max(1, bbox_expansion_px)
            kernel = np.ones((k, k), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=1)

        # Инпейтинг текста по маске
        try:
            cleaned = cv2.inpaint(img, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
        except Exception:
            # В случае проблем с inpaint возвращаем исходное изображение
            cleaned = img

        return cleaned

    # ------------------------------------------------------------------ #
    # ЦВЕТОВАЯ ДЕТЕКЦИЯ СТЕН
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Конвертация HEX -> RGB"""
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
        Автоматическое определение цвета стен

        Параметры:
            img: входное изображение
            min_fraction: минимальная доля пикселей для кластера
            k: количество кластеров для k-means
            max_samples: максимальное количество пикселей для анализа

        Возвращает:
            Кортеж (R, G, B) цвета стен
        """
        h, w = img.shape[:2]
        pixels = img.reshape(-1, 3).astype(np.float32)

        # Подвыборка для ускорения
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

        counts = np.bincount(labels, minlength=K).astype(np.float32)
        fractions = counts / float(labels.size)

        # Оценка яркости (L в Lab) для центров кластеров
        centers_bgr = centers.reshape(K, 1, 3)
        centers_lab = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2LAB).reshape(K, 3)
        L = centers_lab[:, 0]

        # Кластеры с достаточной долей пикселей
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
        Цветовая фильтрация стен с комбинированным RGB + HSV подходом

        Параметры:
            img: входное изображение (BGR)

        Возвращает:
            Бинарная маска стен
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

        # Допуски для HSV
        h_tol = 4
        s_tol = 15
        v_tol = 12

        h = hsv[:, :, 0].astype(np.int16)
        s = hsv[:, :, 1].astype(np.int16)
        v = hsv[:, :, 2].astype(np.int16)

        target_h = int(target_hsv[0])
        target_s = int(target_hsv[1])
        target_v = int(target_hsv[2])

        # Обработка цикличности Hue
        h_diff = np.abs(h - target_h)
        h_diff = np.minimum(h_diff, 180 - h_diff)
        h_match = h_diff <= h_tol

        s_match = np.abs(s - target_s) <= s_tol
        v_match = np.abs(v - target_v) <= v_tol
        hsv_mask = (h_match & s_match & v_match).astype(np.uint8) * 255

        # Пересечение масок RGB и HSV
        combined_mask = cv2.bitwise_and(rgb_mask, hsv_mask)

        # Замыкание мелких дыр
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return combined_mask

    # ------------------------------------------------------------------ #
    # СПЕЦИАЛЬНАЯ ОЧИСТКА КРУГЛЫХ МЕТОК
    # ------------------------------------------------------------------ #

    def remove_circular_labels(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        Удаление наиболее выраженной круглой метки комнаты

        Параметры:
            wall_mask: бинарная маска стен

        Возвращает:
            Очищенная маска стен
        """
        cleaned = wall_mask.copy()

        # Лёгкое размытие для устойчивого поиска окружностей
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

        # HoughCircles возвращает круги по убыванию accumulator value —
        # используем только первый (самый сильный) круг
        if circles is not None:
            circles = np.around(circles[0]).astype(np.uint16)
            if len(circles) > 0:
                cx, cy, r = circles[0]
                cv2.circle(cleaned, (cx, cy), r + 5, 0, -1)

        return cleaned

    # ------------------------------------------------------------------ #
    # ДЕТЕКЦИЯ ЛИНИЙ, ТОЛЩИНА, СОЕДИНЕНИЯ
    # ------------------------------------------------------------------ #

    def detect_wall_lines(self, skeleton: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детекция линий стен через Probabilistic Hough Transform

        Параметры:
            skeleton: скелетизированная маска стен

        Возвращает:
            Список линий в формате (x1, y1, x2, y2)
        """
        lines = cv2.HoughLinesP(
            skeleton,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_wall_length,
            maxLineGap=10
        )

        if lines is None:
            return []

        wall_lines: List[Tuple[int, int, int, int]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= self.min_wall_length:
                wall_lines.append((int(x1), int(y1), int(x2), int(y2)))

        return wall_lines

    def compute_distance_transform(self, wall_mask: np.ndarray) -> np.ndarray:
        """
        Вычисление Distance Transform для измерения толщины стен

        Параметры:
            wall_mask: бинарная маска стен

        Возвращает:
            Distance transform изображение
        """
        dist_transform = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 5)
        return dist_transform

    def measure_thickness_along_line(self,
                                     x1: int, y1: int, x2: int, y2: int,
                                     dist_transform: np.ndarray,
                                     num_samples: int = 50) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        Измерение толщины стены вдоль линии

        Параметры:
            x1, y1, x2, y2: координаты линии
            dist_transform: distance transform изображение
            num_samples: количество точек для измерения

        Возвращает:
            Кортеж (профиль толщины, точки измерения)
        """
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 1:
            return [0.0], [(x1, y1)]

        t_values = np.linspace(0.0, 1.0, num_samples)

        sample_points: List[Tuple[int, int]] = []
        thickness_values: List[float] = []

        h, w = dist_transform.shape

        for t in t_values:
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= x < w and 0 <= y < h:
                sample_points.append((x, y))
                thickness = float(dist_transform[y, x] * 2.0)
                thickness_values.append(thickness)

        if not thickness_values:
            return [0.0], [(x1, y1)]

        return thickness_values, sample_points

    def detect_junctions(self,
                         wall_lines: List[Tuple[int, int, int, int]],
                         dist_transform: np.ndarray,
                         proximity_threshold: int = 15) -> List[WallJunction]:
        """
        Детекция соединений стен (L-образные, T-образные, X-образные)

        Параметры:
            wall_lines: список линий стен
            dist_transform: distance transform изображение
            proximity_threshold: порог близости для группировки концов

        Возвращает:
            Список соединений стен
        """
        endpoints = []
        for idx, (x1, y1, x2, y2) in enumerate(wall_lines):
            endpoints.append((x1, y1, idx, 'start'))
            endpoints.append((x2, y2, idx, 'end'))

        if not endpoints:
            return []

        coords = np.array([(x, y) for x, y, _, _ in endpoints])
        distances = cdist(coords, coords, metric='euclidean')

        junctions: List[WallJunction] = []
        processed = set()
        junction_id = 0

        for i in range(len(endpoints)):
            if i in processed:
                continue

            nearby = np.where(distances[i] < proximity_threshold)[0]

            if len(nearby) >= 3:
                connected_walls = list(set([endpoints[j][2] for j in nearby]))

                if len(connected_walls) >= 2:
                    x_mean = int(np.mean([endpoints[j][0] for j in nearby]))
                    y_mean = int(np.mean([endpoints[j][1] for j in nearby]))

                    h, w = dist_transform.shape
                    if 0 <= x_mean < w and 0 <= y_mean < h:
                        local_thick = float(dist_transform[y_mean, x_mean] * 2.0)
                    else:
                        local_thick = 0.0

                    if len(connected_walls) == 2:
                        junction_type = "L-junction"
                    elif len(connected_walls) == 3:
                        junction_type = "T-junction"
                    else:
                        junction_type = "X-junction"

                    junctions.append(WallJunction(
                        id=junction_id,
                        x=x_mean,
                        y=y_mean,
                        connected_walls=connected_walls,
                        junction_type=junction_type,
                        local_thickness=local_thick
                    ))

                    processed.update(nearby)
                    junction_id += 1

        return junctions

    # ------------------------------------------------------------------ #
    # "УМНАЯ" ФИЛЬТРАЦИЯ СЕГМЕНТОВ СТЕН
    # ------------------------------------------------------------------ #

    def _segment_pass_smart_filters(self,
                                    seg: WallSegment,
                                    global_median_thickness: Optional[float]) -> bool:
        """
        Фильтрация сегментов стен по критериям качества

        Параметры:
            seg: сегмент стены
            global_median_thickness: медианная толщина всех стен

        Возвращает:
            True если сегмент проходит фильтрацию
        """
        if not seg.thickness_profile:
            return False

        prof = np.array(seg.thickness_profile, dtype=np.float32)
        if prof.size == 0:
            return False

        valid_mask = prof > 0
        valid_ratio = float(valid_mask.mean()) if prof.size > 0 else 0.0

        # Проверка доли валидных точек
        if valid_ratio < self.min_valid_thickness_ratio:
            return False

        # Проверка толщины
        if seg.thickness_mean <= 0:
            return False

        # Относительное стандартное отклонение толщины
        if seg.thickness_mean > 0:
            rel_std = seg.thickness_std / (seg.thickness_mean + 1e-6)
            if rel_std > self.max_rel_thickness_std:
                return False

        # Сопоставимость с глобальной толщиной
        if global_median_thickness is not None and global_median_thickness > 0:
            if seg.thickness_mean < 0.3 * global_median_thickness:
                return False
            if seg.thickness_mean > 3.0 * global_median_thickness:
                return False

        # Прямолинейность (угол отклонения от 0° или 90°)
        dx = seg.x2 - seg.x1
        dy = seg.y2 - seg.y1
        angle_deg = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        deviation = min(abs(angle_deg), abs(angle_deg - 90.0))

        if deviation > self.max_angle_deviation_deg:
            return False

        return True

    def analyze_walls(self,
                      wall_lines: List[Tuple[int, int, int, int]],
                      dist_transform: np.ndarray) -> List[WallSegment]:
        """
        Полный анализ стен с измерением толщины и фильтрацией

        Параметры:
            wall_lines: список линий стен
            dist_transform: distance transform изображение

        Возвращает:
            Список отфильтрованных сегментов стен
        """
        raw_segments: List[WallSegment] = []

        for idx, (x1, y1, x2, y2) in enumerate(wall_lines):
            length = float(np.hypot(x2 - x1, y2 - y1))

            thickness_profile, sample_points = self.measure_thickness_along_line(
                x1, y1, x2, y2, dist_transform,
                num_samples=max(20, int(length / 5))
            )

            if thickness_profile:
                arr = np.array(thickness_profile, dtype=np.float32)
                valid_mask = arr > 0

                if valid_mask.any():
                    arr_valid = arr[valid_mask]
                    thickness_mean = float(arr_valid.mean())
                    thickness_min = float(arr_valid.min())
                    thickness_max = float(arr_valid.max())

                    # Для σ используем "сердцевину" профиля
                    n = len(arr_valid)
                    if n >= 5:
                        start = n // 10
                        end = n - start
                        core = arr_valid[start:end] if end > start else arr_valid
                    else:
                        core = arr_valid

                    thickness_std = float(core.std()) if core.size > 1 else 0.0
                else:
                    thickness_mean = thickness_std = thickness_min = thickness_max = 0.0
            else:
                thickness_mean = thickness_std = thickness_min = thickness_max = 0.0

            raw_segments.append(WallSegment(
                id=idx,
                x1=x1, y1=y1, x2=x2, y2=y2,
                length=length,
                thickness_mean=thickness_mean,
                thickness_std=thickness_std,
                thickness_min=thickness_min,
                thickness_max=thickness_max,
                thickness_profile=thickness_profile,
                sample_points=sample_points
            ))

        if not raw_segments:
            return []

        # Глобальная статистика толщины
        thickness_means = np.array(
            [s.thickness_mean for s in raw_segments if s.thickness_mean > 0],
            dtype=np.float32
        )

        if thickness_means.size >= 3:
            global_median = float(np.median(thickness_means))
        else:
            global_median = None

        # Применение фильтрации
        filtered_segments: List[WallSegment] = []
        for seg in raw_segments:
            if self._segment_pass_smart_filters(seg, global_median):
                filtered_segments.append(seg)

        # Перенумерация id сегментов
        for new_id, seg in enumerate(filtered_segments):
            seg.id = new_id

        return filtered_segments

    # ------------------------------------------------------------------ #
    # ПОЛНЫЙ ПАЙПЛАЙН
    # ------------------------------------------------------------------ #

    def process(self, img: np.ndarray) -> Tuple[np.ndarray, List[WallSegment], List[WallJunction], np.ndarray]:
        """
        Полный пайплайн обработки изображения

        Параметры:
            img: входное изображение (BGR)

        Возвращает:
            Кортеж (маска стен, сегменты стен, соединения, distance transform)
        """
        # OCR-очистка текстовых меток
        img_no_text = self.remove_text_labels_ocr(img)

        # Цветовая фильтрация
        wall_mask = self.filter_walls_by_color(img_no_text)

        # Удаление наиболее выраженной круглой метки
        wall_mask = self.remove_circular_labels(wall_mask)

        # Морфологическая очистка
        wall_mask = morphological_cleanup(wall_mask)

        # Скелетизация
        skeleton = skeletonize_walls(wall_mask)

        # Детекция линий
        wall_lines = self.detect_wall_lines(skeleton)

        # Distance Transform
        dist_transform = self.compute_distance_transform(wall_mask)

        # Анализ толщины
        segments = self.analyze_walls(wall_lines, dist_transform)

        # Детекция соединений
        filtered_wall_lines = [(s.x1, s.y1, s.x2, s.y2) for s in segments]
        junctions = self.detect_junctions(filtered_wall_lines, dist_transform)

        return wall_mask, segments, junctions, dist_transform