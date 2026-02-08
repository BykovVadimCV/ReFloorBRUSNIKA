"""
Система анализа планировок помещений - Основной модуль
Версия: 4.2 (Edge-Based Wall Length Measurements)
Автор: Быков Вадим Олегович
Дата: 01.29.2026
Модификация: 02.05.2026

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v4.2:
Измерение длин стен на основе прямых линий, образованных объединёнными
прямоугольниками, а не по центральным линиям отдельных прямоугольников.

Основной модуль системы анализа стен (CV) и дверей/окон (YOLO + CV алгоритмы)
с измерением размеров, отображением размеров стен и экспортом в Sweet Home 3D формат.
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Внутренние модули
from detection.walldetector import ColorBasedWallDetector, WallRectangle
from detection.furnituredetector import YOLODoorWindowDetector, YOLOResults
from detection.windowdetector import AlgorithmicWindowDetector
from detection.space import RoomSizeAnalyzer, RoomMeasurements
from detection.gap import GeometricOpeningDetector
from floorplanexporter import SweetHome3DExporter
from objexporter import ImprovedOBJExporter


class HybridWallAnalyzer:
    """
    Гибридный анализатор стен и проемов с измерением размеров.
    Объединяет YOLO детекцию с алгоритмами компьютерного зрения для комплексного анализа планировок.
    Экспортирует результаты в формат Sweet Home 3D (.sh3d).

    v4.2 МОДИФИКАЦИЯ:
    - Измерение длин стен по объединённым прямоугольникам вместо центральных линий
    - Группировка выровненных прямоугольников
    - Использование внешних границ для расчёта длины
    """

    class _SegmentStub:
        """Псевдо-сегмент стены для расчета измерений."""
        def __init__(self, x1: int, y1: int, x2: int, y2: int, thickness_mean: float = 0.0):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.thickness_mean = thickness_mean

    def __init__(self,
                 yolo_model_path: str,
                 wall_color_hex: str = "8E8780",
                 color_tolerance: int = 3,
                 yolo_conf_threshold: float = 0.5,
                 device: str = "cpu",
                 auto_wall_color: bool = False,
                 enable_measurements: bool = True,
                 edge_distance_ratio: float = 0.15,
                 debug_ocr: bool = False,
                 enable_visuals: bool = True,
                 min_window_len: int = 15,
                 edge_expansion_tolerance: int = 8,
                 max_edge_expansion_px: int = 10,
                 enable_sh3d_export: bool = True,
                 skip_bbox_check: bool = True,
                 wall_height_cm: float = 243.84,
                 snap_tolerance_px: float = 5.0,
                 min_wall_thickness_px: float = 5.0,
                 pixels_to_cm: float = 2.54,
                 auto_create_room: bool = True,
                 show_wall_sizes: bool = True,
                 enable_obj_export: bool = True,
                 obj_include_floor: bool = True,
                 obj_include_ceiling: bool = False):
        """
        Инициализация гибридного анализатора стен.

        Параметры:
            yolo_model_path: Путь к весам YOLO модели
            wall_color_hex: HEX код цвета стен для детекции
            color_tolerance: Допуск RGB для определения цвета
            yolo_conf_threshold: Порог уверенности YOLO
            device: Устройство обработки ('cpu' или 'cuda')
            auto_wall_color: Автоматическое определение цвета стен
            enable_measurements: Включить измерение площади помещений
            edge_distance_ratio: Коэффициент для определения внешних стен
            debug_ocr: Включить отладочный вывод OCR
            enable_visuals: Включить визуализацию результатов
            min_window_len: Минимальная длина окна для детекции
            edge_expansion_tolerance: Допуск для расширения краёв стен
            max_edge_expansion_px: Максимальное расширение краёв в пикселях
            enable_sh3d_export: Включить экспорт в Sweet Home 3D формат
            skip_bbox_check: Пропускать проверку bbox при детекции проёмов
            wall_height_cm: Высота стен в сантиметрах для Sweet Home 3D
            snap_tolerance_px: Допуск привязки в пикселях
            min_wall_thickness_px: Минимальная толщина стены в пикселях
            pixels_to_cm: Коэффициент преобразования пикселей в сантиметры
            auto_create_room: Автоматически создавать полигон комнаты
            show_wall_sizes: Отображать размеры стен на визуализации
            enable_obj_export: Включить экспорт в OBJ формат (v2.0 - Sweet Home 3D compatible)
            obj_include_floor: Включить плоскость пола в OBJ экспорт
            obj_include_ceiling: Включить плоскость потолка в OBJ экспорт
        """
        # Инициализация OCR модели docTR
        self.ocr_model = self._init_doctr()

        # Инициализация YOLO детектора
        self.yolo_detector = YOLODoorWindowDetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            device=device
        )

        # Инициализация детектора стен по цвету
        self.color_detector = ColorBasedWallDetector(
            wall_color_hex=wall_color_hex,
            color_tolerance=color_tolerance,
            auto_wall_color=auto_wall_color,
            min_rect_area_ratio=0.0002,
            rect_visual_gap_px=3,
            ocr_model=self.ocr_model,
            edge_expansion_tolerance=edge_expansion_tolerance,
            max_edge_expansion_px=max_edge_expansion_px
        )

        # Инициализация алгоритмического детектора окон
        self.algo_window_detector = AlgorithmicWindowDetector(
            min_window_len=min_window_len,
            ocr_model=self.ocr_model
        )

        # Сохранение параметров конфигурации
        self.enable_measurements = enable_measurements
        self.enable_visuals = enable_visuals
        self.enable_sh3d_export = enable_sh3d_export
        self.skip_bbox_check = skip_bbox_check
        self.wall_height_cm = wall_height_cm
        self.snap_tolerance_px = snap_tolerance_px
        self.min_wall_thickness_px = min_wall_thickness_px
        self.pixels_to_cm = pixels_to_cm
        self.auto_create_room = auto_create_room
        self.show_wall_sizes = show_wall_sizes
        self.enable_obj_export = enable_obj_export
        self.obj_include_floor = obj_include_floor
        self.obj_include_ceiling = obj_include_ceiling

        self.size_analyzer = None
        self.sh3d_exporter = None

        # Инициализация детектора проёмов
        self.gap_detector = GeometricOpeningDetector(
            max_extension_length=200,
            small_rect_aspect_threshold=3.0,
            small_rect_min_side=10,
            merge_distance=3,
            yolo_iou_threshold=0.3
        )

        # Инициализация анализатора размеров помещения
        if enable_measurements:
            self.size_analyzer = RoomSizeAnalyzer(
                edge_distance_ratio=edge_distance_ratio,
                debug_ocr=debug_ocr
            )

        # Инициализация Sweet Home 3D экспортера
        if enable_sh3d_export:
            self.sh3d_exporter = SweetHome3DExporter(
                min_wall_thickness=min_wall_thickness_px,
                pixels_to_cm=pixels_to_cm,
                wall_height_cm=wall_height_cm
            )

        # Инициализация OBJ экспортера v2.0 (Sweet Home 3D compatible)
        self.obj_exporter = None
        if enable_obj_export:
            self.obj_exporter = ImprovedOBJExporter(
                pixels_to_cm=pixels_to_cm,
                wall_height_cm=wall_height_cm,
                include_floor=obj_include_floor,
                include_ceiling=obj_include_ceiling
            )

    def _init_doctr(self) -> Optional[object]:
        """Инициализация модели docTR для распознавания текста."""
        try:
            from doctr.models import ocr_predictor
            return ocr_predictor(pretrained=True)
        except Exception:
            return None

    # ============================================================
    # МОДИФИЦИРОВАННЫЕ МЕТОДЫ v4.2 - EDGE-BASED MEASUREMENTS
    # ============================================================

    def _rectangles_to_segments(self, rectangles: List[WallRectangle]) -> List[_SegmentStub]:
        """
        Преобразование прямоугольников стен в сегменты для измерений.
        Объединяет соседние прямоугольники в единые линии стен.

        МОДИФИКАЦИЯ v4.2: Использует внешние границы объединённых прямоугольников
        вместо центральных линий отдельных прямоугольников.

        Параметры:
            rectangles: Список обнаруженных прямоугольников стен

        Возвращает:
            Список сегментов с информацией о толщине
        """
        if not rectangles:
            return []

        # Группировка прямоугольников по ориентации
        horizontal_rects = []
        vertical_rects = []

        for r in rectangles:
            if r.width <= 0 or r.height <= 0:
                continue

            if r.width >= r.height:
                horizontal_rects.append(r)
            else:
                vertical_rects.append(r)

        segments: List[HybridWallAnalyzer._SegmentStub] = []

        # Обработка горизонтальных стен
        if horizontal_rects:
            segments.extend(self._merge_aligned_rectangles(horizontal_rects, is_horizontal=True))

        # Обработка вертикальных стен
        if vertical_rects:
            segments.extend(self._merge_aligned_rectangles(vertical_rects, is_horizontal=False))

        return segments

    def _merge_aligned_rectangles(self, rectangles: List[WallRectangle], is_horizontal: bool) -> List[_SegmentStub]:
        """
        Объединение выровненных прямоугольников в единые линейные сегменты.

        Алгоритм:
        1. Группировка прямоугольников по выравниванию (Y для горизонтальных, X для вертикальных)
        2. Сортировка прямоугольников в каждой группе
        3. Объединение непрерывных прямоугольников в сегменты
        4. Расчёт длины по внешним границам

        Параметры:
            rectangles: Список прямоугольников одной ориентации
            is_horizontal: True для горизонтальных, False для вертикальных

        Возвращает:
            Список объединённых сегментов
        """
        if not rectangles:
            return []

        # Допуск для выравнивания (пиксели)
        alignment_tolerance = 5
        # Максимальный зазор для объединения (пиксели)
        gap_tolerance = 3

        # Группировка прямоугольников по выравниванию
        groups = []

        for rect in rectangles:
            placed = False

            for group in groups:
                # Проверка выравнивания с существующей группой
                if self._can_merge_into_group(rect, group, is_horizontal, alignment_tolerance):
                    group.append(rect)
                    placed = True
                    break

            if not placed:
                groups.append([rect])

        # Преобразование групп в сегменты
        segments = []

        for group in groups:
            if not group:
                continue

            # Сортировка прямоугольников в группе
            if is_horizontal:
                group.sort(key=lambda r: r.x1)
            else:
                group.sort(key=lambda r: r.y1)

            # Объединение прямоугольников в непрерывные сегменты
            merged_segments = self._create_merged_segments(group, is_horizontal, gap_tolerance)
            segments.extend(merged_segments)

        return segments

    def _can_merge_into_group(self, rect: WallRectangle, group: List[WallRectangle],
                              is_horizontal: bool, tolerance: int) -> bool:
        """
        Проверка возможности добавления прямоугольника в группу.

        Использует края прямоугольников для выравнивания:
        - Горизонтальные стены: проверка по Y-координатам (нижний край)
        - Вертикальные стены: проверка по X-координатам (правый край)

        Параметры:
            rect: Проверяемый прямоугольник
            group: Группа прямоугольников
            is_horizontal: Ориентация группы
            tolerance: Допуск выравнивания

        Возвращает:
            True если прямоугольник выровнен с группой
        """
        if not group:
            return True

        # Для горизонтальных: проверка выравнивания по Y
        # Для вертикальных: проверка выравнивания по X

        if is_horizontal:
            # Используем нижний край для выравнивания (более стабильный)
            rect_align = rect.y2
            group_align_values = [r.y2 for r in group]
        else:
            # Используем правый край для выравнивания
            rect_align = rect.x2
            group_align_values = [r.x2 for r in group]

        # Проверка выравнивания с любым прямоугольником в группе
        for align_val in group_align_values:
            if abs(rect_align - align_val) <= tolerance:
                return True

        return False

    def _create_merged_segments(self, group: List[WallRectangle],
                                is_horizontal: bool, gap_tolerance: int) -> List[_SegmentStub]:
        """
        Создание объединённых сегментов из группы выровненных прямоугольников.

        Прямоугольники объединяются если зазор между ними не превышает gap_tolerance.
        Это позволяет игнорировать небольшие визуальные разрывы (например, от дверей).

        Параметры:
            group: Группа выровненных прямоугольников (уже отсортированных)
            is_horizontal: Ориентация группы
            gap_tolerance: Максимальный зазор для объединения

        Возвращает:
            Список непрерывных сегментов
        """
        if not group:
            return []

        segments = []
        current_segment = []

        for rect in group:
            if not current_segment:
                current_segment.append(rect)
                continue

            # Проверка непрерывности с последним прямоугольником
            last_rect = current_segment[-1]

            if is_horizontal:
                gap = rect.x1 - last_rect.x2
            else:
                gap = rect.y1 - last_rect.y2

            if gap <= gap_tolerance:
                # Продолжение текущего сегмента
                current_segment.append(rect)
            else:
                # Создание нового сегмента
                segments.append(self._build_segment_from_rectangles(current_segment, is_horizontal))
                current_segment = [rect]

        # Добавление последнего сегмента
        if current_segment:
            segments.append(self._build_segment_from_rectangles(current_segment, is_horizontal))

        return segments

    def _build_segment_from_rectangles(self, rects: List[WallRectangle],
                                       is_horizontal: bool) -> _SegmentStub:
        """
        Построение единого сегмента из списка прямоугольников.
        Использует внешние границы объединённых прямоугольников.

        КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Длина сегмента определяется расстоянием между
        крайними точками объединённых прямоугольников, а не суммой отдельных длин.

        Параметры:
            rects: Список прямоугольников для объединения
            is_horizontal: Ориентация сегмента

        Возвращает:
            Объединённый сегмент с координатами внешних границ
        """
        if not rects:
            raise ValueError("Empty rectangle list")

        # Вычисление границ объединённой области
        x1_min = min(r.x1 for r in rects)
        x2_max = max(r.x2 for r in rects)
        y1_min = min(r.y1 for r in rects)
        y2_max = max(r.y2 for r in rects)

        # Средняя толщина
        avg_thickness = sum(r.height if is_horizontal else r.width for r in rects) / len(rects)

        if is_horizontal:
            # Горизонтальный сегмент: от левого до правого края
            # Используем нижний край стены (y2_max) для более точного представления
            y_line = y2_max
            return HybridWallAnalyzer._SegmentStub(
                x1=int(x1_min),
                y1=int(y_line),
                x2=int(x2_max),
                y2=int(y_line),
                thickness_mean=float(avg_thickness)
            )
        else:
            # Вертикальный сегмент: от верхнего до нижнего края
            # Используем правый край стены (x2_max)
            x_line = x2_max
            return HybridWallAnalyzer._SegmentStub(
                x1=int(x_line),
                y1=int(y1_min),
                x2=int(x_line),
                y2=int(y2_max),
                thickness_mean=float(avg_thickness)
            )

    # ============================================================
    # ОСТАЛЬНЫЕ МЕТОДЫ БЕЗ ИЗМЕНЕНИЙ
    # ============================================================

    def _convert_segments_to_wall_segments(self, segments) -> List[Dict]:
        """Преобразование сегментов в словари для измерений."""
        wall_segments = []
        for i, seg in enumerate(segments):
            length = float(np.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1))
            wall_segments.append({
                'id': i,
                'x1': int(seg.x1), 'y1': int(seg.y1),
                'x2': int(seg.x2), 'y2': int(seg.y2),
                'length_pixels': length,
                'thickness': float(getattr(seg, "thickness_mean", 0.0))
            })
        return wall_segments

    def _refine_living_space_with_doors(self, measurements, wall_mask, yolo_results, non_interior_threshold=0.02):
        """
        Уточнение границ жилой площади на основе расположения дверей.

        Параметры:
            measurements: Данные измерений помещения
            wall_mask: Бинарная маска стен
            yolo_results: Результаты YOLO детекции
            non_interior_threshold: Порог для определения нежилого пространства
        """
        if yolo_results is None or not getattr(yolo_results, "doors", None):
            return
        ab = getattr(measurements, "apartment_boundary", None)
        if ab is None:
            return

        if ab.ndim == 3:
            ab_gray = cv2.cvtColor(ab, cv2.COLOR_BGR2GRAY)
        else:
            ab_gray = ab

        interior_mask = ab_gray > 0
        h, w = interior_mask.shape
        remove_mask = np.zeros_like(interior_mask, dtype=bool)

        for det in yolo_results.doors:
            x1, y1, x2, y2 = det.bbox
            x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
            y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                continue
            sub = interior_mask[y1:y2, x1:x2]
            if sub.size == 0:
                continue

            frac = (sub.size - int(sub.sum())) / float(sub.size)
            if frac > non_interior_threshold:
                remove_mask[y1:y2, x1:x2] = True

        if remove_mask.any():
            new_ab = ab_gray.copy()
            new_ab[remove_mask] = 0
            measurements.apartment_boundary = new_ab
            px_scale = float(getattr(measurements, "pixel_scale", 0.0) or 0.0)
            if px_scale > 0:
                measurements.area_m2 = float((new_ab > 0).sum()) * (px_scale ** 2)

    def _convert_rectangles_for_sh3d(self, rectangles: List[WallRectangle]) -> List[Dict]:
        """Преобразование объектов WallRectangle в словари для Sweet Home 3D формата."""
        return self.color_detector.rectangles_to_dict_list(rectangles)

    def _convert_openings_for_sh3d(self, openings: List) -> List[Dict]:
        """Преобразование обнаруженных проёмов в Sweet Home 3D совместимый формат."""
        converted = []
        for op in openings:
            opening_dict = {
                'x1': int(op.x1),
                'y1': int(op.y1),
                'x2': int(op.x2),
                'y2': int(op.y2),
                'opening_type': getattr(op, 'type', getattr(op, 'opening_type', 'unknown')),
                'is_valid': getattr(op, 'is_valid', True)
            }
            converted.append(opening_dict)
        return converted

    def _convert_algo_windows_for_sh3d(self, algo_windows: List) -> List[Dict]:
        """Преобразование алгоритмических окон в Sweet Home 3D совместимый формат."""
        converted = []
        for win in algo_windows:
            win_dict = {
                'x': int(win.x),
                'y': int(win.y),
                'w': int(win.w),
                'h': int(win.h),
                'id': getattr(win, 'id', 0)
            }
            converted.append(win_dict)
        return converted

    def _convert_yolo_doors_for_sh3d(self, yolo_doors: List) -> List[Dict]:
        """Преобразование YOLO дверей в Sweet Home 3D совместимый формат."""
        converted = []
        for door in yolo_doors:
            bbox = door.bbox if hasattr(door, 'bbox') else door[:4]
            x1, y1, x2, y2 = map(float, bbox)
            converted.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
        return converted

    def _convert_yolo_windows_for_sh3d(self, yolo_windows: List) -> List[Dict]:
        """Преобразование YOLO окон в Sweet Home 3D совместимый формат."""
        converted = []
        for window in yolo_windows:
            bbox = window.bbox if hasattr(window, 'bbox') else window[:4]
            x1, y1, x2, y2 = map(float, bbox)
            converted.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
        return converted

    def _export_sh3d(
        self,
        color_results: Dict,
        yolo_results: YOLOResults,
        measurements: Optional[RoomMeasurements],
        output_dir: str,
        base_name: str
    ) -> Optional[str]:
        """
        Экспорт планировки в формат Sweet Home 3D.

        Параметры:
            color_results: Результаты детекции стен по цвету
            yolo_results: Результаты YOLO детекции
            measurements: Данные измерений помещения
            output_dir: Директория для вывода
            base_name: Базовое имя файла

        Возвращает:
            Путь к созданному .sh3d файлу при успехе, None при ошибке
        """
        if not self.enable_sh3d_export or not self.sh3d_exporter:
            return None

        if not color_results or 'rectangles' not in color_results:
            return None

        try:
            # Создание директории вывода
            sh3d_dir = os.path.join(output_dir, 'sh3d')
            os.makedirs(sh3d_dir, exist_ok=True)
            sh3d_path = os.path.join(sh3d_dir, f"{base_name}.sh3d")

            # Обновление коэффициента преобразования пикселей в см, если доступны измерения
            pixels_to_cm = self.pixels_to_cm
            if measurements and measurements.pixel_scale:
                pixels_to_cm = measurements.pixel_scale * 100.0
                # Обновляем экспортер с новым масштабом
                self.sh3d_exporter.pixels_to_cm = pixels_to_cm

            # Преобразование данных в Sweet Home 3D совместимые форматы
            rectangles_sh3d = self._convert_rectangles_for_sh3d(
                color_results['rectangles']
            )

            # Собираем все окна (алгоритмические + YOLO)
            window_rects = []

            # Алгоритмические окна
            algo_windows = color_results.get('algo_windows', [])
            for win in algo_windows:
                window_rects.append({
                    'x': int(win.x),
                    'y': int(win.y),
                    'width': int(win.w),
                    'height': int(win.h)
                })

            # YOLO окна
            if yolo_results and yolo_results.windows:
                yolo_windows_sh3d = self._convert_yolo_windows_for_sh3d(yolo_results.windows)
                for win in yolo_windows_sh3d:
                    window_rects.append({
                        'x1': win['x1'],
                        'y1': win['y1'],
                        'x2': win['x2'],
                        'y2': win['y2']
                    })

            # Собираем все двери (YOLO + проёмы)
            door_rects = []

            # YOLO двери
            if yolo_results and yolo_results.doors:
                yolo_doors_sh3d = self._convert_yolo_doors_for_sh3d(yolo_results.doors)
                for door in yolo_doors_sh3d:
                    door_rects.append({
                        'x1': door['x1'],
                        'y1': door['y1'],
                        'x2': door['x2'],
                        'y2': door['y2']
                    })

            # Проёмы как двери
            openings = color_results.get('openings', [])
            for opening in openings:
                if getattr(opening, 'is_valid', True):
                    door_rects.append({
                        'x1': int(opening.x1),
                        'y1': int(opening.y1),
                        'x2': int(opening.x2),
                        'y2': int(opening.y2)
                    })

            # Экспорт в .sh3d
            self.sh3d_exporter.export_to_sh3d(
                wall_rectangles=rectangles_sh3d,
                output_path=sh3d_path,
                door_rectangles=door_rects if door_rects else None,
                window_rectangles=window_rects if window_rects else None,
                auto_create_room=self.auto_create_room
            )

            return sh3d_path

        except Exception as e:
            print(f"Sweet Home 3D export error: {e}")
            return None

    def process_image(self, img_path: str, output_dir: str, run_color_detection: bool = True):
        """
        Обработка одного изображения планировки.

        Параметры:
            img_path: Путь к входному изображению
            output_dir: Директория для выходных файлов
            run_color_detection: Включить детекцию стен по цвету

        Возвращает:
            Словарь с результатами детекции и измерениями
        """
        img_name = Path(img_path).name
        base_name = Path(img_path).stem
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")

        # Шаг 1: YOLO детекция дверей
        yolo_results = self.yolo_detector.detect(img, verbose=False)

        color_results = None
        measurements: Optional[RoomMeasurements] = None

        if run_color_detection:
            # Шаг 2: Детекция стен
            wall_mask_raw, rectangles, outline_mask = self.color_detector.process(img)

            # Шаг 3: Алгоритмическая детекция окон
            door_bboxes = []
            if yolo_results and yolo_results.doors:
                door_bboxes = [det.bbox for det in yolo_results.doors]

            ocr_bboxes = self.color_detector.get_ocr_bboxes(img)

            algo_windows = self.algo_window_detector.detect(
                img,
                outline_mask,
                exclude_bboxes=door_bboxes,
                ocr_bboxes=ocr_bboxes
            )

            algo_win_mask = self.algo_window_detector.get_mask(img.shape, algo_windows)

            # Шаг 4: Интеграция окон в стены
            wall_mask = wall_mask_raw.copy()
            combined_opening_mask = np.zeros_like(wall_mask)
            combined_opening_mask = cv2.bitwise_or(combined_opening_mask, algo_win_mask)

            if np.count_nonzero(combined_opening_mask) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                win_mask_expanded = cv2.dilate(combined_opening_mask, kernel, iterations=1)
                wall_mask[win_mask_expanded > 0] = 255

            # Шаг 5: Построение результатов (с новым методом _rectangles_to_segments)
            segments = self._rectangles_to_segments(rectangles)

            color_results = {
                'wall_mask': wall_mask,
                'segments': segments,
                'junctions': [],
                'rectangles': rectangles,
                'outline_mask': outline_mask,
                'openings': [],
                'algo_windows': algo_windows,
                'algo_win_mask': algo_win_mask
            }

            # Шаг 6: Начальная детекция проёмов
            if rectangles:
                try:
                    detected_openings = self.gap_detector.detect_openings(
                        rectangles=rectangles,
                        yolo_results=yolo_results
                    )
                    color_results['openings'] = detected_openings
                except Exception:
                    pass

            # Шаг 7: Сохранение промежуточных результатов
            json_dir = os.path.join(output_dir, 'json')
            os.makedirs(json_dir, exist_ok=True)

            # Сохранение прямоугольников
            with open(os.path.join(json_dir, f"{base_name}_rectangles.json"), "w") as f:
                json.dump(self.color_detector.rectangles_to_dict_list(rectangles), f, indent=2)

            # Сохранение окон
            algo_win_list = [{"id": w.id, "x": w.x, "y": w.y, "w": w.w, "h": w.h} for w in algo_windows]
            with open(os.path.join(json_dir, f"{base_name}_algo_windows.json"), "w") as f:
                json.dump(algo_win_list, f, indent=2)

            # Шаг 8: Измерения
            if self.enable_measurements and segments:
                try:
                    wall_segments_data = self._convert_segments_to_wall_segments(segments)
                    measurements = self.size_analyzer.process(
                        img_path, wall_segments_data, output_dir,
                        wall_mask=outline_mask, original_image=img
                    )
                except Exception:
                    pass

            # Шаг 9: Финальная детекция проёмов с масштабом
            if rectangles:
                try:
                    pixel_scale = measurements.pixel_scale if measurements else None
                    combined_boxes = []
                    if yolo_results and yolo_results.doors:
                        combined_boxes.extend(yolo_results.doors)

                    detected_openings = self.gap_detector.detect_openings(
                        rectangles=rectangles,
                        yolo_results=yolo_results,
                        pixel_scale=pixel_scale,
                        image=img,
                        door_window_boxes=combined_boxes
                    )
                    color_results['openings'] = detected_openings

                except Exception:
                    pass

        # Уточнение жилой площади
        if measurements and color_results:
            self._refine_living_space_with_doors(
                measurements, color_results['wall_mask'], yolo_results
            )

        # Экспорт Sweet Home 3D
        sh3d_path = self._export_sh3d(
            color_results=color_results,
            yolo_results=yolo_results,
            measurements=measurements,
            output_dir=output_dir,
            base_name=base_name
        )

        if sh3d_path and color_results:
            color_results['sh3d_path'] = sh3d_path

        # OBJ Export
        if self.obj_exporter is not None and color_results and 'segments' in color_results:
            obj_path = os.path.join(output_dir, f"{base_name}.obj")
            self.obj_exporter.export_from_segments(color_results['segments'], obj_path)
            color_results['obj_path'] = obj_path

        # Визуализация
        if self.enable_visuals:
            if run_color_detection and color_results:
                summary_path = os.path.join(output_dir, f"{base_name}_summary.png")
                self._visualize_combined_results(
                    img, yolo_results, color_results, summary_path, measurements
                )

                overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                self._save_overlay_image(img, yolo_results, color_results, overlay_path, measurements)

                masks_dir = os.path.join(output_dir, 'masks')
                os.makedirs(masks_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(masks_dir, f"{base_name}_algo_windows.png"),
                    color_results['algo_win_mask']
                )
            else:
                yolo_vis_path = os.path.join(output_dir, f"{base_name}_yolo.png")
                self._visualize_yolo_results(img, yolo_results, yolo_vis_path)

        # Сохранение масок
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(masks_dir, f"{base_name}_yolo_doors.png"),
            yolo_results.combined_door_mask
        )

        if run_color_detection and color_results:
            cv2.imwrite(
                os.path.join(masks_dir, f"{base_name}_color_walls.png"),
                color_results['wall_mask']
            )
            cv2.imwrite(
                os.path.join(masks_dir, f"{base_name}_rect_outline.png"),
                color_results['outline_mask']
            )

        return {
            'yolo': yolo_results,
            'color': color_results,
            'measurements': measurements
        }

    def _visualize_yolo_results(self, img: np.ndarray, yolo_results: YOLOResults, output_path: str):
        """Visualization of YOLO detection results."""
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        yolo_vis = self.yolo_detector.visualize_detections(img, yolo_results)
        ax2.imshow(cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title('YOLO Results')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _build_overlay(
        self,
        img: np.ndarray,
        yolo_results: YOLOResults,
        color_results: Dict,
        measurements: Optional[RoomMeasurements]
    ) -> np.ndarray:
        """Build visualization overlay with all detection results."""
        overlay = img.copy()
        segments = color_results['segments']

        # Determine scale for size display
        pixel_scale = measurements.pixel_scale if measurements else self.pixels_to_cm / 100.0

        # Step 1: Draw walls with measurements
        outer_wall_ids = set()
        if measurements:
            outer_wall_ids = {w['id'] for w in measurements.walls if w['is_outer']}
            # Draw outer crust if available
            if measurements.outer_crust is not None:
                crust_mask = measurements.outer_crust > 0
                # Use semi-transparent overlay for outer walls
                overlay_colored = overlay.copy()
                overlay_colored[crust_mask] = [100, 200, 100]  # Light green
                cv2.addWeighted(overlay, 0.7, overlay_colored, 0.3, 0, overlay)

        # Draw wall segments
        for i, seg in enumerate(segments):
            is_outer = i in outer_wall_ids

            # Wall line colors - more distinct
            if is_outer:
                wall_color = (0, 180, 0)  # Green for outer walls
                line_thickness = 4
            else:
                wall_color = (100, 100, 100)  # Gray for inner walls
                line_thickness = 3

            # Draw wall line
            pt1 = (int(seg.x1), int(seg.y1))
            pt2 = (int(seg.x2), int(seg.y2))
            cv2.line(overlay, pt1, pt2, wall_color, line_thickness)

            # Calculate and display wall measurements
            length_px = np.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)

            # Skip very short segments
            if length_px < 10:
                continue

            length_cm = length_px * pixel_scale * 100  # Convert to cm

            # Format size text
            if length_cm >= 100:
                size_text = f"{length_cm/100:.1f}m"
            else:
                size_text = f"{int(length_cm)}cm"

            # Calculate midpoint for text placement
            mid_x = int((seg.x1 + seg.x2) / 2)
            mid_y = int((seg.y1 + seg.y2) / 2)

            # Determine wall orientation and text offset
            dx = seg.x2 - seg.x1
            dy = seg.y2 - seg.y1
            angle = np.degrees(np.arctan2(dy, dx))

            # Normalize angle to [-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # Determine if wall is more horizontal or vertical
            is_horizontal = abs(angle) < 45

            if is_horizontal:
                # Horizontal wall - place text above or below
                text_offset_x = 0
                text_offset_y = -15 if is_outer else -12
            else:
                # Vertical wall - place text to the side
                text_offset_x = 15 if is_outer else 12
                text_offset_y = 5

            text_x = mid_x + text_offset_x
            text_y = mid_y + text_offset_y

            # Text styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            font_thickness = 1

            # Colors based on wall type
            if is_outer:
                text_color = (255, 255, 255)
                bg_color = (0, 120, 0)
            else:
                text_color = (255, 255, 255)
                bg_color = (60, 60, 60)

            # Get text dimensions
            (text_width, text_height), baseline = cv2.getTextSize(
                size_text, font, font_scale, font_thickness
            )

            # Draw background rectangle with padding
            padding = 3
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_height - padding
            bg_x2 = text_x + text_width + padding
            bg_y2 = text_y + baseline + padding

            # Ensure background is within image bounds
            h, w = overlay.shape[:2]
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(w, bg_x2)
            bg_y2 = min(h, bg_y2)

            # Draw semi-transparent background
            overlay_bg = overlay.copy()
            cv2.rectangle(overlay_bg, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            cv2.addWeighted(overlay, 0.7, overlay_bg, 0.3, 0, overlay)

            # Draw text
            cv2.putText(overlay, size_text, (text_x, text_y),
                       font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Step 2: Draw algorithmic windows (yellow highlights)
        algo_windows = color_results.get('algo_windows', [])
        for win in algo_windows:
            x, y, w, h = win.x, win.y, win.w, win.h

            # Ensure window is within bounds
            img_h, img_w = overlay.shape[:2]
            if y + h <= img_h and x + w <= img_w and y >= 0 and x >= 0:
                # Semi-transparent yellow fill
                sub = overlay[y:y+h, x:x+w]
                if sub.size > 0:
                    yellow_overlay = np.full_like(sub, (0, 255, 255))  # Yellow in BGR
                    cv2.addWeighted(sub, 0.6, yellow_overlay, 0.4, 0, sub)

                # Yellow border
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 255), 2)

        # Step 3: Draw YOLO doors (blue boxes)
        if yolo_results and yolo_results.doors:
            for door in yolo_results.doors:
                x1, y1, x2, y2 = map(int, door.bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), 2)  # Blue
                # Add label
                cv2.putText(overlay, "Door", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)

        # Step 4: Draw YOLO windows (cyan boxes - if any detected)
        if yolo_results and yolo_results.windows:
            for window in yolo_results.windows:
                x1, y1, x2, y2 = map(int, window.bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan

        # Step 5: Draw openings (orange boxes)
        openings = color_results.get('openings', [])
        if openings:
            for opening in openings:
                if getattr(opening, 'is_valid', True):
                    x1, y1, x2, y2 = opening.x1, opening.y1, opening.x2, opening.y2
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 2)  # Orange

        # Step 6: Add measurement info banner at top
        if measurements:
            # Create semi-transparent banner
            banner_height = 35
            banner = overlay[0:banner_height, :].copy()
            banner_bg = np.zeros_like(banner)
            banner_bg[:] = (40, 40, 40)
            cv2.addWeighted(banner, 0.3, banner_bg, 0.7, 0, banner)
            overlay[0:banner_height, :] = banner

            # Add text to banner
            info_text = f"Area: {measurements.area_m2:.1f} sq.m  |  Scale: {measurements.pixel_scale * 100:.2f} cm/px"
            cv2.putText(overlay, info_text, (10, 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return overlay

    def _save_overlay_image(self, img, yolo, color, path, meas):
        """Сохранение изображения с наложением."""
        overlay = self._build_overlay(img, yolo, color, meas)
        cv2.imwrite(path, overlay)

    def _visualize_combined_results(self, img, yolo, color, output_path, measurements):
        """Create combined visualization of results with clean, professional layout."""
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Title styling
        title_props = {'fontsize': 11, 'fontweight': 'bold', 'pad': 10}

        # Row 1: Main visualizations
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Floor Plan', **title_props)
        ax1.axis('off')

        # Overlay with measurements
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self._build_overlay(img, yolo, color, measurements)
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title('Detected Elements & Measurements', **title_props)
        ax2.axis('off')

        # Window mask
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(color['algo_win_mask'], cmap='gray')
        window_count = len(color.get('algo_windows', []))
        ax3.set_title(f'Window Detection ({window_count} found)', **title_props)
        ax3.axis('off')

        # Row 2: Analysis results
        # Apartment boundary
        ax4 = fig.add_subplot(gs[1, 0])
        if measurements and measurements.apartment_boundary is not None:
            ax4.imshow(measurements.apartment_boundary, cmap='viridis')
            ax4.set_title('Apartment Boundary', **title_props)
        else:
            ax4.imshow(np.zeros_like(img[:, :, 0]), cmap='gray')
            ax4.set_title('Apartment Boundary (N/A)', **title_props)
        ax4.axis('off')

        # Wall outline
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(color['outline_mask'], cmap='gray')
        ax5.set_title('Wall Structure Outline', **title_props)
        ax5.axis('off')

        # Information panel with legend
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        # Build info text
        info_lines = []
        info_lines.append("═══ DETECTION SUMMARY ═══")
        info_lines.append("")
        info_lines.append(f"Doors (YOLO):     {len(yolo.doors)}")
        info_lines.append(f"Windows (CV):     {len(color.get('algo_windows', []))}")
        info_lines.append(f"Wall Segments:    {len(color.get('segments', []))}")
        info_lines.append(f"Openings:         {len(color.get('openings', []))}")
        info_lines.append("")

        if measurements:
            info_lines.append("═══ MEASUREMENTS ═══")
            info_lines.append("")
            info_lines.append(f"Total Area:       {measurements.area_m2:.1f} m²")
            info_lines.append(f"Scale:            {measurements.pixel_scale*100:.2f} cm/px")

            outer_walls = sum(1 for w in measurements.walls if w.get('is_outer', False))
            inner_walls = len(measurements.walls) - outer_walls
            info_lines.append(f"Outer Walls:      {outer_walls}")
            info_lines.append(f"Inner Walls:      {inner_walls}")

        info_lines.append("")
        info_lines.append("═══ COLOR LEGEND ═══")
        info_lines.append("")
        info_lines.append("🟢 Green:  Outer walls")
        info_lines.append("⚫ Gray:   Inner walls")
        info_lines.append("🔵 Blue:   Doors (YOLO)")
        info_lines.append("🟡 Yellow: Windows (CV)")
        info_lines.append("🟠 Orange: Openings")

        if 'sh3d_path' in color:
            info_lines.append("")
            info_lines.append("═══ EXPORT ═══")
            info_lines.append("")
            info_lines.append("Sweet Home 3D file created:")
            sh3d_name = Path(color['sh3d_path']).name
            # Split long filename if needed
            if len(sh3d_name) > 25:
                info_lines.append(sh3d_name[:25])
                info_lines.append(sh3d_name[25:])
            else:
                info_lines.append(sh3d_name)

        # Display info text
        info_text = "\n".join(info_lines)
        ax6.text(0.05, 0.98, info_text,
                fontsize=9,
                va='top',
                ha='left',
                family='monospace',
                transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Add overall title
        fig.suptitle('Floor Plan Analysis Results - v4.2 (Edge-Based Measurements)',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()


# ============================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Floor Plan Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input image.png --yolo-model weights.pt --output results/
  python main.py --input images/ --yolo-model weights.pt --wall-height 280 --pixels-to-cm 2.5
  python main.py --input image.png --yolo-model weights.pt --no-wall-sizes  # Disable wall sizes
  python main.py --input image.png --yolo-model weights.pt --include-ceiling  # OBJ with ceiling
  python main.py --input image.png --yolo-model weights.pt --no-floor  # OBJ walls only

v4.2 Changes:
  - Wall lengths now calculated from merged rectangle edges
  - Automatic grouping of aligned rectangles
  - More accurate measurements for continuous walls

OBJ Export v2.0 Features:
  - Sweet Home 3D compatible format
  - Y-up coordinate system (industry standard)
  - Vertex normals for proper lighting
  - Ground plane (optional with --no-floor)
  - Ceiling plane (optional with --include-ceiling)
  - Multi-group walls for material control
        """
    )

    # Input/output parameters
    parser.add_argument('--input', type=str, required=True,
                        help='Path to image or directory')
    parser.add_argument('--output', type=str, default='output_sh3d',
                        help='Output directory for results')

    # YOLO configuration
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Path to YOLO model weights')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Processing device: cpu or cuda')

    # Wall detection configuration
    parser.add_argument('--wall-color', type=str, default='8E8780',
                        help='Wall color HEX code (default: 8E8780)')
    parser.add_argument('--color-tolerance', type=int, default=3,
                        help='RGB color tolerance (default: 3)')
    parser.add_argument('--auto-wall-color', action='store_true',
                        help='Automatic wall color detection')
    parser.add_argument('--skip-color', action='store_true',
                        help='Skip CV wall detection (YOLO only)')

    # Measurement configuration
    parser.add_argument('--disable-measurements', action='store_true',
                        help='Disable area measurements')
    parser.add_argument('--edge-distance-ratio', type=float, default=0.15,
                        help='Coefficient for outer wall detection (default: 0.15)')
    parser.add_argument('--debug-ocr', action='store_true',
                        help='Enable OCR debug output')

    # Window detection configuration
    parser.add_argument('--min-window-len', type=int, default=15,
                        help='Minimum window length for detection (default: 15)')

    # Wall edge expansion configuration
    parser.add_argument('--edge-expansion-tolerance', type=int, default=8,
                        help='Color tolerance for wall edge expansion (default: 8)')
    parser.add_argument('--max-edge-expansion', type=int, default=10,
                        help='Maximum wall edge expansion in pixels (default: 10)')

    # Visualization
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable result visualization')
    parser.add_argument('--no-wall-sizes', action='store_true',
                        help='Do not display wall sizes on visualization')

    # Sweet Home 3D export parameters
    sh3d_group = parser.add_argument_group('Sweet Home 3D Export')
    sh3d_group.add_argument('--disable-sh3d', action='store_true',
                            help='Disable Sweet Home 3D export')
    sh3d_group.add_argument('--no-skip-bbox-check', action='store_true',
                            help='Enable bbox check in opening detector')
    sh3d_group.add_argument('--wall-height', type=float, default=243.84,
                            help='Wall height in centimeters (default: 243.84 = ~8 feet)')
    sh3d_group.add_argument('--snap-tolerance', type=float, default=5.0,
                            help='Snap tolerance in pixels (default: 5.0)')
    sh3d_group.add_argument('--min-wall-thickness', type=float, default=5.0,
                            help='Minimum wall thickness in pixels (default: 5.0)')
    sh3d_group.add_argument('--pixels-to-cm', type=float, default=2.54,
                            help='Pixels to cm conversion coefficient (default: 2.54 = 1 inch)')
    sh3d_group.add_argument('--no-auto-room', action='store_true',
                            help='Do not automatically create room polygon')

    # OBJ export parameters
    obj_group = parser.add_argument_group('OBJ Export')
    obj_group.add_argument('--disable-obj', action='store_true',
                           help='Disable OBJ export')
    obj_group.add_argument('--no-floor', action='store_true',
                           help='Do not include ground plane in OBJ export')
    obj_group.add_argument('--include-ceiling', action='store_true',
                           help='Include ceiling plane in OBJ export')


    args = parser.parse_args()

    # Валидация коэффициента расстояния от края
    if not 0.01 <= args.edge_distance_ratio <= 0.5:
        args.edge_distance_ratio = max(0.01, min(0.5, args.edge_distance_ratio))

    # Создание директории вывода
    os.makedirs(args.output, exist_ok=True)

    # Сбор файлов изображений
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [str(input_path)]
    elif input_path.is_dir():
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend([str(f) for f in input_path.glob(ext)])
        image_files.sort()
    else:
        raise ValueError(f"Path does not exist: {args.input}")

    if not image_files:
        raise ValueError(f"No images found: {args.input}")

    # Создание анализатора
    print(f"\n{'='*60}")
    print(f"Floor Plan Analysis System")
    print(f"Edge-Based Wall Length Measurements")
    print(f"Sweet Home 3D Compatible OBJ Export")
    print(f"{'='*60}\n")

    analyzer = HybridWallAnalyzer(
        yolo_model_path=args.yolo_model,
        wall_color_hex=args.wall_color,
        color_tolerance=args.color_tolerance,
        yolo_conf_threshold=args.yolo_conf,
        device=args.device,
        auto_wall_color=args.auto_wall_color,
        enable_measurements=not args.disable_measurements,
        edge_distance_ratio=args.edge_distance_ratio,
        debug_ocr=args.debug_ocr,
        enable_visuals=not args.no_vis,
        min_window_len=args.min_window_len,
        edge_expansion_tolerance=args.edge_expansion_tolerance,
        max_edge_expansion_px=args.max_edge_expansion,
        enable_sh3d_export=not args.disable_sh3d,
        skip_bbox_check=not args.no_skip_bbox_check,
        wall_height_cm=args.wall_height,
        snap_tolerance_px=args.snap_tolerance,
        min_wall_thickness_px=args.min_wall_thickness,
        pixels_to_cm=args.pixels_to_cm,
        auto_create_room=not args.no_auto_room,
        show_wall_sizes=not args.no_wall_sizes,
        enable_obj_export=not args.disable_obj,
        obj_include_floor=not args.no_floor,
        obj_include_ceiling=args.include_ceiling
    )

    # Processing images
    results = []
    for idx, img_path in enumerate(image_files, 1):
        print(f"\nProcessing [{idx}/{len(image_files)}]: {Path(img_path).name}")
        try:
            result = analyzer.process_image(
                img_path,
                args.output,
                run_color_detection=not args.skip_color
            )

            # Output information about created .sh3d file
            if result.get('color') and 'sh3d_path' in result['color']:
                sh3d_path = result['color']['sh3d_path']
                print(f"  ✓ Sweet Home 3D: {Path(sh3d_path).name}")

            # Output information about created .obj file
            if result.get('color') and 'obj_path' in result['color']:
                obj_path = result['color']['obj_path']
                print(f"  ✓ OBJ Model: {Path(obj_path).name}")

            # Output segment count info
            if result.get('color') and 'segments' in result['color']:
                seg_count = len(result['color']['segments'])
                print(f"  ✓ Wall segments: {seg_count} (merged from rectangles)")

            results.append({
                'path': img_path,
                'success': True,
                'result': result
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'path': img_path,
                'success': False,
                'error': str(e)
            })

    # Generate summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"Processing complete: {successful}/{len(results)} images successful")
    if failed > 0:
        print(f"Errors: {failed} images")
    print(f"Results saved to: {args.output}")

    # Output information about .sh3d files
    sh3d_files = [r for r in results if r['success'] and
                  r.get('result', {}).get('color', {}).get('sh3d_path')]
    if sh3d_files:
        print(f"\n.sh3d files created: {len(sh3d_files)}")
        print("Open them in Sweet Home 3D!")

    # Output information about .obj files
    obj_files = [r for r in results if r['success'] and
                 r.get('result', {}).get('color', {}).get('obj_path')]
    if obj_files:
        print(f"\n.obj files created: {len(obj_files)} (v2.0 - Sweet Home 3D compatible)")
        print("Features: Y-up coordinates, vertex normals, ground plane")
        print("Open them in Blender, SketchUp, Sweet Home 3D, or any 3D modeling software!")


    print(f"\nv4.2 Note: Wall lengths calculated from merged rectangle edges")
    print(f"OBJ v2.0: Full Sweet Home 3D compatibility with Y-up coordinate system")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()