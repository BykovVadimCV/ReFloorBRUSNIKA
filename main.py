"""
Система анализа планировок помещений - Основной модуль
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Основной модуль системы анализа стен (CV) и дверей/окон (YOLO + CV алгоритмы)
с измерением размеров и экспортом в Blueprint3D формат.
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
from floorplanexporter import FloorplanBuilder


class HybridWallAnalyzer:
    """
    Гибридный анализатор стен и проемов с измерением размеров.
    Объединяет YOLO детекцию с алгоритмами компьютерного зрения для комплексного анализа планировок.
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
                 enable_blueprint3d_export: bool = True,
                 skip_bbox_check: bool = True,
                 default_window_width_inches: float = 48.0,
                 default_door_width_inches: float = 65.0,
                 wall_height_cm: float = 300.0,
                 corner_snap_tolerance_px: float = 10.0):
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
            enable_blueprint3d_export: Включить экспорт в Blueprint3D формат
            skip_bbox_check: Пропускать проверку bbox при детекции проёмов
            default_window_width_inches: Ширина окна по умолчанию для Blueprint3D
            default_door_width_inches: Ширина двери по умолчанию для Blueprint3D
            wall_height_cm: Высота стен для Blueprint3D экспорта
            corner_snap_tolerance_px: Допуск привязки углов в пикселях
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
        self.enable_blueprint3d_export = enable_blueprint3d_export
        self.skip_bbox_check = skip_bbox_check
        self.default_window_width_inches = default_window_width_inches
        self.default_door_width_inches = default_door_width_inches
        self.wall_height_cm = wall_height_cm
        self.corner_snap_tolerance_px = corner_snap_tolerance_px

        self.size_analyzer = None
        self.floorplan_builder = None

        # Инициализация детектора проёмов
        self.gap_detector = GeometricOpeningDetector(
            max_extension_length=150,
            small_rect_aspect_threshold=2.0,
            small_rect_min_side=20,
            merge_distance=3,
            yolo_iou_threshold=0.3
        )

        # Инициализация анализатора размеров помещения
        if enable_measurements:
            self.size_analyzer = RoomSizeAnalyzer(
                edge_distance_ratio=edge_distance_ratio,
                debug_ocr=debug_ocr
            )

    def _init_doctr(self) -> Optional[object]:
        """Инициализация модели docTR для распознавания текста."""
        try:
            from doctr.models import ocr_predictor
            return ocr_predictor(pretrained=True)
        except Exception:
            return None

    def _rectangles_to_segments(self, rectangles: List[WallRectangle]) -> List[_SegmentStub]:
        """
        Преобразование прямоугольников стен в сегменты для измерений.

        Параметры:
            rectangles: Список обнаруженных прямоугольников стен

        Возвращает:
            Список сегментов с информацией о толщине
        """
        segments: List[HybridWallAnalyzer._SegmentStub] = []

        for r in rectangles:
            width = r.width
            height = r.height
            if width <= 0 or height <= 0:
                continue

            if width >= height:
                # Горизонтальный сегмент
                y = (r.y1 + r.y2) // 2
                x1, x2 = r.x1, r.x2
                thickness = float(height)
                seg = HybridWallAnalyzer._SegmentStub(x1, y, x2, y, thickness_mean=thickness)
            else:
                # Вертикальный сегмент
                x = (r.x1 + r.x2) // 2
                y1, y2 = r.y1, r.y2
                thickness = float(width)
                seg = HybridWallAnalyzer._SegmentStub(x, y1, x, y2, thickness_mean=thickness)

            segments.append(seg)
        return segments

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

    def _convert_rectangles_for_blueprint3d(self, rectangles: List[WallRectangle]) -> List[Dict]:
        """Преобразование объектов WallRectangle в словари для Blueprint3D формата."""
        return self.color_detector.rectangles_to_dict_list(rectangles)

    def _convert_openings_for_blueprint3d(self, openings: List) -> List[Dict]:
        """Преобразование обнаруженных проёмов в Blueprint3D совместимый формат."""
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

    def _convert_algo_windows_for_blueprint3d(self, algo_windows: List) -> List[Dict]:
        """Преобразование алгоритмических окон в Blueprint3D совместимый формат."""
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

    def _convert_yolo_doors_for_blueprint3d(self, yolo_doors: List) -> List[List[float]]:
        """Преобразование YOLO дверей в Blueprint3D совместимый формат (список bbox)."""
        converted = []
        for door in yolo_doors:
            bbox = door.bbox if hasattr(door, 'bbox') else door[:4]
            converted.append(list(map(float, bbox)))
        return converted

    def _convert_yolo_windows_for_blueprint3d(self, yolo_windows: List) -> List[List[float]]:
        """Преобразование YOLO окон в Blueprint3D совместимый формат (список bbox)."""
        converted = []
        for window in yolo_windows:
            bbox = window.bbox if hasattr(window, 'bbox') else window[:4]
            converted.append(list(map(float, bbox)))
        return converted

    def _create_floorplan_builder(self) -> FloorplanBuilder:
        """Создание нового экземпляра FloorplanBuilder с текущими настройками."""
        return FloorplanBuilder(
            wall_height_cm=self.wall_height_cm,
            corner_snap_tolerance_px=self.corner_snap_tolerance_px,
            min_wall_length_cm=20.0,
            skip_bbox_check=self.skip_bbox_check,
            default_window_width_inches=self.default_window_width_inches,
            default_door_width_inches=self.default_door_width_inches
        )

    def _export_blueprint3d(
        self,
        color_results: Dict,
        yolo_results: YOLOResults,
        measurements: Optional[RoomMeasurements],
        output_dir: str,
        base_name: str
    ) -> Optional[Dict]:
        """
        Экспорт планировки в формат Blueprint3D.

        Параметры:
            color_results: Результаты детекции стен по цвету
            yolo_results: Результаты YOLO детекции
            measurements: Данные измерений помещения
            output_dir: Директория для вывода
            base_name: Базовое имя файла

        Возвращает:
            Словарь Blueprint3D при успехе, None при ошибке
        """
        if not self.enable_blueprint3d_export:
            return None

        if not color_results or 'rectangles' not in color_results:
            return None

        try:
            # Создание директории вывода
            json_dir = os.path.join(output_dir, 'json')
            os.makedirs(json_dir, exist_ok=True)
            blueprint_path = os.path.join(json_dir, f"{base_name}.blueprint3d")

            # Расчёт коэффициента преобразования пикселей в см
            pixels_to_cm = 1.0
            if measurements and measurements.pixel_scale:
                pixels_to_cm = measurements.pixel_scale * 100.0

            # Преобразование данных в Blueprint3D совместимые форматы
            rectangles_bp = self._convert_rectangles_for_blueprint3d(
                color_results['rectangles']
            )

            algo_windows_bp = self._convert_algo_windows_for_blueprint3d(
                color_results.get('algo_windows', [])
            )

            yolo_doors_bp = self._convert_yolo_doors_for_blueprint3d(
                yolo_results.doors if yolo_results else []
            )

            yolo_windows_bp = self._convert_yolo_windows_for_blueprint3d(
                yolo_results.windows if yolo_results else []
            )

            openings_bp = self._convert_openings_for_blueprint3d(
                color_results.get('openings', [])
            )

            # Создание построителя и построение планировки
            builder = self._create_floorplan_builder()

            blueprint_result = builder.build(
                rectangles=rectangles_bp,
                pixels_to_cm=pixels_to_cm,
                algo_windows=algo_windows_bp,
                yolo_doors=yolo_doors_bp,
                yolo_windows=yolo_windows_bp,
                pre_detected_openings=openings_bp if openings_bp else None
            )

            # Сохранение в файл
            with open(blueprint_path, 'w', encoding='utf-8') as f:
                json.dump(blueprint_result, f, indent=2, ensure_ascii=False)

            return blueprint_result

        except Exception:
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
            raise ValueError(f"Невозможно загрузить изображение: {img_path}")

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

            # Шаг 5: Построение результатов
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

        # Экспорт Blueprint3D
        blueprint_result = self._export_blueprint3d(
            color_results=color_results,
            yolo_results=yolo_results,
            measurements=measurements,
            output_dir=output_dir,
            base_name=base_name
        )

        if blueprint_result and color_results:
            color_results['blueprint3d'] = blueprint_result

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
        """Визуализация результатов YOLO детекции."""
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Исходное')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        yolo_vis = self.yolo_detector.visualize_detections(img, yolo_results)
        ax2.imshow(cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title('Результаты YOLO')
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
        """Построение наложения визуализации со всеми результатами детекции."""
        overlay = img.copy()
        segments = color_results['segments']

        # Отрисовка стен
        if measurements:
            outer_wall_ids = {w['id'] for w in measurements.walls if w['is_outer']}
            if measurements.outer_crust is not None:
                crust_mask = measurements.outer_crust > 0
                overlay[crust_mask] = [0, 255, 0]
            for i, seg in enumerate(segments):
                is_outer = i in outer_wall_ids
                color = (0, 255, 0) if is_outer else (128, 128, 128)
                thickness = 3 if is_outer else 2
                cv2.line(overlay, (int(seg.x1), int(seg.y1)), (int(seg.x2), int(seg.y2)), color, thickness)
        else:
            for seg in segments:
                cv2.line(overlay, (int(seg.x1), int(seg.y1)), (int(seg.x2), int(seg.y2)), (0, 255, 0), 2)

        # Отрисовка алгоритмических окон (жёлтый)
        algo_windows = color_results.get('algo_windows', [])
        for win in algo_windows:
            sub = overlay[win.y:win.y+win.h, win.x:win.x+win.w]
            if sub.size > 0:
                white_rect = np.full_like(sub, (255, 255, 0))
                cv2.addWeighted(sub, 0.5, white_rect, 0.5, 0, sub)
            cv2.rectangle(overlay, (win.x, win.y), (win.x+win.w, win.y+win.h), (200, 200, 0), 2)

        # Отрисовка YOLO дверей (синий)
        for door in yolo_results.doors:
            x1, y1, x2, y2 = map(int, door.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Отрисовка YOLO окон (тонкий красный - для отладки)
        for window in yolo_results.windows:
            x1, y1, x2, y2 = map(int, window.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 100), 1)

        # Отрисовка проёмов (оранжевый)
        if 'openings' in color_results and color_results['openings']:
            for opening in color_results['openings']:
                if opening.is_valid:
                    cv2.rectangle(
                        overlay,
                        (opening.x1, opening.y1),
                        (opening.x2, opening.y2),
                        (0, 165, 255), 2
                    )

        # Добавление информации об измерениях
        if measurements:
            info_text = f"{measurements.area_m2:.1f} м², {measurements.pixel_scale * 100:.2f} см/пкс"
            cv2.putText(overlay, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(overlay, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return overlay

    def _save_overlay_image(self, img, yolo, color, path, meas):
        """Сохранение изображения с наложением."""
        overlay = self._build_overlay(img, yolo, color, meas)
        cv2.imwrite(path, overlay)

    def _visualize_combined_results(self, img, yolo, color, output_path, measurements):
        """Создание комбинированной визуализации результатов."""
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 3, figure=fig)

        # Исходное изображение
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Исходное')
        ax1.axis('off')

        # Наложение
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self._build_overlay(img, yolo, color, measurements)
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title('Наложение')
        ax2.axis('off')

        # Маска окон
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(color['algo_win_mask'], cmap='gray')
        ax3.set_title(f"Маска окон ({len(color.get('algo_windows', []))})")
        ax3.axis('off')

        # Жилая площадь
        ax4 = fig.add_subplot(gs[1, 0])
        if measurements and measurements.apartment_boundary is not None:
            ax4.imshow(measurements.apartment_boundary, cmap='gray')
            ax4.set_title('Граница квартиры')
        else:
            ax4.imshow(np.zeros_like(img[:, :, 0]), cmap='gray')
        ax4.axis('off')

        # Контур стен
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(color['outline_mask'], cmap='gray')
        ax5.set_title('Контур стен')
        ax5.axis('off')

        # Информационная панель
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')

        lines = [
            f"Двери (YOLO): {len(yolo.doors)}",
            f"Окна (CV-алго): {len(color.get('algo_windows', []))}",
            f"Сегменты: {len(color.get('segments', []))}"
        ]

        if measurements:
            lines.append(f"Площадь: {measurements.area_m2:.1f} м²")
            lines.append(f"Масштаб: {measurements.pixel_scale*100:.2f} см/пкс")

        if 'blueprint3d' in color:
            bp = color['blueprint3d']
            fp = bp.get('floorplan', {})
            lines.append("")
            lines.append("Экспорт Blueprint3D")
            lines.append(f"Углов: {len(fp.get('corners', {}))}")
            lines.append(f"Стен: {len(fp.get('walls', []))}")
            lines.append(f"Объектов: {len(bp.get('items', []))}")

            # Подсчёт дверей/окон в объектах
            items = bp.get('items', [])
            bp_windows = sum(1 for i in items if i.get('item_type') == 3)
            bp_doors = sum(1 for i in items if i.get('item_type') == 7)
            lines.append(f"  Окна: {bp_windows}")
            lines.append(f"  Двери: {bp_doors}")

        ax6.text(0.05, 0.95, "\n".join(lines), fontsize=10, va='top', family='monospace',
                 transform=ax6.transAxes)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()


# ============================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Система анализа планировок v3.9 (Blueprint3D с топологией графа стен)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py --input image.png --yolo-model weights.pt --output results/
  python main.py --input images/ --yolo-model weights.pt --wall-height 280 --snap-tolerance 15
        """
    )

    # Параметры ввода/вывода
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к изображению или директории')
    parser.add_argument('--output', type=str, default='output_hybrid',
                        help='Директория для результатов')

    # Конфигурация YOLO
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Путь к весам YOLO модели')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='Порог уверенности YOLO (по умолчанию: 0.25)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Устройство обработки: cpu или cuda')

    # Конфигурация детекции стен
    parser.add_argument('--wall-color', type=str, default='8E8780',
                        help='HEX цвет стен (по умолчанию: 8E8780)')
    parser.add_argument('--color-tolerance', type=int, default=3,
                        help='Допуск цвета RGB (по умолчанию: 3)')
    parser.add_argument('--auto-wall-color', action='store_true',
                        help='Автоматическое определение цвета стен')
    parser.add_argument('--skip-color', action='store_true',
                        help='Пропустить CV детекцию стен (только YOLO)')

    # Конфигурация измерений
    parser.add_argument('--disable-measurements', action='store_true',
                        help='Отключить измерения площади')
    parser.add_argument('--edge-distance-ratio', type=float, default=0.15,
                        help='Коэффициент для определения внешних стен (по умолчанию: 0.15)')
    parser.add_argument('--debug-ocr', action='store_true',
                        help='Включить отладку OCR')

    # Конфигурация детекции окон
    parser.add_argument('--min-window-len', type=int, default=15,
                        help='Минимальная длина окна для детекции (по умолчанию: 15)')

    # Конфигурация расширения краёв стен
    parser.add_argument('--edge-expansion-tolerance', type=int, default=8,
                        help='Допуск цвета для расширения краёв стен (по умолчанию: 8)')
    parser.add_argument('--max-edge-expansion', type=int, default=10,
                        help='Максимальное расширение краёв стен в пикселях (по умолчанию: 10)')

    # Визуализация
    parser.add_argument('--no-vis', action='store_true',
                        help='Отключить визуализацию результатов')

    # Параметры экспорта Blueprint3D
    bp3d_group = parser.add_argument_group('Экспорт Blueprint3D')
    bp3d_group.add_argument('--disable-blueprint3d', action='store_true',
                            help='Отключить экспорт Blueprint3D')
    bp3d_group.add_argument('--no-skip-bbox-check', action='store_true',
                            help='Включить проверку bbox в детекторе проёмов')
    bp3d_group.add_argument('--window-width', type=float, default=48.0,
                            help='Ширина окна по умолчанию в дюймах (по умолчанию: 48)')
    bp3d_group.add_argument('--door-width', type=float, default=65.0,
                            help='Ширина двери по умолчанию в дюймах (по умолчанию: 65)')
    bp3d_group.add_argument('--wall-height', type=float, default=300.0,
                            help='Высота стен в сантиметрах (по умолчанию: 300)')
    bp3d_group.add_argument('--snap-tolerance', type=float, default=10.0,
                            help='Допуск привязки углов в пикселях (по умолчанию: 10)')

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
        raise ValueError(f"Путь не существует: {args.input}")

    if not image_files:
        raise ValueError(f"Изображения не найдены: {args.input}")

    # Создание анализатора
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
        enable_blueprint3d_export=not args.disable_blueprint3d,
        skip_bbox_check=not args.no_skip_bbox_check,
        default_window_width_inches=args.window_width,
        default_door_width_inches=args.door_width,
        wall_height_cm=args.wall_height,
        corner_snap_tolerance_px=args.snap_tolerance
    )

    # Обработка изображений
    results = []
    for idx, img_path in enumerate(image_files, 1):
        try:
            result = analyzer.process_image(
                img_path,
                args.output,
                run_color_detection=not args.skip_color
            )
            results.append({
                'path': img_path,
                'success': True,
                'result': result
            })
        except Exception as e:
            results.append({
                'path': img_path,
                'success': False,
                'error': str(e)
            })

    # Генерация итогов
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nОбработка завершена: {successful}/{len(results)} изображений успешно")
    if failed > 0:
        print(f"Ошибки: {failed} изображений")
    print(f"Результаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()