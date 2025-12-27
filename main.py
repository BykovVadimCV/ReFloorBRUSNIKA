"""
Быков Вадим Олегович - 19.12.2025

Гибридный анализ стен (CV) и дверей/окон (YOLO) с измерением размеров - v3.1 (compact)

Изменения v1.3:
- Упрощённый, минималистичный вывод в консоль
- Визуализации по желанию (--no-vis)
- Одна компактная сводная фигура на изображение (2x3 сетка)
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
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Импорт внутренних модулей
from detection.walldetector import ColorBasedWallDetector
from detection.furnituredetector import YOLODoorWindowDetector, YOLOResults
from space import RoomSizeAnalyzer, RoomMeasurements
from floorplanexporter import FloorplanBuilder


# ------------------------------------------------------------------ #
# ГИБРИДНЫЙ АНАЛИЗАТОР С ИЗМЕРЕНИЯМИ
# ------------------------------------------------------------------ #
class HybridWallAnalyzer:
    """Гибридный анализатор стен и проёмов с измерением размеров."""

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
                 enable_visuals: bool = True):
        print("[INIT] Гибридный анализатор v3.1")
        print(f"[INIT] Модель YOLO={Path(yolo_model_path).name}, устройство={device}")
        if auto_wall_color:
            print(f"[INIT] Стены: автоматический цвет, допуск=±{color_tolerance}")
        else:
            print(f"[INIT] Стены: #{wall_color_hex} ±{color_tolerance}")
        print(f"[INIT] Измерения={'ВКЛ' if enable_measurements else 'ВЫКЛ'}, "
              f"визуализации={'ВКЛ' if enable_visuals else 'ВЫКЛ'}")

        self.yolo_detector = YOLODoorWindowDetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            device=device
        )
        self.color_detector = ColorBasedWallDetector(
            wall_color_hex=wall_color_hex,
            color_tolerance=color_tolerance,
            min_wall_length=10,
            hough_threshold=10,
            auto_wall_color=auto_wall_color
        )

        self.enable_measurements = enable_measurements
        self.enable_visuals = enable_visuals
        self.size_analyzer = None
        self.floorplan_builder = None

        if enable_measurements:
            self.size_analyzer = RoomSizeAnalyzer(
                edge_distance_ratio=edge_distance_ratio,
                debug_ocr=debug_ocr
            )
            self.floorplan_builder = FloorplanBuilder(
                wall_height_m=3.0,
                node_merge_tol_pixels=1.5,
                min_wall_length_m=0.2,
            )
        print("[INIT] Инициализация завершена\n")

    def _convert_segments_to_wall_segments(self, segments) -> List[Dict]:
        """Преобразование сегментов детектора в список стен с метаданными."""
        wall_segments = []
        for i, seg in enumerate(segments):
            length = np.sqrt((seg.x2 - seg.x1) ** 2 + (seg.y2 - seg.y1) ** 2)
            wall_segments.append({
                'id': i,
                'x1': seg.x1,
                'y1': seg.y1,
                'x2': seg.x2,
                'y2': seg.y2,
                'length_pixels': length,
                'thickness': float(getattr(seg, "thickness_mean", 0.0))
            })
        return wall_segments

    def _refine_living_space_with_doors(
        self,
        measurements: RoomMeasurements,
        wall_mask: np.ndarray,
        yolo_results: YOLOResults,
        non_interior_threshold: float = 0.02,  # 2 %
    ) -> None:
        """
        Если в bbox двери больше non_interior_threshold пикселей лежит
        ВНЕ интерьера (apartment_boundary), весь bbox двери удаляется
        из интерьера (из apartment_boundary), и площадь пересчитывается.
        """
        if yolo_results is None or not getattr(yolo_results, "doors", None):
            return

        ab = getattr(measurements, "apartment_boundary", None)
        if ab is None:
            return

        # Приводим boundary к одноканальной маске
        if ab.ndim == 3:
            ab_gray = cv2.cvtColor(ab, cv2.COLOR_BGR2GRAY)
        else:
            ab_gray = ab

        if ab_gray.shape[:2] != wall_mask.shape[:2]:
            return

        interior_mask = ab_gray > 0
        h, w = interior_mask.shape
        remove_mask = np.zeros_like(interior_mask, dtype=bool)

        for det in yolo_results.doors:
            x1, y1, x2, y2 = det.bbox

            # Нормализуем координаты bbox
            x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
            y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))

            # Ограничиваем рамками изображения
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w,     x2))   # правая граница как end-index
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h,     y2))   # нижняя граница как end-index

            if x2 <= x1 or y2 <= y1:
                continue

            sub = interior_mask[y1:y2, x1:x2]
            total_px = sub.size
            if total_px == 0:
                continue

            interior_px = int(sub.sum())
            non_interior_px = total_px - interior_px
            frac_non_interior = non_interior_px / float(total_px)

            if frac_non_interior > non_interior_threshold:
                remove_mask[y1:y2, x1:x2] = True

        if not remove_mask.any():
            return

        new_ab = ab_gray.copy()
        new_ab[remove_mask] = 0
        measurements.apartment_boundary = new_ab

        # Пересчёт площади по обновлённому контуру
        px_scale = float(getattr(measurements, "pixel_scale", 0.0) or 0.0)
        if px_scale > 0:
            measurements.area_m2 = float((new_ab > 0).sum()) * (px_scale ** 2)

    def process_image(self, img_path: str, output_dir: str,
                      run_color_detection: bool = True):
        """Обработка одного изображения: стены (цвет), двери/окна (YOLO), измерения."""
        img_name = Path(img_path).name
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Невозможно загрузить: {img_path}")
        h, w = img.shape[:2]

        # Детекция дверей/окон (YOLO)
        yolo_results = self.yolo_detector.detect(img, verbose=False)

        color_results = None
        measurements: Optional[RoomMeasurements] = None

        if run_color_detection:
            wall_mask, segments, junctions, dist_transform = self.color_detector.process(img)

            # --- ОКНА СЧИТАЕМ СТЕНАМИ + РАСШИРЯЕМ НА 3 PX С КАЖДОЙ СТОРОНЫ ---
            if (
                yolo_results is not None and
                getattr(yolo_results, "combined_window_mask", None) is not None
            ):
                win_mask = yolo_results.combined_window_mask
                if win_mask.shape[:2] == wall_mask.shape[:2]:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                    win_mask_expanded = cv2.dilate(win_mask, kernel, iterations=1)
                    wall_mask = wall_mask.copy()
                    wall_mask[win_mask_expanded > 0] = 255
            # ----------------------------------------------------------------

            color_results = {
                'wall_mask': wall_mask,
                'segments': segments,
                'junctions': junctions,
                'dist_transform': dist_transform
            }

            if self.enable_measurements and segments:
                try:
                    wall_segments_data = self._convert_segments_to_wall_segments(segments)
                    measurements = self.size_analyzer.process(
                        img_path,
                        wall_segments_data,
                        output_dir,
                        wall_mask=wall_mask,      # уже с учётом (расширённых) окон как стен
                        original_image=img
                    )
                except Exception as e:
                    print(f"[MEASURE] Ошибка измерений: {e}")
                    import traceback
                    traceback.print_exc()

        # Постобработка жилой площади по дверям (2% правило)
        if measurements is not None and color_results is not None:
            try:
                self._refine_living_space_with_doors(
                    measurements=measurements,
                    wall_mask=color_results['wall_mask'],
                    yolo_results=yolo_results,
                    non_interior_threshold=0.02,
                )
            except Exception as e:
                print(f"[POST] Ошибка пост-обработки жилой площади: {e}")

        base_name = Path(img_path).stem

        # Визуализации (одна сводная фигура + опциональный оверлей)
        if self.enable_visuals:
            if run_color_detection and color_results:
                summary_path = os.path.join(output_dir, f"{base_name}_summary.png")
                self._visualize_combined_results(
                    img, yolo_results, color_results, summary_path, measurements
                )
                overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                self._save_overlay_image(img, yolo_results, color_results, overlay_path, measurements)
            else:
                # Если цветовая детекция выключена, рисуем только YOLO
                yolo_vis_path = os.path.join(output_dir, f"{base_name}_yolo.png")
                self._visualize_yolo_results(img, yolo_results, yolo_vis_path)

        # Сохранение масок
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(os.path.join(masks_dir, f"{base_name}_yolo_doors.png"), yolo_results.combined_door_mask)
        cv2.imwrite(os.path.join(masks_dir, f"{base_name}_yolo_windows.png"), yolo_results.combined_window_mask)

        if run_color_detection and color_results:
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_color_walls.png"), color_results['wall_mask'])
            if measurements is not None:
                if measurements.outer_crust is not None:
                    cv2.imwrite(os.path.join(masks_dir, f"{base_name}_outer_crust.png"), measurements.outer_crust)
                if measurements.apartment_boundary is not None:
                    cv2.imwrite(os.path.join(masks_dir, f"{base_name}_apartment_boundary.png"),
                                measurements.apartment_boundary)

        # Экспорт геометрии стен и floorplan
        if measurements is not None and color_results:
            json_dir = os.path.join(output_dir, 'json')
            os.makedirs(json_dir, exist_ok=True)

            # Экспорт floorplan в формате редактора
            if self.floorplan_builder is not None:
                floorplan_path = os.path.join(json_dir, f"{base_name}_floorplan.json")
                try:
                    floorplan_data = self.floorplan_builder.build_floorplan(
                        measurements=measurements,
                        yolo_results=yolo_results,
                        room_name="A New Room"
                    )
                    with open(floorplan_path, "w", encoding="utf-8") as f_fp:
                        json.dump(floorplan_data, f_fp, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[WARN] Ошибка экспорта планировки: {e}")

        # Краткое резюме по изображению
        doors_n = len(yolo_results.doors)
        windows_n = len(yolo_results.windows)
        seg_n = len(color_results['segments']) if color_results else 0
        if measurements:
            print(f"[IMG] {img_name} | двери={doors_n}, окна={windows_n}, сегменты={seg_n}, "
                  f"площадь={measurements.area_m2:.1f} м², "
                  f"масштаб={measurements.pixel_scale * 100:.2f} см/px")
        else:
            print(f"[IMG] {img_name} | двери={doors_n}, окна={windows_n}, сегменты={seg_n}, площадь=N/A")

        return {
            'yolo': yolo_results,
            'color': color_results,
            'measurements': measurements
        }

    # ------------------------------------------------------------------ #
    # ВИЗУАЛИЗАЦИЯ
    # ------------------------------------------------------------------ #
    def _visualize_yolo_results(self, img: np.ndarray, yolo_results: YOLOResults, output_path: str):
        """Минималистичная фигура: исходное изображение + YOLO + маски."""
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Исходное', fontsize=10)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        yolo_vis = self.yolo_detector.visualize_detections(img, yolo_results)
        ax2.imshow(cv2.cvtColor(yolo_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'YOLO (d={len(yolo_results.doors)}, w={len(yolo_results.windows)})', fontsize=10)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(yolo_results.combined_door_mask, cmap='gray')
        ax3.set_title('Маска дверей', fontsize=10)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(yolo_results.combined_window_mask, cmap='gray')
        ax4.set_title('Маска окон', fontsize=10)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        class_map = np.zeros(img.shape[:2], dtype=np.uint8)
        class_map[yolo_results.combined_door_mask > 0] = 1
        class_map[yolo_results.combined_window_mask > 0] = 2
        cmap_custom = ListedColormap(['black', 'blue', 'green'])
        ax5.imshow(class_map, cmap=cmap_custom, vmin=0, vmax=2)
        legend_elements = [
            mpatches.Patch(color='black', label='BG'),
            mpatches.Patch(color='blue', label='Doors'),
            mpatches.Patch(color='green', label='Windows')
        ]
        ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)
        ax5.set_title('Классы', fontsize=10)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        ax6.text(0.01, 0.99,
                 f"doors={len(yolo_results.doors)}\n"
                 f"windows={len(yolo_results.windows)}",
                 fontsize=9, va='top')
        ax6.set_title('Сводка', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

    def _build_overlay(self, img: np.ndarray, yolo_results: YOLOResults,
                       color_results: Dict,
                       measurements: Optional[RoomMeasurements]) -> np.ndarray:
        """Создание оверлея (формат cv2 BGR) без сохранения на диск."""
        overlay = img.copy()
        segments = color_results['segments']
        junctions = color_results['junctions']

        if measurements:
            outer_wall_ids = {w['id'] for w in measurements.walls if w['is_outer']}
            if measurements.outer_crust is not None:
                crust_mask = measurements.outer_crust > 0
                overlay[crust_mask] = [0, 255, 0]
            for i, seg in enumerate(segments):
                is_outer = i in outer_wall_ids
                color = (0, 255, 0) if is_outer else (128, 128, 128)
                thickness = 3 if is_outer else 2
                cv2.line(overlay, (seg.x1, seg.y1), (seg.x2, seg.y2), color, thickness)
        else:
            for seg in segments:
                cv2.line(overlay, (seg.x1, seg.y1), (seg.x2, seg.y2), (0, 255, 0), 2)

        for junc in junctions:
            cv2.circle(overlay, (junc.x, junc.y), 4, (255, 0, 255), -1)

        for door in yolo_results.doors:
            x1, y1, x2, y2 = map(int, door.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for window in yolo_results.windows:
            x1, y1, x2, y2 = map(int, window.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if measurements:
            info_text = f"{measurements.area_m2:.1f} m², {measurements.pixel_scale * 100:.2f} cm/px"
            cv2.putText(overlay, info_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(overlay, info_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return overlay

    def _save_overlay_image(self, img: np.ndarray, yolo_results: YOLOResults,
                            color_results: Dict, output_path: str,
                            measurements: Optional[RoomMeasurements] = None):
        """Сохранение оверлея с результатами детекции."""
        overlay = self._build_overlay(img, yolo_results, color_results, measurements)
        cv2.imwrite(output_path, overlay)

    def _visualize_combined_results(self, img: np.ndarray, yolo_results: YOLOResults,
                                    color_results: Dict, output_path: str,
                                    measurements: Optional[RoomMeasurements] = None):
        """Компактная сводная фигура 2x3 по одному изображению."""
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 3, figure=fig)

        # 1: исходное
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Исходное', fontsize=10)
        ax1.axis('off')

        # 2: оверлей (стены + двери/окна)
        ax2 = fig.add_subplot(gs[0, 1])
        overlay = self._build_overlay(img, yolo_results, color_results, measurements)
        ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax2.set_title('Стены + двери/окна', fontsize=10)
        ax2.axis('off')

        # 3: маска стен
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(color_results['wall_mask'], cmap='gray')
        ax3.set_title('Маска стен', fontsize=10)
        ax3.axis('off')

        # 4: жилая площадь / контур квартиры
        ax4 = fig.add_subplot(gs[1, 0])
        if measurements and measurements.apartment_boundary is not None:
            comp = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
            living_mask = (measurements.apartment_boundary > 0) & (color_results['wall_mask'] == 0)
            comp[living_mask] = [255, 220, 180]
            comp[color_results['wall_mask'] > 0] = [100, 100, 100]
            if measurements.outer_crust is not None:
                comp[measurements.outer_crust > 0] = [0, 255, 0]
            ax4.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
            ax4.set_title('Жилая площадь / контур', fontsize=10)
        else:
            ax4.imshow(np.zeros_like(img[:, :, 0]), cmap='gray')
            ax4.set_title('Жилая площадь (N/A)', fontsize=10)
        ax4.axis('off')

        # 5: outer crust
        ax5 = fig.add_subplot(gs[1, 1])
        if measurements and measurements.outer_crust is not None:
            ax5.imshow(measurements.outer_crust, cmap='Greens')
            ax5.set_title('Outer crust', fontsize=10)
        else:
            ax5.imshow(np.zeros_like(img[:, :, 0]), cmap='gray')
            ax5.set_title('Outer crust (N/A)', fontsize=10)
        ax5.axis('off')

        # 6: текстовая сводка
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        lines = []

        doors = len(yolo_results.doors)
        windows = len(yolo_results.windows)
        lines.append(f"doors={doors}, windows={windows}")

        wall_mask = color_results['wall_mask']
        total_px = wall_mask.size
        wall_coverage = (wall_mask > 0).sum() / total_px * 100
        lines.append(f"wall px={wall_coverage:.1f}%")

        if measurements:
            lines.append(f"area={measurements.area_m2:.1f} m²")
            lines.append(f"scale={measurements.pixel_scale * 100:.2f} cm/px")
            outer = [w_ for w_ in measurements.walls if w_['is_outer']]
            inner = [w_ for w_ in measurements.walls if not w_['is_outer']]
            lines.append(f"walls: {len(outer)} out / {len(inner)} in")

        ax6.text(0.01, 0.99, "\n".join(lines), fontsize=9, va='top')
        ax6.set_title('Сводка', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------ #
    # ВСПОМОГАТЕЛЬНЫЕ МЕТРИКИ ДЛЯ CLI (НЕ ИСПОЛЬЗУЮТСЯ В КОМПАКТНОМ РЕЖИМЕ)
    # ------------------------------------------------------------------ #
    def _plot_comparison_metrics(self, ax, yolo_results, color_results, measurements=None):
        # Не используется в компактном режиме, оставлено для возможной отладки
        ax.axis('off')


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Гибридный анализатор с измерениями стен v3.1 "
                    "(внешний контур, компактный режим)"
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Путь к изображению или директории с изображениями')
    parser.add_argument('--output', type=str, default='output_hybrid',
                        help='Директория для сохранения результатов')
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Путь к файлу весов YOLOv9 (.pt)')
    parser.add_argument('--wall-color', type=str, default='8E8780',
                        help='HEX-цвет стен для цветовой детекции')
    parser.add_argument('--color-tolerance', type=int, default=3,
                        help='Толерантность цвета RGB')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='Порог уверенности YOLO')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Устройство для YOLO: cpu/cuda/0/1/...')
    parser.add_argument('--skip-color', action='store_true',
                        help='Пропустить цветовую детекцию (только YOLO двери/окна)')
    parser.add_argument('--auto-wall-color', action='store_true',
                        help='Автоматически определить цвет стен (тёмный доминант >3% пикселей)')
    parser.add_argument('--disable-measurements', action='store_true',
                        help='Отключить измерения размеров')
    parser.add_argument('--edge-distance-ratio', type=float, default=0.15,
                        help='Доля от края изображения для определения внешних стен (0.01–0.5)')
    parser.add_argument('--debug-ocr', action='store_true',
                        help='Отладка OCR (вывод всех найденных текстов)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Не сохранять визуализации (фигуры и оверлей)')

    args = parser.parse_args()

    if not 0.01 <= args.edge_distance_ratio <= 0.5:
        args.edge_distance_ratio = max(0.01, min(0.5, args.edge_distance_ratio))

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [str(input_path)]
    elif input_path.is_dir():
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend([str(f) for f in input_path.glob(ext)])
    else:
        raise ValueError(f"Путь не существует: {args.input}")

    print("=" * 60)
    print(f"Изображений: {len(image_files)} | YOLO={Path(args.yolo_model).name} | "
          f"устройство={args.device}")
    if args.auto_wall_color:
        wall_info = "авто"
    else:
        wall_info = f"#{args.wall_color}±{args.color_tolerance}"
    print(f"стены={wall_info} | измерения={'ВКЛ' if not args.disable_measurements else 'ВЫКЛ'} "
          f"| цветовая детекция={'ВЫКЛ' if args.skip_color else 'ВКЛ'} "
          f"| визуализации={'ВЫКЛ' if args.no_vis else 'ВКЛ'}")
    print("=" * 60)

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
        enable_visuals=not args.no_vis
    )

    results = []
    for idx, img_path in enumerate(sorted(image_files), 1):
        print(f"[RUN] {idx}/{len(image_files)} :: обработка {Path(img_path).name}")
        try:
            result = analyzer.process_image(
                img_path,
                args.output,
                run_color_detection=not args.skip_color
            )
            results.append({'path': img_path, 'success': True, 'result': result})
        except Exception as e:
            print(f"[ERR] Ошибка при обработке {Path(img_path).name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'path': img_path, 'success': False, 'error': str(e)})

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    measurements_done = 0
    total_area = 0.0
    for r in results:
        if r['success'] and r['result'].get('measurements'):
            m = r['result']['measurements']
            measurements_done += 1
            total_area += m.area_m2

    print("=" * 60)
    print(f"Готово: успешно={successful}, с ошибками={failed}")
    if measurements_done > 0:
        print(f"Измерения: {measurements_done} изображений, "
              f"средняя площадь={total_area / measurements_done:.1f} м²")
    print(f"Результаты сохранены в: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()