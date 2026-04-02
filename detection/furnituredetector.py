"""
Система анализа планировок помещений - Модуль YOLO-детекции
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

Модуль YOLO-детекции и экспорта масок мебели. На данный момент
можуль нацелен исключительно на обнаружение дверей и окон.
"""

import cv2
import numpy as np
import torch
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# НАСТРОЙКА ПУТЕЙ YOLO
# ============================================================

current_dir = os.getcwd()
yolo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'yolov9')

if os.path.exists(yolo_path):
    if yolo_path not in sys.path:
        sys.path.append(yolo_path)

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
except ImportError as e:
    raise ImportError(f"Не удалось импортировать модули YOLOv9. Проверьте путь и структуру репозитория. Ошибка: {e}")


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class YOLODetection:
    """
    Структура данных для одной YOLO детекции.

    Атрибуты:
        class_id: Идентификатор класса (0 - дверь, 1 - окно)
        class_name: Название класса
        confidence: Уверенность детекции (0-1)
        bbox: Ограничивающий прямоугольник (x1, y1, x2, y2)
        mask: Бинарная маска объекта (опционально)
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None


@dataclass
class YOLOResults:
    """
    Структура данных для результатов YOLO детекции.

    Атрибуты:
        doors: Список обнаруженных дверей
        windows: Список обнаруженных окон
        all_detections: Все детекции
        combined_door_mask: Объединённая маска всех дверей
        combined_window_mask: Объединённая маска всех окон
    """
    doors: List[YOLODetection]
    windows: List[YOLODetection]
    all_detections: List[YOLODetection]
    combined_door_mask: np.ndarray
    combined_window_mask: np.ndarray


# ============================================================
# ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ
# ============================================================

def letterbox(im, new_shape=(1048, 1048), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Изменение размера изображения с сохранением пропорций и добавлением отступов.

    Параметры:
        im: Входное изображение
        new_shape: Целевой размер (высота, ширина)
        color: Цвет заполнения отступов
        auto: Автоматический расчёт отступов
        scaleFill: Заполнение всей области без отступов
        scaleup: Разрешить увеличение изображения
        stride: Шаг для выравнивания размера

    Возвращает:
        Кортеж (изображение с отступами, коэффициент масштабирования, отступы)
    """
    shape = im.shape[:2]  # Текущая форма [высота, ширина]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Расчёт коэффициента масштабирования
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # Расчёт отступов
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    # Изменение размера при необходимости
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Добавление отступов
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


# ============================================================
# ДЕТЕКТОР
# ============================================================

def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class YOLODoorWindowDetector:
    """
    YOLO-детектор дверей и окон на базе нативного YOLOv9.

    Поддерживает детекцию двух классов объектов:
    - Двери (класс 0)
    - Окна (класс 1)
    """

    # Сопоставление ID классов с названиями
    CLASS_NAMES = {
        0: "Door",
        1: "Window"
    }

    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "cpu",
                 max_door_area_fraction: float = 0.05):
        """
        Инициализация YOLO-детектора.

        Параметры:
            model_path: Путь к файлу весов модели (.pt)
            conf_threshold: Порог уверенности для детекции
            iou_threshold: Порог IoU для NMS (Non-Maximum Suppression)
            device: Устройство для инференса (cpu/cuda/0/1/...)
            max_door_area_fraction: Maximum fraction of total image area that a
                single door or window bbox may occupy.  Detections exceeding this
                threshold are almost certainly whole-room false positives produced
                when YOLO pattern-matches an enclosed corridor or hallway to a wide
                doorway.  At typical Brusnika floor-plan scale a real door leaf
                occupies < 2 % of the image; the default of 5 % gives comfortable
                headroom while still killing corridor-sized false positives that
                typically span 10–20 % of the image.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_door_area_fraction = max_door_area_fraction
        self.device = select_device(device)

        # Проверка существования файла модели
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        try:
            # Загрузка модели
            self.model = DetectMultiBackend(str(self.model_path), device=self.device, dnn=False)
            self.stride = self.model.stride
            self.names = self.model.names
            self.pt = self.model.pt
            self.img_size = 640

            # Прогрев модели для оптимизации производительности
            self.model.warmup(imgsz=(1, 3, self.img_size, self.img_size))

        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    @staticmethod
    def _box_nms(detections: List[YOLODetection], iou_thresh: float = 0.40) -> List[YOLODetection]:
        """
        Per-class greedy NMS sorted by descending box area.
        Removes duplicate YOLO detections of the same physical opening
        that survived the model's own cross-class NMS pass.
        """
        if len(detections) <= 1:
            return detections
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )
        kept: List[YOLODetection] = []
        kept_boxes: List[Tuple[float, float, float, float]] = []
        for d in sorted_dets:
            b = (float(d.bbox[0]), float(d.bbox[1]),
                 float(d.bbox[2]), float(d.bbox[3]))
            if all(_iou_xyxy(b, kb) < iou_thresh for kb in kept_boxes):
                kept.append(d)
                kept_boxes.append(b)
        return kept

    def detect(self, img: np.ndarray, verbose: bool = False) -> YOLOResults:
        """
        Детекция дверей и окон на изображении.

        Параметры:
            img: Входное изображение в формате BGR
            verbose: Флаг вывода отладочной информации

        Возвращает:
            YOLOResults с результатами детекции

        Исключения:
            ValueError: При пустом или некорректном изображении
        """
        # Валидация входного изображения
        if img is None or img.size == 0:
            raise ValueError("Пустое изображение")

        h0, w0 = img.shape[:2]

        # Этап 1: Предобработка изображения
        img_input, ratio, pad = letterbox(img, self.img_size, stride=self.stride, auto=True)
        img_input = img_input.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
        img_input = np.ascontiguousarray(img_input)

        # Этап 2: Преобразование в тензор PyTorch
        im = torch.from_numpy(img_input).to(self.device)
        del img_input  # free preprocessed numpy array
        im = im.half() if self.model.fp16 else im.float()
        im /= 255.0  # Нормализация к [0, 1]
        if len(im.shape) == 3:
            im = im[None]  # Добавление batch dimension

        # Этап 3: Инференс модели (no_grad — skip autograd, saves ~2x RAM)
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)

        # Этап 4: Non-Maximum Suppression
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=None, agnostic=False)

        # Этап 5: Обработка результатов
        all_detections = []
        doors = []
        windows = []
        combined_door_mask = np.zeros((h0, w0), dtype=np.uint8)
        combined_window_mask = np.zeros((h0, w0), dtype=np.uint8)

        det = pred[0]
        im_shape = im.shape[2:]  # save shape before freeing tensor
        del im, pred  # free GPU/CPU tensor memory

        if len(det):
            # Масштабирование координат bbox к исходному размеру изображения
            det[:, :4] = scale_boxes(im_shape, det[:, :4], (h0, w0)).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                conf_val = float(conf)
                class_id = int(cls)

                # Ограничение координат размерами изображения
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)

                # Пропуск невалидных bbox
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Создание объекта детекции (no per-detection mask to save RAM)
                detection = YOLODetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf_val,
                    bbox=(x1, y1, x2, y2),
                    mask=None
                )

                all_detections.append(detection)

                # Build bbox mask for combined masks
                bbox_mask = np.zeros((h0, w0), dtype=np.uint8)
                cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

                # Классификация по типу объекта
                if class_id == 0:  # Дверь
                    doors.append(detection)
                    combined_door_mask = cv2.bitwise_or(combined_door_mask, bbox_mask)
                elif class_id == 1:  # Окно
                    windows.append(detection)
                    combined_window_mask = cv2.bitwise_or(combined_window_mask, bbox_mask)

        doors = self._box_nms(doors)
        windows = self._box_nms(windows)

        # ── Physical-size gate ────────────────────────────────────────────
        # Reject any detection whose bbox area exceeds max_door_area_fraction
        # of the full image.  Whole-room false positives (e.g. a narrow corridor
        # that YOLO mistakes for a wide door) always produce oversized boxes,
        # whereas genuine door/window leaves stay well under this limit.
        img_area = h0 * w0
        def _oversized(det: YOLODetection) -> bool:
            bw = det.bbox[2] - det.bbox[0]
            bh = det.bbox[3] - det.bbox[1]
            return (bw * bh) / img_area > self.max_door_area_fraction

        doors   = [d for d in doors   if not _oversized(d)]
        windows = [w for w in windows if not _oversized(w)]
        # ─────────────────────────────────────────────────────────────────

        all_detections = doors + windows

        return YOLOResults(
            doors=doors,
            windows=windows,
            all_detections=all_detections,
            combined_door_mask=combined_door_mask,
            combined_window_mask=combined_window_mask
        )

    def visualize_detections(self,
                             img: np.ndarray,
                             results: YOLOResults,
                             show_boxes: bool = True,
                             show_masks: bool = True,
                             show_labels: bool = True) -> np.ndarray:
        """
        Визуализация результатов детекции на изображении.

        Параметры:
            img: Исходное изображение
            results: Результаты детекции от метода detect()
            show_boxes: Отображать ли ограничивающие прямоугольники
            show_masks: Отображать ли маски объектов
            show_labels: Отображать ли текстовые подписи

        Возвращает:
            Изображение с наложенными визуализациями детекций
        """
        vis_img = img.copy()

        # Цветовая схема для разных классов
        colors = {
            0: (0, 255, 0),   # Зелёный для дверей
            1: (255, 0, 0)    # Синий для окон
        }

        # Наложение масок с прозрачностью
        if show_masks:
            mask_overlay = np.zeros_like(vis_img)
            for detection in results.all_detections:
                if detection.mask is not None:
                    color = colors.get(detection.class_id, (128, 128, 128))
                    mask_overlay[detection.mask > 0] = color

            alpha = 0.4  # Прозрачность масок
            vis_img = cv2.addWeighted(vis_img, 1 - alpha, mask_overlay, alpha, 0)

        # Отрисовка ограничивающих прямоугольников и подписей
        for detection in results.all_detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(detection.class_id, (128, 128, 128))

            # Отрисовка прямоугольника
            if show_boxes:
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Отрисовка подписи с фоном
            if show_labels:
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Фон для текста
                cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)

                # Текст
                cv2.putText(vis_img, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_img