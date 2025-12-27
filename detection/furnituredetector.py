"""
Быков Вадим Олегович - 16.12.2025

Модуль YOLO-детекции дверей и окон - v1.1

ПАЙПЛАЙН:
- Загрузка модели через models.common.DetectMultiBackend
- Ручной препроцессинг (Letterbox) и инференс (PyTorch)
- Детекция дверей (0) и окон (1)
- Постобработка (NMS, Scaling)

СТЕК:
- PyTorch: инференс
- OpenCV: обработка изображений
- NumPy: массивы

Последнее изменение - 22.12.25 - Refactored silent operation
"""

import cv2
import numpy as np
import torch
import sys
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path


# ------------------------------------------------------------------ #
# YOLO PATH SETUP
# ------------------------------------------------------------------ #

current_dir = os.getcwd()
yolo_path = os.path.join(current_dir, 'utils', 'yolov9')

if os.path.exists(yolo_path):
    if yolo_path not in sys.path:
        sys.path.append(yolo_path)

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
except ImportError as e:
    raise ImportError(f"Не удалось импортировать модули YOLOv9. Проверьте путь и структуру репозитория. Ошибка: {e}")


# ------------------------------------------------------------------ #
# DATA STRUCTURES
# ------------------------------------------------------------------ #

@dataclass
class YOLODetection:
    """Структура данных для одной YOLO детекции"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None


@dataclass
class YOLOResults:
    """Структура данных для результатов YOLO детекции"""
    doors: List[YOLODetection]
    windows: List[YOLODetection]
    all_detections: List[YOLODetection]
    combined_door_mask: np.ndarray
    combined_window_mask: np.ndarray


# ------------------------------------------------------------------ #
# PREPROCESSING
# ------------------------------------------------------------------ #

def letterbox(im, new_shape=(1048, 1048), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Изменение размера изображения с сохранением пропорций"""
    shape = im.shape[:2]  # текущая форма [высота, ширина]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


# ------------------------------------------------------------------ #
# DETECTOR
# ------------------------------------------------------------------ #

class YOLODoorWindowDetector:
    """YOLO-детектор дверей и окон на базе нативного YOLOv9"""

    CLASS_NAMES = {
        0: "Door",
        1: "window"
    }

    def __init__(self,
                 model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = "cpu"):
        """
        Инициализация YOLO-детектора
        
        Параметры:
            model_path: путь к файлу весов модели (.pt)
            conf_threshold: порог уверенности для детекции
            iou_threshold: порог IoU для NMS
            device: устройство для инференса (cpu/cuda/0/1/...)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = select_device(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        try:
            self.model = DetectMultiBackend(str(self.model_path), device=self.device, dnn=False)
            self.stride = self.model.stride
            self.names = self.model.names
            self.pt = self.model.pt
            self.img_size = 640

            # Прогрев модели
            self.model.warmup(imgsz=(1, 3, self.img_size, self.img_size))

        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    def detect(self, img: np.ndarray, verbose: bool = False) -> YOLOResults:
        """
        Детекция дверей и окон на изображении
        
        Параметры:
            img: входное изображение (BGR)
            verbose: флаг вывода информации в консоль
            
        Возвращает:
            YOLOResults с результатами детекции
        """
        if img is None or img.size == 0:
            raise ValueError("Пустое изображение")

        original_img = img.copy()
        h0, w0 = img.shape[:2]

        # Препроцессинг
        img_input, ratio, pad = letterbox(img, self.img_size, stride=self.stride, auto=True)
        img_input = img_input.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img_input = np.ascontiguousarray(img_input)

        # Конвертация в тензор
        im = torch.from_numpy(img_input).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        # Инференс
        pred = self.model(im, augment=False, visualize=False)

        # NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=None, agnostic=False)

        # Обработка результатов
        all_detections = []
        doors = []
        windows = []
        combined_door_mask = np.zeros((h0, w0), dtype=np.uint8)
        combined_window_mask = np.zeros((h0, w0), dtype=np.uint8)

        det = pred[0]

        if len(det):
            # Масштабирование bbox к исходному размеру
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], (h0, w0)).round()

            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                conf_val = float(conf)
                class_id = int(cls)

                # Валидация координат
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w0, x2), min(h0, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = self.CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Создание маски из bbox
                bbox_mask = np.zeros((h0, w0), dtype=np.uint8)
                bbox_mask[y1:y2, x1:x2] = 255

                detection = YOLODetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf_val,
                    bbox=(x1, y1, x2, y2),
                    mask=bbox_mask
                )

                all_detections.append(detection)

                if class_id == 0:  # Дверь
                    doors.append(detection)
                    combined_door_mask = cv2.bitwise_or(combined_door_mask, bbox_mask)
                elif class_id == 1:  # Окно
                    windows.append(detection)
                    combined_window_mask = cv2.bitwise_or(combined_window_mask, bbox_mask)

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
        Визуализация результатов детекции
        
        Параметры:
            img: исходное изображение
            results: результаты детекции
            show_boxes: отображать ли bbox
            show_masks: отображать ли маски
            show_labels: отображать ли подписи
            
        Возвращает:
            Изображение с наложенными детекциями
        """
        vis_img = img.copy()

        colors = {
            0: (0, 255, 0),   # Зелёный (двери)
            1: (255, 0, 0)    # Синий (окна)
        }

        # Наложение масок
        if show_masks:
            mask_overlay = np.zeros_like(vis_img)
            for detection in results.all_detections:
                if detection.mask is not None:
                    color = colors.get(detection.class_id, (128, 128, 128))
                    mask_overlay[detection.mask > 0] = color

            alpha = 0.4
            vis_img = cv2.addWeighted(vis_img, 1 - alpha, mask_overlay, alpha, 0)

        # Рисование bbox и подписей
        for detection in results.all_detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors.get(detection.class_id, (128, 128, 128))

            if show_boxes:
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
                cv2.putText(vis_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_img
