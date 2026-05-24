# ReFloorBRUSNIKA

Автоматический анализ планировок квартир Brusnika. На входе — изображение
поэтажного плана, на выходе — файл SweetHome3D (`.sh3d`) и итоговое PNG-превью
с распознанными стенами, дверями, окнами и полигонами помещений.

Стек: Python, OpenCV, YOLOv9, U-Net (segmentation-models-pytorch), docTR OCR,
FastAPI.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Веса YOLO ожидаются по пути `weights/best.pt`.

## Запуск

```bash
# Обработка одного файла или директории
python main.py --input input/plan.png --output output/

# HTTP-сервер (POST /process принимает изображение,
# возвращает SH3D + summary PNG в base64)
python api.py
```

Программный API:

```python
from pipeline import FloorplanPipeline
from core.config import PipelineConfig

pipeline = FloorplanPipeline(PipelineConfig())
pipeline.initialize()
result = pipeline.process("input/plan.png", "output/")
```

## Архитектура

Пайплайн линейный, стадии работают через общие типы из `core.models`:

1. **Препроцессинг** (`pipeline/preprocess.py`) — выравнивание наклона
   и обрезка по границам плана.
2. **Калибровка масштаба** (`pipeline/scale_calibration.py`) — OCR подписей
   с длинами/площадями определяет соотношение пикселей и сантиметров.
3. **Детекция стен** (`detection/walldetector.py`, `walls.py`) — связка из
   цветового, структурного и outline-детекторов плюс классификатор режима.
4. **Детекция проёмов** (`detection/openings.py`) — оркестрирует YOLO для
   дверей/окон, эвристику разрывов в стенах, дуговой детектор дверей и
   опционально U-Net; результаты сливаются в единые объекты `FusedDoor`.
5. **Постобработка** (`pipeline/wall_cleanup.py`) — снэппинг по осям,
   склейка T-стыков, чистка коротких сегментов.
6. **Экспорт и визуализация** (`floorplan_export/`) — сборка `.sh3d` через
   legacy-экспортёр и отрисовка превью с разметкой.

Конфигурация целиком сосредоточена в `PipelineConfig` (`core/config.py`):
веса YOLO, цвет стен Brusnika, режим детекции, высота стен, масштаб,
пороги снэппинга, флаги отладки.

## Модули

### `core/` — общие типы и утилиты

- `config.py` — `PipelineConfig` со всеми настройками пайплайна.
- `models.py` — `BBox`, `Opening`, `WallSegment`, `Room`, `FusedDoor` и др.
- `geometry.py` — IoU, операции над полигонами, преобразования координат.
- `ocr_utils.py` — разбор подписей OCR (площадь, длина).
- `rect_decompose.py` — аппроксимация маски набором повёрнутых прямоугольников.

### `detection/` — детекторы

- `base.py` — иерархия абстрактных классов детекторов.
- `wall_params.py` — константы и пороги для детекции стен.
- `wall_rectangle.py` — `WallRectangle` и морфологические хелперы.
- `walldetector.py` — `BrusnikaWallDetector`, цветовой/outline-варианты,
  классификатор режима.
- `walls.py` — адаптер детекции стен в унифицированные типы.
- `door_arc.py` — `DoorArcDetector` (классическая морфология без DL).
- `door_fusion.py` — `DoorFusion`: объединение YOLO + разрывы + дуги в
  `FusedDoor`.
- `windowdetector.py` — `WindowDetector` (алгоритм параллельных линий).
- `gap.py` — `GapDetector`: поиск проёмов лучами.
- `furnituredetector.py` — инференс YOLO для дверей/окон.
- `unet_inference.py` — U-Net сегментация стен/дверей/окон.
- `openings.py` — `OpeningPipeline`, оркестратор проёмов.

### `pipeline/` — оркестрация

- `__init__.py` — реэкспорт `FloorplanPipeline`.
- `orchestrator.py` — основной класс `FloorplanPipeline`.
- `preprocess.py` — выравнивание и кроп.
- `scale_calibration.py` — калибровка масштаба по OCR.
- `wall_cleanup.py` — постобработка стен.
- `unet_subprocess.py` — изоляция U-Net инференса в отдельном процессе.
- `result.py` — `PipelineResult`.

### `floorplan_export/` — экспорт и визуализация

- `sh3d.py` — адаптер `SH3DExporter`.
- `legacy.py` — оригинальный экспортёр SweetHome3D, не трогать
  (см. ниже о дубликатах типов).
- `visualization.py` — рендер итогового PNG и оверлеев комнат/OCR (`PlotStyle`).

### Прочее

- `main.py` — CLI-точка входа.
- `api.py` — HTTP-сервер FastAPI (`POST /process`, `GET /health`).
- `utils/SH3DtoGLTF/` — опциональный конвертер SH3D в GLTF.
- `utils/yolov9/` — копия YOLOv9.
- `weights/` — веса моделей.
- `input/`, `output/` — рабочие директории по умолчанию.

## Заметка про legacy

`floorplan_export/legacy.py` — исходный экспортёр SweetHome3D со своими
дубликатами `Point`/`Wall`/`Opening`/`Room`/`WallSegment`. Адаптер в
`floorplan_export/sh3d.py` транслирует между ними и типами из `core.models`.
Объединять эти типы нельзя — на границе держится паттерн адаптера.
