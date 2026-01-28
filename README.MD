# ReFloorBRUSNIKA

**Гибридная система автоматического анализа планировок квартир (версия 3.9)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Описание

**ReFloorBRUSNIKA** — комплексное решение для автоматического анализа изображений архитектурных планировок помещений. Система сочетает классические методы компьютерного зрения с глубоким обучением (YOLOv9) и алгоритмическими подходами для точной детекции стен, дверей, окон, измерения размеров и экспорта в 3D-форматы.

### Основные возможности (версия 3.9)

- ✅ **Гибридная детекция стен** по цвету с прямоугольной декомпозицией и расширением краёв
- ✅ **Детекция дверей и окон**: YOLOv9 + алгоритмическая (параллельные линии) + геометрическая (ray casting)
- ✅ **Измерение размеров**: ray-tracing контура квартиры, OCR площадей (docTR), автоматический масштаб, классификация внешних стен
- ✅ **Многоуровневая валидация проёмов** (реальные размеры в метрах, отсутствие тупиков, перекрытий)
- ✅ **Экспорт в Blueprint3D** (.blueprint3d JSON) с корректной геометрией и проёмами
- ✅ **Умный генератор синтетических датасетов** с архитектурными принципами (зонирование, циркуляция, золотое сечение)
- ✅ **FastAPI-бэкенд** для асинхронной обработки и API-доступа
- ✅ **Обширная визуализация и отладка**

---

## Архитектура системы

### Пайплайн обработки
```
Входное изображение
↓
[1] OCR-очистка текста (docTR)
↓
[2] Цветовая фильтрация + декомпозиция стен (walldetector.py)
↓
[3] YOLO + алгоритмическая + геометрическая детекция дверей/окон
↓
[4] Ray-tracing контура квартиры и внешних стен (space.py)
↓
[5] Морфологическая очистка и объединение сегментов
↓
[6] OCR площадей → расчёт масштаба (см/пиксель)
↓
[7] Измерение стен в метрах, классификация outer crust
↓
[8] Экспорт в Blueprint3D (floorplanexporter.py)
↓
[9] Визуализации + JSON-результаты
```

---

## Быстрый старт

### Установка
```bash
git clone https://github.com/BykovVadimCV/ReFloorBRUSNIKA.git
cd ReFloorBRUSNIKA

python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Загрузка весов моделей

Скачайте предобученные веса YOLO и поместите в папку `weights/`:

📥 [YANDEX DISK - Модели YOLO](https://disk.yandex.com/d/BTVHG7RsxPjtTw)

### Использование

#### 1️⃣ CLI-анализ изображений
```bash
python main.py \
  --input ./input_images \
  --output ./results \
  --yolo-model weights.pt \
  --wall-color 8E8780 \
  --color-tolerance 3 \
  --yolo-conf 0.25 \
  --device cuda \
  --wall-height 300
```

**Параметры:**
- `--input` — путь к изображению или папке
- `--output` — директория для результатов
- `--yolo-model` — путь к весам YOLOv9
- `--wall-color` — HEX цвет стен (по умолчанию 8E8780)
- `--auto-wall-color` — автоматическое определение цвета стен
- `--skip-color` — пропустить цветовую детекцию (только YOLO)
- `--disable-measurements` — отключить измерения размеров
- `--no-vis` — не сохранять визуализации
- `--device` — устройство (cpu/cuda/0/1/...)


#### 2️⃣ Запуск FastAPI сервера
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Документация: http://localhost:8000/docs

Эндпоинты:
`POST /analyze` — загрузка и анализ
`GET /jobs/{job_id}` — статус и результаты
`GET /results/{job_id}/...` — скачивание файлов

#### 3️⃣ Генерация синтетического датасета
```bash
python datasetgen.py \
  --output synth_dataset/ \
  --count 1000 \
  --size 1024 \
  --seed 42 \
  --min-wall-thickness 12 \
  --max-wall-thickness 30
```

Генератор создаёт архитектурно-обоснованные планировки с зонированием, реалистичными окнами/дверями, мебелью и аннотациями.

---

## Структура проекта
```
ReFloorBRUSNIKA/
├── main.py                  # Гибридный анализатор и CLI
├── api.py                   # FastAPI-бэкенд
├── floorplanexporter.py     # Экспорт в Blueprint3D
├── detection/
│   ├── walldetector.py      # Цветовая детекция стен
│   ├── windowdetector.py    # Алгоритмическая детекция окон
│   ├── furnituredetector.py # YOLO-детекция дверей/окон
│   ├── gap.py               # Геометрическая детекция проёмов
│   └── space.py             # Измерения, OCR, контур квартиры
├── datasetgen.py            # Умный генератор датасетов
├── weights/                 # Веса YOLOials (не в git)
├── input/                   # Входные изображения
└── results/                 # Результаты (визуализации, JSON, .blueprint3d)
```

---

## Технический стек

### Глубокое обучение
- **YOLOv9** — детекция дверей/окон
- **docTR** — OCR текстовых меток и площадей
- **OpenCV + NumPy** — морфология, контуры, ray casting
- **FastAPI + Uvicorn** — REST API
- **dataclasses, typing** — строгая типизация

### Компьютерное зрение
- **OpenCV** — морфологические операции, Hough Transform, Distance Transform
- **scikit-image** — прецизионная скелетизация
- **NumPy/SciPy** — векторные операции, интерполяция

### Подход к детекции

| Компонент | Метод | Преимущества |
|-----------|-------|-------------|
| **Стены** | Цветовая фильтрация (RGB+HSV) | Точность на чистых планах, независимость от обучения |
| **Двери/Окна** | YOLOv9 (dual-model) | Робастность к вариациям, высокий recall |
| **Контур квартиры** | Raytracing на исходном изображении | Пиксель-точная форма, устойчивость к шуму |
| **Масштаб** | OCR + геометрия | Автоматический расчёт м²/пиксель |

---

## Результаты

Для каждого изображения:
- Визуализации (overlay, маски, debug)
- JSON с данными (стены, проёмы, площади)
- .blueprint3d файл
- Отладочные изображения (outer crust, living space и др.)

---

## Требования к серверу

### Минимальные требования (CPU-режим)
```yaml
Процессор: Intel Core i5-8400 / AMD Ryzen 5 2600 (6 ядер)
ОЗУ: 16 GB RAM
Хранилище: 10 GB SSD (модели + временные файлы)
ОС: Ubuntu 20.04 LTS / Windows 10/11 / macOS 11+
Python: 3.8 - 3.11
```

### Рекомендуемые требования (GPU-режим)
```yaml
Процессор: Intel Core i7-10700K / AMD Ryzen 7 3700X (8+ ядер)
ОЗУ: 32 GB RAM
GPU: NVIDIA RTX 3060 (12 GB VRAM) или выше
  - CUDA: 11.8+
  - cuDNN: 8.6+
Хранилище: 50 GB NVMe SSD
ОС: Ubuntu 22.04 LTS (рекомендуется)
Python: 3.9 - 3.11
```

### Производительность

| Режим | Разрешение | Время обработки | Изображений/час |
|-------|-----------|-----------------|-----------------|
| CPU | 1024×1024 | ~15-25 сек | 150-240 |
| GPU (RTX 3060) | 1024×1024 | ~3-5 сек | 720-1200 |
| GPU (RTX 4090) | 1024×1024 | ~1-2 сек | 1800-3600 |


---
## Лицензия

MIT License — см. файл [LICENSE](LICENSE)

---

## Автор

**Быков Вадим Олегович**

- Email: [intelset.bykov.v@yandex.ru](mailto:intelset.bykov.v@yandex.ru)
- GitHub: [@BykovVadimCV](https://github.com/BykovVadimCV)
- Telegram: [@bb4131295](https://t.me/bb4131295)

---

## Благодарности

- [YOLOv9](https://github.com/WongKinYiu/yolov9) — детекция объектов
- [docTR](https://github.com/mindee/doctr) — OCR
- [OpenCV](https://opencv.org/) — компьютерное зрение
- [Blueprint3D](https://github.com/cvdlab/react-planner) — формат экспорта
