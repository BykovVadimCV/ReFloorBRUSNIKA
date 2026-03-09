"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: FastAPI-сервер обработки планировок (api.py)
Версия: 6.0 — Single-Request Edition
Автор: BykovVadimCV
Дата:  23 февраля 2026 г.

Описание модуля:
  Синхронный REST API на базе FastAPI/uvicorn.
  Принимает изображение планировки формата «Брусника», запускает
  HybridWallAnalyzer в пуле потоков и возвращает единым ответом:
    - Sweet Home 3D файл (.sh3d) в base64
    - GLTF-модель (.gltf) в base64
    - Визуализацию (PNG) в base64
    - Метаданные детекции (двери, окна, площадь)

Эндпоинты:
  GET  /health   — статус сервера
  POST /process  — отправить изображение, получить результат

Архитектура:
  - Один экземпляр HybridWallAnalyzer создаётся при старте (singleton)
  - asyncio.Semaphore ограничивает параллельные задачи
  - CPU-тяжёлая обработка выполняется в run_in_executor (thread pool)
  - Клиент получает полный результат в теле единственного HTTP-ответа

Принципы:
  - Single Responsibility: каждая функция решает одну задачу
  - Все константы конфигурации — в SERVER_CONFIG
  - Сбой инициализации анализатора прерывает старт сервера
  - Никаких фоновых задач и опросов статуса — один запрос, один ответ
================================================================================
"""

import asyncio
import base64
import io
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from main import HybridWallAnalyzer

# ============================================================
# SH3D → GLTF INTEGRATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent
SH3DTOGLTF_PATH = PROJECT_ROOT / "utils" / "SH3DtoGLTF"
sys.path.insert(0, str(SH3DTOGLTF_PATH))

try:
    from sh3d2gltf import convert_sh3d_to_gltf
    GLTF_CONVERTER_AVAILABLE = True
    print("✓ SH3DtoGLTF converter loaded successfully")
except ImportError as e:
    GLTF_CONVERTER_AVAILABLE = False
    convert_sh3d_to_gltf = None
    print(f"⚠️  SH3DtoGLTF not found: {e} — GLTF export disabled")

# ============================================================
# Конфигурация сервера
# ============================================================

SERVER_CONFIG: Dict = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_concurrent_jobs": 1,
    "max_image_size_mb": 20,
    "yolo_model_path": "weights/weights.pt",
    "wall_color_hex": "8E8780",
    "color_tolerance": 3,
    "yolo_conf_threshold": 0.25,
    "device": "cpu",
    "wall_height_cm": 243.84,
    "pixels_to_cm": 2.54,
}

# ============================================================
# Модели ответа
# ============================================================


class DetectionInfo(BaseModel):
    doors: int
    windows: int
    area_sqm: float


class ProcessingResult(BaseModel):
    processing_time_ms: int
    detection_info: DetectionInfo
    files: Dict[str, str]


# ============================================================
# Глобальное состояние
# ============================================================

app = FastAPI(title="Sweet Home 3D Processing API")
processing_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    SERVER_CONFIG["max_concurrent_jobs"]
)
analyzer: Optional[HybridWallAnalyzer] = None

# ============================================================
# Инициализация анализатора
# ============================================================


def _initialize_analyzer() -> None:
    global analyzer
    print("Initializing HybridWallAnalyzer...")
    analyzer = HybridWallAnalyzer(
        yolo_model_path=SERVER_CONFIG["yolo_model_path"],
        wall_color_hex=SERVER_CONFIG["wall_color_hex"],
        color_tolerance=SERVER_CONFIG["color_tolerance"],
        yolo_conf_threshold=SERVER_CONFIG["yolo_conf_threshold"],
        device=SERVER_CONFIG["device"],
        auto_wall_color=False,
        enable_measurements=True,
        enable_visuals=True,
        enable_sh3d_export=True,
        wall_height_cm=SERVER_CONFIG["wall_height_cm"],
        pixels_to_cm=SERVER_CONFIG["pixels_to_cm"],
        auto_create_room=True,
    )
    print("Analyzer initialized")


# ============================================================
# Вспомогательные функции
# ============================================================


def _read_file_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _placeholder_png_base64() -> str:
    img = Image.new("RGB", (400, 300), color=(200, 220, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _find_visualization(temp_path: Path, stem: str, color_data: Dict) -> str:
    candidates = [
        color_data.get("viz_path"),
        str(temp_path / f"{stem}_summary.png"),
        str(temp_path / f"{stem}_overlay.png"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return _read_file_as_base64(path)
    return _placeholder_png_base64()


def _convert_to_gltf(sh3d_path: str) -> str:
    """Возвращает base64-encoded GLTF или пустую строку при неудаче."""
    if not GLTF_CONVERTER_AVAILABLE or convert_sh3d_to_gltf is None:
        print("GLTF conversion skipped (converter unavailable)")
        return ""
    try:
        gltf_path = str(Path(sh3d_path).with_suffix(".gltf"))
        print(f"→ Converting {Path(sh3d_path).name} → GLTF")
        convert_sh3d_to_gltf(sh3d_path, gltf_path)
        if os.path.exists(gltf_path):
            data = _read_file_as_base64(gltf_path)
            print(f"✓ GLTF ready ({os.path.getsize(gltf_path) // 1024} KB)")
            return data
        print("⚠️ GLTF file was not created")
    except Exception as exc:
        print(f"⚠️ GLTF conversion failed: {exc}")
    return ""


def _count_detections(result: Dict) -> tuple[int, int]:
    """Подсчёт дверей и окон из результатов YOLO + цветового анализа."""
    doors = 0
    windows = 0

    yolo = result.get("yolo")
    if yolo is not None:
        if hasattr(yolo, "doors") and yolo.doors:
            doors = len(yolo.doors)
        if hasattr(yolo, "windows") and yolo.windows:
            windows = len(yolo.windows)

    windows += len(result.get("color", {}).get("algo_windows", []))
    return doors, windows


def _extract_area(result: Dict) -> float:
    measurements = result.get("measurements")
    if measurements and hasattr(measurements, "area_m2"):
        return float(measurements.area_m2)
    return 0.0


# ============================================================
# Ядро обработки (CPU-bound, запускается в thread pool)
# ============================================================


def _process_image_sync(image_data: bytes) -> ProcessingResult:
    start = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        stem = "input"
        input_path = temp_path / f"{stem}.png"

        Image.open(io.BytesIO(image_data)).save(input_path)

        result = analyzer.process_image(
            str(input_path),
            str(temp_path),
            run_color_detection=True,
        )

        color_data = result.get("color", {})
        doors, windows = _count_detections(result)
        area_sqm = _extract_area(result)

        sh3d_path = color_data.get("sh3d_path")
        if not sh3d_path or not os.path.exists(sh3d_path):
            raise RuntimeError("Sweet Home 3D file was not created")

        sh3d_b64 = _read_file_as_base64(sh3d_path)
        viz_b64 = _find_visualization(temp_path, stem, color_data)
        gltf_b64 = _convert_to_gltf(sh3d_path)

    elapsed_ms = int((time.time() - start) * 1000)
    print(f"Completed: {doors} doors, {windows} windows, {area_sqm:.1f} m², {elapsed_ms} ms")

    return ProcessingResult(
        processing_time_ms=elapsed_ms,
        detection_info=DetectionInfo(
            doors=doors,
            windows=windows,
            area_sqm=area_sqm,
        ),
        files={
            "sweet_home_3d": sh3d_b64,
            "visualization": viz_b64,
            "gltf": gltf_b64,
        },
    )


# ============================================================
# Эндпоинты
# ============================================================


@app.get("/health")
async def health() -> Dict:
    return {
        "status": "ready" if analyzer is not None else "initializing",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/process", response_model=ProcessingResult)
async def process(image: UploadFile = File(...)) -> ProcessingResult:
    # ---- проверки ----
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Server is still initializing")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > SERVER_CONFIG["max_image_size_mb"]:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image too large ({size_mb:.1f} MB). "
                f"Max: {SERVER_CONFIG['max_image_size_mb']} MB"
            ),
        )

    # ---- обработка (с семафором и оффлоадом в тред) ----
    async with processing_semaphore:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _process_image_sync, image_data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return result


# ============================================================
# Старт сервера
# ============================================================


@app.on_event("startup")
async def on_startup() -> None:
    try:
        _initialize_analyzer()
    except Exception as exc:
        print(f"ERROR: Failed to initialize analyzer: {exc}")
        print("Please check:")
        print("  1. YOLO model path is correct in SERVER_CONFIG")
        print("  2. All dependencies are installed")
        print("  3. Required detection modules are available")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("Sweet Home 3D Processing Server  (single-request mode)")
    print("=" * 60)
    print(f"Host:             {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"Max concurrency:  {SERVER_CONFIG['max_concurrent_jobs']}")
    print(f"Max image size:   {SERVER_CONFIG['max_image_size_mb']} MB")
    print(f"YOLO model:       {SERVER_CONFIG['yolo_model_path']}")
    print("=" * 60)
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        log_level="info",
    )