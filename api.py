"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: FastAPI-сервер обработки планировок (api.py)
Версия: 6.0 — Clean Architecture Edition

Эндпоинты:
  GET  /health   — статус сервера
  POST /process  — отправить изображение, получить SH3D + GLTF + PNG

Архитектура:
  - Один экземпляр FloorplanPipeline создаётся при старте (singleton)
  - asyncio.Semaphore ограничивает параллельные задачи
  - CPU-тяжёлая обработка выполняется в run_in_executor (thread pool)
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

from core.config import PipelineConfig
from pipeline import FloorplanPipeline

# ============================================================
# GLTF CONVERSION (optional)
# ============================================================

PROJECT_ROOT = Path(__file__).parent
SH3DTOGLTF_PATH = PROJECT_ROOT / "utils" / "SH3DtoGLTF"
sys.path.insert(0, str(SH3DTOGLTF_PATH))

try:
    from sh3d2gltf import convert_sh3d_to_gltf
    GLTF_CONVERTER_AVAILABLE = True
    print("✓ SH3DtoGLTF converter loaded")
except ImportError as e:
    GLTF_CONVERTER_AVAILABLE = False
    convert_sh3d_to_gltf = None
    print(f"⚠  SH3DtoGLTF not found: {e} — GLTF export disabled")


# ============================================================
# SERVER CONFIGURATION
# ============================================================

SERVER_CONFIG: Dict = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_concurrent_jobs": 1,
    "max_image_size_mb": 20,
    "yolo_model_path": "weights/best.pt",
    "device": "cpu",
}


# ============================================================
# RESPONSE MODELS
# ============================================================

class DetectionInfo(BaseModel):
    walls: int
    doors: int
    windows: int
    rooms: int
    door_arcs: int
    area_sqm: float


class ProcessingResult(BaseModel):
    processing_time_ms: int
    detection_info: DetectionInfo
    files: Dict[str, str]


# ============================================================
# GLOBAL STATE
# ============================================================

app = FastAPI(
    title="ReFloor Brusnika API",
    description="Floorplan to SweetHome3D conversion service",
    version="6.0",
)
processing_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    SERVER_CONFIG["max_concurrent_jobs"]
)
pipeline: Optional[FloorplanPipeline] = None


# ============================================================
# PIPELINE INITIALIZATION
# ============================================================

def _initialize_pipeline() -> None:
    global pipeline
    print("Initializing FloorplanPipeline...")
    # Use PipelineConfig defaults for all processing parameters — identical to
    # running main.py with default CLI arguments.  Only override infrastructure
    # settings that are specific to the server environment.
    config = PipelineConfig(
        yolo_weights_path=SERVER_CONFIG["yolo_model_path"],
        yolo_device=SERVER_CONFIG["device"],
    )
    pipeline = FloorplanPipeline(config)
    pipeline.initialize()
    print("Pipeline initialized")


# ============================================================
# HELPERS
# ============================================================

def _read_file_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _placeholder_png_base64() -> str:
    img = Image.new("RGB", (400, 300), color=(200, 220, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _convert_to_gltf(sh3d_path: str) -> str:
    """Convert SH3D to GLTF. Returns base64 or empty string."""
    if not GLTF_CONVERTER_AVAILABLE or convert_sh3d_to_gltf is None:
        return ""
    try:
        gltf_path = str(Path(sh3d_path).with_suffix(".gltf"))
        convert_sh3d_to_gltf(sh3d_path, gltf_path)
        if os.path.exists(gltf_path):
            return _read_file_as_base64(gltf_path)
    except Exception as exc:
        print(f"⚠  GLTF conversion failed: {exc}")
    return ""


# ============================================================
# CORE PROCESSING (CPU-bound, runs in thread pool)
# ============================================================

def _process_image_sync(image_data: bytes) -> ProcessingResult:
    start = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        stem = "input"
        input_path = temp_path / f"{stem}.png"

        Image.open(io.BytesIO(image_data)).save(input_path)

        result = pipeline.process(str(input_path), str(temp_path))

        if not result.sh3d_path or not os.path.exists(result.sh3d_path):
            raise RuntimeError("SH3D file was not created")

        sh3d_b64 = _read_file_as_base64(result.sh3d_path)
        gltf_b64 = _convert_to_gltf(result.sh3d_path)

        # Find visualization
        viz_b64 = ""
        for candidate in [result.summary_path, result.rooms_ocr_path]:
            if candidate and os.path.exists(candidate):
                viz_b64 = _read_file_as_base64(candidate)
                break
        if not viz_b64:
            viz_b64 = _placeholder_png_base64()

        area_sqm = 0.0
        if result.raw_measurements and hasattr(result.raw_measurements, 'area_m2'):
            area_sqm = float(result.raw_measurements.area_m2)

    elapsed_ms = int((time.time() - start) * 1000)
    print(
        f"Completed: {result.n_walls} walls, {result.n_doors} doors, "
        f"{result.n_windows} windows, {area_sqm:.1f} m², {elapsed_ms} ms"
    )

    return ProcessingResult(
        processing_time_ms=elapsed_ms,
        detection_info=DetectionInfo(
            walls=result.n_walls,
            doors=result.n_doors,
            windows=result.n_windows,
            rooms=len(result.rooms),
            door_arcs=len(result.door_arcs),
            area_sqm=area_sqm,
        ),
        files={
            "sweet_home_3d": sh3d_b64,
            "visualization": viz_b64,
            "gltf": gltf_b64,
        },
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health() -> Dict:
    return {
        "status": "ready" if pipeline is not None else "initializing",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/process", response_model=ProcessingResult)
async def process(image: UploadFile = File(...)) -> ProcessingResult:
    if pipeline is None:
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

    async with processing_semaphore:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _process_image_sync, image_data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return result


# ============================================================
# SERVER STARTUP
# ============================================================

@app.on_event("startup")
async def on_startup() -> None:
    try:
        _initialize_pipeline()
    except Exception as exc:
        print(f"ERROR: Failed to initialize pipeline: {exc}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("ReFloor Brusnika API Server v6.0")
    print("=" * 60)
    print(f"Host:            {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"Max concurrency: {SERVER_CONFIG['max_concurrent_jobs']}")
    print(f"Max image size:  {SERVER_CONFIG['max_image_size_mb']} MB")
    print(f"YOLO model:      {SERVER_CONFIG['yolo_model_path']}")
    print("=" * 60)
    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        log_level="info",
    )
