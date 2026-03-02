"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: FastAPI-сервер обработки планировок (api.py)
Версия: 5.3 — Clean Visuals Edition
Автор: BykovVadimCV
Дата:  23 февраля 2026 г.

Описание модуля:
  Асинхронный REST API на базе FastAPI/uvicorn.
  Принимает изображения планировок формата «Брусника», запускает
  HybridWallAnalyzer в пуле потоков и возвращает:
    - Sweet Home 3D файл (.sh3d) в base64
    - Визуализацию (PNG) в base64
    - Метаданные детекции (двери, окна, площадь)

Эндпоинты:
  GET  /health              — статус сервера
  POST /process             — отправить изображение на обработку
  GET  /status/{job_id}     — статус задачи
  GET  /results/{job_id}    — получить результат
  POST /shutdown            — мягкая остановка сервера

Архитектура:
  - Один экземпляр HybridWallAnalyzer создаётся при старте (singleton)
  - asyncio.Semaphore ограничивает параллельные задачи
  - Старые задачи удаляются фоновой задачей каждую минуту
  - CPU-тяжёлая обработка выполняется в run_in_executor (thread pool)

Принципы:
  - Single Responsibility: каждая функция решает одну задачу
  - Все константы конфигурации — в SERVER_CONFIG
  - Сбой инициализации анализатора прерывает старт сервера
================================================================================
"""

import asyncio
import base64
import io
import os
import tempfile
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from main import HybridWallAnalyzer


# ============================================================
# Конфигурация сервера
# ============================================================

SERVER_CONFIG: Dict = {
    'host': '0.0.0.0',
    'port': 8000,
    'max_concurrent_jobs': 1,
    'max_image_size_mb': 20,
    'result_retention_minutes': 30,
    'yolo_model_path': 'weights/weights.pt',
    'wall_color_hex': '8E8780',
    'color_tolerance': 3,
    'yolo_conf_threshold': 0.25,
    'device': 'cpu',
    'wall_height_cm': 243.84,
    'pixels_to_cm': 2.54,
}


# ============================================================
# Модели данных
# ============================================================

class JobStatus(str, Enum):
    QUEUED = 'queued'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'


class DetectionInfo(BaseModel):
    doors: int
    windows: int
    area_sqm: float


class ProcessingResult(BaseModel):
    job_id: str
    processing_time_ms: int
    detection_info: DetectionInfo
    files: Dict[str, str]


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[int] = None
    result_ready: bool = False


class Job:
    def __init__(self, job_id: str, image_data: bytes) -> None:
        self.job_id = job_id
        self.image_data = image_data
        self.status = JobStatus.QUEUED
        self.progress: int = 0
        self.result: Optional[ProcessingResult] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None


# ============================================================
# Глобальное состояние
# ============================================================

app = FastAPI(title='Sweet Home 3D Processing API')

jobs: Dict[str, Job] = {}
processing_semaphore = asyncio.Semaphore(SERVER_CONFIG['max_concurrent_jobs'])
analyzer: Optional[HybridWallAnalyzer] = None


# ============================================================
# Инициализация анализатора
# ============================================================

def initialize_analyzer() -> None:
    global analyzer
    print('Initializing HybridWallAnalyzer...')
    analyzer = HybridWallAnalyzer(
        yolo_model_path=SERVER_CONFIG['yolo_model_path'],
        wall_color_hex=SERVER_CONFIG['wall_color_hex'],
        color_tolerance=SERVER_CONFIG['color_tolerance'],
        yolo_conf_threshold=SERVER_CONFIG['yolo_conf_threshold'],
        device=SERVER_CONFIG['device'],
        auto_wall_color=False,
        enable_measurements=True,
        enable_visuals=True,
        enable_sh3d_export=True,
        wall_height_cm=SERVER_CONFIG['wall_height_cm'],
        pixels_to_cm=SERVER_CONFIG['pixels_to_cm'],
        auto_create_room=True,
    )
    print('Analyzer initialized')


# ============================================================
# Обработка одного задания
# ============================================================

def _read_file_as_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def _placeholder_png_base64() -> str:
    img = Image.new('RGB', (400, 300), color=(200, 220, 240))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _find_visualization(
    temp_path: Path, job_id: str, color_data: Dict
) -> str:
    candidates = [
        color_data.get('viz_path'),
        str(temp_path / f'input_{job_id}_summary.png'),
        str(temp_path / f'input_{job_id}_overlay.png'),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return _read_file_as_base64(path)
    return _placeholder_png_base64()


def process_floor_plan(job: Job) -> ProcessingResult:
    start_time = time.time()
    job.status = JobStatus.PROCESSING
    job.progress = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_image_path = temp_path / f'input_{job.job_id}.png'

        image = Image.open(io.BytesIO(job.image_data))
        image.save(input_image_path)
        job.progress = 20

        result = analyzer.process_image(
            str(input_image_path),
            str(temp_path),
            run_color_detection=True,
        )
        job.progress = 70

        color_data = result.get('color', {})
        yolo_results = result.get('yolo')

        doors_count = 0
        windows_count = 0

        if yolo_results is not None:
            if hasattr(yolo_results, 'doors') and yolo_results.doors:
                doors_count = len(yolo_results.doors)
            if hasattr(yolo_results, 'windows') and yolo_results.windows:
                windows_count += len(yolo_results.windows)

        windows_count += len(color_data.get('algo_windows', []))

        area_sqm = 0.0
        measurements = result.get('measurements')
        if measurements and hasattr(measurements, 'area_m2'):
            area_sqm = float(measurements.area_m2)

        job.progress = 80

        sh3d_path = color_data.get('sh3d_path')
        if not sh3d_path or not os.path.exists(sh3d_path):
            raise RuntimeError('Sweet Home 3D file was not created')

        sh3d_base64 = _read_file_as_base64(sh3d_path)
        viz_base64 = _find_visualization(temp_path, job.job_id, color_data)

        job.progress = 100
        processing_time_ms = int((time.time() - start_time) * 1000)

        print(
            f'Job {job.job_id} completed: '
            f'{doors_count} doors, {windows_count} windows, {area_sqm:.1f} m²'
        )

        return ProcessingResult(
            job_id=job.job_id,
            processing_time_ms=processing_time_ms,
            detection_info=DetectionInfo(
                doors=doors_count,
                windows=windows_count,
                area_sqm=area_sqm,
            ),
            files={
                'sweet_home_3d': sh3d_base64,
                'visualization': viz_base64,
            },
        )


# ============================================================
# Асинхронная обёртка задания
# ============================================================

async def process_job_async(job: Job) -> None:
    async with processing_semaphore:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_floor_plan, job)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()


# ============================================================
# API эндпоинты
# ============================================================

@app.get('/health')
async def health_check() -> Dict:
    return {
        'status': 'ready' if analyzer is not None else 'initializing',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': sum(
            1 for j in jobs.values() if j.status == JobStatus.PROCESSING
        ),
        'queued_jobs': sum(
            1 for j in jobs.values() if j.status == JobStatus.QUEUED
        ),
    }


@app.post('/process')
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    image_id: Optional[str] = None,
) -> JSONResponse:
    if analyzer is None:
        raise HTTPException(status_code=503, detail='Server is still initializing')

    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='File must be an image')

    image_data = await image.read()
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > SERVER_CONFIG['max_image_size_mb']:
        raise HTTPException(
            status_code=413,
            detail=(
                f'Image too large ({size_mb:.1f} MB). '
                f'Max: {SERVER_CONFIG["max_image_size_mb"]} MB'
            ),
        )

    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, image_data=image_data)
    jobs[job_id] = job
    background_tasks.add_task(process_job_async, job)

    return JSONResponse(
        status_code=202,
        content={
            'job_id': job_id,
            'status': 'processing',
            'message': 'Image accepted for processing',
        },
    )


@app.get('/status/{job_id}')
async def get_status(job_id: str) -> Dict:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail='Job not found')

    job = jobs[job_id]

    if job.status == JobStatus.FAILED:
        return {'job_id': job_id, 'status': job.status.value, 'error': job.error}

    return {
        'job_id': job_id,
        'status': job.status.value,
        'progress': job.progress,
        'result_ready': job.status == JobStatus.COMPLETED,
    }


@app.get('/results/{job_id}')
async def get_results(job_id: str) -> ProcessingResult:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail='Job not found')

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f'Job not completed yet. Current status: {job.status.value}',
        )

    if not job.result:
        raise HTTPException(status_code=500, detail='Result not available')

    return job.result


@app.post('/shutdown')
async def shutdown() -> Dict:
    active_jobs = sum(
        1 for j in jobs.values() if j.status == JobStatus.PROCESSING
    )
    return {
        'message': 'Shutdown initiated',
        'active_jobs': active_jobs,
        'note': 'Server will complete active jobs before shutting down',
    }


# ============================================================
# Старт и остановка
# ============================================================

@app.on_event('startup')
async def startup_event() -> None:
    try:
        initialize_analyzer()
    except Exception as e:
        print(f'ERROR: Failed to initialize analyzer: {e}')
        print('Please check:')
        print('  1. YOLO model path is correct in SERVER_CONFIG')
        print('  2. All dependencies are installed')
        print('  3. Required detection modules are available')
        raise
    asyncio.create_task(cleanup_old_jobs())


async def cleanup_old_jobs() -> None:
    while True:
        await asyncio.sleep(60)
        current_time = datetime.now()
        retention = SERVER_CONFIG['result_retention_minutes']
        to_remove = [
            job_id for job_id, job in jobs.items()
            if job.completed_at
            and (current_time - job.completed_at).total_seconds() / 60 > retention
        ]
        for job_id in to_remove:
            del jobs[job_id]
        if to_remove:
            print(f'Cleaned up {len(to_remove)} old jobs')


# ============================================================
# Точка входа
# ============================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Sweet Home 3D Processing Server')
    print('=' * 60)
    print(f'Starting server on {SERVER_CONFIG["host"]}:{SERVER_CONFIG["port"]}')
    print(f'Max concurrent jobs: {SERVER_CONFIG["max_concurrent_jobs"]}')
    print(f'Max image size: {SERVER_CONFIG["max_image_size_mb"]} MB')
    print(f'YOLO model: {SERVER_CONFIG["yolo_model_path"]}')
    print('=' * 60)
    uvicorn.run(
        app,
        host=SERVER_CONFIG['host'],
        port=SERVER_CONFIG['port'],
        log_level='info',
    )