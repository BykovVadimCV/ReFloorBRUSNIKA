"""
Система анализа планировок помещений - FastAPI Backend
Версия: 3.9
Автор: Быков Вадим Олегович
Дата: 01.28.2026

REST API для анализа планировок помещений с использованием YOLO и CV алгоритмов.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
import traceback

from main import HybridWallAnalyzer


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

# Директории для хранения файлов
UPLOAD_DIR = Path("./api_uploads")
RESULTS_DIR = Path("./api_results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Путь к модели YOLO (необходимо обновить перед запуском)
YOLO_MODEL_PATH = "D:/ReFloorBRUSNIKA/weights/weights.pt"

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "wall_color_hex": "8E8780",
    "color_tolerance": 3,
    "yolo_conf_threshold": 0.25,
    "device": "cpu",
    "auto_wall_color": False,
    "enable_measurements": True,
    "edge_distance_ratio": 0.15,
    "debug_ocr": False,
    "enable_visuals": True,
    "min_window_len": 15,
    "edge_expansion_tolerance": 8,
    "max_edge_expansion_px": 10,
    "enable_blueprint3d_export": True,
    "skip_bbox_check": True,
    "default_window_width_inches": 48.0,
    "default_door_width_inches": 65.0,
    "wall_height_cm": 300.0,
    "corner_snap_tolerance_px": 10.0,
}

# Глобальный экземпляр анализатора (инициализируется при запуске)
analyzer: Optional[HybridWallAnalyzer] = None

# Хранилище результатов задач в памяти (для продакшна использовать Redis/БД)
job_results: Dict[str, Dict] = {}


# ============================================================
# PYDANTIC МОДЕЛИ
# ============================================================

class AnalysisConfig(BaseModel):
    """Конфигурация для анализа планировки"""
    wall_color_hex: str = Field(default="8E8780", description="Цвет стен в HEX формате")
    color_tolerance: int = Field(default=3, ge=1, le=20, description="Допуск RGB цвета")
    yolo_conf_threshold: float = Field(default=0.25, ge=0.1, le=1.0, description="Порог уверенности YOLO")
    auto_wall_color: bool = Field(default=False, description="Автоопределение цвета стен")
    enable_measurements: bool = Field(default=True, description="Включить измерения площади")
    edge_distance_ratio: float = Field(default=0.15, ge=0.01, le=0.5, description="Коэффициент для внешних стен")
    min_window_len: int = Field(default=15, ge=5, le=100, description="Минимальная длина линии окна")
    edge_expansion_tolerance: int = Field(default=8, ge=1, le=20, description="Допуск для расширения краёв")
    max_edge_expansion_px: int = Field(default=10, ge=0, le=50, description="Макс. расширение краёв в пикселях")
    enable_blueprint3d_export: bool = Field(default=True, description="Включить экспорт Blueprint3D")
    wall_height_cm: float = Field(default=300.0, ge=200, le=500, description="Высота стен в см")
    window_width_inches: float = Field(default=48.0, ge=20, le=100, description="Ширина окна по умолчанию")
    door_width_inches: float = Field(default=65.0, ge=20, le=100, description="Ширина двери по умолчанию")


class JobStatus(BaseModel):
    """Статус задачи анализа"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[str] = None


class AnalysisResult(BaseModel):
    """Результат анализа планировки"""
    job_id: str
    status: str
    image_name: str
    doors_count: int
    windows_count: int
    segments_count: int
    area_m2: Optional[float] = None
    pixel_scale: Optional[float] = None  # см/пиксель
    blueprint3d_available: bool = False
    files: Dict[str, str] = {}  # тип файла -> относительный путь


class AnalysisSummary(BaseModel):
    """Краткая сводка результатов анализа"""
    job_id: str
    image_name: str
    doors: int
    windows: int
    wall_segments: int
    area_m2: Optional[float]
    scale_cm_per_px: Optional[float]
    blueprint3d: Optional[Dict[str, Any]] = None


# ============================================================
# FASTAPI ПРИЛОЖЕНИЕ
# ============================================================

app = FastAPI(
    title="API системы анализа планировок",
    description="REST API для анализа планировок с использованием YOLO и CV алгоритмов",
    version="3.9.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшне настроить конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# СОБЫТИЯ ЖИЗНЕННОГО ЦИКЛА
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация анализатора при запуске сервера"""
    global analyzer
    try:
        analyzer = HybridWallAnalyzer(
            yolo_model_path=YOLO_MODEL_PATH,
            **DEFAULT_CONFIG
        )
    except Exception as e:
        print(f"Ошибка инициализации анализатора: {e}")
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка ресурсов при остановке сервера"""
    pass


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def create_job_id() -> str:
    """Генерация уникального идентификатора задачи"""
    return str(uuid.uuid4())


def save_upload_file(upload_file: UploadFile, destination: Path):
    """Сохранение загруженного файла на диск"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


def get_job_info(job_id: str) -> Dict:
    """Получение информации о задаче"""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail=f"Задача {job_id} не найдена")
    return job_results[job_id]


def process_image_sync(
        job_id: str,
        image_path: Path,
        output_dir: Path,
        config: Optional[AnalysisConfig] = None
):
    """
    Синхронная обработка изображения.
    Обновляет job_results с прогрессом и финальным результатом.
    """
    global job_results

    try:
        # Обновление статуса
        job_results[job_id]["status"] = "processing"
        job_results[job_id]["progress"] = "Начало анализа..."

        # Запуск анализа
        job_results[job_id]["progress"] = "Детекция стен и проёмов..."
        result = analyzer.process_image(
            str(image_path),
            str(output_dir),
            run_color_detection=True
        )

        # Извлечение результатов
        yolo_results = result.get('yolo')
        color_results = result.get('color')
        measurements = result.get('measurements')

        doors_count = len(yolo_results.doors) if yolo_results else 0
        windows_count = len(color_results.get('algo_windows', [])) if color_results else 0
        segments_count = len(color_results.get('segments', [])) if color_results else 0
        area_m2 = measurements.area_m2 if measurements else None
        pixel_scale = measurements.pixel_scale * 100 if measurements else None  # см/пкс

        # Сбор сгенерированных файлов
        base_name = image_path.stem
        files = {}

        # Проверка наличия файлов
        file_patterns = {
            "summary": f"{base_name}_summary.png",
            "overlay": f"{base_name}_overlay.png",
            "wall_mask": f"masks/{base_name}_color_walls.png",
            "outline_mask": f"masks/{base_name}_rect_outline.png",
            "window_mask": f"masks/{base_name}_algo_windows.png",
            "door_mask": f"masks/{base_name}_yolo_doors.png",
            "rectangles_json": f"json/{base_name}_rectangles.json",
            "windows_json": f"json/{base_name}_algo_windows.json",
            "blueprint3d": f"json/{base_name}.blueprint3d",
        }

        for file_key, file_pattern in file_patterns.items():
            file_path = output_dir / file_pattern
            if file_path.exists():
                files[file_key] = f"/results/{job_id}/{file_pattern}"

        # Данные Blueprint3D
        blueprint3d_data = None
        if color_results and 'blueprint3d' in color_results:
            blueprint3d_data = color_results['blueprint3d']

        # Обновление задачи с результатами
        job_results[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": {
                "doors_count": doors_count,
                "windows_count": windows_count,
                "segments_count": segments_count,
                "area_m2": area_m2,
                "pixel_scale": pixel_scale,
                "blueprint3d_available": blueprint3d_data is not None,
                "blueprint3d": blueprint3d_data,
                "files": files
            },
            "progress": "Завершено"
        })

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        job_results[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": error_msg,
            "error_trace": error_trace,
            "progress": "Ошибка"
        })


# ============================================================
# API ЭНДПОИНТЫ
# ============================================================

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "API системы анализа планировок",
        "version": "3.9.0",
        "status": "работает" if analyzer else "не инициализирован"
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "исправен" if analyzer else "неисправен",
        "analyzer_loaded": analyzer is not None,
        "active_jobs": len([j for j in job_results.values() if j["status"] in ["pending", "processing"]])
    }


@app.post("/analyze", response_model=JobStatus)
async def analyze_floor_plan(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="Изображение планировки"),
        config: Optional[str] = Query(None, description="JSON конфигурация")
):
    """
    Загрузка и анализ изображения планировки.
    Возвращает ID задачи для отслеживания прогресса.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Анализатор не инициализирован")

    # Валидация типа файла
    allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Недопустимый тип файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    # Парсинг конфигурации
    analysis_config = None
    if config:
        try:
            config_dict = json.loads(config)
            analysis_config = AnalysisConfig(**config_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Неверная конфигурация: {str(e)}")

    # Создание задачи
    job_id = create_job_id()
    timestamp = datetime.now().isoformat()

    # Сохранение загруженного файла
    upload_path = UPLOAD_DIR / job_id / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    save_upload_file(file, upload_path)

    # Создание директории для результатов
    output_path = RESULTS_DIR / job_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Инициализация записи задачи
    job_results[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": timestamp,
        "completed_at": None,
        "image_name": file.filename,
        "image_path": str(upload_path),
        "output_path": str(output_path),
        "config": analysis_config.dict() if analysis_config else None,
        "error": None,
        "progress": "В очереди"
    }

    # Обработка в фоне
    background_tasks.add_task(
        process_image_sync,
        job_id,
        upload_path,
        output_path,
        analysis_config
    )

    return JobStatus(
        job_id=job_id,
        status="pending",
        created_at=timestamp,
        progress="В очереди на обработку"
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Получение статуса задачи анализа"""
    job_info = get_job_info(job_id)

    return JobStatus(
        job_id=job_id,
        status=job_info["status"],
        created_at=job_info["created_at"],
        completed_at=job_info.get("completed_at"),
        error=job_info.get("error"),
        progress=job_info.get("progress")
    )


@app.get("/jobs/{job_id}/result", response_model=AnalysisResult)
async def get_job_result(job_id: str):
    """Получение полного результата задачи анализа"""
    job_info = get_job_info(job_id)

    if job_info["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Задача ещё не завершена. Текущий статус: {job_info['status']}"
        )

    if job_info["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Задача завершилась с ошибкой: {job_info.get('error', 'Неизвестная ошибка')}"
        )

    result = job_info.get("result", {})

    return AnalysisResult(
        job_id=job_id,
        status=job_info["status"],
        image_name=job_info["image_name"],
        doors_count=result.get("doors_count", 0),
        windows_count=result.get("windows_count", 0),
        segments_count=result.get("segments_count", 0),
        area_m2=result.get("area_m2"),
        pixel_scale=result.get("pixel_scale"),
        blueprint3d_available=result.get("blueprint3d_available", False),
        files=result.get("files", {})
    )


@app.get("/jobs/{job_id}/summary", response_model=AnalysisSummary)
async def get_job_summary(job_id: str):
    """Получение краткой сводки анализа"""
    job_info = get_job_info(job_id)

    if job_info["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Задача не завершена. Текущий статус: {job_info['status']}"
        )

    result = job_info.get("result", {})

    return AnalysisSummary(
        job_id=job_id,
        image_name=job_info["image_name"],
        doors=result.get("doors_count", 0),
        windows=result.get("windows_count", 0),
        wall_segments=result.get("segments_count", 0),
        area_m2=result.get("area_m2"),
        scale_cm_per_px=result.get("pixel_scale"),
        blueprint3d=result.get("blueprint3d")
    )


@app.get("/jobs/{job_id}/blueprint3d")
async def get_blueprint3d(job_id: str):
    """Получение данных Blueprint3D в JSON формате"""
    job_info = get_job_info(job_id)

    if job_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Задача не завершена")

    result = job_info.get("result", {})
    blueprint3d = result.get("blueprint3d")

    if not blueprint3d:
        raise HTTPException(status_code=404, detail="Данные Blueprint3D недоступны")

    return JSONResponse(content=blueprint3d)


@app.get("/results/{job_id}/{file_path:path}")
async def get_result_file(job_id: str, file_path: str):
    """Получение файла результата (изображение, JSON и т.д.)"""
    # Защита от обхода директории
    if ".." in file_path or file_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Недопустимый путь к файлу")

    full_path = RESULTS_DIR / job_id / file_path

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    return FileResponse(
        path=str(full_path),
        filename=full_path.name
    )


@app.get("/jobs")
async def list_jobs(
        status: Optional[str] = Query(None, description="Фильтр по статусу"),
        limit: int = Query(50, ge=1, le=500)
):
    """Список всех задач с опциональной фильтрацией"""
    jobs = list(job_results.values())

    # Фильтрация по статусу
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Сортировка по дате создания (новые первыми)
    jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)

    # Ограничение количества
    jobs = jobs[:limit]

    # Возврат упрощённой информации
    return [
        {
            "job_id": j["job_id"],
            "status": j["status"],
            "image_name": j["image_name"],
            "created_at": j["created_at"],
            "completed_at": j.get("completed_at")
        }
        for j in jobs
    ]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Удаление задачи и её файлов"""
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # Удаление файлов
    try:
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)

        result_dir = RESULTS_DIR / job_id
        if result_dir.exists():
            shutil.rmtree(result_dir)
    except Exception:
        pass

    # Удаление из памяти
    del job_results[job_id]

    return {"message": f"Задача {job_id} удалена"}


@app.get("/config/default")
async def get_default_config():
    """Получение конфигурации по умолчанию"""
    return DEFAULT_CONFIG


# ============================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("Запуск API системы анализа планировок")
    print(f"Директория загрузок: {UPLOAD_DIR}")
    print(f"Директория результатов: {RESULTS_DIR}")
    print(f"Модель YOLO: {YOLO_MODEL_PATH}")
    print("\nДля запуска сервера:")
    print("  uvicorn api:app --reload --host 0.0.0.0 --port 8000")
    print("\nДокументация API доступна по адресу:")
    print("  http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000)