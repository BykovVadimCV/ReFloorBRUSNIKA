"""
Sweet Home 3D Processing API Server - FIXED VERSION
Handles floor plan image processing and converts to Sweet Home 3D files
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import uuid
import time
import base64
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
import io
from PIL import Image
import asyncio
from enum import Enum
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np

# Import your actual processing modules
from main import HybridWallAnalyzer

# ============================================================================
# Configuration
# ============================================================================

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "max_concurrent_jobs": 1,
    "max_image_size_mb": 20,
    "result_retention_minutes": 30,
    # Processing configuration
    "yolo_model_path": "weights/weights.pt",  # UPDATE THIS PATH!
    "wall_color_hex": "8E8780",
    "color_tolerance": 3,
    "yolo_conf_threshold": 0.25,
    "device": "cpu",
    "wall_height_cm": 243.84,
    "pixels_to_cm": 2.54,
}

# ============================================================================
# Data Models
# ============================================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

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
    def __init__(self, job_id: str, image_data: bytes):
        self.job_id = job_id
        self.image_data = image_data
        self.status = JobStatus.QUEUED
        self.progress = 0
        self.result: Optional[ProcessingResult] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="Sweet Home 3D Processing API")

# In-memory job storage (use Redis/database for production)
jobs: Dict[str, Job] = {}
processing_semaphore = asyncio.Semaphore(SERVER_CONFIG["max_concurrent_jobs"])

# Initialize the analyzer once at startup
analyzer: Optional[HybridWallAnalyzer] = None

# ============================================================================
# Processing Functions
# ============================================================================

def initialize_analyzer():
    """Initialize the HybridWallAnalyzer with configuration"""
    global analyzer

    try:
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
            auto_create_room=True
        )
        print("✓ Analyzer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize analyzer: {e}")
        raise

def process_floor_plan(job: Job) -> ProcessingResult:
    """
    Process floor plan image using HybridWallAnalyzer.
    Generates Sweet Home 3D file and visualization.
    """
    start_time = time.time()

    try:
        # Update progress
        job.status = JobStatus.PROCESSING
        job.progress = 10

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save image to temporary file
            input_image_path = temp_path / f"input_{job.job_id}.png"
            image = Image.open(io.BytesIO(job.image_data))
            image.save(input_image_path)

            job.progress = 20

            # Process the image with HybridWallAnalyzer
            print(f"Processing job {job.job_id}...")
            result = analyzer.process_image(
                str(input_image_path),
                str(temp_path),
                run_color_detection=True
            )

            job.progress = 70

            # Extract detection information
            # FIXED: Handle YOLOResults object properly (not as dict)
            color_data = result.get('color', {})
            yolo_results = result.get('yolo')  # This is a YOLOResults object

            # Count doors and windows
            doors_count = 0
            windows_count = 0

            # FIXED: Access YOLOResults attributes directly (not with .get())
            if yolo_results is not None:
                # YOLOResults has .doors and .windows attributes
                if hasattr(yolo_results, 'doors') and yolo_results.doors:
                    doors_count = len(yolo_results.doors)
                if hasattr(yolo_results, 'windows') and yolo_results.windows:
                    windows_count += len(yolo_results.windows)

            # Add algorithmic windows from color detection
            if 'algo_windows' in color_data:
                windows_count += len(color_data['algo_windows'])

            # Get area measurement
            area_sqm = 0.0
            measurements = result.get('measurements')
            if measurements and hasattr(measurements, 'area_m2'):
                area_sqm = float(measurements.area_m2)

            job.progress = 80

            # Read Sweet Home 3D file
            sh3d_path = color_data.get('sh3d_path')
            if not sh3d_path or not os.path.exists(sh3d_path):
                raise Exception("Sweet Home 3D file was not created")

            with open(sh3d_path, 'rb') as f:
                sh3d_data = f.read()
            sh3d_base64 = base64.b64encode(sh3d_data).decode()

            # Read visualization image - try multiple possible paths
            viz_base64 = None
            possible_viz_paths = [
                color_data.get('viz_path'),
                os.path.join(temp_path, f"input_{job.job_id}_summary.png"),
                os.path.join(temp_path, f"input_{job.job_id}_overlay.png"),
            ]

            for viz_path in possible_viz_paths:
                if viz_path and os.path.exists(viz_path):
                    with open(viz_path, 'rb') as f:
                        viz_data = f.read()
                    viz_base64 = base64.b64encode(viz_data).decode()
                    break

            # Create placeholder if no visualization found
            if viz_base64 is None:
                viz_image = Image.new('RGB', (400, 300), color=(200, 220, 240))
                viz_buffer = io.BytesIO()
                viz_image.save(viz_buffer, format='PNG')
                viz_base64 = base64.b64encode(viz_buffer.getvalue()).decode()

            job.progress = 95

            # Complete processing
            job.progress = 100
            processing_time_ms = int((time.time() - start_time) * 1000)

            print(f"✓ Job {job.job_id} completed: {doors_count} doors, {windows_count} windows, {area_sqm:.1f} m²")

            return ProcessingResult(
                job_id=job.job_id,
                processing_time_ms=processing_time_ms,
                detection_info=DetectionInfo(
                    doors=doors_count,
                    windows=windows_count,
                    area_sqm=area_sqm
                ),
                files={
                    "sweet_home_3d": sh3d_base64,
                    "visualization": viz_base64
                }
            )

    except Exception as e:
        print(f"✗ Job {job.job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Processing failed: {str(e)}")

async def process_job_async(job: Job):
    """Background task to process a job"""
    async with processing_semaphore:
        try:
            # Run CPU-bound processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, process_floor_plan, job)

            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Check if server is ready to accept requests"""
    return {
        "status": "ready" if analyzer is not None else "initializing",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in jobs.values() if j.status == JobStatus.PROCESSING]),
        "queued_jobs": len([j for j in jobs.values() if j.status == JobStatus.QUEUED])
    }

@app.post("/process")
async def process_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    image_id: Optional[str] = None
):
    """Submit a floor plan image for processing"""

    # Check if analyzer is initialized
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Server is still initializing")

    # Validate file
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image data
    image_data = await image.read()

    # Check size limit
    size_mb = len(image_data) / (1024 * 1024)
    if size_mb > SERVER_CONFIG["max_image_size_mb"]:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({size_mb:.1f}MB). Max: {SERVER_CONFIG['max_image_size_mb']}MB"
        )

    # Create job
    job_id = str(uuid.uuid4())
    job = Job(job_id=job_id, image_data=image_data)
    jobs[job_id] = job

    # Start background processing
    background_tasks.add_task(process_job_async, job)

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "processing",
            "message": "Image accepted for processing"
        }
    )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check processing status of a job"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status == JobStatus.FAILED:
        return {
            "job_id": job_id,
            "status": job.status.value,
            "error": job.error
        }

    return {
        "job_id": job_id,
        "status": job.status.value,
        "progress": job.progress,
        "result_ready": job.status == JobStatus.COMPLETED
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Retrieve processed files and metadata"""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {job.status.value}"
        )

    if not job.result:
        raise HTTPException(status_code=500, detail="Result not available")

    return job.result

@app.post("/shutdown")
async def shutdown():
    """Graceful server shutdown"""
    # Wait for active jobs to complete
    active_jobs = [j for j in jobs.values() if j.status == JobStatus.PROCESSING]

    return {
        "message": "Shutdown initiated",
        "active_jobs": len(active_jobs),
        "note": "Server will complete active jobs before shutting down"
    }

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize analyzer and start background cleanup task"""
    try:
        initialize_analyzer()
    except Exception as e:
        print(f"ERROR: Failed to initialize analyzer: {e}")
        print("Please check:")
        print("  1. YOLO model path is correct in SERVER_CONFIG")
        print("  2. All dependencies are installed")
        print("  3. Required detection modules are available")
        raise

    # Start cleanup task
    asyncio.create_task(cleanup_old_jobs())

async def cleanup_old_jobs():
    """Periodically remove old completed jobs"""
    while True:
        await asyncio.sleep(60)  # Run every minute

        current_time = datetime.now()
        retention_minutes = SERVER_CONFIG["result_retention_minutes"]

        jobs_to_remove = []
        for job_id, job in jobs.items():
            if job.completed_at:
                age_minutes = (current_time - job.completed_at).total_seconds() / 60
                if age_minutes > retention_minutes:
                    jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del jobs[job_id]

        if jobs_to_remove:
            print(f"Cleaned up {len(jobs_to_remove)} old jobs")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sweet Home 3D Processing Server (FIXED)")
    print("=" * 60)
    print(f"Starting server on {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"Max concurrent jobs: {SERVER_CONFIG['max_concurrent_jobs']}")
    print(f"Max image size: {SERVER_CONFIG['max_image_size_mb']}MB")
    print(f"YOLO model: {SERVER_CONFIG['yolo_model_path']}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        log_level="info"
    )