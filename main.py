"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: CLI-интерфейс (main.py)
Версия: 6.0 — Clean Architecture Edition

Запуск:
  python main.py --input <path> --output <dir> [options]
================================================================================
"""

import argparse
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*torch\.load.*weights_only=False.*")

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn,
)
from rich.text import Text

from core.config import PipelineConfig
from pipeline import FloorplanPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("main")
_console = Console()


# ============================================================
# ARGUMENT PARSER
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Floor Plan Analysis System v6.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', required=True,
                        help='Single image file or directory of images')
    parser.add_argument('--output', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--yolo-model', default='weights/best.pt',
                        help='Path to YOLOv9 .pt weights (default: weights/best.pt)')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--device', default='cpu',
                        help='Inference device: cpu | cuda | 0 | 1 (default: cpu)')
    parser.add_argument('--wall-color', default='8E8780',
                        help='Brusnika wall hex color (default: 8E8780)')
    parser.add_argument('--color-tolerance', type=int, default=3,
                        help='Per-channel color tolerance (default: 3)')
    parser.add_argument('--wall-mode', default='auto',
                        choices=['color', 'structure', 'outline', 'hybrid', 'auto'],
                        help='Wall detection mode (default: auto)')
    parser.add_argument('--structure-method', default='lines',
                        choices=['lines', 'edges', 'contours'],
                        help='Structure detection method (default: lines)')
    parser.add_argument('--wall-height', type=float, default=243.84,
                        help='Wall height in cm for 3D export (default: 243.84)')
    parser.add_argument('--pixels-to-cm', type=float, default=2.54,
                        help='Pixels to cm scale factor (default: 2.54)')
    parser.add_argument('--scale-factor', type=float, default=2.2,
                        help='SweetHome3D scale factor (default: 2.2)')
    parser.add_argument('--disable-measurements', action='store_true',
                        help='Disable OCR room measurement')
    parser.add_argument('--disable-door-arcs', action='store_true',
                        help='Disable door arc detection')
    parser.add_argument('--enable-unet', action='store_true',
                        help='Enable U-Net outline-style detection (experimental)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output and images')
    parser.add_argument('--debug-arcs', action='store_true',
                        help='Save door arc detection debug images')
    return parser


# ============================================================
# IMAGE FILE COLLECTION
# ============================================================

def collect_images(input_path: str) -> list:
    """Collect image paths from a file or directory."""
    inp = Path(input_path)
    if inp.is_file():
        return [str(inp)]
    if inp.is_dir():
        return sorted(
            str(f)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp')
            for f in inp.glob(ext)
        )
    raise ValueError(f"Path not found: {input_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    image_files = collect_images(args.input)
    if not image_files:
        raise ValueError(f"No images found at: {args.input}")

    # Build config from CLI args
    config = PipelineConfig(
        yolo_weights_path=args.yolo_model,
        yolo_confidence=args.yolo_conf,
        yolo_device=args.device,
        wall_color_hex=args.wall_color,
        color_tolerance=args.color_tolerance,
        wall_mode=args.wall_mode,
        structure_method=args.structure_method,
        wall_height_cm=args.wall_height,
        pixels_to_cm=args.pixels_to_cm,
        sh3d_scale_factor=args.scale_factor,
        enable_room_measurement=not args.disable_measurements,
        enable_door_arcs=not args.disable_door_arcs,
        enable_unet=args.enable_unet,
        debug_mode=args.debug,
        debug_arcs=args.debug_arcs,
    )

    pipeline = FloorplanPipeline(config)
    pipeline.initialize()

    _console.print()
    _console.print(Text('  Floor Plan Analysis', style='italic'))
    _console.print()

    ok, fail = 0, 0
    n_stages = 6  # loading, walls, openings, measurement, export, render

    with Progress(
        TextColumn('  {task.description}', justify='left', style='default'),
        BarColumn(bar_width=24, complete_style='white',
                  finished_style='dim', pulse_style='dim white'),
        TaskProgressColumn(style='dim'),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
        expand=False,
    ) as progress:
        for path in image_files:
            name = Path(path).stem
            task_id = progress.add_task(
                f'[dim]{name}[/dim]  loading',
                total=n_stages,
            )

            def _advance(stage: str) -> None:
                progress.advance(task_id)
                progress.update(
                    task_id,
                    description=f'[dim]{name}[/dim]  {stage}',
                )

            try:
                result = pipeline.process(
                    path, args.output, progress_callback=_advance,
                )

                suffix = f'{result.n_walls} walls'
                if len(result.door_arcs) > 0:
                    suffix += f'  {len(result.door_arcs)} arcs'
                if result.sh3d_path:
                    suffix += '  sh3d ok'

                progress.update(
                    task_id,
                    completed=n_stages,
                    description=f'[dim]{name}[/dim]  {suffix}',
                )
                ok += 1

            except Exception as e:
                logger.exception("Failed to process %s", path)
                progress.update(
                    task_id,
                    description=f'[dim]{name}[/dim]  [red]✗ {e}[/red]',
                )
                fail += 1

    _console.print()
    _console.print(
        f'  [dim]{ok}/{ok + fail} completed  ·  {args.output}/[/dim]'
    )
    _console.print()


if __name__ == '__main__':
    main()
