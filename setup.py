#!/usr/bin/env python3
"""
Setup script for Sweet Home 3D Processing API
Creates necessary directories and validates installation
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    dirs = [
        'input',
        'output',
        'output/sh3d',
        'output/viz',
        'logs'
    ]

    print("Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}/")
    print()


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")

    required_packages = [
        'fastapi',
        'uvicorn',
        'requests',
        'PIL',
        'pydantic'
    ]

    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed!")
    return True


def create_example_config():
    """Create example configuration file"""
    config_content = """# Sweet Home 3D API Configuration
# Copy this file and modify as needed

[server]
host = 0.0.0.0
port = 8000
max_concurrent_jobs = 4
max_image_size_mb = 10
result_retention_minutes = 30

[client]
server_url = http://localhost:8000
input_directory = ./input
output_directory = ./output
poll_interval_ms = 500
request_timeout_seconds = 300
max_retries = 3
"""

    config_path = Path('config.ini.example')
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✓ Created example config: {config_path}")


def create_sample_readme_input():
    """Create a README in the input directory"""
    readme_content = """# Input Directory

Place your floor plan images here for processing.

Supported formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- GIF (.gif)

The client will process all images in this directory sequentially.
"""

    readme_path = Path('input/README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"✓ Created input README: {readme_path}")


def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Place your floor plan images in the 'input/' directory")
    print("  2. Start the server:")
    print("     python server.py")
    print("  3. In a new terminal, run the client:")
    print("     python client.py")
    print("\nFor more information, see README.md")
    print("=" * 60)


def main():
    print("=" * 60)
    print("Sweet Home 3D Processing API - Setup")
    print("=" * 60)
    print()

    # Create directories
    create_directories()

    # Check dependencies
    deps_ok = check_dependencies()
    print()

    # Create example files
    create_example_config()
    create_sample_readme_input()
    print()

    # Print next steps
    if deps_ok:
        print_next_steps()
    else:
        print("\n⚠️  Please install missing dependencies first!")
        print("Run: pip install -r requirements.txt")

    return 0 if deps_ok else 1


if __name__ == "__main__":
    sys.exit(main())