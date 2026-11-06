"""Shared paths, limits and safe file helpers for NetraAI.

Every tool reads from ``input/`` and writes to ``output/``. Tools accept
plain filenames only, never paths, so a model-generated argument cannot
escape the project directories.
"""
import os
from pathlib import Path

import cv2

BASE = Path(os.getenv("NETRA_HOME", Path(__file__).resolve().parents[1])).resolve()

IMAGE_IN = BASE / "input" / "images"
VIDEO_IN = BASE / "input" / "videos"
IMAGE_OUT = BASE / "output" / "images"
VIDEO_OUT = BASE / "output" / "videos"
FRAME_OUT = BASE / "output" / "frames"
REPORT_OUT = BASE / "output" / "reports"
AUDIO_OUT = BASE / "output" / "audio"

for _d in (IMAGE_IN, VIDEO_IN, IMAGE_OUT, VIDEO_OUT, FRAME_OUT, REPORT_OUT, AUDIO_OUT):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

MAX_PIXELS = 25_000_000      # reject very large images
MAX_VIDEO_SECONDS = 600      # reject very long videos

LLM_MODEL = os.getenv("NETRA_LLM", "llama3.1:8b")
VISION_MODEL = os.getenv("NETRA_VISION", "llama3.2-vision")

FOLDERS = {"input": IMAGE_IN, "output": IMAGE_OUT}


def safe_path(directory: Path, filename: str, allowed: set) -> Path:
    """Return directory/filename after validating name, extension and existence."""
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Only plain filenames are accepted.")
    if Path(filename).suffix.lower() not in allowed:
        raise ValueError(f"Unsupported file type. Allowed: {sorted(allowed)}")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


def load_image(filename: str, folder: str = "input", grayscale: bool = False):
    """Decode an image from the input or output folder (BGR or grayscale)."""
    if folder not in FOLDERS:
        raise ValueError("folder must be 'input' or 'output'.")
    path = safe_path(FOLDERS[folder], filename, IMAGE_EXT)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError("Unable to decode image.")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise ValueError(f"Image too large ({w}x{h}); limit is {MAX_PIXELS} pixels.")
    return img


def save_image(img, name: str) -> Path:
    """Write an image into output/images and verify that the write succeeded."""
    out = IMAGE_OUT / name
    if not cv2.imwrite(str(out), img):
        raise IOError(f"Could not save {name}.")
    return out


def rel(path) -> str:
    """Path relative to the project root, for readable tool results."""
    return Path(path).resolve().relative_to(BASE).as_posix()
