"""
core/image_processor.py — Image resize, letterbox, and JPEG save pipeline.

Design decisions:
- PIL is used instead of OpenCV for final save because PIL gives finer JPEG quality
  control and easier EXIF stripping via `exif=b""`.
- Letterbox (black-bar padding) preserves aspect ratio rather than stretching.
- All processing is done on CPU synchronously; images are small (640×480) so
  this is fast enough even for batches of 50+ images.
- Input can be a numpy array (from OpenCV, BGR) or a PIL Image or a file path.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

import config

# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def process_frame(
    source: Union[np.ndarray, Image.Image, str, Path],
    output_path: Union[str, Path],
) -> tuple[Path, int, int]:
    """
    Process a single image through the full pipeline:
    1. Convert to PIL RGB if needed.
    2. Letterbox-resize to config.IMAGE_WIDTH × config.IMAGE_HEIGHT.
    3. Save as JPEG at config.JPEG_QUALITY, with no EXIF metadata.

    Args:
        source: numpy array (BGR or RGB), PIL Image, or file path.
        output_path: Destination .jpg path.

    Returns:
        (output_path, width, height)
    """
    pil_img = _to_pil(source)
    pil_img = letterbox(pil_img, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_jpeg(pil_img, output_path)
    return output_path, config.IMAGE_WIDTH, config.IMAGE_HEIGHT


def process_frame_to_array(
    source: Union[np.ndarray, Image.Image, str, Path],
) -> np.ndarray:
    """
    Process a frame (letterbox + resize) and return as a numpy array (RGB uint8).
    Used internally when we need the processed frame for CSRT tracker init.
    """
    pil_img = _to_pil(source)
    pil_img = letterbox(pil_img, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    return np.array(pil_img)


# ─── Letterbox ─────────────────────────────────────────────────────────────────

def letterbox(
    img: Image.Image,
    target_w: int,
    target_h: int,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Resize img to fit within (target_w × target_h) while preserving aspect ratio.
    Pads the shorter dimension with fill_color (default: black).
    Uses LANCZOS resampling for high-quality downscale.
    """
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


# ─── Format Conversion ─────────────────────────────────────────────────────────

def _to_pil(source: Union[np.ndarray, Image.Image, str, Path]) -> Image.Image:
    """Convert any supported input type to a PIL RGB Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, (str, Path)):
        return Image.open(source).convert("RGB")

    if isinstance(source, np.ndarray):
        # OpenCV uses BGR; convert to RGB
        if source.ndim == 3 and source.shape[2] == 3:
            rgb = source[:, :, ::-1]  # BGR → RGB
        elif source.ndim == 3 and source.shape[2] == 4:
            rgb = source[:, :, :3][:, :, ::-1]  # BGRA → RGB
        else:
            rgb = source  # Grayscale or already RGB — best-effort
        return Image.fromarray(rgb.astype(np.uint8))

    raise TypeError(f"Unsupported image source type: {type(source)}")


def numpy_bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to PIL RGB Image (no resize)."""
    return _to_pil(frame)


def pil_to_numpy_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image to an OpenCV BGR numpy array."""
    rgb = np.array(img.convert("RGB"))
    return rgb[:, :, ::-1].copy()


# ─── JPEG Save ─────────────────────────────────────────────────────────────────

def _save_jpeg(img: Image.Image, path: Path) -> None:
    """Save a PIL Image as JPEG with no EXIF metadata."""
    img.save(
        path,
        format="JPEG",
        quality=config.JPEG_QUALITY,
        optimize=True,
        exif=b"",  # Strip all EXIF metadata
    )


# ─── Clipboard Import ──────────────────────────────────────────────────────────

def clipboard_to_pil() -> Image.Image | None:
    """
    Attempt to get an image from the system clipboard.
    Returns a PIL Image or None if clipboard has no image data.

    Uses PyQt6 QClipboard for cross-platform support.
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import Qt

        clipboard = QApplication.clipboard()
        q_image = clipboard.image()

        if q_image.isNull():
            return None

        # Convert QImage to PIL
        q_image = q_image.convertToFormat(QImage.Format.Format_RGB888)
        width = q_image.width()
        height = q_image.height()
        ptr = q_image.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
        return Image.fromarray(arr)

    except Exception:
        return None


# ─── Thumbnail Generation ──────────────────────────────────────────────────────

def make_thumbnail(
    source: Union[np.ndarray, Image.Image, str, Path],
    size: int = 80,
) -> Image.Image:
    """
    Generate a square thumbnail of the given size.
    Crops the center of the image to maintain aspect ratio.
    """
    pil_img = _to_pil(source)
    pil_img = ImageOps.fit(pil_img, (size, size), method=Image.LANCZOS)
    return pil_img
