"""
core/webcam_capture.py — Live webcam preview and burst capture.

Design decisions:
- WebcamWorker runs in a QThread, uses cv2.VideoCapture in its own thread.
- Preview frames are emitted as QImage signals (not numpy arrays) to avoid
  numpy→QImage conversion on the main thread.
- Capture frames are emitted as numpy arrays (for tracker compatibility).
- Capture is throttled to CAPTURE_FPS using monotonic time, not QTimer.
- The worker tries webcam indexes 0–3 and uses the first one that opens.
- Setting cv2.CAP_PROP_BUFFERSIZE = 1 reduces latency (skip stale frames).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtMultimedia import QMediaDevices

from config import CAPTURE_FPS, WEBCAM_BUFFER_SIZE, WEBCAM_INDEXES_TO_TRY


class WebcamWorker(QThread):
    """
    Background thread that drives a webcam.

    Signals:
        frame_ready(QImage): Emitted for every preview frame (~30fps).
        capture_frame(np.ndarray): Emitted at CAPTURE_FPS while capturing.
        error(str): Emitted if webcam cannot be opened.
        camera_opened(int, int): Emitted once camera is ready (width, height).
    """

    frame_ready = pyqtSignal(QImage)
    capture_frame = pyqtSignal(object)   # numpy array
    error = pyqtSignal(str)
    camera_opened = pyqtSignal(int, int)
    max_fps_found = pyqtSignal(int)
    cameras_found = pyqtSignal(list)   # list of (index, name) tuples
    active_camera_index = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._running = False
        self._capturing = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_capture_time = 0.0
        self._capture_interval = 1.0 / CAPTURE_FPS
        self._camera_index: int = 0

    # ─── Control Methods (called from main thread) ─────────────────────────────

    def set_camera_index(self, idx: int) -> None:
        """Set which camera index to open."""
        self._camera_index = idx

    def set_capturing(self, active: bool) -> None:
        """Enable or disable burst capture mode (called when SPACE pressed/released)."""
        if active and not self._capturing:
            self._last_capture_time = 0.0  # Capture immediately on first frame
        self._capturing = active

    def set_fps(self, fps: int) -> None:
        if fps > 0:
            self._capture_interval = 1.0 / fps

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._running = False
        self.wait(3000)  # Wait up to 3 seconds

    # ─── QThread Run ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop: scan cameras, open selected one, emit frames, handle capture."""
        # 1. Scan available cameras using QtMultimedia (the correct way)
        cameras = self.scan_cameras()
        self.cameras_found.emit(cameras)
        
        # 2. Try to open the selected camera index
        cap = self._open_camera(self._camera_index)
        
        # 3. Fallback: if selected index fails but we found cameras, try the first available one
        if cap is None and cameras:
            for idx, name in cameras:
                if idx != self._camera_index:
                    cap = self._open_camera(idx)
                    if cap is not None:
                        self._camera_index = idx
                        break
        
        if cap is None:
            self.error.emit(
                "No webcam found.\n"
                "Please connect a webcam and restart the application."
            )
            return

        self.active_camera_index.emit(self._camera_index)

        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_opened.emit(w, h)
        
        cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if cap_fps <= 0 or cap_fps > 120:
            cap_fps = 30
        self.max_fps_found.emit(cap_fps)

        self._running = True

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    # Try to recover from a momentary read failure
                    time.sleep(0.05)
                    continue

                import config
                h, w = frame.shape[:2]
                if w != config.TARGET_WIDTH or h != config.TARGET_HEIGHT:
                    frame = cv2.resize(frame, (config.TARGET_WIDTH, config.TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                # Emit preview frame as QImage
                q_image = self._bgr_to_qimage(frame)
                self.frame_ready.emit(q_image)

                # Burst capture at CAPTURE_FPS
                if self._capturing:
                    now = time.monotonic()
                    if now - self._last_capture_time >= self._capture_interval:
                        self._last_capture_time = now
                        self.capture_frame.emit(frame.copy())

        finally:
            cap.release()
            self._cap = None

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _open_camera(self, idx: int = 0) -> Optional[cv2.VideoCapture]:
        """Open a specific camera index."""
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, WEBCAM_BUFFER_SIZE)
            import config
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.TARGET_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.TARGET_HEIGHT)
            return cap
        cap.release()
        return None

    @staticmethod
    def scan_cameras() -> list:
        """Use QtMultimedia to detect available cameras and their names."""
        found = []
        devices = QMediaDevices.videoInputs()
        for i, device in enumerate(devices):
            name = device.description()
            if not name:
                name = f"Camera {i}"
            found.append((i, name))
        return found

    @staticmethod
    def _bgr_to_qimage(frame: np.ndarray) -> QImage:
        """Convert an OpenCV BGR frame to a QImage (RGB888)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()  # .copy() ensures data outlives the numpy array


def _is_windows() -> bool:
    """Return True on Windows."""
    import sys
    return sys.platform == "win32"
