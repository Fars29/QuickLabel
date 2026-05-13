"""
ui/dataset_screen.py — Main workspace screen after opening a dataset.

Tabs: Camera (live webcam + burst capture) and Upload (drag-drop + clipboard).
"""

from __future__ import annotations

import os
import config
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap, QKeyEvent
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from config import (
    COLOR_ACCENT, COLOR_BG, COLOR_HIGHLIGHT, COLOR_SURFACE, COLOR_SURFACE2,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_SUCCESS, COLOR_DANGER, THUMBNAIL_SIZE,
)
from core.dataset_manager import DatasetManager
from core.image_processor import make_thumbnail, clipboard_to_pil, _to_pil
from core.webcam_capture import WebcamWorker
from ui.components.class_input import ClassInput
from ui.components.sidebar import Sidebar

_BTN_PRIMARY = f"""
QPushButton {{
    background: {COLOR_HIGHLIGHT}; color: white; border: none;
    border-radius: 8px; padding: 10px 24px; font-size: 13px;
    font-weight: 600; font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{ background: #ff6b7f; }}
QPushButton:pressed {{ background: #c73652; }}
QPushButton:disabled {{ background: {COLOR_ACCENT}; color: {COLOR_TEXT_MUTED}; }}
"""

_BTN_SECONDARY = f"""
QPushButton {{
    background: transparent;
    color: {COLOR_TEXT};
    border: 2px solid {COLOR_ACCENT};
    border-radius: 8px;
    padding: 9px 18px;
    font-size: 13px;
    font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{
    border-color: {COLOR_HIGHLIGHT};
    color: {COLOR_HIGHLIGHT};
}}
QPushButton:pressed {{ background: rgba(233,69,96,0.1); }}
QPushButton:disabled {{ border-color: {COLOR_ACCENT}; color: {COLOR_TEXT_MUTED}; }}
"""


class ImagePreviewDialog(QDialog):
    """Full-size image preview dialog."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.setModal(True)
        self.setStyleSheet(f"background: #0d0d1a;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Scale to 80% of screen size
        screen = QApplication.primaryScreen().availableGeometry()
        max_w = int(screen.width() * 0.8)
        max_h = int(screen.height() * 0.8)
        scaled = pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        img_lbl = QLabel()
        img_lbl.setPixmap(scaled)
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_lbl)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_BTN_PRIMARY)
        close_btn.setFixedHeight(36)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.adjustSize()


class ThumbnailWidget(QFrame):
    """
    Thumbnail card with:
    - Click anywhere → open full-size preview dialog
    - Hover → shows a red trash overlay button to remove
    """

    removed = pyqtSignal(object)

    def __init__(self, index: int, pixmap: QPixmap, show_remove: bool = True, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._show_remove = show_remove
        
        # Adapt size to content
        self.setFixedSize(pixmap.width() + 12, pixmap.height() + 12)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background: {COLOR_SURFACE}; border-radius: 8px; "
            f"border: 2px solid {COLOR_ACCENT}; }}"
        )
        self.setMouseTracking(True)

        # Image label fills card
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self._img_label = QLabel()
        self._img_label.setPixmap(pixmap)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self._img_label)

        # Overlay trash button (hidden by default)
        if show_remove:
            self._trash_btn = QPushButton("🗑", self)
            self._trash_btn.setFixedSize(22, 22)
            self._trash_btn.setStyleSheet(
                f"QPushButton {{ background: {COLOR_HIGHLIGHT}; color: white; "
                f"border: none; border-radius: 4px; font-size: 11px; }}"
                f"QPushButton:hover {{ background: #ff6b7f; }}"
            )
            self._trash_btn.move(self.width() - 26, 4)
            self._trash_btn.setVisible(False)
            self._trash_btn.clicked.connect(lambda: self.removed.emit(self))
        else:
            self._trash_btn = None

    def enterEvent(self, event):
        self.setStyleSheet(
            f"QFrame {{ background: {COLOR_SURFACE}; border-radius: 8px; "
            f"border: 2px solid {COLOR_HIGHLIGHT}; }}"
        )
        if self._trash_btn:
            self._trash_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(
            f"QFrame {{ background: {COLOR_SURFACE}; border-radius: 8px; "
            f"border: 2px solid {COLOR_ACCENT}; }}"
        )
        if self._trash_btn:
            self._trash_btn.setVisible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Only open preview if not clicking the trash button
            if self._trash_btn and self._trash_btn.geometry().contains(event.pos()):
                return
            dlg = ImagePreviewDialog(self._pixmap, self)
            dlg.exec()
        super().mousePressEvent(event)


class DropZone(QWidget):
    """Drag-and-drop zone for images."""

    files_dropped = pyqtSignal(list)  # list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(160)
        self.setStyleSheet(
            f"background: {COLOR_SURFACE}; border: 2px dashed {COLOR_ACCENT}; border-radius: 12px;"
        )
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("📥")
        icon.setStyleSheet("font-size: 36px; background: transparent; border: none;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        lbl = QLabel("Drag images here or click to browse")
        lbl.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 14px; background: transparent; border: none;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        sub = QLabel("Supports JPG, PNG, BMP · Folders allowed")
        sub.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px; background: transparent; border: none;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if files:
            self.files_dropped.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(
                f"background: {COLOR_SURFACE}; border: 2px dashed {COLOR_HIGHLIGHT}; border-radius: 12px;"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            f"background: {COLOR_SURFACE}; border: 2px dashed {COLOR_ACCENT}; border-radius: 12px;"
        )

    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(
            f"background: {COLOR_SURFACE}; border: 2px dashed {COLOR_ACCENT}; border-radius: 12px;"
        )
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            paths.append(os.path.join(root, f))
            elif os.path.isfile(p):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)


class CameraTab(QWidget):
    """Live webcam preview with burst capture."""

    capture_ready = pyqtSignal(object) # np.ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[WebcamWorker] = None
        self._capturing = False
        self._fps = 5
        self._max_fps = 30
        self._camera_index = 0
        self._pulse_timer = QTimer()
        self._pulse_timer.timeout.connect(self._pulse_dot)
        self._pulse_state = False
        self._capturing_via_mouse = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left Controls Sidebar ─────────────────────────────────────────────
        controls = QFrame()
        controls.setFixedWidth(180)
        controls.setStyleSheet(f"background: {COLOR_SURFACE}; border: 1px solid {COLOR_ACCENT}; border-radius: 8px;")
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(10, 14, 10, 14)
        ctrl_layout.setSpacing(10)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Camera selector dropdown
        cam_lbl = QLabel("CAMERA")
        cam_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; border: none;")
        ctrl_layout.addWidget(cam_lbl)

        self._cam_combo = QComboBox()
        self._cam_combo.setFixedHeight(32)
        self._cam_combo.addItem("Searching...")
        self._cam_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLOR_BG}; color: {COLOR_TEXT};
                border: 2px solid {COLOR_ACCENT}; border-radius: 6px;
                padding: 2px 6px; font-size: 12px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {COLOR_TEXT}; }}
            QComboBox QAbstractItemView {{
                background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_ACCENT};
                selection-background-color: {COLOR_ACCENT};
            }}
        """)
        self._cam_combo.currentIndexChanged.connect(self._on_camera_selected)
        ctrl_layout.addWidget(self._cam_combo)

        # Divider
        d1 = QFrame(); d1.setFixedHeight(1)
        d1.setStyleSheet(f"background: {COLOR_ACCENT}; border: none;")
        ctrl_layout.addWidget(d1)

        # FPS label + pill
        fps_lbl = QLabel("CAPTURE FPS")
        fps_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; border: none;")
        ctrl_layout.addWidget(fps_lbl)

        self._fps_pill = QFrame()
        self._fps_pill.setFixedHeight(34)
        self._fps_pill.setStyleSheet(f"background: {COLOR_BG}; border: 2px solid {COLOR_ACCENT}; border-radius: 8px;")
        fps_pill_layout = QHBoxLayout(self._fps_pill)
        fps_pill_layout.setContentsMargins(4, 0, 4, 0)
        fps_pill_layout.setSpacing(0)

        btn_style = (f"QPushButton {{ background: transparent; color: {COLOR_TEXT}; "
                     f"font-size: 20px; border: none; font-weight: 700; padding: 0; }}"
                     f"QPushButton:hover {{ color: {COLOR_HIGHLIGHT}; }}")

        self._minus_btn = QPushButton("−")
        self._minus_btn.setFixedSize(30, 30)
        self._minus_btn.setStyleSheet(btn_style)
        self._minus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._minus_btn.clicked.connect(lambda: self._adjust_fps(-1))

        self._fps_val_label = QLabel(f"{self._fps} fps")
        self._fps_val_label.setStyleSheet("color: white; font-size: 13px; font-weight: 700; background: transparent; border: none;")
        self._fps_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._plus_btn = QPushButton("+")
        self._plus_btn.setFixedSize(30, 30)
        self._plus_btn.setStyleSheet(btn_style)
        self._plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._plus_btn.clicked.connect(lambda: self._adjust_fps(1))

        fps_pill_layout.addWidget(self._minus_btn)
        fps_pill_layout.addWidget(self._fps_val_label, 1)
        fps_pill_layout.addWidget(self._plus_btn)
        ctrl_layout.addWidget(self._fps_pill)

        # Divider
        d2 = QFrame(); d2.setFixedHeight(1)
        d2.setStyleSheet(f"background: {COLOR_ACCENT}; border: none;")
        ctrl_layout.addWidget(d2)

        # Resolution
        res_lbl = QLabel("RESOLUTION")
        res_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 1px; border: none;")
        ctrl_layout.addWidget(res_lbl)

        self._res_combo = QComboBox()
        self._res_combo.addItems(["640x480", "640x640", "800x600", "1280x720"])
        self._res_combo.setFixedHeight(32)
        import config
        curr_res = f"{config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}"
        self._res_combo.setCurrentText(curr_res)
        self._res_combo.setStyleSheet(f"""
            QComboBox {{
                background: {COLOR_BG}; color: {COLOR_TEXT};
                border: 2px solid {COLOR_ACCENT}; border-radius: 6px;
                padding: 2px 6px; font-size: 12px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ width: 0; height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {COLOR_TEXT}; }}
            QComboBox QAbstractItemView {{
                background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_ACCENT};
                selection-background-color: {COLOR_ACCENT};
            }}
        """)
        self._res_combo.currentTextChanged.connect(self._on_res_changed)
        ctrl_layout.addWidget(self._res_combo)

        # Divider
        d3 = QFrame(); d3.setFixedHeight(1)
        d3.setStyleSheet(f"background: {COLOR_ACCENT}; border: none;")
        ctrl_layout.addWidget(d3)

        # Start/Restart Camera button
        self._start_btn = QPushButton("▶  Start Camera")
        self._start_btn.setStyleSheet(_BTN_SECONDARY)
        self._start_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._toggle_camera)
        ctrl_layout.addWidget(self._start_btn)

        ctrl_layout.addStretch()

        # Record pill at the bottom of controls
        self._record_widget = QFrame()
        self._record_widget.setObjectName("recordPill")
        self._record_widget.setGraphicsEffect(__import__('PyQt6.QtWidgets', fromlist=['QGraphicsDropShadowEffect']).QGraphicsDropShadowEffect(blurRadius=15, xOffset=0, yOffset=4, color=__import__('PyQt6.QtGui', fromlist=['QColor']).QColor(0,0,0,80)))
        self._record_widget.setStyleSheet(f"""
            QFrame#recordPill {{ 
                background: {COLOR_SURFACE2}; 
                border-radius: 16px; 
                border: 1px solid {COLOR_ACCENT};
            }}
        """)
        record_layout = QVBoxLayout(self._record_widget)
        record_layout.setContentsMargins(10, 10, 10, 10)
        record_layout.setSpacing(6)
        record_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._record_btn = QPushButton()
        self._record_btn.setFixedSize(32, 32)
        self._record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._record_btn.setStyleSheet(f"""
            QPushButton {{ 
                background: transparent; 
                border: 2px solid {COLOR_DANGER}; 
                border-radius: 16px;
            }}
            QPushButton:hover {{ background: rgba(255, 42, 85, 0.1); }}
        """)
        self._record_inner = QFrame(self._record_btn)
        self._record_inner.setFixedSize(14, 14)
        self._record_inner.move(9, 9)
        self._record_inner.setStyleSheet(f"background: {COLOR_DANGER}; border-radius: 7px;")
        self._record_inner.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._record_btn.clicked.connect(self._toggle_recording)
        record_layout.addWidget(self._record_btn, 0, Qt.AlignmentFlag.AlignCenter)

        self._tip_lbl = QLabel("Hold space")
        self._tip_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px; background: transparent; border: none;")
        record_layout.addWidget(self._tip_lbl, 0, Qt.AlignmentFlag.AlignCenter)

        ctrl_layout.addWidget(self._record_widget)
        root.addWidget(controls)

        # ── Right: Camera Canvas ──────────────────────────────────────────────
        self._preview_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(self._preview_wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._preview_container = QFrame()
        self._preview_container.setStyleSheet(f"""
            QFrame {{ 
                background: #0d0d1a; 
                border: 3px solid {COLOR_ACCENT}; 
                border-radius: 0px; 
            }}
        """)
        preview_container_layout = QVBoxLayout(self._preview_container)
        preview_container_layout.setContentsMargins(0, 0, 0, 0)

        self._preview_label = QLabel("Camera not started")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background: transparent; color: #7b82a8; font-size: 14px; border: none;")
        preview_container_layout.addWidget(self._preview_label)

        wrapper_layout.addWidget(self._preview_container)
        root.addWidget(self._preview_wrapper, 1)


    def _toggle_camera(self):
        if self._worker and self._worker.isRunning():
            self._restart_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self._worker and self._worker.isRunning():
            return
        self._worker = WebcamWorker()
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.capture_frame.connect(self._on_capture)
        self._worker.error.connect(self._on_camera_error)
        self._worker.camera_opened.connect(self._on_camera_opened)
        self._worker.max_fps_found.connect(self._on_max_fps_found)
        self._worker.cameras_found.connect(self._on_cameras_found)
        self._worker.set_camera_index(self._camera_index)
        self._worker.set_fps(self._fps)
        self._worker.start()
        self._start_btn.setText("🔄  Restart Camera")

    def _stop_camera(self):
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._preview_label.setText("Camera stopped")
        self._start_btn.setText("▶  Start Camera")
        self._pulse_timer.stop()
        self._capturing = False

    def _restart_camera(self):
        self._stop_camera()
        QTimer.singleShot(200, self._start_camera)

    def _on_camera_opened(self, w: int, h: int):
        # We removed _status_label, so we can just log or ignore
        pass

    def _on_frame(self, q_image: QImage):
        # Dynamically resize the container to match the aspect ratio of the frame
        # but limited by the available space in the wrapper
        w, h = q_image.width(), q_image.height()
        # Subtracting more space (12px) to ensure the 3px border is fully visible
        avail_w = self._preview_wrapper.width() - 12
        avail_h = self._preview_wrapper.height() - 12
        
        if avail_w > 0 and avail_h > 0:
            pix = QPixmap.fromImage(q_image)
            scaled = pix.scaled(avail_w, avail_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._preview_label.setPixmap(scaled)
            # Add 6px for the 3px border on each side
            self._preview_container.setFixedSize(scaled.width() + 6, scaled.height() + 6)

    def _on_capture(self, frame: np.ndarray):
        self.capture_ready.emit(frame)
        # Visual feedback: flash container border color only
        self._preview_container.setStyleSheet(f"""
            QFrame {{ 
                background: #0d0d1a; 
                border: 3px solid {COLOR_HIGHLIGHT}; 
                border-radius: 0px; 
            }}
        """)
        QTimer.singleShot(100, self._reset_preview_style)

    def _reset_preview_style(self):
        self._preview_container.setStyleSheet(f"""
            QFrame {{ 
                background: #0d0d1a; 
                border: 3px solid {COLOR_ACCENT}; 
                border-radius: 0px; 
            }}
        """)

    def _on_camera_error(self, msg: str):
        self._preview_label.setText(f"⚠ {msg}")
        self._start_btn.setText("▶  Start Camera")

    def _pulse_dot(self):
        self._pulse_state = not self._pulse_state
        # We removed the dot indicator, so this just toggles internal state for now

    def set_capturing(self, active: bool, via_mouse: bool = False):
        if self._capturing_via_mouse and not via_mouse:
            return
        self._capturing = active
        self._capturing_via_mouse = via_mouse if active else False
        if self._worker:
            self._worker.set_capturing(active)
        if active:
            self._record_inner.setStyleSheet(f"background: {COLOR_DANGER}; border-radius: 2px;")
            self._pulse_timer.start(400)
        else:
            self._record_inner.setStyleSheet(f"background: {COLOR_DANGER}; border-radius: 7px;")
            self._pulse_timer.stop()

    def _toggle_recording(self):
        self.set_capturing(not self._capturing, via_mouse=True)

    def stop_camera(self):
        self._stop_camera()

    def _on_cameras_found(self, cameras: list):
        """Populate the camera dropdown with discovered cameras."""
        # Block signals to avoid triggering _on_camera_selected during update
        self._cam_combo.blockSignals(True)
        self._cam_combo.clear()
        if cameras:
            for idx, name in cameras:
                self._cam_combo.addItem(f"{name}  (#{idx})", userData=idx)
            # Select the currently active camera
            for i in range(self._cam_combo.count()):
                if self._cam_combo.itemData(i) == self._camera_index:
                    self._cam_combo.setCurrentIndex(i)
                    break
        else:
            self._cam_combo.addItem("No cameras found")
        self._cam_combo.blockSignals(False)

    def _on_camera_selected(self, combo_index: int):
        """Switch to the selected camera index."""
        idx = self._cam_combo.itemData(combo_index)
        if idx is None:
            return
        if idx != self._camera_index:
            self._camera_index = idx
            if self._worker and self._worker.isRunning():
                self._restart_camera()

    def _on_fps_changed(self, value: int):
        if self._worker:
            self._worker.set_fps(value)

    def _on_max_fps_found(self, max_val: int):
        self._max_fps = max_val
        # If current fps is higher than new max, cap it
        if self._fps > self._max_fps:
            self._adjust_fps(self._max_fps - self._fps)

    def _adjust_fps(self, delta: int):
        new_val = max(1, min(self._max_fps, self._fps + delta))
        if new_val != self._fps:
            self._fps = new_val
            self._fps_val_label.setText(f"{self._fps} fps")
            self._on_fps_changed(self._fps)

    def _on_res_changed(self, text: str):
        parts = text.split('x')
        if len(parts) == 2:
            import config
            config.IMAGE_WIDTH = int(parts[0])
            config.IMAGE_HEIGHT = int(parts[1])
            if self._worker and self._worker.isRunning():
                self._restart_camera()

class UploadTab(QWidget):
    """Drag-and-drop image upload tab."""
    
    files_added = pyqtSignal(list) # list[str]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._drop_zone = DropZone()
        self._drop_zone.files_dropped.connect(self.files_added.emit)
        layout.addWidget(self._drop_zone)

        # Clipboard button
        clip_btn = QPushButton("📋  Import from Clipboard")
        clip_btn.setStyleSheet(_BTN_SECONDARY)
        clip_btn.clicked.connect(self._import_clipboard)
        layout.addWidget(clip_btn)
        layout.addStretch()

    def _import_clipboard(self):
        pil = clipboard_to_pil()
        if pil is None:
            QMessageBox.information(self, "Clipboard", "No image found in clipboard.")
            return
        import tempfile
        tmp = tempfile.mktemp(suffix=".png")
        pil.save(tmp)
        self.files_added.emit([tmp])

    def clear(self):
        pass


class DatasetScreen(QWidget):
    """
    Main workspace: sidebar + tabs (Camera / Upload) + Review button.

    Signals:
        review_requested(class_name, sources): sources is list of np.ndarray or str paths
    """

    review_requested = pyqtSignal(str, list)
    back_requested = pyqtSignal()

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self._class_name = ""
        self._queued_sources = []
        self._build_ui()
        self._refresh_sidebar()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        self._sidebar.sync_requested.connect(self._on_sync)
        self._sidebar.home_requested.connect(self.back_requested.emit)
        root.addWidget(self._sidebar)

        # Main area
        main = QWidget()
        main.setStyleSheet(f"background: {COLOR_BG};")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar
        top = QWidget()
        top.setStyleSheet(f"background: {COLOR_SURFACE2}; border-bottom: 1px solid {COLOR_ACCENT};")
        top.setFixedHeight(64)
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(20, 0, 20, 0)
        top_layout.setSpacing(16)



        _class_lbl = QLabel("Class:")
        _class_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 13px;")
        top_layout.addWidget(_class_lbl)

        self._class_input = ClassInput()
        self._class_input.setFixedWidth(280)
        self._class_input.class_confirmed.connect(self._on_class_confirmed)
        self._class_input.text_changed.connect(self._on_class_text_changed)
        top_layout.addWidget(self._class_input)
        top_layout.addStretch()

        self._review_btn = QPushButton("Annotate →")
        self._review_btn.setStyleSheet(_BTN_PRIMARY)
        self._review_btn.setFixedHeight(38)
        self._review_btn.setEnabled(False)
        self._review_btn.clicked.connect(self._on_review)
        top_layout.addWidget(self._review_btn)

        main_layout.addWidget(top)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {COLOR_BG}; border: none;
            }}
            QTabBar::tab {{
                background: {COLOR_SURFACE}; color: {COLOR_TEXT_MUTED};
                padding: 10px 20px; font-size: 13px; font-family: "Segoe UI", Inter, Arial;
                border: none; border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {COLOR_TEXT}; border-bottom: 2px solid {COLOR_HIGHLIGHT};
            }}
            QTabBar::tab:hover {{ color: {COLOR_TEXT}; }}
        """)

        self._camera_tab = CameraTab()
        self._camera_tab.capture_ready.connect(self._add_source)
        self._upload_tab = UploadTab()
        self._upload_tab.files_added.connect(self._add_sources)
        self._tabs.addTab(self._camera_tab, "🎥  Camera")
        self._tabs.addTab(self._upload_tab, "📂  Upload")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self._tabs, 1)

        # Unified carousel bar
        carousel_bar = QFrame()
        carousel_bar.setStyleSheet(f"background: {COLOR_SURFACE2}; border-top: 1px solid {COLOR_ACCENT};")
        carousel_bar.setFixedHeight(THUMBNAIL_SIZE + 40)
        carousel_layout = QVBoxLayout(carousel_bar)
        carousel_layout.setContentsMargins(12, 8, 12, 8)
        carousel_layout.setSpacing(4)

        info_row = QHBoxLayout()
        self._counter_label = QLabel("0 sources")
        self._counter_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px; font-weight: 600;")
        info_row.addWidget(self._counter_label)
        info_row.addStretch()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(f"color: {COLOR_DANGER}; background: transparent; border: none; font-size: 11px;")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._clear_sources)
        info_row.addWidget(clear_btn)
        carousel_layout.addLayout(info_row)

        self._strip_scroll = QScrollArea()
        self._strip_scroll.setFixedHeight(THUMBNAIL_SIZE + 10)
        self._strip_scroll.setWidgetResizable(True)
        self._strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._strip_scroll.setStyleSheet("background: transparent; border: none;")

        self._strip_widget = QWidget()
        self._strip_widget.setStyleSheet("background: transparent;")
        self._strip_layout = QHBoxLayout(self._strip_widget)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(8)
        self._strip_layout.addStretch()
        self._strip_scroll.setWidget(self._strip_widget)
        carousel_layout.addWidget(self._strip_scroll)
        
        main_layout.addWidget(carousel_bar)

        from config import SIDEBAR_WIDTH
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(2)
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLOR_ACCENT}; }}")
        
        self._splitter.addWidget(self._sidebar)
        self._splitter.addWidget(main)
        
        self._splitter.setSizes([SIDEBAR_WIDTH, 1000])
        
        root.addWidget(self._splitter)

        self._class_input.set_known_classes(self.dm.get_all_class_names())

        if self._tabs.currentIndex() == 0:
            QTimer.singleShot(100, self._camera_tab._start_camera)

    def _on_tab_changed(self, index: int):
        if index == 0:
            self._camera_tab._start_camera()
        else:
            self._camera_tab._stop_camera()

    def _refresh_sidebar(self):
        self._sidebar.refresh(self.dm)

    def _on_class_confirmed(self, name: str):
        self._class_name = name
        self._update_review_btn()

    def _on_class_text_changed(self, text: str):
        self._class_name = text.strip()
        self._update_review_btn()

    def _update_review_btn(self):
        has_class = bool(self._class_name.strip())
        has_images = len(self._queued_sources) > 0
        self._review_btn.setEnabled(has_class and has_images)

    def _add_source(self, source):
        self._queued_sources.append(source)
        self._rebuild_strip()
        self._update_review_btn()

    def _add_sources(self, sources: list):
        for s in sources:
            if s not in self._queued_sources:
                self._queued_sources.append(s)
        self._rebuild_strip()
        self._update_review_btn()

    def _remove_source(self, widget):
        idx = self._strip_layout.indexOf(widget)
        if 0 <= idx < len(self._queued_sources):
            self._queued_sources.pop(idx)
            self._rebuild_strip()
            self._update_review_btn()

    def _clear_sources(self):
        self._queued_sources.clear()
        self._rebuild_strip()
        self._update_review_btn()

    def _rebuild_strip(self):
        while self._strip_layout.count() > 1:
            item = self._strip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        import cv2
        for idx, src in enumerate(self._queued_sources):
            try:
                if isinstance(src, np.ndarray):
                    frame = src
                else:
                    frame = cv2.imread(src)
                
                h, w = frame.shape[:2]
                scale = THUMBNAIL_SIZE / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                thumb_arr = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(thumb_arr, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg)
            except Exception:
                pix = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
                pix.fill()

            tw = ThumbnailWidget(idx, pix, show_remove=True)
            tw.removed.connect(self._remove_source)
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, tw)
        
        self._counter_label.setText(f"{len(self._queued_sources)} source{'s' if len(self._queued_sources) != 1 else ''}")

    def _on_review(self):
        name = self._class_name.strip()
        if not name or not self._queued_sources:
            return
        self.review_requested.emit(name, list(self._queued_sources))

    def _on_sync(self):
        from core.hf_sync import HFPullWorker, HFPushWorker, retrieve_token
        cfg = self.dm.config
        if not cfg or not cfg.is_synced:
            return
        token = retrieve_token(cfg.hf_token_key)
        if not token:
            QMessageBox.warning(self, "Sync Error", "HF token not found in keyring.")
            return
        self._sidebar.set_syncing(True)
        self._do_full_sync(cfg.hf_repo, token)

    def _do_full_sync(self, repo: str, token: str):
        from core.hf_sync import HFPullWorker
        self._pull_worker = HFPullWorker(repo, token, self.dm.root)
        self._pull_worker.finished.connect(self._on_pull_done)
        self._pull_worker.start()

    def _on_pull_done(self, success: bool, remote_coco: dict, error: str):
        if success and remote_coco:
            try:
                self.dm.merge_remote_coco(remote_coco)
                self.dm.save_coco()
            except Exception as e:
                QMessageBox.warning(self, "Merge Error", str(e))
        self._sidebar.set_syncing(False)
        self._refresh_sidebar()

    def refresh(self):
        """Call after returning from review screen."""
        self._class_input.set_known_classes(self.dm.get_all_class_names())
        self._refresh_sidebar()
        self._clear_sources()
        self._class_input.clear()
        self._class_name = ""
        self._review_btn.setEnabled(False)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._camera_tab.set_capturing(True)
            self._update_review_btn()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._camera_tab.set_capturing(False)
            self._update_review_btn()
        super().keyReleaseEvent(event)

    def cleanup(self):
        self._camera_tab.stop_camera()
