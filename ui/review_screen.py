"""
ui/review_screen.py — Bounding box annotation review screen.
"""

from __future__ import annotations

import io
import os
import tempfile
import config
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QKeyEvent, QPixmap, QKeySequence, QImage
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from config import (
    COLOR_ACCENT, COLOR_BG, COLOR_HIGHLIGHT, COLOR_SURFACE, COLOR_SURFACE2,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_SUCCESS, COLOR_WARNING, COLOR_DANGER,
    BBOX_COLOR_PROPAGATED, FILMSTRIP_WIDTH, THUMBNAIL_SIZE,
)
from core.dataset_manager import DatasetManager
from core.image_processor import (
    process_frame, 
    process_frame_to_array,
    make_thumbnail, 
    transform_bbox,
    _to_pil
)
from core.tracker import MultiBBoxTracker
from ui.components.bbox_canvas import BBoxCanvas, BoxState

_BTN_PRIMARY = f"""
QPushButton {{
    background: {COLOR_HIGHLIGHT}; color: {COLOR_BG}; border: none;
    border-radius: 8px; padding: 10px 20px; font-size: 13px;
    font-weight: 700; font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{ background: #33e5ff; }}
QPushButton:pressed {{ background: #00a2cc; }}
QPushButton:disabled {{ background: {COLOR_ACCENT}; color: {COLOR_TEXT_MUTED}; }}
"""

_BTN_NAV = f"""
QPushButton {{
    background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
    border: 1px solid {COLOR_ACCENT};
    border-radius: 8px; padding: 8px 16px; font-size: 13px;
    font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{ background: {COLOR_ACCENT}; border-color: {COLOR_HIGHLIGHT}; }}
QPushButton:pressed {{ background: {COLOR_BG}; }}
QPushButton:disabled {{ background: {COLOR_BG}; color: {COLOR_TEXT_MUTED}; border-color: {COLOR_BG}; }}
"""


class PropagationWorker(QThread):
    """Background thread to run CSRT tracking + GrabCut refinement."""
    finished = pyqtSignal(list, list) # (bboxes, states)
    error = pyqtSignal(str)

    def __init__(self, tracker: MultiBBoxTracker, curr_frame: np.ndarray, scaled_anns: list, next_frame: np.ndarray, inv_scale: tuple[float, float]):
        super().__init__()
        self.tracker = tracker
        self.curr_frame = curr_frame
        self.scaled_anns = scaled_anns
        self.next_frame = next_frame
        self.inv_scale = inv_scale

    def run(self):
        try:
            self.tracker.reset()
            ok = self.tracker.init(self.curr_frame, self.scaled_anns)
            if not ok:
                raise RuntimeError("Tracker init failed")
                
            results = self.tracker.update(self.next_frame)
            
            final_bboxes = []
            final_states = []
            inv_sx, inv_sy = self.inv_scale
            
            for tracked_bbox, is_refined in results:
                x, y, w, h = tracked_bbox
                final_bboxes.append([
                    x * inv_sx,
                    y * inv_sy,
                    w * inv_sx,
                    h * inv_sy,
                ])
                final_states.append(BoxState.REFINED if is_refined else BoxState.PROPAGATED)
                
            self.finished.emit(final_bboxes, final_states)
        except Exception as e:
            self.error.emit(str(e))

class BBoxListWidget(QWidget):
    """Custom widget that forwards mouse clicks to select the list item."""
    def __init__(self, list_widget, item, parent=None):
        super().__init__(parent)
        self.list_widget = list_widget
        self.item = item
        
    def mousePressEvent(self, event):
        self.list_widget.setCurrentItem(self.item)
        super().mousePressEvent(event)

class SyncProgressDialog(QDialog):
    """Modal progress dialog for merge+sync operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Saving & Syncing")
        self.setMinimumWidth(480)
        self.setModal(True)
        self.setStyleSheet(f"background: {COLOR_SURFACE}; color: {COLOR_TEXT}; font-family: 'Segoe UI', Inter, Arial;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(14)

        self._title_label = QLabel("Processing…")
        self._title_label.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {COLOR_TEXT};")
        layout.addWidget(self._title_label)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 13px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Overall progress
        self._overall_label = QLabel("Overall Progress")
        self._overall_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self._overall_label)
        
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: {COLOR_BG}; border-radius: 4px; border: none; }}"
            f"QProgressBar::chunk {{ background: {COLOR_HIGHLIGHT}; border-radius: 4px; }}"
        )
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        # File progress
        self._file_label = QLabel("File Progress")
        self._file_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self._file_label.setVisible(False)
        layout.addWidget(self._file_label)

        self._file_progress = QProgressBar()
        self._file_progress.setRange(0, 100)
        self._file_progress.setFixedHeight(8)
        self._file_progress.setStyleSheet(
            f"QProgressBar {{ background: {COLOR_BG}; border-radius: 4px; border: none; }}"
            f"QProgressBar::chunk {{ background: {COLOR_SUCCESS}; border-radius: 4px; }}"
        )
        self._file_progress.setTextVisible(False)
        self._file_progress.setVisible(False)
        layout.addWidget(self._file_progress)

    def set_status(self, title: str, detail: str = ""):
        self._title_label.setText(title)
        self._status_label.setText(detail)

    def set_determinate(self, value: int, maximum: int):
        self._progress.setRange(0, maximum)
        self._progress.setValue(value)
        if maximum > 0:
            self._overall_label.setText(f"Overall Progress: {value} / {maximum}")
            
    def set_file_progress(self, filename: str, bytes_sent: int, total_bytes: int, speed: str):
        self._file_label.setVisible(True)
        self._file_progress.setVisible(True)
        
        size_kb = total_bytes / 1024
        self._file_label.setText(f"Uploading: {filename} ({size_kb:.1f} KB) - {speed}")
        
        self._file_progress.setRange(0, total_bytes)
        self._file_progress.setValue(bytes_sent)

    def hide_file_progress(self):
        self._file_label.setVisible(False)
        self._file_progress.setVisible(False)

class ReviewThumbnail(QFrame):
    """Miniature for the filmstrip with hover trash icon."""
    removed = pyqtSignal()
    
    def __init__(self, pixmap: QPixmap, index_text: str, parent=None):
        super().__init__(parent)
        self.setFixedSize(FILMSTRIP_WIDTH - 4, THUMBNAIL_SIZE + 20)
        self.setStyleSheet("QFrame { background: transparent; border: none; }")
        self.setMouseTracking(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        self.img_lbl = QLabel()
        self.img_lbl.setPixmap(pixmap)
        self.img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.img_lbl)
        
        self.txt_lbl = QLabel(index_text)
        self.txt_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        self.txt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txt_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.txt_lbl)
        
        self.trash = QPushButton("🗑", self)
        self.trash.setFixedSize(24, 24)
        self.trash.setCursor(Qt.CursorShape.PointingHandCursor)
        self.trash.setStyleSheet(f"""
            QPushButton {{ 
                background: rgba(0, 0, 0, 0.4); color: white; border: none; font-size: 14px; border-radius: 4px;
            }}
            QPushButton:hover {{ background: #e74c3c; }}
        """)
        self.trash.move(self.width() - 32, 4)
        self.trash.setVisible(False)
        self.trash.clicked.connect(self.removed.emit)

    def enterEvent(self, event):
        self.trash.setVisible(True)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.trash.setVisible(False)
        super().leaveEvent(event)

class ReviewScreen(QWidget):
    """
    Review & annotate bounding boxes for a batch of images.

    Signals:
        done(class_name, saved_paths, annotations_per_image)
        cancelled()
    """

    done = pyqtSignal(str, list, list)
    cancelled = pyqtSignal()

    def __init__(
        self,
        class_name: str,
        sources: list,
        dataset_manager: DatasetManager,
        parent=None,
    ):
        super().__init__(parent)
        self.class_name = class_name
        self.dm = dataset_manager
        # sources: list of np.ndarray or str/Path
        self._sources = sources
        self._current_idx = 0
        self._annotations: list[list[list[float]]] = [[] for _ in sources]
        self._states: list[list[BoxState]] = [[] for _ in sources]
        self._tracker = MultiBBoxTracker()
        self._processed_frames: dict[int, np.ndarray] = {}
        self._is_current_modified = False
        self._propagating = False  # FIX B: Guard flag
        self._prop_worker: Optional[PropagationWorker] = None
        self._build_ui()
        self._load_image(0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._splitter = __import__('PyQt6.QtWidgets', fromlist=['QSplitter']).QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(2)
        self._splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLOR_ACCENT}; }}")
        main_layout.addWidget(self._splitter)

        # ── Left: filmstrip ────────────────────────────────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(FILMSTRIP_WIDTH + 20)
        left_panel.setStyleSheet(f"background: {COLOR_SURFACE}; border: none;")

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 12, 8, 12)
        left_layout.setSpacing(6)

        film_label = QLabel("IMAGES")
        film_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        left_layout.addWidget(film_label)

        self._filmstrip = QListWidget()
        self._filmstrip.setStyleSheet(f"""
            QListWidget {{ background: transparent; border: none; outline: 0; }}
            QListWidget::item {{ padding: 2px; border-radius: 8px; margin-bottom: 4px; }}
            QListWidget::item:selected {{ background: {COLOR_ACCENT}; }}
        """)
        self._filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._filmstrip.currentRowChanged.connect(self._on_filmstrip_select)
        left_layout.addWidget(self._filmstrip, 1)



        self._progress_label = QLabel("0 / 0 annotated")
        self._progress_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        left_layout.addWidget(self._progress_label)

        self._splitter.addWidget(left_panel)

        # ── Center: canvas ─────────────────────────────────────────────────────
        center = QWidget()
        center.setStyleSheet(f"background: {COLOR_BG};")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # Top bar
        top_bar = QWidget()
        top_bar.setStyleSheet(f"background: {COLOR_SURFACE2}; border-bottom: 1px solid {COLOR_ACCENT};")
        top_bar.setFixedHeight(56)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(12, 0, 12, 0)
        top_bar_layout.setSpacing(10)

        # Back button at top left
        back_btn = QPushButton("← Back")
        back_btn.setFixedWidth(80)
        back_btn.setFixedHeight(34)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_ACCENT}; border-radius: 6px;
                font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: {COLOR_ACCENT}; border-color: {COLOR_HIGHLIGHT}; }}
        """)
        back_btn.clicked.connect(self.cancelled.emit)
        top_bar_layout.addWidget(back_btn)

        # Class badge (cleaner, bordered)
        self._class_badge = QLabel(self.class_name)
        self._class_badge.setStyleSheet(
            f"color: {COLOR_TEXT}; background: {COLOR_BG}; "
            f"border: 1px solid {COLOR_HIGHLIGHT}; border-radius: 6px; "
            f"padding: 0px 12px; font-size: 13px; font-weight: 700; "
            f"margin-left: 4px;"
        )
        self._class_badge.setFixedHeight(28)
        top_bar_layout.addWidget(self._class_badge)

        top_bar_layout.addStretch()

        # Hints - formatted cleanly
        hints_container = QFrame()
        hints_container.setStyleSheet("background: transparent; border: none;")
        hints_layout = QHBoxLayout(hints_container)
        hints_layout.setContentsMargins(0, 0, 0, 0)
        hints_layout.setSpacing(16)

        def make_hint(key, action):
            h = QLabel(f"<span style='color:{COLOR_TEXT_MUTED}'>{action}:</span> <span style='color:{COLOR_TEXT}'>{key}</span>")
            h.setStyleSheet("font-size: 12px; font-family: Segoe UI, Inter;")
            return h

        hints_layout.addWidget(make_hint("Click & Drag", "Draw"))
        hints_layout.addWidget(make_hint("Ctrl + Scroll", "Zoom"))
        hints_layout.addWidget(make_hint("Canc", "Delete"))
        
        top_bar_layout.addWidget(hints_container)
        top_bar_layout.addStretch()

        # Save Dataset at top right
        self._done_btn = QPushButton("✓ Save Dataset")
        self._done_btn.setStyleSheet(f"""
            QPushButton {{
                background: #061a11; 
                color: #00f5a0;
                border: 2px solid #00f5a0; 
                border-radius: 8px;
                padding: 8px 20px; 
                font-size: 13px; 
                font-weight: 800;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{ 
                background: #00f5a0; 
                color: #061a11;
                border: 2px solid #00f5a0;
            }}
            QPushButton:pressed {{ 
                background: #00c480;
                top: 1px;
            }}
        """)
        self._done_btn.setFixedHeight(40)
        self._done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._done_btn.clicked.connect(self._on_done)
        top_bar_layout.addWidget(self._done_btn)

        center_layout.addWidget(top_bar)

        self._canvas = BBoxCanvas()
        self._canvas.annotations_changed.connect(self._on_annotations_changed)
        self._canvas.box_selected.connect(self._on_box_selected)
        center_layout.addWidget(self._canvas, 1)

        # Bottom nav bar (Pagination only)
        nav_bar = QWidget()
        nav_bar.setStyleSheet(f"background: {COLOR_SURFACE}; border-top: 1px solid {COLOR_ACCENT};")
        nav_bar.setFixedHeight(60)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(16, 0, 16, 0)
        nav_layout.setSpacing(20)

        nav_layout.addStretch()

        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.setStyleSheet(_BTN_NAV)
        self._prev_btn.setFixedWidth(100)
        self._prev_btn.clicked.connect(self._go_prev)
        nav_layout.addWidget(self._prev_btn)

        self._img_counter = QLabel("1 / 1")
        self._img_counter.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 14px; font-weight: 600; min-width: 80px;")
        self._img_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._img_counter)

        # Removed _prop_status label as requested

        self._next_btn = QPushButton("Next →")
        self._next_btn.setStyleSheet(_BTN_NAV)
        self._next_btn.setFixedWidth(100)
        self._next_btn.clicked.connect(self._go_next)
        nav_layout.addWidget(self._next_btn)

        nav_layout.addStretch()

        center_layout.addWidget(nav_bar)
        self._splitter.addWidget(center)

        # ── Right: bbox list ───────────────────────────────────────────────────
        right_panel = QWidget()
        right_panel.setMinimumWidth(280)
        right_panel.setStyleSheet(f"background: {COLOR_SURFACE}; border-left: none; border-top: none; border-bottom: none; border-right: none;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(8)

        boxes_label = QLabel("BOUNDING BOXES")
        boxes_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
        right_layout.addWidget(boxes_label)

        self._bbox_list = QListWidget()
        self._bbox_list.setStyleSheet(f"""
            QListWidget {{ background: transparent; border: none; outline: 0; color: {COLOR_TEXT}; font-size: 12px; }}
            QListWidget::item {{ padding: 0px; border-radius: 6px; margin: 2px 0; }}
            QListWidget::item:selected {{ background: {COLOR_ACCENT}; color: {COLOR_TEXT}; }}
            QListWidget::item:hover {{ background: {COLOR_ACCENT}; }}
        """)
        self._bbox_list.currentRowChanged.connect(self._on_bbox_list_select)
        right_layout.addWidget(self._bbox_list, 1)

        self._splitter.addWidget(right_panel)
        
        # Set initial sizes
        self._splitter.setSizes([FILMSTRIP_WIDTH + 20, 800, 320])

        # Shortcuts
        from PyQt6.QtGui import QShortcut
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._go_next)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._go_prev)
        QShortcut(QKeySequence(Qt.Key.Key_D), self, self._canvas.delete_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._canvas.delete_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, self._canvas.delete_selected)

        # Populate filmstrip
        self._populate_filmstrip()
        self._update_nav()

    # ─── Filmstrip ─────────────────────────────────────────────────────────────

    def _populate_filmstrip(self):
        self._filmstrip.clear()
        for i, src in enumerate(self._sources):
            try:
                pil = _to_pil(src)
                thumb = make_thumbnail(pil, THUMBNAIL_SIZE)
                buf = io.BytesIO()
                thumb.save(buf, format="PNG")
                buf.seek(0)
                pix = QPixmap()
                pix.loadFromData(buf.read())
            except Exception:
                pix = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
                pix.fill(QColor(COLOR_SURFACE))

            from PyQt6.QtGui import QIcon
            item = QListWidgetItem()
            thumb_widget = ReviewThumbnail(pix, f"{i+1}")
            thumb_widget.removed.connect(lambda idx=i: self._delete_image_by_index(idx))
            item.setSizeHint(thumb_widget.sizeHint())
            
            self._filmstrip.addItem(item)
            self._filmstrip.setItemWidget(item, thumb_widget)

        self._filmstrip.setCurrentRow(0)

    def _update_filmstrip_item(self, idx: int):
        item = self._filmstrip.item(idx)
        if item is None:
            return
        has_ann = bool(self._annotations[idx])
        color = COLOR_SUCCESS if has_ann else COLOR_TEXT_MUTED
        item.setForeground(QColor(color))

    # ─── Navigation ────────────────────────────────────────────────────────────

    def _save_current(self):
        """Save canvas annotations back to memory for current image."""
        self._canvas.deselect_all()  # Ensure nothing is amber in saved memory
        self._annotations[self._current_idx] = self._canvas.get_annotations()
        self._states[self._current_idx] = self._canvas.get_annotation_states()
        self._update_filmstrip_item(self._current_idx)
        self._update_progress_label()

    def _load_image(self, idx: int):
        self._current_idx = idx
        self._is_current_modified = False
        src = self._sources[idx]

        # Store a consistently-sized BGR frame for the tracker.
        # We use the configured resolution so that all frames are the same size.
        try:
            from core.image_processor import process_frame_to_array
            rgb_arr, _ = process_frame_to_array(src)          # Config RGB
            frame_bgr = rgb_arr[:, :, ::-1].copy()        # → BGR for OpenCV
        except Exception:
            import config
            frame_bgr = np.zeros((config.TARGET_HEIGHT, config.TARGET_WIDTH, 3), dtype=np.uint8)

        self._processed_frames[idx] = frame_bgr  # always config resolution BGR
        self._canvas.load_image(src)              # canvas uses ORIGINAL resolution

        # Set existing annotations (in original-image coords)
        ann = self._annotations[idx]
        states = self._states[idx] if self._states[idx] else [BoxState.CONFIRMED] * len(ann)
        self._canvas.set_annotations(ann, states)

        self._update_nav()
        self._update_bbox_list()
        self._is_current_modified = False

    def _go_next(self):
        was_modified = self._is_current_modified
        self._save_current()

        next_idx = self._current_idx + 1
        if next_idx >= len(self._sources):
            return

        # CSRT propagation if next image has no annotations OR current was manually modified
        if self._annotations[self._current_idx] and (
            not self._annotations[next_idx] or was_modified
        ):
            self._propagate_to(next_idx)
        else:
            self._load_image(next_idx)

        self._filmstrip.blockSignals(True)
        self._filmstrip.setCurrentRow(next_idx)
        self._filmstrip.blockSignals(False)

    def _go_prev(self):
        self._save_current()
        if self._current_idx > 0:
            prev_idx = self._current_idx - 1
            # Never propagate when going backward, only load
            self._load_image(prev_idx)
            
            self._filmstrip.blockSignals(True)
            self._filmstrip.setCurrentRow(self._current_idx)
            self._filmstrip.blockSignals(False)

    def _on_filmstrip_select(self, row: int):
        if row == self._current_idx or row < 0:
            return
        was_modified = self._is_current_modified
        self._save_current()
        # Only force propagation if jumping FORWARD to an immediately adjacent frame
        is_adjacent_forward = (row - self._current_idx) == 1
        if self._annotations[self._current_idx] and (
            not self._annotations[row] or (was_modified and is_adjacent_forward)
        ):
            self._propagate_to(row)
        else:
            self._load_image(row)

    def _propagate_to(self, next_idx: int):
        """
        Use CSRT tracker to propagate bboxes from the current frame to next_idx.
        FIX A & B: Now runs in background thread and guards against concurrency.
        """
        if self._propagating:
            return  # Drop overlapping calls (FIX B)

        curr_frame_640 = self._processed_frames.get(self._current_idx)
        curr_anns = self._annotations[self._current_idx]

        if curr_frame_640 is None or not curr_anns:
            self._load_image(next_idx)
            return

        # 1. Expand context and scale
        curr_src = self._sources[self._current_idx]
        try:
            curr_pil = _to_pil(curr_src)
            orig_w, orig_h = curr_pil.size
        except Exception:
            self._load_image(next_idx)
            return

        track_h, track_w = curr_frame_640.shape[:2]
        scale_x, scale_y = track_w / orig_w, track_h / orig_h
        scaled_anns = [[b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y] for b in curr_anns]

        next_src = self._sources[next_idx]
        try:
            from core.image_processor import process_frame_to_array
            next_rgb, _ = process_frame_to_array(next_src)
            next_bgr = next_rgb[:, :, ::-1].copy()
            next_pil = _to_pil(next_src)
            next_orig_w, next_orig_h = next_pil.size
        except Exception:
            self._load_image(next_idx)
            return

        self._processed_frames[next_idx] = next_bgr
        inv_scale = (next_orig_w / track_w, next_orig_h / track_h)

        # 2. Start background worker (FIX A)
        self._propagating = True
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._filmstrip.setEnabled(False)

        self._prop_worker = PropagationWorker(
            self._tracker, curr_frame_640, scaled_anns, next_bgr, inv_scale
        )
        
        def on_finished(bboxes, states):
            self._propagating = False
            self._prev_btn.setEnabled(True)
            self._next_btn.setEnabled(True)
            self._filmstrip.setEnabled(True)
            
            self._annotations[next_idx] = bboxes
            self._states[next_idx] = states
            self._load_image(next_idx)
            
            # Sync filmstrip selection
            self._filmstrip.blockSignals(True)
            self._filmstrip.setCurrentRow(next_idx)
            self._filmstrip.blockSignals(False)
            self._update_filmstrip_item(next_idx)

        def on_error(err):
            self._propagating = False
            self._prev_btn.setEnabled(True)
            self._next_btn.setEnabled(True)
            self._filmstrip.setEnabled(True)
            
            # Fallback: copy verbatim
            self._annotations[next_idx] = list(curr_anns)
            self._states[next_idx] = [BoxState.PROPAGATED] * len(curr_anns)
            self._load_image(next_idx)

        self._prop_worker.finished.connect(on_finished)
        self._prop_worker.error.connect(on_error)
        self._prop_worker.start()

    # ─── Annotations UI sync ───────────────────────────────────────────────────

    def _on_box_selected(self, idx: int):
        """Sync canvas selection → right panel list (block signals to avoid loop)."""
        self._bbox_list.blockSignals(True)
        self._bbox_list.setCurrentRow(idx)
        if idx >= 0:
            item = self._bbox_list.item(idx)
            if item:
                self._bbox_list.scrollToItem(item)
        self._bbox_list.blockSignals(False)

    def _on_annotations_changed(self):
        self._is_current_modified = True
        self._update_bbox_list()

    def _on_bbox_list_select(self, row: int):
        if row >= 0:
            self._canvas.select_box(row)

    def _update_bbox_list(self):
        current_row = self._bbox_list.currentRow()
        self._bbox_list.blockSignals(True)
        self._bbox_list.clear()
        bboxes = self._canvas.get_annotations()
        states = self._canvas.get_annotation_states()
        for i, (bbox, state) in enumerate(zip(bboxes, states)):
            x, y, w, h = [round(v) for v in bbox]
            
            if state in (BoxState.REFINED, BoxState.PROPAGATED):
                state_icon = "🔳"
            else:
                state_icon = "🟩"
            
            item = QListWidgetItem()
            self._bbox_list.addItem(item)
            
            widget = BBoxListWidget(self._bbox_list, item)
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(16, 8, 16, 8)
            layout.setSpacing(12)
            layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            
            lbl = QLabel(f"{state_icon} Box {i+1}: [{x},{y},{w},{h}]")
            lbl.setStyleSheet("background: transparent; border: none; color: white; font-size: 13px;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            layout.addWidget(lbl, 1)
            
            trash = QPushButton("🗑")
            trash.setFixedSize(28, 28)
            trash.setCursor(Qt.CursorShape.PointingHandCursor)
            trash.setStyleSheet(f"""
                QPushButton {{ 
                    background: transparent; color: {COLOR_TEXT_MUTED}; border: none; border-radius: 4px; font-size: 14px;
                }}
                QPushButton:hover {{ background: #e74c3c; color: white; }}
            """)
            trash.setCursor(Qt.CursorShape.PointingHandCursor)
            trash.clicked.connect(lambda _, idx=i: self._delete_box_by_index(idx))
            layout.addWidget(trash)
            
            item.setSizeHint(widget.sizeHint())
            self._bbox_list.setItemWidget(item, widget)

        if 0 <= current_row < self._bbox_list.count():
            self._bbox_list.setCurrentRow(current_row)
        self._bbox_list.blockSignals(False)

    def _delete_box_by_index(self, idx: int):
        self._canvas.select_box(idx)
        self._canvas.delete_selected()

    def _delete_current_image(self):
        self._delete_image_by_index(self._current_idx)

    def _delete_image_by_index(self, idx: int):
        if idx < 0 or idx >= len(self._sources):
            return
        
        self._sources.pop(idx)
        self._annotations.pop(idx)
        self._states.pop(idx)
        
        new_frames = {}
        for old_idx, frame in self._processed_frames.items():
            if old_idx < idx:
                new_frames[old_idx] = frame
            elif old_idx > idx:
                new_frames[old_idx - 1] = frame
        self._processed_frames = new_frames

        if len(self._sources) == 0:
            self.cancelled.emit()
            return
            
        if self._current_idx == idx:
            next_idx = min(idx, len(self._sources) - 1)
            self._current_idx = next_idx
            self._populate_filmstrip()
            self._load_image(next_idx)
        else:
            if self._current_idx > idx:
                self._current_idx -= 1
            self._populate_filmstrip()
            self._update_nav()
            self._filmstrip.setCurrentRow(self._current_idx)

    def _update_nav(self):
        n = len(self._sources)
        idx = self._current_idx
        self._img_counter.setText(f"{idx+1} / {n}")
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < n - 1)

    def _update_progress_label(self):
        annotated = sum(1 for a in self._annotations if a)
        self._progress_label.setText(f"{annotated} / {len(self._sources)} annotated")

    # ─── Done / Save / Sync ────────────────────────────────────────────────────

    def _on_done(self):
        self._save_current()

        # Check for unannotated images
        unannotated = sum(1 for a in self._annotations if not a)
        if unannotated > 0:
            reply = QMessageBox.question(
                self, "Unannotated Images",
                f"{unannotated} image(s) have no bounding boxes and will be skipped.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        dialog = SyncProgressDialog(self)
        dialog.show()

        try:
            # We don't close the dialog here if it goes async
            self._execute_save(dialog)
        except Exception as exc:
            dialog.close()
            QMessageBox.critical(self, "Save Error", str(exc))
            return

    def _execute_save(self, progress_dialog: SyncProgressDialog):
        """Process images, update COCO JSON, optionally sync with HF."""
        from PyQt6.QtWidgets import QApplication

        self._progress_dialog = progress_dialog
        cfg = self.dm.config

        # Step 1: Pull remote if HF-backed
        if cfg and cfg.is_synced:
            from core.hf_sync import HFPullWorker, retrieve_token
            token = retrieve_token(cfg.hf_token_key)
            if token:
                progress_dialog.set_status("Checking for remote changes…")
                self._pull_worker = HFPullWorker(cfg.hf_repo, token, self.dm.root)
                
                def on_pull_done(success, remote_coco, error):
                    if success and remote_coco:
                        self.dm.merge_remote_coco(remote_coco)
                    # Proceed even if pull fails (offline mode)
                    self._continue_save_after_pull(progress_dialog)

                self._pull_worker.finished.connect(on_pull_done)
                self._pull_worker.start()
                return # Async wait
        
        self._continue_save_after_pull(progress_dialog)

    def _continue_save_after_pull(self, progress_dialog: SyncProgressDialog):
        """Step 2 & 3: Save images and update local COCO."""
        from PyQt6.QtWidgets import QApplication
        cfg = self.dm.config

        saved_paths: list[Path] = []
        annotations_per_image: list[list[list[float]]] = []
        start_num = self.dm.next_image_number(self.class_name)

        for i, (src, bboxes) in enumerate(zip(self._sources, self._annotations)):
            if not bboxes:
                continue
            num = start_num + len(saved_paths)
            out_path = self.dm.build_image_path(self.class_name, num)
            try:
                from core.image_processor import transform_bbox
                _, _, _, transform = process_frame(src, out_path)
                scale, px, py = transform
                
                # Transform each bbox to match the letterboxed output image
                transformed = [transform_bbox(bb, scale, px, py) for bb in bboxes]
                
                saved_paths.append(out_path)
                annotations_per_image.append(transformed)
            except OSError as exc:
                if "No space left" in str(exc) or "disk" in str(exc).lower():
                    raise OSError("Disk full — could not save images. Free up space and try again.")
                raise
            progress_dialog.set_determinate(i + 1, len(self._sources))
            QApplication.processEvents()

        # Step 3: Update COCO JSON
        progress_dialog.set_status("Updating annotation file…")
        QApplication.processEvents()
        self.dm.add_batch(self.class_name, saved_paths, annotations_per_image)
        self.dm.save_coco()

        # Step 4: Push to HF
        if cfg and cfg.is_synced:
            progress_dialog.set_status("Syncing with Hugging Face...", "Identifying local changes...")
            progress_dialog.set_determinate(0, 0)
            
            # Start async push of all changed files (including new images and COCO)
            self._do_push_sync_async(cfg, saved_paths, annotations_per_image)
            return  # The done signal will be emitted in the async callback
            
        # Emit done signal if not syncing
        self._finish_save(saved_paths, annotations_per_image)

    def _finish_save(self, saved_paths, annotations_per_image):
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
            
        self.done.emit(
            self.class_name,
            [str(p) for p in saved_paths],
            annotations_per_image,
        )

    def _do_push_sync_async(self, cfg, saved_paths: list[Path], annotations_per_image: list):
        from core.hf_sync import HFPushWorker, retrieve_token
        token = retrieve_token(cfg.hf_token_key)
        if not token:
            self._finish_save(saved_paths, annotations_per_image)
            return
            
        changed = {str(p.relative_to(self.dm.root)) for p in saved_paths}
        coco_rel = str(Path("annotations") / "instances_all.json")
        changed.add(coco_rel)
        
        self._push_worker = HFPushWorker(cfg.hf_repo, token, self.dm.root, changed, only_missing=True)
        dialog = self._progress_dialog
        
        if dialog:
            self._push_worker.status.connect(lambda s1, s2: dialog.set_status(s1, s2))
            self._push_worker.progress_overall.connect(
                lambda up, tot: dialog.set_determinate(up, tot)
            )
            self._push_worker.progress_file.connect(
                lambda name, sent, tot, spd: dialog.set_file_progress(name, sent, tot, spd)
            )
        
        def on_push_done(success, message):
            if not success:
                QMessageBox.warning(
                    self, "Sync Warning",
                    f"Images saved locally but HF push failed:\n{message}\n\nYou can retry sync from the sidebar."
                )
            self._finish_save(saved_paths, annotations_per_image)
            
        self._push_worker.finished.connect(on_push_done)
        self._push_worker.start()
