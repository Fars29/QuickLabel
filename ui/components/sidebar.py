"""
ui/components/sidebar.py — Left sidebar with dataset info and class list.

Design decisions:
- The sidebar is a QWidget with a fixed width; no QDockWidget (simpler).
- Class chips are custom QFrame widgets with name + count badge, styled as pills.
- "Sync with HF" button is hidden when the dataset is local-only.
- The sidebar refreshes via refresh(dataset_manager) — no direct state storage.
- sync_requested signal is emitted when the user clicks "Sync with HF".
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from config import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_HIGHLIGHT,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE2,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    SIDEBAR_WIDTH,
)

_STYLE_SIDEBAR = f"""
    #sidebar {{
        background: {COLOR_SURFACE};
    }}
"""

_STYLE_HEADING = f"""
    color: {COLOR_TEXT};
    font-size: 15px;
    font-weight: 700;
    font-family: "Segoe UI", Inter, Arial;
    padding: 0 0 2px 0;
"""

_STYLE_SUBHEADING = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
    font-family: "Segoe UI", Inter, Arial;
"""

_STYLE_SECTION_LABEL = f"""
    color: {COLOR_TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    font-family: "Segoe UI", Inter, Arial;
    margin-top: 8px;
"""

_STYLE_SYNC_BTN = f"""
QPushButton {{
    background: {COLOR_HIGHLIGHT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
    font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{
    background: #ff6b7f;
}}
QPushButton:pressed {{
    background: #c73652;
}}
QPushButton:disabled {{
    background: {COLOR_ACCENT};
    color: {COLOR_TEXT_MUTED};
}}
"""

_STYLE_DIVIDER = f"background: {COLOR_ACCENT}; max-height: 1px;"


class ClassChip(QFrame):
    """A pill-shaped widget showing a class name and its image count badge."""

    clicked = pyqtSignal(str)

    def __init__(self, class_name: str, count: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.class_name = class_name
        self._build_ui(class_name, count)

    def _build_ui(self, class_name: str, count: int) -> None:
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {COLOR_ACCENT};
                border-radius: 8px;
                border: 1px solid transparent;
            }}
            QFrame:hover {{
                border-color: {COLOR_HIGHLIGHT};
            }}
            """
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(36)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        name_label = QLabel(class_name)
        name_label.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 13px; font-family: 'Segoe UI', Inter, Arial; background: transparent;"
        )
        layout.addWidget(name_label)

        layout.addStretch()

        badge = QLabel(str(count))
        badge.setStyleSheet(
            f"""
            color: white;
            background: {COLOR_HIGHLIGHT};
            border-radius: 10px;
            padding: 1px 7px;
            font-size: 11px;
            font-weight: 700;
            font-family: "Segoe UI", Inter, Arial;
            """
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self.clicked.emit(self.class_name)
        super().mousePressEvent(event)


class Sidebar(QWidget):
    """
    Left sidebar panel.

    Signals:
        sync_requested(): User clicked "Sync with HF".
        home_requested(): User clicked "Home".
        class_selected(str): User clicked a class chip.
    """

    sync_requested = pyqtSignal()
    home_requested = pyqtSignal()
    class_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setObjectName("sidebar")
        self.setStyleSheet(_STYLE_SIDEBAR)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header area ──────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"background: {COLOR_BG};")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 16, 12, 16)
        header_layout.setSpacing(6)

        # Home Button
        self._home_btn = QPushButton("🏠  Home")
        self._home_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._home_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_SURFACE2};
                color: {COLOR_TEXT};
                border: 2px solid {COLOR_ACCENT};
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                font-weight: 700;
                font-family: "Segoe UI", Inter, Arial;
                text-align: left;
            }}
            QPushButton:hover {{
                background: {COLOR_ACCENT};
                border-color: {COLOR_HIGHLIGHT};
            }}
            QPushButton:pressed {{
                background: {COLOR_SURFACE};
            }}
        """)
        self._home_btn.clicked.connect(self.home_requested.emit)
        header_layout.addWidget(self._home_btn)

        outer.addWidget(header)

        # Divider
        div = QFrame()
        div.setStyleSheet(_STYLE_DIVIDER)
        div.setFixedHeight(1)
        outer.addWidget(div)

        # ── Dataset info ─────────────────────────────────────────────────────
        info_widget = QWidget()
        info_widget.setStyleSheet(f"background: {COLOR_SURFACE};")
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(16, 16, 16, 12)
        info_layout.setSpacing(4)

        self._dataset_name_label = QLabel("No dataset open")
        self._dataset_name_label.setStyleSheet(_STYLE_HEADING)
        self._dataset_name_label.setWordWrap(True)
        info_layout.addWidget(self._dataset_name_label)

        self._image_count_label = QLabel("0 images")
        self._image_count_label.setStyleSheet(_STYLE_SUBHEADING)
        info_layout.addWidget(self._image_count_label)

        self._hf_label = QLabel()
        self._hf_label.setStyleSheet(
            f"color: {COLOR_SUCCESS}; font-size: 11px; font-family: 'Segoe UI', Inter, Arial;"
        )
        self._hf_label.setVisible(False)
        info_layout.addWidget(self._hf_label)

        outer.addWidget(info_widget)

        # Divider
        div2 = QFrame()
        div2.setStyleSheet(_STYLE_DIVIDER)
        div2.setFixedHeight(1)
        outer.addWidget(div2)

        # ── Classes section ───────────────────────────────────────────────────
        classes_container = QWidget()
        classes_container.setStyleSheet(f"background: {COLOR_SURFACE};")
        classes_layout = QVBoxLayout(classes_container)
        classes_layout.setContentsMargins(16, 12, 16, 8)
        classes_layout.setSpacing(6)

        classes_header = QLabel("CLASSES")
        classes_header.setStyleSheet(_STYLE_SECTION_LABEL)
        classes_layout.addWidget(classes_header)

        # Scrollable chip area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
        )
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._chips_widget = QWidget()
        self._chips_widget.setStyleSheet("background: transparent;")
        self._chips_layout = QVBoxLayout(self._chips_widget)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        self._chips_layout.addStretch()

        self._scroll.setWidget(self._chips_widget)
        classes_layout.addWidget(self._scroll)

        self._empty_label = QLabel("No classes yet.\nCapture some images to get started.")
        self._empty_label.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 12px; font-family: 'Segoe UI', Inter, Arial;"
        )
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        classes_layout.addWidget(self._empty_label)

        outer.addWidget(classes_container, 1)  # stretch factor 1

        # ── Sync button ───────────────────────────────────────────────────────
        bottom_widget = QWidget()
        bottom_widget.setStyleSheet(f"background: {COLOR_SURFACE};")
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(16, 8, 16, 20)

        div3 = QFrame()
        div3.setStyleSheet(_STYLE_DIVIDER)
        div3.setFixedHeight(1)
        bottom_layout.addWidget(div3)
        bottom_layout.addSpacing(8)

        self._sync_btn = QPushButton("☁  Sync with HF")
        self._sync_btn.setStyleSheet(_STYLE_SYNC_BTN)
        self._sync_btn.setFixedHeight(42)
        self._sync_btn.setVisible(False)
        self._sync_btn.clicked.connect(self.sync_requested.emit)
        bottom_layout.addWidget(self._sync_btn)

        outer.addWidget(bottom_widget)

    # ─── Public API ────────────────────────────────────────────────────────────

    def refresh(self, dataset_manager) -> None:
        """
        Refresh all displayed information from the DatasetManager.
        Call this whenever the dataset state changes.
        """
        if dataset_manager is None or dataset_manager.config is None:
            self._dataset_name_label.setText("No dataset open")
            self._image_count_label.setText("0 images")
            self._hf_label.setVisible(False)
            self._sync_btn.setVisible(False)
            self._update_chips({})
            return

        cfg = dataset_manager.config
        self._dataset_name_label.setText(cfg.name)

        total = dataset_manager.total_image_count()
        self._image_count_label.setText(f"{total} image{'s' if total != 1 else ''}")

        if cfg.is_synced:
            self._hf_label.setText(f"🤗 {cfg.hf_repo}")
            self._hf_label.setVisible(True)
            self._sync_btn.setVisible(True)
        else:
            self._hf_label.setVisible(False)
            self._sync_btn.setVisible(False)

        counts = dataset_manager.get_class_counts()
        self._update_chips(counts)

    def set_syncing(self, is_syncing: bool) -> None:
        """Disable/enable the sync button during a sync operation."""
        self._sync_btn.setEnabled(not is_syncing)
        self._sync_btn.setText("⟳ Syncing…" if is_syncing else "☁  Sync with HF")

    def _update_chips(self, counts: dict[str, int]) -> None:
        """Rebuild the class chip list."""
        # Remove all existing chips (but not the stretch at the end)
        while self._chips_layout.count() > 1:
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        has_classes = bool(counts)
        self._empty_label.setVisible(not has_classes)

        for class_name, count in sorted(counts.items()):
            chip = ClassChip(class_name, count)
            chip.clicked.connect(self.class_selected.emit)
            self._chips_layout.insertWidget(
                self._chips_layout.count() - 1, chip
            )
