"""
main.py — Application entry point for QuickLabel.
"""

from __future__ import annotations

import sys
import os


def main() -> None:
    # Add project root to path so all imports work
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont, QIcon

    # High-DPI support (must be set before QApplication)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("QuickLabel")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("QuickLabel")

    # Apply global dark stylesheet
    app.setStyleSheet(_global_stylesheet())

    # Set default font
    font = QFont("Segoe UI", 13)
    font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
    app.setFont(font)

    from app import App
    window = App()
    window.showMaximized()

    sys.exit(app.exec())


def _global_stylesheet() -> str:
    from config import (
        COLOR_BG, COLOR_SURFACE, COLOR_SURFACE2, COLOR_ACCENT,
        COLOR_HIGHLIGHT, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_SUCCESS,
    )
    return f"""
    /* ── Reset ──────────────────────────────────────────── */
    * {{
        font-family: "Segoe UI", Inter, Arial, sans-serif;
        font-size: 13px;
    }}

    /* ── Root windows: gradient BG ─────────────────────── */
    QMainWindow, QDialog, QMessageBox, QInputDialog {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:1,
            stop:0 #0d0e17, stop:1 #141729
        );
        color: {COLOR_TEXT};
    }}

    /* ── Generic widget: inherit BG from parent, no artifacts */
    QWidget {{
        background: transparent;
        color: {COLOR_TEXT};
        border: none;
    }}

    /* ── Labels: always transparent background ──────────── */
    QLabel {{
        background: transparent;
        border: none;
    }}

    /* ── Frames/panels with explicit surface color ─────── */
    QFrame[class="panel"] {{
        background: {COLOR_SURFACE};
    }}

    /* ── Scrollbars ────────────────────────────────────── */
    QScrollBar:vertical {{
        background: {COLOR_BG};
        width: 14px;
        margin: 2px;
        border-radius: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLOR_ACCENT};
        border-radius: 6px;
        min-height: 32px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {COLOR_HIGHLIGHT}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

    QScrollBar:horizontal {{
        background: {COLOR_BG};
        height: 14px;
        margin: 2px;
        border-radius: 6px;
    }}
    QScrollBar::handle:horizontal {{
        background: {COLOR_ACCENT};
        border-radius: 6px;
        min-width: 32px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {COLOR_HIGHLIGHT}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }}

    /* ── Tooltip ────────────────────────────────────────── */
    QToolTip {{
        background: {COLOR_SURFACE2};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 6px;
        padding: 5px 10px;
    }}

    /* ── List widgets ───────────────────────────────────── */
    QListWidget {{
        background: transparent;
        border: none;
        outline: none;
    }}
    QListWidget::item {{
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QListWidget::item:selected {{
        background: {COLOR_ACCENT};
        color: {COLOR_TEXT};
    }}
    QListWidget::item:hover {{
        background: {COLOR_ACCENT};
    }}
    """


if __name__ == "__main__":
    main()
