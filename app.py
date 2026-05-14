"""
app.py — Main application window with QStackedWidget routing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QWidget

from config import APP_NAME, APP_VERSION, WINDOW_MIN_HEIGHT, WINDOW_MIN_WIDTH
from core.dataset_manager import DatasetManager
from core.session import save_session
from ui.welcome_screen import WelcomeScreen


class App(QMainWindow):
    """
    Central application window.
    Uses QStackedWidget to route between:
      0 — WelcomeScreen
      1 — DatasetScreen
      2 — ReviewScreen (pushed dynamically)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowIcon(QIcon("QuickLabel.png"))
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resize(1400, 860)

        self._dm: DatasetManager | None = None
        self._dataset_screen = None
        self._review_screen = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Welcome screen (always index 0)
        self._welcome = WelcomeScreen()
        self._welcome.dataset_opened.connect(self._on_dataset_opened)
        self._stack.addWidget(self._welcome)

        self._stack.setCurrentIndex(0)

    # ─── Navigation ────────────────────────────────────────────────────────────

    def _on_dataset_opened(self, dm: DatasetManager) -> None:
        """Switch to the DatasetScreen after a dataset is created/opened."""
        self._dm = dm
        if dm.config and dm.config.local_path:
            save_session(dm.config.local_path)
        self._show_dataset_screen()

    def _show_dataset_screen(self) -> None:
        from ui.dataset_screen import DatasetScreen

        # Remove old dataset screen if exists
        if self._dataset_screen is not None:
            self._stack.removeWidget(self._dataset_screen)
            self._dataset_screen.cleanup()
            self._dataset_screen.deleteLater()
            self._dataset_screen = None

        self._dataset_screen = DatasetScreen(self._dm)
        self._dataset_screen.review_requested.connect(self._on_review_requested)
        self._dataset_screen.back_requested.connect(self._go_welcome)
        self._stack.addWidget(self._dataset_screen)
        self._stack.setCurrentWidget(self._dataset_screen)
        self.setWindowTitle(f"{APP_NAME} — {self._dm.config.name}")

    def _on_review_requested(self, class_name: str, sources: list) -> None:
        if self._dataset_screen is not None:
            self._dataset_screen.cleanup()
        from ui.review_screen import ReviewScreen

        # Remove old review screen
        if self._review_screen is not None:
            self._stack.removeWidget(self._review_screen)
            self._review_screen.deleteLater()
            self._review_screen = None

        self._review_screen = ReviewScreen(class_name, sources, self._dm)
        self._review_screen.done.connect(self._on_review_done)
        self._review_screen.cancelled.connect(self._go_dataset)
        self._stack.addWidget(self._review_screen)
        self._stack.setCurrentWidget(self._review_screen)

    def _on_review_done(self, class_name: str, saved_paths: list, annotations: list) -> None:
        if self._dataset_screen is not None:
            self._dataset_screen.refresh()
        self._go_dataset()

    def _go_dataset(self) -> None:
        if self._dataset_screen is not None:
            self._stack.setCurrentWidget(self._dataset_screen)

        # Clean up review screen
        if self._review_screen is not None:
            self._stack.removeWidget(self._review_screen)
            self._review_screen.deleteLater()
            self._review_screen = None

    def _go_welcome(self) -> None:
        if self._dataset_screen is not None:
            self._dataset_screen.cleanup()
        self._stack.setCurrentWidget(self._welcome)
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")

    def closeEvent(self, event) -> None:
        """Clean up webcam thread on close."""
        if self._dataset_screen is not None:
            self._dataset_screen.cleanup()
        super().closeEvent(event)
