"""
ui/welcome_screen.py — Opening screen with interactive cards.

Changes from v1:
- "New Synced Dataset" renamed to "Hugging Face Dataset" (covers both new & existing HF repos)
- Added "Resume Last Session" banner if a previous dataset is found in AppData session cache
- Buttons have visible borders/outlines for clarity
- Cleaner layout with more breathing room
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import (
    APP_VERSION,
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_HIGHLIGHT,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE2,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)
from core.dataset_manager import DatasetManager
from core.hf_sync import HFValidateWorker, store_token
from core.session import load_session, save_session

# ── Shared button styles ───────────────────────────────────────────────────────

_BTN_PRIMARY = f"""
QPushButton {{
    background: {COLOR_HIGHLIGHT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 11px 28px;
    font-size: 13px;
    font-weight: 600;
    font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{ background: #33dbff; }}
QPushButton:pressed {{ background: #00a2cc; }}
QPushButton:disabled {{ background: {COLOR_ACCENT}; color: {COLOR_TEXT_MUTED}; }}
"""

_BTN_OUTLINE = f"""
QPushButton {{
    background: {COLOR_SURFACE2}; color: {COLOR_TEXT};
    border: 1px solid {COLOR_ACCENT}; border-radius: 6px;
    padding: 9px 22px;
    font-size: 12px; font-weight: 600;
    font-family: "Segoe UI", Inter, Arial;
}}
QPushButton:hover {{ background: {COLOR_ACCENT}; border-color: {COLOR_HIGHLIGHT}; }}
QPushButton:pressed {{ background: {COLOR_BG}; }}
"""

_INPUT_STYLE = f"""
QLineEdit {{
    background: {COLOR_BG};
    color: {COLOR_TEXT};
    border: 2px solid {COLOR_ACCENT};
    border-radius: 8px;
    padding: 9px 13px;
    font-size: 13px;
    font-family: "Segoe UI", Inter, Arial;
}}
QLineEdit:focus {{ border-color: {COLOR_HIGHLIGHT}; }}
"""


# ── Clickable card ─────────────────────────────────────────────────────────────

class ClickableCard(QFrame):
    """Large interactive card that emits clicked on press."""

    clicked = pyqtSignal()

    def __init__(self, icon: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self._normal_style = f"""
            QFrame {{
                background: {COLOR_SURFACE};
                border: 2px solid {COLOR_ACCENT};
                border-radius: 16px;
            }}
        """
        self._hover_style = f"""
            QFrame {{
                background: {COLOR_SURFACE};
                border: 2px solid {COLOR_HIGHLIGHT};
                border-radius: 16px;
            }}
        """
        self.setStyleSheet(self._normal_style)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(260, 210)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(10)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 38px; background: transparent; border: none;")
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 15px; font-weight: 700; "
            f"font-family: 'Segoe UI', Inter, Arial; background: transparent; border: none;"
        )
        title_lbl.setWordWrap(True)
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 12px; "
            f"font-family: 'Segoe UI', Inter, Arial; background: transparent; border: none;"
        )
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)
        layout.addStretch()

    def enterEvent(self, event):
        self.setStyleSheet(self._hover_style)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self._normal_style)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ── HF Setup Dialog ────────────────────────────────────────────────────────────

class HFSetupDialog(QDialog):
    """
    Modal dialog for Hugging Face dataset setup.
    Works for both creating a new repo and re-connecting to an existing one.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face Dataset")
        self.setMinimumWidth(500)
        self.setStyleSheet(
            f"background: {COLOR_SURFACE}; color: {COLOR_TEXT}; "
            f"font-family: 'Segoe UI', Inter, Arial;"
        )
        self._token: Optional[str] = None
        self._repo: Optional[str] = None
        self._worker: Optional[HFValidateWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(14)

        heading = QLabel("🤗  Hugging Face Dataset")
        heading.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 18px; font-weight: 700;")
        layout.addWidget(heading)

        sub = QLabel(
            "Enter your HF token and repository name.\n"
            "Works for both new and existing repositories — existing data will be downloaded automatically.\n"
            "Your token is stored securely in the OS keyring, never in a file."
        )
        sub.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px; line-height: 1.5;")
        sub.setWordWrap(True)
        layout.addWidget(sub)

        layout.addSpacing(6)

        # Token
        lbl1 = QLabel("HuggingFace Token (write access required)")
        lbl1.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(lbl1)

        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setStyleSheet(_INPUT_STYLE)
        self._token_input.setMinimumHeight(42)
        layout.addWidget(self._token_input)

        # Repo
        lbl2 = QLabel("Repository name  (e.g.  myorg/my-dataset)")
        lbl2.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 13px; font-weight: 600;")
        layout.addWidget(lbl2)

        self._repo_input = QLineEdit()
        self._repo_input.setPlaceholderText("username/dataset-name")
        self._repo_input.setStyleSheet(_INPUT_STYLE)
        self._repo_input.setMinimumHeight(42)
        layout.addWidget(self._repo_input)

        # Status + progress
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: {COLOR_BG}; border-radius: 3px; border: none; }}"
            f"QProgressBar::chunk {{ background: {COLOR_HIGHLIGHT}; border-radius: 3px; }}"
        )
        layout.addWidget(self._progress)

        layout.addSpacing(4)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_BTN_OUTLINE)
        cancel_btn.setMinimumHeight(40)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        btn_row.addSpacing(10)

        self._connect_btn = QPushButton("Validate & Connect")
        self._connect_btn.setStyleSheet(_BTN_PRIMARY)
        self._connect_btn.setMinimumHeight(40)
        self._connect_btn.clicked.connect(self._on_connect)
        btn_row.addWidget(self._connect_btn)

        layout.addLayout(btn_row)

    def _on_connect(self):
        token = self._token_input.text().strip()
        repo = self._repo_input.text().strip()

        if not token or not repo:
            self._set_status("⚠ Please fill in both fields.", COLOR_HIGHLIGHT)
            return
        if "/" not in repo:
            self._set_status("⚠ Repository must be: username/repo-name", COLOR_HIGHLIGHT)
            return

        self._connect_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._set_status("Validating token…", COLOR_TEXT_MUTED)

        self._worker = HFValidateWorker(repo, token, create_if_missing=True)
        self._worker.finished.connect(self._on_validate_done)
        self._worker.start()

    def _on_validate_done(self, success: bool, msg: str):
        self._progress.setVisible(False)
        self._connect_btn.setEnabled(True)
        if success:
            self._token = self._token_input.text().strip()
            self._repo = self._repo_input.text().strip()
            self._set_status(f"✓ Connected as {msg}", COLOR_SUCCESS)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(700, self.accept)
        else:
            self._set_status(f"✗ {msg}", COLOR_HIGHLIGHT)

    def _set_status(self, text: str, color: str):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color}; font-size: 12px;")

    def get_result(self):
        return self._token, self._repo


# ── Welcome Screen ─────────────────────────────────────────────────────────────

class WelcomeScreen(QWidget):
    """
    Opening screen. Emits dataset_opened(DatasetManager) when a dataset
    is created or opened successfully.
    """

    dataset_opened = pyqtSignal(object)  # DatasetManager

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COLOR_BG};")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setStyleSheet(
            f"background: {COLOR_SURFACE}; border-bottom: 1px solid {COLOR_ACCENT};"
        )
        top_bar.setFixedHeight(56)
        tbl = QHBoxLayout(top_bar)
        tbl.setContentsMargins(28, 0, 28, 0)

        tbl.addStretch()

        ver = QLabel(f"v{APP_VERSION}")
        ver.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px; background: transparent;")
        tbl.addWidget(ver)
        outer.addWidget(top_bar)

        # ── Main content ──────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background: {COLOR_BG};")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(80, 60, 80, 40)
        cl.setSpacing(0)
        cl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Heading
        h1 = QLabel("Welcome to QuickLabel")
        h1.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 34px; font-weight: 700; background: transparent;"
        )
        h1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(h1)
        cl.addSpacing(10)

        sub = QLabel(
            "Build COCO-format object detection datasets collaboratively.\n"
            "Annotate with bounding boxes. Sync your team via Hugging Face Hub."
        )
        sub.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 14px; background: transparent;"
        )
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        cl.addWidget(sub)
        cl.addSpacing(40)

        # ── Resume banner (shown only if a previous session exists) ───────────
        last = load_session()
        if last:
            resume_frame = QFrame()
            resume_frame.setStyleSheet(
                f"QFrame {{ background: {COLOR_SURFACE}; border: 2px solid {COLOR_SUCCESS}; "
                f"border-radius: 12px; }}"
            )
            resume_frame.setMaximumWidth(820)
            rf_layout = QHBoxLayout(resume_frame)
            rf_layout.setContentsMargins(20, 14, 20, 14)

            resume_icon = QLabel("🔄")
            resume_icon.setStyleSheet("font-size: 22px; background: transparent; border: none;")
            rf_layout.addWidget(resume_icon)

            resume_lbl = QLabel(f"<b>Resume last session</b><br>"
                                f"<span style='color:{COLOR_TEXT_MUTED}; font-size:12px;'>{last}</span>")
            resume_lbl.setStyleSheet(
                f"color: {COLOR_TEXT}; font-size: 13px; background: transparent; border: none;"
            )
            resume_lbl.setWordWrap(True)
            rf_layout.addWidget(resume_lbl, 1)

            resume_btn = QPushButton("  Resume  ")
            resume_btn.setStyleSheet(_BTN_OUTLINE)
            resume_btn.setMinimumHeight(38)
            resume_btn.clicked.connect(lambda: self._open_folder(last))
            rf_layout.addWidget(resume_btn)

            center_wrap = QHBoxLayout()
            center_wrap.addStretch()
            center_wrap.addWidget(resume_frame)
            center_wrap.addStretch()
            cl.addLayout(center_wrap)
            cl.addSpacing(32)

        # ── Cards row ─────────────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(24)
        cards_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        local_card = ClickableCard(
            "📁",
            "New Local Dataset",
            "Create a dataset stored only on this machine.\nPerfect for solo work or offline use.",
        )
        local_card.clicked.connect(self._on_new_local)
        cards_row.addWidget(local_card)

        hf_card = ClickableCard(
            "🤗",
            "Hugging Face Dataset",
            "Connect to an HF Hub repository.\nCreates it if it doesn't exist yet. Perfect for team collaboration.",
        )
        hf_card.clicked.connect(self._on_hf_dataset)
        cards_row.addWidget(hf_card)

        open_card = ClickableCard(
            "📂",
            "Open Local Dataset",
            "Resume work on an existing QuickLabel dataset folder stored on this PC.",
        )
        open_card.clicked.connect(self._on_open)
        cards_row.addWidget(open_card)

        cl.addLayout(cards_row)
        cl.addStretch()
        outer.addWidget(content, 1)

        # ── Footer ────────────────────────────────────────────────────────────
        footer = QLabel("COCO JSON format · JPEG 640×480 · Detic / YOLO ready")
        footer.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px; padding: 12px; background: transparent;"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(footer)

    # ─── Handlers ──────────────────────────────────────────────────────────────

    def _on_new_local(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder for new dataset", os.path.expanduser("~")
        )
        if not folder:
            return

        name, ok = self._ask_name("New Local Dataset", "Dataset name:")
        if not ok or not name.strip():
            return

        dm = DatasetManager()
        try:
            dm.create_local(folder, name.strip())
        except Exception as exc:
            self._show_error("Failed to create dataset", str(exc))
            return

        save_session(folder)
        self.dataset_opened.emit(dm)

    def _on_hf_dataset(self):
        """Handle both new and existing HF datasets."""
        dialog = HFSetupDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        token, repo = dialog.get_result()
        if not token or not repo:
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select local folder for the dataset", os.path.expanduser("~")
        )
        if not folder:
            return

        name, ok = self._ask_name(
            "Dataset Name",
            "Dataset name:",
            default=repo.split("/")[-1],
        )
        if not ok or not name.strip():
            return

        token_key = repo.replace("/", "_").replace("-", "_")
        try:
            store_token(token_key, token)
        except Exception as exc:
            self._show_error("Keyring error", f"Could not store token: {exc}")
            return

        dm = DatasetManager()
        try:
            dm.create_synced(folder, name.strip(), repo, token_key)
        except Exception as exc:
            self._show_error("Failed to create dataset", str(exc))
            return

        save_session(folder)
        self.dataset_opened.emit(dm)

    def _on_open(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Open existing dataset folder", os.path.expanduser("~")
        )
        if not folder:
            return
        self._open_folder(folder)

    def _open_folder(self, folder: str):
        dm = DatasetManager()
        try:
            dm.open(folder)
        except FileNotFoundError as exc:
            self._show_error("Not a QuickLabel dataset", str(exc))
            return
        except ValueError as exc:
            self._show_error("Corrupt dataset", str(exc))
            return
        except Exception as exc:
            self._show_error("Failed to open dataset", str(exc))
            return

        save_session(folder)
        self.dataset_opened.emit(dm)

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _ask_name(self, title: str, label: str, default: str = ""):
        from PyQt6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, label, text=default)

    def _show_error(self, title: str, message: str):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setStyleSheet(
            f"background: {COLOR_SURFACE}; color: {COLOR_TEXT}; "
            f"font-family: 'Segoe UI', Inter, Arial;"
        )
        msg.exec()
