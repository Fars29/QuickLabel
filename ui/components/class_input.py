"""
ui/components/class_input.py — Class name input with fuzzy autocomplete.

Design decisions:
- The dropdown is a QListWidget rendered as a floating overlay over the parent
  widget, not inside a layout. This avoids layout jitter when the list appears.
- Fuzzy matching uses difflib.get_close_matches (no extra dep, cutoff=0.3).
- The "new class" warning label is shown inline below the input.
- Pressing Enter or clicking a list item confirms the selection.
- The widget emits class_confirmed(str) when the user finalises their choice.
"""

from __future__ import annotations

import difflib
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_HIGHLIGHT,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_WARNING,
)

_STYLE_INPUT = f"""
QLineEdit {{
    background: rgba(0, 0, 0, 0.2);
    color: {COLOR_TEXT};
    border: 2px solid {COLOR_ACCENT};
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 13px;
    font-family: "Segoe UI", Inter, Arial;
}}
QLineEdit:focus {{
    background: rgba(0, 0, 0, 0.3);
    border-color: {COLOR_HIGHLIGHT};
}}
"""

_STYLE_DROPDOWN = f"""
QListWidget {{
    background: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 2px solid {COLOR_HIGHLIGHT};
    border-radius: 8px;
    font-size: 13px;
    font-family: "Segoe UI", Inter, Arial;
    padding: 4px;
    outline: 0;
}}
QListWidget::item {{
    padding: 2px 8px;
    border-radius: 4px;
}}
QListWidget::item:hover {{
    background: {COLOR_ACCENT};
}}
QListWidget::item:selected {{
    background: {COLOR_HIGHLIGHT};
    color: white;
}}
"""


class ClassInput(QWidget):
    """
    A text input with live fuzzy-autocomplete for class names.

    Signals:
        class_confirmed(str): Emitted when user confirms a class name.
    """

    class_confirmed = pyqtSignal(str)
    text_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._known_classes: list[str] = []
        self._confirmed_name: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Input field
        self._input = QLineEdit()
        self._input.setPlaceholderText("Class name (e.g. Cardboard)…")
        self._input.setStyleSheet(_STYLE_INPUT)
        self._input.setMinimumHeight(38)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.returnPressed.connect(self._on_enter)
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

        # Status label
        self._status_label = QLabel()
        self._status_label.setStyleSheet(
            f"color: {COLOR_WARNING}; font-size: 11px; padding-left: 4px;"
        )
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Dropdown (floating overlay on the main window)
        self._dropdown = QListWidget()
        # Use ToolTip flag to ensure it overlays without stealing focus
        self._dropdown.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint)
        self._dropdown.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._dropdown.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._dropdown.setStyleSheet(_STYLE_DROPDOWN + """
            QScrollBar:vertical {
                background: #0d0e17;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: #1e2d5a;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {COLOR_HIGHLIGHT};
            }}
        """)
        self._dropdown.setFixedWidth(300)
        self._dropdown.setVisible(False)
        self._dropdown.itemClicked.connect(self._on_item_clicked)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def eventFilter(self, obj, event):
        if obj is self._input:
            if event.type() == event.Type.FocusOut:
                # Delay check slightly to allow dropdown clicks to register
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(100, self._check_focus_out)
            elif event.type() == event.Type.KeyPress:
                if self._dropdown.isVisible():
                    if event.key() == Qt.Key.Key_Down:
                        curr = self._dropdown.currentRow()
                        self._dropdown.setCurrentRow(min(curr + 1, self._dropdown.count() - 1))
                        return True
                    if event.key() == Qt.Key.Key_Up:
                        curr = self._dropdown.currentRow()
                        self._dropdown.setCurrentRow(max(curr - 1, 0))
                        return True
                    if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                        if self._dropdown.currentRow() >= 0:
                            item = self._dropdown.currentItem()
                            if item:
                                self._on_item_clicked(item)
                                return True
                    if event.key() == Qt.Key.Key_Escape:
                        self._dropdown.setVisible(False)
                        return True
        return super().eventFilter(obj, event)

    def _check_focus_out(self):
        if not self._dropdown.isVisible() or not self._dropdown.underMouse():
            text = self._input.text().strip()
            if text and text != self._confirmed_name:
                self._confirm(text)

    # ─── Public API ────────────────────────────────────────────────────────────

    def set_known_classes(self, classes: list[str]) -> None:
        """Update the list of known classes for autocomplete."""
        self._known_classes = sorted(classes)

    def get_class_name(self) -> str:
        """Return the current text in the input field (trimmed)."""
        return self._input.text().strip()

    def get_confirmed_name(self) -> Optional[str]:
        """Return the confirmed class name, or None if not yet confirmed."""
        return self._confirmed_name

    def clear(self) -> None:
        """Reset the input and status."""
        self._input.clear()
        self._confirmed_name = None
        self._status_label.setVisible(False)
        self._dropdown.setVisible(False)

    def set_class_name(self, name: str) -> None:
        """Programmatically set the class name and confirm it."""
        self._input.setText(name)
        self._confirm(name)

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _on_text_changed(self, text: str) -> None:
        text = text.strip()
        self._confirmed_name = None  # Reset confirmation when text changes
        self.text_changed.emit(text)

        if not text:
            self._dropdown.setVisible(False)
            self._status_label.setVisible(False)
            return

        # Fuzzy match (more results allowed now)
        matches = difflib.get_close_matches(
            text, self._known_classes, n=15, cutoff=0.2
        )

        # Also include exact prefix matches
        prefix_matches = [c for c in self._known_classes if c.lower().startswith(text.lower())]
        for pm in prefix_matches:
            if pm not in matches:
                matches.insert(0, pm)

        # Update dropdown
        self._dropdown.clear()
        for match in matches:
            item = QListWidgetItem(match)
            self._dropdown.addItem(item)

        if matches:
            self._position_and_size_dropdown(len(matches))
            self._dropdown.show()
        else:
            self._dropdown.setVisible(False)

        # Status label
        exact = any(c.lower() == text.lower() for c in self._known_classes)
        if exact:
            matched = next(c for c in self._known_classes if c.lower() == text.lower())
            self._status_label.setText(f"\u2713 Will be added to existing class '{matched}'")
            self._status_label.setStyleSheet(
                f"color: {COLOR_SUCCESS}; font-size: 11px; padding-left: 4px;"
            )
        else:
            self._status_label.setText("⚠ New class — will be created")
            self._status_label.setStyleSheet(
                f"color: {COLOR_WARNING}; font-size: 11px; padding-left: 4px;"
            )
        self._status_label.setVisible(True)

    def _on_enter(self) -> None:
        text = self._input.text().strip()
        if text:
            self._confirm(text)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        name = item.text()
        self._input.setText(name)
        self._confirm(name)

    def _confirm(self, name: str) -> None:
        self._confirmed_name = name
        self._dropdown.setVisible(False)
        self._status_label.setVisible(False)
        self.class_confirmed.emit(name)

    def _position_and_size_dropdown(self, num_items: int) -> None:
        """Position the dropdown below the input field and adjust its height."""
        # Calculate height: ~32px per item + small padding
        item_h = 32
        total_h = min(400, (num_items * item_h) + 10)
        self._dropdown.setFixedHeight(total_h)
        
        global_pos = self._input.mapToGlobal(self._input.rect().bottomLeft())
        # No vertical gap, flush with the bottom of the input
        self._dropdown.move(global_pos.x(), global_pos.y())
        self._dropdown.setFixedWidth(self._input.width())
        self._dropdown.raise_()

    def hideEvent(self, event) -> None:
        self._dropdown.setVisible(False)
        super().hideEvent(event)
