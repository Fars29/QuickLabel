"""
ui/components/bbox_canvas.py — Interactive bounding box drawing and editing canvas.

Design decisions:
- Uses QGraphicsScene / QGraphicsView for pixel-accurate rendering and transforms.
- Each bounding box is a BBoxItem (QGraphicsRectItem subclass) with 8 resize handles
  (QGraphicsEllipseItem), all grouped via QGraphicsItemGroup.
- Coordinate space: the scene coordinate IS the image pixel coordinate. The view
  applies a scale transform so the image fills the canvas while maintaining aspect
  ratio. All coordinates exposed via get_annotations() are in image space.
- Box states: DRAWING, CONFIRMED, PROPAGATED, SELECTED — each has a distinct color.
- Mouse events:
    - Press on empty area → start DRAWING
    - Press on handle → resize
    - Press on box center → move
  Cursor changes automatically based on what's under the mouse.
- Press Del → delete selected box.
- Right-click → context menu → Delete.
- The canvas emits annotations_changed() when any box is added/removed/moved.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QCursor,
    QKeyEvent,
    QMouseEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMenu,
    QWidget,
)

from config import (
    BBOX_COLOR_CONFIRMED,
    BBOX_COLOR_DRAWING,
    BBOX_COLOR_PROPAGATED,
    BBOX_COLOR_SELECTED,
    COLOR_BG,
)
from core.image_processor import _to_pil

_HANDLE_RADIUS = 10.0
_HANDLE_POSITIONS = [
    (0.0, 0.0),   # top-left
    (0.5, 0.0),   # top-center
    (1.0, 0.0),   # top-right
    (1.0, 0.5),   # right-center
    (1.0, 1.0),   # bottom-right
    (0.5, 1.0),   # bottom-center
    (0.0, 1.0),   # bottom-left
    (0.0, 0.5),   # left-center
]

# Cursor for each handle index
_HANDLE_CURSORS = [
    Qt.CursorShape.SizeFDiagCursor,   # TL
    Qt.CursorShape.SizeVerCursor,     # TC
    Qt.CursorShape.SizeBDiagCursor,   # TR
    Qt.CursorShape.SizeHorCursor,     # RC
    Qt.CursorShape.SizeFDiagCursor,   # BR
    Qt.CursorShape.SizeVerCursor,     # BC
    Qt.CursorShape.SizeBDiagCursor,   # BL
    Qt.CursorShape.SizeHorCursor,     # LC
]


class BoxState(Enum):
    DRAWING = auto()
    CONFIRMED = auto()
    PROPAGATED = auto()
    SELECTED = auto()


def _state_color(state: BoxState) -> str:
    mapping = {
        BoxState.DRAWING: BBOX_COLOR_DRAWING,
        BoxState.CONFIRMED: BBOX_COLOR_CONFIRMED,
        BoxState.PROPAGATED: BBOX_COLOR_PROPAGATED,
        BoxState.SELECTED: BBOX_COLOR_SELECTED,
    }
    return mapping.get(state, BBOX_COLOR_CONFIRMED)


class HandleItem(QGraphicsEllipseItem):
    """A small circle handle on a BBoxItem for resizing."""

    def __init__(self, index: int, parent: "BBoxItem") -> None:
        r = _HANDLE_RADIUS
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.handle_index = index
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(_HANDLE_CURSORS[index]))
        self._update_appearance(BBOX_COLOR_CONFIRMED)

    def _update_appearance(self, color: str) -> None:
        pen = QPen(QColor(color), 1.5)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(color)))

    def set_color(self, color: str) -> None:
        self._update_appearance(color)

    def position_at(self, rx: float, ry: float, rect: QRectF) -> None:
        """Move handle to fractional position (rx, ry) within rect."""
        x = rect.left() + rx * rect.width()
        y = rect.top() + ry * rect.height()
        self.setPos(x, y)


class BBoxItem(QGraphicsRectItem):
    """
    A single bounding box on the canvas.
    Manages its own 8 resize handles.
    """

    def __init__(
        self,
        rect: QRectF,
        state: BoxState = BoxState.CONFIRMED,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(rect, parent)
        self._state = state
        self._handles: list[HandleItem] = []
        self._drag_handle_idx: Optional[int] = None
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_rect: Optional[QRectF] = None

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))

        self._create_handles()
        self._apply_style()

    def set_state(self, state: BoxState) -> None:
        self._state = state
        self._apply_style()

    def get_state(self) -> BoxState:
        return self._state

    def _apply_style(self) -> None:
        color_str = _state_color(self._state)
        color = QColor(color_str)

        if self._state == BoxState.DRAWING:
            pen = QPen(color, 4, Qt.PenStyle.DashLine)
        else:
            pen = QPen(color, 4)

        self.setPen(pen)
        self.setBrush(QBrush(QColor(0, 0, 0, 0)))  # Transparent fill

        for h in self._handles:
            h.set_color(color_str)
            h.setVisible(self._state == BoxState.SELECTED)

        self._update_handle_positions()

    def _create_handles(self) -> None:
        for idx, (rx, ry) in enumerate(_HANDLE_POSITIONS):
            h = HandleItem(idx, self)
            self._handles.append(h)
        self._update_handle_positions()

    def _update_handle_positions(self) -> None:
        rect = self.rect()
        for h, (rx, ry) in zip(self._handles, _HANDLE_POSITIONS):
            h.position_at(rx, ry, rect)

    # ─── Bbox in image coords ──────────────────────────────────────────────────

    def get_xywh(self) -> list[float]:
        """Return [x, y, w, h] in scene (image pixel) coordinates."""
        scene_rect = self.mapToScene(self.rect()).boundingRect()
        return [
            scene_rect.x(),
            scene_rect.y(),
            scene_rect.width(),
            scene_rect.height(),
        ]

    def set_rect_from_xywh(self, x: float, y: float, w: float, h: float) -> None:
        self.setRect(QRectF(x, y, w, h))
        self._update_handle_positions()

    # ─── Mouse interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if pressing on a handle
            for idx, h in enumerate(self._handles):
                if h.contains(self.mapFromScene(event.scenePos())):
                    self._drag_handle_idx = idx
                    self._drag_start_pos = event.scenePos()
                    self._drag_start_rect = self.rect()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_handle_idx is not None and self._drag_start_rect is not None:
            delta = event.scenePos() - self._drag_start_pos  # type: ignore[operator]
            self._resize_by_handle(self._drag_handle_idx, delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._drag_handle_idx = None
        self._drag_start_pos = None
        self._drag_start_rect = None
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_handle_positions()
        return super().itemChange(change, value)

    def _resize_by_handle(self, idx: int, delta: QPointF) -> None:
        """Resize the rect by dragging a specific handle."""
        r = QRectF(self._drag_start_rect)  # type: ignore[arg-type]
        dx, dy = delta.x(), delta.y()

        # Handle index → which edges to move
        # TL=0, TC=1, TR=2, RC=3, BR=4, BC=5, BL=6, LC=7
        if idx == 0:   r.setTopLeft(r.topLeft() + QPointF(dx, dy))
        elif idx == 1: r.setTop(r.top() + dy)
        elif idx == 2: r.setTopRight(r.topRight() + QPointF(dx, dy))
        elif idx == 3: r.setRight(r.right() + dx)
        elif idx == 4: r.setBottomRight(r.bottomRight() + QPointF(dx, dy))
        elif idx == 5: r.setBottom(r.bottom() + dy)
        elif idx == 6: r.setBottomLeft(r.bottomLeft() + QPointF(dx, dy))
        elif idx == 7: r.setLeft(r.left() + dx)

        # Enforce minimum size
        if r.width() < 4:
            r.setWidth(4)
        if r.height() < 4:
            r.setHeight(4)

        self.setRect(r.normalized())
        self._update_handle_positions()


# ─── Main Canvas ───────────────────────────────────────────────────────────────

class BBoxCanvas(QGraphicsView):
    """
    The main annotation canvas.

    Signals:
        annotations_changed(): Emitted when boxes are added/modified/deleted.
        box_selected(int): Index of selected box (-1 = none).
    """

    annotations_changed = pyqtSignal()
    box_selected = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._boxes: list[BBoxItem] = []
        self._selected_box: Optional[BBoxItem] = None
        self._drawing = False
        self._draw_start: Optional[QPointF] = None
        self._draw_rect_item: Optional[BBoxItem] = None
        self._image_size: tuple[int, int] = (640, 480)  # (w, h)

        self._setup_view()

    def _setup_view(self) -> None:
        from PyQt6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet(f"background: {COLOR_BG}; border: none;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ─── Image Loading ─────────────────────────────────────────────────────────

    def load_image(self, source) -> None:
        """
        Load an image into the canvas. Clears existing boxes.
        source can be: numpy array (BGR or RGB), PIL Image, str/Path.
        """
        self._scene.clear()
        self._boxes = []
        self._selected_box = None
        self._draw_rect_item = None

        pixmap = self._source_to_pixmap(source)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._image_size = (pixmap.width(), pixmap.height())
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._fit_view()

    def _source_to_pixmap(self, source) -> QPixmap:
        """Convert any image source to QPixmap."""
        from PyQt6.QtGui import QImage
        import numpy as np

        if isinstance(source, QPixmap):
            return source

        try:
            pil_img = _to_pil(source)
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            data = pil_img.tobytes("raw", "RGB")
            q_img = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(q_img)
        except Exception:
            # Blank fallback
            px = QPixmap(640, 480)
            px.fill(QColor(COLOR_BG))
            return px

    def _fit_view(self) -> None:
        """Scale view so image fills the canvas preserving aspect ratio."""
        if self._pixmap_item is None:
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._fit_view()

    # ─── Annotation Management ─────────────────────────────────────────────────

    def set_annotations(
        self,
        bboxes_xywh: list[list[float]],
        states: Optional[list[BoxState]] = None,
    ) -> None:
        """
        Set the boxes on the canvas (replacing any existing boxes).
        bboxes_xywh: list of [x, y, w, h] in image pixel coords.
        states: parallel list of BoxState (default: CONFIRMED).
        """
        # Remove existing box items but keep pixmap
        for box in self._boxes:
            self._scene.removeItem(box)
        self._boxes = []
        self._selected_box = None

        if states is None:
            states = [BoxState.CONFIRMED] * len(bboxes_xywh)

        for bbox, state in zip(bboxes_xywh, states):
            x, y, w, h = bbox
            rect = QRectF(x, y, w, h)
            item = BBoxItem(rect, state)
            self._scene.addItem(item)
            self._boxes.append(item)

        self.annotations_changed.emit()

    def get_annotations(self) -> list[list[float]]:
        """Return all boxes as list of [x, y, w, h] in image pixel coords."""
        result = []
        for box in self._boxes:
            if box.get_state() == BoxState.DRAWING:
                continue
            xywh = box.get_xywh()
            # Clamp to image bounds
            x, y, w, h = xywh
            iw, ih = self._image_size
            x = max(0.0, min(x, float(iw)))
            y = max(0.0, min(y, float(ih)))
            w = max(1.0, min(w, float(iw) - x))
            h = max(1.0, min(h, float(ih) - y))
            result.append([x, y, w, h])
        return result

    def get_annotation_states(self) -> list[BoxState]:
        """Return state for each box (parallel to get_annotations())."""
        return [
            box.get_state()
            for box in self._boxes
            if box.get_state() != BoxState.DRAWING
        ]

    def clear_boxes(self) -> None:
        """Remove all bounding boxes from the canvas."""
        for box in self._boxes:
            self._scene.removeItem(box)
        self._boxes = []
        self._selected_box = None
        self.annotations_changed.emit()

    def delete_selected(self) -> None:
        """Delete the currently selected box."""
        if self._selected_box and self._selected_box in self._boxes:
            self._scene.removeItem(self._selected_box)
            idx = self._boxes.index(self._selected_box)
            self._boxes.remove(self._selected_box)
            self._selected_box = None
            self.box_selected.emit(-1)
            self.annotations_changed.emit()

    def select_box(self, index: int) -> None:
        """Select a box by index (for right-panel sync)."""
        if 0 <= index < len(self._boxes):
            self._deselect_all()
            self._boxes[index].set_state(BoxState.SELECTED)
            self._selected_box = self._boxes[index]
            self.box_selected.emit(index)



    def _deselect_all(self) -> None:
        for box in self._boxes:
            if box.get_state() == BoxState.SELECTED:
                box.set_state(BoxState.CONFIRMED)
        self._selected_box = None

    # ─── Mouse Events ──────────────────────────────────────────────────────────

    def _scene_pos(self, event: QMouseEvent) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _hit_box(self, scene_pos: QPointF) -> Optional[BBoxItem]:
        """Return the topmost BBoxItem under scene_pos, or None."""
        items = self._scene.items(scene_pos)
        for item in items:
            if isinstance(item, BBoxItem):
                return item
            if isinstance(item, HandleItem):
                return item.parentItem()  # type: ignore[return-value]
        return None

    def _hit_handle(self, scene_pos: QPointF, box: BBoxItem) -> int:
        """Return handle index under scene_pos on box, or -1."""
        for idx, h in enumerate(box._handles):
            h_scene = box.mapToScene(h.pos())
            if (scene_pos - h_scene).manhattanLength() <= _HANDLE_RADIUS * 2:
                return idx
        return -1

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._pixmap_item is None:
            return

        scene_pos = self._scene_pos(event)

        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_box(scene_pos)

            if hit is not None:
                # Check if clicking a handle
                handle_idx = self._hit_handle(scene_pos, hit)
                if handle_idx >= 0:
                    # Handle drag → let BBoxItem handle it
                    hit._drag_handle_idx = handle_idx
                    hit._drag_start_pos = scene_pos
                    hit._drag_start_rect = hit.rect()
                    self._select_box_item(hit)
                    event.accept()
                    return

                # Clicking body → select and allow move
                self._select_box_item(hit)
                super().mousePressEvent(event)
                return

            # No hit → start drawing
            self._deselect_all()
            self._drawing = True
            self._draw_start = scene_pos
            draw_rect = BBoxItem(QRectF(scene_pos, scene_pos), BoxState.DRAWING)
            self._scene.addItem(draw_rect)
            self._draw_rect_item = draw_rect

        elif event.button() == Qt.MouseButton.RightButton:
            hit = self._hit_box(scene_pos)
            if hit is not None:
                self._select_box_item(hit)
                self._show_context_menu(event.globalPosition().toPoint())
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        scene_pos = self._scene_pos(event)

        if self._drawing and self._draw_rect_item and self._draw_start:
            # Update the in-progress rectangle
            rect = QRectF(self._draw_start, scene_pos).normalized()
            self._draw_rect_item.setRect(rect)
            self._draw_rect_item._update_handle_positions()
            event.accept()
            return

        # Handle resize drag (delegated to BBoxItem)
        if self._selected_box and self._selected_box._drag_handle_idx is not None:
            delta = scene_pos - self._selected_box._drag_start_pos  # type: ignore[operator]
            self._selected_box._resize_by_handle(
                self._selected_box._drag_handle_idx, delta
            )
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drawing and self._draw_rect_item and self._draw_start:
            scene_pos = self._scene_pos(event)
            rect = QRectF(self._draw_start, scene_pos).normalized()

            # Minimum box size
            if rect.width() < 4 or rect.height() < 4:
                self._scene.removeItem(self._draw_rect_item)
            else:
                self._draw_rect_item.setRect(rect)
                self._draw_rect_item.set_state(BoxState.CONFIRMED)
                self._boxes.append(self._draw_rect_item)
                self._select_box_item(self._draw_rect_item)
                self.annotations_changed.emit()

            self._drawing = False
            self._draw_start = None
            self._draw_rect_item = None
            event.accept()
            return

        if self._selected_box:
            self._selected_box._drag_handle_idx = None
            self._selected_box._drag_start_pos = None
            self._selected_box._drag_start_rect = None
            self.annotations_changed.emit()

        super().mouseReleaseEvent(event)

    def _select_box_item(self, item: BBoxItem) -> None:
        self._deselect_all()
        prev_state = item.get_state()
        item.set_state(BoxState.SELECTED)
        self._selected_box = item
        idx = self._boxes.index(item) if item in self._boxes else -1
        self.box_selected.emit(idx)

    # ─── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_D, Qt.Key.Key_Delete):
            self.delete_selected()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._deselect_all()
            self._selected_box = None
            self.box_selected.emit(-1)
            return
        super().keyPressEvent(event)

    # ─── Context Menu ──────────────────────────────────────────────────────────

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            f"""
            QMenu {{
                background: #16213e;
                color: #e8e8f0;
                border: 1px solid #0f3460;
                border-radius: 6px;
                padding: 4px;
                font-family: "Segoe UI", Inter, Arial;
                font-size: 13px;
            }}
            QMenu::item {{ padding: 6px 20px 6px 12px; border-radius: 4px; }}
            QMenu::item:selected {{ background: #e94560; }}
            """
        )
        delete_action = menu.addAction("🗑  Delete Box")
        action = menu.exec(global_pos)
        if action == delete_action:
            self.delete_selected()

    # ─── Zoom ──────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        """Zoom in/out with Ctrl+Scroll."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)
