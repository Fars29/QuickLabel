"""
core/tracker.py — CSRT bounding box tracker for frame-to-frame propagation.

Best practice (2024):
- One tracker instance per bbox being tracked.
- init() once with the source frame + bbox, then update() on each subsequent frame.
- Frames must be the same size between init() and update() — we handle this by
  tracking in a shared "working" resolution (original image size) rather than
  the display size.
- CSRT does not expose a raw confidence score; we use an area-change heuristic.
- If update() returns False (lost tracking), fall back to the last known bbox.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import TRACKER_AREA_CHANGE_THRESHOLD

log = logging.getLogger(__name__)

# ── OpenCV availability check ──────────────────────────────────────────────────
try:
    import cv2
    # Check for CSRT in main namespace (modern opencv-contrib)
    if hasattr(cv2, "TrackerCSRT_create"):
        _TRACKER_FACTORY = cv2.TrackerCSRT_create
    elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        _TRACKER_FACTORY = cv2.legacy.TrackerCSRT_create
    else:
        _TRACKER_FACTORY = None

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    _TRACKER_FACTORY = None

_CSRT_AVAILABLE = _CV2_AVAILABLE and (_TRACKER_FACTORY is not None)


def _make_tracker():
    """Instantiate a fresh CSRT tracker."""
    if _TRACKER_FACTORY is None:
        raise RuntimeError(
            "CSRT tracker unavailable. Install: pip install opencv-contrib-python"
        )
    return _TRACKER_FACTORY()


# ── Single-box tracker ─────────────────────────────────────────────────────────

class BBoxTracker:
    """
    Wraps the OpenCV CSRT tracker for a single bounding box.

    All coordinates are [x, y, w, h] in image-pixel space (COCO format).
    """

    def __init__(self) -> None:
        self._tracker = None
        self._last_bbox: Optional[list[float]] = None
        self._init_area: float = 0.0

    @property
    def available(self) -> bool:
        return _CSRT_AVAILABLE

    def init(self, frame_bgr: np.ndarray, bbox_xywh: list[float]) -> bool:
        """
        Initialise tracker on frame_bgr with bbox_xywh=[x,y,w,h].
        Must be called before update(). Returns True on success.
        """
        if not _CSRT_AVAILABLE:
            log.warning("CSRT tracker not available; skipping tracking")
            return False

        try:
            self._tracker = _make_tracker()
            fh, fw = frame_bgr.shape[:2]
            x, y, w, h = (int(round(v)) for v in bbox_xywh)
            # Clamp strictly inside frame
            x = max(0, min(x, fw - 2))
            y = max(0, min(y, fh - 2))
            w = max(2, min(w, fw - x))
            h = max(2, min(h, fh - y))

            # In OpenCV >= 4.5.1, tracker.init() returns None on success, not True.
            # In older OpenCV versions (legacy), it returns True/False.
            ok = self._tracker.init(frame_bgr, (x, y, w, h))
            if ok is None or ok:
                self._last_bbox = [float(x), float(y), float(w), float(h)]
                self._init_area = float(w * h)
                log.debug("Tracker init OK: bbox=%s, frame=%dx%d", (x, y, w, h), fw, fh)
                return True
            else:
                log.warning("Tracker init returned False for bbox=%s", (x, y, w, h))
                return False
        except Exception as exc:
            log.error("Tracker init exception: %s", exc)
            self._tracker = None
            return False

    def update(self, frame_bgr: np.ndarray) -> tuple[list[float], bool]:
        """
        Track on the next frame. Returns (bbox_xywh, is_confident).
        Falls back to last known bbox if tracking fails.
        """
        if self._tracker is None or self._last_bbox is None:
            return self._last_bbox or [0.0, 0.0, 50.0, 50.0], False

        try:
            ok, rect = self._tracker.update(frame_bgr)
        except Exception as exc:
            log.error("Tracker update exception: %s", exc)
            return self._last_bbox, False

        if not ok or rect is None:
            log.debug("Tracker lost object; using last bbox")
            return self._last_bbox, False

        x, y, w, h = (float(v) for v in rect)

        # Confidence: reject if area changed dramatically
        new_area = w * h
        if self._init_area > 0:
            ratio = abs(new_area - self._init_area) / self._init_area
            confident = ratio < TRACKER_AREA_CHANGE_THRESHOLD
        else:
            confident = True

        new_bbox = [x, y, w, h]
        self._last_bbox = new_bbox
        log.debug("Tracker update: bbox=%s confident=%s", new_bbox, confident)
        return new_bbox, confident

    def reset(self) -> None:
        self._tracker = None
        self._last_bbox = None
        self._init_area = 0.0


# ── Multi-box tracker ─────────────────────────────────────────────────────────

class MultiBBoxTracker:
    """One CSRT tracker per bounding box."""

    def __init__(self) -> None:
        self._trackers: list[BBoxTracker] = []

    def init(self, frame_bgr: np.ndarray, bboxes_xywh: list[list[float]]) -> bool:
        """Init one tracker per bbox. Returns True if ALL succeeded."""
        self._trackers = []
        all_ok = True
        for bbox in bboxes_xywh:
            t = BBoxTracker()
            ok = t.init(frame_bgr, bbox)
            if not ok:
                all_ok = False
                log.warning("Tracker init failed for bbox=%s", bbox)
            self._trackers.append(t)
        return all_ok and len(self._trackers) > 0

    def update(self, frame_bgr: np.ndarray) -> list[tuple[list[float], bool]]:
        """Update all trackers; returns list of (bbox, confident)."""
        return [t.update(frame_bgr) for t in self._trackers]

    def reset(self) -> None:
        for t in self._trackers:
            t.reset()
        self._trackers = []

    @property
    def count(self) -> int:
        return len(self._trackers)
