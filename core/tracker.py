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
import config
from config import TRACKER_AREA_CHANGE_THRESHOLD, ENABLE_GRABCUT_REFINEMENT

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
    """Instantiate a fresh CSRT tracker with tuned parameters."""
    if _TRACKER_FACTORY is None:
        raise RuntimeError(
            "CSRT tracker unavailable. Install: pip install opencv-contrib-python"
        )
    
    try:
        # Check for Params class location
        if hasattr(cv2, "TrackerCSRT_Params"):
            params = cv2.TrackerCSRT_Params()
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_Params"):
            params = cv2.legacy.TrackerCSRT_Params()
        else:
            return _TRACKER_FACTORY()

        params.use_segmentation = True
        params.use_channel_weights = True
        params.psr_threshold = 0.06
        params.padding = 2.0
        params.filter_lr = 0.02
        params.num_hog_channels_used = 18
        params.window_function = "hann"
        
        return _TRACKER_FACTORY(params)
    except Exception as exc:
        log.warning("Failed to init CSRT params, using defaults: %s", exc)
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
        new_bbox = [x, y, w, h]
        self._last_bbox = new_bbox

        # Baseline confidence (CSRT area check)
        confident = True
        if self._init_area > 0:
            area_ratio = abs((w * h) - self._init_area) / self._init_area
            confident = area_ratio < TRACKER_AREA_CHANGE_THRESHOLD

        # ── Refinement (GrabCut) ──────────────────────────────────────────────
        if ENABLE_GRABCUT_REFINEMENT:
            refined_bbox, refined_ok = self._refine_grabcut(frame_bgr, x, y, w, h)
            if refined_ok:
                self._last_bbox = list(refined_bbox)
                log.debug("Refinement OK: %s -> %s", new_bbox, self._last_bbox)
                return self._last_bbox, True  # refined=True means high confidence

        log.debug("Tracker update: bbox=%s confident=%s", new_bbox, confident)
        return new_bbox, confident

    def _refine_grabcut(self, frame: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[tuple[float, float, float, float], bool]:
        """Snap bbox to object contours using GrabCut."""
        try:
            import cv2
            # STEP 1 — Expand bbox for context
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x0 = max(0, int(x - pad_x))
            y0 = max(0, int(y - pad_y))
            x1 = min(frame.shape[1], int(x + w + pad_x))
            y1 = min(frame.shape[0], int(y + h + pad_y))
            
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                return (x, y, w, h), False

            # STEP 2 — GrabCut on the ROI
            mask = np.zeros(roi.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # rect is the CSRT bbox relative to the expanded ROI
            # Note: GrabCut rect is (x, y, w, h) relative to ROI
            gc_rect = (pad_x, pad_y, int(w), int(h))
            
            cv2.grabCut(roi, mask, gc_rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # STEP 3 — Build foreground mask
            fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            # STEP 4 — Find contours and tight bbox
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return (x, y, w, h), False
            
            largest = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest)
            
            # Convert back to full frame coordinates
            refined = (float(x0 + rx), float(y0 + ry), float(rw), float(rh))

            # STEP 5 — Sanity check (reject if area changed too much)
            original_area = w * h
            refined_area = rw * rh
            ratio = refined_area / original_area if original_area > 0 else 0
            
            if 0.40 <= ratio <= 1.60:
                return refined, True
            else:
                return (x, y, w, h), False
        except Exception as exc:
            log.warning("GrabCut refinement failed: %s", exc)
            return (x, y, w, h), False

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
