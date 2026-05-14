"""
config.py — Global constants for QuickLabel.

Design decisions:
- All magic numbers and color values live here, never scattered across UI files.
- Colors are defined as hex strings compatible with both Qt stylesheets and QColor().
- Paths are constructed at runtime relative to the dataset root, not hardcoded.
"""

from __future__ import annotations

# ─── App Metadata ──────────────────────────────────────────────────────────────
APP_NAME = "QuickLabel"
APP_VERSION = "1.0.0"
APP_AUTHOR = "QuickLabel Team"

# ─── Image Processing ──────────────────────────────────────────────────────────
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
JPEG_QUALITY = 85           # 1–95; 85 is a good trade-off for Detic training
CAPTURE_FPS = 5             # Frames per second captured while SPACE is held

# ─── Dataset Structure ─────────────────────────────────────────────────────────
DATASET_CONFIG_FILE = "dataset_config.json"
ANNOTATIONS_DIR = "annotations"
IMAGES_DIR = "images"
COCO_JSON_FILE = "instances_all.json"
COCO_JSON_BACKUP_FILE = "instances_all.json.bak"

# ─── HuggingFace ───────────────────────────────────────────────────────────────
HF_KEYRING_SERVICE = "QuickLabel"           # Keyring service name
HF_REPO_TYPE = "dataset"                    # All repos are dataset repos
HF_CACHE_DIR = ".hf_cache"                  # Local HF cache subfolder

# ─── Color Palette (Liquid Glass Dark Theme) ──────────────────────────────────
# NOTE: Qt stylesheets do NOT reliably support rgba() for background properties on
# all widget types — use solid hex only; translucency is achieved via painter effects.
COLOR_BG = "#0d0e17"           # Deep navy-black background
COLOR_SURFACE = "#13162b"      # Card / panel surface (solid, slightly lighter than BG)
COLOR_SURFACE2 = "#1a1f3a"     # Secondary surface (e.g. top bars, dialogs)
COLOR_ACCENT = "#1e2d5a"       # Accent borders / dividers
COLOR_HIGHLIGHT = "#00d2ff"    # Vibrant primary action color (Electric Blue)
COLOR_TEXT = "#f0f2ff"         # Crisp near-white text
COLOR_TEXT_MUTED = "#7b82a8"   # Muted text
COLOR_SUCCESS = "#00f5a0"      # Neon green for success / confirmed
COLOR_WARNING = "#ffb000"      # Neon amber / selected
COLOR_INFO = "#00d2ff"         # Cyan
COLOR_DANGER = "#ff2a55"       # Bright red for delete

# Bounding box canvas colors
BBOX_COLOR_DRAWING = "#00f5a0"    # In-progress draw — green dashed
BBOX_COLOR_CONFIRMED = "#00f5a0"  # Confirmed annotation
BBOX_COLOR_REFINED = "#00f5a0"    # High-confidence GrabCut refined (Green)
BBOX_COLOR_PROPAGATED = "#00f5a0" # CSRT-only propagation (Green)
BBOX_COLOR_SELECTED = "#ffb000"   # Currently selected box (Amber)

# ─── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY = "Segoe UI, Inter, Arial, sans-serif"
FONT_SIZE_HEADING = 20      # px
FONT_SIZE_SUBHEADING = 16
FONT_SIZE_BODY = 13
FONT_SIZE_SMALL = 11

# ─── UI Layout ─────────────────────────────────────────────────────────────────
SIDEBAR_WIDTH = 260         # px
FILMSTRIP_WIDTH = 180       # px (review screen left panel - increased size)
BBOX_PANEL_WIDTH = 220      # px (review screen right panel)
THUMBNAIL_SIZE = 80         # px (smaller thumbnails)
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 750

# ─── Tracker ───────────────────────────────────────────────────────────────────
TRACKER_CONFIDENCE_THRESHOLD = 0.4   # Below this → show "low confidence" warning
# CSRT does not expose a raw score; we use IoU of predicted vs. expected motion.
# A bbox area change > this factor flags low confidence:
TRACKER_AREA_CHANGE_THRESHOLD = 0.5  # e.g. area shrank/grew by >50%
ENABLE_GRABCUT_REFINEMENT = True     # Use GrabCut to snap bbox to object contours

# ─── Webcam ────────────────────────────────────────────────────────────────────
WEBCAM_INDEXES_TO_TRY = [0, 1, 2, 3]   # Try these V4L/DirectShow indexes
WEBCAM_BUFFER_SIZE = 1                  # Camera buffer (minimize latency)
