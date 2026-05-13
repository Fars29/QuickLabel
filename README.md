# QuickLabel 🏷️

> **Professional collaborative dataset annotation tool for object detection** — Optimized for training Detic/YOLO models with a premium, high-performance interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-00d2ff?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/Engine-OpenCV%204.9+-white?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)

---

## ✨ Features

### 🎨 Premium "Liquid Glass" Interface
*   **Cyber Blue Theme**: High-contrast, dark-mode design with Electric Blue highlights.
*   **Fluid Interactions**: Micro-animations, smooth transitions, and responsive hover effects.
*   **Intuitive Layout**: Streamlined workspace designed for maximum efficiency during long annotation sessions.

### 🖼️ Intelligent Image Pipeline
*   **Letterbox Resizing**: Automatically preserves original aspect ratios using `LANCZOS` resampling, centering images on a black canvas of the target resolution.
*   **Dynamic Resolution**: Easily switch between standard resolutions (640x480, 1280x720, etc.) without distorting existing annotations.
*   **BBox Coordinate Mapping**: Seamless transformation between original and letterboxed coordinate spaces.

### 🎥 Multi-Source Acquisition
*   **Live Webcam Capture**: High-speed burst-capture at configurable FPS.
*   **Advanced Drag & Drop**: Batch upload individual files or nested directory structures.
*   **Smart Clipboard**: Paste screenshots or copied images directly into your dataset.

### 🤖 Assisted Annotation
*   **CSRT Tracker Propagation**: High-accuracy object tracking that propagates bounding boxes across frames, reducing manual work by up to 90%.
*   **Professional BBox Editor**: 8-handle resizing, fluid movement, and multi-box management.
*   **Fuzzy Class Autocomplete**: Intelligent class name suggestions based on existing dataset categories.

### 🤝 Enterprise Sync & Format
*   **HuggingFace Hub Sync**: Bi-directional push/pull with automatic COCO JSON merging.
*   **Secure Keyring**: Industry-standard OS credential storage for HF tokens.
*   **COCO JSON Standard**: Fully compatible with Detic, YOLO, and Detectron2 training pipelines.

---

## 🚀 Getting Started

### Prerequisites
*   Python **3.10+**
*   Git

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/QuickLabel.git
cd QuickLabel

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Pro Tip**: Use `opencv-contrib-python` to enable the advanced CSRT tracker. Avoid installing `opencv-python` simultaneously to prevent namespace conflicts.

### Run
```bash
python main.py
```

---

## 🛠️ Configuration (`config.py`)

Tailor QuickLabel to your specific training needs:
*   `TARGET_WIDTH` / `TARGET_HEIGHT`: Set your model's input size (default 640x480).
*   `JPEG_QUALITY`: Balance between file size and detail (default 85).
*   `CAPTURE_FPS`: Adjust webcam burst speed.
*   `COLOR_HIGHLIGHT`: Change the brand color from Electric Blue to your preference.

---

## 🎨 Visual Identity

| Element | Color | Role |
| :--- | :--- | :--- |
| **Highlight** | `#00d2ff` | Primary Actions / Selection (Electric Blue) |
| **Background** | `#0d0e17` | Deep Navy Black |
| **Surface** | `#13162b` | Card & Panel Surfaces |
| **Success** | `#00f5a0` | Confirmed Annotations |
| **Danger** | `#e74c3c` | Deletion / Critical Actions |

---

## 📂 Dataset Structure
QuickLabel organizes data in a clean, model-ready format:
```text
my_dataset/
├── dataset_config.json          # Local session metadata
├── annotations/
│   └── instances_all.json       # Master COCO JSON
└── images/
    ├── Category_A/
    │   ├── Category_A_001.jpg
    │   └── Category_A_002.jpg
    └── Category_B/
        └── Category_B_001.jpg
```

---

## 🛡️ Technical Stack
*   **UI Framework**: PyQt6 (Custom QSS)
*   **Graphics Engine**: QGraphicsScene with coordinate mapping logic
*   **Computer Vision**: OpenCV 4.9+ (Contrib) & CSRT Tracker
*   **Image Processing**: Pillow (LANCZOS, padding, metadata)
*   **Data Serialization**: Pydantic v2
*   **Sync Engine**: HuggingFace Hub API via `huggingface_hub`

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements
*   [COCO Format](https://cocodataset.org/#format-data)
*   [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub)
*   [Detic](https://github.com/facebookresearch/Detic) — primary target for this tool's output.

---
*Created by the QuickLabel Team with ❤️ for the Robotics Community.*
