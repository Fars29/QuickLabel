# QuickLabel 🏷️

> **Professional collaborative dataset annotation tool for object detection** — optimized for training Detic/YOLO models in robotics contexts.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-orange?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey?style=flat-square)

---

## Features

- 🎥 **Live webcam capture** — hold SPACE to burst-capture at 5fps
- 📂 **Drag & drop image upload** — individual files or entire folders
- 🖱️ **Professional bbox editor** — draw, resize (8 handles), move, multi-box support
- 🤖 **CSRT tracker propagation** — automatically propagates bboxes frame-to-frame
- 🤝 **HuggingFace Hub sync** — push/pull with automatic COCO JSON merge
- 🔐 **Secure token storage** — HF tokens stored in OS keyring (never plaintext)
- 📦 **COCO JSON output** — standard format, single `instances_all.json`
- 🌑 **Dark-themed UI** — professional design with custom Qt Style Sheets

---

## Installation

### Prerequisites

- Python **3.10+**
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/QuickLabel.git
cd QuickLabel

# Create a virtual environment (recommended)
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Important**: Do **not** install `opencv-python` alongside `opencv-contrib-python`. The contrib package includes all standard OpenCV features **plus** the CSRT tracker. Having both installed causes import conflicts.

### Run

```bash
python main.py
```

---

## Usage Guide

### Creating a Dataset

1. Launch QuickLabel
2. Choose one of:
   - **New Local Dataset** — stored on your machine only
   - **New Synced Dataset** — linked to a Hugging Face dataset repository (for team collaboration)
   - **Open Existing Dataset** — resume work on a previously created dataset

### Capturing Images

1. Enter a **class name** in the input field (supports autocomplete from existing classes)
2. Switch to the **Camera** tab — live webcam preview appears
3. **Hold SPACE** to capture images at 5fps — a capture counter shows progress
4. Click **Review →** to annotate

### Uploading Images

1. Switch to the **Upload** tab
2. **Drag & drop** image files or folders onto the drop zone, or click to browse
3. Use **Import from Clipboard** to paste a screenshot
4. Remove unwanted images with the ✕ button on each thumbnail
5. Click **Review →** to annotate

### Annotating Bounding Boxes

- **Click and drag** on the canvas to draw a bounding box
- **Drag handles** (8 points) to resize
- **Drag the box center** to move it
- **Right-click → Delete** or press **D** to delete selected box
- Boxes automatically propagate to the next frame via CSRT tracker

| Box Color | Meaning |
|-----------|---------|
| 🟩 Green | Confirmed annotation |
| 🩵 Cyan | Auto-propagated (review recommended) |
| 🟧 Orange | Selected |
| ⬜ White dashed | Currently drawing |

### Syncing with HuggingFace

Click **Sync with HF** in the sidebar to manually push/pull. This also happens automatically when you click **Done** in the review screen.

---

## Dataset Structure

```
my_dataset/
├── dataset_config.json          # Local config (gitignored)
├── annotations/
│   └── instances_all.json       # Master COCO JSON file
└── images/
    ├── Cardboard/
    │   ├── Cardboard_001.jpg
    │   └── Cardboard_002.jpg
    └── Plastic/
        └── Plastic_001.jpg
```

### COCO JSON Format

```json
{
  "info": { "description": "QuickLabel dataset", "version": "1.0" },
  "categories": [
    { "id": 1, "name": "Cardboard", "supercategory": "object" }
  ],
  "images": [
    { "id": 1, "file_name": "images/Cardboard/Cardboard_001.jpg", "width": 640, "height": 480 }
  ],
  "annotations": [
    { "id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": w_h, "iscrowd": 0 }
  ]
}
```

---

## Configuration

Edit `config.py` to change:
- Output image resolution (default: 640×480)
- JPEG quality (default: 85)
- Webcam capture FPS (default: 5)
- Color theme

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | PyQt6 |
| Bbox Canvas | QGraphicsScene / QGraphicsView |
| Computer Vision | OpenCV 4.9+ (contrib) |
| Image Processing | Pillow (LANCZOS resize, letterbox) |
| Tracker | CSRT (Channel and Spatial Reliability Tracker) |
| Data Models | Pydantic v2 |
| HF Integration | huggingface_hub |
| Token Security | keyring (OS Credential Manager) |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [OpenCV Tracking API](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)
- [HuggingFace Hub](https://huggingface.co/docs/huggingface_hub)
- [Detic](https://github.com/facebookresearch/Detic) — the primary model this tool targets
