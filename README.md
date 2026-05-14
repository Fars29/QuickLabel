# QuickLabel

**Collaborative dataset annotation for object detection — built for speed.**

QuickLabel lets small teams build COCO-format datasets together without friction. Capture images from a webcam, drag in files, draw bounding boxes, and push everything to a shared Hugging Face repository in one click.

---

## Install

```bash
git clone https://github.com/YOUR_USERNAME/QuickLabel.git
cd QuickLabel
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

> Requires `opencv-contrib-python` for the CSRT tracker. Do not install `opencv-python` alongside it.

---

## How it works

1. **Create or open a dataset** — local folder or backed by a Hugging Face repo.
2. **Set a class name** — existing classes appear as suggestions while you type.
3. **Capture or upload images** — hold Space to burst-capture from your webcam at 5 fps, or drag files/folders into the upload tab.
4. **Annotate** — draw bounding boxes on the first frame; CSRT tracking propagates them automatically to the following frames.
5. **Done** — QuickLabel pulls the latest remote changes, merges them, saves your images and updates the COCO JSON, then pushes everything back to Hugging Face.

Output is a standard COCO `instances_all.json` ready to feed directly into Detic, Detectron2, or any YOLO pipeline.

---

## Dataset layout

```
my_dataset/
├── dataset_config.json
├── annotations/
│   └── instances_all.json
└── images/
    ├── CardBoard/
    │   ├── CardBoard_001.jpg
    │   └── CardBoard_002.jpg
    └── Plastic/
        └── Plastic_001.jpg
```

Images are saved as JPEG at quality 85, letterboxed to the target resolution (default 640×480) with aspect ratio preserved.

---

## Team sync

QuickLabel uses the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) as its sync backend. Before each push it pulls remote changes and merges the COCO JSON automatically, re-numbering IDs to avoid conflicts. Your HF token is stored in the OS keyring, never in plaintext.

---

## License

MIT
