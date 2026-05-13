"""
core/dataset_manager.py — Single source of truth for dataset state.

Design decisions:
- DatasetManager is NOT a singleton via module state; instead the App creates one
  instance and passes it to every screen via constructor injection. This keeps tests
  clean and avoids hidden global state.
- All writes are atomic: write to a temp file, then rename. A .bak is kept before
  each write so corrupt writes are recoverable.
- merge_remote_coco() re-numbers ALL IDs from scratch rather than trying to find
  the highest remote ID — this is simpler and guaranteed conflict-free.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from models.annotation import (
    COCOAnnotation,
    COCOCategory,
    COCODataset,
    COCOImage,
    DatasetConfig,
)
from config import (
    ANNOTATIONS_DIR,
    COCO_JSON_BACKUP_FILE,
    COCO_JSON_FILE,
    DATASET_CONFIG_FILE,
    IMAGES_DIR,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)


class DatasetManager:
    """
    Manages a single QuickLabel dataset on disk.

    Lifecycle:
        dm = DatasetManager()
        dm.create_local(folder, name)   # OR
        dm.create_synced(...)           # OR
        dm.open(folder)
        # ... use dm.dataset, dm.config ...
        dm.save_coco()
    """

    def __init__(self) -> None:
        self.config: Optional[DatasetConfig] = None
        self.dataset: Optional[COCODataset] = None
        self._root: Optional[Path] = None

    # ─── Path Helpers ──────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        if self._root is None:
            raise RuntimeError("No dataset is open.")
        return self._root

    @property
    def annotations_dir(self) -> Path:
        return self.root / ANNOTATIONS_DIR

    @property
    def images_dir(self) -> Path:
        return self.root / IMAGES_DIR

    @property
    def coco_json_path(self) -> Path:
        return self.annotations_dir / COCO_JSON_FILE

    @property
    def config_path(self) -> Path:
        return self.root / DATASET_CONFIG_FILE

    # ─── Dataset Creation ──────────────────────────────────────────────────────

    def create_local(self, folder: str, name: str) -> None:
        """
        Create a new local-only dataset in the given folder.
        Initialises the folder structure and writes an empty COCO JSON.
        """
        self._root = Path(folder)
        self._init_folder_structure()

        self.config = DatasetConfig(name=name, local_path=str(self._root))
        self.dataset = COCODataset()

        self._write_config()
        self.save_coco()

    def create_synced(
        self, folder: str, name: str, hf_repo: str, hf_token_key: str
    ) -> None:
        """
        Create a new HuggingFace-backed dataset.
        The caller (hf_sync.py) is responsible for the actual HF operations;
        this method just initialises the local state.
        """
        self._root = Path(folder)
        self._init_folder_structure()

        self.config = DatasetConfig(
            name=name,
            local_path=str(self._root),
            hf_repo=hf_repo,
            hf_token_key=hf_token_key,
        )
        self.dataset = COCODataset()

        self._write_config()
        self.save_coco()

    def open(self, folder: str) -> None:
        """
        Open an existing dataset folder.
        Raises FileNotFoundError if dataset_config.json is missing.
        Raises ValueError if instances_all.json is corrupt.
        """
        self._root = Path(folder)
        config_path = self._root / DATASET_CONFIG_FILE

        if not config_path.exists():
            raise FileNotFoundError(
                f"No dataset_config.json found in {folder}.\n"
                "Please select a valid QuickLabel dataset folder."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = DatasetConfig.model_validate(json.load(f))

        # Update local_path in case folder was moved
        self.config.local_path = str(self._root)

        if self.coco_json_path.exists():
            self._load_coco()
        else:
            # Initialise empty if COCO file is missing
            self.dataset = COCODataset()
            self.save_coco()

    # ─── COCO JSON I/O ─────────────────────────────────────────────────────────

    def _load_coco(self) -> None:
        """Load instances_all.json. Raises ValueError on corrupt data."""
        try:
            with open(self.coco_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.dataset = COCODataset.model_validate(raw)
        except (json.JSONDecodeError, Exception) as exc:
            # Try to restore from backup
            bak = self.annotations_dir / COCO_JSON_BACKUP_FILE
            if bak.exists():
                try:
                    with open(bak, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self.dataset = COCODataset.model_validate(raw)
                    return
                except Exception:
                    pass
            raise ValueError(
                f"instances_all.json is corrupt and no valid backup exists.\n"
                f"Original error: {exc}"
            )

    def save_coco(self) -> None:
        """
        Atomically write instances_all.json.
        - Backs up the current file to .bak first.
        - Writes to a temp file, then renames (atomic on same filesystem).
        """
        if self.dataset is None:
            return

        annotations_dir = self.annotations_dir
        annotations_dir.mkdir(parents=True, exist_ok=True)

        coco_path = self.coco_json_path
        bak_path = annotations_dir / COCO_JSON_BACKUP_FILE

        # Back up existing file
        if coco_path.exists():
            shutil.copy2(coco_path, bak_path)

        # Validate before writing
        errors = self.dataset.validate_integrity()
        if errors:
            raise ValueError(
                "COCO dataset integrity check failed before save:\n"
                + "\n".join(errors)
            )

        # Write atomically
        data = self.dataset.model_dump()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                dir=annotations_dir,
                delete=False,
            ) as tmp:
                json.dump(data, tmp, indent=2, ensure_ascii=False)
                tmp_path = tmp.name

            os.replace(tmp_path, coco_path)
        except Exception:
            # Clean up temp file if rename failed
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ─── Config I/O ────────────────────────────────────────────────────────────

    def _write_config(self) -> None:
        """Write dataset_config.json (no HF token — just the keyring key name)."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.model_dump(), f, indent=2)

    # ─── Folder Structure ──────────────────────────────────────────────────────

    def _init_folder_structure(self) -> None:
        """Create the required directory skeleton."""
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / ANNOTATIONS_DIR).mkdir(exist_ok=True)
        (self._root / IMAGES_DIR).mkdir(exist_ok=True)

    def get_class_folder(self, class_name: str) -> Path:
        """Return the path for a class's image folder, creating it if needed."""
        folder = self.images_dir / class_name
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # ─── Image Numbering ───────────────────────────────────────────────────────

    def next_image_number(self, class_name: str) -> int:
        """
        Scan the class folder for files matching ClassName_NNN.jpg
        and return the next sequential number.
        Example: if Cardboard_007.jpg is the highest, returns 8.
        """
        folder = self.images_dir / class_name
        if not folder.exists():
            return 1

        pattern = re.compile(
            rf"^{re.escape(class_name)}_(\d+)\.jpg$", re.IGNORECASE
        )
        max_num = 0
        for entry in folder.iterdir():
            m = pattern.match(entry.name)
            if m:
                num = int(m.group(1))
                if num > max_num:
                    max_num = num

        return max_num + 1

    def build_image_filename(self, class_name: str, number: int) -> str:
        """Return filename like 'Cardboard_008.jpg' (3-digit zero-padded)."""
        return f"{class_name}_{number:03d}.jpg"

    def build_image_path(self, class_name: str, number: int) -> Path:
        """Return absolute path for an image file."""
        return self.get_class_folder(class_name) / self.build_image_filename(
            class_name, number
        )

    def image_file_name_relative(self, class_name: str, number: int) -> str:
        """Return COCO-style relative file_name: 'images/ClassName/ClassName_NNN.jpg'."""
        return f"{IMAGES_DIR}/{class_name}/{self.build_image_filename(class_name, number)}"

    # ─── Category Management ───────────────────────────────────────────────────

    def get_or_create_category(self, class_name: str) -> COCOCategory:
        """Return existing category for class_name, or create a new one."""
        if self.dataset is None:
            raise RuntimeError("No dataset loaded.")

        existing = self.dataset.get_category_by_name(class_name)
        if existing:
            return existing

        new_cat = COCOCategory(
            id=self.dataset.next_category_id(),
            name=class_name,
            supercategory="object",
        )
        self.dataset.categories.append(new_cat)
        return new_cat

    # ─── Add Images & Annotations ──────────────────────────────────────────────

    def add_batch(
        self,
        class_name: str,
        saved_image_paths: list[Path],
        annotations_per_image: list[list[list[float]]],
    ) -> None:
        """
        Add a completed batch to the COCO dataset.

        Args:
            class_name: Category name.
            saved_image_paths: List of absolute paths to already-saved JPEG files.
            annotations_per_image: Parallel list; each element is a list of
                [x, y, w, h] bboxes for that image (may be empty).
        """
        if self.dataset is None:
            raise RuntimeError("No dataset loaded.")

        category = self.get_or_create_category(class_name)

        for abs_path, bboxes in zip(saved_image_paths, annotations_per_image):
            # Build relative file_name from the path
            rel = abs_path.relative_to(self.root)
            file_name = rel.as_posix()

            coco_image = COCOImage(
                id=self.dataset.next_image_id(),
                file_name=file_name,
                width=TARGET_WIDTH,
                height=TARGET_HEIGHT,
            )
            self.dataset.images.append(coco_image)

            for bbox in bboxes:
                x, y, w, h = bbox
                ann = COCOAnnotation(
                    id=self.dataset.next_annotation_id(),
                    image_id=coco_image.id,
                    category_id=category.id,
                    bbox=[float(x), float(y), float(w), float(h)],
                    area=float(w * h),
                    iscrowd=0,
                )
                self.dataset.annotations.append(ann)

    # ─── Class Stats ───────────────────────────────────────────────────────────

    def get_class_counts(self) -> dict[str, int]:
        """Return {class_name: image_count} for the sidebar display."""
        if self.dataset is None:
            return {}
        return self.dataset.image_count_by_category()

    def get_all_class_names(self) -> list[str]:
        """Return sorted list of all category names."""
        if self.dataset is None:
            return []
        return sorted(cat.name for cat in self.dataset.categories)

    def total_image_count(self) -> int:
        """Return total number of images in the dataset."""
        if self.dataset is None:
            return 0
        return len(self.dataset.images)

    # ─── Remote Merge ──────────────────────────────────────────────────────────

    def merge_remote_coco(self, remote_data: dict) -> None:
        """
        Merge a remote COCO JSON dict into the local dataset.

        Strategy:
        - Build a union of all categories (by name, deduped).
        - Build a union of all images (by file_name, deduped).
        - Build a union of all annotations (by image+category+bbox combo, deduped).
        - Re-number all IDs sequentially from 1.

        This is safe because file_name is the real identity of an image, not the ID.
        """
        if self.dataset is None:
            self.dataset = COCODataset()

        try:
            remote = COCODataset.model_validate(remote_data)
        except Exception as exc:
            raise ValueError(f"Remote COCO JSON is invalid: {exc}")

        # ── Merge categories by name ──────────────────────────────────────────
        all_cat_names: dict[str, str] = {}  # name_lower -> original_name
        for cat in self.dataset.categories:
            all_cat_names[cat.name.lower()] = cat.name
        for cat in remote.categories:
            if cat.name.lower() not in all_cat_names:
                all_cat_names[cat.name.lower()] = cat.name

        new_categories = [
            COCOCategory(id=idx + 1, name=name, supercategory="object")
            for idx, name in enumerate(sorted(all_cat_names.values()))
        ]
        cat_name_to_id = {cat.name: cat.id for cat in new_categories}

        # ── Merge images by file_name ─────────────────────────────────────────
        all_images: dict[str, COCOImage] = {}
        for img in self.dataset.images:
            all_images[img.file_name] = img
        for img in remote.images:
            if img.file_name not in all_images:
                all_images[img.file_name] = img

        # Build old_image_id -> file_name maps for both datasets
        local_id_to_fname = {img.id: img.file_name for img in self.dataset.images}
        remote_id_to_fname = {img.id: img.file_name for img in remote.images}

        # Re-assign image IDs
        new_images: list[COCOImage] = []
        fname_to_new_id: dict[str, int] = {}
        for new_id, (fname, img) in enumerate(all_images.items(), start=1):
            new_img = img.model_copy(update={"id": new_id})
            new_images.append(new_img)
            fname_to_new_id[fname] = new_id

        # ── Merge annotations (deduplicate by content) ────────────────────────
        # Key: (file_name, category_name, rounded_bbox)
        seen_annotations: set[tuple] = set()
        new_annotations: list[COCOAnnotation] = []
        ann_id_counter = 1

        def _add_ann(
            ann: COCOAnnotation,
            id_to_fname: dict[int, str],
            remote_cats: list[COCOCategory],
            is_remote: bool,
        ) -> None:
            nonlocal ann_id_counter
            fname = id_to_fname.get(ann.image_id)
            if fname is None:
                return  # Orphan annotation

            # Find original category name
            if is_remote:
                cat_obj = next(
                    (c for c in remote_cats if c.id == ann.category_id), None
                )
            else:
                cat_obj = self.dataset.get_category_by_id(ann.category_id)

            if cat_obj is None:
                return  # Orphan annotation

            new_image_id = fname_to_new_id.get(fname)
            new_cat_id = cat_name_to_id.get(cat_obj.name)
            if new_image_id is None or new_cat_id is None:
                return

            bbox_rounded = tuple(round(v, 1) for v in ann.bbox)
            key = (fname, cat_obj.name, bbox_rounded)
            if key in seen_annotations:
                return

            seen_annotations.add(key)
            new_annotations.append(
                ann.model_copy(
                    update={
                        "id": ann_id_counter,
                        "image_id": new_image_id,
                        "category_id": new_cat_id,
                    }
                )
            )
            ann_id_counter += 1

        for ann in self.dataset.annotations:
            _add_ann(ann, local_id_to_fname, self.dataset.categories, is_remote=False)
        for ann in remote.annotations:
            _add_ann(ann, remote_id_to_fname, remote.categories, is_remote=True)

        self.dataset.categories = new_categories
        self.dataset.images = new_images
        self.dataset.annotations = new_annotations
