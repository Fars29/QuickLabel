"""
models/annotation.py — Pydantic v2 data models for COCO-format annotations.

Design decisions:
- Use Pydantic v2 (model_validator, field_validator) for strict validation.
- All models are fully serialisable to/from JSON via model_dump() / model_validate().
- bbox is stored as [x, y, w, h] (COCO standard — top-left origin, absolute pixels).
- DatasetConfig is the local metadata file; it stores the HF repo info but NOT the token
  (the token key name is stored so we can retrieve the actual token from keyring).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# ─── COCO Sub-models ───────────────────────────────────────────────────────────

class COCOInfo(BaseModel):
    description: str = "QuickLabel dataset"
    version: str = "1.0"
    year: int = Field(default_factory=lambda: datetime.now().year)
    contributor: str = "QuickLabel"
    date_created: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y/%m/%d")
    )


class COCOLicense(BaseModel):
    id: int
    name: str
    url: str = ""


class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: str = "object"


class COCOImage(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    date_captured: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    license: int = 0
    coco_url: str = ""
    flickr_url: str = ""


class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    # bbox: [x, y, width, height] — COCO standard (absolute pixel coords, top-left origin)
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    area: float
    iscrowd: int = 0
    segmentation: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_area_if_zero(self) -> "COCOAnnotation":
        """Auto-compute area from bbox if not provided or zero."""
        if self.area == 0 and len(self.bbox) == 4:
            self.area = float(self.bbox[2] * self.bbox[3])
        return self


# ─── Root COCO Dataset ─────────────────────────────────────────────────────────

class COCODataset(BaseModel):
    info: COCOInfo = Field(default_factory=COCOInfo)
    licenses: list[COCOLicense] = Field(default_factory=list)
    categories: list[COCOCategory] = Field(default_factory=list)
    images: list[COCOImage] = Field(default_factory=list)
    annotations: list[COCOAnnotation] = Field(default_factory=list)

    # ── Helper accessors ──────────────────────────────────────────────────────

    def next_image_id(self) -> int:
        """Return the next available image ID (max existing + 1, or 1)."""
        if not self.images:
            return 1
        return max(img.id for img in self.images) + 1

    def next_annotation_id(self) -> int:
        """Return the next available annotation ID."""
        if not self.annotations:
            return 1
        return max(ann.id for ann in self.annotations) + 1

    def next_category_id(self) -> int:
        """Return the next available category ID."""
        if not self.categories:
            return 1
        return max(cat.id for cat in self.categories) + 1

    def get_category_by_name(self, name: str) -> Optional[COCOCategory]:
        """Find a category by name (case-insensitive)."""
        name_lower = name.lower()
        for cat in self.categories:
            if cat.name.lower() == name_lower:
                return cat
        return None

    def get_category_by_id(self, cat_id: int) -> Optional[COCOCategory]:
        for cat in self.categories:
            if cat.id == cat_id:
                return cat
        return None

    def get_image_by_id(self, image_id: int) -> Optional[COCOImage]:
        for img in self.images:
            if img.id == image_id:
                return img
        return None

    def get_annotations_for_image(self, image_id: int) -> list[COCOAnnotation]:
        return [ann for ann in self.annotations if ann.image_id == image_id]

    def validate_integrity(self) -> list[str]:
        """
        Run integrity checks. Returns a list of error messages (empty = valid).
        Checks:
        - Duplicate image IDs
        - Duplicate annotation IDs
        - Annotations referencing missing image IDs
        - Annotations referencing missing category IDs
        """
        errors: list[str] = []

        image_ids = [img.id for img in self.images]
        if len(image_ids) != len(set(image_ids)):
            errors.append("Duplicate image IDs detected.")

        ann_ids = [ann.id for ann in self.annotations]
        if len(ann_ids) != len(set(ann_ids)):
            errors.append("Duplicate annotation IDs detected.")

        image_id_set = set(image_ids)
        category_id_set = {cat.id for cat in self.categories}

        for ann in self.annotations:
            if ann.image_id not in image_id_set:
                errors.append(
                    f"Annotation {ann.id} references missing image_id {ann.image_id}."
                )
            if ann.category_id not in category_id_set:
                errors.append(
                    f"Annotation {ann.id} references missing category_id {ann.category_id}."
                )

        return errors

    def image_count_by_category(self) -> dict[str, int]:
        """
        Return {category_name: count_of_images_with_that_category}.
        An image is counted once per category, even if it has multiple bboxes.
        """
        cat_image_sets: dict[int, set[int]] = {}
        for ann in self.annotations:
            cat_image_sets.setdefault(ann.category_id, set()).add(ann.image_id)

        result: dict[str, int] = {}
        for cat in self.categories:
            result[cat.name] = len(cat_image_sets.get(cat.id, set()))
        return result


# ─── Local Dataset Config ──────────────────────────────────────────────────────

class DatasetConfig(BaseModel):
    """
    Stored as dataset_config.json in the dataset root.
    NEVER stores the actual HF token — only the keyring key name.
    """
    name: str
    local_path: str                        # Absolute path to dataset root
    hf_repo: Optional[str] = None          # e.g. "myorg/my-dataset" or None
    hf_token_key: Optional[str] = None     # Keyring key name for HF token
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def is_synced(self) -> bool:
        """True if this dataset is backed by a HuggingFace repo."""
        return self.hf_repo is not None and self.hf_token_key is not None
