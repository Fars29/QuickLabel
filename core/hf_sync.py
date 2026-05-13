"""
core/hf_sync.py — HuggingFace Hub push/pull integration.

Design decisions:
- HFSync is a pure Python class; it does NOT inherit QThread.
- HFSyncWorker wraps it as a QThread so the UI never blocks.
- Per-file upload (upload_file) is used instead of upload_folder — this avoids
  re-uploading unchanged images in large datasets.
- snapshot_download with local_dir= is used for pull, so files land directly
  in the dataset folder (no cache indirection).
- Token is retrieved from keyring here, not stored in this class.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import keyring
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from PyQt6.QtCore import QThread, pyqtSignal

from config import (
    ANNOTATIONS_DIR,
    COCO_JSON_FILE,
    HF_CACHE_DIR,
    HF_KEYRING_SERVICE,
    HF_REPO_TYPE,
    IMAGES_DIR,
)


class HFSync:
    """
    Low-level HuggingFace Hub operations.
    All methods are synchronous and may raise exceptions.
    """

    def __init__(self, repo_id: str, token: str) -> None:
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)

    # ─── Repository Management ─────────────────────────────────────────────────

    def create_repo_if_missing(self) -> None:
        """Create the dataset repository on HF Hub if it doesn't already exist."""
        try:
            self.api.repo_info(repo_id=self.repo_id, repo_type=HF_REPO_TYPE)
        except RepositoryNotFoundError:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type=HF_REPO_TYPE,
                private=True,
            )

    def validate_token(self) -> str:
        """Validate the token by fetching the current user. Returns username."""
        user = self.api.whoami()
        return user["name"]

    # ─── Pull ──────────────────────────────────────────────────────────────────

    def pull(self, local_dir: Path) -> dict:
        """
        Download the latest state from HF Hub into local_dir.
        Returns the remote COCO JSON data as a dict (for merge), or {} if not found.

        Strategy:
        - Download to a temp directory first.
        - Copy only the files we care about (images/ and annotations/).
        - Return the remote COCO JSON for the caller to merge.
        """
        with tempfile.TemporaryDirectory(prefix="ql_hf_pull_") as tmp_dir:
            try:
                snapshot_download(
                    repo_id=self.repo_id,
                    repo_type=HF_REPO_TYPE,
                    local_dir=tmp_dir,
                    token=self.token,
                    ignore_patterns=["*.git*", ".gitattributes"],
                )
            except RepositoryNotFoundError:
                return {}

            tmp_path = Path(tmp_dir)

            # Copy new image files
            remote_images_dir = tmp_path / IMAGES_DIR
            if remote_images_dir.exists():
                local_images = local_dir / IMAGES_DIR
                for img_file in remote_images_dir.rglob("*.jpg"):
                    rel = img_file.relative_to(remote_images_dir)
                    dest = local_images / rel
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_file, dest)

            # Read remote COCO JSON
            remote_json = tmp_path / ANNOTATIONS_DIR / COCO_JSON_FILE
            if remote_json.exists():
                with open(remote_json, "r", encoding="utf-8") as f:
                    return json.load(f)

        return {}

    # ─── Push ──────────────────────────────────────────────────────────────────

    def push(
        self,
        local_dir: Path,
        changed_files: Optional[set[str]] = None,
    ) -> int:
        """
        Push changed files to HF Hub.

        Args:
            local_dir: Dataset root directory.
            changed_files: Set of relative paths (from local_dir) to upload.
                           If None, uploads everything under images/ and annotations/.

        Returns:
            Number of files uploaded.
        """
        uploaded = 0

        if changed_files is None:
            # Collect all images and the annotations JSON
            changed_files = set()
            for img_file in (local_dir / IMAGES_DIR).rglob("*.jpg"):
                changed_files.add(str(img_file.relative_to(local_dir)))
            coco_path = local_dir / ANNOTATIONS_DIR / COCO_JSON_FILE
            if coco_path.exists():
                changed_files.add(str(coco_path.relative_to(local_dir)))

        for rel_path in changed_files:
            abs_path = local_dir / rel_path
            if not abs_path.exists():
                continue
            self.api.upload_file(
                path_or_fileobj=str(abs_path),
                path_in_repo=rel_path.replace("\\", "/"),
                repo_id=self.repo_id,
                repo_type=HF_REPO_TYPE,
                token=self.token,
            )
            uploaded += 1

        return uploaded


# ─── QThread Workers ───────────────────────────────────────────────────────────

class HFValidateWorker(QThread):
    """Validate HF token and create repo in a background thread."""

    finished = pyqtSignal(bool, str)  # (success, username_or_error)

    def __init__(self, repo_id: str, token: str, create_if_missing: bool = True) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.create_if_missing = create_if_missing

    def run(self) -> None:
        try:
            sync = HFSync(self.repo_id, self.token)
            username = sync.validate_token()
            if self.create_if_missing:
                sync.create_repo_if_missing()
            self.finished.emit(True, username)
        except HfHubHTTPError as exc:
            self.finished.emit(False, f"Authentication failed: {exc}")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class HFPullWorker(QThread):
    """Pull latest remote state in a background thread."""

    progress = pyqtSignal(str)        # status message
    finished = pyqtSignal(bool, dict, str)  # (success, remote_coco_dict, error)

    def __init__(self, repo_id: str, token: str, local_dir: Path) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir

    def run(self) -> None:
        try:
            self.progress.emit("Connecting to Hugging Face Hub…")
            sync = HFSync(self.repo_id, self.token)
            self.progress.emit("Downloading remote changes…")
            remote_coco = sync.pull(self.local_dir)
            self.finished.emit(True, remote_coco, "")
        except Exception as exc:
            self.finished.emit(False, {}, str(exc))


class HFPushWorker(QThread):
    """Push local changes to HF Hub in a background thread."""

    progress = pyqtSignal(int, int)   # (uploaded, total)
    finished = pyqtSignal(bool, int, str)  # (success, count, error)

    def __init__(
        self,
        repo_id: str,
        token: str,
        local_dir: Path,
        changed_files: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir
        self.changed_files = changed_files

    def run(self) -> None:
        try:
            sync = HFSync(self.repo_id, self.token)
            count = sync.push(self.local_dir, self.changed_files)
            self.finished.emit(True, count, "")
        except Exception as exc:
            self.finished.emit(False, 0, str(exc))


# ─── Token Helpers ─────────────────────────────────────────────────────────────

def store_token(token_key: str, token: str) -> None:
    """Store an HF token in the OS keyring under token_key."""
    keyring.set_password(HF_KEYRING_SERVICE, token_key, token)


def retrieve_token(token_key: str) -> Optional[str]:
    """Retrieve an HF token from the OS keyring. Returns None if not found."""
    return keyring.get_password(HF_KEYRING_SERVICE, token_key)


def delete_token(token_key: str) -> None:
    """Delete an HF token from the OS keyring."""
    try:
        keyring.delete_password(HF_KEYRING_SERVICE, token_key)
    except keyring.errors.PasswordDeleteError:
        pass
