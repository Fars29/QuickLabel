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

import io
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Callable

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

class ProgressIO(io.BufferedIOBase):
    """Wrapper for file-like objects to track upload progress."""
    def __init__(self, filename: str, fileobj, callback: Callable[[str, int, int, str], None]):
        self.filename = filename
        self.fileobj = fileobj
        self.callback = callback
        try:
            self.total_size = os.fstat(fileobj.fileno()).st_size
        except (AttributeError, io.UnsupportedOperation):
            # Fallback if fileno() is not available
            self.total_size = 0
            
        self.bytes_read = 0
        self.start_time = time.time()

    def read(self, size=-1):
        chunk = self.fileobj.read(size)
        if chunk:
            self.bytes_read += len(chunk)
            elapsed = time.time() - self.start_time
            speed_val = (self.bytes_read / 1024) / elapsed if elapsed > 0 else 0
            speed_str = f"{speed_val:.1f} KB/s" if speed_val < 1024 else f"{speed_val/1024:.1f} MB/s"
            self.callback(self.filename, self.bytes_read, self.total_size, speed_str)
        return chunk

    def seek(self, offset, whence=0):
        return self.fileobj.seek(offset, whence)

    def tell(self):
        return self.fileobj.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.fileobj, name)


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
        Smart pull: downloads only new/changed files by comparing file sizes.
        
        Conflict resolution: if a local file exists with the SAME name but DIFFERENT size
        as the remote, the local file is renamed to the next free number, and the remote
        version is downloaded in its place. The returned remote_coco will reflect the
        remote state; the caller (merge_remote_coco) handles re-numbering.
        
        Returns the remote COCO JSON data as a dict (for merge), or {} if not found.
        """
        try:
            remote_files = list(self.api.list_repo_tree(
                repo_id=self.repo_id,
                repo_type=HF_REPO_TYPE,
                token=self.token,
                recursive=True,
            ))
        except RepositoryNotFoundError:
            return {}

        remote_coco: dict = {}
        coco_rel = f"{ANNOTATIONS_DIR}/{COCO_JSON_FILE}"

        for entry in remote_files:
            # Skip directories (no 'size' attribute)
            if not hasattr(entry, "size"):
                continue
            path_in_repo: str = entry.path

            # Only sync images and the COCO JSON
            is_image = path_in_repo.startswith(IMAGES_DIR + "/") and path_in_repo.endswith(".jpg")
            is_coco = path_in_repo == coco_rel
            if not is_image and not is_coco:
                continue

            local_path = local_dir / path_in_repo.replace("/", os.sep)

            if local_path.exists():
                local_size = local_path.stat().st_size
                if local_size == entry.size:
                    # Identical — skip download
                    if is_coco:
                        with open(local_path, "r", encoding="utf-8") as f:
                            remote_coco = json.load(f)
                    continue
                
                if is_image:
                    # CONFLICT: same name, different size.
                    # Rename local file to next free number so it is preserved.
                    self._resolve_image_conflict(local_dir, local_path)

            if is_coco:
                # Download remote COCO to a TEMP location — never overwrite the local
                # COCO JSON on disk (it may have already been updated by conflict renames).
                with tempfile.TemporaryDirectory(prefix="ql_coco_pull_") as tmp_dir:
                    from huggingface_hub import hf_hub_download
                    tmp_coco = hf_hub_download(
                        repo_id=self.repo_id,
                        repo_type=HF_REPO_TYPE,
                        filename=path_in_repo,
                        token=self.token,
                        local_dir=tmp_dir,
                    )
                    with open(tmp_coco, "r", encoding="utf-8") as f:
                        remote_coco = json.load(f)
                # tmp_dir and its contents are deleted here — local COCO untouched
            else:
                # Images can be downloaded directly
                local_path.parent.mkdir(parents=True, exist_ok=True)
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=self.repo_id,
                    repo_type=HF_REPO_TYPE,
                    filename=path_in_repo,
                    token=self.token,
                    local_dir=str(local_dir),
                )

        return remote_coco

    def _resolve_image_conflict(self, local_dir: Path, conflict_path: Path) -> Path:
        """
        Rename a conflicting local image to the next free number for its class.
        
        Example: if 'images/Carta/Carta_045.jpg' conflicts with the remote,
        it is renamed to 'images/Carta/Carta_046.jpg' (or higher if needed),
        and the COCO JSON on disk is updated to reflect the new filename.
        
        Returns the new path.
        """
        # Parse class name and number from filename
        filename = conflict_path.stem  # e.g. "Carta_045"
        m = re.match(r"^(.+)_(\d+)$", filename)
        if not m:
            return conflict_path  # Unexpected name format, leave it

        class_name = m.group(1)
        class_folder = conflict_path.parent

        # Find all existing numbers for this class
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$", re.IGNORECASE)
        used_numbers: set[int] = set()
        for f in class_folder.iterdir():
            mm = pattern.match(f.name)
            if mm:
                used_numbers.add(int(mm.group(1)))

        # Find next free number
        n = 1
        while n in used_numbers:
            n += 1

        new_filename = f"{class_name}_{n:03d}.jpg"
        new_path = class_folder / new_filename
        conflict_path.rename(new_path)

        # Update COCO JSON on disk to point to the new filename
        coco_path = local_dir / ANNOTATIONS_DIR / COCO_JSON_FILE
        if coco_path.exists():
            try:
                with open(coco_path, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
                
                old_rel = f"{IMAGES_DIR}/{class_name}/{conflict_path.name}"
                new_rel = f"{IMAGES_DIR}/{class_name}/{new_filename}"
                
                for img in coco_data.get("images", []):
                    if img.get("file_name") == old_rel:
                        img["file_name"] = new_rel
                
                with open(coco_path, "w", encoding="utf-8") as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Non-fatal: COCO will be re-merged anyway

        return new_path



    # ─── Push ──────────────────────────────────────────────────────────────────

    def push(
        self,
        local_dir: Path,
        changed_files: Optional[set[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        file_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        only_missing: bool = False,
    ) -> int:
        """
        Push local files to HF Hub.
        If only_missing is True, it will first list remote files and only upload
        those that don't exist on remote or have different sizes.
        """
        uploaded = 0

        if changed_files is None:
            changed_files = set()
            for img_file in (local_dir / IMAGES_DIR).rglob("*.jpg"):
                changed_files.add(str(img_file.relative_to(local_dir)))
            coco_path = local_dir / ANNOTATIONS_DIR / COCO_JSON_FILE
            if coco_path.exists():
                changed_files.add(str(coco_path.relative_to(local_dir)))

        # Suffix the COCO JSON if it exists, as it always needs pushing after a merge
        coco_rel = f"{ANNOTATIONS_DIR}/{COCO_JSON_FILE}"
        
        remote_metadata = {}
        if only_missing:
            try:
                remote_files = self.api.list_repo_tree(
                    repo_id=self.repo_id,
                    repo_type=HF_REPO_TYPE,
                    token=self.token,
                    recursive=True,
                )
                for entry in remote_files:
                    if hasattr(entry, "size"):
                        remote_metadata[entry.path] = entry.size
            except Exception:
                pass

        files_to_upload = []
        for rel_path in changed_files:
            abs_path = local_dir / rel_path
            if not abs_path.exists():
                continue
            
            repo_path = rel_path.replace("\\", "/")
            if only_missing and repo_path in remote_metadata:
                if repo_path != coco_rel: # Always push COCO
                    if abs_path.stat().st_size == remote_metadata[repo_path]:
                        continue
            
            files_to_upload.append(rel_path)

        total = len(files_to_upload)
        for i, rel_path in enumerate(files_to_upload):
            abs_path = local_dir / rel_path
            
            if progress_callback:
                progress_callback(i, total)

            with open(abs_path, "rb") as f:
                wrapped_file = ProgressIO(rel_path, f, file_progress_callback) if file_progress_callback else f
                self.api.upload_file(
                    path_or_fileobj=wrapped_file,
                    path_in_repo=rel_path.replace("\\", "/"),
                    repo_id=self.repo_id,
                    repo_type=HF_REPO_TYPE,
                    token=self.token,
                )
            uploaded += 1

        if progress_callback:
            progress_callback(total, total)

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
    """Pull remote changes in a background thread."""
    status = pyqtSignal(str, str)
    finished = pyqtSignal(bool, dict, str)

    def __init__(self, repo_id: str, token: str, local_dir: Path) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir

    def run(self) -> None:
        try:
            sync = HFSync(self.repo_id, self.token)
            self.status.emit("Syncing...", "Checking remote changes...")
            remote_coco = sync.pull(self.local_dir)
            self.finished.emit(True, remote_coco or {}, "")
        except Exception as exc:
            self.finished.emit(False, {}, str(exc))

class HFPushWorker(QThread):
    """Push local changes in a background thread."""
    status = pyqtSignal(str, str)
    progress_overall = pyqtSignal(int, int)
    progress_file = pyqtSignal(str, int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, repo_id: str, token: str, local_dir: Path, changed_files: Optional[set[str]] = None, only_missing: bool = False) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir
        self.changed_files = changed_files
        self.only_missing = only_missing

    def run(self) -> None:
        try:
            sync = HFSync(self.repo_id, self.token)
            self.status.emit("Syncing...", "Pushing local changes...")
            count = sync.push(
                self.local_dir,
                self.changed_files,
                progress_callback=self.progress_overall.emit,
                file_progress_callback=self.progress_file.emit,
                only_missing=self.only_missing
            )
            self.finished.emit(True, f"Successfully pushed {count} files.")
        except Exception as exc:
            self.finished.emit(False, str(exc))

class HFFullSyncWorker(QThread):
    """
    Perform a full bidirectional sync:
    1. Pull remote state & COCO.
    2. Merge (auto or via signal).
    3. Push local changes back to remote.
    """
    status = pyqtSignal(str, str)
    progress_overall = pyqtSignal(int, int)
    progress_file = pyqtSignal(str, int, int, str)
    remote_coco_ready = pyqtSignal(dict)
    finished = pyqtSignal(bool, str)

    def __init__(self, repo_id: str, token: str, local_dir: Path, dm=None) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.token = token
        self.local_dir = local_dir
        self.dm = dm

    def run(self) -> None:
        try:
            sync = HFSync(self.repo_id, self.token)
            
            # STEP 1: PULL
            self.status.emit("Syncing...", "Checking remote changes...")
            remote_coco = sync.pull(self.local_dir)
            
            if remote_coco:
                self.remote_coco_ready.emit(remote_coco)
                if self.dm:
                    self.dm.merge_remote_coco(remote_coco)
                    self.dm.save_coco()

            # STEP 2: PUSH
            self.status.emit("Syncing...", "Identifying local changes...")
            pushed_count = sync.push(
                self.local_dir,
                progress_callback=self.progress_overall.emit,
                file_progress_callback=self.progress_file.emit,
                only_missing=True
            )

            msg = f"Sync complete: {pushed_count} files pushed." if pushed_count > 0 else "Already up to date."
            self.finished.emit(True, msg)
        except Exception as exc:
            self.finished.emit(False, str(exc))


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
