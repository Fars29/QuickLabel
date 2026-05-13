"""
core/session.py — Persistent session storage in AppData.

Saves the last opened dataset path so the app can offer to resume on launch.
Stored in: %APPDATA%/QuickLabel/session.json  (Windows)
           ~/.config/QuickLabel/session.json   (Linux/macOS)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def _session_dir() -> Path:
    """Return the platform-appropriate config directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home())
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    d = Path(base) / "QuickLabel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_file() -> Path:
    return _session_dir() / "session.json"


def save_session(dataset_path: str) -> None:
    """Persist the last used dataset folder path."""
    try:
        with open(_session_file(), "w", encoding="utf-8") as f:
            json.dump({"last_dataset": dataset_path}, f)
    except Exception:
        pass  # Non-fatal


def load_session() -> Optional[str]:
    """
    Return the last used dataset folder path, or None if not found / invalid.
    Validates that the folder and dataset_config.json still exist.
    """
    try:
        path = _session_file()
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        folder = data.get("last_dataset", "")
        if folder and Path(folder).is_dir() and (Path(folder) / "dataset_config.json").exists():
            return folder
        return None
    except Exception:
        return None


def clear_session() -> None:
    """Remove saved session (e.g. after dataset is deleted)."""
    try:
        _session_file().unlink(missing_ok=True)
    except Exception:
        pass
