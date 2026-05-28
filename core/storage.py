"""
File-based storage for Photo Cores.

Layout on disk
--------------
data/photocores/<key>/
    metadata.json     ← analysis state (box, dividers, params, lithologies)
    processed.jpg     ← the downscaled (+optionally deskewed) working image
    overlay.jpg       ← cached overlay (verde+amarillo) for the preview
    rows/
        row_00.jpg
        row_01.jpg
        …

`<key>` is the first 12 hex chars of the MD5 of the original upload bytes.
The same image uploaded twice always lands in the same folder.

All the heavy state lives on disk; `st.session_state` only mirrors the
metadata + lazy numpy caches for the active rerun.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


BASE_DIR = Path("data") / "photocores"


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def pc_dir(key: str) -> Path:
    d = base_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def processed_path(key: str) -> Path:
    return pc_dir(key) / "processed.jpg"


def overlay_path(key: str) -> Path:
    return pc_dir(key) / "overlay.jpg"


def metadata_path(key: str) -> Path:
    return pc_dir(key) / "metadata.json"


def rows_dir(key: str) -> Path:
    d = pc_dir(key) / "rows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def row_path(key: str, index: int) -> Path:
    return rows_dir(key) / f"row_{index:02d}.jpg"


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_image(key: str, img: np.ndarray, quality: int = 88) -> Path:
    """Save the working RGB image as JPEG."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    path = processed_path(key)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def load_processed_image(key: str) -> np.ndarray | None:
    p = processed_path(key)
    if not p.exists():
        return None
    bgr = cv2.imread(str(p))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def save_overlay(key: str, img: np.ndarray, quality: int = 80) -> Path:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    path = overlay_path(key)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def save_rows(key: str, row_imgs: list[np.ndarray], quality: int = 85) -> list[Path]:
    """Wipe previous row files and save new ones. Returns list of paths."""
    rd = rows_dir(key)
    for f in rd.glob("row_*.jpg"):
        try:
            f.unlink()
        except OSError:
            pass
    paths: list[Path] = []
    for i, img in enumerate(row_imgs):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
        p = row_path(key, i)
        cv2.imwrite(str(p), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(p)
    return paths


def get_row_paths(key: str) -> list[Path]:
    rd = pc_dir(key) / "rows"
    if not rd.exists():
        return []
    return sorted(rd.glob("row_*.jpg"))


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(key: str, metadata: dict) -> Path:
    payload = _make_json_safe(metadata)
    payload.setdefault("hash", key)
    payload.setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
    if "created_at" not in payload:
        # Inherit from existing file if it has one
        existing = load_metadata(key)
        payload["created_at"] = (existing or {}).get(
            "created_at", payload["updated_at"]
        )
    path = metadata_path(key)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_metadata(key: str) -> dict | None:
    p = metadata_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Listing / cleanup
# ─────────────────────────────────────────────────────────────────────────────

def list_photocore_keys() -> list[str]:
    """Return existing photocore keys, ordered by created_at desc."""
    if not BASE_DIR.exists():
        return []
    keys = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
    # Sort by created_at when available, else by folder mtime
    def _sort_key(k: str):
        meta = load_metadata(k)
        if meta and meta.get("created_at"):
            return meta["created_at"]
        try:
            return datetime.fromtimestamp((BASE_DIR / k).stat().st_mtime).isoformat()
        except OSError:
            return "0"
    return sorted(keys, key=_sort_key, reverse=True)


def delete_photocore(key: str) -> bool:
    d = BASE_DIR / key
    if not d.exists():
        return False
    try:
        shutil.rmtree(d)
        return True
    except OSError:
        return False


def delete_all() -> int:
    if not BASE_DIR.exists():
        return 0
    n = 0
    for d in BASE_DIR.iterdir():
        if d.is_dir() and delete_photocore(d.name):
            n += 1
    return n


def folder_size_mb(key: str) -> float:
    """Total size of one photocore folder in MB."""
    d = BASE_DIR / key
    if not d.exists():
        return 0.0
    total = 0
    for f in d.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_json_safe(obj):
    """Convert numpy / Path / datetime values into JSON-friendly equivalents."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="seconds")
    return obj
