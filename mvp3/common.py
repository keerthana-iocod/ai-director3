"""common.py — small helpers shared by every step.

- Finds the project folder, so paths work from anywhere.
- Loads config.yaml.
- Keeps library caches INSIDE the project (not in your home folder or /tmp).
- Says where each clip's files go: data/<clip>/ (tracking results) and outputs/<clip>/ (videos, logs).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

# The project root is the folder that contains this "mvp3" folder.
ROOT = Path(__file__).resolve().parent.parent

# Must be set BEFORE ultralytics / huggingface_hub are imported (they read these at import time).
(ROOT / ".yolo_config").mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))


def load_config() -> dict[str, Any]:
    """Read config.yaml into a plain dict."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def expand(p: str) -> Path:
    """A path from config.yaml: ~ = your home folder; relative = relative to the project."""
    path = Path(os.path.expanduser(p))
    return path if path.is_absolute() else ROOT / path


def clip_config(cfg: dict, clip: str) -> dict:
    if clip not in cfg["clips"]:
        raise SystemExit(f"Unknown clip '{clip}'. Clips in config.yaml: {', '.join(cfg['clips'])}")
    return cfg["clips"][clip]


def data_dir(clip: str) -> Path:
    """data/<clip>/ : tracking results (tracks.jsonl), so directing/rendering never re-runs detection."""
    d = ROOT / "data" / clip
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(clip: str) -> Path:
    """outputs/<clip>/ : program.mp4, debug.mp4, director_log.csv, summary.txt."""
    d = ROOT / "outputs" / clip
    d.mkdir(parents=True, exist_ok=True)
    return d


def ffmpeg_exe() -> str:
    """Path to ffmpeg: a system one if installed, otherwise the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
