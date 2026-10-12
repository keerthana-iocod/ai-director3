"""sources.py — where the camera frames come from.

CameraRig gives you ONE frame from EVERY camera at the same moment, frame after frame
(a generator, so only the current frames are ever in memory).

Two modes (see `clips:` in config.yaml):
  split  one wide video cut into N side-by-side strips. The strips touch, so we know exactly where
         a camera's right edge meets the next camera's left edge (`offsets`): that is what makes the
         ball hand-off between cameras possible.
  files  separate, real camera videos. `start_seconds` per camera lines them up in time.
         No hand-off: we don't know how the cameras overlap without calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from mvp3.common import expand


@dataclass
class Tick:
    index: int               # frame number (0 = first frame used)
    time: float              # seconds since the first frame used
    cams: list[np.ndarray]   # one BGR image per camera


class CameraRig:
    def __init__(self, clip_cfg: dict):
        self.mode = clip_cfg["mode"]
        if self.mode == "split":
            self.videos = [expand(clip_cfg["video"])]
            self.num_cams = int(clip_cfg["num_cams"])
            self.starts = [0.0]
        elif self.mode == "files":
            self.videos = [expand(c["video"]) for c in clip_cfg["cams"]]
            self.num_cams = len(self.videos)
            self.starts = [float(c.get("start_seconds", 0.0)) for c in clip_cfg["cams"]]
        else:
            raise SystemExit(f"Unknown clip mode '{self.mode}' (use split or files)")
        for v in self.videos:
            if not v.exists():
                raise SystemExit(f"Video not found: {v}")

        infos = [self._info(v) for v in self.videos]
        self.fps = infos[0]["fps"]
        for v, info in zip(self.videos, infos):
            if abs(info["fps"] - self.fps) > 0.01:
                raise SystemExit(f"{v.name} is {info['fps']:.2f} fps, but {self.videos[0].name} is {self.fps:.2f} fps. "
                                 "Convert all cameras to the same fps first.")
        if self.mode == "split":
            w, h = infos[0]["width"], infos[0]["height"]
            strip = w // self.num_cams
            self.cam_sizes = [(strip, h)] * self.num_cams
            self.offsets = [i * strip for i in range(self.num_cams)]   # x of each camera's left edge
            self.frames = infos[0]["frames"]
        else:
            self.cam_sizes = [(i["width"], i["height"]) for i in infos]
            self.offsets = None
            self.frames = min(i["frames"] - round(s * self.fps) for i, s in zip(infos, self.starts))
        self.audio_video = self.videos[0] if self.mode == "split" else \
            self.videos[int(clip_cfg.get("audio_from", 1)) - 1]
        self.audio_start = 0.0 if self.mode == "split" else self.starts[int(clip_cfg.get("audio_from", 1)) - 1]

    @staticmethod
    def _info(video: Path) -> dict:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video}")
        info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        return info

    @property
    def handoff_possible(self) -> bool:
        return self.offsets is not None

    def ticks(self, start_seconds: float = 0.0, max_seconds: float | None = None) -> Iterator[Tick]:
        """Yield the frames of all cameras, one moment at a time."""
        caps = [cv2.VideoCapture(str(v)) for v in self.videos]
        try:
            for cap, s in zip(caps, self.starts):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round((s + start_seconds) * self.fps))
            index = 0
            while True:
                t = index / self.fps
                if max_seconds is not None and t >= max_seconds:
                    break
                frames = []
                for cap in caps:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    frames.append(frame)
                if self.mode == "split":
                    w = self.cam_sizes[0][0]
                    cams = [frames[0][:, i * w:(i + 1) * w] for i in range(self.num_cams)]
                else:
                    cams = frames
                yield Tick(index, t, cams)
                index += 1
        finally:
            for cap in caps:
                cap.release()
