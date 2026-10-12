"""detector.py — the football YOLO model (one instance, shared by all cameras).

Model: gianpaj/football-players-detection-1 on Hugging Face (YOLOv8x fine-tuned on football;
classes 0 ball, 1 goalkeeper, 2 player, 3 referee). Downloaded once into models/best.pt.

Two kinds of search:
  full(images)  the whole camera picture, all cameras in one batch. Finds players and ball
                candidates. Slow (~0.5 s per camera at 1280 on an M1), so it only runs every few
                frames, and at 1280 only while the ball is lost (640 otherwise).
  patch(image)  a small zoomed crop around where the ball should be, at its native resolution.
                The ball is many more pixels here, so it is found far more often. Fast (~60 ms).
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field

import numpy as np

import mvp3.common  # noqa: F401  (sets cache folders before ultralytics is imported)
from mvp3.common import expand

warnings.filterwarnings("ignore", category=FutureWarning)

BALL, GOALKEEPER, PLAYER, REFEREE = 0, 1, 2, 3


@dataclass
class Detections:
    balls: list[dict] = field(default_factory=list)      # [{"bbox": [x1,y1,x2,y2], "conf": 0.7}, ...]
    players: list[list[float]] = field(default_factory=list)   # player + goalkeeper boxes


def ensure_weights(cfg: dict) -> str:
    """Download the model from Hugging Face the first time; return the local path."""
    dcfg = cfg["detector"]
    path = expand(dcfg["weights"])
    if not path.exists():
        from huggingface_hub import hf_hub_download

        print(f"[detector] downloading {dcfg['hf_repo']} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(dcfg["hf_repo"], dcfg["hf_file"]), path)
    return str(path)


class Detector:
    def __init__(self, cfg: dict):
        import torch
        from ultralytics import YOLO

        self.cfg = cfg["detector"]
        self.model = YOLO(ensure_weights(cfg))
        wanted = self.cfg["device"]
        if wanted == "mps" and not torch.backends.mps.is_available():
            wanted = "cpu"
        if wanted == "cuda" and not torch.cuda.is_available():
            wanted = "cpu"
        self.device = wanted
        self.calls_full = 0
        self.calls_patch = 0

    def full(self, images: list[np.ndarray], imgsz: int) -> list[Detections]:
        """Search whole camera pictures (one batch). One Detections per image."""
        self.calls_full += len(images)
        results = self.model.predict(images, imgsz=imgsz, device=self.device, verbose=False,
                                     conf=min(self.cfg["ball_conf_global"], self.cfg["player_conf"]))
        return [self._read(r, self.cfg["ball_conf_global"]) for r in results]

    def patch(self, image: np.ndarray, offset: tuple[int, int]) -> list[dict]:
        """Search a small crop at its own resolution. Returns ball candidates in CAMERA coordinates."""
        self.calls_patch += 1
        size = max(32, (max(image.shape[:2]) // 32) * 32)
        r = self.model.predict(image, imgsz=size, device=self.device, verbose=False,
                               conf=self.cfg["ball_conf_patch"])[0]
        ox, oy = offset
        return [{"bbox": [b["bbox"][0] + ox, b["bbox"][1] + oy, b["bbox"][2] + ox, b["bbox"][3] + oy],
                 "conf": b["conf"]} for b in self._read(r, self.cfg["ball_conf_patch"]).balls]

    def _read(self, result, ball_conf: float) -> Detections:
        boxes = result.boxes
        raw = Detections()
        if boxes is None or len(boxes) == 0:
            return raw
        for xyxy, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int),
                                   boxes.conf.cpu().numpy()):
            box = [round(float(v), 1) for v in xyxy]
            if cls == BALL and conf >= ball_conf:
                raw.balls.append({"bbox": box, "conf": round(float(conf), 3)})
            elif cls in (PLAYER, GOALKEEPER) and conf >= self.cfg["player_conf"]:
                raw.players.append(box)
        return raw

