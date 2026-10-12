"""ptz.py — virtual zoom inside the live camera (the "PTZ" idea from ai-director2).

A real PTZ camera Pans, Tilts and Zooms. We fake it: the program shows a 16:9 crop of the live
camera's picture, and the crop moves smoothly to keep the ball in frame:

  - aim slightly AHEAD of the ball (lead_frames), so the shot shows where play is going;
  - don't move for tiny ball movements (deadband), move smoothly otherwise (smoothing, max_pan_px);
  - zoom in while the ball is tracked, zoom out when it is fast or lost, so the viewer sees context;
  - ball lost: aim at the middle of the players instead (the action is usually where the players are);
  - on a camera cut, jump straight to the new target (a cut is a cut, no slide across).
"""
from __future__ import annotations

import numpy as np

from mvp3.ball_tracker import DETECTED, BallState

ASPECT = 16 / 9


class VirtualCamera:
    def __init__(self, cfg: dict):
        self.cfg = cfg["ptz"]
        self.cam: int | None = None
        self.cx = self.cy = 0.0
        self.zoom = 1.0

    def update(self, cam: int, cam_size: tuple[int, int], ball: BallState, players: list[list[float]]) -> tuple[int, int, int, int]:
        """Move the crop one frame toward its target. Returns the crop box (x1, y1, x2, y2) in camera pixels."""
        w, h = cam_size
        tx, ty, tzoom = self._target(w, h, ball, players)
        if cam != self.cam:   # cut: snap to the target
            self.cam, self.cx, self.cy, self.zoom = cam, tx, ty, tzoom
        else:
            c = self.cfg
            self.zoom += (tzoom - self.zoom) * c["zoom_smoothing"]
            crop_w, crop_h = self._crop_size(w, h)
            for axis, target, span in (("x", tx, crop_w), ("y", ty, crop_h)):
                pos = self.cx if axis == "x" else self.cy
                gap = target - pos
                if abs(gap) < c["deadband"] * span:
                    continue
                step = float(np.clip(gap * c["smoothing"], -c["max_pan_px"], c["max_pan_px"]))
                if axis == "x":
                    self.cx += step
                else:
                    self.cy += step
        return self._box(w, h)

    def _target(self, w, h, ball: BallState, players) -> tuple[float, float, float]:
        c = self.cfg
        if ball.visible and ball.center is not None:
            vx, vy = ball.velocity
            tx = ball.center[0] + vx * c["lead_frames"]
            ty = ball.center[1] + vy * c["lead_frames"]
            fast = ball.speed >= c["fast_ball_px"]
            zoom = c["zoom_fast_ball"] if fast or ball.status != DETECTED else c["zoom_tracking"]
            return tx, ty, zoom
        if players:
            feet = np.array([[(p[0] + p[2]) / 2, p[3]] for p in players])
            return float(np.median(feet[:, 0])), float(np.median(feet[:, 1])), c["zoom_lost"]
        return w / 2, h / 2, c["zoom_lost"]

    def _crop_size(self, w: int, h: int) -> tuple[float, float]:
        """The biggest 16:9 box that fits the camera, divided by the zoom."""
        base_w = min(w, h * ASPECT)
        crop_w = base_w / max(1.0, self.zoom)
        return crop_w, crop_w / ASPECT

    def _box(self, w: int, h: int) -> tuple[int, int, int, int]:
        crop_w, crop_h = self._crop_size(w, h)
        x1 = float(np.clip(self.cx - crop_w / 2, 0, w - crop_w))
        y1 = float(np.clip(self.cy - crop_h / 2, 0, h - crop_h))
        # keep the stored centre inside the reachable area, so the crop doesn't "stick" at an edge
        self.cx, self.cy = x1 + crop_w / 2, y1 + crop_h / 2
        return int(x1), int(y1), int(x1 + crop_w), int(y1 + crop_h)
