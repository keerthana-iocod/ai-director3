"""director.py — picks which camera is live on every frame. Ported from MVP 1 (ai-director/src/director.py).

Each frame, every camera gets a score:
  ball detected                       +100
  ball predicted (or just handed over) +60
  ball near the horizontal centre       +0..+30
  players visible                       +1 each, max +10

Rules that stop the output from flickering:
  - Minimum shot length: never cut sooner than 2 s after the previous cut.
  - Hysteresis: another camera must beat the live one for 0.5 s in a row before we cut.
  - No ball anywhere: stay; after 2 s, go to the fallback camera (the middle one).

What's better than MVP 1: the ball states come from the high-recall tracker with hand-off, so the
ball is "seen" far more often and the next camera already has it when it crosses a border.
All numbers come from config.yaml ("director").
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from mvp3.ball_tracker import DETECTED, PREDICTED, BallState


@dataclass
class Decision:
    cam: int                       # live camera, 1-based
    scores: list[float]
    reason: str
    events: list[dict] = field(default_factory=list)   # CUT / BALL_LOST / BALL_FOUND


def score_camera(ball: BallState, num_players: int, cam_width: int, cfg: dict) -> float:
    score = 0.0
    if ball.status == DETECTED:
        score += cfg["score_detected"]
    elif ball.status == PREDICTED:
        score += cfg["score_predicted"]
    if ball.visible and ball.center is not None:
        offset = abs(ball.center[0] - cam_width / 2) / (cam_width / 2)   # 0 = middle, 1 = edge
        score += cfg["score_center_max"] * max(0.0, 1.0 - offset)
    score += min(num_players * cfg["score_per_player"], cfg["score_players_max"])
    return round(score, 1)


class Director:
    def __init__(self, cfg: dict, fps: float, num_cams: int):
        self.cfg = cfg["director"]
        self.fps = fps
        self.num_cams = num_cams
        self.min_shot = round(self.cfg["min_shot_seconds"] * fps)
        self.hysteresis = max(1, round(self.cfg["hysteresis_seconds"] * fps))
        self.lost_limit = round(self.cfg["lost_fallback_seconds"] * fps)
        self.current = min(self.cfg["start_cam"], num_cams)
        self.shot_start = 0
        self.challenger: int | None = None
        self.challenger_frames = 0
        self.lost_frames = 0
        self.ball_visible = False

    def decide(self, frame: int, balls: list[BallState], player_counts: list[int], cam_widths: list[int]) -> Decision:
        scores = [score_camera(b, n, w, self.cfg) for b, n, w in zip(balls, player_counts, cam_widths)]
        events: list[dict] = []
        can_cut = frame - self.shot_start >= self.min_shot
        has_ball = any(b.visible for b in balls)
        if has_ball and not self.ball_visible and frame > 0:
            events.append({"type": "BALL_FOUND", "message": f"ball found on cam{self._best(scores)}"})
        if not has_ball and self.ball_visible:
            events.append({"type": "BALL_LOST", "message": "ball lost on all cameras"})
        self.ball_visible = has_ball

        if has_ball:
            self.lost_frames = 0
            reason = self._follow(scores, can_cut, frame, events)
        else:
            self.lost_frames += 1
            reason = self._lost(can_cut, frame, events)
        return Decision(self.current, scores, reason, events)

    def _best(self, scores: list[float]) -> int:
        return max(range(self.num_cams), key=lambda i: scores[i]) + 1

    def _follow(self, scores, can_cut, frame, events) -> str:
        best = self._best(scores)
        if best == self.current or scores[best - 1] <= scores[self.current - 1]:
            self.challenger, self.challenger_frames = None, 0
            return "live cam has the best score"
        if best == self.challenger:
            self.challenger_frames += 1
        else:
            self.challenger, self.challenger_frames = best, 1
        if self.challenger_frames < self.hysteresis:
            return f"cam{best} better, waiting (hysteresis {self.challenger_frames}/{self.hysteresis})"
        if not can_cut:
            return f"cam{best} better, waiting (min shot length)"
        return self._cut(best, frame, f"cam{best} scored higher for {self.cfg['hysteresis_seconds']}s", events)

    def _lost(self, can_cut, frame, events) -> str:
        self.challenger, self.challenger_frames = None, 0
        fallback = self.cfg["fallback_cam"]
        if self.lost_frames > self.lost_limit and self.current != fallback:
            if can_cut:
                return self._cut(fallback, frame, f"ball lost > {self.cfg['lost_fallback_seconds']}s, fallback", events)
            return "ball lost, waiting to go to fallback (min shot length)"
        return "no ball anywhere, holding current cam"

    def _cut(self, new_cam: int, frame: int, reason: str, events: list[dict]) -> str:
        events.append({"type": "CUT", "message": f"cam{self.current} -> cam{new_cam} ({reason})",
                       "from": self.current, "to": new_cam,
                       "shot_seconds": round((frame - self.shot_start) / self.fps, 2)})
        self.current = new_cam
        self.shot_start = frame
        self.challenger, self.challenger_frames = None, 0
        return "CUT: " + reason


class DecisionLog:
    """outputs/<clip>/director_log.csv, one row per frame."""

    def __init__(self, path: Path, num_cams: int):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["frame", "time", "live_cam"] + [f"score_cam{i + 1}" for i in range(num_cams)]
                             + [f"ball_cam{i + 1}" for i in range(num_cams)] + ["zoom", "reason"])

    def write(self, frame: int, t: float, d: Decision, balls: list[BallState], zoom: float) -> None:
        self.writer.writerow([frame, f"{t:.3f}", d.cam, *d.scores, *[b.status for b in balls], f"{zoom:.2f}", d.reason])

    def close(self) -> None:
        self.file.close()


def shot_lengths(live: list[int], fps: float) -> list[float]:
    """Per-frame live camera list -> list of shot durations in seconds."""
    if not live:
        return []
    lengths, run = [], 1
    for prev, cur in zip(live, live[1:]):
        if cur == prev:
            run += 1
        else:
            lengths.append(run / fps)
            run = 1
    lengths.append(run / fps)
    return lengths
