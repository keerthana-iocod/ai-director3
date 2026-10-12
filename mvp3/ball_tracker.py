"""ball_tracker.py — follows the ball in ONE camera (every camera gets its own BallTracker).

Ported from ai-director2's HighRecallBallTracker (cam_director/ball_tracker/), with fixes:

  1. Prediction (Kalman filter): where should the ball be now, given where it was and how fast it
     moved? ai-director2 added 18 px/frame² of "gravity" downwards on every frame. On a fixed wide
     camera the image's "down" is NOT height (a rolling ball doesn't fall down the picture), so
     predicted balls slid off the pitch. Here: constant velocity + a little drag (the ball slows down).
  2. Zoomed search ("temporal RoI"): look for the ball in a small crop around the prediction at full
     resolution. The ball is far bigger there than in a shrunk full frame, so it is found much more
     often (the main idea from ai-director2).
  3. Plausibility check: size, shape and distance from the prediction reject socks, boots and lines.
  4. Full-frame search (every few frames, done for all cameras at once by the engine): starts a new
     track, or takes over when the current track is shaky and a much surer ball shows up elsewhere.
  5. Hand-off (NEW in MVP 3): when the ball leaves the neighbouring camera, this camera is told where
     it should enter (`seed`) and searches there first, so the ball is picked up immediately.

Every frame ends with a status: detected / predicted / lost.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

DETECTED, PREDICTED, LOST = "detected", "predicted", "lost"


@dataclass
class BallState:
    status: str = LOST                          # detected / predicted / lost
    center: tuple[float, float] | None = None   # camera pixels
    bbox: list[float] | None = None
    conf: float = 0.0
    source: str = "lost"                        # patch / full / handoff / predicted / lost
    velocity: tuple[float, float] = (0.0, 0.0)  # px per frame
    missed: int = 0                             # frames since the last real detection

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.center:
            d["center"] = [round(v, 1) for v in self.center]
        if self.bbox:
            d["bbox"] = [round(v, 1) for v in self.bbox]
        d["velocity"] = [round(v, 2) for v in self.velocity]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BallState":
        return cls(status=d["status"], center=tuple(d["center"]) if d.get("center") else None,
                   bbox=d.get("bbox"), conf=d.get("conf", 0.0), source=d.get("source", "lost"),
                   velocity=tuple(d.get("velocity", (0.0, 0.0))), missed=d.get("missed", 0))

    @property
    def visible(self) -> bool:
        return self.status in (DETECTED, PREDICTED)

    @property
    def speed(self) -> float:
        return math.hypot(*self.velocity)


class BallKalman:
    """Kalman filter on [x, y, vx, vy] (position + velocity per frame), with drag."""

    def __init__(self, drag: float):
        self.drag = drag
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1 - drag, 0], [0, 0, 0, 1 - drag]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.diag([1.0, 1.0, 4.0, 4.0])    # how much the motion can change on its own
        self.R = np.diag([4.0, 4.0])              # how noisy a detection is (px²)
        self.ready = False

    def start(self, cx: float, cy: float) -> None:
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])
        self.ready = True

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def correct(self, cx: float, cy: float) -> tuple[float, float]:
        y = np.array([cx, cy]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def uncertainty(self) -> float:
        """Rough position uncertainty in px (1 standard deviation)."""
        return math.sqrt(max(self.P[0, 0], self.P[1, 1]))


class BallTracker:
    def __init__(self, cfg: dict, cam_size: tuple[int, int]):
        self.cfg = cfg["ball_tracker"]
        self.handoff_conf = self.cfg["handoff_conf"]
        self.width, self.height = cam_size
        self.kalman = BallKalman(self.cfg["drag"])
        self.missed = 0
        self.last_size = (12.0, 12.0)
        self.last_conf = 0.0
        self.seed_point: tuple[float, float] | None = None
        self.seed_left = 0
        self.stats = {"frames": 0, DETECTED: 0, PREDICTED: 0, LOST: 0, "patch": 0, "full": 0, "handoff": 0}

    # ------------------------------------------------------------------ public
    @property
    def tracking(self) -> bool:
        return self.kalman.ready

    def seed(self, point: tuple[float, float], frames: int) -> None:
        """Hand-off: the ball should enter this camera near `point` soon. Search there first."""
        if not self.tracking:
            self.seed_point, self.seed_left = point, frames

    def reset(self) -> None:
        self.kalman = BallKalman(self.cfg["drag"])
        self.missed = 0

    def step(self, frame: np.ndarray, detector, full_balls: list[dict] | None) -> BallState:
        """Process one frame. `full_balls` = this camera's full-frame candidates if the engine ran a
        full search this frame, else None."""
        self.stats["frames"] += 1
        state = self._step(frame, detector, full_balls)
        self.stats[state.status] += 1
        if state.status == DETECTED:
            self.stats[state.source] += 1
        return state

    # ------------------------------------------------------------------ the logic
    def _step(self, frame, detector, full_balls) -> BallState:
        if self.tracking:
            predicted = self.kalman.predict()
            radius = self._radius()
            found = self._best(self._search_patch(frame, detector, predicted, radius), predicted, radius,
                               self.cfg["min_conf_tracking"])
            source = "patch"
            if found is None and full_balls is not None:
                found = self._best(full_balls, predicted, radius, self.cfg["min_conf_tracking"])
                source = "full"
                # Shaky track (already missing frames) + a much surer ball elsewhere: switch to it.
                if found is None and self.missed > 0:
                    surest = self._best(full_balls, None, None, self.cfg["reacquire_conf"])
                    if surest is not None:
                        self.kalman.start(*_center(surest["bbox"]))
                        return self._hit(surest, "full", corrected=False)
            if found is not None:
                return self._hit(found, source)
            return self._miss()

        # Not tracking yet: first the hand-off hint, then a full-frame search if one ran.
        if self.seed_left > 0 and self.seed_point is not None:
            self.seed_left -= 1
            radius = self.cfg["patch_size_max"] / 2
            point = (min(max(self.seed_point[0], 0), self.width - 1), min(max(self.seed_point[1], 0), self.height - 1))
            found = self._best(self._search_patch(frame, detector, point, radius), point, radius, self.handoff_conf)
            if found is not None:
                self.seed_left = 0
                self.kalman.start(*_center(found["bbox"]))
                return self._hit(found, "handoff", corrected=False)
        if full_balls:
            found = self._best(full_balls, None, None, self.cfg["start_conf"])
            if found is not None:
                self.kalman.start(*_center(found["bbox"]))
                return self._hit(found, "full", corrected=False)
        return BallState()

    def _hit(self, cand: dict, source: str, corrected: bool = True) -> BallState:
        cx, cy = _center(cand["bbox"])
        if corrected:
            cx, cy = self.kalman.correct(cx, cy)
        b = cand["bbox"]
        self.last_size = (b[2] - b[0], b[3] - b[1])
        self.last_conf = cand["conf"]
        self.missed = 0
        return BallState(DETECTED, (cx, cy), self._box(cx, cy), cand["conf"], source, self.kalman.velocity, 0)

    def _miss(self) -> BallState:
        self.missed += 1
        cx, cy = self.kalman.position
        outside = not (-20 <= cx <= self.width + 20 and -20 <= cy <= self.height + 20)
        if self.missed > self.cfg["max_predict_frames"] or outside:
            self.reset()
            return BallState(missed=self.missed)
        conf = max(0.05, self.last_conf * (1 - self.missed / (self.cfg["max_predict_frames"] + 1)))
        return BallState(PREDICTED, (cx, cy), self._box(cx, cy), round(conf, 3), "predicted",
                         self.kalman.velocity, self.missed)

    def _radius(self) -> float:
        return max(self.cfg["search_radius"], self.kalman.uncertainty * 2.5 + self.missed * 15.0)

    def _search_patch(self, frame, detector, point, radius) -> list[dict]:
        size = int(np.clip(radius * 3, self.cfg["patch_size"], self.cfg["patch_size_max"])) // 32 * 32
        size = min(size, (min(self.width, self.height) // 32) * 32)
        x1 = int(np.clip(point[0] - size / 2, 0, self.width - size))
        y1 = int(np.clip(point[1] - size / 2, 0, self.height - size))
        return detector.patch(frame[y1:y1 + size, x1:x1 + size], (x1, y1))

    def _best(self, cands: list[dict], predicted, radius, min_conf: float) -> dict | None:
        """The most plausible candidate, or None. Lower penalty = better."""
        best, best_penalty = None, float("inf")
        for c in cands:
            if c["conf"] < min_conf or not self._looks_like_ball(c["bbox"]):
                continue
            if predicted is not None:
                dist = math.dist(_center(c["bbox"]), predicted)
                if dist > radius:
                    continue
                penalty = 0.6 * dist / radius + 0.4 * (1 - c["conf"])
            else:
                penalty = 1 - c["conf"]
            if penalty < best_penalty:
                best, best_penalty = c, penalty
        return best

    def _looks_like_ball(self, b: list[float]) -> bool:
        w, h = b[2] - b[0], b[3] - b[1]
        lo, hi = self.cfg["min_ball_px"], self.cfg["max_ball_px"]
        if not (lo <= w <= hi and lo <= h <= hi):
            return False
        return max(w / h, h / w) <= self.cfg["max_aspect"]

    def _box(self, cx: float, cy: float) -> list[float]:
        w, h = self.last_size
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
