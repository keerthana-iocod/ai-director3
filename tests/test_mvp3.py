"""Unit tests for MVP 3 (no video, no model needed: a fake detector stands in for YOLO).

Run:  venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import numpy as np

from mvp3.ball_tracker import DETECTED, LOST, PREDICTED, BallKalman, BallState, BallTracker
from mvp3.common import load_config
from mvp3.director import Director, shot_lengths
from mvp3.handoff import hand_off
from mvp3.ptz import VirtualCamera

CFG = load_config()
SIZE = (853, 640)
FRAME = np.zeros((640, 853, 3), dtype=np.uint8)


class FakeDetector:
    """Returns a ball at `ball` (camera pixels) if it falls inside the searched patch."""

    def __init__(self, ball=None, conf=0.8):
        self.ball, self.conf = ball, conf

    def patch(self, image, offset):
        if self.ball is None:
            return []
        x, y = self.ball
        ox, oy = offset
        if ox <= x < ox + image.shape[1] and oy <= y < oy + image.shape[0]:
            return [{"bbox": [x - 5, y - 5, x + 5, y + 5], "conf": self.conf}]
        return []


def ball_at(x, y, conf=0.8):
    return [{"bbox": [x - 5, y - 5, x + 5, y + 5], "conf": conf}]


# ------------------------------------------------------------------ Kalman: no fake gravity
def test_kalman_predicts_straight_line_without_falling():
    k = BallKalman(drag=0.0)
    k.start(100, 300)
    for i in range(1, 6):
        k.predict()
        k.correct(100 + 10 * i, 300)
    x, y = k.predict()
    assert 150 < x < 170          # keeps moving right
    assert abs(y - 300) < 2       # does NOT slide down the picture (ai-director2's gravity bug)


# ------------------------------------------------------------------ tracker
def test_tracker_needs_a_sure_ball_to_start():
    t = BallTracker(CFG, SIZE)
    assert t.step(FRAME, FakeDetector(), ball_at(400, 300, conf=0.2)).status == LOST   # below start_conf
    s = t.step(FRAME, FakeDetector(), ball_at(400, 300, conf=0.6))
    assert s.status == DETECTED and s.source == "full"


def test_tracker_follows_with_zoomed_patch_then_predicts_then_loses():
    t = BallTracker(CFG, SIZE)
    t.step(FRAME, FakeDetector(), ball_at(400, 300))
    s = t.step(FRAME, FakeDetector(ball=(405, 300)), None)
    assert s.status == DETECTED and s.source == "patch"
    s = t.step(FRAME, FakeDetector(), None)                    # detector misses it
    assert s.status == PREDICTED
    for _ in range(CFG["ball_tracker"]["max_predict_frames"] + 1):
        s = t.step(FRAME, FakeDetector(), None)
    assert s.status == LOST


def test_tracker_rejects_a_far_jump_and_non_ball_shapes():
    t = BallTracker(CFG, SIZE)
    t.step(FRAME, FakeDetector(), ball_at(400, 300))
    far = t.step(FRAME, FakeDetector(), ball_at(800, 50, conf=0.4))      # too far from the prediction
    assert far.status == PREDICTED
    t2 = BallTracker(CFG, SIZE)
    sock = [{"bbox": [400, 300, 404, 340], "conf": 0.9}]                   # tall and thin
    assert t2.step(FRAME, FakeDetector(), sock).status == LOST


def test_tracker_drops_a_ball_that_never_moves_like_the_penalty_spot():
    t = BallTracker(CFG, SIZE)
    spot = FakeDetector(ball=(400, 300))
    t.step(FRAME, spot, ball_at(400, 300))
    statuses = [t.step(FRAME, spot, ball_at(400, 300)).status for _ in range(CFG["ball_tracker"]["static_frames"] + 5)]
    assert statuses[0] == DETECTED and statuses[-1] == LOST          # marked as a static spot
    assert t.step(FRAME, spot, ball_at(400, 300, conf=0.9)).status == LOST     # never restarts there
    s = t.step(FRAME, FakeDetector(), ball_at(500, 320, conf=0.6))             # a real ball elsewhere
    assert s.status == DETECTED


def test_handoff_seeds_the_next_camera():
    trackers = [BallTracker(CFG, SIZE) for _ in range(3)]
    states = [BallState(DETECTED, (840.0, 300.0), velocity=(6.0, 0.0)), BallState(), BallState()]
    seeded = hand_off(states, trackers, [0, 853, 1706], [853] * 3, edge_px=60, seed_frames=10)
    assert seeded == [(0, 1)]
    assert trackers[1].seed_point == (840.0 - 853, 300.0)       # just left of cam2's left edge
    s = trackers[1].step(FRAME, FakeDetector(ball=(8, 300)), None)
    assert s.status == DETECTED and s.source == "handoff"


def test_no_handoff_when_ball_moves_away_from_the_border():
    trackers = [BallTracker(CFG, SIZE) for _ in range(3)]
    states = [BallState(DETECTED, (840.0, 300.0), velocity=(-6.0, 0.0)), BallState(), BallState()]
    assert hand_off(states, trackers, [0, 853, 1706], [853] * 3, 60, 10) == []


# ------------------------------------------------------------------ director
def test_director_respects_min_shot_and_hysteresis():
    fps = 25
    d = Director(CFG, fps, 3)
    live = []
    for f in range(200):
        on_cam = 0 if f < 100 else 2                      # ball moves from cam1 to cam3 at 4 s
        balls = [BallState(DETECTED, (400.0, 300.0)) if c == on_cam else BallState() for c in range(3)]
        live.append(d.decide(f, balls, [5, 5, 5], [853] * 3).cam)
    assert live[0] == 2 and live[-1] == 3
    assert min(shot_lengths(live, fps)) >= CFG["director"]["min_shot_seconds"] - 1e-9
    cut_to_3 = live.index(3)
    assert cut_to_3 >= 100 + round(CFG["director"]["hysteresis_seconds"] * fps) - 1


# ------------------------------------------------------------------ virtual camera
def test_virtual_camera_stays_inside_and_follows_smoothly():
    v = VirtualCamera(CFG)
    box = v.update(2, SIZE, BallState(DETECTED, (100.0, 100.0)), [])
    assert 0 <= box[0] < box[2] <= SIZE[0] and 0 <= box[1] < box[3] <= SIZE[1]
    assert abs((box[2] - box[0]) / (box[3] - box[1]) - 16 / 9) < 0.02
    before = box
    box = v.update(2, SIZE, BallState(DETECTED, (800.0, 600.0)), [])
    assert abs(box[0] - before[0]) <= CFG["ptz"]["max_pan_px"] + 1        # no whip pan
    snap = v.update(3, SIZE, BallState(DETECTED, (800.0, 600.0)), [])     # a cut snaps
    assert snap[2] >= 780


def test_tracker_drops_two_marks_even_when_the_track_hops_between_them():
    t = BallTracker(CFG, SIZE)
    t.step(FRAME, FakeDetector(), ball_at(400, 300))
    statuses = []
    for i in range(2 * CFG["ball_tracker"]["static_frames"] + 10):
        mark = (400, 300) if (i // 10) % 2 == 0 else (430, 320)          # hops every 10 frames
        statuses.append(t.step(FRAME, FakeDetector(ball=mark), ball_at(*mark)).status)
    assert len(t.static_spots) == 2 and statuses[-1] == LOST                # both marks caught
