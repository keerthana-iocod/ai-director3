"""handoff.py — passes the ball from one camera to the next (split mode).

In split mode the cameras touch: cam1's right edge IS cam2's left edge. So when cam1 sees the ball
near its right edge moving right, we know exactly where it will appear in cam2, and tell cam2's
tracker to search there (BallTracker.seed). Without this, cam2 would only find the ball at its next
full-frame search, up to `global_every` frames later, and the director would cut late.

With separate real cameras we don't know how their pictures overlap (that needs calibration), so
hand-off is off in `files` mode.
"""
from __future__ import annotations

from mvp3.ball_tracker import BallState, BallTracker


def hand_off(states: list[BallState], trackers: list[BallTracker], offsets: list[int],
             cam_widths: list[int], edge_px: float, seed_frames: int) -> list[tuple[int, int]]:
    """Seed neighbouring cameras for every ball at a shared border. Returns [(from_cam, to_cam), ...] (0-based)."""
    seeded = []
    for i, s in enumerate(states):
        if not s.visible or s.center is None:
            continue
        x, y = s.center
        vx = s.velocity[0]
        # Next position, a few frames ahead, so a fast ball doesn't skip the edge zone.
        ahead = x + vx * 3
        if i + 1 < len(states) and (x > cam_widths[i] - edge_px or ahead >= cam_widths[i]) and vx >= 0:
            j = i + 1
        elif i > 0 and (x < edge_px or ahead < 0) and vx <= 0:
            j = i - 1
        else:
            continue
        if trackers[j].tracking:
            continue
        # Same point on the panorama, in the neighbour's own pixels.
        jx = x + offsets[i] - offsets[j]
        trackers[j].seed((jx, y), seed_frames)
        seeded.append((i, j))
    return seeded
