"""track.py — Step 1: follow the ball (and count players) in every camera, frame by frame.

For every frame:
  1. Every `global_every` frames: full-frame YOLO search on all cameras at once -> players + ball
     candidates. Cheap (640) while some camera tracks the ball; big (1280) and more often while
     no camera has it.
  2. Every camera's BallTracker: zoomed search around its prediction, plausibility check, Kalman.
  3. Hand-off: a ball at a shared border is announced to the neighbouring camera.
  4. One line per frame is appended to data/<clip>/tracks.jsonl.

This is the slow part (YOLO). It saves everything, so Step 2 (direct.py: director, virtual zoom,
videos) can be re-run and tuned in seconds without detecting again. The loop itself is live-shaped:
it never looks at a future frame.

Run:  venv/bin/python -m mvp3.track pano                 (whole clip)
      venv/bin/python -m mvp3.track pano --max-seconds 20   (quick test)
"""
from __future__ import annotations

import argparse
import json
import time

from mvp3.ball_tracker import BallTracker
from mvp3.common import clip_config, data_dir, load_config
from mvp3.detector import Detector
from mvp3.handoff import hand_off
from mvp3.sources import CameraRig


def track(cfg: dict, clip: str, start_seconds: float = 0.0, max_seconds: float | None = None) -> dict:
    rig = CameraRig(clip_config(cfg, clip))
    detector = Detector(cfg)
    trackers = [BallTracker(cfg, size) for size in rig.cam_sizes]
    dcfg, hcfg = cfg["detector"], cfg["handoff"]
    use_handoff = hcfg["enabled"] and rig.handoff_possible
    out = data_dir(clip)
    part = out / "tracks.jsonl.part"

    print(f"[track] {clip}: {rig.num_cams} cams {rig.cam_sizes[0][0]}x{rig.cam_sizes[0][1]} @ {rig.fps:.0f} fps, "
          f"device={detector.device}, hand-off={'on' if use_handoff else 'off'}")
    players = [[] for _ in range(rig.num_cams)]
    any_visible = False
    baseline = {"full_frames": 0, "full_any_ball": 0, "tracker_any_ball_same_frames": 0}
    handoffs = 0
    started = time.perf_counter()
    frames = 0

    with open(part, "w") as f:
        for tick in rig.ticks(start_seconds, max_seconds):
            every = dcfg["global_every"] if any_visible else dcfg["global_every_lost"]
            imgsz = dcfg["global_imgsz_tracking"] if any_visible else dcfg["global_imgsz_lost"]
            run_full = tick.index % every == 0
            full_balls = [None] * rig.num_cams
            if run_full:
                results = detector.full(tick.cams, imgsz)
                for c, det in enumerate(results):
                    full_balls[c] = det.balls
                    players[c] = det.players

            states = [trk.step(img, detector, full_balls[c]) for c, (trk, img) in enumerate(zip(trackers, tick.cams))]
            any_visible = any(s.visible for s in states)

            seeded = []
            if use_handoff:
                seeded = hand_off(states, trackers, rig.offsets, [w for w, _ in rig.cam_sizes],
                                  hcfg["edge_px"], hcfg["seed_frames"])
                handoffs += len(seeded)

            if run_full:   # "plain detector" baseline: would a single full-frame search have found a ball?
                baseline["full_frames"] += 1
                start_conf = cfg["ball_tracker"]["start_conf"]
                baseline["full_any_ball"] += any(any(b["conf"] >= start_conf for b in fb) for fb in full_balls)
                baseline["tracker_any_ball_same_frames"] += any(s.status == "detected" for s in states)

            row = {"i": tick.index, "t": round(tick.time, 3), "full": run_full,
                   "cams": [{"ball": s.to_dict(), "players": [[round(v) for v in p] for p in players[c]]}
                            for c, s in enumerate(states)],
                   "handoff": [[a + 1, b + 1] for a, b in seeded]}
            f.write(json.dumps(row) + "\n")
            frames += 1
            if frames % 125 == 0:
                fps = frames / (time.perf_counter() - started)
                status = " ".join(f"cam{c + 1}:{s.status[0]}" for c, s in enumerate(states))
                print(f"[track] t={tick.time:6.1f}s  {status}  {fps:.1f} fps")

    part.replace(out / "tracks.jsonl")
    elapsed = time.perf_counter() - started
    summary = {
        "clip": clip, "frames": frames, "fps_video": rig.fps, "seconds": round(frames / rig.fps, 1),
        "processing_fps": round(frames / elapsed, 2), "realtime_factor": round(frames / rig.fps / elapsed, 3),
        "full_searches": detector.calls_full, "patch_searches": detector.calls_patch, "handoffs": handoffs,
        "per_cam": [t.stats for t in trackers], "baseline": baseline,
        "cam_sizes": rig.cam_sizes, "num_cams": rig.num_cams, "start_seconds": start_seconds,
    }
    (out / "tracking_summary.json").write_text(json.dumps(summary, indent=1))
    print_summary(summary)
    return summary


def print_summary(s: dict) -> None:
    print(f"\n[track] {s['clip']}: {s['frames']} frames ({s['seconds']} s) in "
          f"{s['frames'] / s['processing_fps']:.0f} s -> {s['processing_fps']} fps "
          f"({s['realtime_factor']}x real time)")
    for c, st in enumerate(s["per_cam"], start=1):
        n = max(1, st["frames"])
        print(f"  cam{c}: ball detected {100 * st['detected'] / n:4.1f}%  predicted {100 * st['predicted'] / n:4.1f}%  "
              f"lost {100 * st['lost'] / n:4.1f}%   (found by zoomed patch {st['patch']}, full search {st['full']}, "
              f"hand-off {st['handoff']})")
    b = s["baseline"]
    if b["full_frames"]:
        print(f"  plain detector (full search only) found a ball on {100 * b['full_any_ball'] / b['full_frames']:.0f}% "
              f"of its frames; the tracker detected one on {100 * b['tracker_any_ball_same_frames'] / b['full_frames']:.0f}% "
              f"of the same frames")
    print(f"  hand-offs: {s['handoffs']}   searches: {s['full_searches']} full, {s['patch_searches']} zoomed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()
    track(load_config(), args.clip, args.start_seconds, args.max_seconds)


if __name__ == "__main__":
    main()
