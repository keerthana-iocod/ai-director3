"""direct.py — Step 2: direct the show from the saved tracks and render it.

Reads data/<clip>/tracks.jsonl (from track.py; no detection runs here, so this takes seconds to
minutes, and you can tune the director/zoom numbers in config.yaml and re-run freely).

For every frame:
  1. Director: which camera is live (MVP 1 rules, fed by the high-recall ball tracker).
  2. Virtual camera: a smooth 16:9 crop of the live camera that follows the ball.
  3. Render:
       outputs/<clip>/program.mp4   the broadcast: the crop, 1280x720, "CAM n" label, original sound
       outputs/<clip>/debug.mp4     all cameras side by side: ball marker + trail (green detected,
                                    yellow predicted, cyan = found through hand-off), player boxes,
                                    red border + white crop box on the live camera, scores at the bottom
       outputs/<clip>/director_log.csv, events.jsonl, summary.txt

Run:  venv/bin/python -m mvp3.direct pano
      venv/bin/python -m mvp3.direct pano --no-render    (numbers only, very fast)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from mvp3.ball_tracker import DETECTED, PREDICTED, BallState
from mvp3.common import clip_config, data_dir, ffmpeg_exe, load_config, output_dir
from mvp3.director import DecisionLog, Director, shot_lengths
from mvp3.ptz import VirtualCamera
from mvp3.sources import CameraRig

GREEN, YELLOW, CYAN, RED, WHITE, GREY = (80, 220, 80), (0, 220, 255), (255, 220, 0), (40, 40, 230), (255, 255, 255), (150, 150, 150)


def load_tracks(clip: str) -> tuple[list[dict], dict]:
    d = data_dir(clip)
    if not (d / "tracks.jsonl").exists():
        raise SystemExit(f"No tracks yet. Run first: venv/bin/python -m mvp3.track {clip}")
    rows = [json.loads(line) for line in (d / "tracks.jsonl").read_text().splitlines() if line.strip()]
    return rows, json.loads((d / "tracking_summary.json").read_text())


def direct(cfg: dict, clip: str, render: bool = True) -> dict:
    rows, tsum = load_tracks(clip)
    rig = CameraRig(clip_config(cfg, clip))
    sizes = [tuple(s) for s in tsum["cam_sizes"]]
    widths = [w for w, _ in sizes]
    out = output_dir(clip)
    director = Director(cfg, rig.fps, rig.num_cams)
    vcam = VirtualCamera(cfg)
    log = DecisionLog(out / "director_log.csv", rig.num_cams)
    events_file = open(out / "events.jsonl", "w")
    rcfg = cfg["render"]

    program = debug = None
    frames_iter = rig.ticks(tsum["start_seconds"], len(rows) / rig.fps + 0.5 / rig.fps) if render else None
    trails = [deque(maxlen=rcfg["trail_frames"]) for _ in range(rig.num_cams)]
    live_cams, zooms = [], []
    on_air = Counter()

    for row in rows:
        balls = [BallState.from_dict(c["ball"]) for c in row["cams"]]
        players = [c["players"] for c in row["cams"]]
        d = director.decide(row["i"], balls, [len(p) for p in players], widths)
        live = d.cam - 1
        box = vcam.update(d.cam, sizes[live], balls[live], players[live])
        log.write(row["i"], row["t"], d, balls, vcam.zoom)
        for e in d.events + [{"type": "HANDOFF", "message": f"cam{a} -> cam{b}"} for a, b in row["handoff"]]:
            events_file.write(json.dumps({"frame": row["i"], "time": row["t"], **e}) + "\n")
        live_cams.append(d.cam)
        zooms.append(vcam.zoom)
        _count_on_air(on_air, balls, live, box)

        if render:
            tick = next(frames_iter, None)
            if tick is None:
                break
            for c, b in enumerate(balls):
                trails[c].append((b.center, b.status, b.source) if b.visible else None)
            prog = _program_frame(tick.cams[live], box, d.cam, rcfg["program_size"])
            dbg = _debug_frame(tick.cams, balls, players, trails, live, box, d, row["t"], rcfg["debug_tile_height"])
            if program is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                program = cv2.VideoWriter(str(out / "program_noaudio.mp4"), fourcc, rig.fps, tuple(rcfg["program_size"]))
                debug = cv2.VideoWriter(str(out / "debug_raw.mp4"), fourcc, rig.fps, (dbg.shape[1], dbg.shape[0]))
            program.write(prog)
            debug.write(dbg)
            if row["i"] % 250 == 0:
                print(f"[direct] rendering t={row['t']:6.1f}s")

    log.close()
    events_file.close()
    if program is not None:
        program.release()
        debug.release()
        _finish_videos(rig, out, tsum["start_seconds"], len(live_cams) / rig.fps)

    summary = _summary(clip, live_cams, zooms, on_air, rig.fps, rig.num_cams, out)
    (out / "summary.txt").write_text(summary)
    print(summary)
    return {"live": live_cams}


# ---------------------------------------------------------------------------------------- numbers
def _count_on_air(on_air: Counter, balls: list[BallState], live: int, box) -> None:
    """Director quality: when SOME camera has the ball, is it the live one, and inside the crop?"""
    if not any(b.status == DETECTED for b in balls):
        return
    on_air["ball_frames"] += 1
    b = balls[live]
    if b.status == DETECTED:
        on_air["live_has_ball"] += 1
        x, y = b.center
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            on_air["ball_in_crop"] += 1


def _summary(clip, live, zooms, on_air, fps, num_cams, out: Path) -> str:
    shots = shot_lengths(live, fps)
    cuts = len(shots) - 1
    share = Counter(live)
    n = max(1, on_air["ball_frames"])
    lines = [
        f"MVP 3 director summary — {clip}",
        f"length {len(live) / fps:.1f} s, {cuts} cuts, shots: shortest {min(shots):.2f} s, "
        f"average {sum(shots) / len(shots):.2f} s, longest {max(shots):.2f} s",
        "time live per camera: " + ", ".join(f"cam{c} {100 * share[c] / len(live):.0f}%" for c in range(1, num_cams + 1)),
        f"frames where some camera detects the ball: {on_air['ball_frames']}",
        f"  ...and the LIVE camera has it: {100 * on_air['live_has_ball'] / n:.0f}%",
        f"  ...and it is inside the virtual-zoom crop: {100 * on_air['ball_in_crop'] / n:.0f}%",
        f"average zoom {np.mean(zooms):.2f}x",
        f"files: {out}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------- drawing
def _program_frame(img: np.ndarray, box, cam: int, size) -> np.ndarray:
    x1, y1, x2, y2 = box
    frame = cv2.resize(img[y1:y2, x1:x2], tuple(size), interpolation=cv2.INTER_LINEAR)
    _label(frame, f"CAM {cam}", (16, 34), 0.8)
    return frame


def _debug_frame(cams, balls, players, trails, live, box, d, t, tile_h) -> np.ndarray:
    tiles = []
    for c, img in enumerate(cams):
        h, w = img.shape[:2]
        s = tile_h / h
        tile = cv2.resize(img, (round(w * s), tile_h), interpolation=cv2.INTER_AREA)
        for p in players[c]:
            cv2.rectangle(tile, (int(p[0] * s), int(p[1] * s)), (int(p[2] * s), int(p[3] * s)), GREY, 1)
        pts = [(int(pt[0][0] * s), int(pt[0][1] * s), pt[1], pt[2]) for pt in trails[c] if pt]
        for a, b in zip(pts, pts[1:]):
            cv2.line(tile, a[:2], b[:2], _color(b[2], b[3]), 2)
        ball = balls[c]
        if ball.visible:
            x, y = int(ball.center[0] * s), int(ball.center[1] * s)
            cv2.circle(tile, (x, y), 9, _color(ball.status, ball.source), 2)
        if c == live:
            cv2.rectangle(tile, (int(box[0] * s), int(box[1] * s)), (int(box[2] * s), int(box[3] * s)), WHITE, 2)
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), RED, 6)
            _label(tile, "LIVE", (tile.shape[1] - 70, 30), 0.7, RED)
        _label(tile, f"cam{c + 1}  {ball.status}{' (' + ball.source + ')' if ball.status == DETECTED else ''}", (8, 26), 0.55)
        tiles.append(tile)
    strip = np.hstack(tiles)
    bar = np.zeros((44, strip.shape[1], 3), dtype=np.uint8)
    text = f"t={t:6.2f}s  live=cam{d.cam}  scores " + " / ".join(f"{s:.0f}" for s in d.scores) + f"   {d.reason}"
    _label(bar, text[:140], (10, 29), 0.55)
    return np.vstack([strip, bar])


def _color(status: str, source: str):
    if status == PREDICTED:
        return YELLOW
    return CYAN if source == "handoff" else GREEN


def _label(img, text, org, scale, color=WHITE) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _finish_videos(rig: CameraRig, out: Path, start: float, duration: float) -> None:
    """Re-encode to H.264 (plays everywhere) and copy the original sound into program.mp4."""
    ff = ffmpeg_exe()
    audio_at = rig.audio_start + start
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out / "program_noaudio.mp4"),
                    "-ss", f"{audio_at:.3f}", "-t", f"{duration:.3f}", "-i", str(rig.audio_video),
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-c:a", "aac", "-shortest", str(out / "program.mp4")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out / "debug_raw.mp4"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "23", str(out / "debug.mp4")], check=True)
    (out / "program_noaudio.mp4").unlink(missing_ok=True)
    (out / "debug_raw.mp4").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--no-render", action="store_true", help="only the director numbers, no videos")
    args = parser.parse_args()
    direct(load_config(), args.clip, render=not args.no_render)


if __name__ == "__main__":
    main()
