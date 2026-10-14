# AI Director — MVP 3: camera switching + ball tracking

MVP 3 combines the two earlier prototypes:

| From | What MVP 3 takes |
|---|---|
| **MVP 1** (`~/ai-director/ai-director`) | Several cameras filming the same moment. A director picks the live camera on every frame (ball detected +100, predicted +60, near the centre up to +30, players up to +10; minimum 2 s shots; hysteresis so it doesn't flicker). |
| **ai-director2** (`~/ai-director2/football-ai-director/cam_director`) | The high-recall ball tracker: Kalman prediction, a zoomed search patch around the predicted ball, and a plausibility check. Also the virtual zoom ("PTZ") that follows the ball. |
| **New in MVP 3** | The ball is **handed off** between cameras: when it leaves cam1 on the right, cam2 already searches where it will enter. Also fixes ai-director2's tracker, whose fake "gravity" pulled predicted balls down the picture. |

```
3 cameras ─► per camera: full YOLO search (every few frames) + zoomed patch search (every frame)
          ─► Kalman tracker per camera: detected / predicted / lost
          ─► hand-off at the borders between cameras
          ─► director: which camera is live
          ─► virtual zoom: a smooth 16:9 crop of the live camera follows the ball
          ─► program.mp4 (the broadcast) + debug.mp4 (all cameras with markers) + logs
```

No Roboflow and no API key. The detector is the football YOLO model
[`gianpaj/football-players-detection-1`](https://huggingface.co/gianpaj/football-players-detection-1),
downloaded once into `models/best.pt`.

## Install (once)

```bash
cd ~/ai-director3
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Run

```bash
venv/bin/python -m mvp3.run pano --max-seconds 30     # quick test: first 30 s
venv/bin/python -m mvp3.run pano                      # whole clip (~30 min on an M1)
venv/bin/python -m mvp3.run pano --skip-tracking      # after changing director/zoom settings: seconds
venv/bin/python -m pytest tests -q                    # unit tests (no video or model needed)
```

The two steps can also run separately:

| Step | Command | What it does | Speed (M1) |
|---|---|---|---|
| 1. Track | `venv/bin/python -m mvp3.track pano` | YOLO + ball trackers + hand-off → `data/pano/tracks.jsonl` | slow (~2 fps) |
| 2. Direct | `venv/bin/python -m mvp3.direct pano` | director + virtual zoom + videos, from the saved tracks | fast |

## Outputs (`outputs/<clip>/`)

| File | What you see |
|---|---|
| `program.mp4` | The broadcast: the zoomed crop of the live camera, 1280×720, "CAM n" label, original sound |
| `debug.mp4` | All cameras side by side. Ball circle and trail: **green** = detected, **yellow** = predicted, **cyan** = picked up through hand-off. Grey boxes = players. The live camera has a red border and "LIVE"; the white box is the virtual-zoom crop. Bottom bar: time, scores, the director's reason. |
| `director_log.csv` | One row per frame: live camera, scores, ball status per camera, zoom, reason |
| `events.jsonl` | CUT / BALL_LOST / BALL_FOUND / HANDOFF events |
| `summary.txt` | Cuts, shot lengths, time live per camera, and how often the live camera (and the crop) has the ball |

`data/<clip>/tracking_summary.json` has the tracker numbers: per camera detected / predicted / lost, how
many balls were found by the zoomed patch vs the full search vs hand-off, and a "plain detector" baseline.

## Test videos

Set in `config.yaml` → `clips:`.

| Clip | What | Why |
|---|---|---|
| `pano` (default test) | Fixed panoramic camera, 2560×640, 141 s, cut into 3 cameras | The ball crosses all 3 cameras: tests switching, tracking and hand-off together |
| `spurs` | Tottenham–Watford tactical camera, 1920×1080, 300 s | Longer; the ball stays mostly in the middle strip |
| `alfheim` | **3 real cameras** (left / middle / right) from the [Alfheim dataset](https://datasets.simula.no/alfheim/), Tromsø–Strømsgodset, 1280×960, 30 fps, 60 s, no sound | Real multi-camera footage. Download first: `venv/bin/python -m mvp3.fetch_alfheim` (`--first`/`--chunks` pick another window, 3 s per chunk). Hand-off is off (separate cameras). |
| your own | `mode: files`, 3 separate synced videos | Real multi-camera footage (phones on tripods, the Alfheim or SoccerTrack v2 datasets). Hand-off is off in this mode (needs calibration). |

## Where things are

| Path | What |
|---|---|
| `config.yaml` | every tunable number |
| `mvp3/sources.py` | cameras: one wide video cut into strips, or separate synced files |
| `mvp3/detector.py` | the football YOLO model: full-frame search and zoomed patch search |
| `mvp3/ball_tracker.py` | per-camera ball tracker (Kalman + zoomed patch + plausibility check) |
| `mvp3/handoff.py` | passes the ball to the neighbouring camera at a shared border |
| `mvp3/track.py` | Step 1: runs all of the above frame by frame, saves the tracks |
| `mvp3/director.py` | MVP 1's camera-switching rules |
| `mvp3/ptz.py` | virtual zoom inside the live camera |
| `mvp3/direct.py` | Step 2: director + zoom + rendering + summary |
| `mvp3/run.py` | both steps in one command |
| `tests/` | unit tests with a fake detector |

## Known limits

- **Not real time on an M1.** Tracking runs at about 2 frames per second. The YOLOv8x model is large;
  a smaller model fine-tuned on football, an NVIDIA GPU, or a CoreML export are the ways to speed it up.
- **Hand-off needs touching cameras.** It works for split panoramas only. Separate real cameras need
  calibration first.
- **Pitch marks.** The penalty spot looks like a ball to YOLO (0.7 confidence). A "ball" that sits still for
  3 s is marked as a static spot and ignored after that, so a real ball resting there that long (a penalty) is missed.
- **No ground truth.** The numbers say how often the ball was tracked, not whether every marker was on
  the real ball: check `debug.mp4`.
