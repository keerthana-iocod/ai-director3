"""run.py — one command: track (Step 1) then direct + render (Step 2).

Run:  venv/bin/python -m mvp3.run pano
      venv/bin/python -m mvp3.run pano --max-seconds 30      (quick test on the first 30 s)
      venv/bin/python -m mvp3.run pano --skip-tracking        (reuse tracks, re-direct + re-render only)
"""
from __future__ import annotations

import argparse

from mvp3.common import data_dir, load_config
from mvp3.direct import direct
from mvp3.track import track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--skip-tracking", action="store_true", help="reuse data/<clip>/tracks.jsonl")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if not args.skip_tracking or not (data_dir(args.clip) / "tracks.jsonl").exists():
        track(cfg, args.clip, args.start_seconds, args.max_seconds)
    direct(cfg, args.clip, render=not args.no_render)


if __name__ == "__main__":
    main()
