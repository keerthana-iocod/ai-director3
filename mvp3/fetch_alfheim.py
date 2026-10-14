"""fetch_alfheim.py — download 3 real, synced cameras from the Alfheim dataset.

Alfheim (https://datasets.simula.no/alfheim/) filmed Tromsø IL matches with 3 fixed cameras side by side
(cam 0 = left, 1 = middle, 2 = right). Each camera is stored as raw H.264 chunks of 3 s (90 frames,
1280x960, 30 fps, no sound). A chunk number is the same moment in every camera, so taking the same
chunk numbers from all 3 gives synced videos.

This downloads chunks `--first` .. `--first + --chunks - 1` for all 3 cameras and joins them into
data/alfheim/cam1.mp4, cam2.mp4, cam3.mp4 (cam1 = Alfheim camera 0). Chunks are cached, so running it
again with the same window downloads nothing.

Run:  venv/bin/python -m mvp3.fetch_alfheim                     (60 s from chunk 156, ~6 min into the match)
      venv/bin/python -m mvp3.fetch_alfheim --first 300 --chunks 40
"""
from __future__ import annotations

import argparse
import re
import subprocess
import urllib.parse

from mvp3.common import data_dir, ffmpeg_exe

BASE = "https://datasets.simula.no/downloads/alfheim"


def _get(url: str, dest=None) -> bytes:
    # The server sends the wrong intermediate certificate, so Python's ssl rejects it. macOS curl copes.
    args = ["curl", "-sSfL", "--retry", "3", url] + (["-o", str(dest)] if dest else [])
    return subprocess.run(args, check=True, capture_output=True).stdout


def _chunk_names(match: str, half: str, cam: int) -> dict[int, str]:
    """Chunk number -> file name, from the camera's directory listing."""
    url = f"{BASE}/{match}/{urllib.parse.quote(half)}/{cam}/"
    html = _get(url).decode()
    names = {urllib.parse.unquote(n) for n in re.findall(r'href="\./(\d{4}_[^"]+\.h264)"', html)}
    return {int(n[:4]): n for n in names}


def fetch(match: str, half: str, first: int, chunks: int) -> None:
    out = data_dir("alfheim")
    raw = out / "raw"
    wanted = list(range(first, first + chunks))
    # 1. Download every camera first. cam1.mp4..cam3.mp4 are only replaced once all 3 are complete,
    #    so a run started meanwhile never mixes two matches (or reads a half-written video).
    all_parts = []
    for cam in range(3):
        names = _chunk_names(match, half, cam)
        missing = [n for n in wanted if n not in names]
        if missing:
            raise SystemExit(f"Camera {cam} has no chunks {missing}. It has {min(names)}..{max(names)}.")
        cam_dir = raw / f"{match}_{half.replace(' ', '')}_{cam}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for n in wanted:
            part = cam_dir / f"{n:04d}.h264"
            if not part.exists():
                url = f"{BASE}/{match}/{urllib.parse.quote(half)}/{cam}/{urllib.parse.quote(names[n])}"
                print(f"cam{cam + 1}: chunk {n} ...", flush=True)
                tmp = part.with_suffix(".part")
                _get(url, tmp)
                tmp.rename(part)
            parts.append(part)
        all_parts.append(parts)

    # 2. Raw H.264 chunks of one camera are one stream cut in pieces: joining the bytes is enough.
    for cam, parts in enumerate(all_parts):
        joined = raw / f"cam{cam + 1}_joined.h264"
        with open(joined, "wb") as f:
            for part in parts:
                f.write(part.read_bytes())
        video = out / f"cam{cam + 1}.mp4"
        tmp = out / f"cam{cam + 1}.tmp.mp4"
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-r", "30", "-f", "h264", "-i", str(joined),
                        "-c:v", "copy", "-r", "30", str(tmp)], check=True)
        joined.unlink()
        tmp.replace(video)
        print(f"-> {video}")
    (out / "source.txt").write_text(f"{match} {half}, chunks {first}..{first + chunks - 1} "
                                    f"({first * 3 // 60}:{first * 3 % 60:02d} into the recording, {chunks * 3} s)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match", default="2013-11-03", help="2013-11-03 (Tromsø–Strømsgodset) or 2013-11-07")
    parser.add_argument("--half", default="First Half")
    parser.add_argument("--first", type=int, default=156, help="first chunk number (3 s each)")
    parser.add_argument("--chunks", type=int, default=20, help="how many 3 s chunks (20 = 60 s)")
    args = parser.parse_args()
    fetch(args.match, args.half, args.first, args.chunks)


if __name__ == "__main__":
    main()
