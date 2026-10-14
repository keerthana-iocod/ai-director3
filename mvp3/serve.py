"""serve.py — watch the results in a browser.

Serves outputs/ on a local port, with one page per clip: program.mp4 (the broadcast), debug.mp4 (all
cameras with the ball markers), the summary and the cut/ball events.

Run:  venv/bin/python -m mvp3.serve              then open http://localhost:8000
      venv/bin/python -m mvp3.serve --port 8080
"""
from __future__ import annotations

import argparse
import html
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mvp3.common import ROOT

OUTPUTS = ROOT / "outputs"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Director results</title>
<style>
  body {{ margin: 0; padding: 16px; background: #111; color: #ddd; font: 15px/1.5 system-ui, sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 40px; border-top: 1px solid #333; padding-top: 16px; }}
  video {{ width: 100%; background: #000; border-radius: 6px; }}
  pre {{ background: #1b1b1b; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  .muted {{ color: #888; }}
</style></head><body><main>
<h1>AI Director — results</h1>
<p class="muted">Re-run a clip with <code>venv/bin/python -m mvp3.run &lt;clip&gt;</code>, then reload this page.</p>
{clips}
</main></body></html>"""


def _clip_section(clip_dir) -> str:
    name = clip_dir.name
    parts = [f"<h2>{html.escape(name)}</h2>"]
    source = ROOT / "data" / name / "source.txt"
    if source.exists():
        parts.append(f'<p class="muted">{html.escape(source.read_text().strip())}</p>')
    for video, label in (("program.mp4", "Broadcast (live camera, virtual zoom)"),
                         ("debug.mp4", "All cameras: green = ball detected, yellow = predicted, red border = live")):
        if (clip_dir / video).exists():
            parts.append(f"<h3>{label}</h3><video controls preload=\"metadata\" "
                         f"src=\"/{name}/{video}?v={int((clip_dir / video).stat().st_mtime)}\"></video>")
    if (clip_dir / "summary.txt").exists():
        parts.append(f"<h3>Summary</h3><pre>{html.escape((clip_dir / 'summary.txt').read_text())}</pre>")
    if (clip_dir / "events.jsonl").exists():
        lines = []
        for line in (clip_dir / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            lines.append(f"{e['time']:7.2f}s  {e['type']:<10} {e.get('message', '')}")
        parts.append(f"<h3>Events</h3><pre>{html.escape(chr(10).join(lines))}</pre>")
    return "\n".join(parts)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            clips = sorted((d for d in OUTPUTS.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
            body = PAGE.format(clips="\n".join(_clip_section(d) for d in clips) or "<p>No results yet.</p>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.path = self.path.split("?")[0]
        rng = self.headers.get("Range", "")
        path = Path(self.translate_path(self.path))
        if not (rng.startswith("bytes=") and path.is_file()):
            super().do_GET()
            return
        # Byte ranges: Safari will not play an mp4 without them, and every browser needs them to seek.
        size = path.stat().st_size
        start_s, _, end_s = rng[6:].split(",")[0].partition("-")
        start = int(start_s) if start_s else max(0, size - int(end_s))
        end = min(int(end_s), size - 1) if start_s and end_s else size - 1
        if start >= size or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):   # the browser stopped this request
                    return
                left -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    OUTPUTS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(OUTPUTS)))
    print(f"Serving results at http://localhost:{args.port}  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
