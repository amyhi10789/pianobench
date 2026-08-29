"""Launch a local website for human thumbs-up/down video evaluation.

Run from anywhere with:
    python human_video_evaluator.py

The app reads videos recursively from data/videos and does not modify them.
Press Ctrl+C in this terminal to stop the website.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_ROOT = PROJECT_ROOT / "data" / "videos"
RESULTS_PATH = PROJECT_ROOT / "human_evaluation_results.json"
VIDEO_EXTENSIONS = {".mp4", ".webm", ".ogg", ".ogv", ".mov", ".m4v"}
RESULTS_LOCK = threading.Lock()


def load_results() -> dict[str, int]:
    """Load cumulative totals, using empty totals for a missing/invalid file."""
    defaults = {"good": 0, "bad": 0, "evaluations": 0}
    try:
        stored = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return {
            key: max(0, int(stored.get(key, default)))
            for key, default in defaults.items()
        }
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return defaults


def add_results(good: int, bad: int) -> dict[str, int]:
    """Add one completed evaluation to the persistent cumulative totals."""
    with RESULTS_LOCK:
        totals = load_results()
        totals["good"] += good
        totals["bad"] += bad
        totals["evaluations"] += 1
        RESULTS_PATH.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
        return totals


def natural_key(path: Path) -> list[object]:
    """Sort paths so, for example, level2 comes before level10."""
    relative = path.relative_to(VIDEO_ROOT).as_posix().lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", relative)]


def find_videos() -> list[Path]:
    if not VIDEO_ROOT.is_dir():
        return []
    return sorted(
        (path for path in VIDEO_ROOT.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS),
        key=natural_key,
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PianoBench Human Evaluation</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #10131a; color: #f7f8fb; }
    main { width: min(1050px, 94vw); min-height: 100vh; margin: auto; display: grid; place-items: center; padding: 28px 0; }
    section { width: 100%; text-align: center; }
    h1 { font-size: clamp(2rem, 5vw, 4rem); margin: 0 0 18px; }
    p { color: #b9c0cf; font-size: 1.1rem; }
    button { border: 0; cursor: pointer; font: inherit; font-weight: 800; transition: transform .12s, filter .12s; }
    button:hover { transform: scale(1.025); filter: brightness(1.1); }
    #start { padding: 22px 40px; border-radius: 18px; font-size: clamp(1.3rem, 3vw, 2rem); background: #7c5cff; color: white; }
    #review, #results, #empty { display: none; }
    .topline { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 12px; color: #b9c0cf; }
    #filename { overflow-wrap: anywhere; text-align: right; }
    video { display: block; width: 100%; max-height: 64vh; border-radius: 14px; background: black; box-shadow: 0 18px 50px #0008; }
    .votes { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 22px; }
    .vote { min-height: 130px; border-radius: 22px; color: white; font-size: clamp(3.5rem, 9vw, 6rem); }
    .good { background: #198754; }
    .bad { background: #dc3545; }
    .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 32px auto; max-width: 800px; }
    .result { padding: 35px 15px; border-radius: 22px; font-size: clamp(1.5rem, 4vw, 2.8rem); font-weight: 900; }
    .result span { display: block; font-size: clamp(3rem, 9vw, 6rem); margin-top: 10px; }
    .overall { margin: 38px auto; padding-top: 28px; max-width: 800px; border-top: 1px solid #3a4050; }
    .overall h2 { font-size: clamp(1.7rem, 4vw, 2.5rem); margin-bottom: 8px; }
    .restart { padding: 15px 28px; border-radius: 12px; background: #7c5cff; color: white; }
    @media (max-width: 600px) { .votes, .result-grid { grid-template-columns: 1fr; } .vote { min-height: 90px; } }
  </style>
</head>
<body><main>
  <section id="welcome">
    <h1>PianoBench</h1>
    <p>Review every video and decide whether it is good or bad.</p>
    <button id="start">Evaluate videos</button>
  </section>
  <section id="empty">
    <h1>No videos found</h1>
    <p>Put videos inside <code>data/videos</code>, then refresh this page.</p>
  </section>
  <section id="review">
    <div class="topline"><strong id="progress"></strong><span id="filename"></span></div>
    <video id="video" controls playsinline preload="metadata"></video>
    <div class="votes">
      <button class="vote good" aria-label="Good video" title="Good">👍</button>
      <button class="vote bad" aria-label="Bad video" title="Bad">👎</button>
    </div>
  </section>
  <section id="results">
    <h1>Evaluation complete</h1>
    <p id="summary"></p>
    <div class="result-grid">
      <div class="result good">Good<span id="good-percent"></span></div>
      <div class="result bad">Bad<span id="bad-percent"></span></div>
    </div>
    <div class="overall">
      <h2>Overall saved results</h2>
      <p id="overall-summary">Saving and loading cumulative results...</p>
      <div class="result-grid">
        <div class="result good">Good<span id="overall-good-percent"></span></div>
        <div class="result bad">Bad<span id="overall-bad-percent"></span></div>
      </div>
    </div>
    <button class="restart">Evaluate again</button>
  </section>
</main>
<script>
  const videos = __VIDEOS__;
  let index = 0, good = 0, bad = 0;
  const $ = id => document.getElementById(id);
  function show(id) {
    for (const name of ['welcome', 'empty', 'review', 'results']) $(name).style.display = name === id ? 'block' : 'none';
  }
  function loadVideo() {
    const item = videos[index];
    $('progress').textContent = `Video ${index + 1} of ${videos.length}`;
    $('filename').textContent = item.name;
    $('video').src = item.url;
    $('video').load();
  }
  function start() {
    index = good = bad = 0;
    if (!videos.length) return show('empty');
    show('review'); loadVideo();
  }
  async function finish() {
    const total = good + bad;
    $('video').pause(); $('video').removeAttribute('src');
    $('good-percent').textContent = `${(100 * good / total).toFixed(1)}%`;
    $('bad-percent').textContent = `${(100 * bad / total).toFixed(1)}%`;
    $('summary').textContent = `This evaluation: ${good} good and ${bad} bad out of ${total} videos`;
    show('results');
    try {
      const response = await fetch('/results', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({good, bad})
      });
      if (!response.ok) throw new Error('Could not save results');
      const saved = await response.json();
      const overallTotal = saved.good + saved.bad;
      const goodPercent = overallTotal ? 100 * saved.good / overallTotal : 0;
      const badPercent = overallTotal ? 100 * saved.bad / overallTotal : 0;
      $('overall-good-percent').textContent = `${goodPercent.toFixed(1)}%`;
      $('overall-bad-percent').textContent = `${badPercent.toFixed(1)}%`;
      $('overall-summary').textContent = `${saved.good} good and ${saved.bad} bad across ${saved.evaluations} completed evaluation(s)`;
    } catch (error) {
      $('overall-summary').textContent = 'The cumulative results could not be saved or loaded.';
    }
  }
  function vote(isGood) {
    isGood ? good++ : bad++;
    index++;
    if (index < videos.length) return loadVideo();
    finish();
  }
  $('start').addEventListener('click', start);
  document.querySelector('.good.vote').addEventListener('click', () => vote(true));
  document.querySelector('.bad.vote').addEventListener('click', () => vote(false));
  document.querySelector('.restart').addEventListener('click', start);
</script></body></html>"""


class EvaluationHandler(BaseHTTPRequestHandler):
    videos: list[Path] = []

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        request_path = urlparse(self.path).path
        if request_path == "/":
            items = [
                {
                    "name": path.relative_to(VIDEO_ROOT).as_posix(),
                    "url": f"/video/{index}",
                }
                for index, path in enumerate(self.videos)
            ]
            page = PAGE.replace("__VIDEOS__", json.dumps(items)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return

        match = re.fullmatch(r"/video/(\d+)", unquote(request_path))
        if match and int(match.group(1)) < len(self.videos):
            self.serve_video(self.videos[int(match.group(1))])
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlparse(self.path).path != "/results":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024:
                raise ValueError("Invalid request size")
            submitted = json.loads(self.rfile.read(length))
            good = int(submitted["good"])
            bad = int(submitted["bad"])
            if good < 0 or bad < 0 or good + bad != len(self.videos):
                raise ValueError("Results must contain one vote per video")
            totals = add_results(good, bad)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.send_error(400, "Invalid evaluation results")
            return

        response = json.dumps(totals).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def serve_video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                self.send_error(416)
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(int(match.group(2) or end), end)
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as video:
            video.seek(start)
            remaining = length
            while remaining:
                chunk = video.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the PianoBench human video evaluator.")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: choose automatically)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically")
    args = parser.parse_args()

    EvaluationHandler.videos = find_videos()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), EvaluationHandler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Found {len(EvaluationHandler.videos)} video(s) in {VIDEO_ROOT}")
    print(f"Human evaluator: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
