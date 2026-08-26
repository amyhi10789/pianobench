"""Visual piano-key accuracy scoring using sampled frames and a VLM."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gpt-5.4-mini"
FRAME_OFFSETS = (-0.60, -0.30, -0.10, 0.10, 0.30, 0.60)


@dataclass
class VideoAccuracyResult:
    score: float
    expected_note: str
    expected_press_time: float
    details: str


def _ffmpeg_executable() -> str:
    """Resolve ffmpeg, allowing a bundled imageio-ffmpeg binary as fallback."""
    configured = os.environ.get("PIANOBENCH_FFMPEG")
    if configured:
        return configured
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError(
        "No usable ffmpeg binary found. Install imageio-ffmpeg or set "
        "PIANOBENCH_FFMPEG to a working ffmpeg.exe path."
    )


def _extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
        "-i", str(video_path), "-frames:v", "1", "-vf",
        "scale=1024:-2:force_original_aspect_ratio=decrease",
        "-q:v", "3", "-y", str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not produce a frame at {timestamp:.3f}s")


def _sample_frames(video_path: str, expected_time: float) -> list[tuple[float, str]]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    samples: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, offset in enumerate(FRAME_OFFSETS):
            timestamp = max(0.0, expected_time + offset)
            frame_path = Path(temp_dir) / f"frame_{index}.jpg"
            _extract_frame(path, timestamp, frame_path)
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
            samples.append((timestamp, f"data:image/jpeg;base64,{encoded}"))
    return samples


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observed_note": {"type": ["string", "null"]},
            "press_time": {"type": ["number", "null"]},
            "extra_press_count": {"type": "integer", "minimum": 0},
            "one_hand_only": {"type": "boolean"},
            "target_key_visibly_moves": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string"},
        },
        "required": [
            "observed_note", "press_time", "extra_press_count", "one_hand_only",
            "target_key_visibly_moves", "confidence", "explanation",
        ],
        "additionalProperties": False,
    }


def _call_vision_model(
    frames: list[tuple[float, str]], expectation: dict[str, Any]
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    expected_note = str(expectation["note"])
    expected_time = float(expectation["press_time"])
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": (
            "Analyze these chronologically ordered frames from a fixed-view piano video. "
            "Determine which named piano key the finger actually depresses, using the "
            "two-black-key/three-black-key pattern to locate pitches. Distinguish hovering "
            "or contact from visible downward key travel. Estimate press_time by "
            "interpolating between labeled frames. Count unintended additional key presses. "
            "If evidence is insufficient, use null and lower confidence; never assume the "
            f"requested note was played.\nRequested note: {expected_note}\n"
            f"Requested press time: {expected_time:.3f}s\nFrame timestamps follow."
        ),
    }]
    for timestamp, image_url in frames:
        content.append({"type": "input_text", "text": f"Frame at {timestamp:.3f}s"})
        content.append({"type": "input_image", "image_url": image_url, "detail": "high"})

    payload = {
        "model": os.environ.get("PIANOBENCH_VISION_MODEL", DEFAULT_MODEL),
        "store": False,
        "input": [{"role": "user", "content": content}],
        "text": {"format": {
            "type": "json_schema", "name": "piano_key_press_analysis", "strict": True,
            "schema": _response_schema(),
        }},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vision API returned HTTP {exc.code}: {error_body}") from exc

    output_text = body.get("output_text")
    if not output_text:
        for item in body.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    output_text = part.get("text")
                    break
            if output_text:
                break
    if not output_text:
        raise RuntimeError("Vision API response contained no output text")
    return json.loads(output_text)


def detect_key_presses(
    video_path: str, expectation: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return one structured VLM observation for the expected press window."""
    if expectation is None:
        raise ValueError("expectation is required for timestamp-aware frame sampling")
    frames = _sample_frames(video_path, float(expectation["press_time"]))
    return [_call_vision_model(frames, expectation)]


def _score_observation(
    observation: dict[str, Any], expected_note: str, expected_time: float, tolerance: float
) -> tuple[float, dict[str, float]]:
    note_score = 1.0 if observation.get("observed_note") == expected_note else 0.0
    press_time = observation.get("press_time")
    timing_score = 0.0 if press_time is None else max(
        0.0, 1.0 - abs(float(press_time) - expected_time) / max(tolerance, 1e-6)
    )
    clarity_score = 1.0 if observation.get("target_key_visibly_moves") else 0.0
    extras = max(0, int(observation.get("extra_press_count", 0)))
    extras_score = max(0.0, 1.0 - 0.5 * extras)
    hand_score = 1.0 if observation.get("one_hand_only") else 0.0
    components = {
        "note": note_score, "timing": timing_score, "visible_motion": clarity_score,
        "no_extra_presses": extras_score, "one_hand": hand_score,
    }
    score = (
        0.40 * note_score + 0.25 * timing_score + 0.15 * clarity_score
        + 0.10 * extras_score + 0.10 * hand_score
    )
    return score, components


def score_video_accuracy(
    video_path: str,
    expectation: dict[str, Any],
    *,
    timing_tolerance_seconds: float = 0.5,
    use_placeholder_demo: bool = True,
) -> VideoAccuracyResult:
    """Score note identity, timing, key motion, extra presses, and hand count."""
    if expectation.get("notes"):
        expected_notes = [str(note) for note in expectation["notes"]]
        return VideoAccuracyResult(
            score=0.0,
            expected_note=",".join(expected_notes),
            expected_press_time=float(expectation.get("press_time", 0.0)),
            details=(
                "Sequence visual scoring is not implemented yet; "
                f"expected ordered notes={expected_notes}. Audio sequence scoring still runs."
            ),
        )
    expected_note = str(expectation["note"])
    expected_t = float(expectation["press_time"])
    try:
        observation = detect_key_presses(video_path, expectation)[0]
        score, components = _score_observation(
            observation, expected_note, expected_t, timing_tolerance_seconds
        )
        details = (
            f"VLM observation={observation}; components={components}; "
            f"expected={expected_note} @ {expected_t:.3f}s"
        )
    except Exception as exc:
        if use_placeholder_demo:
            score = 0.65
            details = f"Placeholder demo score (0.65); visual analysis failed: {exc}"
        else:
            score = 0.0
            details = f"Visual analysis failed: {exc}"
    return VideoAccuracyResult(
        score=round(score, 3), expected_note=expected_note,
        expected_press_time=expected_t, details=details,
    )
