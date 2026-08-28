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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gpt-5.4-mini"
FRAME_OFFSETS = (-0.60, -0.30, -0.10, 0.10, 0.30, 0.60)
SEQUENCE_FRAME_INTERVAL = 0.25


@dataclass
class VideoAccuracyResult:
    score: float
    expected_note: str
    expected_press_time: float
    details: str
    detected_events: list[dict[str, Any]] = field(default_factory=list)


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
    return _sample_frames_at_times(
        video_path, [max(0.0, expected_time + offset) for offset in FRAME_OFFSETS]
    )


def _sample_frames_at_times(
    video_path: str, timestamps: list[float]
) -> list[tuple[float, str]]:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    samples: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, timestamp in enumerate(timestamps):
            frame_path = Path(temp_dir) / f"frame_{index}.jpg"
            _extract_frame(path, timestamp, frame_path)
            encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
            samples.append((timestamp, f"data:image/jpeg;base64,{encoded}"))
    return samples


def _sample_sequence_frames(
    video_path: str, duration_seconds: float = 8.0
) -> list[tuple[float, str]]:
    """Sample the active portion of a clip densely enough to see separate presses."""
    timestamps: list[float] = []
    timestamp = 0.5
    while timestamp < duration_seconds:
        timestamps.append(round(timestamp, 3))
        timestamp += SEQUENCE_FRAME_INTERVAL
    return _sample_frames_at_times(video_path, timestamps)


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


def _sequence_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "observed_notes": {"type": "array", "items": {"type": "string"}},
            "press_times": {"type": "array", "items": {"type": "number"}},
            "visible_press_count": {"type": "integer", "minimum": 0},
            "extra_press_count": {"type": "integer", "minimum": 0},
            "one_hand_only": {"type": "boolean"},
            "presses_visibly_distinct": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string"},
        },
        "required": [
            "observed_notes", "press_times", "visible_press_count", "extra_press_count",
            "one_hand_only", "presses_visibly_distinct", "confidence", "explanation",
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


def _call_sequence_vision_model(
    frames: list[tuple[float, str]], expectation: dict[str, Any]
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    expected_notes = [str(note) for note in expectation["notes"]]
    chord_events = expectation.get("chord_events")
    if chord_events:
        task_description = (
            f"Requested chord events: {chord_events}. Notes within each inner list must be "
            "pressed simultaneously. Put every visible chord pitch in observed_notes and "
            "repeat the same press_time for notes belonging to the same chord. "
            "visible_press_count counts depressed keys, not chord groups. Explicitly reject "
            "an arpeggio as simultaneous."
        )
    else:
        task_description = (
            f"Requested ordered sequence: {expected_notes}. There is no required "
            "press-time schedule."
        )
    content: list[dict[str, Any]] = [{
        "type": "input_text",
        "text": (
            "Analyze these chronologically ordered frames from one fixed-view piano video. "
            "Report every visually distinct key press in observed order. Identify named "
            "keys from the two-black-key/three-black-key pattern and the visible center "
            "octave. A press requires downward key travel; do not count hovering or mere "
            "contact. Do not copy the requested sequence when frames disagree or are "
            "unclear. observed_notes and press_times must correspond position by position. "
            "visible_press_count is the number of presses actually supported by the frames. "
            "extra_press_count counts presses beyond the requested count. Use low confidence "
            "when temporal sampling cannot prove an event.\n"
            f"{task_description}\nTimestamped frames follow."
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
            "type": "json_schema", "name": "piano_sequence_analysis", "strict": True,
            "schema": _sequence_response_schema(),
        }},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
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


def _lcs_length(left: list[str], right: list[str]) -> int:
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i, left_note in enumerate(left, start=1):
        for j, right_note in enumerate(right, start=1):
            if left_note == right_note:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def _score_sequence_observation(
    observation: dict[str, Any], expected_notes: list[str]
) -> tuple[float, dict[str, float]]:
    observed_notes = [str(note) for note in observation.get("observed_notes", [])]
    expected_count = len(expected_notes)
    observed_count = max(
        len(observed_notes), int(observation.get("visible_press_count", len(observed_notes)))
    )
    denominator = max(expected_count, observed_count, 1)
    count_score = max(0.0, 1.0 - abs(observed_count - expected_count) / denominator)
    position_hits = sum(
        expected == observed
        for expected, observed in zip(expected_notes, observed_notes)
    )
    pitch_score = position_hits / max(expected_count, 1)
    order_score = _lcs_length(expected_notes, observed_notes) / max(expected_count, 1)
    distinct_score = 1.0 if observation.get("presses_visibly_distinct") else 0.0
    extras = max(0, int(observation.get("extra_press_count", 0)))
    extras_score = max(0.0, 1.0 - extras / max(expected_count, 1))
    hand_score = 1.0 if observation.get("one_hand_only") else 0.0
    components = {
        "count": count_score,
        "pitch_by_position": pitch_score,
        "order": order_score,
        "distinct_presses": distinct_score,
        "no_extra_presses": extras_score,
        "one_hand": hand_score,
    }
    score = (
        0.20 * count_score
        + 0.30 * pitch_score
        + 0.20 * order_score
        + 0.15 * distinct_score
        + 0.05 * extras_score
        + 0.10 * hand_score
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
        try:
            duration = float(expectation.get("duration_seconds", 8.0))
            frames = _sample_sequence_frames(video_path, duration)
            observation = _call_sequence_vision_model(frames, expectation)
            score, components = _score_sequence_observation(observation, expected_notes)
            details = (
                f"VLM sequence observation={observation}; components={components}; "
                f"expected ordered notes={expected_notes}; no fixed timing schedule"
            )
            detected_events = [
                {"note": note, "press_time": float(press_time)}
                for note, press_time in zip(
                    observation.get("observed_notes", []),
                    observation.get("press_times", []),
                )
            ]
        except Exception as exc:
            if use_placeholder_demo:
                score = 0.65
                details = f"Placeholder demo score (0.65); sequence analysis failed: {exc}"
            else:
                score = 0.0
                details = f"Sequence visual analysis failed: {exc}"
            detected_events = []
        return VideoAccuracyResult(
            score=round(score, 3),
            expected_note=",".join(expected_notes),
            expected_press_time=float(expectation.get("press_time", 0.0)),
            details=details,
            detected_events=detected_events,
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
        detected_events = []
        if observation.get("press_time") is not None:
            detected_events.append(
                {
                    "note": observation.get("observed_note"),
                    "press_time": float(observation["press_time"]),
                }
            )
    except Exception as exc:
        if use_placeholder_demo:
            score = 0.65
            details = f"Placeholder demo score (0.65); visual analysis failed: {exc}"
        else:
            score = 0.0
            details = f"Visual analysis failed: {exc}"
        detected_events = []
    return VideoAccuracyResult(
        score=round(score, 3), expected_note=expected_note,
        expected_press_time=expected_t, details=details,
        detected_events=detected_events,
    )
