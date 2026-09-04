#!/usr/bin/env python3
"""Retry failed visual scoring while preserving valid stored audio scores.

This recovery command is intended for rows whose visual detector failed before
producing events. It reloads the API key from ``.env``, reruns visual analysis,
and recomputes AV alignment from fresh acoustic-onset measurements.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(EVAL_DIR))

from metrics.av_alignment import score_av_alignment  # noqa: E402
from metrics.video_accuracy import _ffmpeg_executable, score_video_accuracy  # noqa: E402

SAMPLE_RATE = 22_050
MIN_ONSET_SEPARATION = 0.12
RESULT_FIELDS = [
    "eval_id", "prompt_id", "level", "model_id", "model_name", "video",
    "expected_note", "audio_accuracy", "video_accuracy", "av_alignment", "overall",
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_video(raw: str) -> Path:
    path = Path(raw)
    return (path if path.is_file() else ROOT / "data" / path).resolve()


def parsed_detected_notes(details: str) -> list[str]:
    match = re.search(r"detected=(\[.*?\]); count=", details)
    return list(ast.literal_eval(match.group(1))) if match else []


def parsed_detected_chords(details: str) -> list[list[str]] | None:
    marker = "detected_chords="
    if marker not in details:
        return None
    return list(ast.literal_eval(details.split(marker, 1)[1]))


def expected_onset_count(row: dict) -> int:
    chords = parsed_detected_chords(row["audio_details"])
    return len(chords) if chords is not None else len(parsed_detected_notes(row["audio_details"]))


def detect_onsets(video_path: Path, max_time: float | None, count: int) -> list[float]:
    """Find the strongest separated spectral attacks without redoing pitch scoring."""
    if count <= 0:
        return []
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, timeout=30)
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    if max_time is not None:
        samples = samples[: int(max_time * SAMPLE_RATE)]
    frame_length, hop = 1024, 256
    if samples.size < frame_length:
        return [0.0] if samples.size else []
    starts = np.arange(0, samples.size - frame_length + 1, hop)
    window = np.hanning(frame_length).astype(np.float32)
    spectra = np.asarray([
        np.abs(np.fft.rfft(samples[start : start + frame_length] * window))
        for start in starts
    ])
    flux = np.maximum(np.diff(spectra, axis=0), 0.0).sum(axis=1)
    chosen: list[float] = []
    for index in np.argsort(flux)[::-1]:
        onset = float(starts[index + 1] / SAMPLE_RATE)
        if all(abs(onset - prior) >= MIN_ONSET_SEPARATION for prior in chosen):
            chosen.append(onset)
            if len(chosen) == count:
                break
    return sorted(chosen)


def rebuild_audio_events(row: dict, onsets: list[float]) -> list[dict]:
    chords = parsed_detected_chords(row["audio_details"])
    if chords is not None:
        events: list[dict] = []
        for onset, chord in zip(onsets, chords):
            events.extend({"note": note, "onset": onset} for note in chord)
        return events
    return [
        {"note": note, "onset": onset}
        for note, onset in zip(parsed_detected_notes(row["audio_details"]), onsets)
    ]


def save(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scores.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_visual_with_retries(video_path: Path, expectation: dict, eval_id: str):
    last_details = ""
    for attempt in range(1, 4):
        result = score_video_accuracy(
            str(video_path), expectation, use_placeholder_demo=False
        )
        last_details = result.details
        if not result.details.startswith(("Sequence visual analysis failed:", "Visual analysis failed:")):
            return result
        print(f"RETRY {eval_id} attempt={attempt} {result.details}", flush=True)
        if attempt < 3:
            time.sleep(2 * attempt)
    raise RuntimeError(f"Visual scoring failed for {eval_id}: {last_details}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing", type=Path, default=EVAL_DIR / "results" / "scores.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    old_by_id = {row["eval_id"]: row for row in load_json(args.existing)}
    expectations: dict[int, dict] = {}
    updated: list[dict] = []
    for item in manifest["evaluations"]:
        eval_id = item["id"]
        level = int(item["level"])
        print(f"START {eval_id}", flush=True)
        if level not in expectations:
            expectations[level] = load_json(
                ROOT / "data" / "expectations" / f"level{level}.json"
            )["prompts"]
        expectation = expectations[level][item["prompt_id"]]
        row = dict(old_by_id[eval_id])
        video_path = resolve_video(item["video_path"])
        max_time = expectation.get("duration_seconds")
        if max_time is None and expectation.get("release_time") is not None:
            max_time = float(expectation["release_time"]) + 2.0
        onsets = detect_onsets(
            video_path,
            float(max_time) if max_time is not None else None,
            expected_onset_count(row),
        )
        video = score_visual_with_retries(video_path, expectation, eval_id)
        alignment = score_av_alignment(
            str(video_path), expectation,
            audio_events=rebuild_audio_events(row, onsets),
            video_events=video.detected_events,
            use_placeholder_demo=False,
        )
        row.update({
            "video_accuracy": video.score,
            "video_details": video.details,
            "av_alignment": alignment.score,
            "alignment_details": alignment.details,
            "alignment_events": alignment.event_alignments,
            "overall": round(
                0.35 * row["audio_accuracy"] + 0.35 * video.score + 0.30 * alignment.score,
                3,
            ),
            "retry_provenance": (
                "Preserved valid stored audio score; reran visual scoring; recomputed AV "
                "alignment from freshly detected acoustic attacks."
            ),
        })
        updated.append(row)
        save(updated, args.out)
        print(
            f"DONE {eval_id} A={row['audio_accuracy']:.3f} V={video.score:.3f} "
            f"AV={alignment.score:.3f} O={row['overall']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
