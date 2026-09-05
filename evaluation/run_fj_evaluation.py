#!/usr/bin/env python3
"""Evaluate only prompts F-J without touching the canonical score files.

The runner discovers the requested non-Gemini videos, checkpoints after every
completed clip, and retries detector failures with demo/placeholder scoring
disabled. It is intentionally separate from ``evaluate.py`` so an interrupted
batch can resume without repeating successful API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT = EVAL_DIR / "results" / "f-j-2026-09-04"
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(EVAL_DIR))

from evaluate import DEFAULT_WEIGHTS, overall_score  # noqa: E402
from metrics.audio_accuracy import score_audio_accuracy  # noqa: E402
from metrics.av_alignment import score_av_alignment  # noqa: E402
from metrics.video_accuracy import _ffmpeg_executable, score_video_accuracy  # noqa: E402


MODEL_NAMES = {
    "cosmos3": "Cosmos 3",
    "cosmos3-super": "Cosmos 3 Super",
    "joyai-echo": "JoyAI Echo",
    "minimax-h3": "MiniMax H3",
    "dreamx-creator": "DreamX Creator",
    "ltx-2.5": "LTX-2.5",
}
MODEL_ORDER = tuple(MODEL_NAMES)
LETTERS = "fghij"


PROMPT_SPECS: dict[str, dict[str, Any]] = {
    "1F": {"note": "F4", "midi": 65, "press_time": 3.0, "hold_seconds": 0.5, "release_time": 3.5, "num_notes": 1},
    "1G": {"note": "B4", "midi": 71, "press_time": 3.0, "hold_seconds": 0.5, "release_time": 3.5, "num_notes": 1},
    "1H": {"note": "C5", "midi": 72, "press_time": 3.0, "hold_seconds": 0.5, "release_time": 3.5, "num_notes": 1},
    "1I": {"note": "A3", "midi": 57, "press_time": 3.0, "hold_seconds": 0.5, "release_time": 3.5, "num_notes": 1},
    "1J": {"note": "E3", "midi": 52, "press_time": 3.0, "hold_seconds": 0.5, "release_time": 3.5, "num_notes": 1},
    "2F": {"notes": ["E4", "F4", "G4", "A4"], "midis": [64, 65, 67, 69], "num_notes": 4},
    "2G": {"notes": ["D4", "F4", "A4"], "midis": [62, 65, 69], "num_notes": 3},
    "2H": {"notes": ["G4", "A4", "B4", "C5"], "midis": [67, 69, 71, 72], "num_notes": 4},
    "2I": {"notes": ["C4", "E4", "F4", "G4"], "midis": [60, 64, 65, 67], "num_notes": 4},
    "2J": {"notes": ["F4", "A4", "B4"], "midis": [65, 69, 71], "num_notes": 3},
    "3F": {"notes": ["A4", "G4", "F4", "E4"], "midis": [69, 67, 65, 64], "num_notes": 4},
    "3G": {"notes": ["C5", "B4", "A4", "G4"], "midis": [72, 71, 69, 67], "num_notes": 4},
    "3H": {"notes": ["A4", "G4", "E4"], "midis": [69, 67, 64], "num_notes": 3},
    "3I": {"notes": ["G4", "F4", "D4", "C4"], "midis": [67, 65, 62, 60], "num_notes": 4},
    "3J": {"notes": ["B4", "A4", "F4"], "midis": [71, 69, 65], "num_notes": 3},
    "4F": {"notes": ["D4", "D4", "D4"], "midis": [62, 62, 62], "num_notes": 3},
    "4G": {"notes": ["A4", "A4", "A4", "A4"], "midis": [69, 69, 69, 69], "num_notes": 4},
    "4H": {"notes": ["G4", "G4", "E4", "G4"], "midis": [67, 67, 64, 67], "num_notes": 4},
    "4I": {"notes": ["F4", "F4", "F4"], "midis": [65, 65, 65], "num_notes": 3},
    "4J": {"notes": ["C4", "E4", "C4", "E4"], "midis": [60, 64, 60, 64], "num_notes": 4},
    "5F": {"notes": ["F4", "A4"], "midis": [65, 69], "num_notes": 2, "chord_events": [["F4", "A4"]], "num_chord_events": 1},
    "5G": {"notes": ["G4", "B4", "D5"], "midis": [67, 71, 74], "num_notes": 3, "chord_events": [["G4", "B4", "D5"]], "num_chord_events": 1},
    "5H": {"notes": ["E4", "G4", "A4", "C5"], "midis": [64, 67, 69, 72], "num_notes": 4, "chord_events": [["E4", "G4"], ["A4", "C5"]], "num_chord_events": 2},
    "5I": {"notes": ["E4", "G4", "B4"], "midis": [64, 67, 71], "num_notes": 3, "chord_events": [["E4", "G4", "B4"]], "num_chord_events": 1},
    "5J": {"notes": ["C4", "G4", "E4", "A4"], "midis": [60, 67, 64, 69], "num_notes": 4, "chord_events": [["C4", "G4"], ["E4", "A4"]], "num_chord_events": 2},
}

RESULT_FIELDS = [
    "eval_id", "prompt_id", "level", "model_id", "model_name", "video",
    "expected_note", "audio_accuracy", "video_accuracy", "av_alignment", "overall",
]


def expectation(prompt_id: str) -> dict[str, Any]:
    value = dict(PROMPT_SPECS[prompt_id])
    value["prompt_file"] = f"prompts/level{prompt_id[0]}/prompt_{prompt_id.lower()}.txt"
    return value


def requested_jobs(selected_models: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    jobs: list[dict] = []
    missing: list[dict] = []
    for model_id in MODEL_ORDER:
        if selected_models is not None and model_id not in selected_models:
            continue
        for level in range(1, 6):
            for letter in LETTERS:
                prompt_id = f"{level}{letter.upper()}"
                path = DATA_DIR / "videos" / model_id / f"level{level}" / f"{level}{letter}.mp4"
                separate_audio: Path | None = None
                if not path.is_file() and model_id == "dreamx-creator" and prompt_id == "3I":
                    split_video = path.with_name("3i.video.mp4")
                    split_audio = path.with_name("3i.wav")
                    if split_video.is_file() and split_audio.is_file():
                        path, separate_audio = split_video, split_audio
                item = {
                    "id": f"eval_{prompt_id.lower()}_{model_id}",
                    "prompt_id": prompt_id,
                    "level": level,
                    "model_id": model_id,
                    "model_name": MODEL_NAMES[model_id],
                    "video_path": str(path.resolve()),
                }
                if path.is_file():
                    if separate_audio is not None:
                        item["separate_audio_path"] = str(separate_audio.resolve())
                    jobs.append(item)
                else:
                    missing.append(item)
    return jobs, missing


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temp.replace(path)


def save_rows(rows_by_id: dict[str, dict[str, Any]], out_dir: Path) -> None:
    order = {model: index for index, model in enumerate(MODEL_ORDER)}
    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (order[row["model_id"]], row["level"], row["prompt_id"]),
    )
    atomic_json(out_dir / "scores.json", rows)
    csv_path = out_dir / "scores.csv"
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)


def write_metadata(out_dir: Path, jobs: list[dict], missing: list[dict]) -> None:
    metadata = out_dir / "metadata"
    manifest_jobs = [{k: v for k, v in job.items() if k != "separate_audio_path"} for job in jobs]
    atomic_json(metadata / "manifest_available.json", {
        "project": "pianobench",
        "scope": "non-Gemini prompts F-J only",
        "models": [{"id": model, "display_name": MODEL_NAMES[model]} for model in MODEL_ORDER],
        "evaluations": manifest_jobs,
    })
    atomic_json(metadata / "missing_videos.json", missing)
    for level in range(1, 6):
        prompts = {
            prompt_id: expectation(prompt_id)
            for prompt_id in PROMPT_SPECS
            if prompt_id.startswith(str(level))
        }
        atomic_json(metadata / "expectations" / f"level{level}.json", {
            "level": level,
            "scope": "F-J only; derived directly from data/prompts",
            "prompts": prompts,
        })


def prepared_video(job: dict, out_dir: Path) -> Path:
    video = Path(job["video_path"])
    audio_raw = job.get("separate_audio_path")
    if audio_raw is None:
        return video
    target = out_dir / "prepared_media" / job["model_id"] / f"level{job['level']}" / f"{job['prompt_id'].lower()}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(), "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio_raw), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", str(target),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return target


def retry_component(
    label: str,
    eval_id: str,
    function: Callable[[], Any],
    failure_prefixes: tuple[str, ...],
    max_attempts: int,
) -> tuple[Any, int]:
    last_details = ""
    for attempt in range(1, max_attempts + 1):
        result = function()
        last_details = str(result.details)
        placeholder = "placeholder demo score" in last_details.lower()
        failed = last_details.startswith(failure_prefixes)
        if not placeholder and not failed:
            return result, attempt
        print(f"RETRY {eval_id} component={label} attempt={attempt}: {last_details}", flush=True)
        if attempt < max_attempts:
            time.sleep(5 * attempt)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts: {last_details}")


def evaluate_job(job: dict, out_dir: Path, max_attempts: int) -> dict[str, Any]:
    eval_id = job["id"]
    video_path = prepared_video(job, out_dir)
    spec = expectation(job["prompt_id"])
    audio, audio_attempts = retry_component(
        "audio", eval_id,
        lambda: score_audio_accuracy(str(video_path), spec, use_placeholder_demo=False),
        ("Audio detection failed:",), max_attempts,
    )
    video, video_attempts = retry_component(
        "video", eval_id,
        lambda: score_video_accuracy(str(video_path), spec, use_placeholder_demo=False),
        ("Sequence visual analysis failed:", "Visual analysis failed:"), max_attempts,
    )
    alignment = score_av_alignment(
        str(video_path), spec,
        audio_events=audio.detected_events,
        video_events=video.detected_events,
        use_placeholder_demo=False,
    )
    score = overall_score(audio.score, video.score, alignment.score, DEFAULT_WEIGHTS)
    return {
        "video": str(video_path),
        "level": job["level"],
        "expected_note": spec.get("note") or spec.get("notes"),
        "audio_accuracy": audio.score,
        "video_accuracy": video.score,
        "av_alignment": alignment.score,
        "overall": round(score, 3),
        "audio_details": audio.details,
        "audio_components": {
            "count_accuracy": getattr(audio, "count_accuracy", None),
            "pitch_accuracy": getattr(audio, "pitch_accuracy", None),
            "order_accuracy": getattr(audio, "order_accuracy", None),
            "timing_accuracy": getattr(audio, "timing_accuracy", None),
            "duration_accuracy": getattr(audio, "duration_accuracy", None),
        },
        "video_details": video.details,
        "alignment_details": alignment.details,
        "alignment_events": alignment.event_alignments,
        "eval_id": eval_id,
        "prompt_id": job["prompt_id"],
        "model_id": job["model_id"],
        "model_name": job["model_name"],
        "fallback_used": False,
        "attempts": {"audio": audio_attempts, "video": video_attempts},
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["eval_id"]: row for row in json.load(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Limit pending jobs for a smoke test")
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER)
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1:
        parser.error("--workers and --max-attempts must be positive")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.models) if args.models else None
    jobs, missing = requested_jobs(selected)
    write_metadata(out_dir, jobs, missing)
    rows_by_id = load_existing(out_dir / "scores.json")
    failures_path = out_dir / "failures.json"
    failures = load_existing(failures_path)
    pending = [job for job in jobs if job["id"] not in rows_by_id]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"REQUESTED={len(jobs) + len(missing)} AVAILABLE={len(jobs)} "
        f"MISSING={len(missing)} COMPLETE={len(rows_by_id)} PENDING={len(pending)} "
        f"WORKERS={args.workers} FALLBACKS=DISABLED",
        flush=True,
    )
    # Audio pitch tracking is CPU-bound, so use processes rather than threads.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_jobs = {
            pool.submit(evaluate_job, job, out_dir, args.max_attempts): job
            for job in pending
        }
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            eval_id = job["id"]
            try:
                row = future.result()
            except Exception as exc:
                failures[eval_id] = {"eval_id": eval_id, **job, "error": str(exc)}
                atomic_json(failures_path, list(failures.values()))
                print(f"FAILED {eval_id}: {exc}", flush=True)
                continue
            rows_by_id[eval_id] = row
            failures.pop(eval_id, None)
            save_rows(rows_by_id, out_dir)
            atomic_json(failures_path, list(failures.values()))
            print(
                f"DONE {eval_id} A={row['audio_accuracy']:.3f} "
                f"V={row['video_accuracy']:.3f} AV={row['av_alignment']:.3f} "
                f"O={row['overall']:.3f} ({len(rows_by_id)}/{len(jobs)})",
                flush=True,
            )

    print(
        f"FINISHED COMPLETE={len(rows_by_id)}/{len(jobs)} FAILURES={len(failures)} "
        f"MISSING={len(missing)} OUT={out_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
