#!/usr/bin/env python3
"""
Evaluate one or more generated videos against per-level prompt expectations.

Usage (from the project root):

    python evaluation/evaluate.py

Or evaluate a single video:

    python evaluation/evaluate.py \\
        --video data/videos/gemini/level1/piano_vid_new_1.mp4 \\
        --prompt-id 1A
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running this file directly without installing a package.
ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXPECTATIONS_DIR = DATA_DIR / "expectations"
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(EVAL_DIR))

from metrics.audio_accuracy import score_audio_accuracy  # noqa: E402
from metrics.av_alignment import score_av_alignment  # noqa: E402
from metrics.video_accuracy import score_video_accuracy  # noqa: E402


DEFAULT_WEIGHTS = {
    "audio_accuracy": 0.35,
    "video_accuracy": 0.35,
    "av_alignment": 0.30,
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def level_from_prompt_id(prompt_id: str) -> int:
    """Map prompt ids like 1A / 2B to integer level."""
    if not prompt_id or not prompt_id[0].isdigit():
        raise ValueError(f"Cannot infer level from prompt id: {prompt_id!r}")
    return int(prompt_id[0])


def expectations_path_for_level(level: int) -> Path:
    return EXPECTATIONS_DIR / f"level{level}.json"


def load_expectations_for_level(level: int) -> dict:
    path = expectations_path_for_level(level)
    if not path.is_file():
        raise FileNotFoundError(f"Expectations not found for level {level}: {path}")
    return load_json(path)


def overall_score(audio: float, video: float, align: float, weights: dict[str, float]) -> float:
    return (
        weights["audio_accuracy"] * audio
        + weights["video_accuracy"] * video
        + weights["av_alignment"] * align
    )


def evaluate_one(
    video_path: Path,
    expectation: dict,
    *,
    level: int,
    use_placeholder_demo: bool = True,
) -> dict:
    audio = score_audio_accuracy(
        str(video_path), expectation, use_placeholder_demo=use_placeholder_demo
    )
    video = score_video_accuracy(
        str(video_path), expectation, use_placeholder_demo=use_placeholder_demo
    )
    align = score_av_alignment(
        str(video_path), expectation, use_placeholder_demo=use_placeholder_demo
    )
    total = overall_score(audio.score, video.score, align.score, DEFAULT_WEIGHTS)
    return {
        "video": str(video_path),
        "level": level,
        "expected_note": expectation.get("note") or expectation.get("notes"),
        "audio_accuracy": audio.score,
        "video_accuracy": video.score,
        "av_alignment": align.score,
        "overall": round(total, 3),
        "audio_details": audio.details,
        "audio_components": {
            "count_accuracy": getattr(audio, "count_accuracy", None),
            "pitch_accuracy": getattr(audio, "pitch_accuracy", None),
            "order_accuracy": getattr(audio, "order_accuracy", None),
            "timing_accuracy": getattr(audio, "timing_accuracy", None),
            "duration_accuracy": getattr(audio, "duration_accuracy", None),
        },
        "video_details": video.details,
        "alignment_details": align.details,
    }


def resolve_video_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_file():
        return path
    candidate = DATA_DIR / raw
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Video not found: {raw}")


def run_from_manifest(
    manifest_path: Path,
    *,
    use_placeholder_demo: bool,
) -> list[dict]:
    manifest = load_json(manifest_path)
    model_names = {m["id"]: m["display_name"] for m in manifest.get("models", [])}
    expectations_cache: dict[int, dict] = {}
    rows: list[dict] = []

    for item in manifest.get("evaluations", []):
        prompt_id = item["prompt_id"]
        level = int(item.get("level") or level_from_prompt_id(prompt_id))
        if level not in expectations_cache:
            expectations_cache[level] = load_expectations_for_level(level)
        expectation = expectations_cache[level]["prompts"][prompt_id]
        video_path = resolve_video_path(item["video_path"])
        result = evaluate_one(
            video_path,
            expectation,
            level=level,
            use_placeholder_demo=use_placeholder_demo,
        )
        result["eval_id"] = item["id"]
        result["prompt_id"] = prompt_id
        result["model_id"] = item["model_id"]
        result["model_name"] = model_names.get(item["model_id"], item["model_id"])
        rows.append(result)
    return rows


def save_results(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scores.json"
    csv_path = out_dir / "scores.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "eval_id",
        "prompt_id",
        "level",
        "model_id",
        "model_name",
        "video",
        "expected_note",
        "audio_accuracy",
        "video_accuracy",
        "av_alignment",
        "overall",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No evaluation rows.")
        return
    print()
    print(
        f"{'prompt':<8}{'lvl':<5}{'model':<22}"
        f"{'audio':>8}{'video':>8}{'align':>8}{'overall':>10}"
    )
    print("-" * 69)
    for row in rows:
        model = row.get("model_name") or row.get("model_id") or "-"
        prompt = row.get("prompt_id") or "-"
        level = row.get("level", "-")
        print(
            f"{prompt:<8}{level:<5}{model:<22}"
            f"{row['audio_accuracy']:>8.3f}"
            f"{row['video_accuracy']:>8.3f}"
            f"{row['av_alignment']:>8.3f}"
            f"{row['overall']:>10.3f}"
        )
    print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score generated piano videos vs prompts.")
    p.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIR / "manifest.json",
        help="JSON list of (prompt, model, video) evaluations",
    )
    p.add_argument(
        "--expectations-dir",
        type=Path,
        default=EXPECTATIONS_DIR,
        help="Directory of per-level expectations (levelN.json)",
    )
    p.add_argument("--video", type=Path, help="Score a single video instead of the manifest")
    p.add_argument("--prompt-id", default="1A", help="Prompt id when using --video")
    p.add_argument(
        "--level",
        type=int,
        help="Level when using --video (default: inferred from prompt id)",
    )
    p.add_argument(
        "--no-demo",
        action="store_true",
        help="Disable placeholder demo scores (real detectors only)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=EVAL_DIR / "results",
        help="Folder for scores.json and scores.csv",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    global EXPECTATIONS_DIR
    EXPECTATIONS_DIR = args.expectations_dir
    use_demo = not args.no_demo

    if args.video:
        level = args.level or level_from_prompt_id(args.prompt_id)
        expectations = load_expectations_for_level(level)
        expectation = expectations["prompts"][args.prompt_id]
        video_path = resolve_video_path(str(args.video))
        row = evaluate_one(
            video_path, expectation, level=level, use_placeholder_demo=use_demo
        )
        row["eval_id"] = "single"
        row["prompt_id"] = args.prompt_id
        row["model_id"] = "manual"
        row["model_name"] = "manual"
        rows = [row]
    else:
        rows = run_from_manifest(args.manifest, use_placeholder_demo=use_demo)

    print_table(rows)
    save_results(rows, args.out)


if __name__ == "__main__":
    main()
