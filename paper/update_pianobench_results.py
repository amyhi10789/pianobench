#!/usr/bin/env python3
"""Regenerate the PianoBench results paper from canonical evaluation scores.

The script intentionally refuses to write the paper unless the score artifact
contains one non-fallback result for every prompt/model pair (75 rows total).
Run it from any directory; repository-relative defaults are resolved from this
file's location.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = REPO_ROOT / "evaluation" / "results" / "scores.json"
DEFAULT_TEX = REPO_ROOT / "paper" / "pianobench_results.tex"

MODEL_ORDER = ("gemini", "cosmos3", "minimax-h3")
MODEL_NAMES = {
    "gemini": "Gemini",
    "cosmos3": "Cosmos 3",
    "minimax-h3": "MiniMax H3",
}
MODEL_LABELS = {
    "gemini": "gemini-results",
    "cosmos3": "cosmos-results",
    "minimax-h3": "minimax-results",
}
METRICS = ("audio_accuracy", "video_accuracy", "av_alignment", "overall")
METRIC_NAMES = {
    "audio_accuracy": "audio accuracy",
    "video_accuracy": "video accuracy",
    "av_alignment": "audio--visual alignment",
    "overall": "overall score",
}

PROMPT_META = {
    "1A": ("Single note", "C4"),
    "1B": ("Single note", "E4"),
    "1C": ("Single note", "G4"),
    "1D": ("Single note", "D4"),
    "1E": ("Single note", "A4"),
    "2A": ("Ascending", "C4--D4--E4--F4"),
    "2B": ("Ascending", "C4--E4--G4"),
    "2C": ("Ascending", "F4--G4--A4--B4"),
    "2D": ("Ascending", "D4--E4--G4--A4"),
    "2E": ("Ascending", "C4--D4--F4--A4"),
    "3A": ("Descending", "F4--E4--D4--C4"),
    "3B": ("Descending", "G4--E4--C4"),
    "3C": ("Descending", "B4--A4--G4--F4"),
    "3D": ("Descending", "A4--F4--E4--C4"),
    "3E": ("Descending", "B4--G4--D4"),
    "4A": ("Repeated", "C4--C4--C4"),
    "4B": ("Repeated", "E4--E4--E4--E4"),
    "4C": ("Repeated", "C4--C4--E4--C4"),
    "4D": ("Repeated", "G4--G4--G4"),
    "4E": ("Repeated", "E4--G4--G4--E4"),
    "5A": ("Chord", "C4+E4"),
    "5B": ("Chord", "C4+E4+G4"),
    "5C": ("Chord", "(C4+E4), then (F4+A4)"),
    "5D": ("Chord", "D4+A4"),
    "5E": ("Chord", "(C4+E4+G4), then (D4+F4+A4)"),
}
PROMPT_ORDER = tuple(PROMPT_META)


def _score(value: Any, *, row_id: str, metric: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{row_id}: {metric} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{row_id}: {metric} is outside [0, 1]: {result}")
    return result


def load_complete_scores(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")

    grouped: dict[str, dict[str, dict[str, Any]]] = {
        model_id: {} for model_id in MODEL_ORDER
    }
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        model_id = row.get("model_id")
        if model_id not in grouped:
            raise ValueError(f"row {index} has unexpected model_id {model_id!r}")
        prompt_id = str(row.get("prompt_id", "")).upper()
        row_id = str(row.get("eval_id", f"row {index}"))
        if prompt_id not in PROMPT_META:
            raise ValueError(f"{row_id}: unexpected prompt_id {prompt_id!r}")
        if prompt_id in grouped[model_id]:
            raise ValueError(f"duplicate result for {model_id}/{prompt_id}")
        for metric in METRICS:
            _score(row.get(metric), row_id=row_id, metric=metric)

        # The old evaluator failure announced itself in video_details.  Refuse
        # to turn such sentinel values into benchmark claims.
        details = str(row.get("video_details", "")).casefold()
        if "placeholder demo score" in details or "evaluator fallback" in details:
            raise ValueError(f"{row_id}: visual result is still an evaluator fallback")
        grouped[model_id][prompt_id] = row

    expected = set(PROMPT_ORDER)
    complete: dict[str, list[dict[str, Any]]] = {}
    for model_id in MODEL_ORDER:
        actual = set(grouped[model_id])
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"{model_id} is incomplete: missing={missing or 'none'}, "
                f"extra={extra or 'none'}"
            )
        complete[model_id] = [grouped[model_id][prompt] for prompt in PROMPT_ORDER]

    if len(raw) != len(MODEL_ORDER) * len(PROMPT_ORDER):
        raise ValueError(f"expected exactly 75 rows, found {len(raw)}")
    return complete


def means(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    materialized = list(rows)
    return {
        metric: fmean(float(row[metric]) for row in materialized)
        for metric in METRICS
    }


def family_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    families: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        family, _ = PROMPT_META[str(row["prompt_id"]).upper()]
        families[family].append(float(row["overall"]))
    return {family: fmean(values) for family, values in families.items()}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def model_list(aggregates: dict[str, dict[str, float]]) -> str:
    return ", ".join(
        f"{MODEL_NAMES[model_id]} ({fmt(aggregates[model_id]['overall'])})"
        for model_id in MODEL_ORDER
    )


def metric_leader_clause(aggregates: dict[str, dict[str, float]]) -> str:
    clauses = []
    for metric in ("audio_accuracy", "video_accuracy", "av_alignment"):
        leader = max(MODEL_ORDER, key=lambda model_id: aggregates[model_id][metric])
        clauses.append(
            f"{MODEL_NAMES[leader]} leads {METRIC_NAMES[metric]} "
            f"({fmt(aggregates[leader][metric])})"
        )
    return "; ".join(clauses)


def baseline_prose(
    scores: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, float]],
) -> str:
    ranking = sorted(
        MODEL_ORDER, key=lambda model_id: aggregates[model_id]["overall"], reverse=True
    )
    ranking_text = ", ".join(
        f"{MODEL_NAMES[model_id]} ({fmt(aggregates[model_id]['overall'])})"
        for model_id in ranking
    )
    component_leaders = {
        metric: max(MODEL_ORDER, key=lambda model_id: aggregates[model_id][metric])
        for metric in ("audio_accuracy", "video_accuracy", "av_alignment")
    }
    unique_leaders = set(component_leaders.values())
    if len(unique_leaders) == 1:
        leader = next(iter(unique_leaders))
        leader_sentence = (
            f"{MODEL_NAMES[leader]} also leads all three diagnostic components: "
            f"audio accuracy ({fmt(aggregates[leader]['audio_accuracy'])}), "
            f"video accuracy ({fmt(aggregates[leader]['video_accuracy'])}), and "
            f"audio--visual alignment ({fmt(aggregates[leader]['av_alignment'])})."
        )
    else:
        leader_sentence = metric_leader_clause(aggregates) + "."

    ranges = []
    for model_id in MODEL_ORDER:
        by_family = family_means(scores[model_id])
        best = max(by_family, key=by_family.get)
        worst = min(by_family, key=by_family.get)
        ranges.append(
            f"{MODEL_NAMES[model_id]} ranges from {fmt(by_family[worst])} on "
            f"{worst.lower()} prompts to {fmt(by_family[best])} on "
            f"{best.lower()} prompts"
        )
    range_sentence = "; ".join(ranges) + "."
    return (
        f"Across all 25 prompts, the overall ranking is {ranking_text}. "
        f"{leader_sentence} The family-level results are likewise diagnostic: "
        f"{range_sentence} These differences show why a benchmark for physically "
        "grounded audiovisual generation should expose component and task-family "
        "metrics rather than only a single holistic quality score."
    )


def summary_table(aggregates: dict[str, dict[str, float]]) -> str:
    rows = []
    for model_id in MODEL_ORDER:
        aggregate = aggregates[model_id]
        rows.append(
            f"    {MODEL_NAMES[model_id]} & 25 & "
            f"{fmt(aggregate['audio_accuracy'])} & "
            f"{fmt(aggregate['video_accuracy'])} & "
            f"{fmt(aggregate['av_alignment'])} & "
            f"{fmt(aggregate['overall'])} \\\\"
        )
    return "\n".join(rows)


def per_video_table(
    model_id: str,
    rows: list[dict[str, Any]],
    aggregate: dict[str, float],
) -> str:
    body: list[str] = []
    prior_level: int | None = None
    for row in rows:
        prompt_id = str(row["prompt_id"]).upper()
        level = int(prompt_id[0])
        if prior_level is not None and level != prior_level:
            body.append(r"\addlinespace")
        family, target = PROMPT_META[prompt_id]
        body.append(
            f"{prompt_id} & {family} & {target} & "
            f"{fmt(float(row['audio_accuracy']))} & "
            f"{fmt(float(row['video_accuracy']))} & "
            f"{fmt(float(row['av_alignment']))} & "
            f"{fmt(float(row['overall']))} \\\\"
        )
        prior_level = level

    body_text = "\n".join(body)
    model_name = MODEL_NAMES[model_id]
    label = MODEL_LABELS[model_id]
    return rf"""\subsection{{{model_name}}}

\begin{{longtable}}{{@{{}}llp{{3.0cm}}rrrr@{{}}}}
\caption{{Per-video PianoBench scores for {model_name}. A, V, and AV denote audio accuracy, video accuracy, and audio--visual alignment.}}
\label{{tab:{label}}}\\
\toprule
Prompt & Family & Target event(s) & A & V & AV & Overall \\
\midrule
\endfirsthead

\multicolumn{{7}}{{c}}{{\tablename\ \thetable{{}} continued}} \\
\toprule
Prompt & Family & Target event(s) & A & V & AV & Overall \\
\midrule
\endhead

\midrule
\multicolumn{{7}}{{r}}{{Continued on next page}} \\
\endfoot

\bottomrule
\endlastfoot

\small
{body_text}
\midrule
\multicolumn{{3}}{{r}}{{\textit{{Mean ($n=25$)}}}} & {fmt(aggregate['audio_accuracy'])} & {fmt(aggregate['video_accuracy'])} & {fmt(aggregate['av_alignment'])} & {fmt(aggregate['overall'])} \\
\end{{longtable}}"""


def render(scores: dict[str, list[dict[str, Any]]]) -> str:
    aggregates = {model_id: means(scores[model_id]) for model_id in MODEL_ORDER}
    overall_scores = model_list(aggregates)
    leaders = metric_leader_clause(aggregates)
    results_prose = baseline_prose(scores, aggregates)
    summary_rows = summary_table(aggregates)
    appendices = "\n\n".join(
        per_video_table(model_id, scores[model_id], aggregates[model_id])
        for model_id in MODEL_ORDER
    )

    return rf"""\documentclass{{article}}

% NeurIPS 2026 style file (submission / anonymized track).
% Get neurips_2026.sty from https://neurips.cc and place it in this
% same directory before compiling. See note at the end of this file.
\usepackage{{neurips_2026}}
% For the camera-ready version, switch to:
%  \usepackage[main,final]{{neurips_2026}}
% For an arXiv preprint, switch to:
%  \usepackage[preprint]{{neurips_2026}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{amsfonts}}
\usepackage{{nicefrac}}
\usepackage{{microtype}}
\usepackage{{xcolor}}
\usepackage{{longtable}}

\title{{PianoBench: A Benchmark for Audio-Visual Alignment in Text-to-Video Generation}}

\author{{%
  Author Name \\
  \texttt{{author@example.edu}}
}}

\begin{{document}}

\maketitle

\begin{{abstract}}
Text-to-video systems are commonly judged by visual plausibility, yet a convincing performance may still violate the discrete actions, acoustic content, and cross-modal timing specified by a prompt. We introduce PianoBench, a benchmark for measuring instruction following in generated piano-performance videos. PianoBench contains 25 controlled prompts organized into five task families---single notes, ascending sequences, descending sequences, repeated notes, and chords---with machine-readable expectations for pitch, event count, order, simultaneity, timing, and physical key motion. The baseline suite contains 75 videos from Gemini, Cosmos 3, and MiniMax H3, with 25 videos per model. Our evaluation decomposes performance into audio accuracy, video accuracy, and audio--visual alignment, then combines them with weights of 0.35, 0.35, and 0.30. Across the three 25-video baselines, the mean overall scores are {overall_scores}. The component results reveal distinct strengths: {leaders}. PianoBench therefore contributes a reproducible, event-level protocol that separates looking correct, sounding correct, and being synchronized, enabling diagnostic comparison beyond holistic video-quality judgments.
\end{{abstract}}

\section{{Benchmark Scope}}

PianoBench evaluates whether a generated video executes an explicitly requested musical action rather than merely depicting a plausible pianist. Its 25 prompts form five equally sized families: isolated notes with prescribed onset and hold duration; ascending note sequences; descending note sequences; repeated-note patterns; and simultaneous or successive chords. All prompts request one visible right hand, a stable overhead keyboard view, isolated acoustic-piano audio, no extra notes, and synchronization between visible key travel and sound onset. Each of three generation systems contributes one video per prompt, yielding 75 benchmark instances.

The benchmark is designed to expose three separable capabilities: symbolic musical correctness, physically legible visual execution, and temporal agreement between modalities. The prompt files, expectation records, model registry, videos, evaluation code, and result artifacts are maintained together in the repository so that every reported score can be traced to a concrete prompt--video pair.

\section{{Evaluation Protocol}}

All metrics lie in $[0,1]$, with higher values indicating better agreement with the prompt. Audio accuracy combines event-count, pitch, order, onset-time, and duration components. Monophonic prompts use onset detection and pYIN pitch estimation; chord prompts use a multipitch analysis path that preserves simultaneous note groups. Video accuracy uses a vision--language model to analyze timestamped frames and score note identity, event count and order, visible key movement, unintended presses, and hand-count constraints. Audio--visual alignment pairs detected acoustic onsets with detected visual press times and linearly reduces credit to zero at a lag of 0.5 seconds. The aggregate score is
\begin{{equation}}
S_{{\mathrm{{overall}}}} =
0.35S_{{\mathrm{{audio}}}} +
0.35S_{{\mathrm{{video}}}} +
0.30S_{{\mathrm{{AV}}}}.
\end{{equation}}

The baseline reported here uses the checked-in outputs of \texttt{{evaluation/evaluate.py}} and the vision configuration \texttt{{gpt-5.4-mini}}. For the MiniMax H3 Level 2--5 recovery run, already-valid audio scores were retained, visual scoring was rerun after duration-aware frame sampling was introduced, and AV alignment was recomputed from freshly detected acoustic attacks. Table~\ref{{tab:model-summary}} reports macro averages over all 25 prompts per model; Appendix~\ref{{app:per-video}} preserves every per-video score.

\section{{Baseline Results}}

\begin{{table}}[htbp]
  \caption{{Model-level macro averages over all 25 benchmark prompts per model.}}
  \label{{tab:model-summary}}
  \centering
  \begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
    \toprule
    Model & $n$ & Audio & Video & AV align. & Overall \\
    \midrule
{summary_rows}
    \bottomrule
  \end{{tabular}}
\end{{table}}

{results_prose}

\section{{Limitations and Intended Use}}

PianoBench is a compact diagnostic benchmark, not a comprehensive measure of video-generation quality. It covers one instrument, a constrained camera configuration, white-key pitches in the middle register, three model baselines, and one generated video per prompt--model pair. Audio scoring relies on heuristic signal processing, while visual scoring relies on sampled frames and a vision--language judge; both require calibration against human annotations. Future benchmark releases should add multiple generations per prompt, diverse keyboards and viewpoints, human inter-rater measurements, uncertainty estimates, and versioned evaluator outputs.

The benchmark should be used diagnostically: audio, video, and alignment scores identify different failure mechanisms, while the overall score provides a convenience summary under declared weights. Results should not be interpreted as a general-purpose ranking of the underlying generation systems beyond this prompt suite and evaluator version.

\section{{Reproducibility}}

The benchmark manifest is stored in \texttt{{data/manifest.json}}; prompt expectations are stored under \texttt{{data/expectations}}; and the complete detailed and tabular outputs are stored in \texttt{{evaluation/results/scores.json}} and \texttt{{evaluation/results/scores.csv}}. Running \texttt{{python evaluation/evaluate.py}} from the repository root regenerates the results when the required audio dependencies, FFmpeg, and vision API credentials are available. The targeted recovery procedure is recorded in \texttt{{evaluation/retry\_visual\_scores.py}}. Running \texttt{{python paper/update\_pianobench\_results.py}} then synchronizes every reported table value and aggregate with the canonical JSON artifact.

\begin{{ack}}
Do not include this section in the anonymized submission, only in the final paper.
\end{{ack}}

\section*{{References}}

\medskip
{{
\small
% Add references here in your chosen consistent citation style, e.g.:
% [1] Author, A. (2026) Title of work. \textit{{Venue}}.
}}

\appendix

\section{{Per-Video Baseline Scores}}
\label{{app:per-video}}

Tables~\ref{{tab:gemini-results}}--\ref{{tab:minimax-results}} report all 75 per-video baseline scores, with 25 prompt results for each model.

{appendices}

\end{{document}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help=f"canonical score JSON (default: {DEFAULT_SCORES})",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=DEFAULT_TEX,
        help=f"LaTeX output (default: {DEFAULT_TEX})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and render in memory without changing the LaTeX file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_complete_scores(args.scores.resolve())
    rendered = render(scores)
    if args.check:
        print("Validated 75 complete rows; LaTeX render succeeded (no file written).")
        return
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.tex.with_suffix(args.tex.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(args.tex)
    print(f"Updated {args.tex} from 75 complete results in {args.scores}.")


if __name__ == "__main__":
    main()
