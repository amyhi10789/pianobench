"""Measure synchronization between detected audio notes and visual key presses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AlignmentResult:
    score: float
    audio_onset: float | None
    video_press_time: float | None
    lag_seconds: float | None
    details: str
    event_alignments: list[dict[str, Any]] = field(default_factory=list)


def lag_to_score(lag_seconds: float, soft_limit: float = 0.3) -> float:
    """Give full credit at zero lag and taper linearly to zero at soft_limit."""
    lag = abs(lag_seconds)
    if lag >= soft_limit:
        return 0.0
    return 1.0 - lag / max(soft_limit, 1e-6)


def _expected_notes(expectation: dict[str, Any]) -> list[str]:
    if expectation.get("notes"):
        return [str(note) for note in expectation["notes"]]
    if expectation.get("note") is not None:
        return [str(expectation["note"])]
    return []


def score_av_alignment(
    video_path: str,
    expectation: dict[str, Any],
    *,
    audio_events: list[dict[str, Any]] | None = None,
    video_events: list[dict[str, Any]] | None = None,
    soft_limit_seconds: float = 0.5,
    use_placeholder_demo: bool = True,
) -> AlignmentResult:
    """Compare already-detected ordered audio and video events note by note.

    Synchronization is intentionally independent of pitch correctness. Events
    are paired one-to-one by nearest time, so an extra/missed detector event
    does not shift every later pair. Unmatched expected events receive zero.
    """
    _ = video_path, use_placeholder_demo
    audio_events = list(audio_events or [])
    video_events = list(video_events or [])
    expected_notes = _expected_notes(expectation)

    if not expected_notes:
        return AlignmentResult(
            score=0.0,
            audio_onset=None,
            video_press_time=None,
            lag_seconds=None,
            details="No expected notes were provided for AV alignment.",
        )

    audio_events.sort(key=lambda event: float(event.get("onset", float("inf"))))
    video_events.sort(key=lambda event: float(event.get("press_time", float("inf"))))
    candidates: list[tuple[float, int, int]] = []
    for audio_index, audio in enumerate(audio_events):
        if audio.get("onset") is None:
            continue
        for video_index, video in enumerate(video_events):
            if video.get("press_time") is None:
                continue
            distance = abs(float(audio["onset"]) - float(video["press_time"]))
            candidates.append((distance, audio_index, video_index))
    candidates.sort()
    matched_audio: set[int] = set()
    matched_video: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _distance, audio_index, video_index in candidates:
        if audio_index in matched_audio or video_index in matched_video:
            continue
        matched_audio.add(audio_index)
        matched_video.add(video_index)
        pairs.append((audio_index, video_index))
    pairs.sort(key=lambda pair: float(video_events[pair[1]]["press_time"]))

    alignments: list[dict[str, Any]] = []
    event_scores: list[float] = []
    for index in range(len(expected_notes)):
        pair = pairs[index] if index < len(pairs) else None
        audio = audio_events[pair[0]] if pair else None
        video = video_events[pair[1]] if pair else None
        expected_note = expected_notes[index]
        audio_note = audio.get("note") if audio else None
        video_note = video.get("note") if video else None
        audio_t = audio.get("onset") if audio else None
        video_t = video.get("press_time") if video else None
        notes_match = (
            audio_note is not None
            and video_note is not None
            and str(audio_note) == str(video_note)
        )

        lag: float | None = None
        if audio_t is not None and video_t is not None:
            lag = float(audio_t) - float(video_t)
        event_score = lag_to_score(lag, soft_limit_seconds) if lag is not None else 0.0
        event_scores.append(event_score)
        alignments.append(
            {
                "index": index,
                "expected_note": expected_note,
                "audio_note": audio_note,
                "video_note": video_note,
                "audio_onset": round(float(audio_t), 3) if audio_t is not None else None,
                "video_press_time": round(float(video_t), 3) if video_t is not None else None,
                "lag_seconds": round(lag, 3) if lag is not None else None,
                "notes_match": notes_match,
                "pitch_match_required": False,
                "score": round(event_score, 3),
            }
        )

    score = sum(event_scores) / len(expected_notes)
    measured = [event for event in alignments if event["lag_seconds"] is not None]
    first = measured[0] if measured else None
    details = (
        f"Measured {len(measured)}/{len(expected_notes)} expected AV event pairs; "
        f"soft_limit={soft_limit_seconds:.3f}s; pitch-independent nearest-time matching; "
        f"events={alignments}"
    )
    return AlignmentResult(
        score=round(score, 3),
        audio_onset=first["audio_onset"] if first else None,
        video_press_time=first["video_press_time"] if first else None,
        lag_seconds=first["lag_seconds"] if first else None,
        details=details,
        event_alignments=alignments,
    )
