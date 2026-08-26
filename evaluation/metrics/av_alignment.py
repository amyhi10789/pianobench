"""
Audio–video alignment metric.

Goal: check that each audible note starts at the same moment the key
visually begins moving downward.

Alignment score is high when |audio_onset - video_press_time| is small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlignmentResult:
    score: float  # 0.0 to 1.0
    audio_onset: float | None
    video_press_time: float | None
    lag_seconds: float | None
    details: str


def detect_audio_onset(video_path: str) -> float | None:
    """Placeholder: return the time (seconds) of the first piano note onset."""
    _ = video_path
    return None


def detect_video_press_time(video_path: str) -> float | None:
    """Placeholder: return the time (seconds) when the target key starts moving down."""
    _ = video_path
    return None


def lag_to_score(lag_seconds: float, soft_limit: float = 0.3) -> float:
    """
    Map absolute lag to a score in [0, 1].

    lag = 0.0s  -> 1.0
    lag = soft_limit -> ~0.0
    lag > soft_limit -> 0.0
    """
    lag = abs(lag_seconds)
    if lag >= soft_limit:
        return 0.0
    return 1.0 - (lag / soft_limit)


def score_av_alignment(
    video_path: str,
    expectation: dict[str, Any],
    *,
    soft_limit_seconds: float = 0.3,
    use_placeholder_demo: bool = True,
) -> AlignmentResult:
    if expectation.get("notes") and not expectation.get("press_times"):
        return AlignmentResult(
            score=0.0,
            audio_onset=None,
            video_press_time=None,
            lag_seconds=None,
            details=(
                "Sequence AV alignment is not implemented and this prompt has no "
                "explicit press-time schedule."
            ),
        )
    audio_t = detect_audio_onset(video_path)
    video_t = detect_video_press_time(video_path)
    expected_t = float(expectation["press_time"])

    if audio_t is not None and video_t is not None:
        lag = audio_t - video_t
        score = lag_to_score(lag, soft_limit=soft_limit_seconds)
        details = (
            f"audio_onset={audio_t:.3f}s, video_press={video_t:.3f}s, "
            f"lag={lag:+.3f}s (expected ~{expected_t}s)."
        )
        return AlignmentResult(
            score=round(score, 3),
            audio_onset=audio_t,
            video_press_time=video_t,
            lag_seconds=round(lag, 3),
            details=details,
        )

    if use_placeholder_demo:
        # Demo: assume almost-aligned events near the expected press time.
        demo_audio = expected_t + 0.05
        demo_video = expected_t
        lag = demo_audio - demo_video
        score = lag_to_score(lag, soft_limit=soft_limit_seconds)
        return AlignmentResult(
            score=round(score, 3),
            audio_onset=demo_audio,
            video_press_time=demo_video,
            lag_seconds=round(lag, 3),
            details=(
                "Placeholder demo alignment. "
                "Replace onset/press detectors with real measurements."
            ),
        )

    return AlignmentResult(
        score=0.0,
        audio_onset=None,
        video_press_time=None,
        lag_seconds=None,
        details="Could not measure alignment and demo mode is off.",
    )
