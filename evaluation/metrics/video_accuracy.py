"""
Video accuracy metric.

Goal: check whether the visible key press matches the prompt
(correct key, timing, one hand, no extra presses).

This version is a stub with a clear place to plug in computer-vision
code later (e.g. key region tracking, hand detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VideoAccuracyResult:
    score: float  # 0.0 to 1.0
    expected_note: str
    expected_press_time: float
    details: str


def detect_key_presses(video_path: str) -> list[dict[str, Any]]:
    """
    Placeholder visual detector.

    TODO (advanced): for each frame, detect which piano key moves down and when.
    Return a list like: [{"note": "C4", "press_time": 3.05}, ...]
    """
    _ = video_path
    return []


def score_video_accuracy(
    video_path: str,
    expectation: dict[str, Any],
    *,
    timing_tolerance_seconds: float = 0.5,
    use_placeholder_demo: bool = True,
) -> VideoAccuracyResult:
    """
    Compare detected key presses to the prompt expectation.
    """
    expected_note = str(expectation["note"])
    expected_t = float(expectation["press_time"])
    presses = detect_key_presses(video_path)

    if presses:
        best = 0.0
        for press in presses:
            note_ok = press.get("note") == expected_note
            t = float(press.get("press_time", -999))
            timing_ok = abs(t - expected_t) <= timing_tolerance_seconds
            if note_ok and timing_ok:
                best = 1.0
                break
            if note_ok or timing_ok:
                best = max(best, 0.5)
        score = best
        details = f"Detected presses={presses}; expected {expected_note} @ {expected_t}s."
    elif use_placeholder_demo:
        score = 0.65
        details = (
            "Placeholder demo score (0.65). "
            "Replace detect_key_presses() with real visual key tracking."
        )
    else:
        score = 0.0
        details = "No presses detected and demo mode is off."

    return VideoAccuracyResult(
        score=round(score, 3),
        expected_note=expected_note,
        expected_press_time=expected_t,
        details=details,
    )
