"""
Audio accuracy metric (shared across prompt levels).

Scores how well the heard piano notes match the prompt expectation using
librosa for onset detection and probabilistic YIN (pYIN) pitch tracking.

Component scores (English labels, all in [0, 1]):
  - count_accuracy:    right number of notes (no extras / missing)
  - pitch_accuracy:    correct pitch classes / MIDI notes after alignment
  - order_accuracy:    detected sequence order matches expected order
  - timing_accuracy:   onset times near expected press times
  - duration_accuracy: note hold lengths near expected hold duration

The overall score is a weighted mean of these components.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import librosa
except ImportError:  # pragma: no cover - handled at score time
    librosa = None  # type: ignore


# --- defaults -----------------------------------------------------------------

DEFAULT_SR = 22050
DEFAULT_WEIGHTS = {
    "count_accuracy": 0.25,
    "pitch_accuracy": 0.35,
    "order_accuracy": 0.20,
    "timing_accuracy": 0.15,
    "duration_accuracy": 0.05,
}

# Onset / pitch heuristics tuned for short clean piano clips.
MIN_ONSET_SEP_SECONDS = 0.12
ONSET_ENERGY_PERCENTILE = 70.0
PITCH_WINDOW_SECONDS = 0.35
PITCH_HOP_SECONDS = 0.01
DEFAULT_TIMING_TOLERANCE = 0.5  # seconds for full timing credit taper
DEFAULT_DURATION_TOLERANCE = 0.35


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class DetectedNote:
    note: str
    midi: int
    onset: float
    duration: float
    confidence: float


@dataclass
class AudioAccuracyResult:
    score: float
    expected_note: str
    detected_notes: list[str]
    details: str
    expected_notes: list[str] = field(default_factory=list)
    count_accuracy: float = 0.0
    pitch_accuracy: float = 0.0
    order_accuracy: float = 0.0
    timing_accuracy: float = 0.0
    duration_accuracy: float = 0.0
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    detected_events: list[dict[str, Any]] = field(default_factory=list)


# --- note / expectation helpers ----------------------------------------------


def midi_to_note_name(midi: int | float) -> str:
    m = int(round(float(midi)))
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"


def note_name_to_midi(note: str) -> int:
    """Parse names like C4, Eb3, F#5 into MIDI numbers."""
    note = note.strip().replace("♯", "#").replace("♭", "b")
    if len(note) < 2:
        raise ValueError(f"Invalid note name: {note!r}")

    if note[1:2] in ("#", "b"):
        letter, accidental, octave_s = note[0].upper(), note[1], note[2:]
        name = letter + accidental
    else:
        letter, octave_s = note[0].upper(), note[1:]
        name = letter

    # Normalize flats to sharps used in NOTE_NAMES.
    flat_to_sharp = {
        "Db": "C#",
        "Eb": "D#",
        "Gb": "F#",
        "Ab": "G#",
        "Bb": "A#",
        "Cb": "B",
        "Fb": "E",
    }
    if name.endswith("b"):
        name = flat_to_sharp.get(name, name)

    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown pitch class in note: {note!r}")
    octave = int(octave_s)
    return NOTE_NAMES.index(name) + (octave + 1) * 12


def expected_note_sequence(expectation: dict[str, Any]) -> list[str]:
    """
    Build the expected note list from an expectation dict.

    Supports:
      - notes: ["C4", "E4", ...]
      - note: "C4" (singular-note prompts)
      - midis / midi as fallbacks
      - num_notes to pad/truncate when only one pitch is given
    """
    if "notes" in expectation and expectation["notes"]:
        notes = [str(n) for n in expectation["notes"]]
    elif "midis" in expectation and expectation["midis"]:
        notes = [midi_to_note_name(m) for m in expectation["midis"]]
    elif "note" in expectation and expectation["note"] is not None:
        notes = [str(expectation["note"])]
    elif "midi" in expectation and expectation["midi"] is not None:
        notes = [midi_to_note_name(expectation["midi"])]
    else:
        raise KeyError("Expectation must include note(s) or midi(s).")

    num = expectation.get("num_notes")
    if num is not None:
        n = int(num)
        if len(notes) == 1 and n > 1:
            notes = notes * n
        elif len(notes) > n:
            notes = notes[:n]
    return notes


def expected_onset_times(expectation: dict[str, Any], n: int) -> list[float | None]:
    if "press_times" in expectation and expectation["press_times"]:
        times = [float(t) for t in expectation["press_times"]]
    elif "press_time" in expectation and expectation["press_time"] is not None:
        times = [float(expectation["press_time"])]
    else:
        return [None] * n

    if len(times) == 1 and n > 1:
        # No schedule given for a sequence — timing component becomes N/A (score 1 later).
        return [times[0]] + [None] * (n - 1)
    if len(times) < n:
        times = times + [None] * (n - len(times))
    return times[:n]


def expected_hold_durations(expectation: dict[str, Any], n: int) -> list[float | None]:
    if "hold_seconds_list" in expectation and expectation["hold_seconds_list"]:
        holds = [float(h) for h in expectation["hold_seconds_list"]]
    elif "hold_seconds" in expectation and expectation["hold_seconds"] is not None:
        holds = [float(expectation["hold_seconds"])]
    else:
        return [None] * n

    if len(holds) == 1 and n > 1:
        holds = holds * n
    if len(holds) < n:
        holds = holds + [None] * (n - len(holds))
    return holds[:n]


# --- audio loading / detection -----------------------------------------------


def _extract_wav_with_ffmpeg(video_path: Path, wav_path: Path, sr: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_mono_audio(video_path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load mono audio from a video/audio file via librosa, with ffmpeg fallback."""
    if librosa is None:
        raise ImportError(
            "librosa is required for audio accuracy. "
            "Install with: pip install -r requirements.txt"
        )

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video/audio not found: {video_path}")

    try:
        y, file_sr = librosa.load(str(path), sr=sr, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio stream")
        return y.astype(np.float32), int(file_sr)
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "audio.wav"
            _extract_wav_with_ffmpeg(path, wav_path, sr)
            y, file_sr = librosa.load(str(wav_path), sr=sr, mono=True)
            if y.size == 0:
                raise ValueError(f"Could not extract audio from {video_path}")
            return y.astype(np.float32), int(file_sr)


def _frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        return np.array([float(np.sqrt(np.mean(y**2)))], dtype=np.float32)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(frames**2, axis=0)).astype(np.float32)


def detect_notes_from_audio(
    video_path: str,
    *,
    sr: int = DEFAULT_SR,
    min_onset_sep: float = MIN_ONSET_SEP_SECONDS,
    max_time: float | None = None,
    min_confidence: float = 0.25,
    midi_min: int = 36,
    midi_max: int = 96,
) -> list[DetectedNote]:
    """
    Detect ordered piano note events: onset time, pitch, and rough duration.
    """
    y, sr = load_mono_audio(video_path, sr=sr)
    duration = float(len(y) / sr)
    if duration <= 0:
        return []
    if max_time is not None:
        duration = min(duration, float(max_time))

    hop_length = 256
    frame_length = 1024
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="time",
        backtrack=True,
        delta=0.07,
        wait=max(1, int(min_onset_sep * sr / hop_length)),
    )

    rms = _frame_rms(y, frame_length=frame_length, hop_length=hop_length)
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    energy_gate = float(np.percentile(rms, ONSET_ENERGY_PERCENTILE)) if rms.size else 0.0

    # Keep onsets that sit on reasonably energetic frames (filters soft noise clicks).
    filtered: list[float] = []
    for t in onset_times:
        if t > duration + 0.05:
            continue
        idx = int(np.argmin(np.abs(rms_times - t))) if rms_times.size else 0
        if rms.size == 0 or rms[idx] >= energy_gate * 0.5:
            if not filtered or (t - filtered[-1]) >= min_onset_sep:
                filtered.append(float(t))

    if not filtered:
        # Fallback: peak of onset envelope if anything energetic exists.
        if onset_env.size and float(np.max(onset_env)) > 0:
            peak_frame = int(np.argmax(onset_env))
            peak_t = float(librosa.frames_to_time(peak_frame, sr=sr, hop_length=hop_length))
            if peak_t <= duration + 0.05:
                filtered = [peak_t]

    fmin = librosa.note_to_hz("A0")
    fmax = librosa.note_to_hz("C8")
    frame_length_pyin = 2048
    hop_pyin = max(1, int(PITCH_HOP_SECONDS * sr))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length_pyin,
        hop_length=hop_pyin,
    )
    times_f0 = librosa.times_like(f0, sr=sr, hop_length=hop_pyin)

    events: list[DetectedNote] = []
    for i, onset in enumerate(filtered):
        next_onset = filtered[i + 1] if i + 1 < len(filtered) else duration
        window_end = min(onset + PITCH_WINDOW_SECONDS, next_onset - 0.02, duration)
        if window_end <= onset + 0.03:
            window_end = min(onset + PITCH_WINDOW_SECONDS, duration)

        mask = (times_f0 >= onset) & (times_f0 <= window_end) & voiced_flag & np.isfinite(f0)
        if not np.any(mask):
            # Slightly wider rescue window.
            mask = (
                (times_f0 >= onset)
                & (times_f0 <= min(onset + 0.6, duration))
                & np.isfinite(f0)
            )
            if np.any(mask) and voiced_flag is not None:
                voiced_mask = mask & voiced_flag
                if np.any(voiced_mask):
                    mask = voiced_mask

        if not np.any(mask):
            continue

        freqs = f0[mask]
        probs = voiced_prob[mask] if voiced_prob is not None else np.ones_like(freqs)
        # Prefer higher-confidence frames.
        weights = np.clip(probs.astype(np.float64), 0.05, 1.0)
        midi_vals = librosa.hz_to_midi(freqs)
        midi = int(round(float(np.average(midi_vals, weights=weights))))
        conf = float(np.average(weights))

        # Duration: time until RMS drops below a fraction of local peak, or next onset.
        onset_idx = int(np.argmin(np.abs(rms_times - onset))) if rms_times.size else 0
        search_end_t = min(next_onset, onset + 2.0, duration)
        if rms.size:
            end_idx = int(np.argmin(np.abs(rms_times - search_end_t)))
            segment = rms[onset_idx : max(onset_idx + 1, end_idx + 1)]
            local_peak = float(np.max(segment)) if segment.size else 0.0
            thresh = max(local_peak * 0.25, energy_gate * 0.2)
            release_idx = onset_idx
            for j in range(onset_idx, min(len(rms), end_idx + 1)):
                release_idx = j
                if j > onset_idx + 2 and rms[j] < thresh:
                    break
            release_t = float(rms_times[release_idx]) if rms_times.size else onset
        else:
            release_t = min(next_onset, onset + 0.5)

        note_dur = max(0.05, min(release_t, next_onset) - onset)
        if conf < min_confidence or midi < midi_min or midi > midi_max:
            continue
        events.append(
            DetectedNote(
                note=midi_to_note_name(midi),
                midi=midi,
                onset=float(onset),
                duration=float(note_dur),
                confidence=conf,
            )
        )

    return events


# --- scoring ------------------------------------------------------------------


def _count_accuracy(n_det: int, n_exp: int) -> float:
    if n_exp == 0 and n_det == 0:
        return 1.0
    denom = max(n_exp, n_det, 1)
    return max(0.0, 1.0 - abs(n_det - n_exp) / denom)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, start=1):
        for j, y in enumerate(b, start=1):
            if x == y:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def _align_by_onset(
    expected_notes: list[str],
    expected_onsets: list[float | None],
    detected: list[DetectedNote],
) -> list[tuple[int | None, int | None]]:
    """
    Greedy one-to-one alignment: for each expected event, pick the closest
    unused detection in time (or left-to-right if no expected onset).
    Returns list of (expected_index, detected_index).
    """
    n_exp, n_det = len(expected_notes), len(detected)
    pairs: list[tuple[int | None, int | None]] = []
    used: set[int] = set()

    # If we have expected onsets for all, match by nearest time.
    if any(t is not None for t in expected_onsets) and n_det:
        for i, t_exp in enumerate(expected_onsets):
            if t_exp is None:
                continue
            best_j, best_dist = None, float("inf")
            for j, ev in enumerate(detected):
                if j in used:
                    continue
                dist = abs(ev.onset - t_exp)
                if dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is not None:
                used.add(best_j)
                pairs.append((i, best_j))
        for i in range(n_exp):
            if all(p[0] != i for p in pairs):
                pairs.append((i, None))
        for j in range(n_det):
            if j not in used:
                pairs.append((None, j))
        pairs.sort(key=lambda p: (p[0] is None, p[0] if p[0] is not None else 10**9, p[1] or 0))
        return pairs

    # Pure sequential zip for order-focused cases.
    for i in range(max(n_exp, n_det)):
        pairs.append((i if i < n_exp else None, i if i < n_det else None))
    return pairs


def _pitch_and_timing_from_alignment(
    expected_notes: list[str],
    expected_onsets: list[float | None],
    expected_holds: list[float | None],
    detected: list[DetectedNote],
    pairs: list[tuple[int | None, int | None]],
    *,
    timing_tolerance: float,
    duration_tolerance: float,
) -> tuple[float, float, float]:
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    if not expected_notes:
        return 1.0, 1.0, 1.0

    pitch_hits = 0
    timing_scores: list[float] = []
    duration_scores: list[float] = []

    for i, j in matched:
        exp_note = expected_notes[i]
        det = detected[j]
        if det.note == exp_note or det.midi == note_name_to_midi(exp_note):
            pitch_hits += 1

        t_exp = expected_onsets[i] if i < len(expected_onsets) else None
        if t_exp is not None:
            lag = abs(det.onset - t_exp)
            timing_scores.append(max(0.0, 1.0 - lag / max(timing_tolerance, 1e-6)))

        h_exp = expected_holds[i] if i < len(expected_holds) else None
        if h_exp is not None and h_exp > 0:
            err = abs(det.duration - h_exp)
            duration_scores.append(max(0.0, 1.0 - err / max(duration_tolerance, 1e-6)))

    # Unmatched expected notes count as pitch misses.
    pitch_acc = pitch_hits / max(len(expected_notes), 1)

    if not timing_scores:
        # No expected schedule to compare — do not punish.
        timing_acc = 1.0 if matched else 0.0
    else:
        # Also penalize missing expected timed notes.
        while len(timing_scores) < len([t for t in expected_onsets if t is not None]):
            timing_scores.append(0.0)
        timing_acc = float(np.mean(timing_scores)) if timing_scores else 0.0

    if not duration_scores:
        duration_acc = 1.0 if matched else 0.0
    else:
        while len(duration_scores) < len([h for h in expected_holds if h is not None]):
            duration_scores.append(0.0)
        duration_acc = float(np.mean(duration_scores)) if duration_scores else 0.0

    return pitch_acc, timing_acc, duration_acc


def _order_accuracy(expected_notes: list[str], detected_notes: list[str]) -> float:
    """
    Sequence-order score.

    - Singular expected note: trivial 1.0 if anything was detected (pitch is
      scored separately so a wrong pitch is not double-penalized here).
    - Sequences: longest common subsequence recall vs the expected stream,
      so extras hurt less than missing/reordered target notes.
    """
    if not expected_notes and not detected_notes:
        return 1.0
    if not expected_notes:
        return 0.0
    if not detected_notes:
        return 0.0
    if len(expected_notes) == 1:
        return 1.0
    lcs = _lcs_length(expected_notes, detected_notes)
    return lcs / len(expected_notes)


def combine_component_scores(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    total_w = sum(w.values()) or 1.0
    return float(sum(components[k] * w[k] for k in w) / total_w)


def score_audio_accuracy(
    video_path: str,
    expectation: dict[str, Any],
    *,
    use_placeholder_demo: bool = True,
    weights: dict[str, float] | None = None,
    timing_tolerance_seconds: float = DEFAULT_TIMING_TOLERANCE,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE,
) -> AudioAccuracyResult:
    """
    Compare detected audio notes to the prompt expectation.

    If detection dependencies fail and use_placeholder_demo is True, returns a
    soft demo score so the pipeline still runs.
    """
    expected_notes = expected_note_sequence(expectation)
    expected_label = expected_notes[0] if len(expected_notes) == 1 else ",".join(expected_notes)
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    max_time = None
    # Prefer explicit duration on the prompt, else a typical 8s clip.
    if "duration_seconds" in expectation:
        max_time = float(expectation["duration_seconds"])
    elif expectation.get("release_time") is not None:
        max_time = float(expectation["release_time"]) + 2.0

    try:
        detected_events = detect_notes_from_audio(video_path, max_time=max_time)
    except Exception as exc:
        if use_placeholder_demo:
            return AudioAccuracyResult(
                score=0.7,
                expected_note=expected_label,
                expected_notes=expected_notes,
                detected_notes=list(expected_notes),
                details=(
                    f"Placeholder demo score (0.7); detection failed: {exc}. "
                    "Install deps (librosa/soundfile/ffmpeg) for real scoring."
                ),
                count_accuracy=0.7,
                pitch_accuracy=0.7,
                order_accuracy=0.7,
                timing_accuracy=0.7,
                duration_accuracy=0.7,
                weights=w,
            )
        return AudioAccuracyResult(
            score=0.0,
            expected_note=expected_label,
            expected_notes=expected_notes,
            detected_notes=[],
            details=f"Audio detection failed: {exc}",
            weights=w,
        )

    detected_notes = [e.note for e in detected_events]
    n_exp, n_det = len(expected_notes), len(detected_events)
    onsets = expected_onset_times(expectation, n_exp)
    holds = expected_hold_durations(expectation, n_exp)
    pairs = _align_by_onset(expected_notes, onsets, detected_events)

    count_acc = _count_accuracy(n_det, n_exp)
    order_acc = _order_accuracy(expected_notes, detected_notes)
    pitch_acc, timing_acc, duration_acc = _pitch_and_timing_from_alignment(
        expected_notes,
        onsets,
        holds,
        detected_events,
        pairs,
        timing_tolerance=timing_tolerance_seconds,
        duration_tolerance=duration_tolerance_seconds,
    )

    # If nothing was detected at all, allow optional demo fallback.
    if n_det == 0 and use_placeholder_demo:
        return AudioAccuracyResult(
            score=0.7,
            expected_note=expected_label,
            expected_notes=expected_notes,
            detected_notes=[],
            details=(
                "Placeholder demo score (0.7); no onsets/pitches detected in audio. "
                "Check that the video has an audio track and piano energy."
            ),
            count_accuracy=0.7,
            pitch_accuracy=0.7,
            order_accuracy=0.7,
            timing_accuracy=0.7,
            duration_accuracy=0.7,
            weights=w,
            detected_events=[],
        )

    components = {
        "count_accuracy": count_acc,
        "pitch_accuracy": pitch_acc,
        "order_accuracy": order_acc,
        "timing_accuracy": timing_acc,
        "duration_accuracy": duration_acc,
    }
    overall = combine_component_scores(components, w)

    details = (
        f"expected={expected_notes}, detected={detected_notes}; "
        f"count={count_acc:.3f}, pitch={pitch_acc:.3f}, order={order_acc:.3f}, "
        f"timing={timing_acc:.3f}, duration={duration_acc:.3f}; "
        f"weights={w}"
    )

    return AudioAccuracyResult(
        score=round(overall, 3),
        expected_note=expected_label,
        expected_notes=expected_notes,
        detected_notes=detected_notes,
        details=details,
        count_accuracy=round(count_acc, 3),
        pitch_accuracy=round(pitch_acc, 3),
        order_accuracy=round(order_acc, 3),
        timing_accuracy=round(timing_acc, 3),
        duration_accuracy=round(duration_acc, 3),
        weights=w,
        detected_events=[asdict(e) for e in detected_events],
    )
