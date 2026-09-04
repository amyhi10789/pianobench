import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics.video_accuracy import (
    _probe_video_duration,
    _sample_sequence_frames,
)


class VideoDurationTests(unittest.TestCase):
    @patch("evaluation.metrics.video_accuracy.subprocess.run")
    @patch("evaluation.metrics.video_accuracy._ffmpeg_executable", return_value="ffmpeg")
    def test_probe_parses_ffmpeg_container_duration(self, _executable, run):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Duration: 00:00:05.18, start: 0.000000, bitrate: 964 kb/s",
        )

        self.assertAlmostEqual(_probe_video_duration(Path("short.mp4")), 5.18)

    @patch("evaluation.metrics.video_accuracy._sample_frames_at_times")
    @patch("evaluation.metrics.video_accuracy._probe_video_duration", return_value=5.18)
    def test_sequence_sampling_stops_before_short_video_ends(self, _probe, sample):
        sample.return_value = [(5.0, "frame")]

        result = _sample_sequence_frames("short.mp4", duration_seconds=8.0)

        timestamps = sample.call_args.args[1]
        self.assertEqual(result, [(5.0, "frame")])
        self.assertEqual(timestamps[0], 0.5)
        self.assertEqual(timestamps[-1], 5.0)
        self.assertNotIn(5.25, timestamps)

    @patch("evaluation.metrics.video_accuracy._sample_frames_at_times")
    @patch("evaluation.metrics.video_accuracy._probe_video_duration", return_value=10.0)
    def test_sequence_sampling_keeps_requested_duration_cap(self, _probe, sample):
        sample.return_value = []

        _sample_sequence_frames("long.mp4", duration_seconds=8.0)

        timestamps = sample.call_args.args[1]
        self.assertEqual(timestamps[-1], 7.75)


if __name__ == "__main__":
    unittest.main()
