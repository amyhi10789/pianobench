import unittest
from unittest.mock import patch

from evaluation.metrics.audio_accuracy import DetectedNote, score_audio_accuracy


def note(name, midi, onset):
    return DetectedNote(name, midi, onset, 0.5, 0.9)


class AudioAccuracyChordTests(unittest.TestCase):
    @patch("evaluation.metrics.audio_accuracy.detect_chords_from_audio")
    def test_notes_inside_one_chord_are_unordered(self, detect):
        detect.return_value = [note("E4", 64, 1.02), note("C4", 60, 1.00)]
        result = score_audio_accuracy(
            "unused.wav",
            {"notes": ["C4", "E4"], "chord_events": [["C4", "E4"]]},
            use_placeholder_demo=False,
        )
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.pitch_accuracy, 1.0)
        self.assertEqual(result.order_accuracy, 1.0)

    @patch("evaluation.metrics.audio_accuracy.detect_chords_from_audio")
    def test_arpeggiated_notes_do_not_count_as_one_chord(self, detect):
        detect.return_value = [note("C4", 60, 1.0), note("E4", 64, 1.3)]
        result = score_audio_accuracy(
            "unused.wav",
            {"notes": ["C4", "E4"], "chord_events": [["C4", "E4"]]},
            use_placeholder_demo=False,
        )
        self.assertLess(result.pitch_accuracy, 1.0)
        self.assertEqual(result.order_accuracy, 0.0)
        self.assertIn("detected_chords=[['C4'], ['E4']]", result.details)

    @patch("evaluation.metrics.audio_accuracy.detect_notes_from_audio")
    def test_sequential_notes_still_use_monophonic_path(self, detect):
        detect.return_value = [note("C4", 60, 1.0), note("E4", 64, 2.0)]
        result = score_audio_accuracy(
            "unused.wav", {"notes": ["C4", "E4"]}, use_placeholder_demo=False
        )
        self.assertEqual(result.score, 1.0)
        detect.assert_called_once_with("unused.wav", max_time=None)


if __name__ == "__main__":
    unittest.main()
