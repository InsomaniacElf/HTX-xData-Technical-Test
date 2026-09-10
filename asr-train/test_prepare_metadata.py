"""Checks for leakage and overlap accounting, independent of the real corpus."""
import unittest
from prepare_metadata import features, overlap_stats, split_for, video_key


class MetadataTests(unittest.TestCase):
    def test_video_id_ignores_title_variants(self):
        self.assertEqual(video_key("20200101--abc123--title.TextGrid"), "abc123")
        self.assertEqual(video_key("20200101--abc123--other title.TextGrid"), "abc123")

    def test_split_deterministic(self):
        self.assertEqual(split_for("abc123"), split_for("abc123"))

    def test_nested_overlap_is_not_double_counted(self):
        total, overlap, different = overlap_stats([(0, 10, "A"), (2, 8, "B"), (3, 4, "C")])
        self.assertEqual((total, overlap, different), (10, 6, 6))

    def test_touching_intervals_do_not_overlap(self):
        self.assertEqual(overlap_stats([(0, 2, "A"), (2, 3, "B")]), (3, 0, 0))

    def test_singlish_and_short_speech_retained_as_flags(self):
        result = features({"start_time": "1", "end_time": "1.4", "text": "wah lah"})
        self.assertTrue(result[-1])
        self.assertIn("short", result[-2])
        self.assertNotIn("empty_or_nonlexical_text", result[-2])

    def test_bad_timestamp(self):
        self.assertIn("invalid_timestamp", features({"start_time": "nan", "end_time": "2", "text": "hi"})[-2])


if __name__ == "__main__":
    unittest.main()
