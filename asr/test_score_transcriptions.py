import csv
import tempfile
import unittest
from pathlib import Path
from score_transcriptions import normalize, score


class MetricTests(unittest.TestCase):
    def test_unicode_and_apostrophes(self):
        self.assertEqual(normalize("Wah, don't! \u4f60\u597d"), "wah don't \u4f60\u597d")

    def test_pooled_rates_and_failure_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input.csv", Path(folder) / "scored.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["text", "generated_text", "asr_error"])
                writer.writerows([["one two three", "one two", ""], ["four", "five six", ""],
                                 ["failed", "", "TimeoutError"], ["", "inserted", ""]])
            result = score(source, output)
            self.assertEqual(result["reference_words"], 4)
            self.assertEqual(result["word_edits"], 4)
            self.assertEqual(result["corpus_wer"], 1)
            self.assertEqual(result["coverage"], .75)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[2]["wer_base"], "")
            self.assertEqual(rows[3]["wer_base"], "")


if __name__ == "__main__":
    unittest.main()
