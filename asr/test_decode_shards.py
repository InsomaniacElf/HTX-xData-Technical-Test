"""Verify sharding covers every held-out row exactly once, including repeats."""
import csv
import tempfile
import unittest
from pathlib import Path
from ycsep_decode import load_tdk_rows, TDK_CHANNEL


class ShardTests(unittest.TestCase):
    def test_partition_is_complete_and_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.csv"
            fields = ["channel", "file", "speaker", "start_time", "end_time", "text", "audio"]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for i in range(17):
                    writer.writerow(dict(zip(fields, [TDK_CHANNEL, str(i), "S", "0", "1", "hi", "https://example.test/audio"])))
                    writer.writerow(dict(zip(fields, ["other", str(i), "S", "0", "1", "hi", "https://example.test/audio"])))
            shards = [load_tdk_rows(str(source), None, i, 4) for i in range(4)]
            ids = [r["file"] for shard in shards for r in shard]
            self.assertEqual(len(ids), 17)
            self.assertEqual(set(ids), {str(i) for i in range(17)})
            self.assertTrue(all(r["channel"] == TDK_CHANNEL for s in shards for r in s))
            self.assertEqual([len(s) for s in shards], [5, 4, 4, 4])

    def test_invalid_partition_rejected(self):
        with self.assertRaises(ValueError):
            load_tdk_rows("unused", None, 4, 4)


if __name__ == "__main__":
    unittest.main()
