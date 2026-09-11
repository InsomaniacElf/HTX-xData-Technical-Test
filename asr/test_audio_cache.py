import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cache_audio_shard import unpack_verified
from ycsep_decode import download_audio


class AudioCacheTests(unittest.TestCase):
    def fixture(self, root, name="abc.mp3", payload=b"test", digest=None):
        record = {"cache_name": "abc.mp3", "error": "", "bytes": 4,
                  "sha256": digest or hashlib.sha256(b"test").hexdigest()}
        (root / "sources.jsonl").write_text(json.dumps(record) + "\n")
        with tarfile.open(root / "audio.tar", "w") as archive:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    def test_verified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(unpack_verified(root, root / "out"), 1)
            self.assertEqual((root / "out/abc.mp3").read_bytes(), b"test")

    def test_rejects_path_and_corruption(self):
        for name, payload in [("../abc.mp3", b"test"), ("abc.mp3", b"evil")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, name, payload)
                with self.assertRaises(ValueError):
                    unpack_verified(root, root / "out")

    def test_cache_only_never_downloads(self):
        row = {"file": "example.TextGrid", "audio": "https://example.invalid/a.mp3",
               "start_time": "0", "end_time": "1"}
        with patch("ycsep_decode.http_session") as network:
            with self.assertRaises(ValueError):
                download_audio(row, io.BytesIO(), None, cache_only=True)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
