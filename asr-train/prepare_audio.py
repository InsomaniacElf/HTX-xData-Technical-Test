"""Download frozen candidates and create validated 16 kHz mono NeMo manifests.

Uses CPU/ffmpeg only. All failed selections remain recorded, and are not replaced
using validation results. Re-running validates and reuses existing WAV files.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode

import requests

_local = threading.local()


def session():
    if not hasattr(_local, "http"):
        _local.http = requests.Session()
    return _local.http


def validate(path, expected):
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Expected PCM16 mono 16 kHz")
        duration = audio.getnframes() / 16000
        if not .2 <= duration <= 20.2 or abs(duration-expected) > max(.15, expected * .05):
            raise ValueError("Decoded/metadata duration mismatch")
        if not any(audio.readframes(audio.getnframes())):
            raise ValueError("Empty or zero waveform")
    return duration


def fetch(row, root, ffmpeg):
    if row["channel"] == "The_Daily_Ketchup_Podcast":
        raise ValueError("TDK cannot enter training or validation")
    key = hashlib.sha256(json.dumps({k: row[k] for k in
        ["channel", "file", "start_time", "end_time", "audio"]}, sort_keys=True).encode()).hexdigest()
    relative = f"audio/{key}.wav"
    destination = root / relative
    if destination.exists():
        try:
            duration = validate(destination, row["duration"])
            return dict(row, audio_filepath=relative, duration=duration), None
        except (ValueError, wave.Error, EOFError):
            pass
    original = "https://a3s.fi/swift/v1/YCSEP_v2/" + quote(row["file"].removesuffix(".TextGrid") + ".wav")
    fallback = "https://ycsep.corpora.li/clip?" + urlencode({"url": original,
        "start": f"{float(row['start_time']):.3f}", "end": f"{float(row['end_time']):.3f}", "fmt": "mp3"})
    errors = []
    for attempt in range(2):
        for url in (row["audio"], fallback):
            try:
                with tempfile.TemporaryDirectory(dir=root) as folder:
                    raw, wav = Path(folder) / "input.mp3", Path(folder) / "output.wav"
                    with session().get(url, timeout=(10, 45), stream=True) as response:
                        response.raise_for_status()
                        with raw.open("wb") as handle:
                            size = 0
                            for block in response.iter_content(65536):
                                size += len(block)
                                if size > 64 * 1024 * 1024:
                                    raise ValueError("Audio exceeds 64 MiB")
                                handle.write(block)
                    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-threads", "1", "-y",
                        "-i", str(raw), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
                    duration = validate(wav, row["duration"])
                    wav.replace(destination)
                return dict(row, audio_filepath=relative, duration=duration,
                            resolved_audio_source="csv_audio" if url == row["audio"] else "ycsep_v2_clip"), None
            except (requests.RequestException, subprocess.SubprocessError, ValueError, OSError, wave.Error, EOFError) as exc:
                errors.append(type(exc).__name__)
        time.sleep(attempt + 1)
    return row, {"source_row": row["source_row"], "errors": errors}


def prepare(selection, output, split, workers, limit=0):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be installed and on PATH")
    root = Path(output)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    with Path(selection).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if limit:
        rows = rows[:limit]
    if not rows or any(r["split"] != split or r["channel"] == "The_Daily_Ketchup_Podcast" for r in rows):
        raise ValueError("Invalid or contaminated selection")
    started, accepted, failures = time.perf_counter(), [], []
    with (root / f"{split}-progress.jsonl").open("a", encoding="utf-8") as progress, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, r, root, ffmpeg) for r in rows]
        for count, future in enumerate(as_completed(futures), 1):
            row, error = future.result()
            if error:
                failures.append(error)
            else:
                accepted.append(row)
            progress.write(json.dumps({"source_row": row["source_row"], "error": error}) + "\n")
            if count % 100 == 0:
                progress.flush()
                print(json.dumps({"split": split, "processed": count, "total": len(rows),
                    "failures": len(failures), "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
    with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(accepted, key=lambda r: r["source_row"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"split": split, "selected": len(rows), "accepted": len(accepted),
        "hours": sum(r["duration"] for r in accepted)/3600, "failures": failures,
        "elapsed_seconds": time.perf_counter()-started}
    (root / f"{split}-result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} selected clips failed validation; inspect and resolve before training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error("workers must be positive and limit nonnegative")
    prepare(**vars(args))
