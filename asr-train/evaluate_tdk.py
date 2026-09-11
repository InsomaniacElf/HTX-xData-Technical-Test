"""Evaluate one frozen checkpoint on TDK, preserving base-model predictions.

Task 3b permits direct inference. Audio preprocessing matches asr_api.py.
The resume journal is bound to the model, input CSV and shard configuration.
"""
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asr"))
from ycsep_decode import audio_cache_name, row_key, TDK_CHANNEL
from cache_audio_shard import fetch, unpack_verified


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def baseline_identity(rows):
    # Derived comparison columns do not change the underlying evaluation inputs.
    derived = {"generated_text_ft", "asr_error_ft", "wer_ft", "cer_ft", "wer_base", "cer_base"}
    checksum = hashlib.sha256()
    for row in rows:
        checksum.update(json.dumps({k: v for k, v in row.items() if k not in derived},
                                   sort_keys=True, ensure_ascii=False).encode())
        checksum.update(b"\n")
    return checksum.hexdigest()


def load_base(path, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid shard configuration")
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not {"text", "generated_text", "asr_error", "channel"} <= set(fields or []):
            raise ValueError("A Task 2c baseline CSV with explicit error status is required")
        rows = list(reader)
    if not rows or any(r["channel"] != TDK_CHANNEL or r["asr_error"] for r in rows):
        raise ValueError("Baseline must contain successful TDK rows only")
    return fields, rows[index::count]


def prepare(row, folder, cache):
    import librosa
    import soundfile as sf
    name = audio_cache_name(row)
    try:
        mp3 = cache / name if cache else folder / name
        if cache:
            if not mp3.is_file():
                raise FileNotFoundError("Required baseline MP3 cache entry missing")
        else:
            payload, _, error = fetch(row, retries=2, source_mode="source-wav")
            if error:
                return None, error
            mp3.write_bytes(payload)
        samples, _ = librosa.load(mp3, sr=16000, mono=True)
        wav = folder / (row_key(row) + ".wav")
        sf.write(wav, samples, 16000)
        return wav, ""
    except Exception as exc:
        return None, type(exc).__name__


def evaluate(base, model, output, batch_size=8, workers=8, shard_index=0,
             num_shards=1, cache=None):
    import torch
    from nemo.collections.asr.models import ASRModel

    if min(batch_size, workers) < 1:
        raise ValueError("Batch size and workers must be positive")
    fields, rows = load_base(base, shard_index, num_shards)
    if not rows:
        raise ValueError("Empty evaluation shard")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    config = {"base_input_sha256": baseline_identity(rows), "model_sha256": digest(model),
              "shard_index": shard_index, "num_shards": num_shards,
              "audio": "verified_baseline_cache" if cache else "source_wav_reencoded_mp3",
              "checkpoint_selection": "Frozen before TDK evaluation; no TDK model selection"}
    config_path = root / "evaluation-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration mismatch")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    net = ASRModel.restore_from(str(model), map_location="cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="tdk-ft-cache-") as cache_folder, \
            sqlite3.connect(root / "evaluation.sqlite3") as db:
        cache_path = Path(cache_folder) if cache else None
        if cache:
            unpack_verified(cache, cache_path)
        db.execute("CREATE TABLE IF NOT EXISTS predictions (key TEXT PRIMARY KEY, text TEXT, error TEXT)")
        done = {r[0] for r in db.execute("SELECT key FROM predictions WHERE error=''")}
        pending = [r for r in rows if row_key(r) not in done]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset+batch_size]
                with tempfile.TemporaryDirectory(prefix="tdk-ft-") as directory:
                    prepared = list(pool.map(lambda r: prepare(r, Path(directory), cache_path), batch))
                    valid = [(row, path) for row, (path, error) in zip(batch, prepared) if not error]
                    for row, (_, error) in zip(batch, prepared):
                        if error:
                            db.execute("INSERT OR REPLACE INTO predictions VALUES (?, '', ?)", (row_key(row), error))
                    if valid:
                        with torch.inference_mode():
                            predictions = list(net.transcribe([str(p) for _, p in valid], batch_size=batch_size))
                        if len(predictions) != len(valid):
                            raise RuntimeError("Model output count mismatch")
                        for (row, _), prediction in zip(valid, predictions):
                            text = prediction.text if hasattr(prediction, "text") else str(prediction)
                            db.execute("INSERT OR REPLACE INTO predictions VALUES (?, ?, '')", (row_key(row), text))
                    db.commit()
                if offset % (batch_size * 25) == 0:
                    print(json.dumps({"processed_this_run": min(offset+batch_size, len(pending)),
                        "pending_at_start": len(pending), "elapsed_seconds": time.perf_counter()-started}), flush=True)
        destination = root / "TDK_subset_ft.csv"
        temporary = destination.with_suffix(".csv.tmp")
        extra = ["generated_text_ft", "asr_error_ft"]
        failures = 0
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields) + [f for f in extra if f not in fields])
            writer.writeheader()
            for row in rows:
                record = db.execute("SELECT text, error FROM predictions WHERE key=?", (row_key(row),)).fetchone()
                text, error = record or ("", "not_processed")
                failures += bool(error)
                writer.writerow(dict(row, generated_text_ft=text, asr_error_ft=error))
        os.replace(temporary, destination)
    result = dict(config, rows=len(rows), failed=failures, elapsed_seconds=time.perf_counter()-started,
                  source_csv_sha256=digest(base))
    (root / "evaluation-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{failures} evaluation rows failed; successful predictions are preserved")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "model", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cache", help="Task 2c verified CPU-stage archive folder")
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
