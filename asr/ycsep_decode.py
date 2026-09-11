"""Decode The Daily Ketchup YCSEP clips through the local ASR API.

Task 2c requires all rows from The Daily Ketchup Podcast in YCSEP_static.csv to
be transcribed into TDK_subset.csv. The script is resumable because the full
subset contains more than 200k short clips and public audio endpoints may fail
intermittently.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import quote, urlencode

import requests


DEFAULT_CSV_PATH = r"C:\Users\Sneha\Downloads\YCSEP_static.csv"
DEFAULT_API_URL = "http://localhost:8001/asr"
DEFAULT_OUTPUT_PATH = Path("asr") / "TDK_subset.csv"
TDK_CHANNEL = "The_Daily_Ketchup_Podcast"
V2_AUDIO_BASE = "https://a3s.fi/swift/v1/YCSEP_v2/"
V2_CLIP_ENDPOINT = "https://ycsep.corpora.li/clip"
_thread_state = threading.local()


def http_session():
    """Each worker reuses its own connections; Session is not shared across threads."""
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session()
    return _thread_state.session


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_token != hyp_token)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def corpus_wer(rows: list[dict[str, str]]) -> float:
    edits = 0
    words = 0
    for row in rows:
        ref_words = normalize_words(row.get("text", ""))
        hyp_words = normalize_words(row.get("generated_text", ""))
        edits += edit_distance(ref_words, hyp_words)
        words += len(ref_words)
    return edits / words if words else 0.0


def row_key(row: dict[str, str]) -> str:
    identity = {
        "channel": row.get("channel", ""),
        "file": row.get("file", ""),
        "speaker": row.get("speaker", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "audio": row.get("audio", ""),
        "text": row.get("text", ""),
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fallback_clip_url(row: dict[str, str]) -> str:
    stem = row["file"].removesuffix(".TextGrid")
    base = V2_AUDIO_BASE + quote(stem + ".wav")
    return V2_CLIP_ENDPOINT + "?" + urlencode({
        "url": base,
        "start": f"{float(row['start_time']):.3f}",
        "end": f"{float(row['end_time']):.3f}",
        "fmt": "mp3",
    })


def audio_cache_name(row):
    return hashlib.sha256((row["audio"] + "|" + fallback_clip_url(row)).encode()).hexdigest() + ".mp3"


def download_audio(row: dict[str, str], handle, cache_dir: Path | None, cache_only=False):
    urls = [row["audio"], fallback_clip_url(row)]
    cached = cache_dir / audio_cache_name(row) if cache_dir else None
    if cached and cached.exists():
        with cached.open("rb") as source:
            shutil.copyfileobj(source, handle)
        return "cache"

    if cache_only:
        raise ValueError("Required cached MP3 is missing")

    last_error = None
    for url in urls:
        handle.seek(0)
        handle.truncate()
        try:
            timeout = (5, 20) if url == row["audio"] else (10, 60)
            with http_session().get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                size = 0
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > 64 * 1024 * 1024:
                        raise ValueError("Audio download exceeds 64 MiB")
                    handle.write(chunk)
            if size == 0:
                raise ValueError("Empty audio response")
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                handle.seek(0)
                with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as destination:
                    temporary_cache = Path(destination.name)
                    shutil.copyfileobj(handle, destination)
                os.replace(temporary_cache, cached)
            return "csv_audio" if url == row["audio"] else "ycsep_v2_clip"
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
    raise last_error or RuntimeError("No audio URL was available")


def parse_response(result: dict[str, object]) -> tuple[str, float]:
    if not isinstance(result, dict):
        raise ValueError("API response must be a JSON object")
    if not isinstance(result.get("transcription"), str):
        raise ValueError("API response transcription must be a string")
    duration = result.get("duration")
    if not isinstance(duration, str):
        raise ValueError("API response duration must be a string")
    seconds = float(duration)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("API response duration must be finite and nonnegative")
    return result["transcription"], seconds


def transcribe_row(row: dict[str, str], api_url: str, retries: int,
                   cache_dir: Path | None, cache_only=False) -> tuple[str, float | None, str]:
    for attempt in range(retries + 1):
        try:
            with tempfile.TemporaryFile() as audio:
                download_audio(row, audio, cache_dir, cache_only)
                audio.seek(0)
                response = http_session().post(
                    api_url,
                    files={"file": ("audio.mp3", audio, "audio/mpeg")},
                    timeout=(15, 600),
                )
            response.raise_for_status()
            transcription, duration = parse_response(response.json())
            return transcription, duration, ""
        except (requests.RequestException, ValueError, KeyError) as exc:
            error = type(exc).__name__
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    return "", None, error


def load_tdk_rows(csv_path: str, limit: int | None, shard_index=0,
                  num_shards=1) -> list[dict[str, str]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard index/count")
    if limit is not None and limit < 1:
        raise ValueError("Row limit must be positive or None")
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"channel", "file", "speaker", "start_time", "end_time", "text", "audio"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"YCSEP CSV must contain: {', '.join(sorted(required))}")
        rows = []
        position = 0
        for row in reader:
            if row["channel"] == TDK_CHANNEL:
                selected = position % num_shards == shard_index
                position += 1
                if not selected:
                    continue
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError("No The Daily Ketchup rows found")
    return rows


def export_csv(rows: list[dict[str, str]], database: sqlite3.Connection,
               output_path: Path) -> tuple[int, int, float]:
    fields = list(rows[0].keys()) + ["generated_text", "asr_duration_seconds", "asr_error"]
    temporary = output_path.with_suffix(".csv.tmp")
    successful_rows = []
    failures = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = database.execute(
                "SELECT generated_text, duration_seconds, error FROM results WHERE row_key=?",
                (row_key(row),),
            ).fetchone()
            generated_text, duration, error = record or ("", "", "not_processed")
            exported = dict(row, generated_text=generated_text,
                            asr_duration_seconds="" if duration is None else duration,
                            asr_error=error)
            if error:
                failures += 1
            else:
                successful_rows.append(exported)
            writer.writerow(exported)
    os.replace(temporary, output_path)
    return len(successful_rows), failures, corpus_wer(successful_rows)


def decode_tdk_subset(csv_path: str, api_url: str, output_path: Path, limit: int | None,
                      workers: int, retries: int, journal_path: Path,
                      cache_dir: Path | None, shard_index=0, num_shards=1,
                      cache_only=False) -> dict[str, object]:
    if workers < 1 or retries < 0:
        raise ValueError("Workers must be positive and retries nonnegative")
    rows = load_tdk_rows(csv_path, limit, shard_index, num_shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with sqlite3.connect(journal_path) as database:
        database.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            "row_key TEXT PRIMARY KEY, generated_text TEXT, duration_seconds REAL, error TEXT)"
        )
        done = {record[0] for record in database.execute("SELECT row_key FROM results WHERE error=''")}
        pending = iter(row for row in rows if row_key(row) not in done)
        processed = 0
        next_report = 100
        last_report = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}

            def refill() -> None:
                while len(futures) < workers * 2:
                    row = next(pending, None)
                    if row is None:
                        break
                    futures[pool.submit(transcribe_row, row, api_url, retries, cache_dir, cache_only)] = row

            refill()
            while futures:
                finished, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = futures.pop(future)
                    text, duration, error = future.result()
                    database.execute(
                        "INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                        (row_key(row), text, duration, error),
                    )
                    processed += 1
                database.commit()
                if processed >= next_report or time.perf_counter() - last_report >= 30:
                    print(json.dumps({
                        "processed_this_run": processed,
                        "total_rows": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "pending_requests": len(futures),
                    }), flush=True)
                    next_report = (processed // 100 + 1) * 100
                    last_report = time.perf_counter()
                refill()

        successful, failures, wer = export_csv(rows, database, output_path)

    summary = {
        "rows_in_output": len(rows),
        "successful_transcriptions": successful,
        "failed_transcriptions": failures,
        "corpus_wer_successful_rows": wer,
        "processed_this_run": processed,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "full_tdk" if limit is None else f"first_{limit}_tdk_rows",
        "execution_path": "multipart_http_api",
        "api_url": api_url,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "cache_only": cache_only,
        "assumption": "WER is computed after lowercasing and simple word-token normalization.",
    }
    if num_shards > 1:
        summary["scope"] = f"tdk_shard_{shard_index}_of_{num_shards}"
    output_path.with_name(output_path.stem + "_summary.txt").write_text(
        "\n".join([
            "Task 2c ASR comparison summary",
            f"Rows in output: {summary['rows_in_output']}",
            f"Successful transcriptions: {summary['successful_transcriptions']}",
            f"Failed transcriptions: {summary['failed_transcriptions']}",
            f"Corpus WER on successful rows: {summary['corpus_wer_successful_rows']:.4f}",
            f"Scope: {summary['scope']}",
            f"Elapsed seconds: {summary['elapsed_seconds']:.2f}",
            f"Assumption: {summary['assumption']}",
        ]) + "\n",
        encoding="utf-8",
    )
    output_path.with_suffix(".run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{failures} rows failed or remain missing; rerun to resume")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to YCSEP_static.csv")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="ASR API endpoint")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path")
    parser.add_argument("--limit", type=int, default=5, help="TDK rows to process. Use 0 for all rows.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API/download workers")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3, help="Retries per row")
    parser.add_argument("--journal", default="asr/TDK_subset.sqlite3", help="Resume journal path")
    parser.add_argument("--cache-dir", default="asr/audio_cache", help="Optional downloaded MP3 cache directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    row_limit = None if args.limit == 0 else args.limit
    decode_tdk_subset(
        csv_path=args.csv,
        api_url=args.api_url,
        output_path=Path(args.output),
        limit=row_limit,
        workers=args.workers,
        retries=args.retries,
        journal_path=Path(args.journal),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
