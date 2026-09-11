"""CPU-only MP3 staging for the API decoder, with an auditable source manifest."""
import argparse
import hashlib
import io
import json
import tarfile
import tempfile
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ycsep_decode import audio_cache_name, download_audio, load_tdk_rows, row_key
from wav_source import clip_wav
from recover_audio_cache import recover


def fetch(row, retries, source_mode="clip-service"):
    for attempt in range(retries + 1):
        try:
            if source_mode == "source-wav":
                wav = clip_wav(row["file"], float(row["start_time"]), float(row["end_time"]))
                encoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                    "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
                    input=wav, capture_output=True, check=True, timeout=60)
                payload, source = encoded.stdout, "ycsep_source_wav_pcm_to_mp3"
                if not payload:
                    raise ValueError("MP3 encoding produced no data")
            else:
                with tempfile.TemporaryFile() as handle:
                    source = download_audio(row, handle, None)
                    handle.seek(0)
                    payload = handle.read()
            return payload, source, ""
        except Exception as exc:
            error = type(exc).__name__
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    return None, None, error


def fetch_or_resume(row, retries, source_mode, resumed, directory):
    record = resumed.get(row_key(row))
    if record:
        if record['cache_name'] != audio_cache_name(row):
            raise ValueError('Resume row/cache identity mismatch')
        return (directory / record['cache_name']).read_bytes(), record['source'], ''
    return fetch(row, retries, source_mode)


def stage(args):
    with tempfile.TemporaryDirectory(prefix='tdk-resume-') as directory:
        records = recover(args.resume_cache, directory) if getattr(args, 'resume_cache', None) else []
        return _stage(args, {r['row_key']: r for r in records}, Path(directory))


def _stage(args, resumed, directory):
    rows = load_tdk_rows(args.csv, args.limit or None, args.shard_index, args.num_shards)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started, failures, sources = time.perf_counter(), 0, {}
    # Bounded windows prevent completed futures retaining the entire audio corpus.
    with tarfile.open(output / "audio.tar", "w") as archive, \
            (output / "sources.jsonl").open("w", encoding="utf-8") as manifest, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(rows), args.workers * 4):
            futures = {pool.submit(fetch_or_resume, row, args.retries, args.source_mode, resumed, directory): row
                       for row in rows[offset:offset + args.workers * 4]}
            for future in as_completed(futures):
                row = futures[future]
                payload, source, error = future.result()
                record = {"row_key": row_key(row), "cache_name": audio_cache_name(row),
                          "source": source, "error": error}
                if error:
                    failures += 1
                else:
                    record.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                    info = tarfile.TarInfo(record["cache_name"])
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
                    sources[source] = sources.get(source, 0) + 1
                manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            print(json.dumps({"processed": min(offset + args.workers * 4, len(rows)),
                "total": len(rows), "failures": failures,
                "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
            processed = min(offset + args.workers * 4, len(rows))
            if processed >= 64 and failures / processed > .9:
                raise RuntimeError("More than 90% downloads failed; aborting to avoid an error-only run")
    result = {"rows": len(rows), "failures": failures, "sources": sources,
              "reused_rows": sum(row_key(r) in resumed for r in rows),
              "shard_index": args.shard_index, "num_shards": args.num_shards,
              "elapsed_seconds": time.perf_counter() - started}
    (output / "cache_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures == len(rows):
        raise RuntimeError("No usable downloads; GPU dependency must not run")


def unpack_verified(folder, destination):
    """Never trust archive paths; copy only named regular MP3s and verify their bytes."""
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (folder / "sources.jsonl").read_text().splitlines()]
    expected = {row["cache_name"]: row for row in records if not row["error"]}
    seen = set()
    with tarfile.open(folder / "audio.tar", "r") as archive:
        for member in archive:
            if (not member.isfile() or member.name not in expected
                    or Path(member.name).name != member.name or "\\" in member.name):
                raise ValueError("Unexpected archive entry")
            record = expected[member.name]
            if member.size != record["bytes"] or member.size > 64 * 1024 * 1024:
                raise ValueError("Cache size mismatch")
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("Cache hash mismatch")
            (destination / member.name).write_bytes(payload)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive is missing manifest entries")
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-mode", choices=["clip-service", "source-wav"], default="clip-service")
    parser.add_argument("--resume-cache", help="Interrupted cache folder; only hash-verified MP3s are reused")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or args.limit < 0:
        parser.error("Invalid workers, retries, or limit")
    stage(args)
