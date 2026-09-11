"""Run a genuine localhost ASR API plus a bounded, resumable decoding shard.

Used by Azure ML command jobs. The API is the same asr_api.py tested in Docker;
all generated_text values are obtained through multipart POST /asr requests.
"""
import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import requests
from ycsep_decode import decode_tdk_subset
from cache_audio_shard import unpack_verified


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cache_context = tempfile.TemporaryDirectory(prefix="tdk-cache-") if args.cache else None
    cache = Path(cache_context.name) if cache_context else None
    if args.cache:
        unpack_verified(args.cache, cache)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with (output / "api.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "asr_api:app",
            "--app-dir", str(Path(__file__).resolve().parent), "--host", "127.0.0.1",
            "--port", "8001", "--workers", "1"], stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"API exited with code {server.returncode}; inspect api.log")
                try:
                    response = requests.get("http://127.0.0.1:8001/ping", timeout=3)
                    if response.status_code == 200 and response.json() == "pong":
                        break
                except (requests.RequestException, ValueError):
                    pass
                print(json.dumps({"event": "waiting_for_api", "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
                time.sleep(10)
            else:
                raise TimeoutError("API did not become ready within 15 minutes")
            startup_seconds = time.perf_counter() - started
            name = f"TDK_subset_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
            result = decode_tdk_subset(args.csv, "http://127.0.0.1:8001/asr",
                output / f"{name}.csv", None if args.limit == 0 else args.limit,
                args.workers, args.retries, output / f"{name}.sqlite3", cache,
                args.shard_index, args.num_shards, cache_only=bool(args.cache))
            result.update(startup_seconds=startup_seconds,
                total_seconds=time.perf_counter()-started,
                package_versions={k: importlib.metadata.version(k) for k in
                    ["torch", "nemo_toolkit", "fastapi", "librosa", "requests"]})
            (output / "job_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            if cache_context:
                cache_context.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", help="Verified CPU-stage MP3 archive directory")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or args.workers < 1 or args.retries < 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid row limit, workers, retries or shard configuration")
    run(args)
