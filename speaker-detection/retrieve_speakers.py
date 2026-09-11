"""Score all YCSEP annotated speaker/video groups against the supplied reference.

This is retrieval, not a guarantee of exact diarization. Group identity assumes
YCSEP speaker labels are consistent within a video. Failed groups remain visible.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import sqlite3
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asr"))
from wav_source import clip_wav


def collect(csv_path):
    groups, invalid = {}, []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            try:
                start, end = float(row["start_time"]), float(row["end_time"])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError("Invalid timestamp")
            except ValueError:
                invalid.append(index)
                continue
            speaker = row["speaker"].strip()
            if speaker.casefold() in {"", "nan", "none", "unknown"}:
                speaker = f"unlabelled-{index}"
            key = (row["channel"], row["file"], speaker)
            group = groups.setdefault(key, {"segments": [], "pool": [], "fallback": None})
            group["segments"].append((start, end))
            candidate = {"audio": row["audio"], "file": row["file"], "start": start,
                         "end": end, "row": index, "duration": end-start}
            if group["fallback"] is None or candidate["duration"] > group["fallback"]["duration"]:
                group["fallback"] = candidate
            if 2 <= end-start <= 15:
                group["pool"].append(candidate)
                if len(group["pool"]) > 24:
                    group["pool"].sort(key=lambda r: hashlib.sha256(r["audio"].encode()).digest())
                    del group["pool"][24:]
    return groups, invalid


def representatives(group):
    pool = group["pool"] or [group["fallback"]]
    chosen = [max(pool, key=lambda row: min(row["duration"], 5))]
    while len(chosen) < min(3, len(pool)):
        remaining = [row for row in pool if row not in chosen]
        chosen.append(max(remaining, key=lambda r: min(abs(r["start"]-s["start"]) for s in chosen)))
    return chosen


_local = threading.local()


def fetch(row):
    import librosa
    import numpy as np
    import requests
    start, end = row["start"], row["end"]
    if end-start > 5:
        start += (end-start-5)/2
        end = start+5
    for attempt in range(3):
        try:
            payload = clip_wav(row["file"], start, end)
            samples, _ = librosa.load(io.BytesIO(payload), sr=16000, mono=True)
            if len(samples) < 4000 or not np.isfinite(samples).all() or np.max(np.abs(samples)) == 0:
                raise ValueError("Insufficient or invalid audio")
            return samples, "ycsep_source_wav_range", ""
        except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
            error = type(exc).__name__ + ": " + str(exc)[:240]
            time.sleep(attempt+1)
    return None, None, error


def fetch_via_clip_service(row):
    """Retained alternative; the current run uses source WAV ranges during outage."""
    import librosa
    import numpy as np
    import requests
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    source = "https://a3s.fi/swift/v1/YCSEP_v2/" + quote(row["file"].removesuffix(".TextGrid") + ".wav")
    fallback = "https://ycsep.corpora.li/clip?" + urlencode({"url": source,
        "start": f"{row['start']:.3f}", "end": f"{row['end']:.3f}", "fmt": "mp3"})
    error = "unknown"
    for attempt in range(2):
        for url in [row["audio"], fallback]:
            try:
                timeout = (5, 20) if url == row["audio"] else (10, 60)
                with _local.session.get(url, timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    payload = bytearray()
                    for block in response.iter_content(65536):
                        payload.extend(block)
                        if len(payload) > 64*1024*1024:
                            raise ValueError("Audio exceeds size limit")
                samples, _ = librosa.load(io.BytesIO(payload), sr=16000, mono=True, duration=15)
                if len(samples) < 8000 or not np.isfinite(samples).all() or np.max(np.abs(samples)) == 0:
                    raise ValueError("Insufficient or invalid audio")
                if len(samples) > 80000:
                    offset = (len(samples)-80000)//2
                    samples = samples[offset:offset+80000]
                return samples, "csv_audio" if url == row["audio"] else "ycsep_v2_clip", ""
            except (requests.RequestException, ValueError, RuntimeError, OSError) as exc:
                error = type(exc).__name__ + ": " + str(exc)[:240]
        time.sleep(attempt+1)
    return None, None, error


def run(args):
    import numpy as np
    import torch
    from pyannote.audio import Inference
    from embedding_model import load_embedding_model

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    groups, invalid = collect(args.csv)
    keys = sorted(groups)
    if args.limit_groups:
        keys = keys[:args.limit_groups]
    reference = np.load(args.reference)["prototype"]
    model, checkpoint = load_embedding_model()
    inference = Inference(model, window="whole")
    inference.to(torch.device(args.device))
    with Path(args.csv).open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    config = {"csv_sha256": source_hash, "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "window_policy": "Up to three temporally separated clips per group, central 5 s maximum",
              "groups_total": len(groups), "groups_selected": len(keys)}
    config_path = output / "retrieval-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Resume configuration mismatch; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = {row["row"]: row for key in keys for row in representatives(groups[key])}
    with sqlite3.connect(output / "embeddings.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS embeddings (row_id INTEGER PRIMARY KEY, vector BLOB, source TEXT, error TEXT)")
        done = {r[0] for r in db.execute("SELECT row_id FROM embeddings WHERE error=''")}
        pending = [r for index, r in rows.items() if index not in done]
        completed, failures = 0, 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(pending), args.workers * 4):
                futures = {pool.submit(fetch, r): r for r in pending[offset:offset+args.workers*4]}
                for future in as_completed(futures):
                    row = futures[future]
                    samples, source, error = future.result()
                    vector = None
                    if not error:
                        if len(samples) < 16000:
                            samples = np.pad(samples, (0, 16000-len(samples)))
                        with torch.inference_mode():
                            value = np.asarray(inference({"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": 16000})).reshape(-1)
                        norm = np.linalg.norm(value)
                        if not np.isfinite(value).all() or norm <= 0:
                            error = "InvalidEmbedding"
                        else:
                            vector = (value/norm).astype("float32").tobytes()
                    failures += bool(error)
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)", (row["row"], vector, source, error))
                    db.commit()
                    completed += 1
                print(json.dumps({"processed_this_run": completed, "pending_at_start": len(pending),
                    "failures": failures, "elapsed_seconds": round(time.perf_counter()-started, 1)}), flush=True)
        scores = []
        for key in keys:
            similarities, failed = [], []
            selected = representatives(groups[key])
            for row in selected:
                record = db.execute("SELECT vector, error FROM embeddings WHERE row_id=?", (row["row"],)).fetchone()
                if record is None or record[1]:
                    failed.append(row["row"])
                else:
                    similarities.append(float(np.frombuffer(record[0], dtype="float32") @ reference))
            scores.append({"channel": key[0], "file": key[1], "speaker": key[2],
                "median_similarity": float(np.median(similarities)) if similarities else None,
                "similarities": similarities, "failed_rows": failed,
                "representative_rows": [r["row"] for r in selected], "segments": groups[key]["segments"]})
    scores.sort(key=lambda r: r["median_similarity"] if r["median_similarity"] is not None else -2, reverse=True)
    with (output / "group-scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"groups_total": len(groups), "groups_processed": len(keys), "representative_clips": len(rows),
        "failed_groups": sum(bool(r["failed_rows"]) for r in scores), "invalid_timestamp_rows": invalid,
        "elapsed_seconds": time.perf_counter()-started,
        "status": "retrieval_scores_only", "threshold_selected": False,
        "boundary_source": "YCSEP annotations; not independently verified"}
    (output / "retrieval-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if report["failed_groups"]:
        raise RuntimeError("Some representative clips failed; inspect the journal and resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--limit-groups", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.limit_groups < 0:
        parser.error("Invalid worker count or group limit")
    run(args)
