"""Validate complete API shard coverage before producing the Task 2c CSV."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from ycsep_decode import load_tdk_rows


def merge(source, shards, output):
    source, output = Path(source), Path(output)
    rows = load_tdk_rows(source, None)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    found, reports = {}, []
    count = len(shards)
    if not count:
        raise ValueError("No shards supplied")
    for item in shards:
        path = Path(item)
        report = json.loads(path.with_suffix(".run.json").read_text(encoding="utf-8"))
        job = json.loads((path.parent / "job_result.json").read_text(encoding="utf-8"))
        index = report["shard_index"]
        if (report["execution_path"] != "multipart_http_api"
                or report["num_shards"] != count or index in found
                or not 0 <= index < count or report["failed_transcriptions"] != 0
                or job["source_csv_sha256"] != source_hash):
            raise ValueError(f"Invalid or incomplete API provenance: {path.name}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        expected = rows[index::count]
        if len(predictions) != len(expected) or report["rows_in_output"] != len(expected):
            raise ValueError(f"Shard does not contain all expected rows: {index}")
        for original, prediction in zip(expected, predictions):
            if any(prediction.get(key) != value for key, value in original.items()):
                raise ValueError(f"Source row/order mismatch in shard {index}")
            duration = float(prediction["asr_duration_seconds"])
            if prediction["asr_error"] or not math.isfinite(duration) or duration < 0:
                raise ValueError(f"Failed inference in shard {index}")
        found[index] = predictions
        reports.append({"shard_index": index, "rows": len(predictions),
                        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    fields = list(rows[0]) + ["generated_text", "asr_duration_seconds", "asr_error"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for position in range(len(rows)):
            writer.writerow(found[position % count][position // count])
    os.replace(temporary, output)
    result = {"rows": len(rows), "failed": 0, "source_csv_sha256": source_hash,
              "execution_path": "multipart_http_api", "shards": reports,
              "audio_provenance": "Consult each CPU cache sources.jsonl; source WAV fallback is not byte-identical to original MP3"}
    output.with_suffix(".provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.shards, args.output), indent=2))
