"""CPU-only YCSEP audit and source-video-disjoint candidate split.

This produces candidate lists, not audio-ready NeMo manifests. No network or GPU
is used. Suspicious speech rates are flags, not grounds for automatic exclusion.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

TDK = "The_Daily_Ketchup_Podcast"
BUCKETS = [(0.3, "<0.3"), (0.5, "0.3-0.5"), (1, "0.5-1"),
           (5, "1-5"), (10, "5-10"), (15, "10-15"),
           (20, "15-20"), (float("inf"), ">=20")]
SINGLISH = {"lah", "leh", "lor", "meh", "sia", "wah", "walao", "shiok", "paiseh"}


def video_key(filename):
    # YCSEP names include date--YouTube ID--title. Channel labels are not IDs.
    parts = filename.split("--", 2)
    return parts[1] if len(parts) == 3 else filename.removesuffix(".TextGrid")


def split_for(video, seed=2026, fraction=0.1):
    value = int(hashlib.sha256(f"{seed}/{video}".encode()).hexdigest()[:16], 16)
    return "validation" if value / 2**64 < fraction else "train"


def features(row):
    try:
        start, end = float(row["start_time"]), float(row["end_time"])
        valid = math.isfinite(start) and math.isfinite(end) and 0 <= start < end
    except (TypeError, ValueError):
        start, end, valid = 0, 0, False
    duration = end - start if valid else 0
    words = re.findall(r"\w+(?:['\u2019]\w+)*", row["text"].lower())
    count = len(words)
    wps = count / duration if valid else 0
    flags = []
    if not valid:
        flags.append("invalid_timestamp")
    if not words:
        flags.append("empty_or_nonlexical_text")
    if valid and duration < 0.5:
        flags.append("short")
    if duration > 20:
        flags.append("long")
    if duration >= 2 and wps < 0.5:
        flags.append("sparse_text")
    if duration >= 5 and count <= 1:
        flags.append("extreme_sparse")
    if wps > 7.5:
        flags.append("dense_text")
    if words and all(w in {"um", "uh", "hmm", "erm", "ah"} for w in words):
        flags.append("filler_only")
    if words and len(set(words)) == 1 and count >= 4:
        flags.append("repeated_tokens")
    if re.search(r"(.)\1{5,}", row["text"]):
        flags.append("repeated_characters")
    return start, end, duration, count, wps, flags, bool(set(words) & SINGLISH)


def overlap_stats(intervals):
    """Sweep events; union seconds with >=2 active intervals, without double count."""
    events = []
    for start, end, speaker in intervals:
        events.extend([(start, 1, speaker), (end, -1, speaker)])
    active = Counter()
    total = overlap = different = 0.0
    previous = events[0][0] if events else 0
    for stamp, delta, speaker in sorted(events):
        span = stamp - previous
        if sum(active.values()) > 0:
            total += span
        if sum(active.values()) >= 2:
            overlap += span
        if len(active) >= 2:
            different += span
        active[speaker] += delta
        if active[speaker] == 0:
            del active[speaker]
        previous = stamp
    return total, overlap, different


def prepare(source, output, seed=2026, fraction=0.1):
    started = time.perf_counter()
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    heldout_videos, heldout_urls = set(), set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"channel", "file", "speaker", "start_time", "end_time", "text", "audio"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required YCSEP columns")
        for row in reader:
            if row["channel"] == TDK:
                heldout_videos.add(video_key(row["file"]))
                heldout_urls.add(row["audio"])
    counts, hours, flags_count, buckets, exclusions = [Counter() for _ in range(5)]
    videos, intervals = defaultdict(set), defaultdict(list)
    split_videos, split_hours, split_counts = defaultdict(set), Counter(), Counter()
    seen_urls = set()
    fields = ["source_row", "channel", "video", "duration", "word_count", "characters",
              "words_per_second", "characters_per_second", "duration_bucket",
              "singlish_indicator", "flags", "split", "exclusion_reason"]
    with source.open(encoding="utf-8-sig", newline="") as handle, \
            (output / "data_quality_report.csv").open("w", encoding="utf-8", newline="") as quality, \
            (output / "train_candidates.jsonl").open("w", encoding="utf-8") as train, \
            (output / "validation_candidates.jsonl").open("w", encoding="utf-8") as validation:
        writer = csv.DictWriter(quality, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(csv.DictReader(handle)):
            channel, video = row["channel"], video_key(row["file"])
            start, end, duration, wc, wps, flags, singlish = features(row)
            counts[channel] += 1
            hours[channel] += duration / 3600
            videos[channel].add(video)
            bucket = next(name for bound, name in BUCKETS if duration < bound)
            buckets[bucket] += 1
            if duration:
                intervals[video].append((start, end, row["speaker"]))
            reason = ""
            if video in heldout_videos or row["audio"] in heldout_urls:
                split, reason = "test", "tdk_holdout"
            else:
                split = split_for(video, seed, fraction)
                if "invalid_timestamp" in flags or not wc:
                    reason = "invalid_timestamp_or_empty_text"
                elif not row["audio"].startswith("https://"):
                    reason = "invalid_audio_url"
                elif row["audio"] in seen_urls:
                    reason = "duplicate_audio_url"
                elif not 0.3 <= duration <= 20:
                    reason = "pilot_duration_outside_0.3_to_20"
                seen_urls.add(row["audio"])
                # Test metadata is not included in training-quality decision counts.
                flags_count.update(flags)
            if reason:
                exclusions[reason] += 1
            else:
                split_videos[split].add(video)
                split_counts[split] += 1
                split_hours[split] += duration / 3600
                candidate = {**row, "source_row": index, "video": video, "duration": duration,
                             "quality_flags": flags, "split": split}
                (train if split == "train" else validation).write(json.dumps(candidate, ensure_ascii=False) + "\n")
            writer.writerow(dict(zip(fields, [index, channel, video, duration, wc, len(row["text"]),
                wps, len(row["text"]) / duration if duration else 0, bucket, singlish,
                "|".join(flags), split, reason])))
    assert split_videos["train"].isdisjoint(split_videos["validation"])
    assert not (split_videos["train"] | split_videos["validation"]) & heldout_videos
    if not all(split_counts[k] for k in ("train", "validation")):
        raise ValueError("One split is empty; inspect source/video groups")
    with (output / "video_overlap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "union_seconds", "metadata_overlap_seconds", "different_speaker_overlap_seconds"])
        for video, spans in sorted(intervals.items()):
            writer.writerow([video, *overlap_stats(spans)])
    report = {"source_sha256": digest.hexdigest(), "seed": seed, "validation_fraction": fraction,
        "rows": sum(counts.values()), "channel_rows": dict(counts), "channel_segment_hours": dict(hours),
        "channel_videos": {k: len(v) for k, v in videos.items()}, "duration_buckets": dict(buckets),
        "non_tdk_flags": dict(flags_count), "exclusions": dict(exclusions),
        "split_rows": dict(split_counts), "split_segment_hours": dict(split_hours),
        "split_videos": {k: len(v) for k, v in split_videos.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Metadata overlap does not establish acoustic overlap",
            "Segment hours may double-count overlapping speech",
            "Video-disjoint is not necessarily speaker-disjoint",
            "URL deduplication cannot detect re-encoded duplicate audio",
            "Candidates are not downloaded, audio-validated, or final NeMo manifests"]}
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.csv, args.output)
