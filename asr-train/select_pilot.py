"""Freeze channel-balanced, video-diverse pilot candidates before audio download."""
import argparse
import hashlib
import heapq
import json
from collections import defaultdict
from pathlib import Path


def select(path, hours, seed=2026, per_video=128):
    groups = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["channel"] == "The_Daily_Ketchup_Podcast":
                raise ValueError("TDK contamination in development candidates")
            priority = int(hashlib.sha256(f"{seed}/{row['source_row']}".encode()).hexdigest(), 16)
            heap = groups[(row["channel"], row["video"])]
            item = (-priority, row["source_row"], row)
            if len(heap) < per_video:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    channels = sorted({key[0] for key in groups})
    if not channels or hours <= 0:
        raise ValueError("Need positive target hours and nonempty candidates")
    selected, summary = [], {}
    for channel in channels:
        videos = sorted([key for key in groups if key[0] == channel],
            key=lambda key: hashlib.sha256(f"{seed}/{key[1]}".encode()).hexdigest())
        pools = [sorted(groups[key], reverse=True) for key in videos]
        remaining = target = hours * 3600 / len(channels)
        for rank in range(per_video):
            for pool in pools:
                if rank < len(pool):
                    row = pool[rank][2]
                    if row["duration"] <= remaining:
                        selected.append(row)
                        remaining -= row["duration"]
        summary[channel] = {"hours": (target-remaining)/3600, "target_hours": target/3600}
        if remaining > target * .05:
            raise ValueError(f"Insufficient candidate reservoir for {channel}; enlarge per_video")
    return sorted(selected, key=lambda r: r["source_row"]), summary


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    groups, report = [], {}
    for split, hours in [("train", args.train_hours), ("validation", args.validation_hours)]:
        rows, summary = select(Path(args.audit) / f"{split}_candidates.jsonl", hours)
        groups.append({r["video"] for r in rows})
        with (output / f"{split}-selected.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report[split] = {"rows": len(rows), "videos": len(groups[-1]), "channels": summary}
    if groups[0] & groups[1]:
        raise ValueError("Training/validation video leakage")
    report.update(seed=2026, policy="Equal channel hours, round-robin videos, hash-selected candidates",
                  status="Frozen candidates; audio validity not yet established")
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-hours", type=float, default=10)
    parser.add_argument("--validation-hours", type=float, default=2)
    main(parser.parse_args())
