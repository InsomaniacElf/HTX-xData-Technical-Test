"""Create CPU-only plots and numerical diagnostics from the metadata audit."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(folder):
    folder = Path(folder)
    audit = json.loads((folder / "audit.json").read_text())
    rng, sample, checks = random.Random(2026), [], Counter()
    non_tdk = 0
    with (folder / "data_quality_report.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                continue
            non_tdk += 1
            duration, words = float(row["duration"]), int(row["word_count"])
            wps = float(row["words_per_second"])
            for key, condition in {
                "duration_ge5_words_le1": duration >= 5 and words <= 1,
                "duration_ge5_words_le2": duration >= 5 and words <= 2,
                "duration_ge10_words_le3": duration >= 10 and words <= 3,
                "duration_le1_words_ge8": duration <= 1 and words >= 8,
                "duration_le2_words_ge15": duration <= 2 and words >= 15,
                "wps_lt0.5": wps < 0.5,
                "wps_gt7.5": wps > 7.5,
                "singlish_indicator": row["singlish_indicator"] == "True",
            }.items():
                checks[key] += bool(condition)
            point = (duration, words, wps)
            if len(sample) < 20000:
                sample.append(point)
            else:
                position = rng.randrange(non_tdk)
                if position < len(sample):
                    sample[position] = point
    with (folder / "video_overlap.csv").open(encoding="utf-8", newline="") as handle:
        spans = list(csv.DictReader(handle))
    total = sum(float(r["union_seconds"]) for r in spans)
    overlap = sum(float(r["metadata_overlap_seconds"]) for r in spans)
    different = sum(float(r["different_speaker_overlap_seconds"]) for r in spans)
    result = {"non_tdk_rows": non_tdk, "non_tdk_mismatch_counts": dict(checks),
        "all_corpus_union_hours": total / 3600,
        "all_corpus_metadata_overlap_hours": overlap / 3600,
        "all_corpus_different_speaker_overlap_hours": different / 3600,
        "all_corpus_overlap_fraction_of_union": overlap / total if total else None,
        "plot_sample_rows": len(sample), "sample_seed": 2026}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    axes[0, 0].barh(list(audit["channel_segment_hours"]), list(audit["channel_segment_hours"].values()), color="#24786c")
    axes[0, 0].set_xlabel("Annotated segment-hours (overlap can count twice)")
    order = ["<0.3", "0.3-0.5", "0.5-1", "1-5", "5-10", "10-15", "15-20", ">=20"]
    axes[0, 1].bar(order, [audit["duration_buckets"].get(k, 0) for k in order], color="#526db0")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].set(xlabel="Duration (s), all channels", ylabel="Segments")
    axes[1, 0].hexbin([p[0] for p in sample], [p[1] for p in sample], gridsize=45, bins="log", mincnt=1)
    axes[1, 0].set(xlabel="Metadata duration (s)", ylabel="Word count", title="Non-TDK uniform sample, n=20,000")
    axes[1, 1].hist([p[2] for p in sample], bins=[0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 25, 50, 100], color="#b44c69")
    axes[1, 1].set(xlabel="Words/second (sample; display limited to 100)", ylabel="Segments", yscale="log")
    fig.savefig(folder / "metadata_diagnostics.png", dpi=150)
    plt.close(fig)
    (folder / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    summarize(parser.parse_args().audit_dir)
