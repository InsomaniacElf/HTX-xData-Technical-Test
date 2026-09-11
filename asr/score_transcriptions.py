"""Add row WER/CER and pooled corpus metrics to an existing inference CSV.

Unicode letters/digits and internal apostrophes are retained. Punctuation/case
are ignored; CER excludes whitespace. Original transcripts are never changed.
Empty-reference row rates are blank (undefined); insertion counts still enter
pooled metrics. Failed requests are excluded and coverage is reported explicitly.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from jiwer import process_characters, process_words


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text))


def score(source, output, prediction="generated_text", suffix="base", error_column="asr_error"):
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output path to preserve the inference artifact")
    counts = dict(rows=0, scored=0, failed=0, word_edits=0, reference_words=0,
                  character_edits=0, reference_characters=0, empty_references=0)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with source.open(encoding="utf-8-sig", newline="") as handle, \
            temporary.open("w", encoding="utf-8", newline="") as dest:
        reader = csv.DictReader(handle)
        if not {"text", prediction, error_column} <= set(reader.fieldnames or []):
            raise ValueError("CSV must contain reference, prediction and explicit error-status columns")
        extra = [f"wer_{suffix}", f"cer_{suffix}"]
        writer = csv.DictWriter(dest, fieldnames=list(reader.fieldnames) + [k for k in extra if k not in reader.fieldnames])
        writer.writeheader()
        for row in reader:
            counts["rows"] += 1
            row.update({k: "" for k in extra})
            if row[error_column]:
                counts["failed"] += 1
            else:
                ref, hyp = normalize(row["text"]), normalize(row[prediction])
                nw, nc = len(ref.split()), len(ref.replace(" ", ""))
                # JiWER 3.1 rejects empty references; every hypothesis unit is an insertion.
                if nw:
                    words = process_words(ref, hyp)
                    chars = process_characters(ref.replace(" ", ""), hyp.replace(" ", ""))
                    we = words.substitutions + words.deletions + words.insertions
                    ce = chars.substitutions + chars.deletions + chars.insertions
                else:
                    we, ce = len(hyp.split()), len(hyp.replace(" ", ""))
                row[extra[0]] = we / nw if nw else ""
                row[extra[1]] = ce / nc if nc else ""
                counts["scored"] += 1
                counts["empty_references"] += not bool(nw)
                counts["word_edits"] += we
                counts["reference_words"] += nw
                counts["character_edits"] += ce
                counts["reference_characters"] += nc
            writer.writerow(row)
    os.replace(temporary, output)
    counts["corpus_wer"] = counts["word_edits"] / counts["reference_words"] if counts["reference_words"] else None
    counts["corpus_cer"] = counts["character_edits"] / counts["reference_characters"] if counts["reference_characters"] else None
    counts["coverage"] = counts["scored"] / counts["rows"] if counts["rows"] else 0
    counts["normalization"] = "unicode_nfkc_casefold_words_apostrophes_v1; CER excludes spaces"
    counts["prediction_column"] = prediction
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    counts["source_sha256"] = digest.hexdigest()
    output.with_suffix(".metrics.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction", default="generated_text")
    parser.add_argument("--suffix", default="base")
    parser.add_argument("--error-column", default="asr_error")
    args = parser.parse_args()
    score(args.input, args.output, args.prediction, args.suffix, args.error_column)
