import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd
import requests


DEFAULT_CSV_PATH = r"C:\Users\Sneha\Downloads\YCSEP_static.csv"
DEFAULT_API_URL = "http://localhost:8001/asr"
DEFAULT_OUTPUT_PATH = Path("asr") / "TDK_subset.csv"
TDK_CHANNEL = "The_Daily_Ketchup_Podcast"


def normalize_words(text: str) -> list[str]:
    """Lowercase and keep simple word tokens for a lightweight WER comparison."""
    import re

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


def corpus_wer(references: pd.Series, hypotheses: pd.Series) -> float:
    edits = 0
    words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_words = normalize_words(reference)
        hyp_words = normalize_words(hypothesis)
        edits += edit_distance(ref_words, hyp_words)
        words += len(ref_words)
    return edits / words if words else 0.0


def transcribe_audio_url(audio_url: str, api_url: str) -> tuple[str, float]:
    """Download one YCSEP MP3 segment and send it to the local ASR API."""
    response = requests.get(audio_url, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(response.content)
        audio_path = tmp.name

    try:
        with open(audio_path, "rb") as audio_file:
            api_response = requests.post(
                api_url,
                files={"file": ("audio.mp3", audio_file, "audio/mpeg")},
                timeout=120,
            )

        api_response.raise_for_status()
        result = api_response.json()
        return result["transcription"], float(result["duration"])

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def decode_tdk_subset(
    csv_path: str,
    api_url: str,
    output_path: Path,
    limit: int | None,
) -> pd.DataFrame:
    """Create TDK_subset.csv with Parakeet base-model transcriptions."""
    df = pd.read_csv(csv_path)
    tdk = df[df["channel"] == TDK_CHANNEL].copy()

    if limit is not None:
        tdk = tdk.head(limit).copy()

    generated_text = []
    asr_duration = []
    asr_error = []
    total_rows = len(tdk)

    for item_no, (index, row) in enumerate(tdk.iterrows(), start=1):
        print(f"[{item_no}/{total_rows}] Transcribing row {index}")

        try:
            transcription, duration = transcribe_audio_url(row["audio"], api_url)
            generated_text.append(transcription)
            asr_duration.append(duration)
            asr_error.append("")
            print(f"Reference:  {row['text']}")
            print(f"Generated:  {transcription}")
            print(f"Duration:   {duration:.2f}s")

        except Exception as exc:
            generated_text.append("")
            asr_duration.append("")
            asr_error.append(str(exc))
            print(f"ERROR: {exc}")

        print()

    tdk["generated_text"] = generated_text
    tdk["asr_duration_seconds"] = asr_duration
    tdk["asr_error"] = asr_error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tdk.to_csv(output_path, index=False)

    successful = tdk[tdk["asr_error"].fillna("").eq("")]
    wer = corpus_wer(successful["text"], successful["generated_text"]) if len(successful) else float("nan")
    summary_path = output_path.with_name(output_path.stem + "_summary.txt")
    summary_path.write_text(
        "\n".join(
            [
                "Task 2c ASR comparison summary",
                f"Rows in output: {len(tdk)}",
                f"Successful transcriptions: {len(successful)}",
                f"Failed transcriptions: {len(tdk) - len(successful)}",
                f"Corpus WER on successful rows: {wer:.4f}",
                "Assumption: WER is computed after lowercasing and simple word-token normalization.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tdk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode The Daily Ketchup YCSEP clips through the ASR API."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to YCSEP_static.csv")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="ASR API endpoint")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV path for the TDK subset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of TDK rows to process. Use --limit 0 for all rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    row_limit = None if args.limit == 0 else args.limit
    decoded = decode_tdk_subset(
        csv_path=args.csv,
        api_url=args.api_url,
        output_path=Path(args.output),
        limit=row_limit,
    )
    print(f"Saved {len(decoded)} rows to {args.output}")
