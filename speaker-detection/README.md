# Task 5: Speaker Retrieval

`sp-detect-5.ipynb` is the required executed notebook. The submitted method uses
only YCSEP audio and `pyannote/embedding`. It searches 6,823 speaker/video groups
through up to three separated representative clips per group, then exports the
annotated intervals of matched within-video speaker labels.

The outputs contain two predicted videos and 795 intervals. The 0.49 similarity
threshold is an unlabeled gap heuristic, not calibrated identity confidence.
Seventy-two groups were unscorable because their clips were too short. Complete
recall, speaker-label consistency and exact acoustic boundaries are not verified.
The reference was resampled to mono 16 kHz, not denoised. Human review was unsure;
it was not converted into positive or negative ground-truth labels.

## CPU Replay

From the repository root, build the analysis image as in the main README, then:

```powershell
docker run --rm -v "${PWD}:/workspace" htx-analysis python run_notebooks.py --tasks 5
```

This reads packaged compressed scores and provenance under `results/`, plots the
similarity distribution and regenerates `detected_titles.txt` and
`detected_timestamps.json`. It does not download audio or run embeddings again.
Title metadata is cached; no YouTube audio is downloaded.

## New GPU Retrieval

Build the ASR image first (`docker compose build asr-api`), then:

```powershell
docker build -f speaker-detection/Dockerfile -t htx-speaker .
```

Use this image with `--gpus all`, mount the repository at `/workspace`, and work
from `/workspace`. Accept the model's Hugging Face access conditions and supply
`HF_TOKEN` securely as an environment variable; never place credentials in Git.
The reference is downloaded from the supplied assessment URL by `prepare_reference.py`.
Alternatively place the supplied file at `test_docs/speaker-reference/speaker_reference_1min.wav`.

```bash
python speaker-detection/prepare_reference.py --output test_docs/speaker-reference --device cuda
python speaker-detection/retrieve_speakers.py --csv PATH/YCSEP_static.csv --reference test_docs/speaker-reference/reference-embeddings.npz --output test_docs/speaker-run --device cuda --workers 16
python speaker-detection/export_detections.py --scores test_docs/speaker-run/group-scores.jsonl --output speaker-detection --threshold 0.49 --threshold-note "Unlabeled gap heuristic; annotation boundaries assumed" --title-cache speaker-detection/results/title-provenance.json
```

The notebook can run the same steps by setting `RUN_SPEAKER_RETRIEVAL=1`,
`YCSEP_CSV`, `SPEAKER_REFERENCE` (directory), `SPEAKER_OUTPUT` and `SPEAKER_DEVICE=cuda`.
Do not reuse stale scores when changing model, reference or retrieval configuration.
The original full retrieval first pass took 62.9 minutes; this is not a runtime guarantee.

Future work: label cross-recording examples for threshold calibration, inspect
unknown groups, and evaluate full-recording re-diarization separately. Sortformer
and the alternative Colab VAD pipeline are not part of the submitted predictions.
