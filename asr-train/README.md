# Task 3: Fine-Tuning and Evaluation

The executed Task 3a notebook records full-parameter Parakeet fine-tuning on
10.046 hours of non-TDK audio with 2.023 hours of video-disjoint validation.
Training completed 1,200 updates in 73.05 minutes. Native validation WER improved
from 0.322042 to 0.265402; the best checkpoint is update 1,200. Full-TDK evaluation
is complete: WER 24.4272% -> 21.3120%, CER 18.1281% -> 15.8369%, 218,011 paired
rows and no failed predictions. Native validation WER uses NeMo's metric and is
not directly comparable to the explicitly normalized TDK metric. Both executed
notebooks and Python helpers are supplied. See results/E1 and results/E1-TDK.

Run `python download_artifacts.py --model` from the repository root to restore
the CSV and selected model. The model is stored at
`asr-train/models/parakeet-tdt-0.6b-v3-ycsep.nemo`. CPU replay requires only the CSV,
not the model. Use the root README's analysis Docker and `run_notebooks.py`.

Run the dependency-free audit from the repository root:

```powershell
python asr-train/prepare_metadata.py --csv PATH/YCSEP_static.csv --output test_docs/test/runtime/metadata-audit
```

It creates data_quality_report.csv, video_overlap.csv, audit.json and deterministic
train/validation candidate JSONL files. They contain audio URLs, not validated
local WAV paths. Do not pass them directly to NeMo training.

For plots, metrics and notebook schema validation, dependencies are recorded in
requirements-analysis.txt. A CPU-only Docker environment is provided:

```powershell
docker build -f asr-train/Dockerfile.analysis -t htx-analysis .
docker run --rm -v "${PWD}:/workspace" htx-analysis python asr-train/prepare_metadata.py --csv /workspace/PATH/YCSEP_static.csv --output /workspace/test_docs/test/runtime/metadata-audit
```

Default notebook execution replays the packaged E1 evidence without training.
For a new GPU run, build `asr-train/Dockerfile`, use `--gpus all` and mount this
repository at `/workspace`. Set `RUN_TRAINING=1`, `YCSEP_CSV` to the source CSV,
and `YCSEP_EXPERIMENT` to a fresh output directory before executing Task 3a.
Optionally set YCSEP_AUDIT, YCSEP_SELECTION and YCSEP_AUDIO to scratch folders.
The notebook prepares the deterministic split/audio and calls the training helper.
FFmpeg and dependencies are installed by that Dockerfile. Network access is
required to retrieve source audio and the base model. No denoising or neural
quality filtering was used in E1; difficult-but-valid speech was not discarded
based on base-model WER.

For new Task 3b inference, set `RUN_TDK_INFERENCE=1`, `YCSEP_MODEL` to the restored
model and optionally `TDK_CACHE` to the verified baseline MP3 cache. Use a fresh
TDK_EVALUATION output directory. Without a cache, source WAV ranges are recovered
again, so report this execution path and do not claim original cached-byte identity.
The default replay recomputes full paired WER/CER and 2,000 video-bootstrap draws.

Freeze a bounded, channel-balanced pilot and prepare its audio on CPU:

```powershell
python asr-train/select_pilot.py --audit test_docs/test/runtime/metadata-audit --output test_docs/test/runtime/pilot-selection --train-hours 10 --validation-hours 2
python asr-train/prepare_audio.py --selection test_docs/test/runtime/pilot-selection/validation-selected.jsonl --output test_docs/test/runtime/pilot-audio --split validation --workers 16
python asr-train/prepare_audio.py --selection test_docs/test/runtime/pilot-selection/train-selected.jsonl --output test_docs/test/runtime/pilot-audio --split train --workers 16
```

Audio preparation requires FFmpeg on PATH and requests. It preserves transcripts,
checks decoded duration, sample rate, channels and nonzero waveform, records failed
selections, and reuses valid local WAVs on restart. The selected 2,133 validation
clips passed these checks in 1,046.05 seconds on the development machine. Audio
validity is not proof of transcription accuracy. Training and validation retain
disjoint source videos; TDK is excluded from both.

Score existing base-model output without downloading audio or using a GPU:

```powershell
python asr/score_transcriptions.py --input asr/TDK_subset.csv --output test_docs/test/runtime/TDK_scored.csv
```

The scorer requires explicit asr_error status, preserves the input, and adds
wer_base/cer_base. For future fine-tuned output, set --prediction generated_text_ft
--suffix ft --error-column asr_error_ft. Final comparisons must use identical
reference rows and report missing predictions for each model.

## Strategy Review

The proposed split, pilot and checkpoint strategy is sound. UTMOS, SIGMOS,
SortFormer, neural curation tiers and multiple ablations are too much to make
prerequisites for the first fine-tuning run. Start with evidence from metadata,
sampled audio and a working baseline. Preserve difficult but correctly aligned
accent/code-switching data. The Parakeet model card lists English but not Mandarin
or Malay; record code-switching as a possible limitation, not guaranteed support.

Data curation is a training choice. Keep the validation distribution fixed and
document objective invalid-audio exclusions. Hash grouping gives video-disjoint,
not speaker-disjoint, evaluation. TDK is final test only.

Working notes, verification scripts and intermediate artifacts are kept locally
and excluded from the submission repository.
