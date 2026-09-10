# Task 3: Current Preparation Milestone

This directory currently implements CPU metadata auditing and contains the Task
3a notebook groundwork. Fine-tuning has not been completed by these files.

Run the dependency-free audit from the repository root:

```powershell
python asr-train/prepare_metadata.py --csv PATH/YCSEP_static.csv --output test_docs/test/runtime/metadata-audit
python -m unittest discover -s asr-train -p "test_*.py"
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

Set YCSEP_CSV and optionally YCSEP_AUDIT before executing ycsep-train-3a.ipynb.
The notebook records assumptions, selection policy and pending GPU work.

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

See ../WORKFLOW.md for evidence and next actions.
