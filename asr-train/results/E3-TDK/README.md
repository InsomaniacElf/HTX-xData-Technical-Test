# E3 Full Daily Ketchup Evaluation

All 218,011 rows across 367 videos passed strict merge checks: zero failed
predictions, exact baseline row order, original shard identity, baseline CSV
hashes, and the same verified cached MP3 audio. The downloaded model SHA-256 is
`c7c194c87490455b673cc7ad713d53a760231e236ad5dfb365eab585c8a81877`.

| Metric | Base | E3 |
| --- | --- | --- |
| Corpus WER | 24.4272% | 19.0689% |
| Corpus CER | 18.1281% | 14.2119% |
| Word edits | 607,513 | 474,249 |
| Reference words | 2,487,035 | 2,487,035 |

WER decreases by 5.3583 percentage points, or 21.9360% relative to base.
The 2,000-draw video-cluster bootstrap interval for E3-minus-base WER is
[-5.4961, -5.2234] percentage points. Video independence is an approximation.
All 367 videos improve in pooled WER; individual rows include 80,758 improvements,
28,279 regressions and 108,974 ties. This is not a guarantee of accurate text on
each clip. The previously published E1 full-TDK WER was 21.3120%.

E3 was motivated and selected by non-TDK validation, not these TDK results.
Its best native validation WER was 24.6470% at step 3,000. No convergence claim
is made. E4's inconsistent resumed validation score and selected artifact are
excluded, not substituted into this evaluation.

The public `artifacts.json` identifies the matching E3 model and complete CSV
in release `submission-e3-20260913`. The E1 release is retained as history.
The existing `merge_evaluation_shards.py`, `score_transcriptions.py`,
`compare_models.py` and `analyze_errors.py` produced the evidence in this folder.
