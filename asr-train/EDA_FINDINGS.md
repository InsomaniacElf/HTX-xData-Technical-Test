# YCSEP Metadata Audit: First Results

Source SHA256: a76ea7653b2025bb934e14ecba1c6e5aebdfdb836af6264fb870b7b03316ce83

The full local CPU metadata audit took 107.39 seconds. Plot generation and package
setup are separate from this timing. No model training was performed.

| Finding | Observed Result | Decision |
| --- | --- | --- |
| Corpus size | 757,072 rows; 1,312 videos across six channels | Keep source row IDs and video identities in all derived data. |
| TDK holdout | 218,011 rows, 367 videos, 172.91 hours | All held out of training and validation. |
| Non-TDK pool | 539,061 rows | Apply pilot curation only here. |
| Initial training candidates | 433,320 rows, 852 videos, 387.03 hours | Sample a smaller pilot; do not train on every candidate automatically. |
| Initial validation candidates | 54,217 rows, 93 videos, 48.39 hours | Freeze a manageable video-diverse evaluation sample before training. |
| Duration exclusions, non-TDK | 51,524 rows outside 0.3-20 seconds | Pragmatic pilot policy; not a claim all excluded speech is bad. |
| Very short segments, all data | 74,070 below 0.3 s; 47,091 from 0.3 to 0.5 s | Short turns are frequent; keep valid 0.3-0.5 s conversational turns eligible. |
| Dense text, non-TDK | 23,853 rows above 7.5 words/s | Inspect sampled audio for alignment errors; do not delete automatically. |
| Sparse text, non-TDK | 522 rows below 0.5 words/s | VAD on sampled cases can separate silence from incomplete transcription. |
| Extreme sparse cases | 39 clips >=5 s with <=1 word; 58 with <=2 words | High-priority manual/audio review sample. |
| Extreme dense cases | 166 clips <=1 s with >=8 words; 41 clips <=2 s with >=15 words | Review timestamp correctness before neural quality scoring. |
| Singlish indicator | 7,958 non-TDK rows contain one of nine listed terms | Preserve these terms; this simple indicator misses many dialect features. |
| Metadata overlap | Zero overlapping seconds in this CSV; union audio 620.29 h | Do not add corpus-wide diarization on the basis of timestamp overlap. This does not establish absence of acoustic overlap. |

The corpus is strongly imbalanced: Yah_Lah_BUT contributes 288.32 hours, compared
with 15.93 hours for Historyogi. A pilot should control channel representation.
The plotted 20,000-row uniform non-TDK sample shows the expected duration/text
relationship plus outliers. These outliers motivate targeted validation, not a
global MOS-scoring pipeline.

The train/validation split groups by YouTube ID parsed from source filename,
falling back to source filename when needed. Its seed is 2026. Split video sets
are checked for disjointness and exclusion of all TDK videos. Speaker independence
and re-encoded audio duplication are not established by this metadata audit.

Generated local evidence is in test_docs/test/runtime/metadata-audit: audit.json,
diagnostics.json, metadata_diagnostics.png, data_quality_report.csv,
video_overlap.csv and candidate JSONL files. Regenerate using prepare_metadata.py
and summarize_eda.py. The large intermediate files stay outside Git.
