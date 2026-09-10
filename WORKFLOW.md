# HTX xData Workflow

Status is evidence-based. A notebook scaffold is not a completed experiment.
Commits record real milestones; dates and authorship are never manufactured.

| Task / To Do | Status | Done / Evidence, Decisions and Next Action |
| --- | --- | --- |
| 2c: full TDK decoding | Running, final acceptance pending | 12 Azure shards submitted; 11 Running and 1 Queued at this check. Verify successful row counts and increasing throughput, not status alone. Existing committed CSV contains five clips. |
| 2c: execution provenance | Open requirement gap | Azure fast harness calls Parakeet directly. Assignment requests API calls. Preserve source/model/preprocessing provenance and resolve API compliance before claiming completion. Batch-average duration is not measured per-request API latency. |
| 2c: WER/CER | Scorer tested; full results pending | Added Unicode-aware row WER/CER and pooled metrics with coverage. Two edge-case tests pass. Existing five clips: WER 0.368421, CER 0.222615; not full-corpus results. Input inference CSV preserved. |
| 2d: container code | Committed | Commit 2122a85 contains Dockerfile and asr-api Compose service. Compose validation passes. |
| 2d: real container acceptance | Blocked locally | Docker engine unavailable. Build, ping, real MP3 inference and cleanup checks still needed. Configuration validation alone does not prove runtime success. |
| 3a: strategy review | Done | Video-disjoint non-TDK validation; retain tokenizer/front end; conservative pilot before scaling. UTMOS/SIGMOS optional after evidence. Group by source video across channels. Speakers may recur across videos. |
| 3a: metadata EDA and splits | Done for initial candidate policy | Full 757,072-row audit took 107.39 s locally. TDK 218,011 rows held out; 433,320 train and 54,217 validation candidates. Six focused tests pass. Plots generated and inspected. Zero metadata overlap; acoustic overlap still unknown. See asr-train/EDA_FINDINGS.md. |
| 3a: audio pilot | To do | Validate/decode only selected training and validation candidates. Compare waveform duration against timestamps. No GPU spending during downloads. |
| 3a: fine-tuning notebook | Preparation scaffold validated | ycsep-train-3a.ipynb schema and Python cell syntax valid. Includes audit and plotting cells, rationale and planned configuration. Not a completed fine-tuning notebook; GPU execution and training curves pending. |
| 3a: baseline / pilot / selected model | To do | Base validation WER/CER, then pilot. Choose checkpoint using non-TDK validation only. Save actual timing, configuration and model artifact. |
| 3b: frozen-model TDK evaluation | To do | Required notebook ycsep-train-3b.ipynb; generated_text_ft and paired corpus WER/CER. Do not repeatedly select models against TDK. |
| 4: training report | To do | Report actual improvements and regressions, uncertainty by source video, limitations and proposed experiments. No invented metrics. |
| 5: speaker plan | Reviewed | Separate curation from ASR. Calibration needs manually verified cross-recording positives/negatives; reference-only positives overestimate generalization. |
| 5: retrieval and localization | To do | Use pyannote/embedding. Cache embeddings; audit retrieval misses. Clean regions for embedding estimation, but retain Speaker X overlap in final timestamps. Exactness requires boundary verification. |
| 6: essay | To do | At most 700 words plus references; final PDF must be checked. |
| 7: submission | To do | Final audit of executability and required artifacts, then user-controlled submission. |

## Execution and Timing

Local CPU: metadata EDA, splits, selected audio preparation, metrics, reports.
Azure GPU: finish healthy Task 2c work first; benchmark training before allocating more jobs.
Do not submit another large GPU run while current throughput/failures are unknown.
Record measured script wall time separately from Azure queue, setup and training time.
Historical task completion times are unknown unless supported by logs.

## Strategy Sources

- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/pyannote/embedding
- https://jitsi.github.io/jiwer/
