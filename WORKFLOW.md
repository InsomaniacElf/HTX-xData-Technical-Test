# HTX xData Workflow

Status is evidence-based. A notebook scaffold is not a completed experiment.
Commits record real milestones; dates and authorship are never manufactured.

| Task / To Do | Status | Done / Evidence, Decisions and Next Action |
| --- | --- | --- |
| 2c: full TDK decoding | Corrected pilot passed; full run pending | Old 12 jobs canceled; 4,544 direct-inference rows retained as reference only. Azure job task2c-api-pilot-20260911-0627 completed 256/256 actual API requests in 185.38 s after 50.06 s startup. Local 128-row API test also passed. Full 218,011-row artifact remains incomplete. |
| 2c: execution provenance | API path verified on pilot | run_api_shard.py starts the same asr_api.py and calls multipart POST /asr. Client uses thread-local persistent sessions and exact/disjoint modulo shards. Added CPU cache archives with per-file SHA256 and source labels; GPU cache-only mode cannot silently download missing files. Five shard/cache tests pass. CPU-to-GPU Azure integration pilot submitted; full pipeline not yet validated. |
| 2c: WER/CER | Scorer tested; full results pending | Added Unicode-aware row WER/CER and pooled metrics with coverage. Two edge-case tests pass. Existing five clips: WER 0.368421, CER 0.222615; not full-corpus results. Input inference CSV preserved. |
| 2d: container code | Committed | Commit 2122a85 contains Dockerfile and asr-api Compose service. Compose validation passes. |
| 2d: real container acceptance | Passed | Normal Docker build and pip check passed; NeMo imports passed. RTX 4090 GPU passthrough verified. Real GET /ping returned pong; POST /asr returned string transcription and duration 5.579523 s on first inference. Temporary MP3/WAV count before/after: 0/0. Readiness plus test took 73.42 s (excludes image build). See asr/DOCKER_VALIDATION.md. |
| 3a: strategy review | Done | Video-disjoint non-TDK validation; retain tokenizer/front end; conservative pilot before scaling. UTMOS/SIGMOS optional after evidence. Group by source video across channels. Speakers may recur across videos. |
| 3a: metadata EDA and splits | Done for initial candidate policy | Full 757,072-row audit took 107.39 s locally. TDK 218,011 rows held out; 433,320 train and 54,217 validation candidates. Six focused tests pass. Plots generated and inspected. Zero metadata overlap; acoustic overlap still unknown. See asr-train/EDA_FINDINGS.md. |
| 3a: audio pilot | Validation prepared; training download running | Frozen candidates: 10 h channel-balanced training, 2 h validation, video-disjoint and no TDK. All 2,133 validation clips decoded to mono 16 kHz PCM and passed duration checks in 1,046.05 s locally. Actual decoded validation duration 2.02285 h. Training preparation is still running; no training result claimed. |
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

Read-only monitoring script: test_docs/test/azure_task2c_fast/monitor_jobs.py.
It checks statuses and streamed logs, writes monitor-latest.json and history,
and flags repeated unchanged logs for review. It does not cancel/restart jobs,
prove successful CSV coverage, or fix download bottlenecks automatically.

## Strategy Sources

- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/pyannote/embedding
- https://jitsi.github.io/jiwer/
