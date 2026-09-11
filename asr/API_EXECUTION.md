# Task 2c Execution Evidence

The required decoding path is MP3 -> multipart HTTP POST /asr -> generated_text.
Azure is an execution location, not a substitute for the API. run_api_shard.py
starts asr_api.py with one Uvicorn process and waits for readiness with a bounded
timeout and process-exit checks.

## Verified Pilot

- Azure job: task2c-api-pilot-20260911-0627.
- Scope: first 256 TDK rows, not the complete held-out set.
- Successful: 256; failures: 0.
- Decoder wall time: 185.379 s; model startup: 50.059 s; total: 236.055 s.
- Execution: multipart_http_api, 32 client workers, one GPU API process.
- Local 128-row test: 128 successful, 0 failed, 69.374 s wall time.
- Warm local requests with already downloaded MP3: approximately 0.06 s each.

The source downloads dominate the end-to-end pilot. More download workers are
not equivalent to more GPU inference processes. CPU staging is therefore being
integration-tested before full deployment. Cache archives preserve MP3 bytes,
per-file hashes and original/fallback URL-source labels. They do not contain
precomputed transcriptions. Cache-only GPU execution still calls /asr for every
row, and reports missing cached audio as an error, never a successful blank.

The previous direct-inference reference run is not evidence of API compliance.
Its recovered rows are not merged into the final API output. The full
TDK_subset.csv and canonical pooled WER/CER remain pending.

## Tests

```powershell
python -m unittest discover -s asr -p test_decode_shards.py
python -m unittest discover -s asr -p test_audio_cache.py
```

The cache pipeline is optional execution infrastructure. The ordinary local
decoder continues to download audio and call the user-specified API directly.
