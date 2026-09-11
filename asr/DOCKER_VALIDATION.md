# Task 2d: Docker Acceptance Evidence

Verified locally on 2026-09-11 with Docker Desktop 29.6.1 and an NVIDIA GeForce
RTX 4090 Laptop GPU (16 GB). The service name is asr-api.

- Image build succeeded with normal apt timestamp checks.
- python -m pip check: no broken requirements.
- NeMo ASR, FastAPI and SoundFile imports succeeded during build.
- GET /ping returned the JSON string "pong".
- A real MP3 POST to /asr returned HTTP 200 and application/json.
- transcription and duration were both strings; first inference duration was
  5.579523 seconds. This is a cold inference measurement, not steady throughput.
- Temporary /tmp/tmp*.mp3 and /tmp/tmp*.wav counts were zero before and after.
- The user's original source MP3 remained intact.
- Readiness plus acceptance took 73.42 seconds, excluding image build/pull time.

Tested image ID:
sha256:169b39624a2a0b15e5ede3bad228c196af4214a7ff37f52fc9b01c38544675d6

The full local machine-readable result is stored in
test_docs/test/runtime/docker-acceptance.json. The generated transcript is not a
ground-truth accuracy assertion. This test establishes service/container behavior;
it does not establish completion of the full Task 2c dataset.

## Reproduce

From the repository root, with Docker GPU support enabled:

```powershell
docker compose build asr-api
docker compose up -d asr-api
python asr/test_docker_service.py --audio PATH/real-speech.mp3
docker compose ps asr-api
```

The test uses only Python's standard library on the host. Model dependencies are
installed in the image. Its bounded readiness wait allows the first model
download/load; the named model-cache volume avoids re-downloading on restart.

An initial build failed because the host and Docker clocks were eight hours slow.
The user corrected the clock. The final Dockerfile contains no timestamp-check
override. A temporary image-layer failure during the clock correction was
resolved by pulling the base image again after time synchronization.

NeMo 2.5 requires JiWER below 4. The requirements and scorer were made compatible
with JiWER 3.1, and the metric edge-case tests pass against that version.
