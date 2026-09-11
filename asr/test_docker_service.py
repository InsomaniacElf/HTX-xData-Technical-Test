"""Task 2d acceptance test against the real running Compose ASR service.

Run from the repository root after docker compose up -d asr-api. Uses an existing
speech MP3 supplied with --audio. Does not delete the user's source audio.
"""
import argparse
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, encoding="utf-8").strip()


def temporary_audio(container):
    query = "import glob,json; print(json.dumps(sorted(glob.glob('/tmp/tmp*.mp3')+glob.glob('/tmp/tmp*.wav'))))"
    return set(json.loads(docker("exec", container, "python", "-c", query)))


def check(audio, base_url, output, startup_timeout=900):
    started = time.perf_counter()
    audio = Path(audio)
    if not audio.is_file() or audio.suffix.lower() != ".mp3":
        raise ValueError("--audio must identify an existing speech MP3")
    container = docker("compose", "ps", "-q", "asr-api")
    if not container:
        raise RuntimeError("Start the Compose service asr-api before testing")
    deadline = time.monotonic() + startup_timeout
    while True:
        if docker("inspect", "--format", "{{.State.Running}}", container) != "true":
            raise RuntimeError("ASR container exited; inspect docker compose logs asr-api")
        try:
            with urllib.request.urlopen(base_url + "/ping", timeout=5) as response:
                assert json.load(response) == "pong", "GET /ping must return pong"
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if time.monotonic() >= deadline:
                raise TimeoutError("ASR startup exceeded the bounded readiness timeout")
            time.sleep(3)
    before = temporary_audio(container)
    boundary = "htx-" + uuid.uuid4().hex
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
               f"filename=\"sample.mp3\"\r\nContent-Type: audio/mpeg\r\n\r\n").encode()
    payload += audio.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(base_url + "/asr", data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        assert response.headers.get_content_type() == "application/json"
        result = json.load(response)
    assert isinstance(result.get("transcription"), str), "transcription must be a string"
    assert result["transcription"].strip(), "Speech sample returned an empty transcription"
    assert isinstance(result.get("duration"), str), "duration must be a string"
    seconds = float(result["duration"])
    assert math.isfinite(seconds) and seconds > 0, "inference duration must be positive and finite"
    after = temporary_audio(container)
    assert after == before, f"Temporary audio changed after processing: {after - before}"
    report = {"task": "2d", "status": "passed", "ping": "pong", "response": result,
        "temporary_audio_before": len(before), "temporary_audio_after": len(after),
        "source_audio_preserved": audio.exists(),
        "container_image_id": docker("inspect", "--format", "{{.Image}}", container),
        "elapsed_seconds_including_readiness": time.perf_counter() - started}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", default="test_docs/test/runtime/docker-acceptance.json")
    args = parser.parse_args()
    check(args.audio, args.base_url.rstrip("/"), args.output)
