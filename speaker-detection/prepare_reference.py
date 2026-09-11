"""Download the supplied reference and inspect pyannote speaker embeddings.

Energy screening is not speaker diarization. Reference consistency statistics
are diagnostics, not a calibrated cross-recording verification threshold.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

REFERENCE_URL = "https://www.dropbox.com/scl/fi/qz7tfnvd13jv5mtx4dveo/speaker_reference_1min.wav?rlkey=pw1flpvlitft2z0tms337rn5d&st=t44yityj&dl=1"


def prepare(output, device="cpu"):
    import librosa
    import numpy as np
    import requests
    import soundfile as sf
    import torch
    from pyannote.audio import Inference
    from embedding_model import load_embedding_model

    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "speaker_reference_1min.wav"
    if not raw.exists():
        response = requests.get(REFERENCE_URL, timeout=(10, 90))
        response.raise_for_status()
        if not response.content.startswith(b"RIFF") or len(response.content) > 32 * 1024 * 1024:
            raise ValueError("Expected a bounded WAV reference, not an HTML response")
        raw.write_bytes(response.content)
    samples, _ = librosa.load(raw, sr=16000, mono=True)
    if not 30 <= len(samples)/16000 <= 120 or not np.isfinite(samples).all():
        raise ValueError("Invalid reference duration or samples")
    sf.write(output / "reference-16k.wav", samples, 16000, subtype="PCM_16")
    model, checkpoint = load_embedding_model()
    if model is None:
        raise RuntimeError("Model access unavailable; check the named Hugging Face secret")
    inference = Inference(model, window="whole")
    inference.to(torch.device(device))
    vectors, windows = [], []
    for start in range(0, len(samples)-80000+1, 80000):
        chunk = samples[start:start+80000]
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < .005:
            continue
        value = np.asarray(inference({"waveform": torch.from_numpy(chunk).unsqueeze(0),
                                      "sample_rate": 16000})).reshape(-1)
        norm = np.linalg.norm(value)
        if not np.isfinite(value).all() or norm <= 0:
            raise ValueError("Invalid reference embedding")
        vectors.append(value/norm)
        windows.append({"start": start/16000, "end": (start+80000)/16000, "rms": rms})
    if len(vectors) < 2:
        raise ValueError("Not enough reference windows")
    vectors = np.stack(vectors)
    prototype = vectors.mean(axis=0)
    prototype /= np.linalg.norm(prototype)
    np.savez(output / "reference-embeddings.npz", windows=vectors, prototype=prototype)
    similarity = vectors @ vectors.T
    pairwise = similarity[np.triu_indices(len(vectors), 1)]
    report = {"model": "pyannote/embedding", "duration": len(samples)/16000,
        "model_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "windows": windows,
        "embedding_dimension": int(vectors.shape[1]),
        "pairwise_cosine_min": float(pairwise.min()),
        "pairwise_cosine_median": float(np.median(pairwise)),
        "elapsed_seconds": time.perf_counter()-started,
        "calibration_status": "Within-reference diagnostic only; no cross-recording threshold selected"}
    (output / "reference-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    prepare(**vars(parser.parse_args()))
