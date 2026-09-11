"""Read exact PCM frame ranges from original YCSEP WAV objects, without a clip server."""
import io
import math
import wave
from functools import lru_cache
from urllib.parse import quote

import requests

BASE = "https://a3s.fi/swift/v1/YCSEP_v2/"


def byte_range(url, start, end):
    expected = end-start+1
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                      timeout=(10, 45), stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("Source did not honor the requested byte range")
        result = bytearray()
        for block in response.iter_content(65536):
            result.extend(block)
            if len(result) > expected:
                raise ValueError("Source exceeded requested byte range")
    if len(result) != expected:
        raise ValueError("Truncated source byte range")
    return bytes(result)


@lru_cache(maxsize=2048)
def header(filename):
    url = BASE + quote(filename.removesuffix(".TextGrid") + ".wav", safe="")
    raw = io.BytesIO(byte_range(url, 0, 4095))
    with wave.open(raw, "rb") as source:
        params = source.getparams()
        offset = raw.tell()
    if params.comptype != "NONE" or params.sampwidth not in (1, 2, 3, 4):
        raise ValueError("Expected uncompressed PCM WAV")
    return url, params, offset


def clip_wav(filename, start, end):
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("Invalid segment interval")
    url, params, offset = header(filename)
    first = math.floor(start * params.framerate)
    last = min(math.ceil(end * params.framerate), params.nframes)
    if first >= last or (end * params.framerate) > params.nframes + params.framerate * .15:
        raise ValueError("Segment is outside the source audio")
    frame_bytes = params.nchannels * params.sampwidth
    size = (last-first) * frame_bytes
    if size > 64 * 1024 * 1024:
        raise ValueError("Segment exceeds size limit")
    pcm = byte_range(url, offset+first*frame_bytes, offset+last*frame_bytes-1)
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setparams(params)
        wav.writeframes(pcm)
    return target.getvalue()
