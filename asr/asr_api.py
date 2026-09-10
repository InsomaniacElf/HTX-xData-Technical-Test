from fastapi import FastAPI, File, UploadFile
import time
import tempfile
import os

import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr

app = FastAPI()

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v3"
)

@app.get("/ping")
def ping():
    return "pong"

@app.post("/asr")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        input_path = tmp.name
        tmp.write(await file.read())

    try:
        # Convert MP3 → 16 kHz mono WAV
        audio, _ = librosa.load(
            input_path,
            sr=16000,
            mono=True
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name

        sf.write(wav_path, audio, 16000)

        # Measure ONLY Parakeet transcription time
        start = time.perf_counter()

        result = asr_model.transcribe([wav_path])

        duration = time.perf_counter() - start

        transcription = result[0].text if hasattr(result[0], "text") else str(result[0])

        return {
            "transcription": transcription,
            "duration": f"{duration:.6f}"
        }

    finally:
        # Delete temporary files
        if os.path.exists(input_path):
            os.remove(input_path)

        if "wav_path" in locals() and os.path.exists(wav_path):
            os.remove(wav_path)
