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