FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip check

RUN python -c "import nemo.collections.asr, fastapi, soundfile"

COPY asr ./asr

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/ping', timeout=3)"

CMD ["uvicorn", "asr.asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
