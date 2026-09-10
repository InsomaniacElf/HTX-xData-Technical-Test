FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY asr ./asr

EXPOSE 8001

CMD ["uvicorn", "asr.asr_api:app", "--host", "0.0.0.0", "--port", "8001"]
