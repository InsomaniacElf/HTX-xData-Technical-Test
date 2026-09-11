ARG ASR_IMAGE=htx-xdata-technical-test-asr-api:latest
FROM ${ASR_IMAGE}
ENV NUMBA_CUDA_USE_NVIDIA_BINDING=0 PYTHONFAULTHANDLER=1
COPY asr-train/requirements-training.txt /app/requirements-training.txt
COPY asr-train/requirements-analysis.txt /app/requirements-analysis.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-training.txt -r /app/requirements-analysis.txt
RUN python -m pip check
COPY asr-train /app/asr-train
HEALTHCHECK NONE
ENTRYPOINT ["python", "/app/asr-train/train_parakeet.py"]
CMD ["--help"]
