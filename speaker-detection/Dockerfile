ARG ASR_IMAGE=htx-xdata-technical-test-asr-api:latest
FROM ${ASR_IMAGE}
COPY speaker-detection/requirements-speaker.txt /app/requirements-speaker.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt -r /app/requirements-speaker.txt
RUN python -m pip check
HEALTHCHECK NONE
COPY speaker-detection /app/speaker-detection
CMD ["python", "-c", "from pyannote.audio import Model, Inference; print('Speaker environment ready')"]
