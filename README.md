# HTX xData Technical Test

Technical test submission for the HTX xData Data Scientist position.

## Setup and Run Instructions

### Prerequisites

- Python 3.11
- Docker Desktop with Linux containers
- NVIDIA GPU, compatible driver, and NVIDIA Container Toolkit for model inference and training
- Internet access for the initial model and artifact downloads
- Approximately 6 GB of free disk space when restoring both large deliverables

Run all commands from the repository root.

### Clone the repository

```bash
git clone https://github.com/InsomaniacElf/HTX-xData-Technical-Test.git
cd HTX-xData-Technical-Test
```

### Restore large deliverables

The complete `asr/TDK_subset.csv` and fine-tuned model are distributed as public GitHub release assets because they exceed normal Git file-size limits. `download_artifacts.py` verifies their file sizes and SHA-256 checksums using `artifacts.json`.

Restore the complete evaluation CSV:

```bash
python download_artifacts.py
```

Restore the CSV and fine-tuned model:

```bash
python download_artifacts.py --model
```

The model is restored to `asr-train/models/parakeet-tdt-0.6b-v3-ycsep.nemo`.

### Run the submitted notebooks on CPU

Restore `asr/TDK_subset.csv` first, then build the analysis image and execute all required notebooks:

```bash
docker build -f asr-train/Dockerfile.analysis -t htx-analysis .
docker run --rm -v "${PWD}:/workspace" htx-analysis python run_notebooks.py
```

Run an individual notebook task:

```bash
docker run --rm -v "${PWD}:/workspace" htx-analysis python run_notebooks.py --tasks 3a
docker run --rm -v "${PWD}:/workspace" htx-analysis python run_notebooks.py --tasks 3b
docker run --rm -v "${PWD}:/workspace" htx-analysis python run_notebooks.py --tasks 5
```

The default notebook mode reproduces the submitted analyses from recorded results. It does not repeat full model training or full-corpus inference.

### Run the ASR API

```bash
docker compose up --build asr-api
```

Check the service and transcribe an MP3 file:

```bash
curl http://localhost:8001/ping
curl -F "file=@/path/to/speech.mp3" http://localhost:8001/asr
```

Stop the service:

```bash
docker compose down
```

### Run Task 2 without Docker

Docker is the recommended setup because it includes the required system libraries and FFmpeg. For a native Linux or WSL2 installation, create an isolated Python 3.11 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the API:

```bash
uvicorn asr.asr_api:app --host 0.0.0.0 --port 8001
```

Decode all Daily Ketchup rows from YCSEP:

```bash
python asr/ycsep_decode.py --csv /path/to/YCSEP_static.csv --limit 0
```

### Run new fine-tuning, evaluation, or speaker detection

Detailed GPU execution instructions and required environment variables are available in:

- [`asr-train/README.md`](asr-train/README.md) for Tasks 3a and 3b
- [`speaker-detection/README.md`](speaker-detection/README.md) for Task 5

Build the task images after building the ASR image:

```bash
docker compose build asr-api
docker build -f asr-train/Dockerfile -t htx-training .
docker build -f speaker-detection/Dockerfile -t htx-speaker .
```

New GPU runs require the YCSEP metadata/audio inputs, model access, and explicit output paths described in the task-specific READMEs.

## Tasks

### General instructions

- Complete all tasks 1 to 7 independently.
- Submit all required deliverables within 7 calendar days of receiving the test.
- You may make reasonable assumptions where information is not specified, but state those assumptions clearly in each task.
- You may reference external technical literature where needed.
- Submit deliverables to the sender of the test via email.
- All submitted artifacts must run without errors; if a Python environment setup is needed, include the setup scripts.
- Code should be properly documented and commented.

## Repository setup

### Task 1: Create the main repository

- Create a public Git repository on a platform such as GitHub or GitLab.
- Add `requirements.txt` for Python dependencies.
- Add `.gitignore`.
- Add `README.md` with setup and run instructions.

### Deliverables

- Public repository URL
- `requirements.txt`
- `.gitignore`
- `README.md`

## ASR deployment

### Task 2: Build the `asr` module

Create a directory named `asr` in the repository. Keep all code for this task inside it.

#### 2.1 Create health-check API

- Implement a GET endpoint:
  - `http://localhost:8001/ping`
- Response:
  - `"pong"`

#### 2.2 Create ASR inference API

- Use NVIDIA Parakeet ASR model:
  - [Parakeet model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- Create file:
  - `asr_api.py`
- Implement endpoint:
  - `http://localhost:8001/asr`
- Input:
  - `multipart/form-data`
  - `file`: binary of an MP3 audio file
- Output JSON:
  - `transcription`: transcribed text
  - `duration`: transcription time in seconds

#### 2.3 Decode YCSEP subset

- Use dataset:
  - YCSEP: Youtube Corpus of Singapore English Podcasts
  - [YCSEP dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/B7JRID)
- Create file:
  - `ycsep_decode.py`
- Steps:
  - Read `YCSEP_static.csv`
  - Filter rows from **The Daily Ketchup** channel
  - Create subset file named `TDK_subset.csv`
  - Call the ASR API from Task 2.2 on all audio files listed in the `audio` column
  - Save predictions into a new column:
    - `generated_text`
  - Save the updated CSV in the same folder
  - Compare `generated_text` against the reference transcription in `YCSEP_static.csv`

#### 2.4 Dockerize the ASR API

- Containerize `asr_api.py`
- Create `Dockerfile`
- Use service name:
  - `asr-api`
- After each file is processed successfully, delete the uploaded file.

### Deliverables

- `asr/asr_api.py`
- `asr/ycsep_decode.py`
- `asr/TDK_subset.csv`
- `asr/Dockerfile`

### Run Task 2 locally

Docker is the recommended reproducible setup, including system libraries and FFmpeg. For a native installation, use Python 3.11 on Linux/WSL2 with FFmpeg, libsndfile and an appropriate NVIDIA driver available. Create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the ASR API:

```bash
uvicorn asr.asr_api:app --host 0.0.0.0 --port 8001
```

Check the service:

```bash
curl http://localhost:8001/ping
```

Decode The Daily Ketchup clips from YCSEP:

```bash
python asr/ycsep_decode.py --csv /path/to/YCSEP_static.csv --limit 0
```

Run the Dockerized ASR API with service name `asr-api`:

```bash
docker compose up --build asr-api
```

Docker requires NVIDIA GPU support. Dependencies are pinned in `requirements.txt` and installed inside the image; no host virtual environment is needed for Docker. The first startup downloads Parakeet into a persistent model-cache volume.

Check the running container using a real MP3:

```bash
curl http://localhost:8001/ping
curl -F "file=@/path/to/speech.mp3" http://localhost:8001/asr
```

Local GPU acceptance passed on 11 September 2026: ping and multipart inference returned HTTP 200, the response fields were strings, and temporary MP3/WAV file counts were zero before and after the request. Verification scripts and logs are kept locally rather than included as submission deliverables.

## ASR fine-tuning

### Task 3: Build the `asr-train` module

Create a directory named `asr-train` in the repository. Keep all code for this task inside it.

#### 3.1 Fine-tune the Parakeet model

- Create notebook:
  - `ycsep-train-3a.ipynb`
- Use:
  - TensorFlow or PyTorch
- Fine-tune the Parakeet model on YCSEP.
- Exclude **all clips from The Daily Ketchup** from training and reserve them for evaluation.
- In the notebook, explain:
  - preprocessing
  - tokenizer
  - feature extraction
  - training pipeline
  - hyperparameters
- Visualize:
  - training metrics
  - validation metrics
- Interpret the visualizations.
- Rename the fine-tuned model:
  - `parakeet-tdt-0.6b-v3-ycsep`

#### 3.2 Evaluate the fine-tuned model

- Create notebook:
  - `ycsep-train-3b.ipynb`
- Use the fine-tuned model:
  - `parakeet-tdt-0.6b-v3-ycsep`
- Transcribe the audio clips in `TDK_subset.csv`
- Save predictions in a new column:
  - `generated_text_ft`
- Compare `generated_text_ft` against ground truth
- Report overall performance using a metric of your choice

### Deliverables

- `asr-train/ycsep-train-3a.ipynb`
- `asr-train/ycsep-train-3b.ipynb`
- Fine-tuned model artifacts
- Updated evaluation outputs

## Analysis reports

### Task 4: Compare base vs fine-tuned model

- Compare:
  - `generated_text` from Task 2c
  - `generated_text_ft` from Task 3b
- Describe key observations.
- Propose steps to improve fine-tuning results.
- Include suggested datasets and experiments.
- Save report as:
  - `training-report.pdf`

## Speaker diarization

### Task 5: Speaker detection

Create a directory named `speaker-detection` in the repository. Keep all code for this task inside it.

- Use reference clip for Speaker X:
  - [Speaker reference clip](https://www.dropbox.com/scl/fi/qz7tfnvd13jv5mtx4dveo/speaker_reference_1min.wav?rlkey=pw1flpvlitft2z0tms337rn5d&st=t44yityj&dl=0)
- Use speaker embedding model:
  - [pyannote embedding](https://huggingface.co/pyannote/embedding)

#### 5.1 Identify matching videos

- Create notebook:
  - `sp-detect-5.ipynb`
- Use the speaker embedding model to find all YCSEP YouTube videos containing Speaker X.
- Save the list of matching YouTube titles into:
  - `detected_titles.txt`

#### 5.2 Extract Speaker X segments

- From the identified videos, extract all exact segments where Speaker X is speaking.
- Save timestamps in:
  - `detected_timestamps.json`
- Store timestamps in ascending order using the specified JSON format.

### Deliverables

- `speaker-detection/sp-detect-5.ipynb`
- `speaker-detection/detected_titles.txt`
- `speaker-detection/detected_timestamps.json`

## Essay and submission

### Task 6: Write speech-to-speech essay

- Write an essay of no more than 700 words.
- Topic:
  - Design of a conversational speech-to-speech pipeline
- Include:
  - chosen design
  - possible challenges
  - ways to address those challenges
- Add references/materials used; these do not count toward the 700-word limit.
- Save as:
  - `essay-s2s.pdf`

### Task 7: Final submission

- Submit the Git repository URL via email.
- You may remove your repository after receiving confirmation that your submission has been received and reviewed.
