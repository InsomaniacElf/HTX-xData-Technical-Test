# parakeet-tdt-0.6b-v3-ycsep

This is a modified model derived from NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
Credit: NVIDIA, the base model's authors. The base model is provided under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain that attribution
and license notice when sharing the derived model. No NVIDIA endorsement is implied.

Modification: full-parameter PyTorch/NeMo fine-tuning on a deterministic,
channel-balanced 10.046-hour non-TDK subset of the
[YCSEP static corpus](https://doi.org/10.7910/DVN/B7JRID), selected by non-TDK
validation WER at update 1,200. The pretrained tokenizer and frontend are retained.
The model is an assessment/research artifact, not a production-certified ASR service.

Download using `python download_artifacts.py --model` from the repository root.
The manifest records model SHA-256
`8860f2759272a5585b7d8034492a15c2ee10ca68ff3db75698dea5a36e903e2d`.
The three public release parts reconstruct one standard NeMo `.nemo` artifact.
Use the pinned environment in `asr-train/Dockerfile`; inference requires mono
16 kHz audio. `evaluate_tdk.py` performs the same decoding/resampling as the API.

Evaluation: all 218,011 Daily Ketchup rows were excluded from training and
validation. Full held-out normalized WER is 21.3120%, versus 24.4272% for the base.
See `results/E1-TDK/` for coverage, provenance, counts and uncertainty, and the
root `training-report.pdf` for limitations. This model still produces omissions,
repetitions and incorrect words. Language, speaker, noise and alignment shifts
can degrade performance. Transcriptions must not be treated as verified facts.

No denoising, neural quality filtering, tokenizer replacement or E2 expansion
results are part of this artifact. TDK must not be reused to select future models.
