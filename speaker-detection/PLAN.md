# Task 5: Execution Plan and Acceptance Criteria

Status: reviewed preparation plan; no new detections claimed.

1. Validate the reference waveform; retain several clean speech excerpts and
   use pyannote/embedding to cache normalized reference vectors.
2. Retrieve candidates across all YCSEP videos using cached embeddings. ASR
   transcript filters must not exclude speaker-search inputs. Record search
   coverage, download errors and short/uncertain segments.
3. Calibrate using manually verified positive and negative segments. Reference
   excerpts from the same recording are a sanity check, not an independent
   estimate of cross-recording performance. Do not assume arbitrary YCSEP voices
   are negatives without checking. Reserve independent clips for evaluation.
4. Audit a random sample of rejected videos for missed Speaker X appearances.
   Candidate-only refinement cannot recover voices missed during retrieval.
5. Refine candidate videos on original audio with VAD and, if needed, diarization.
   Verify the source timeline and video title mapping. Keep clean single-speaker
   regions for embedding estimation, but include overlapping speech in final
   timestamps when Speaker X is active. Do not merge across unverified silence or
   another speaker just because neighboring windows have high scores.
6. Review boundary accuracy and uncertainty. Short turns and code-switching still
   count. Model outputs are estimated boundaries, not automatically exact labels.
7. Execute sp-detect-5.ipynb and export detected_titles.txt and
   detected_timestamps.json, deterministically ordered, with finite positive
   intervals in source-video seconds. No placeholder output counts as completion.

Start with pyannote/embedding as required. SortFormer is optional, introduced only
after checking speaker-count, duration and overlap constraints on candidate audio.
No new GPU jobs should compete with Task 2c until its progress is healthy.

Model reference: https://huggingface.co/pyannote/embedding
