"""Bounded PyTorch/NeMo fine-tuning; TDK never participates in model selection."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path


def resolve_manifests(train_path, validation_path, output):
    manifests, groups, reports = [], [], []
    for split, path in [("train", train_path), ("validation", validation_path)]:
        path = Path(path).resolve()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows or any(r["channel"] == "The_Daily_Ketchup_Podcast" for r in rows):
            raise ValueError("Empty split or TDK contamination")
        if any(r.get("split") != split or not .2 <= r["duration"] <= 20.2 for r in rows):
            raise ValueError("Manifest split or validated duration is invalid")
        groups.append({r["video"] for r in rows})
        for row in rows:
            audio = Path(row["audio_filepath"])
            if not audio.is_absolute():
                audio = path.parent / audio
            if not audio.is_file():
                raise FileNotFoundError(audio)
            row["audio_filepath"] = str(audio.resolve())
        manifests.append(rows)
        reports.append({"split": split, "rows": len(rows), "videos": len(groups[-1]),
                        "hours": sum(r["duration"] for r in rows)/3600,
                        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if groups[0] & groups[1]:
        raise ValueError("Source-video leakage between training and validation")
    paths = []
    for split, rows in zip(["train", "validation"], manifests):
        path = output / f"resolved-{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        paths.append(path)
    return paths, reports


def train(train_manifest, validation_manifest, output, max_steps=1200, batch_size=8,
          learning_rate=1e-5, validation_interval=200, workers=4, max_minutes=120,
          accumulation=4, early_stopping_patience=0, early_stopping_min_delta=0.0005,
          retain_optimizer_state=False):
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    if min(max_steps, batch_size, validation_interval, max_minutes, accumulation) < 1 or learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if early_stopping_patience < 0 or not math.isfinite(early_stopping_min_delta) or early_stopping_min_delta < 0:
        raise ValueError("Invalid early-stopping configuration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training-result.json").exists():
        raise FileExistsError("Use a new experiment directory; existing results must not be overwritten")
    paths, data_report = resolve_manifests(train_manifest, validation_manifest, output)
    seed_everything(2026, workers=True)
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val_wer",
        mode="min", save_top_k=1, save_last=False, save_weights_only=not retain_optimizer_state,
        filename="parakeet-{step}-{val_wer:.4f}")
    stopping = EarlyStopping(monitor="val_wer", mode="min", patience=early_stopping_patience,
        min_delta=early_stopping_min_delta, strict=True, check_finite=True,
        check_on_train_epoch_end=False, verbose=True) if early_stopping_patience else None
    callbacks = [checkpoint] + ([stopping] if stopping else [])
    logger = CSVLogger(save_dir=str(output), name="training")
    trainer = Trainer(accelerator="gpu", devices=1, precision="bf16-mixed",
        max_steps=max_steps, max_epochs=-1, max_time={"minutes": max_minutes},
        accumulate_grad_batches=accumulation, gradient_clip_val=1.0,
        log_every_n_steps=1, val_check_interval=validation_interval,
        check_val_every_n_epoch=None, callbacks=callbacks, logger=logger,
        num_sanity_val_steps=2, enable_progress_bar=False, deterministic=False)
    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.set_trainer(trainer)
    for is_training, path in zip([True, False], paths):
        cfg = OmegaConf.create({"manifest_filepath": str(path), "sample_rate": 16000,
            "batch_size": batch_size, "shuffle": is_training, "num_workers": workers,
            "pin_memory": True, "min_duration": 0.2, "max_duration": 20.2,
            "use_start_end_token": False, "is_tarred": False})
        if is_training:
            model.setup_training_data(cfg)
        else:
            model.setup_validation_data(cfg)
    model.setup_optimization(OmegaConf.create({"name": "adamw", "lr": learning_rate,
        "betas": [0.9, 0.98], "weight_decay": 0.001}))
    with open_dict(model.cfg):
        model.cfg.name = "parakeet-tdt-0.6b-v3-ycsep"
    config = {"max_steps": max_steps, "batch_size": batch_size, "accumulation": accumulation,
        "learning_rate": learning_rate, "precision": "bf16-mixed", "seed": 2026,
        "checkpoint_metric": "val_wer", "all_layers_trainable": True,
        "data": data_report, "validation_interval_microbatches": validation_interval,
        "max_minutes": max_minutes, "tokenizer": "Unchanged pretrained tokenizer",
        "tokenizer_vocab_size": model.tokenizer.vocab_size,
        "checkpoint_policy": ("Best full Lightning checkpoint including optimizer state" if retain_optimizer_state
                              else "Best weights only; optimizer restart state is not retained"),
        "early_stopping": {"patience_validation_checks": early_stopping_patience,
                           "min_delta_absolute_wer": early_stopping_min_delta},
        "package_versions": {p: importlib.metadata.version(p) for p in
            ["torch", "nemo_toolkit", "lightning", "transformers", "numba", "numba-cuda"]}}
    (output / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    OmegaConf.save(model.cfg, output / "model-config.yaml")
    baseline_started = time.perf_counter()
    baseline = trainer.validate(model, verbose=False)
    (output / "baseline-validation.json").write_text(json.dumps({"metrics": baseline,
        "elapsed_seconds": time.perf_counter()-baseline_started,
        "metric_definition": "NeMo native validation WER; final TDK comparison uses canonical scorer"}, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model)
    elapsed = time.perf_counter() - started
    if not checkpoint.best_model_path:
        raise RuntimeError("No validation-selected checkpoint; training is incomplete")
    # Only load the checkpoint created by this process in its own output directory.
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    del state
    # Exported inference artifacts must not require this machine's training paths.
    with open_dict(model.cfg):
        for split in ("train_ds", "validation_ds", "test_ds"):
            if model.cfg.get(split) is not None:
                model.cfg[split].manifest_filepath = None
    model.save_to(str(output / "parakeet-tdt-0.6b-v3-ycsep.nemo"))
    stop_reason = ("validation_plateau" if stopping and stopping.wait_count >= early_stopping_patience
                   else "max_steps" if trainer.global_step >= max_steps else "time_limit_or_trainer_stop")
    result = {"elapsed_training_seconds": elapsed, "optimizer_steps": trainer.global_step,
              "best_checkpoint": str(Path(checkpoint.best_model_path).name),
              "best_validation_wer": float(checkpoint.best_model_score),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "gpu": torch.cuda.get_device_name(), "status": "completed",
              "baseline_validation": baseline, "stop_reason": stop_reason,
              "early_stopping_wait_count": stopping.wait_count if stopping else None,
              "convergence_note": "A plateau criterion is not proof of mathematical or global convergence"}
    (output / "training-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0005)
    parser.add_argument("--retain-optimizer-state", action="store_true")
    train(**vars(parser.parse_args()))
