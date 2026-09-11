"""Validate a completed baseline shard before evaluating the frozen model."""
import argparse
import json
from pathlib import Path
from evaluate_tdk import evaluate, load_base


def run(base_folder, cache, model, output, original_index):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Dedicated GPU evaluation requires CUDA')
    folder = Path(base_folder)
    base = folder / f'TDK_subset_shard_{original_index:02d}_of_08.csv'
    if not (folder / 'job_result.json').is_file():
        raise ValueError('Task 2c completion record is missing')
    _, rows = load_base(base, 0, 1)
    expected = len(range(original_index, 218011, 8))
    if len(rows) != expected:
        raise ValueError(f'Incomplete baseline shard: {len(rows)} != {expected}')
    result = evaluate(base, model, output, batch_size=8, workers=8, cache=cache)
    result.update(original_shard_index=original_index, original_num_shards=8,
                  gpu=torch.cuda.get_device_name(), baseline_directory=str(folder))
    (Path(output) / 'shard-provenance.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['base-folder', 'cache', 'model', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--original-index', type=int, choices=range(8), required=True)
    run(**vars(parser.parse_args()))
