"""Paired corpus analysis with source-video cluster bootstrap, not row independence."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asr"))
from score_transcriptions import normalize
from jiwer import process_words, process_characters


def edits(reference, hypothesis, character=False):
    if not reference:
        return len(hypothesis) if character else len(hypothesis.split())
    result = (process_characters if character else process_words)(reference, hypothesis)
    return result.substitutions + result.deletions + result.insertions


def compare(path, output, bootstrap_samples=2000):
    import numpy as np
    import matplotlib.pyplot as plt

    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(lambda: np.zeros(6, dtype=np.int64))
    examples = []
    rows = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row['channel'] != 'The_Daily_Ketchup_Podcast' or row['asr_error'] or row['asr_error_ft']:
                raise ValueError("Paired analysis requires successful TDK predictions for both models")
            ref, base, ft = [normalize(row[key]) for key in ['text', 'generated_text', 'generated_text_ft']]
            words, chars = len(ref.split()), len(ref.replace(' ', ''))
            be, fe = edits(ref, base), edits(ref, ft)
            bc, fc = edits(ref.replace(' ', ''), base.replace(' ', ''), True), edits(ref.replace(' ', ''), ft.replace(' ', ''), True)
            groups[row['file']] += [be, fe, words, bc, fc, chars]
            rows += 1
            if words >= 5 and be != fe:
                examples.append({key: row[key] for key in ['file', 'start_time', 'end_time', 'text', 'generated_text', 'generated_text_ft']}
                                | {'base_edits': be, 'ft_edits': fe, 'reference_words': words})
    if not rows:
        raise ValueError("Empty paired evaluation")
    totals = np.stack(list(groups.values()))
    be, fe, words, bc, fc, chars = totals.sum(axis=0).tolist()
    if not words or not chars:
        raise ValueError("No reference words/characters")
    rng = np.random.default_rng(2026)
    draws = totals[rng.integers(0, len(totals), size=(bootstrap_samples, len(totals)))].sum(axis=1)
    draws = draws[draws[:, 2] > 0]
    delta = (draws[:, 1]-draws[:, 0]) / draws[:, 2]
    comparison = {'scope': 'full_tdk' if rows == 218011 else 'partial_tdk', 'rows': rows,
        'videos': len(groups), 'failed_rows': 0, 'base_wer': be/words, 'ft_wer': fe/words,
        'base_cer': bc/chars, 'ft_cer': fc/chars, 'delta_wer': (fe-be)/words,
        'relative_wer_reduction': (be-fe)/be if be else None,
        'delta_wer_95pct_video_bootstrap': np.quantile(delta, [.025, .975]).tolist() if len(groups) > 1 else None,
        'valid_bootstrap_draws': len(draws),
        'bootstrap_samples': bootstrap_samples, 'seed': 2026,
        'normalization': 'unicode_nfkc_casefold_words_apostrophes_v1; CER excludes spaces',
        'bootstrap_assumption': 'Videos treated as independent clusters; recurring speakers can violate this assumption'}
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    examples.sort(key=lambda r: (r['ft_edits']-r['base_edits']) / r['reference_words'])
    (output / 'error-examples.json').write_text(json.dumps({
        'selection': 'Largest row WER decreases/increases among references with at least five words; diagnostic, not representative',
        'improvements': [r for r in examples if r['ft_edits'] < r['base_edits']][:8],
        'regressions': [r for r in reversed(examples) if r['ft_edits'] > r['base_edits']][:8]}, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output / 'by-video.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['file', 'base_word_edits', 'ft_word_edits', 'reference_words', 'base_char_edits', 'ft_char_edits', 'reference_characters'])
        writer.writerows((key, *values.tolist()) for key, values in sorted(groups.items()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for axis, metric in zip(axes, ['wer', 'cer']):
        axis.bar(['Base', 'Fine-tuned'], [comparison['base_'+metric], comparison['ft_'+metric]], color=['#477c91', '#c4654e'])
        axis.set_ylabel(metric.upper())
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output / 'paired-comparison.png', dpi=160)
    plt.close(fig)
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare(args.input, args.output, args.bootstrap_samples), indent=2))
