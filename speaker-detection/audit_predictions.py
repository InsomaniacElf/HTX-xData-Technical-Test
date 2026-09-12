"""Offline sensitivity and consistency audit; no identity labels are inferred."""
import argparse
import gzip
import json
from pathlib import Path


def audit(scores, output):
    with gzip.open(scores, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    valid = [r for r in rows if r['median_similarity'] is not None]
    valid.sort(key=lambda r:r['median_similarity'], reverse=True)
    threshold = .49
    selected = [r for r in valid if r['median_similarity'] >= threshold]
    report = {'groups':len(rows), 'unscorable':len(rows)-len(valid),
        'threshold_sweep':[], 'selected_groups':[], 'near_threshold':[],
        'interpretation':'Unlabelled robustness checks, not precision/recall or verified identity. Original outputs unchanged.'}
    for t in [.35,.40,.44,.46,.48,.49,.50,.52,.54,.56,.60]:
        matches = [r for r in valid if r['median_similarity'] >= t]
        report['threshold_sweep'].append({'threshold':t, 'groups':len(matches), 'videos':len({r['file'] for r in matches})})
    def summarize(row):
        sims = row['similarities']
        return {'file':row['file'], 'speaker':row['speaker'], 'median_similarity':row['median_similarity'],
            'similarities':sims, 'representative_rows':row['representative_rows'],
            'failed_rows':row['failed_rows'], 'clips_above_threshold':sum(s>=threshold for s in sims),
            'all_scored_clips_above_threshold':bool(sims) and min(sims)>=threshold,
            'annotated_segments':len(row['segments']), 'human_identity':'unsure'}
    report['selected_groups'] = [summarize(r) for r in selected]
    report['near_threshold'] = [summarize(r) for r in valid if .44 <= r['median_similarity'] < threshold]
    report['review_top_rejected'] = [summarize(r) for r in valid if r['median_similarity'] < threshold][:10]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--output', required=True)
    audit(**vars(parser.parse_args()))
