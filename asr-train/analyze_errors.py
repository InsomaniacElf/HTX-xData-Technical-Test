"""Descriptive post-test error slices; never use these slices for checkpoint selection."""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'asr'))
from score_transcriptions import normalize


def analyze(source, output):
    groups = defaultdict(lambda: Counter(rows=0, words=0, base_edits=0, ft_edits=0))
    outcomes = Counter()
    with Path(source).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['asr_error'] or row['asr_error_ft']:
                raise ValueError('Paired successful rows required')
            reference = normalize(row['text'])
            words = len(reference.split())
            if not words:
                raise ValueError('This descriptive analysis requires nonempty references')
            base = round(float(row['wer_base']) * words)
            ft = round(float(row['wer_ft']) * words)
            outcomes['improved' if ft < base else 'regressed' if ft > base else 'unchanged'] += 1
            duration = float(row['end_time']) - float(row['start_time'])
            bucket = '<2s' if duration < 2 else '2-5s' if duration < 5 else '5-10s' if duration < 10 else '>=10s'
            for name in ['all', 'duration:'+bucket]:
                groups[name].update(rows=1, words=words, base_edits=base, ft_edits=ft)
    slices = {key: dict(value, base_wer=value['base_edits']/value['words'],
                        ft_wer=value['ft_edits']/value['words']) for key,value in groups.items()}
    videos = Counter()
    with (Path(output)/'by-video.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            difference = int(row['ft_word_edits'])-int(row['base_word_edits'])
            videos['improved' if difference<0 else 'regressed' if difference>0 else 'unchanged'] += 1
    result = {'interpretation':'Post-test descriptive slices, not tuning criteria. Short segments and reference alignment may confound results.',
              'row_outcomes':dict(outcomes), 'video_outcomes':dict(videos), 'slices':slices}
    (Path(output)/'error-slices.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output), indent=2))
