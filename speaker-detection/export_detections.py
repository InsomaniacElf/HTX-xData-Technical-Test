"""Export threshold-based speaker predictions with explicit coverage and provenance."""
import argparse
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def video_identity(filename):
    match = re.fullmatch(r"\d{8}--([A-Za-z0-9_-]{11})--(.+)\.TextGrid", filename)
    if not match:
        raise ValueError("Unexpected YCSEP video filename")
    return match.group(1), match.group(2)


def merge_segments(segments):
    result = []
    for start, end in sorted(segments):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid output timestamp")
        if result and start <= result[-1]['end_time']:
            result[-1]['end_time'] = max(result[-1]['end_time'], end)
        else:
            result.append({'start_time': start, 'end_time': end})
    return result


def resolve_title(item):
    identifier, fallback = item
    try:
        response = requests.get('https://www.youtube.com/oembed',
            params={'url': 'https://www.youtube.com/watch?v='+identifier, 'format': 'json'}, timeout=(5, 15))
        response.raise_for_status()
        title = response.json()['title']
        if not isinstance(title, str) or not title.strip():
            raise ValueError('Invalid YouTube title')
        return identifier, {'title': title, 'source': 'youtube_oembed'}
    except (requests.RequestException, ValueError, KeyError):
        return identifier, {'title': fallback, 'source': 'dataset_filename_unverified'}


def export(scores, output, threshold, threshold_note, title_cache=None):
    if not math.isfinite(threshold) or not -1 <= threshold <= 1 or not threshold_note.strip():
        raise ValueError('A finite cosine threshold and its justification are required')
    scores, output = Path(scores), Path(output)
    rows = [json.loads(line) for line in scores.read_text(encoding='utf-8').splitlines()]
    retrieval = json.loads((scores.parent / 'retrieval-result.json').read_text())
    if len(rows) != retrieval['groups_total'] or retrieval['groups_processed'] != retrieval['groups_total']:
        raise ValueError('Full-corpus retrieval scores are required for final exports')
    if retrieval.get('retryable_failures', retrieval.get('failed_groups', 0)):
        raise ValueError('Resolve retryable retrieval failures before exporting')
    selected = [r for r in rows if r['median_similarity'] is not None and r['median_similarity'] >= threshold]
    videos = {}
    for row in selected:
        identifier, title = video_identity(row['file'])
        video = videos.setdefault(identifier, {'fallback': title, 'segments': []})
        video['segments'].extend(row['segments'])
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(title_cache) if title_cache else output / 'title-provenance.json'
    titles = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    missing = [(key, video['fallback']) for key, video in videos.items() if key not in titles]
    with ThreadPoolExecutor(max_workers=4) as pool:
        titles.update(dict(pool.map(resolve_title, missing)))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(titles, indent=2, ensure_ascii=False), encoding='utf-8')
    detections = [{'title': titles[key]['title'], 'segments': merge_segments(video['segments'])}
                  for key, video in sorted(videos.items(), key=lambda item: titles[item[0]]['title'])]
    (output / 'detected_timestamps.json').write_text(json.dumps(detections, indent=2, ensure_ascii=False), encoding='utf-8')
    unique_titles = sorted({r['title'] for r in detections})
    (output / 'detected_titles.txt').write_text(''.join(title+'\n' for title in unique_titles), encoding='utf-8')
    report = {'model': 'pyannote/embedding', 'threshold': threshold, 'threshold_note': threshold_note,
        'source_sha256': hashlib.sha256(scores.read_bytes()).hexdigest(),
        'videos_detected': len(videos), 'speaker_groups_selected': len(selected),
        'groups_searched': len(rows), 'unscorable_groups': retrieval.get('unscorable_groups'),
        'unverified_titles': [key for key in videos if titles[key]['source'] != 'youtube_oembed'],
        'boundary_source': 'YCSEP annotated speaker intervals; independently verified boundary accuracy unavailable',
        'interpretation': 'Threshold-based predictions, not a measured guarantee of complete speaker recall',
        'sensitivity_videos': {str(round(t, 3)): len({r['file'] for r in rows if r['median_similarity'] is not None and r['median_similarity'] >= t})
                               for t in [max(-1, threshold-.05), threshold, min(1, threshold+.05)]}}
    (output / 'detection-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--threshold-note', required=True)
    parser.add_argument('--title-cache')
    print(json.dumps(export(**vars(parser.parse_args())), indent=2))
