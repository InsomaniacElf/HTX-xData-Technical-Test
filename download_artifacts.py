"""Restore large submission deliverables from the public GitHub release.

Uses only the Python standard library. Downloads are verified before replacement.
The default restores asr/TDK_subset.csv; --model also restores the NeMo model.
"""
import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(part, temporary):
    """Use bounded ranges so interrupted multi-GB transfers resume safely."""
    expected = part['bytes']
    if temporary.exists() and temporary.stat().st_size > expected:
        raise ValueError('Partial download exceeds the expected artifact size')
    last_report = -1
    while not temporary.exists() or temporary.stat().st_size < expected:
        start = temporary.stat().st_size if temporary.exists() else 0
        end = min(start + 32 * 1024 * 1024, expected) - 1
        request = urllib.request.Request(part['url'], headers={
            'User-Agent': 'HTX-artifact-setup', 'Range': f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{expected}':
                        raise ValueError('Artifact server did not honor the exact requested byte range')
                    # Stage a complete range before appending, preserving a resumable prefix.
                    block = response.read(end-start+2)
                    if len(block) != end-start+1:
                        raise ValueError('Incomplete or oversized artifact range')
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        with temporary.open('ab') as output:
            output.write(block)
        percent = int(100 * (end+1) / expected)
        if percent // 10 != last_report:
            print(f'Downloading {part["name"]}: {percent}%', flush=True)
            last_report = percent // 10


def restore(root, artifact, cache):
    target = root / artifact['path']
    if target.exists() and sha256(target) == artifact['sha256']:
        print(f'Already verified: {artifact["path"]}', flush=True)
        return
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for part in artifact['parts']:
        path = cache / part['name']
        if not path.exists() or sha256(path) != part['sha256']:
            temporary = path.with_suffix(path.suffix + '.download')
            download(part, temporary)
            if sha256(temporary) != part['sha256']:
                raise ValueError(f'Checksum mismatch: {part["name"]}')
            temporary.replace(path)
        parts.append(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.restore')
    with temporary.open('wb') as output:
        for part in parts:
            opener = gzip.open if artifact['encoding'] == 'gzip' else Path.open
            with opener(part, 'rb') as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    if temporary.stat().st_size != artifact['bytes'] or sha256(temporary) != artifact['sha256']:
        raise ValueError(f'Restored artifact does not match manifest: {artifact["path"]}')
    temporary.replace(target)
    print(f'Restored and verified: {artifact["path"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='store_true', help='Also download the 2.51 GB selected model')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'artifacts.json').read_text(encoding='utf-8'))
    for artifact in manifest['artifacts']:
        if artifact['kind'] == 'dataset' or args.model:
            restore(root, artifact, root / 'test_docs/artifact-cache')
