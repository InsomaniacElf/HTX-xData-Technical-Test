"""Recover only hash-verified, manifest-committed MP3s from an interrupted archive."""
import hashlib
import json
import re
import tarfile
from pathlib import Path


def recover(folder, destination):
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    lines = (folder / 'sources.jsonl').read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index == len(lines)-1:
                break
            raise
        if not row['error']:
            if not re.fullmatch(r'[a-f0-9]{64}\.mp3', row['cache_name']):
                raise ValueError('Unsafe cache filename')
            records.append(row)
    expected = {r['cache_name']: r for r in records}
    verified = set()
    with tarfile.open(folder / 'audio.tar', 'r|') as archive:
        try:
            for member in archive:
                if not member.isfile() or member.size > 64*1024*1024 or not re.fullmatch(r'[a-f0-9]{64}\.mp3', member.name):
                    raise ValueError('Unsafe cache archive entry')
                record = expected.get(member.name)
                if not record:
                    continue
                payload = archive.extractfile(member).read()
                if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
                    raise ValueError('Cache payload integrity mismatch')
                (destination / member.name).write_bytes(payload)
                verified.add(member.name)
        except (tarfile.ReadError, EOFError):
            # An interrupted final member is not eligible; earlier verified members survive.
            pass
    return [r for r in records if r['cache_name'] in verified]
