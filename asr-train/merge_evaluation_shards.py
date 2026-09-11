"""Merge pre-partitioned FT shards only after exact baseline/provenance checks."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

EXPECTED_ROWS = 218011
SHARDS = 8

def merge(base, base_shards, ft_shards, model, output):
    base, output = Path(base), Path(output)
    with Path(model).open('rb') as handle:
        model_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    with base.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, original = reader.fieldnames, list(reader)
    if len(original) != EXPECTED_ROWS or len(base_shards) != SHARDS or len(ft_shards) != SHARDS:
        raise ValueError('Full 218,011-row baseline and eight shard pairs are required')
    if any(row.get('channel') != 'The_Daily_Ketchup_Podcast' or row.get('asr_error') != '' for row in original):
        raise ValueError('Baseline must contain only successful TDK rows')
    if output.resolve() == base.resolve():
        raise ValueError('Do not overwrite the raw baseline')
    found, reports = {}, []
    for base_path, ft_path in zip(map(Path, base_shards), map(Path, ft_shards)):
        result = json.loads((ft_path.parent/'shard-provenance.json').read_text())
        index = result['original_shard_index']
        if index not in range(SHARDS) or index in found or result['original_num_shards'] != SHARDS:
            raise ValueError('Duplicate or invalid original shard identity')
        if result['failed'] or result['model_sha256'] != model_hash or result['audio'] != 'verified_baseline_cache':
            raise ValueError('Failed evaluation, wrong checkpoint or unmatched audio')
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != result['source_csv_sha256']:
            raise ValueError('Baseline artifact hash mismatch')
        with ft_path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        expected = original[index::SHARDS]
        if len(rows) != len(expected) or result['rows'] != len(expected):
            raise ValueError('Incomplete FT shard')
        for source, row in zip(expected, rows):
            if any(row.get(key) != source[key] for key in fields):
                raise ValueError('FT source row/order differs from baseline')
            if row.get('asr_error_ft') != '' or 'generated_text_ft' not in row:
                raise ValueError('Unresolved FT prediction')
        found[index] = rows
        reports.append({'index':index,'rows':len(rows),'csv_sha256':hashlib.sha256(ft_path.read_bytes()).hexdigest()})
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields+['generated_text_ft','asr_error_ft'])
        writer.writeheader()
        for position in range(len(original)):
            writer.writerow(found[position%SHARDS][position//SHARDS])
    os.replace(temporary,output)
    report={'rows':len(original),'failed':0,'model_sha256':model_hash,'shards':reports,
            'audio':'verified_baseline_cache','output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'baseline_sha256':hashlib.sha256(base.read_bytes()).hexdigest()}
    output.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['base','model','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--base-shards',nargs=8,required=True)
    parser.add_argument('--ft-shards',nargs=8,required=True)
    print(json.dumps(merge(**vars(parser.parse_args())),indent=2))
