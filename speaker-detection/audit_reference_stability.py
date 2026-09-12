"""Re-score cached embeddings against reference ablations without GPU inference."""
import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np


def audit(reference, database, scores, config, output):
    reference, database = Path(reference), Path(database)
    provenance = json.loads(Path(config).read_text())
    if hashlib.sha256(reference.read_bytes()).hexdigest() != provenance['reference_sha256']:
        raise ValueError('Reference does not match retrieval provenance')
    with gzip.open(scores, 'rt', encoding='utf-8') as handle:
        groups = [json.loads(line) for line in handle if line.strip()]
    with np.load(reference, allow_pickle=False) as data:
        windows, original = data['windows'], data['prototype']
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        vectors = {row:np.frombuffer(vector,dtype='float32').copy()
                   for row,vector,error in db.execute('SELECT row_id, vector, error FROM embeddings')
                   if vector is not None and not error}
    usable, matrices = [], []
    for group in groups:
        matrix = [vectors[i] for i in group['representative_rows'] if i in vectors]
        if group['median_similarity'] is None:
            continue
        if not matrix:
            raise ValueError('Cached embedding coverage incomplete')
        matrix = np.stack(matrix)
        if not np.isclose(np.median(matrix @ original), group['median_similarity'], atol=1e-5):
            raise ValueError('Cached embedding scores do not reproduce the published baseline')
        usable.append(group)
        matrices.append(matrix)
    selected = {i for i,g in enumerate(usable) if g['median_similarity'] >= .49}
    variants = [('first_half',windows[:len(windows)//2]),('second_half',windows[len(windows)//2:])]
    variants += [('omit_window_'+str(i),np.delete(windows,i,axis=0)) for i in range(len(windows))]
    results = []
    for name, subset in variants:
        prototype = subset.mean(axis=0)
        prototype /= np.linalg.norm(prototype)
        values = [float(np.median(matrix @ prototype)) for matrix in matrices]
        ranking = sorted(range(len(values)),key=lambda i:values[i],reverse=True)
        results.append({'variant':name, 'original_selected_ranks':[ranking.index(i)+1 for i in sorted(selected)],
            'top_three':[{'file':usable[i]['file'],'speaker':usable[i]['speaker'],'score':values[i]} for i in ranking[:3]]})
    report = {'verified_groups':len(usable), 'reference_windows':len(windows), 'variants':results,
        'interpretation':'Rank stability only. A changed reference needs its own threshold calibration. No identity labels or boundary verification.'}
    Path(output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'verified_groups':len(usable),'ranks':{v['variant']:v['original_selected_ranks'] for v in results}},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['reference','database','scores','config','output']:
        parser.add_argument('--'+name,required=True)
    audit(**vars(parser.parse_args()))
