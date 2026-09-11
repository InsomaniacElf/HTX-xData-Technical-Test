"""Execute recorded-result notebooks without starting training or GPU inference."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', choices=['3a', '3b', '5'], default=['3a', '3b', '5'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if any(os.environ.get(key) == '1' for key in ['RUN_TRAINING', 'RUN_TDK_INFERENCE', 'RUN_SPEAKER_RETRIEVAL']):
        raise ValueError('Replay only: unset inference/training flags. Use the notebooks directly for new runs.')
    paths = {'3a':'asr-train/ycsep-train-3a.ipynb', '3b':'asr-train/ycsep-train-3b.ipynb',
             '5':'speaker-detection/sp-detect-5.ipynb'}
    for task in args.tasks:
        path = root / paths[task]
        notebook = nbformat.read(path, as_version=4)
        print('Executing recorded-result notebook: '+paths[task], flush=True)
        NotebookClient(notebook, timeout=1800, kernel_name='python3',
                       resources={'metadata':{'path':str(root)}}).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print('Passed: '+task, flush=True)


if __name__ == '__main__':
    main()
