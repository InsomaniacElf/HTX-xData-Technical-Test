"""Run the same notebook training function on a packed, validated audio dataset."""
import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from train_parakeet import train


def run(args):
    with tempfile.TemporaryDirectory(prefix="ycsep-training-") as folder:
        root = Path(folder)
        with tarfile.open(args.archive, "r:*") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                allowed = member.name in {"train.jsonl", "validation.jsonl"} or (
                    len(path.parts) == 2 and path.parts[0] == "audio" and path.suffix == ".wav")
                if not member.isfile() or not allowed or ".." in path.parts or "\\" in member.name:
                    raise ValueError("Unexpected training archive member")
                if member.size > 64 * 1024 * 1024:
                    raise ValueError("Oversized training archive member")
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    shutil.copyfileobj(archive.extractfile(member), handle)
        return train(root / "train.jsonl", root / "validation.jsonl", args.output,
                     max_steps=args.max_steps, max_minutes=args.max_minutes, batch_size=args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    run(parser.parse_args())
