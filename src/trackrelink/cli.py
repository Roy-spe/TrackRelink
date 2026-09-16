"""Apply the frozen repair to one MOT-format sequence."""
import argparse
import json
from pathlib import Path

from .inference import FrozenModel, load_embeddings, repair_rows
from .tracklet_repair import load_mot_rows, write_mot_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, required=True, help='Input MOT-format CSV/text')
    parser.add_argument('--embeddings', type=Path, required=True, help='NPZ keyed by host track ID')
    parser.add_argument('--model', type=Path, required=True, help='Frozen model NPZ')
    parser.add_argument('--output', type=Path, required=True, help='New output MOT file')
    args = parser.parse_args()
    if args.output.resolve() in {args.tracks.resolve(), args.embeddings.resolve(), args.model.resolve()}:
        parser.error('Output must be different from every input')
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    output, report = repair_rows(load_mot_rows(args.tracks), load_embeddings(args.embeddings), FrozenModel.load(args.model))
    write_mot_rows(args.output, output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
