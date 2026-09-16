"""Replay every candidate and ID-only output row in the two Figure 1 videos."""
from pathlib import Path
import json
import numpy as np

from trackrelink.inference import FrozenModel
from trackrelink.tracklet_repair import (build_tracklets, generate_tracklet_candidates, load_mot_rows,
                                        merge_relabel_map, mot_rows_to_arrays, relabel_mot_rows)

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = FrozenModel.load(ROOT / 'models/trackrelink_frozen.npz')
    evidence = json.loads((ROOT / 'examples/data/cases.json').read_text())
    assert model.threshold == evidence['threshold']
    count_scores = count_rows = 0
    for case in evidence['cases']:
        seq = case['sequence']
        with np.load(ROOT / f'examples/data/{seq}_features.npz', allow_pickle=False) as saved:
            q, selected = model.select(saved['x'], saved['pred'], saved['succ'])
            np.testing.assert_allclose(q, saved['frozen_score'], atol=1e-12, rtol=0)
            assert set(map(int, selected)) == set(case['accepted_indices'])
            k = case['candidate_index']
            assert (int(saved['pred'][k]), int(saved['succ'][k])) == (case['predecessor'], case['successor'])
            assert abs(float(q[k]) - case['score']) < 1e-12
            assert (k in selected) == case['accepted']
            selected_pairs = {(int(saved['pred'][i]), int(saved['succ'][i])) for i in selected}
        rows = load_mot_rows(ROOT / f'examples/data/{seq}_host.txt')
        expected = load_mot_rows(ROOT / f'examples/data/{seq}_F.txt')
        tracklets = build_tracklets(*mot_rows_to_arrays(rows))
        candidates = generate_tracklet_candidates(tracklets)
        merges = [c for c in candidates if (c.predecessor_id, c.successor_id) in selected_pairs]
        assert len(merges) == len(selected_pairs)
        output = relabel_mot_rows(rows, merge_relabel_map(tracklets, merges))
        assert len(output) == len(expected) == case['verified_rows']
        for before, after, reference in zip(rows, output, expected):
            assert before[:1] + before[2:] == after[:1] + after[2:]
            assert int(float(after[1])) == int(float(reference[1]))
            # Saved output serialization may use a different numeric precision.
            np.testing.assert_allclose(np.asarray(after, float), np.asarray(reference, float), atol=1e-8, rtol=0)
        count_scores += len(q)
        count_rows += len(rows)
        print(f'{seq}: {case["predecessor"]} -> {case["successor"]}, q={q[k]:.6f}, '
              f'{"link" if case["accepted"] else "reject"}; {len(selected)} total selected links')
    print(f'PASS: {count_scores} frozen scores and {count_rows} output rows; only IDs change.')


if __name__ == '__main__':
    main()
