from __future__ import annotations

import unittest

import numpy as np

from trackrelink.repair_model import feasible_rows, video_grouped_splits, video_weights


class RepairModelTests(unittest.TestCase):
    def test_grouped_splits_never_mix_video_rows(self) -> None:
        sequences = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
        for train, heldout in video_grouped_splits(labels, sequences, 4, 2027):
            self.assertFalse(set(sequences[train]) & set(sequences[heldout]))

    def test_video_weights_give_every_video_equal_total_weight(self) -> None:
        groups = np.asarray(["a", "a", "a", "b"])
        weights = video_weights(groups)
        self.assertAlmostEqual(float(weights[groups == "a"].sum()), 1.0)
        self.assertAlmostEqual(float(weights[groups == "b"].sum()), 1.0)

    def test_feasible_rows_enforces_reciprocal_one_to_one_action(self) -> None:
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.95])
        sequences = np.asarray(["a", "a", "a", "b"])
        predecessors = np.asarray([1, 1, 2, 1])
        successors = np.asarray([3, 4, 3, 3])
        selected = feasible_rows(
            np.arange(4), probabilities, sequences, predecessors, successors, 0.5
        )
        np.testing.assert_array_equal(selected, np.asarray([0, 3]))


if __name__ == "__main__":
    unittest.main()
