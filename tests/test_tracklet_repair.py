from __future__ import annotations

import unittest

import numpy as np

from trackrelink.tracklet_repair import (
    Tracklet,
    build_tracklets,
    ensemble_abstention_scores,
    generate_tracklet_candidates,
    infer_tracklet_identities,
    merge_relabel_map,
    relabel_mot_rows,
    select_oracle_merges,
    tracklet_appearance_features,
    tracklet_bidirectional_motion_features,
    tracklet_competition_features,
    tracklet_exclusivity_features,
    tracklet_occupancy_features,
)


def _tracklet(track_id: int, frames: list[int], x: float) -> Tracklet:
    boxes = np.asarray([[x + frame, 0, x + frame + 10, 10] for frame in frames], dtype=np.float32)
    return Tracklet(track_id, np.asarray(frames, dtype=np.int32), boxes)


class TrackletRepairTests(unittest.TestCase):
    def test_build_tracklets_sorts_rows_within_identity(self) -> None:
        tracklets = build_tracklets(
            frame_ids=np.asarray([3, 1, 2, 1]),
            track_ids=np.asarray([5, 5, 5, 8]),
            boxes=np.asarray(
                [[3, 0, 13, 10], [1, 0, 11, 10], [2, 0, 12, 10], [30, 0, 40, 10]],
                dtype=np.float32,
            ),
        )

        self.assertEqual([value.track_id for value in tracklets], [5, 8])
        np.testing.assert_array_equal(tracklets[0].frame_ids, np.asarray([1, 2, 3]))

    def test_candidate_generation_is_label_blind_and_motion_gated(self) -> None:
        predecessor = _tracklet(1, [1, 2, 3], 0)
        close_successor = _tracklet(2, [5, 6], 0)
        far_successor = _tracklet(3, [5, 6], 1000)

        candidates = generate_tracklet_candidates(
            [predecessor, close_successor, far_successor],
            maximum_missing_frames=3,
            maximum_normalized_center_distance=2.0,
        )

        self.assertEqual([(value.predecessor_id, value.successor_id) for value in candidates], [(1, 2)])
        self.assertEqual(candidates[0].missing_frames, 1)

    def test_identity_inference_and_oracle_merge_are_one_to_one(self) -> None:
        first = _tracklet(1, [1, 2], 0)
        second = _tracklet(2, [4, 5], 0)
        duplicate = _tracklet(3, [4, 5], 0.5)
        ground_truth = {
            frame: (
                np.asarray([[frame, 0, frame + 10, 10]], dtype=np.float32),
                np.asarray([7], dtype=np.int32),
            )
            for frame in [1, 2, 4, 5]
        }
        identities = infer_tracklet_identities(
            [first, second, duplicate],
            ground_truth,
            minimum_matched_observations=1,
            minimum_purity=1.0,
        )
        candidates = generate_tracklet_candidates(
            [first, second, duplicate], maximum_missing_frames=2
        )

        selected = select_oracle_merges(candidates, identities)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].predecessor_id, 1)
        self.assertIn(selected[0].successor_id, {2, 3})

    def test_merge_map_and_row_relabel_form_identity_chain(self) -> None:
        tracklets = [_tracklet(4, [1], 0), _tracklet(8, [3], 0), _tracklet(12, [5], 0)]
        candidates = generate_tracklet_candidates(tracklets, maximum_missing_frames=2)
        selected = [
            next(value for value in candidates if (value.predecessor_id, value.successor_id) == (4, 8)),
            next(value for value in candidates if (value.predecessor_id, value.successor_id) == (8, 12)),
        ]

        relabel = merge_relabel_map(tracklets, selected)
        rows = [["1", "4", "0", "0", "10", "10"], ["5", "12", "0", "0", "10", "10"]]
        repaired = relabel_mot_rows(rows, relabel)

        self.assertEqual(relabel, {4: 4, 8: 4, 12: 4})
        self.assertEqual([row[1] for row in repaired], ["4", "4"])

    def test_appearance_features_capture_endpoint_agreement(self) -> None:
        predecessor = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
        same_successor = np.asarray([[0.9, 0.1], [1.0, 0.0]], dtype=np.float32)
        different_successor = np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)

        same = tracklet_appearance_features(predecessor, same_successor)
        different = tracklet_appearance_features(predecessor, different_successor)

        self.assertEqual(same[0], 1.0)
        self.assertGreater(same[1], different[1])
        self.assertGreater(same[2], different[2])

    def test_appearance_features_expose_missing_observations(self) -> None:
        features = tracklet_appearance_features(
            np.empty((0, 2), dtype=np.float32),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[1], 0.0)
        self.assertEqual(features[-2], 0.0)

    def test_competition_features_encode_margins_and_reciprocal_best(self) -> None:
        features = tracklet_competition_features(
            appearance_scores=np.asarray([[0.9, 0.8], [0.4, 0.7], [0.6, 0.5]]),
            predecessor_ids=np.asarray([1, 1, 2]),
            successor_ids=np.asarray([3, 4, 4]),
        )

        self.assertAlmostEqual(float(features[0, 0]), 0.5)
        self.assertEqual(features[0, 4], 1.0)
        self.assertEqual(features[1, 4], 0.0)
        self.assertGreater(features[2, 1], 0.0)

    def test_ensemble_abstention_scores_penalize_disagreement(self) -> None:
        scores = ensemble_abstention_scores(
            np.asarray([[0.8, 0.8, 0.8], [0.6, 0.8, 1.0]], dtype=np.float64)
        )

        self.assertAlmostEqual(float(scores["mean"][0]), 0.8)
        self.assertAlmostEqual(float(scores["mean_minus_std"][0]), 0.8)
        self.assertLess(scores["mean_minus_std"][1], scores["mean"][1])
        self.assertAlmostEqual(float(scores["minimum"][1]), 0.6)

    def test_occupancy_features_detect_a_track_inside_the_gap(self) -> None:
        predecessor = _tracklet(1, [1, 2], 0)
        successor = _tracklet(2, [5, 6], 0)
        blocker = _tracklet(3, [3, 4], 0)
        candidate = generate_tracklet_candidates([predecessor, successor])[0]

        features = tracklet_occupancy_features(
            candidate, [predecessor, successor, blocker]
        )

        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[1], 1.0)
        self.assertAlmostEqual(float(features[5]), float(np.log1p(1)))

    def test_exclusivity_features_reward_reciprocal_geometry_best(self) -> None:
        predecessors = [_tracklet(1, [1, 2], 0), _tracklet(2, [1, 2], 30)]
        successors = [_tracklet(3, [4, 5], 0), _tracklet(4, [4, 5], 30)]
        tracklets = predecessors + successors
        candidates = generate_tracklet_candidates(
            tracklets, maximum_normalized_center_distance=10.0
        )
        features = tracklet_exclusivity_features(candidates, tracklets)
        pair_to_index = {
            (candidate.predecessor_id, candidate.successor_id): index
            for index, candidate in enumerate(candidates)
        }
        correct = pair_to_index[(1, 3)]
        crossed = pair_to_index[(1, 4)]

        self.assertGreater(features[correct, 8], 0.0)
        self.assertGreater(features[correct, 10], features[crossed, 10])
        self.assertEqual(features[correct, 12], 1.0)
        self.assertEqual(features[crossed, 12], 0.0)

    def test_bidirectional_motion_recovers_constant_velocity_continuation(self) -> None:
        predecessor = _tracklet(1, [1, 2, 3], 0)
        successor = _tracklet(2, [5, 6, 7], 0)
        candidate = generate_tracklet_candidates([predecessor, successor])[0]

        features = tracklet_bidirectional_motion_features(
            candidate, [predecessor, successor]
        )

        self.assertAlmostEqual(float(features[0]), 0.0)
        self.assertAlmostEqual(float(features[1]), 1.0)
        self.assertAlmostEqual(float(features[2]), 0.0)
        self.assertAlmostEqual(float(features[3]), 1.0)
        self.assertAlmostEqual(float(features[4]), 1.0)
        self.assertAlmostEqual(float(features[5]), 0.0)


if __name__ == "__main__":
    unittest.main()
