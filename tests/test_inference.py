import tempfile
import unittest
from pathlib import Path
import numpy as np

from trackrelink.inference import FrozenModel, load_embeddings, repair_rows

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def test_frozen_model_empty_sequence_preserves_empty_output(self):
        output, report = repair_rows([], {}, FrozenModel.load(ROOT / 'models/trackrelink_frozen.npz'))
        self.assertEqual(output, [])
        self.assertEqual(report['selected_count'], 0)

    def test_no_candidate_preserves_all_non_id_fields(self):
        rows = [['1', '7', '10.00', '20.00', '5.0', '6.0', '.900', '-1', '-1', '-1']]
        output, report = repair_rows(rows, {}, FrozenModel.load(ROOT / 'models/trackrelink_frozen.npz'))
        self.assertEqual(output, rows)
        self.assertEqual(report['candidate_count'], 0)

    def test_malformed_embedding_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'embeddings.npz'
            np.savez(path, **{'1': np.ones((2, 4)), '2': np.ones((2, 5))})
            with self.assertRaises(ValueError):
                load_embeddings(path)

    def test_nonfinite_features_are_rejected(self):
        model = FrozenModel.load(ROOT / 'models/trackrelink_frozen.npz')
        with self.assertRaises(ValueError):
            model.score(np.full((1, 39), np.nan))
