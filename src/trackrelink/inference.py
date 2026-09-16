"""Frozen scoring and ID-only repair. Ground truth is never used here."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import expit

from .repair_features import FULL_TRACKRELINK_FEATURE_NAMES, build_tracklet_feature_bundle
from .repair_model import feasible_rows
from .tracklet_repair import merge_relabel_map, relabel_mot_rows


@dataclass(frozen=True)
class FrozenModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float

    @classmethod
    def load(cls, path: str | Path) -> 'FrozenModel':
        with np.load(path, allow_pickle=False) as data:
            if tuple(map(str, data['feature_names'])) != FULL_TRACKRELINK_FEATURE_NAMES:
                raise ValueError('Model feature names/order differ from the runtime schema')
            model = cls(np.asarray(data['scaler_mean'], dtype=np.float64),
                        np.asarray(data['scaler_scale'], dtype=np.float64),
                        np.asarray(data['coefficients'], dtype=np.float64).reshape(-1),
                        float(np.asarray(data['intercept']).reshape(-1)[0]),
                        float(np.asarray(data['probability_threshold']).reshape(-1)[0]))
        if any(v.shape != (39,) for v in (model.mean, model.scale, model.coefficients)):
            raise ValueError('Expected a 39-feature frozen model')
        if not all(np.all(np.isfinite(v)) for v in (model.mean, model.scale, model.coefficients)):
            raise ValueError('Non-finite model parameters')
        if np.any(model.scale <= 0) or not np.isfinite(model.intercept) or not 0 <= model.threshold <= 1:
            raise ValueError('Invalid model scale, intercept or threshold')
        return model

    def score(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 39 or not np.all(np.isfinite(x)):
            raise ValueError('Features must be a finite N x 39 array in model order')
        return expit(((x - self.mean) / self.scale) @ self.coefficients + self.intercept)

    def select(self, features: np.ndarray, predecessors: np.ndarray,
               successors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.score(features)
        a, b = np.asarray(predecessors), np.asarray(successors)
        if a.shape != q.shape or b.shape != q.shape:
            raise ValueError('Candidate IDs must align with feature rows')
        selected = feasible_rows(np.arange(len(q)), q, np.full(len(q), 'sequence'), a, b, self.threshold)
        return q, selected


def load_embeddings(path: str | Path) -> dict[int, np.ndarray]:
    """NPZ keys are track IDs; arrays contain time-ordered matched descriptors."""
    result = {}
    dimension = None
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            track_id = int(key)
            if track_id in result:
                raise ValueError('Duplicate numeric track ID in embedding archive')
            values = np.asarray(data[key], dtype=np.float32)
            if values.ndim != 2 or not np.all(np.isfinite(values)):
                raise ValueError(f'Invalid embedding matrix for track {track_id}')
            if len(values):
                if values.shape[1] == 0 or (dimension is not None and dimension != values.shape[1]):
                    raise ValueError('Nonempty embedding matrices must share a positive width')
                dimension = values.shape[1]
            result[track_id] = values
    return result


def repair_rows(rows: list[list[str]], embeddings: Mapping[int, np.ndarray], model: FrozenModel):
    bundle = build_tracklet_feature_bundle(rows, embeddings)
    q, selected = model.select(bundle.full_features, bundle.predecessor_ids, bundle.successor_ids)
    mapping = merge_relabel_map(list(bundle.tracklets), [bundle.candidates[int(i)] for i in selected])
    output = relabel_mot_rows(rows, mapping)
    if len(output) != len(rows) or any(a[:1] + a[2:] != b[:1] + b[2:] for a, b in zip(rows, output)):
        raise RuntimeError('Repair changed a non-ID field')
    return output, {
        'threshold': model.threshold,
        'candidate_count': len(q),
        'selected_count': len(selected),
        'row_count': len(rows),
        'only_ids_changed': True,
        'links': [{'predecessor': int(bundle.predecessor_ids[i]),
                   'successor': int(bundle.successor_ids[i]), 'score': float(q[i])} for i in selected],
    }
