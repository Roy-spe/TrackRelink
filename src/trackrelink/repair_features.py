from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from trackrelink.tracklet_repair import (
    TRACKLET_APPEARANCE_FEATURE_NAMES,
    TRACKLET_BIDIRECTIONAL_MOTION_FEATURE_NAMES,
    TRACKLET_COMPETITION_FEATURE_NAMES,
    TRACKLET_FEATURE_NAMES,
    Tracklet,
    TrackletCandidate,
    build_tracklets,
    generate_tracklet_candidates,
    mot_rows_to_arrays,
    tracklet_appearance_features,
    tracklet_bidirectional_motion_features,
    tracklet_candidate_features,
    tracklet_competition_features,
)


FULL_TRACKRELINK_FEATURE_NAMES = tuple(
    list(TRACKLET_FEATURE_NAMES)
    + list(TRACKLET_APPEARANCE_FEATURE_NAMES)
    + list(TRACKLET_COMPETITION_FEATURE_NAMES)
    + list(TRACKLET_BIDIRECTIONAL_MOTION_FEATURE_NAMES)
)


@dataclass(frozen=True)
class TrackletFeatureBundle:
    tracklets: tuple[Tracklet, ...]
    candidates: tuple[TrackletCandidate, ...]
    predecessor_ids: np.ndarray
    successor_ids: np.ndarray
    geometry: np.ndarray
    appearance: np.ndarray
    competition: np.ndarray
    bidirectional_motion: np.ndarray
    full_features: np.ndarray


def build_tracklet_feature_bundle(
    rows: list[list[str]],
    track_embeddings: Mapping[int, np.ndarray],
) -> TrackletFeatureBundle:
    tracklets = build_tracklets(*mot_rows_to_arrays(rows))
    candidates = generate_tracklet_candidates(tracklets)
    predecessor_ids = np.asarray(
        [candidate.predecessor_id for candidate in candidates], dtype=np.int64
    )
    successor_ids = np.asarray(
        [candidate.successor_id for candidate in candidates], dtype=np.int64
    )
    geometry = (
        np.stack([tracklet_candidate_features(candidate) for candidate in candidates])
        if candidates
        else np.empty((0, len(TRACKLET_FEATURE_NAMES)), dtype=np.float32)
    )
    appearance = (
        np.stack(
            [
                tracklet_appearance_features(
                    track_embeddings.get(
                        candidate.predecessor_id,
                        np.empty((0, 0), dtype=np.float32),
                    ),
                    track_embeddings.get(
                        candidate.successor_id,
                        np.empty((0, 0), dtype=np.float32),
                    ),
                )
                for candidate in candidates
            ]
        )
        if candidates
        else np.empty((0, len(TRACKLET_APPEARANCE_FEATURE_NAMES)), dtype=np.float32)
    )
    competition = tracklet_competition_features(
        appearance[:, [1, 2]], predecessor_ids, successor_ids
    )
    bidirectional_motion = (
        np.stack(
            [tracklet_bidirectional_motion_features(candidate, tracklets) for candidate in candidates]
        )
        if candidates
        else np.empty(
            (0, len(TRACKLET_BIDIRECTIONAL_MOTION_FEATURE_NAMES)), dtype=np.float32
        )
    )
    full = np.column_stack(
        [geometry, appearance, competition, bidirectional_motion]
    ).astype(np.float32, copy=False)
    if full.shape != (len(candidates), len(FULL_TRACKRELINK_FEATURE_NAMES)):
        raise RuntimeError("TrackRelink feature bundle has an invalid shape")
    if not np.all(np.isfinite(full)):
        raise RuntimeError("TrackRelink feature bundle contains non-finite values")
    return TrackletFeatureBundle(
        tracklets=tuple(tracklets),
        candidates=tuple(candidates),
        predecessor_ids=predecessor_ids,
        successor_ids=successor_ids,
        geometry=geometry,
        appearance=appearance,
        competition=competition,
        bidirectional_motion=bidirectional_motion,
        full_features=full,
    )
