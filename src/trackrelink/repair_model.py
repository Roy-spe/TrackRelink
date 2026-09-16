from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


REGULARIZATION_C_GRID = (0.01, 0.1, 1.0, 10.0)
MINIMUM_TRAINING_PRECISION = 0.95
MINIMUM_TRAINING_TRUE_MERGES = 5
MINIMUM_TRAINING_POSITIVE_VIDEOS = 3


def video_grouped_splits(
    labels: np.ndarray,
    sequences: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    sequences = np.asarray(sequences)
    if len(labels) != len(sequences):
        raise ValueError("labels and sequences must align")
    unique_sequences = np.unique(sequences)
    if len(unique_sequences) < n_splits:
        raise ValueError("There are fewer unique videos than grouped folds")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train_video_indices, test_video_indices in splitter.split(unique_sequences):
        train_videos = unique_sequences[train_video_indices]
        test_videos = unique_sequences[test_video_indices]
        result.append(
            (
                np.flatnonzero(np.isin(sequences, train_videos)),
                np.flatnonzero(np.isin(sequences, test_videos)),
            )
        )
    return result


def video_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(np.float64)


def fit_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    test_features: np.ndarray,
    c_value: float,
    random_seed: int = 2027,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=2000,
        random_state=int(random_seed),
    )
    model.fit(
        scaler.transform(train_features),
        train_labels,
        sample_weight=video_weights(train_groups),
    )
    return model.predict_proba(scaler.transform(test_features))[:, 1]


def feasible_rows(
    indices: np.ndarray,
    probabilities: np.ndarray,
    sequences: np.ndarray,
    predecessor_ids: np.ndarray,
    successor_ids: np.ndarray,
    threshold: float,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    selected: list[int] = []
    for sequence in sorted(map(str, np.unique(sequences[indices]))):
        sequence_rows = indices[
            (sequences[indices] == sequence) & (probabilities[indices] >= threshold)
        ]
        ordered = sorted(
            map(int, sequence_rows),
            key=lambda index: (
                -float(probabilities[index]),
                int(predecessor_ids[index]),
                int(successor_ids[index]),
            ),
        )
        used_predecessors: set[int] = set()
        used_successors: set[int] = set()
        for index in ordered:
            predecessor = int(predecessor_ids[index])
            successor = int(successor_ids[index])
            if predecessor in used_predecessors or successor in used_successors:
                continue
            used_predecessors.add(predecessor)
            used_successors.add(successor)
            selected.append(index)
    return np.asarray(sorted(selected), dtype=np.int64)


def selection_metrics(
    selected: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=np.int64)
    positives = int(labels[selected].sum()) if len(selected) else 0
    selected_videos = set(map(str, sequences[selected])) if len(selected) else set()
    positive_videos = (
        set(map(str, sequences[selected][labels[selected] == 1])) if len(selected) else set()
    )
    return {
        "selected": int(len(selected)),
        "true_merges": positives,
        "false_merges": int(len(selected) - positives),
        "precision": 0.0 if len(selected) == 0 else float(positives / len(selected)),
        "selected_videos": len(selected_videos),
        "positive_videos": len(positive_videos),
    }


def choose_threshold(
    fit_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    predecessor_ids: np.ndarray,
    successor_ids: np.ndarray,
    minimum_precision: float = MINIMUM_TRAINING_PRECISION,
    minimum_true_merges: int = MINIMUM_TRAINING_TRUE_MERGES,
    minimum_positive_videos: int = MINIMUM_TRAINING_POSITIVE_VIDEOS,
) -> tuple[float | None, dict[str, float | int] | None]:
    thresholds = np.unique(probabilities[fit_indices])[::-1]
    eligible: list[tuple[int, float, float, dict[str, float | int]]] = []
    for threshold in thresholds:
        selected = feasible_rows(
            fit_indices,
            probabilities,
            sequences,
            predecessor_ids,
            successor_ids,
            float(threshold),
        )
        metrics = selection_metrics(selected, labels, sequences)
        if (
            metrics["precision"] >= minimum_precision
            and metrics["true_merges"] >= minimum_true_merges
            and metrics["positive_videos"] >= minimum_positive_videos
        ):
            eligible.append(
                (
                    int(metrics["true_merges"]),
                    float(metrics["precision"]),
                    float(threshold),
                    metrics,
                )
            )
    if not eligible:
        return None, None
    _, _, threshold, metrics = max(eligible, key=lambda value: (value[0], value[1], value[2]))
    return threshold, metrics


def fit_standardized_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    c_value: float,
    random_seed: int = 2027,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=2000,
        random_state=int(random_seed),
    )
    model.fit(
        scaler.transform(features),
        labels,
        sample_weight=video_weights(sequences),
    )
    return scaler, model


def model_payload(
    scaler: StandardScaler,
    model: LogisticRegression,
    feature_names: list[str] | tuple[str, ...],
    c_value: float,
    probability_threshold: float,
) -> dict[str, Any]:
    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "classes": model.classes_,
        "feature_names": np.asarray(feature_names),
        "c": np.asarray([c_value]),
        "probability_threshold": np.asarray([probability_threshold]),
    }
