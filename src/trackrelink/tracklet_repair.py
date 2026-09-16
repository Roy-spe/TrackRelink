from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_iou


@dataclass(frozen=True)
class Tracklet:
    track_id: int
    frame_ids: np.ndarray
    boxes: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_ids.ndim != 1:
            raise ValueError("frame_ids must be one-dimensional")
        if self.boxes.shape != (len(self.frame_ids), 4):
            raise ValueError("boxes must align with frame_ids")
        if len(self.frame_ids) == 0:
            raise ValueError("tracklets cannot be empty")
        if np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("tracklet frame IDs must be strictly increasing")

    @property
    def start_frame(self) -> int:
        return int(self.frame_ids[0])

    @property
    def end_frame(self) -> int:
        return int(self.frame_ids[-1])


@dataclass(frozen=True)
class TrackletCandidate:
    predecessor_id: int
    successor_id: int
    missing_frames: int
    normalized_center_distance: float
    normalized_stationary_distance: float
    absolute_log_area_ratio: float
    absolute_log_aspect_ratio: float
    predicted_iou: float
    endpoint_iou: float
    direction_cosine: float
    normalized_velocity: float
    predecessor_length: int
    successor_length: int
    predecessor_density: float
    successor_density: float


@dataclass(frozen=True)
class TrackletIdentity:
    gt_id: int
    matched_observations: int
    purity: float
    valid: bool


TRACKLET_FEATURE_NAMES = [
    "missing_frames",
    "log1p_missing_frames",
    "normalized_center_distance",
    "normalized_stationary_distance",
    "absolute_log_area_ratio",
    "absolute_log_aspect_ratio",
    "predicted_iou",
    "endpoint_iou",
    "direction_cosine",
    "normalized_velocity",
    "log1p_predecessor_length",
    "log1p_successor_length",
    "absolute_log_length_ratio",
    "predecessor_density",
    "successor_density",
]

TRACKLET_APPEARANCE_FEATURE_NAMES = [
    "appearance_available",
    "appearance_centroid_cosine",
    "appearance_endpoint_mean_cosine",
    "appearance_endpoint_max_cosine",
    "appearance_endpoint_min_cosine",
    "appearance_endpoint_std_cosine",
    "predecessor_appearance_consistency",
    "successor_appearance_consistency",
    "log1p_predecessor_embedding_count",
    "log1p_successor_embedding_count",
]

TRACKLET_COMPETITION_FEATURE_NAMES = [
    "centroid_predecessor_margin",
    "centroid_successor_margin",
    "endpoint_predecessor_margin",
    "endpoint_successor_margin",
    "centroid_mutual_best",
    "endpoint_mutual_best",
    "log1p_predecessor_candidate_count",
    "log1p_successor_candidate_count",
]

TRACKLET_EXCLUSIVITY_FEATURE_NAMES = [
    "occupancy_available",
    "gap_occupied_fraction",
    "gap_mean_nearest_center_distance",
    "gap_min_nearest_center_distance",
    "gap_max_interpolated_iou",
    "log1p_gap_blocker_count",
    "log1p_predecessor_endpoint_neighbors",
    "log1p_successor_endpoint_neighbors",
    "motion_predecessor_margin",
    "motion_successor_margin",
    "predicted_iou_predecessor_margin",
    "predicted_iou_successor_margin",
    "motion_mutual_best",
    "predicted_iou_mutual_best",
]

TRACKLET_BIDIRECTIONAL_MOTION_FEATURE_NAMES = [
    "backward_normalized_center_distance",
    "backward_predicted_iou",
    "forward_backward_midpoint_disagreement",
    "endpoint_velocity_cosine",
    "successor_direction_cosine",
    "normalized_velocity_disagreement",
]


def load_mot_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"MOT row has fewer than six columns: {path}")
            rows.append(row)
    return rows


def mot_rows_to_arrays(
    rows: list[list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_ids: list[int] = []
    track_ids: list[int] = []
    boxes: list[list[float]] = []
    for row in rows:
        frame_id = int(float(row[0]))
        track_id = int(float(row[1]))
        x, y, width, height = (float(value) for value in row[2:6])
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Non-positive MOT box for track {track_id}")
        frame_ids.append(frame_id)
        track_ids.append(track_id)
        boxes.append([x, y, x + width, y + height])
    return (
        np.asarray(frame_ids, dtype=np.int32),
        np.asarray(track_ids, dtype=np.int32),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    )


def build_tracklets(
    frame_ids: np.ndarray,
    track_ids: np.ndarray,
    boxes: np.ndarray,
) -> list[Tracklet]:
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    track_ids = np.asarray(track_ids, dtype=np.int32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if frame_ids.ndim != 1 or track_ids.shape != frame_ids.shape:
        raise ValueError("frame_ids and track_ids must be aligned vectors")
    if boxes.shape != (len(frame_ids), 4):
        raise ValueError("boxes must have shape [rows, 4]")
    result: list[Tracklet] = []
    for track_id in sorted(map(int, np.unique(track_ids))):
        indices = np.flatnonzero(track_ids == track_id)
        order = np.argsort(frame_ids[indices], kind="stable")
        indices = indices[order]
        selected_frames = frame_ids[indices]
        if len(np.unique(selected_frames)) != len(selected_frames):
            raise ValueError(f"Track {track_id} has duplicate output rows in one frame")
        result.append(
            Tracklet(
                track_id=track_id,
                frame_ids=selected_frames.copy(),
                boxes=boxes[indices].copy(),
            )
        )
    return result


def _centers(boxes: np.ndarray) -> np.ndarray:
    return 0.5 * (boxes[:, :2] + boxes[:, 2:])


def _endpoint_velocity(tracklet: Tracklet, observations: int = 5) -> np.ndarray:
    count = min(int(observations), len(tracklet.frame_ids))
    if count < 2:
        return np.zeros(2, dtype=np.float64)
    frames = tracklet.frame_ids[-count:].astype(np.float64)
    centers = _centers(tracklet.boxes[-count:].astype(np.float64))
    frame_deltas = np.diff(frames)
    valid = frame_deltas > 0
    if not np.any(valid):
        return np.zeros(2, dtype=np.float64)
    velocities = np.diff(centers, axis=0)[valid] / frame_deltas[valid, None]
    return np.median(velocities, axis=0)


def _initial_velocity(tracklet: Tracklet, observations: int = 5) -> np.ndarray:
    count = min(int(observations), len(tracklet.frame_ids))
    if count < 2:
        return np.zeros(2, dtype=np.float64)
    frames = tracklet.frame_ids[:count].astype(np.float64)
    centers = _centers(tracklet.boxes[:count].astype(np.float64))
    frame_deltas = np.diff(frames)
    valid = frame_deltas > 0
    if not np.any(valid):
        return np.zeros(2, dtype=np.float64)
    velocities = np.diff(centers, axis=0)[valid] / frame_deltas[valid, None]
    return np.median(velocities, axis=0)


def generate_tracklet_candidates(
    tracklets: list[Tracklet],
    maximum_missing_frames: int = 60,
    maximum_normalized_center_distance: float = 4.0,
    maximum_absolute_log_area_ratio: float = 1.4,
) -> list[TrackletCandidate]:
    """Generate label-blind, temporally feasible tracklet successors."""
    if maximum_missing_frames < 0:
        raise ValueError("maximum_missing_frames cannot be negative")
    if maximum_normalized_center_distance <= 0.0:
        raise ValueError("maximum_normalized_center_distance must be positive")
    if maximum_absolute_log_area_ratio <= 0.0:
        raise ValueError("maximum_absolute_log_area_ratio must be positive")
    candidates: list[TrackletCandidate] = []
    ordered = sorted(tracklets, key=lambda value: (value.start_frame, value.track_id))
    for predecessor in ordered:
        last_box = predecessor.boxes[-1].astype(np.float64)
        last_center = _centers(last_box.reshape(1, 4))[0]
        last_wh = np.maximum(1e-6, last_box[2:] - last_box[:2])
        last_area = float(np.prod(last_wh))
        velocity = _endpoint_velocity(predecessor)
        for successor in ordered:
            if successor.track_id == predecessor.track_id:
                continue
            frame_delta = successor.start_frame - predecessor.end_frame
            if frame_delta <= 0:
                continue
            missing_frames = frame_delta - 1
            if missing_frames > maximum_missing_frames:
                continue
            first_box = successor.boxes[0].astype(np.float64)
            first_center = _centers(first_box.reshape(1, 4))[0]
            first_wh = np.maximum(1e-6, first_box[2:] - first_box[:2])
            first_area = float(np.prod(first_wh))
            predicted_center = last_center + velocity * frame_delta
            scale = math.sqrt(0.5 * (last_area + first_area))
            normalized_distance = float(
                np.linalg.norm(predicted_center - first_center) / max(scale, 1e-6)
            )
            normalized_stationary_distance = float(
                np.linalg.norm(last_center - first_center) / max(scale, 1e-6)
            )
            absolute_log_area_ratio = float(abs(math.log(first_area / last_area)))
            last_aspect = float(last_wh[0] / last_wh[1])
            first_aspect = float(first_wh[0] / first_wh[1])
            absolute_log_aspect_ratio = float(abs(math.log(first_aspect / last_aspect)))
            if normalized_distance > maximum_normalized_center_distance:
                continue
            if absolute_log_area_ratio > maximum_absolute_log_area_ratio:
                continue
            predicted_box = np.concatenate(
                [predicted_center - 0.5 * last_wh, predicted_center + 0.5 * last_wh]
            ).reshape(1, 4)
            predicted_iou = float(box_iou(predicted_box, first_box.reshape(1, 4))[0, 0])
            endpoint_iou = float(box_iou(last_box.reshape(1, 4), first_box.reshape(1, 4))[0, 0])
            displacement = first_center - last_center
            velocity_norm = float(np.linalg.norm(velocity))
            displacement_norm = float(np.linalg.norm(displacement))
            direction_cosine = (
                0.0
                if velocity_norm <= 1e-12 or displacement_norm <= 1e-12
                else float(np.dot(velocity, displacement) / (velocity_norm * displacement_norm))
            )
            predecessor_duration = predecessor.end_frame - predecessor.start_frame + 1
            successor_duration = successor.end_frame - successor.start_frame + 1
            candidates.append(
                TrackletCandidate(
                    predecessor_id=predecessor.track_id,
                    successor_id=successor.track_id,
                    missing_frames=missing_frames,
                    normalized_center_distance=normalized_distance,
                    normalized_stationary_distance=normalized_stationary_distance,
                    absolute_log_area_ratio=absolute_log_area_ratio,
                    absolute_log_aspect_ratio=absolute_log_aspect_ratio,
                    predicted_iou=predicted_iou,
                    endpoint_iou=endpoint_iou,
                    direction_cosine=direction_cosine,
                    normalized_velocity=float(velocity_norm / max(math.sqrt(last_area), 1e-6)),
                    predecessor_length=len(predecessor.frame_ids),
                    successor_length=len(successor.frame_ids),
                    predecessor_density=float(len(predecessor.frame_ids) / predecessor_duration),
                    successor_density=float(len(successor.frame_ids) / successor_duration),
                )
            )
    return sorted(
        candidates,
        key=lambda value: (
            value.predecessor_id,
            value.successor_id,
            value.missing_frames,
        ),
    )


def infer_tracklet_identities(
    tracklets: list[Tracklet],
    ground_truth: dict[int, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
    minimum_matched_observations: int = 3,
    minimum_purity: float = 0.9,
) -> dict[int, TrackletIdentity]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if minimum_matched_observations < 1:
        raise ValueError("minimum_matched_observations must be positive")
    if not 0.0 < minimum_purity <= 1.0:
        raise ValueError("minimum_purity must be in (0, 1]")
    observations_by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for tracklet in tracklets:
        for frame_id, box in zip(tracklet.frame_ids, tracklet.boxes):
            observations_by_frame[int(frame_id)].append((tracklet.track_id, box))
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for frame_id, observations in observations_by_frame.items():
        gt_boxes, gt_ids = ground_truth.get(
            frame_id,
            (np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.int32)),
        )
        if len(gt_boxes) == 0:
            continue
        output_boxes = np.asarray([value[1] for value in observations], dtype=np.float32)
        ious = box_iou(output_boxes, gt_boxes)
        output_indices, gt_indices = linear_sum_assignment(-ious)
        for output_index, gt_index in zip(output_indices, gt_indices):
            if ious[output_index, gt_index] < iou_threshold:
                continue
            track_id = int(observations[int(output_index)][0])
            counts[track_id][int(gt_ids[int(gt_index)])] += 1
    identities: dict[int, TrackletIdentity] = {}
    for tracklet in tracklets:
        identity_counts = counts.get(tracklet.track_id, Counter())
        total = int(sum(identity_counts.values()))
        if total == 0:
            identities[tracklet.track_id] = TrackletIdentity(-1, 0, 0.0, False)
            continue
        gt_id, modal_count = identity_counts.most_common(1)[0]
        purity = float(modal_count / total)
        valid = total >= minimum_matched_observations and purity >= minimum_purity
        identities[tracklet.track_id] = TrackletIdentity(
            int(gt_id) if valid else -1,
            total,
            purity,
            valid,
        )
    return identities


def select_oracle_merges(
    candidates: list[TrackletCandidate],
    identities: dict[int, TrackletIdentity],
) -> list[TrackletCandidate]:
    """Select GT-scored edges subject to one predecessor/one successor feasibility."""
    beneficial: list[TrackletCandidate] = []
    for candidate in candidates:
        predecessor = identities[candidate.predecessor_id]
        successor = identities[candidate.successor_id]
        if not predecessor.valid or not successor.valid:
            continue
        if predecessor.gt_id != successor.gt_id:
            continue
        beneficial.append(candidate)
    beneficial.sort(
        key=lambda value: (
            -min(value.predecessor_length, value.successor_length),
            value.missing_frames,
            value.normalized_center_distance,
            -value.predicted_iou,
            value.predecessor_id,
            value.successor_id,
        )
    )
    used_predecessors: set[int] = set()
    used_successors: set[int] = set()
    selected: list[TrackletCandidate] = []
    for candidate in beneficial:
        if candidate.predecessor_id in used_predecessors:
            continue
        if candidate.successor_id in used_successors:
            continue
        used_predecessors.add(candidate.predecessor_id)
        used_successors.add(candidate.successor_id)
        selected.append(candidate)
    return sorted(selected, key=lambda value: (value.predecessor_id, value.successor_id))


def merge_relabel_map(
    tracklets: list[Tracklet],
    merges: list[TrackletCandidate],
) -> dict[int, int]:
    track_ids = sorted(tracklet.track_id for tracklet in tracklets)
    parent = {track_id: track_id for track_id in track_ids}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    for merge in merges:
        left = find(merge.predecessor_id)
        right = find(merge.successor_id)
        if left != right:
            root = min(left, right)
            parent[left] = root
            parent[right] = root
    return {track_id: find(track_id) for track_id in track_ids}


def relabel_mot_rows(
    rows: list[list[str]], relabel: dict[int, int]
) -> list[list[str]]:
    result: list[list[str]] = []
    for row in rows:
        copied = list(row)
        track_id = int(float(copied[1]))
        copied[1] = str(relabel.get(track_id, track_id))
        result.append(copied)
    return result


def write_mot_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def candidate_label_counts(
    candidates: list[TrackletCandidate],
    identities: dict[int, TrackletIdentity],
) -> dict[str, int]:
    counts = {"same_identity": 0, "different_identity": 0, "invalid_identity": 0}
    for candidate in candidates:
        predecessor = identities[candidate.predecessor_id]
        successor = identities[candidate.successor_id]
        if not predecessor.valid or not successor.valid:
            counts["invalid_identity"] += 1
        elif predecessor.gt_id == successor.gt_id:
            counts["same_identity"] += 1
        else:
            counts["different_identity"] += 1
    return counts


def tracklet_candidate_features(candidate: TrackletCandidate) -> np.ndarray:
    values = np.asarray(
        [
            candidate.missing_frames,
            math.log1p(candidate.missing_frames),
            candidate.normalized_center_distance,
            candidate.normalized_stationary_distance,
            candidate.absolute_log_area_ratio,
            candidate.absolute_log_aspect_ratio,
            candidate.predicted_iou,
            candidate.endpoint_iou,
            candidate.direction_cosine,
            candidate.normalized_velocity,
            math.log1p(candidate.predecessor_length),
            math.log1p(candidate.successor_length),
            abs(math.log(candidate.successor_length / candidate.predecessor_length)),
            candidate.predecessor_density,
            candidate.successor_density,
        ],
        dtype=np.float32,
    )
    if values.shape != (len(TRACKLET_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("Tracklet candidate feature vector is invalid")
    return values


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Appearance observations must have shape [observations, dimensions]")
    if len(values) == 0:
        return values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.all(np.isfinite(values)):
        raise ValueError("Appearance observations must be finite and non-zero")
    return values / norms


def tracklet_appearance_features(
    predecessor_embeddings: np.ndarray,
    successor_embeddings: np.ndarray,
    endpoint_observations: int = 5,
) -> np.ndarray:
    """Summarize label-free ReID agreement between two ordered tracklets."""
    if endpoint_observations < 1:
        raise ValueError("endpoint_observations must be positive")
    predecessor = _normalized_rows(predecessor_embeddings)
    successor = _normalized_rows(successor_embeddings)
    predecessor_count = len(predecessor)
    successor_count = len(successor)
    if predecessor_count == 0 or successor_count == 0:
        values = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                math.log1p(predecessor_count),
                math.log1p(successor_count),
            ],
            dtype=np.float32,
        )
    else:
        predecessor_centroid = predecessor.mean(axis=0)
        successor_centroid = successor.mean(axis=0)
        predecessor_centroid /= max(np.linalg.norm(predecessor_centroid), 1e-12)
        successor_centroid /= max(np.linalg.norm(successor_centroid), 1e-12)
        endpoint_cosines = (
            predecessor[-endpoint_observations:] @ successor[:endpoint_observations].T
        ).reshape(-1)
        predecessor_consistency = float(np.mean(predecessor @ predecessor_centroid))
        successor_consistency = float(np.mean(successor @ successor_centroid))
        values = np.asarray(
            [
                1.0,
                float(predecessor_centroid @ successor_centroid),
                float(endpoint_cosines.mean()),
                float(endpoint_cosines.max()),
                float(endpoint_cosines.min()),
                float(endpoint_cosines.std()),
                predecessor_consistency,
                successor_consistency,
                math.log1p(predecessor_count),
                math.log1p(successor_count),
            ],
            dtype=np.float32,
        )
    if values.shape != (len(TRACKLET_APPEARANCE_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError("Tracklet appearance feature vector is invalid")
    return values


def tracklet_competition_features(
    appearance_scores: np.ndarray,
    predecessor_ids: np.ndarray,
    successor_ids: np.ndarray,
) -> np.ndarray:
    """Add row/column appearance margins and reciprocal-best indicators."""
    scores = np.asarray(appearance_scores, dtype=np.float64)
    predecessors = np.asarray(predecessor_ids, dtype=np.int64)
    successors = np.asarray(successor_ids, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("appearance_scores must contain centroid and endpoint columns")
    if predecessors.shape != (len(scores),) or successors.shape != (len(scores),):
        raise ValueError("candidate IDs must align with appearance scores")
    if not np.all(np.isfinite(scores)):
        raise ValueError("appearance scores must be finite")
    output = np.zeros((len(scores), len(TRACKLET_COMPETITION_FEATURE_NAMES)), dtype=np.float32)
    predecessor_groups = {
        int(value): np.flatnonzero(predecessors == value) for value in np.unique(predecessors)
    }
    successor_groups = {
        int(value): np.flatnonzero(successors == value) for value in np.unique(successors)
    }
    for cue in range(2):
        predecessor_best: dict[int, int] = {}
        successor_best: dict[int, int] = {}
        for value, indices in predecessor_groups.items():
            predecessor_best[value] = int(indices[np.argmax(scores[indices, cue])])
        for value, indices in successor_groups.items():
            successor_best[value] = int(indices[np.argmax(scores[indices, cue])])
        for index in range(len(scores)):
            predecessor_indices = predecessor_groups[int(predecessors[index])]
            successor_indices = successor_groups[int(successors[index])]
            predecessor_other = predecessor_indices[predecessor_indices != index]
            successor_other = successor_indices[successor_indices != index]
            output[index, 2 * cue] = (
                0.0
                if len(predecessor_other) == 0
                else scores[index, cue] - np.max(scores[predecessor_other, cue])
            )
            output[index, 2 * cue + 1] = (
                0.0
                if len(successor_other) == 0
                else scores[index, cue] - np.max(scores[successor_other, cue])
            )
            output[index, 4 + cue] = float(
                predecessor_best[int(predecessors[index])] == index
                and successor_best[int(successors[index])] == index
            )
    for index in range(len(scores)):
        output[index, 6] = math.log1p(len(predecessor_groups[int(predecessors[index])]))
        output[index, 7] = math.log1p(len(successor_groups[int(successors[index])]))
    return output


def tracklet_bidirectional_motion_features(
    candidate: TrackletCandidate, tracklets: list[Tracklet]
) -> np.ndarray:
    """Measure constant-velocity agreement from both candidate endpoints."""
    tracklet_by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    if candidate.predecessor_id not in tracklet_by_id or candidate.successor_id not in tracklet_by_id:
        raise ValueError("candidate endpoint is absent from tracklets")
    predecessor = tracklet_by_id[candidate.predecessor_id]
    successor = tracklet_by_id[candidate.successor_id]
    frame_delta = successor.start_frame - predecessor.end_frame
    if frame_delta <= 0:
        raise ValueError("bidirectional motion requires temporally ordered tracklets")
    predecessor_box = predecessor.boxes[-1].astype(np.float64)
    successor_box = successor.boxes[0].astype(np.float64)
    predecessor_center = _centers(predecessor_box.reshape(1, 4))[0]
    successor_center = _centers(successor_box.reshape(1, 4))[0]
    predecessor_wh = np.maximum(1e-6, predecessor_box[2:] - predecessor_box[:2])
    successor_wh = np.maximum(1e-6, successor_box[2:] - successor_box[:2])
    scale = max(
        math.sqrt(0.5 * (float(np.prod(predecessor_wh)) + float(np.prod(successor_wh)))),
        1e-6,
    )
    predecessor_velocity = _endpoint_velocity(predecessor)
    successor_velocity = _initial_velocity(successor)
    backward_center = successor_center - successor_velocity * frame_delta
    backward_distance = float(np.linalg.norm(backward_center - predecessor_center) / scale)
    backward_box = np.concatenate(
        [backward_center - 0.5 * successor_wh, backward_center + 0.5 * successor_wh]
    ).reshape(1, 4)
    backward_iou = float(box_iou(backward_box, predecessor_box.reshape(1, 4))[0, 0])
    forward_midpoint = predecessor_center + 0.5 * predecessor_velocity * frame_delta
    backward_midpoint = successor_center - 0.5 * successor_velocity * frame_delta
    midpoint_disagreement = float(np.linalg.norm(forward_midpoint - backward_midpoint) / scale)
    predecessor_speed = float(np.linalg.norm(predecessor_velocity))
    successor_speed = float(np.linalg.norm(successor_velocity))
    velocity_cosine = (
        0.0
        if predecessor_speed <= 1e-12 or successor_speed <= 1e-12
        else float(
            np.dot(predecessor_velocity, successor_velocity)
            / (predecessor_speed * successor_speed)
        )
    )
    displacement = successor_center - predecessor_center
    displacement_norm = float(np.linalg.norm(displacement))
    successor_direction_cosine = (
        0.0
        if successor_speed <= 1e-12 or displacement_norm <= 1e-12
        else float(
            np.dot(successor_velocity, displacement)
            / (successor_speed * displacement_norm)
        )
    )
    velocity_disagreement = float(
        np.linalg.norm(predecessor_velocity - successor_velocity) / scale
    )
    values = np.asarray(
        [
            backward_distance,
            backward_iou,
            midpoint_disagreement,
            velocity_cosine,
            successor_direction_cosine,
            velocity_disagreement,
        ],
        dtype=np.float32,
    )
    if values.shape != (len(TRACKLET_BIDIRECTIONAL_MOTION_FEATURE_NAMES),) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("Tracklet bidirectional motion feature vector is invalid")
    return values


def tracklet_occupancy_features(
    candidate: TrackletCandidate,
    tracklets: list[Tracklet],
    occupancy_distance_threshold: float = 1.0,
    neighbor_distance_threshold: float = 2.0,
) -> np.ndarray:
    """Describe third-party occupancy along a proposed tracklet continuation."""
    if occupancy_distance_threshold <= 0.0 or neighbor_distance_threshold <= 0.0:
        raise ValueError("occupancy and neighbor distance thresholds must be positive")
    tracklet_by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    observations_by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for tracklet in tracklets:
        for frame_id, box in zip(tracklet.frame_ids, tracklet.boxes):
            observations_by_frame[int(frame_id)].append((tracklet.track_id, box.astype(np.float64)))
    return _tracklet_occupancy_features(
        candidate,
        tracklet_by_id,
        observations_by_frame,
        occupancy_distance_threshold,
        neighbor_distance_threshold,
    )


def _tracklet_occupancy_features(
    candidate: TrackletCandidate,
    tracklet_by_id: dict[int, Tracklet],
    observations_by_frame: dict[int, list[tuple[int, np.ndarray]]],
    occupancy_distance_threshold: float,
    neighbor_distance_threshold: float,
) -> np.ndarray:
    if candidate.predecessor_id not in tracklet_by_id or candidate.successor_id not in tracklet_by_id:
        raise ValueError("candidate endpoint is absent from tracklets")
    predecessor = tracklet_by_id[candidate.predecessor_id]
    successor = tracklet_by_id[candidate.successor_id]
    excluded = {candidate.predecessor_id, candidate.successor_id}

    def endpoint_neighbors(frame_id: int, target_box: np.ndarray) -> int:
        target_center = _centers(target_box.reshape(1, 4))[0]
        target_wh = np.maximum(1e-6, target_box[2:] - target_box[:2])
        scale = max(math.sqrt(float(np.prod(target_wh))), 1e-6)
        count = 0
        for track_id, box in observations_by_frame.get(frame_id, []):
            if track_id in excluded:
                continue
            center = _centers(box.reshape(1, 4))[0]
            if float(np.linalg.norm(center - target_center) / scale) <= neighbor_distance_threshold:
                count += 1
        return count

    predecessor_neighbors = endpoint_neighbors(predecessor.end_frame, predecessor.boxes[-1])
    successor_neighbors = endpoint_neighbors(successor.start_frame, successor.boxes[0])
    gap_frames = list(range(predecessor.end_frame + 1, successor.start_frame))
    if not gap_frames:
        values = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                math.log1p(predecessor_neighbors),
                math.log1p(successor_neighbors),
            ],
            dtype=np.float32,
        )
        return values

    last_box = predecessor.boxes[-1].astype(np.float64)
    first_box = successor.boxes[0].astype(np.float64)
    frame_delta = successor.start_frame - predecessor.end_frame
    nearest_distances: list[float] = []
    maximum_ious: list[float] = []
    occupied_frames = 0
    blocker_ids: set[int] = set()
    for frame_id in gap_frames:
        fraction = (frame_id - predecessor.end_frame) / frame_delta
        interpolated = (1.0 - fraction) * last_box + fraction * first_box
        center = _centers(interpolated.reshape(1, 4))[0]
        wh = np.maximum(1e-6, interpolated[2:] - interpolated[:2])
        scale = max(math.sqrt(float(np.prod(wh))), 1e-6)
        third_party = [
            (track_id, box)
            for track_id, box in observations_by_frame.get(frame_id, [])
            if track_id not in excluded
        ]
        if not third_party:
            nearest_distances.append(8.0)
            maximum_ious.append(0.0)
            continue
        boxes = np.asarray([box for _, box in third_party], dtype=np.float64)
        distances = np.linalg.norm(_centers(boxes) - center, axis=1) / scale
        ious = box_iou(interpolated.reshape(1, 4), boxes)[0]
        nearest_distances.append(float(min(8.0, distances.min())))
        maximum_ious.append(float(ious.max()))
        blocking = (distances <= occupancy_distance_threshold) | (ious > 0.1)
        if np.any(blocking):
            occupied_frames += 1
            blocker_ids.update(
                int(third_party[index][0]) for index in np.flatnonzero(blocking)
            )
    values = np.asarray(
        [
            1.0,
            occupied_frames / len(gap_frames),
            float(np.mean(nearest_distances)),
            float(np.min(nearest_distances)),
            float(np.max(maximum_ious)),
            math.log1p(len(blocker_ids)),
            math.log1p(predecessor_neighbors),
            math.log1p(successor_neighbors),
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Tracklet occupancy feature vector is invalid")
    return values


def tracklet_exclusivity_features(
    candidates: list[TrackletCandidate], tracklets: list[Tracklet]
) -> np.ndarray:
    """Combine path occupancy with reciprocal geometry competition cues."""
    output = np.zeros(
        (len(candidates), len(TRACKLET_EXCLUSIVITY_FEATURE_NAMES)), dtype=np.float32
    )
    if not candidates:
        return output
    tracklet_by_id = {tracklet.track_id: tracklet for tracklet in tracklets}
    observations_by_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for tracklet in tracklets:
        for frame_id, box in zip(tracklet.frame_ids, tracklet.boxes):
            observations_by_frame[int(frame_id)].append((tracklet.track_id, box.astype(np.float64)))
    predecessors = np.asarray([value.predecessor_id for value in candidates], dtype=np.int64)
    successors = np.asarray([value.successor_id for value in candidates], dtype=np.int64)
    motion = np.asarray(
        [value.normalized_center_distance for value in candidates], dtype=np.float64
    )
    predicted_iou = np.asarray([value.predicted_iou for value in candidates], dtype=np.float64)
    for index, candidate in enumerate(candidates):
        output[index, :8] = _tracklet_occupancy_features(
            candidate,
            tracklet_by_id,
            observations_by_frame,
            occupancy_distance_threshold=1.0,
            neighbor_distance_threshold=2.0,
        )
    predecessor_groups = {
        int(value): np.flatnonzero(predecessors == value) for value in np.unique(predecessors)
    }
    successor_groups = {
        int(value): np.flatnonzero(successors == value) for value in np.unique(successors)
    }
    motion_predecessor_best = {
        value: int(indices[np.argmin(motion[indices])])
        for value, indices in predecessor_groups.items()
    }
    motion_successor_best = {
        value: int(indices[np.argmin(motion[indices])])
        for value, indices in successor_groups.items()
    }
    iou_predecessor_best = {
        value: int(indices[np.argmax(predicted_iou[indices])])
        for value, indices in predecessor_groups.items()
    }
    iou_successor_best = {
        value: int(indices[np.argmax(predicted_iou[indices])])
        for value, indices in successor_groups.items()
    }
    for index in range(len(candidates)):
        predecessor = int(predecessors[index])
        successor = int(successors[index])
        predecessor_other = predecessor_groups[predecessor][
            predecessor_groups[predecessor] != index
        ]
        successor_other = successor_groups[successor][successor_groups[successor] != index]
        output[index, 8] = (
            0.0
            if len(predecessor_other) == 0
            else float(np.min(motion[predecessor_other]) - motion[index])
        )
        output[index, 9] = (
            0.0
            if len(successor_other) == 0
            else float(np.min(motion[successor_other]) - motion[index])
        )
        output[index, 10] = (
            0.0
            if len(predecessor_other) == 0
            else float(predicted_iou[index] - np.max(predicted_iou[predecessor_other]))
        )
        output[index, 11] = (
            0.0
            if len(successor_other) == 0
            else float(predicted_iou[index] - np.max(predicted_iou[successor_other]))
        )
        output[index, 12] = float(
            motion_predecessor_best[predecessor] == index
            and motion_successor_best[successor] == index
        )
        output[index, 13] = float(
            iou_predecessor_best[predecessor] == index
            and iou_successor_best[successor] == index
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Tracklet exclusivity feature matrix is invalid")
    return output


def ensemble_abstention_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Return fixed lower-confidence score variants for repeated grouped models."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [candidates, ensemble members>=2]")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("ensemble probabilities must be finite values in [0, 1]")
    mean = values.mean(axis=1)
    standard_deviation = values.std(axis=1)
    return {
        "mean": mean,
        "mean_minus_half_std": mean - 0.5 * standard_deviation,
        "mean_minus_std": mean - standard_deviation,
        "minimum": values.min(axis=1),
    }
