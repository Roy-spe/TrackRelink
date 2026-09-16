"""Bounding-box intersection over union."""
import numpy as np


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 4:
        raise ValueError("first boxes must have shape [N, 4]")
    if second.ndim != 2 or second.shape[1] != 4:
        raise ValueError("second boxes must have shape [M, 4]")
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    first_area = np.maximum(0.0, first[:, 2] - first[:, 0]) * np.maximum(
        0.0, first[:, 3] - first[:, 1]
    )
    second_area = np.maximum(0.0, second[:, 2] - second[:, 0]) * np.maximum(
        0.0, second[:, 3] - second[:, 1]
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
