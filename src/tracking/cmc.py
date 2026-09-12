"""
Camera Motion Compensation (CMC) for BoT-SORT multi-object tracking.

Compensates for background camera motion (pan, tilt, zoom, translation) by applying
affine transformations to Kalman filter state vectors, covariances, and bounding boxes.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from src.common.types import BoundingBox


def build_cmc_matrix(affine_matrix: np.ndarray) -> np.ndarray:
    """
    Constructs an 8x8 block-diagonal transformation matrix T for an 8-state Kalman Filter:
        x = [x_c, y_c, w, h, v_x, v_y, v_w, v_h]^T

    Given 2x3 affine matrix M = [A | t], where A is the 2x2 linear component (rotation/scale):
        T = [ A   0   0   0 ]
            [ 0  I_2  0   0 ]
            [ 0   0   A   0 ]
            [ 0   0   0  I_2]

    Args:
        affine_matrix: Array of shape (2, 3) representing the affine warp.

    Returns:
        Array of shape (8, 8).
    """
    M = np.asarray(affine_matrix, dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError(f"Affine matrix must have shape (2, 3), got {M.shape}")

    A = M[:2, :2]
    T = np.eye(8, dtype=np.float64)
    T[:2, :2] = A
    T[4:6, 4:6] = A
    return T


def apply_cmc_transform(
    mean: np.ndarray,
    covariance: np.ndarray,
    affine_matrix: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Camera Motion Compensation (CMC) affine transformation to an 8-state Kalman vector and covariance:
        [x_c', y_c']^T = A * [x_c, y_c]^T + t
        [v_x', v_y']^T = A * [v_x, v_y]^T
        P' = T * P * T^T

    Args:
        mean: Kalman filter state vector of shape (8,).
        covariance: Kalman filter covariance matrix of shape (8, 8).
        affine_matrix: Affine transformation matrix of shape (2, 3) or None.

    Returns:
        Tuple of (transformed_mean [8,], transformed_covariance [8, 8]).
    """
    if affine_matrix is None:
        return np.asarray(mean, dtype=np.float64).copy(), np.asarray(covariance, dtype=np.float64).copy()

    M = np.asarray(affine_matrix, dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError(f"Affine matrix must have shape (2, 3), got {M.shape}")

    A = M[:2, :2]
    t = M[:2, 2]

    m = np.asarray(mean, dtype=np.float64).copy()
    cov = np.asarray(covariance, dtype=np.float64).copy()

    # Transform position: [x_c, y_c]^T = A * [x_c, y_c]^T + t
    m[:2] = np.dot(A, m[:2]) + t

    # Transform velocity: [v_x, v_y]^T = A * [v_x, v_y]^T
    m[4:6] = np.dot(A, m[4:6])

    # Transform covariance
    T = build_cmc_matrix(M)
    cov = np.linalg.multi_dot((T, cov, T.T))

    return m, cov


def transform_box_cmc(bbox: BoundingBox, affine_matrix: np.ndarray) -> BoundingBox:
    """
    Applies affine camera motion compensation to a single bounding box center.

    Args:
        bbox: BoundingBox instance.
        affine_matrix: 2x3 affine transformation matrix.

    Returns:
        New BoundingBox with compensated center_x, center_y.
    """
    M = np.asarray(affine_matrix, dtype=np.float64)
    if M.shape != (2, 3):
        return bbox

    center = np.array([bbox.center_x, bbox.center_y], dtype=np.float64)
    new_center = np.dot(M[:2, :2], center) + M[:2, 2]

    return BoundingBox(
        center_x=float(new_center[0]),
        center_y=float(new_center[1]),
        width=bbox.width,
        height=bbox.height,
        confidence=bbox.confidence,
        class_id=bbox.class_id,
    )


def compute_camera_motion_compensation(
    curr_frame: np.ndarray,
    prev_frame: np.ndarray,
) -> np.ndarray:
    """
    Estimates 2x3 affine background camera motion between consecutive frames using
    sparse Lucas-Kanade optical flow and RANSAC.

    Falls back to identity transformation matrix if motion cannot be estimated reliably.
    """
    identity = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    if curr_frame is None or prev_frame is None:
        return identity

    if curr_frame.size == 0 or prev_frame.size == 0:
        return identity

    # Convert to grayscale
    if curr_frame.ndim == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    if prev_frame.ndim == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if curr_gray.shape != prev_gray.shape:
        return identity

    # Detect features to track in prev_gray
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
    )

    if prev_pts is None or len(prev_pts) < 4:
        return identity

    # Calculate optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    if curr_pts is None or status is None:
        return identity

    good_prev = prev_pts[status.ravel() == 1]
    good_curr = curr_pts[status.ravel() == 1]

    if len(good_prev) < 4:
        return identity

    # Estimate 2x3 affine transformation matrix with RANSAC
    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
    if M is None or inliers is None or np.sum(inliers) < 4:
        return identity

    return M.astype(np.float64)
