"""
Dynamic planar homography estimation and camera motion compensation (CMC) for FRC 2026 REBUILT.

Implements:
- Direct Linear Transform (DLT) normalized via Hartley normalization and SVD.
- Non-linear Levenberg-Marquardt optimization minimizing geometric reprojection error.
- 2D pixel coordinate to 2D field coordinate (x, y in inches) projection.
- Sparse optical flow Camera Motion Compensation (CMC) for background affine estimation.
- Broadcast camera cut detection triggered by residual divergence or tracking breakdown.
- HomographyActor adhering to HomographyActorBase and HomographyActorProtocol.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from src.common.interfaces import HomographyActorBase, HomographyActorProtocol
from src.common.types import (
    HomographyEstimate,
    RawDecodedFrame,
    TagDetectionResult,
)
from src.vision.apriltag_detector import AprilTagDetector

logger = logging.getLogger(__name__)


def compute_normalized_dlt(
    field_pts: np.ndarray,
    image_pts: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Computes initial 3x3 homography matrix H mapping field coordinates (X, Y)
    to image coordinates (u, v) using normalized Direct Linear Transform (DLT) via SVD.

    Hartley normalization translates points to centroid and scales average distance
    to sqrt(2), guaranteeing condition number < 10^6.

    Args:
        field_pts: Array of shape (N, 2) in physical field inches.
        image_pts: Array of shape (N, 2) in image pixels.

    Returns:
        Tuple of:
            - H: 3x3 homography matrix normalized with H[2, 2] = 1.0 (float64)
            - condition_number: Condition number of the SVD system matrix A.

    Raises:
        ValueError: If fewer than 4 correspondences are provided or points are degenerate.
    """
    field_pts = np.asarray(field_pts, dtype=np.float64)
    image_pts = np.asarray(image_pts, dtype=np.float64)

    n_points = field_pts.shape[0]
    if n_points < 4:
        raise ValueError(f"Normalized DLT requires >= 4 point correspondences, got {n_points}")
    if image_pts.shape[0] != n_points:
        raise ValueError(
            f"Point correspondence mismatch: {n_points} field pts vs {image_pts.shape[0]} image pts"
        )

    # 1. Hartley Normalization
    mean_f = np.mean(field_pts, axis=0)
    mean_i = np.mean(image_pts, axis=0)

    dist_f = np.mean(np.linalg.norm(field_pts - mean_f, axis=1))
    dist_i = np.mean(np.linalg.norm(image_pts - mean_i, axis=1))

    if dist_f < 1e-9 or dist_i < 1e-9:
        raise ValueError("Degenerate point configuration: points are identical or coincident")

    s_f = np.sqrt(2.0) / dist_f
    s_i = np.sqrt(2.0) / dist_i

    T_field = np.array([
        [s_f, 0.0, -s_f * mean_f[0]],
        [0.0, s_f, -s_f * mean_f[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    T_image = np.array([
        [s_i, 0.0, -s_i * mean_i[0]],
        [0.0, s_i, -s_i * mean_i[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    norm_f = (field_pts - mean_f) * s_f
    norm_i = (image_pts - mean_i) * s_i

    # 2. Formulate 2N x 9 matrix A: p_i x (H * P_i) = 0
    A = np.zeros((2 * n_points, 9), dtype=np.float64)
    for i in range(n_points):
        x, y = norm_f[i, 0], norm_f[i, 1]
        u, v = norm_i[i, 0], norm_i[i, 1]
        A[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # 3. SVD of A
    _, S, Vt = np.linalg.svd(A)

    # Condition number: ratio of maximum to minimum singular value (or second-to-last if S[-1] near 0)
    if S[-1] > 1e-9:
        cond_num = float(S[0] / S[-1])
    else:
        cond_num = float(S[0] / S[-2]) if len(S) >= 2 and S[-2] > 1e-9 else 1.0

    # 4. Denormalize homography
    h_tilde = Vt[-1].reshape(3, 3)
    H_init = np.linalg.inv(T_image) @ h_tilde @ T_field

    if abs(H_init[2, 2]) > 1e-12:
        H_init = H_init / H_init[2, 2]
    else:
        H_init = H_init / np.linalg.norm(H_init)

    return H_init, cond_num


def optimize_homography_lm(
    H_init: np.ndarray,
    field_pts: np.ndarray,
    image_pts: np.ndarray,
    max_nfev: int = 200,
) -> Tuple[np.ndarray, float, int, bool]:
    """
    Refines initial homography matrix using Levenberg-Marquardt non-linear optimization
    minimizing geometric reprojection error in pixel coordinates.

    Objective: min_H sum_i || p_i - H * P_i ||^2

    Args:
        H_init: Initial 3x3 homography matrix from DLT.
        field_pts: Array of shape (N, 2) in physical field inches.
        image_pts: Array of shape (N, 2) in image pixels.
        max_nfev: Maximum number of function evaluations for LM optimizer.

    Returns:
        Tuple of:
            - H_opt: Refined 3x3 homography matrix (float64)
            - rmse: Root-mean-square reprojection error in pixels.
            - iterations: Number of function evaluations performed.
            - converged: Boolean indicating successful convergence.
    """
    field_pts = np.asarray(field_pts, dtype=np.float64)
    image_pts = np.asarray(image_pts, dtype=np.float64)

    if abs(H_init[2, 2]) > 1e-12:
        H_norm = H_init / H_init[2, 2]
    else:
        H_norm = H_init

    p0 = H_norm.ravel()[:8]
    pts_h = np.column_stack([field_pts, np.ones(len(field_pts), dtype=np.float64)])

    def residual_fn(params: np.ndarray) -> np.ndarray:
        H = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], 1.0],
        ], dtype=np.float64)
        proj = (H @ pts_h.T).T
        z = proj[:, 2:3]
        z_safe = np.where(np.abs(z) < 1e-9, 1e-9 * np.sign(z) + 1e-9, z)
        proj_uv = proj[:, :2] / z_safe
        return (proj_uv - image_pts).ravel()

    res = least_squares(
        residual_fn,
        p0,
        method="lm",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev,
    )

    H_opt = np.array([
        [res.x[0], res.x[1], res.x[2]],
        [res.x[3], res.x[4], res.x[5]],
        [res.x[6], res.x[7], 1.0],
    ], dtype=np.float64)

    residuals = residual_fn(res.x).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    converged = bool(res.success and rmse < 10.0)
    iterations = int(res.nfev)

    return H_opt, rmse, iterations, converged


def project_pixels_to_field_batch(
    pixel_coords: np.ndarray,
    h_matrix: np.ndarray,
) -> np.ndarray:
    """Transforms batch of 2D pixel coordinates into physical field coordinates (x, y) in inches."""
    pts = np.asarray(pixel_coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)

    H_inv = np.linalg.inv(h_matrix)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    proj = (H_inv @ pts_h.T).T
    z = proj[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    return proj[:, :2] / z_safe
 
 
def estimate_camera_pose_pnp(
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    camera_matrix: Optional[np.ndarray] = None,
    image_shape: Tuple[int, int] = (1080, 1920),
    optimize_focal: bool = True,
    default_focal_length: float = 1250.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, bool]:
    """
    Solves Perspective-n-Point (PnP) pose estimation using 3D CAD object coordinates
    and 2D image coordinates. Computes camera rotation, translation, camera optical center,
    and the induced ground-plane homography H_floor.

    Args:
        object_points_3d: Array of shape (N, 3) in physical field inches (X, Y, Z).
        image_points_2d: Array of shape (N, 2) in image pixels (u, v).
        camera_matrix: Optional 3x3 intrinsic matrix K.
        image_shape: (height, width) of image, default (1080, 1920).
        optimize_focal: Whether to optimize focal length if camera_matrix is None.
        default_focal_length: Fallback focal length if optimization is skipped or fails.

    Returns:
        Tuple of:
            - rvec: (3, 1) rotation vector (float64)
            - tvec: (3, 1) translation vector (float64)
            - camera_center: (3,) camera optical center C = -R^T T in field inches
            - H_floor: (3, 3) ground plane homography matrix K [r1, r2, t] (normalized H[2,2]=1.0)
            - reproj_rmse: Reprojection root-mean-square error in pixels
            - focal_length: Effective focal length f in pixels
            - success: True if PnP solved successfully with acceptable residual
    """
    obj_pts = np.asarray(object_points_3d, dtype=np.float64)
    img_pts = np.asarray(image_points_2d, dtype=np.float64)
    n_pts = len(obj_pts)
    if n_pts < 4:
        raise ValueError(f"estimate_camera_pose_pnp requires >= 4 points, got {n_pts}")

    h_img, w_img = image_shape
    cx = float(w_img) / 2.0
    cy = float(h_img) / 2.0
    scale = float(w_img) / 1920.0

    if camera_matrix is not None:
        K = np.asarray(camera_matrix, dtype=np.float64)
        f_val = float(K[0, 0])
    elif optimize_focal and n_pts >= 5:
        # Optimize focal length using bounded 1D scalar minimization
        def pnp_cost(f_candidate: float) -> float:
            K_cand = np.array([
                [f_candidate, 0.0, cx],
                [0.0, f_candidate, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            ok, r_tmp, t_tmp = cv2.solvePnP(
                obj_pts, img_pts, K_cand, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return 1e6
            try:
                r_tmp, t_tmp = cv2.solvePnPRefineLM(obj_pts, img_pts, K_cand, None, r_tmp, t_tmp)
            except Exception:
                pass
            p_cand, _ = cv2.projectPoints(obj_pts, r_tmp, t_tmp, K_cand, None)
            res = p_cand.reshape(-1, 2) - img_pts
            return float(np.sqrt(np.mean(np.sum(res**2, axis=1))))

        f_low = max(400.0 * scale, 300.0)
        f_high = min(3500.0 * scale, 4000.0)
        try:
            res_opt = minimize_scalar(pnp_cost, bounds=(f_low, f_high), method="bounded")
            f_val = float(res_opt.x) if (res_opt.success and res_opt.fun < 100.0) else default_focal_length * scale
        except Exception:
            f_val = default_focal_length * scale

        K = np.array([
            [f_val, 0.0, cx],
            [0.0, f_val, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    else:
        f_val = default_focal_length * scale
        K = np.array([
            [f_val, 0.0, cx],
            [0.0, f_val, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    # Solve PnP with RANSAC for >= 5 points, or standard iterative for 4 points
    if n_pts >= 5:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            iterationsCount=1000,
        )
        if ok and inliers is not None and len(inliers) >= 4:
            obj_inliers = obj_pts[inliers.ravel()]
            img_inliers = img_pts[inliers.ravel()]
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_inliers, img_inliers, K, None, rvec, tvec)
            except Exception:
                pass
    else:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, None, rvec, tvec)
            except Exception:
                pass

    if not ok or rvec is None or tvec is None:
        return (
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            999.0,
            f_val,
            False,
        )

    # Calculate camera optical center in world coordinates: C = -R^T * T
    R, _ = cv2.Rodrigues(rvec)
    camera_center = (-R.T @ tvec).ravel().astype(np.float64)

    # Compute carpet floor homography (Z=0 plane):
    # [u, v, 1]^T ~ K [r1, r2, t] [X, Y, 1]^T
    H_floor = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    if abs(H_floor[2, 2]) > 1e-12:
        H_floor /= H_floor[2, 2]

    # Calculate reprojection RMSE
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    errs = proj_pts.reshape(-1, 2) - img_pts
    rmse = float(np.sqrt(np.mean(np.sum(errs**2, axis=1))))
    success = bool(rmse < 10.0 and abs(H_floor[2, 2]) > 1e-6)

    return rvec, tvec, camera_center, H_floor, rmse, f_val, success


def project_pixel_to_field_3d(
    pixel_u: float,
    pixel_v: float,
    elevation_inch: float = 0.0,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    camera_center: Optional[np.ndarray] = None,
    h_floor: Optional[np.ndarray] = None,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Projects 2D pixel coordinates (u, v) into physical 3D field coordinates (X, Y, Z) in inches
    at a given plane elevation Z = elevation_inch, and computes Euclidean depth from camera.

    Args:
        pixel_u: Pixel X coordinate.
        pixel_v: Pixel Y coordinate.
        elevation_inch: Target elevation Z in inches (0.0 for carpet floor, 2.955 for fuel ball center).
        rvec: Optional camera rotation vector (3, 1).
        tvec: Optional camera translation vector (3, 1).
        camera_matrix: Optional 3x3 intrinsic matrix K.
        camera_center: Optional 3D camera optical center (X, Y, Z) in inches.
        h_floor: Optional 3x3 floor homography matrix.

    Returns:
        Tuple of:
            - field_coords_3d: (X, Y, Z) in physical inches
            - depth_inch: Euclidean distance || P - C || in physical inches (0.0 if camera center unknown)
    """
    u = float(pixel_u)
    v = float(pixel_v)
    z_target = float(elevation_inch)

    # 1. Preferred method: 3D ray intersection with horizontal plane
    if rvec is not None and tvec is not None and camera_matrix is not None:
        R, _ = cv2.Rodrigues(rvec)
        C = camera_center if camera_center is not None else (-R.T @ tvec).ravel().astype(np.float64)
        K_inv = np.linalg.inv(camera_matrix)
        p_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        d_world = (R.T @ p_cam).ravel()

        if abs(d_world[2]) > 1e-9:
            lam = (z_target - C[2]) / d_world[2]
            P = C + lam * d_world
            X, Y, Z = float(P[0]), float(P[1]), z_target
            depth = float(np.linalg.norm(P - C))
            return (X, Y, Z), depth

    # 2. Fallback method: floor homography inversion
    if h_floor is not None:
        H_inv = np.linalg.inv(h_floor)
        p = np.array([u, v, 1.0], dtype=np.float64)
        P = H_inv @ p
        z_div = P[2] if abs(P[2]) > 1e-12 else 1e-12
        X = float(P[0] / z_div)
        Y = float(P[1] / z_div)
        Z = z_target
        depth = 0.0
        if camera_center is not None:
            depth = float(np.linalg.norm(np.array([X, Y, Z]) - camera_center))
        return (X, Y, Z), depth

    return (u, v, z_target), 0.0


def project_pixels_to_field_3d_batch(
    pixel_coords: np.ndarray,
    elevation_inch: float = 0.0,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
    camera_matrix: Optional[np.ndarray] = None,
    camera_center: Optional[np.ndarray] = None,
    h_floor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch projection of 2D pixel coordinates (N, 2) to 3D field coordinates (N, 3)
    and Euclidean depth values (N,).
    """
    pts = np.asarray(pixel_coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    coords_3d = np.zeros((n, 3), dtype=np.float64)
    depths = np.zeros(n, dtype=np.float64)

    for i in range(n):
        c, d = project_pixel_to_field_3d(
            pixel_u=pts[i, 0],
            pixel_v=pts[i, 1],
            elevation_inch=elevation_inch,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            camera_center=camera_center,
            h_floor=h_floor,
        )
        coords_3d[i] = c
        depths[i] = d

    return coords_3d, depths


def compute_cmc_affine_detailed(
    curr_frame: np.ndarray,
    prev_frame: np.ndarray,
    curr_gray: Optional[np.ndarray] = None,
    prev_gray: Optional[np.ndarray] = None,
    pts_prev: Optional[np.ndarray] = None,
    return_pts: bool = False,
) -> Union[Tuple[np.ndarray, bool, float], Tuple[np.ndarray, bool, float, Optional[np.ndarray]]]:
    """
    Calculates 2x3 affine background transformation matrix for BoT-SORT CMC
    using Lucas-Kanade sparse optical flow and RANSAC.
    Optimized with grayscale caching and temporal feature point reuse.

    Returns:
        Tuple of:
            - affine_cmc: 2x3 transformation matrix (float64)
            - tracking_breakdown: True if feature tracking failed / inlier count is insufficient.
            - inlier_ratio: Proportion of tracked features matching affine motion.
            - (optional if return_pts=True) next_pts: Successfully tracked points for next frame.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if curr_frame is None or prev_frame is None or curr_frame.size == 0 or prev_frame.size == 0:
        if return_pts:
            return identity, False, 0.0, None
        return identity, False, 0.0

    if prev_gray is None:
        if len(prev_frame.shape) == 3 and prev_frame.shape[2] == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame

    if curr_gray is None:
        if len(curr_frame.shape) == 3 and curr_frame.shape[2] == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame

    if pts_prev is not None and len(pts_prev) >= 40:
        tracked_pts = np.asarray(pts_prev, dtype=np.float32)
    else:
        gft_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=3,
        )
        tracked_pts = np.asarray(gft_pts, dtype=np.float32) if gft_pts is not None else None

    if tracked_pts is None or len(tracked_pts) < 4:
        if return_pts:
            return identity, False, 0.0, None
        return identity, False, 0.0

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        tracked_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )


    valid_mask = status.ravel() == 1
    valid_prev = tracked_pts[valid_mask]
    valid_curr = pts_curr[valid_mask]

    if len(valid_prev) < 4:
        if return_pts:
            return identity, True, 0.0, None
        return identity, True, 0.0

    M, inliers = cv2.estimateAffine2D(
        valid_prev,
        valid_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
    )

    if M is None:
        if return_pts:
            return identity, True, 0.0, None
        return identity, True, 0.0

    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    inlier_ratio = inlier_count / max(len(tracked_pts), 1)

    # Feature tracking breakdown occurs if inlier ratio is low or too few inliers survive RANSAC
    breakdown = bool(inlier_ratio < 0.25 or inlier_count < 8)

    next_pts: Optional[np.ndarray] = None
    if inliers is not None and not breakdown and len(valid_curr) > 0:
        surviving = valid_curr[inliers.ravel() == 1]
        if len(surviving) >= 30:
            next_pts = surviving.reshape(-1, 1, 2).astype(np.float32)

    if return_pts:
        return M.astype(np.float64), breakdown, float(inlier_ratio), next_pts
    return M.astype(np.float64), breakdown, float(inlier_ratio)



def detect_camera_cut(
    curr_h: Optional[np.ndarray],
    prev_h: Optional[np.ndarray],
    curr_rmse: float,
    prev_rmse: float,
    tracking_breakdown: bool = False,
    max_reproj_displacement: float = 120.0,
    rmse_spike_ratio: float = 3.0,
) -> bool:
    """
    Detects broadcast camera cut based on:
    1. Feature tracking breakdown in Lucas-Kanade optical flow / RANSAC affine.
    2. Homography residual divergence (sudden RMSE spike).
    3. Projective displacement divergence of canonical field coordinates.
    """
    if tracking_breakdown:
        return True

    if curr_rmse > 5.0:
        return True
    if prev_rmse > 0.0 and (curr_rmse / max(prev_rmse, 0.1)) > rmse_spike_ratio:
        return True

    if curr_h is not None and prev_h is not None:
        probe_pts = np.array([
            [0.0, 0.0, 1.0],
            [651.2, 0.0, 1.0],
            [651.2, 317.7, 1.0],
            [0.0, 317.7, 1.0],
            [325.6, 158.85, 1.0],
        ], dtype=np.float64)

        proj_prev = (prev_h @ probe_pts.T).T
        proj_curr = (curr_h @ probe_pts.T).T

        uv_prev = proj_prev[:, :2] / np.where(np.abs(proj_prev[:, 2:3]) < 1e-9, 1e-9, proj_prev[:, 2:3])
        uv_curr = proj_curr[:, :2] / np.where(np.abs(proj_curr[:, 2:3]) < 1e-9, 1e-9, proj_curr[:, 2:3])

        displacements = np.linalg.norm(uv_curr - uv_prev, axis=1)
        if np.max(displacements) > max_reproj_displacement:
            return True

    return False


class HomographyActor(HomographyActorBase):
    """
    Dynamic Homography Ray pipeline actor for the FRC 2026 REBUILT scouting architecture.

    Subclasses HomographyActorBase and satisfies HomographyActorProtocol.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.detector = AprilTagDetector(config_path=config_path)
        self._prev_homography: Optional[HomographyEstimate] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

    async def detect_apriltags(self, frame: RawDecodedFrame) -> List[TagDetectionResult]:
        """Locates up to 32 AprilTag markers on field elements within a single frame."""
        return self.detector.detect_from_frame(frame)

    async def estimate_homography(
        self,
        frame: RawDecodedFrame,
        detections: List[TagDetectionResult],
        prev_homography: Optional[HomographyEstimate] = None,
    ) -> HomographyEstimate:
        """
        Computes 3x3 homography matrix via normalized DLT + Levenberg-Marquardt optimization.
        Estimates Camera Motion Compensation affine transform and checks for broadcast cuts.
        """
        prev_est = prev_homography if prev_homography is not None else self._prev_homography
        prev_h = prev_est.h_matrix if prev_est is not None else None
        prev_rmse = prev_est.lm_rmse if prev_est is not None else 0.0

        curr_gray = None
        if frame.image_bgr is not None:
            if frame.image_bgr.ndim == 3 and frame.image_bgr.shape[2] == 3:
                curr_gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = frame.image_bgr

        # Compute CMC affine transformation relative to previous frame if available
        if self._prev_frame is not None and frame.image_bgr is not None:
            cmc_res = compute_cmc_affine_detailed(
                frame.image_bgr,
                self._prev_frame,
                curr_gray=curr_gray,
                prev_gray=self._prev_gray,
                pts_prev=self._prev_pts,
                return_pts=True,
            )
            cmc_affine, tracking_breakdown, _, self._prev_pts = cmc_res  # type: ignore[misc]
        else:
            cmc_affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            tracking_breakdown = False
            self._prev_pts = None

        # In multi-camera split-screen broadcast feeds (e.g. 1080p FRC broadcasts),
        # the upper portion of the frame contains the primary wide-angle field camera
        # while the lower portion contains close-up robot/floor inset cameras.
        # Filter detections to the wide-angle camera view (y <= 0.65 * H) if >= 4 tags exist there.
        h_frame = frame.image_bgr.shape[0] if (frame.image_bgr is not None and frame.image_bgr.size > 0) else 1080
        top_dets = [d for d in detections if d.center_px[1] <= 0.65 * h_frame]
        effective_dets = top_dets if len(top_dets) >= 4 else detections

        # Extract 2D and 3D point correspondences: field (X, Y, Z) <-> image (u, v)
        field_pts_list: List[Tuple[float, float]] = []
        image_pts_list: List[Tuple[float, float]] = []
        field_pts_3d_list: List[Tuple[float, float, float]] = []

        for det in effective_dets:
            field_pts_list.append((det.field_coords_inch[0], det.field_coords_inch[1]))
            image_pts_list.append((det.center_px[0], det.center_px[1]))
            field_pts_3d_list.append(det.field_coords_inch)

        n_pts = len(field_pts_list)
        if n_pts >= 4:
            field_pts = np.array(field_pts_list, dtype=np.float64)
            image_pts = np.array(image_pts_list, dtype=np.float64)
            field_pts_3d = np.array(field_pts_3d_list, dtype=np.float64)

            # RANSAC outlier filtering for >= 5 correspondences to eliminate spurious tag fits
            if n_pts >= 5:
                _, inlier_mask = cv2.findHomography(
                    field_pts, image_pts, method=cv2.RANSAC, ransacReprojThreshold=50.0
                )
                if inlier_mask is not None and np.sum(inlier_mask) >= 4:
                    inliers = inlier_mask.ravel() == 1
                    field_pts = field_pts[inliers]
                    image_pts = image_pts[inliers]
                    field_pts_3d = field_pts_3d[inliers]

            # Step 1: Normalized DLT
            H_init, dlt_cond = compute_normalized_dlt(field_pts, image_pts)

            # Step 2: Levenberg-Marquardt Non-Linear Optimization
            H_lm, lm_rmse, lm_iters, lm_converged = optimize_homography_lm(
                H_init, field_pts, image_pts
            )

            # Step 3: Perspective-n-Point (PnP) 3D Camera Pose & Ground Plane Estimation
            w_frame = frame.image_bgr.shape[1] if (frame.image_bgr is not None and frame.image_bgr.size > 0) else 1920
            try:
                rvec_pnp, tvec_pnp, cam_center, H_floor, pnp_rmse, f_pnp, pnp_success = estimate_camera_pose_pnp(
                    object_points_3d=field_pts_3d,
                    image_points_2d=image_pts,
                    image_shape=(h_frame, w_frame),
                )
            except Exception as e:
                logger.debug("PnP estimation error: %s", e)
                pnp_success = False
                rvec_pnp, tvec_pnp, cam_center, H_floor, pnp_rmse, f_pnp = None, None, None, None, 999.0, None

            # Select PnP floor homography if valid and well-conditioned with low RMSE (< 2.5 px)
            if pnp_success and (pnp_rmse < 2.5 or (not lm_converged and pnp_rmse < 5.0)):
                H_opt = H_floor
                final_rmse = pnp_rmse
                converged = True
            else:
                H_opt = H_lm
                final_rmse = lm_rmse
                converged = lm_converged

            # Step 4: Camera Cut Detection
            is_cut = detect_camera_cut(
                curr_h=H_opt,
                prev_h=prev_h,
                curr_rmse=final_rmse,
                prev_rmse=prev_rmse,
                tracking_breakdown=tracking_breakdown,
            )

            # Compute Euclidean distance from camera optical center to each visible tag
            visible_tag_ids = [int(d.tag_id) for d in effective_dets]
            tag_distances: Dict[int, float] = {}
            if pnp_success and cam_center is not None:
                for det in effective_dets:
                    p_tag = np.array(det.field_coords_inch, dtype=np.float64)
                    dist = float(np.linalg.norm(p_tag - cam_center))
                    tag_distances[int(det.tag_id)] = dist

            estimate = HomographyEstimate(
                frame_index=frame.metadata.frame_index,
                timestamp_ms=frame.metadata.pts_ms,
                h_matrix=H_opt,
                dlt_condition=dlt_cond,
                lm_rmse=final_rmse,
                lm_iterations=lm_iters,
                converged=converged,
                camera_cut=is_cut,
                affine_cmc=cmc_affine,
                camera_pose_rvec=rvec_pnp if pnp_success else None,
                camera_pose_tvec=tvec_pnp if pnp_success else None,
                camera_center=cam_center if pnp_success else None,
                focal_length=f_pnp if pnp_success else None,
                visible_tags=visible_tag_ids,
                tag_distances=tag_distances,
            )
        else:
            # Fewer than 4 point correspondences: fallback or propagate with CMC
            is_cut = True if prev_est is not None else False
            if prev_h is not None and prev_est is not None:
                # Propagate previous homography with affine CMC
                A_3x3 = np.vstack([cmc_affine, [0.0, 0.0, 1.0]])
                propagated_h = A_3x3 @ prev_h
                if abs(propagated_h[2, 2]) > 1e-12:
                    propagated_h /= propagated_h[2, 2]
                h_mat = propagated_h
                dlt_cond = prev_est.dlt_condition
                lm_rmse = prev_est.lm_rmse
            else:
                h_mat = np.eye(3, dtype=np.float64)
                dlt_cond = 1.0
                lm_rmse = 999.0

            vis_tags = [int(d.tag_id) for d in effective_dets] if effective_dets else []
            estimate = HomographyEstimate(
                frame_index=frame.metadata.frame_index,
                timestamp_ms=frame.metadata.pts_ms,
                h_matrix=h_mat,
                dlt_condition=dlt_cond,
                lm_rmse=lm_rmse,
                lm_iterations=0,
                converged=False,
                camera_cut=is_cut,
                affine_cmc=cmc_affine,
                visible_tags=vis_tags,
                tag_distances={},
            )

        # Update actor state
        self._prev_homography = estimate
        if frame.image_bgr is not None:
            self._prev_frame = frame.image_bgr.copy()
            self._prev_gray = curr_gray

        return estimate

    async def estimate_batch(
        self,
        frames: List[RawDecodedFrame],
        tags_list: Optional[List[List[TagDetectionResult]]] = None,
    ) -> List[HomographyEstimate]:
        """
        Processes a temporal batch of frames efficiently on the actor,
        running AprilTag detection and sequential homography / optical flow with minimal overhead.

        Args:
            frames: Sequence of RawDecodedFrame objects to process.
            tags_list: Optional pre-computed AprilTag detections for each frame.

        Returns:
            List of HomographyEstimate corresponding to each frame in the batch.
        """
        if not frames:
            return []

        # 1. Run AprilTag detection for any frames without precomputed tags
        if tags_list is None:
            tags_list = [self.detector.detect_from_frame(f) for f in frames]

        # 2. Sequential homography estimation with cached optical flow features
        estimates: List[HomographyEstimate] = []
        for frame, tags in zip(frames, tags_list):
            est = await self.estimate_homography(
                frame=frame,
                detections=tags,
                prev_homography=self._prev_homography,
            )
            estimates.append(est)

        return estimates

    async def project_pixel_to_field(
        self,
        pixel_coords: Tuple[float, float],
        homography: HomographyEstimate,
    ) -> Tuple[float, float]:
        """Transforms 2D pixel coordinates (u, v) into physical field coordinates (x, y) in inches."""
        u, v = float(pixel_coords[0]), float(pixel_coords[1])
        H = homography.h_matrix
        if np.linalg.cond(H) > 1e12:
            raise ValueError("Homography matrix is singular or ill-conditioned")

        H_inv = np.linalg.inv(H)
        p = np.array([u, v, 1.0], dtype=np.float64)
        P = H_inv @ p
        if abs(P[2]) < 1e-12:
            raise ValueError(f"Projected point at infinity: P[2]={P[2]}")

        x = float(P[0] / P[2])
        y = float(P[1] / P[2])
        return (x, y)

    async def project_pixel_to_field_3d(
        self,
        pixel_coords: Tuple[float, float],
        homography: HomographyEstimate,
        elevation_inch: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], float]:
        """
        Transforms 2D pixel coordinates (u, v) into physical 3D field coordinates (X, Y, Z) in inches
        at the specified elevation Z = elevation_inch, and computes Euclidean distance (depth) from camera.
        """
        u, v = float(pixel_coords[0]), float(pixel_coords[1])
        cam_matrix = None
        if homography.focal_length is not None:
            f = homography.focal_length
            cam_matrix = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)

        return project_pixel_to_field_3d(
            pixel_u=u,
            pixel_v=v,
            elevation_inch=elevation_inch,
            rvec=homography.camera_pose_rvec,
            tvec=homography.camera_pose_tvec,
            camera_matrix=cam_matrix,
            camera_center=homography.camera_center,
            h_floor=homography.h_matrix,
        )

    async def compute_camera_motion_compensation(
        self,
        curr_frame: np.ndarray,
        prev_frame: np.ndarray,
    ) -> np.ndarray:
        """Calculates 2x3 affine background transformation matrix for BoT-SORT CMC."""
        cmc_res = compute_cmc_affine_detailed(curr_frame, prev_frame)
        return cmc_res[0]

