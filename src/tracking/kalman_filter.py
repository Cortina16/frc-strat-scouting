"""
Modified 8-State Kalman Filter for BoT-SORT multi-object tracking.

Tracks robot bounding boxes directly via:
    x = [x_c, y_c, w, h, v_x, v_y, v_w, v_h]^T
without aspect-ratio parametrization (w/h), ensuring accurate scaling when
robots approach or recede from the camera.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch


def predict_multi_tracks(
    means: np.ndarray,
    covariances: np.ndarray,
    std_weight_position: float = 1.0 / 20.0,
    std_weight_velocity: float = 1.0 / 160.0,
    affine_matrix: Optional[np.ndarray] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized multi-track Kalman filter prediction and covariance updates.
    Predicts all active tracks simultaneously using PyTorch tensors on CUDA:
        X_{new} = F * X
        P_{new} = F * P * F^T + Q
    Optionally applies Camera Motion Compensation (CMC) affine transformation:
        X_{cmc} = CMC(X_{new})
        P_{cmc} = T * P_{new} * T^T

    Args:
        means: Array of shape (N, 8) representing state vectors of N tracks.
        covariances: Array of shape (N, 8, 8) representing covariance matrices of N tracks.
        std_weight_position: Position noise standard deviation weight.
        std_weight_velocity: Velocity noise standard deviation weight.
        affine_matrix: Optional 2x3 affine matrix for CMC.
        device: PyTorch device ('cuda:0' preferred if available).

    Returns:
        Tuple of (predicted_means [N, 8], predicted_covariances [N, 8, 8]).
    """
    N = means.shape[0]
    if N == 0:
        return means.copy(), covariances.copy()

    use_cuda = torch.cuda.is_available()
    dev = device if device is not None else (torch.device("cuda:0") if use_cuda else torch.device("cpu"))

    means_t = torch.as_tensor(means, dtype=torch.float64, device=dev)
    covs_t = torch.as_tensor(covariances, dtype=torch.float64, device=dev)

    # State transition matrix F (8x8)
    F = torch.eye(8, dtype=torch.float64, device=dev)
    for j in range(4):
        F[j, j + 4] = 1.0

    # Dimension bounds for noise
    h = torch.clamp(means_t[:, 3], min=1.0)
    w = torch.clamp(means_t[:, 2], min=1.0)
    ref_dim = 0.5 * (w + h)

    std_pos = std_weight_position * ref_dim
    std_vel = std_weight_velocity * ref_dim

    # Process noise covariance Q (N, 8, 8)
    Q = torch.zeros((N, 8, 8), dtype=torch.float64, device=dev)
    for j in range(4):
        Q[:, j, j] = std_pos ** 2
        Q[:, j + 4, j + 4] = std_vel ** 2

    # Vectorized state prediction: X_new = X @ F^T
    pred_means_t = torch.matmul(means_t, F.t())

    # Vectorized covariance prediction: P_new = F @ P @ F^T + Q
    F_batch = F.unsqueeze(0).expand(N, -1, -1)
    F_t_batch = F.t().unsqueeze(0).expand(N, -1, -1)
    pred_covs_t = torch.bmm(torch.bmm(F_batch, covs_t), F_t_batch) + Q

    # Optional CMC affine transformation
    if affine_matrix is not None:
        M = np.asarray(affine_matrix, dtype=np.float64)
        if M.shape == (2, 3):
            M_t = torch.as_tensor(M, dtype=torch.float64, device=dev)
            A = M_t[:2, :2]
            t = M_t[:2, 2]

            # Transform position [xc, yc]
            pred_means_t[:, :2] = torch.matmul(pred_means_t[:, :2], A.t()) + t
            # Transform velocity [vx, vy]
            pred_means_t[:, 4:6] = torch.matmul(pred_means_t[:, 4:6], A.t())

            # 8x8 block-diagonal transformation matrix T
            T = torch.eye(8, dtype=torch.float64, device=dev)
            T[:2, :2] = A
            T[4:6, 4:6] = A

            T_batch = T.unsqueeze(0).expand(N, -1, -1)
            T_t_batch = T.t().unsqueeze(0).expand(N, -1, -1)
            pred_covs_t = torch.bmm(torch.bmm(T_batch, pred_covs_t), T_t_batch)

    res_means = pred_means_t.cpu().numpy()
    res_covs = pred_covs_t.cpu().numpy()
    return res_means, res_covs


class KalmanFilter:
    """
    An 8-dimensional Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state vector is:
        x = [x_c, y_c, w, h, v_x, v_y, v_w, v_h]^T
    where (x_c, y_c) is the bounding box center, (w, h) are width and height,
    and (v_x, v_y, v_w, v_h) are their respective linear velocities.

    The 4-dimensional measurement vector is:
        z = [x_c, y_c, w, h]^T
    """

    def __init__(
        self,
        initial_measurement: Optional[np.ndarray] = None,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
    ) -> None:
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(8, 8, dtype=np.float64)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(4, 8, dtype=np.float64)

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        if initial_measurement is not None:
            self.initiate(initial_measurement)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initializes track state vector and covariance matrix from an initial bounding box.

        Args:
            measurement: Bounding box [x_c, y_c, w, h] of shape (4,).

        Returns:
            Tuple of (mean [8,], covariance [8, 8]).
        """
        meas = np.asarray(measurement, dtype=np.float64).ravel()
        if len(meas) != 4:
            raise ValueError(f"Measurement must have 4 elements [xc, yc, w, h], got {len(meas)}")

        mean = np.zeros(8, dtype=np.float64)
        mean[:4] = meas

        h = max(float(meas[3]), 1.0)
        w = max(float(meas[2]), 1.0)
        ref_dim = 0.5 * (w + h)

        std = [
            2.0 * self._std_weight_position * ref_dim,
            2.0 * self._std_weight_position * ref_dim,
            2.0 * self._std_weight_position * ref_dim,
            2.0 * self._std_weight_position * ref_dim,
            10.0 * self._std_weight_velocity * ref_dim,
            10.0 * self._std_weight_velocity * ref_dim,
            10.0 * self._std_weight_velocity * ref_dim,
            10.0 * self._std_weight_velocity * ref_dim,
        ]

        covariance = np.diag(np.square(std))
        self.mean = mean
        self.covariance = covariance
        return self.mean, self.covariance

    def predict(
        self,
        mean: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Kalman filter prediction step:
            x_{k|k-1} = F * x_{k-1|k-1}
            P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

        Can be called with internal state or statelessly with provided mean/covariance.
        """
        if mean is None:
            if self.mean is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass mean.")
            mean = self.mean
        if covariance is None:
            if self.covariance is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass covariance.")
            covariance = self.covariance

        m = np.asarray(mean, dtype=np.float64).copy()
        cov = np.asarray(covariance, dtype=np.float64).copy()

        h = max(float(m[3]), 1.0)
        w = max(float(m[2]), 1.0)
        ref_dim = 0.5 * (w + h)

        std_pos = [
            self._std_weight_position * ref_dim,
            self._std_weight_position * ref_dim,
            self._std_weight_position * ref_dim,
            self._std_weight_position * ref_dim,
        ]
        std_vel = [
            self._std_weight_velocity * ref_dim,
            self._std_weight_velocity * ref_dim,
            self._std_weight_velocity * ref_dim,
            self._std_weight_velocity * ref_dim,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        predicted_mean = np.dot(self._motion_mat, m)
        predicted_covariance = np.linalg.multi_dot((
            self._motion_mat, cov, self._motion_mat.T
        )) + motion_cov

        self.mean = predicted_mean
        self.covariance = predicted_covariance
        return predicted_mean, predicted_covariance

    def update(
        self,
        measurement: np.ndarray,
        mean: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        confidence: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Kalman filter measurement correction step:
            y = z - H * x
            S = H * P * H^T + R
            K = P * H^T * S^{-1}
            x' = x + K * y
            P' = (I - K * H) * P * (I - K * H)^T + K * R * K^T (Joseph form)
        """
        if mean is None:
            if self.mean is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass mean.")
            mean = self.mean
        if covariance is None:
            if self.covariance is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass covariance.")
            covariance = self.covariance

        m = np.asarray(mean, dtype=np.float64).copy()
        cov = np.asarray(covariance, dtype=np.float64).copy()
        meas = np.asarray(measurement, dtype=np.float64).ravel()

        h = max(float(meas[3]), 1.0)
        w = max(float(meas[2]), 1.0)
        ref_dim = 0.5 * (w + h)

        # Confidence scaling: lower confidence -> higher measurement noise R
        conf_clamped = max(0.20, min(1.0, float(confidence)))
        r_scale = 1.0 / (conf_clamped ** 2)

        # Aspect ratio occlusion guard: check if measured AR deviates from state AR due to partial occlusion
        pred_w = max(float(m[2]), 1.0)
        pred_h = max(float(m[3]), 1.0)
        pred_ar = pred_w / pred_h
        meas_ar = w / h
        ar_ratio = max(meas_ar / pred_ar, pred_ar / meas_ar)

        dim_scale = 1.0
        if ar_ratio > 1.35:
            dim_scale = min(4.0, ar_ratio)

        std = [
            self._std_weight_position * ref_dim * np.sqrt(r_scale),
            self._std_weight_position * ref_dim * np.sqrt(r_scale),
            self._std_weight_position * ref_dim * np.sqrt(r_scale * dim_scale),
            self._std_weight_position * ref_dim * np.sqrt(r_scale * dim_scale),
        ]
        R = np.diag(np.square(std))

        projected_mean = np.dot(self._update_mat, m)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, cov, self._update_mat.T
        )) + R

        innovation = meas - projected_mean

        # Kalman gain: K = P * H^T * inv(S)
        # Solve S * K^T = H * P^T => K^T = solve(S, H * P)
        K = np.linalg.solve(projected_cov, np.dot(cov, self._update_mat.T).T).T

        updated_mean = m + np.dot(K, innovation)

        # Joseph form update for guaranteed symmetry & positive semi-definiteness
        I_KH = np.eye(8, dtype=np.float64) - np.dot(K, self._update_mat)
        updated_covariance = np.linalg.multi_dot((I_KH, cov, I_KH.T)) + np.linalg.multi_dot((K, R, K.T))

        self.mean = updated_mean
        self.covariance = updated_covariance
        return updated_mean, updated_covariance

    def apply_cmc(
        self,
        affine_matrix: Optional[np.ndarray],
        mean: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Camera Motion Compensation (CMC) affine transformation to state and covariance.

        Given affine transformation M = [A | t] of shape (2, 3):
            [xc', yc']^T = A * [xc, yc]^T + t
            [vx', vy']^T = A * [vx, vy]^T
            P' = T * P * T^T
        where T = diag(A, I_2, A, I_2).
        """
        if mean is None:
            if self.mean is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass mean.")
            mean = self.mean
        if covariance is None:
            if self.covariance is None:
                raise RuntimeError("KalmanFilter not initiated; call initiate() first or pass covariance.")
            covariance = self.covariance

        if affine_matrix is None:
            return mean, covariance

        M = np.asarray(affine_matrix, dtype=np.float64)
        if M.shape != (2, 3):
            raise ValueError(f"Affine matrix must have shape (2, 3), got {M.shape}")

        A = M[:2, :2]
        t = M[:2, 2]

        m = np.asarray(mean, dtype=np.float64).copy()
        cov = np.asarray(covariance, dtype=np.float64).copy()

        # Transform position [xc, yc]
        m[:2] = np.dot(A, m[:2]) + t
        # Transform velocity [vx, vy]
        m[4:6] = np.dot(A, m[4:6])

        # 8x8 transformation matrix T
        T = np.eye(8, dtype=np.float64)
        T[:2, :2] = A
        T[4:6, 4:6] = A

        # Transform covariance: P' = T * P * T^T
        cov = np.linalg.multi_dot((T, cov, T.T))

        self.mean = m
        self.covariance = cov
        return m, cov

    @property
    def state_vector(self) -> np.ndarray:
        """Returns the current 8-dimensional state vector."""
        if self.mean is None:
            raise RuntimeError("KalmanFilter state vector not initialized.")
        return self.mean

    @property
    def bbox(self) -> np.ndarray:
        """Returns the bounding box coordinates [xc, yc, w, h]."""
        if self.mean is None:
            raise RuntimeError("KalmanFilter state vector not initialized.")
        return self.mean[:4]

    @property
    def velocity(self) -> np.ndarray:
        """Returns the velocity vector [vx, vy, vw, vh]."""
        if self.mean is None:
            raise RuntimeError("KalmanFilter state vector not initialized.")
        return self.mean[4:]

    @classmethod
    def multi_predict(
        cls,
        filters: Sequence[KalmanFilter],
        affine_matrix: Optional[np.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Predicts multiple KalmanFilter instances simultaneously in a single batched GPU/CUDA operation.
        Updates each filter's mean and covariance in-place.
        """
        valid_filters = [kf for kf in filters if kf.mean is not None and kf.covariance is not None]
        if not valid_filters:
            return
        means = np.stack([kf.mean for kf in valid_filters])
        covariances = np.stack([kf.covariance for kf in valid_filters])
        std_p = valid_filters[0]._std_weight_position
        std_v = valid_filters[0]._std_weight_velocity
        pred_means, pred_covs = predict_multi_tracks(
            means=means,
            covariances=covariances,
            std_weight_position=std_p,
            std_weight_velocity=std_v,
            affine_matrix=affine_matrix,
            device=device,
        )
        for i, kf in enumerate(valid_filters):
            kf.mean = pred_means[i]
            kf.covariance = pred_covs[i]

