"""
Extended Kalman Filter (EKF) and Proximity-Emulated IoU for FRC Robot Tracking.

Provides:
- RobotEKF: 6-state Extended Kalman Filter modeling physical field kinematics
  [X, Y, V_X, V_Y, W_px, H_px]^T with non-linear homography projective measurement
  h(x) and analytical Jacobian H_j. Supports multi-observation sequential updates
  that contract covariance when corroborating detections are observed in proximity.
- emulate_iou: Distance-aware IoU emulator that computes standard box IoU when boxes
  intersect, and smoothly emulates IoU in [0, 1] based on spatial proximity when boxes
  do not intersect.
- pairwise_emulate_iou: Vectorized pairwise emulated IoU matrix computation.
- fuse_bounding_boxes: Merges multiple proximity detections into a unified bounding box.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import numpy as np

from src.common.types import BoundingBox

logger = logging.getLogger("frc.vision.robot_ekf")


def emulate_iou(
    box1: BoundingBox,
    box2: BoundingBox,
    proximity_thresh_px: float = 160.0,
    field_coords1: Optional[Tuple[float, float]] = None,
    field_coords2: Optional[Tuple[float, float]] = None,
    proximity_thresh_inch: float = 65.0,
) -> float:
    """
    Calculates emulated Intersection-over-Union (IoU) between two bounding boxes.

    If the bounding boxes physically intersect, standard IoU is computed.
    If the bounding boxes do not intersect (IoU = 0), an emulated IoU value is
    computed based on spatial proximity:
        IoU_emulated = max(0.0, 1.0 - d / D_prox)
    where d is the spatial gap/distance between the boxes in pixel or field space.

    Args:
        box1: First bounding box.
        box2: Second bounding box.
        proximity_thresh_px: Pixel proximity threshold (D_prox). Gap distance above
            this yields 0.0 emulated IoU. Default is 120.0 px.
        field_coords1: Optional (X, Y) field coordinates for box1 in inches.
        field_coords2: Optional (X, Y) field coordinates for box2 in inches.
        proximity_thresh_inch: Field distance threshold in inches. Default is 48.0 inches.

    Returns:
        Emulated IoU in range [0.0, 1.0].
    """
    # 1. Standard Box IoU
    x1_min = box1.center_x - box1.width * 0.5
    y1_min = box1.center_y - box1.height * 0.5
    x1_max = box1.center_x + box1.width * 0.5
    y1_max = box1.center_y + box1.height * 0.5

    x2_min = box2.center_x - box2.width * 0.5
    y2_min = box2.center_y - box2.height * 0.5
    x2_max = box2.center_x + box2.width * 0.5
    y2_max = box2.center_y + box2.height * 0.5

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(1.0, box1.width * box1.height)
    area2 = max(1.0, box2.width * box2.height)
    union_area = area1 + area2 - inter_area
    std_iou = float(inter_area / union_area) if union_area > 0 else 0.0

    # If boxes intersect, return standard IoU directly
    if std_iou > 0.0:
        return std_iou

    # 2. Pixel Proximity Gap Calculation
    # Shortest distance between the two non-overlapping rectangles
    dx = max(0.0, abs(box1.center_x - box2.center_x) - 0.5 * (box1.width + box2.width))
    dy = max(0.0, abs(box1.center_y - box2.center_y) - 0.5 * (box1.height + box2.height))
    gap_px = float(np.hypot(dx, dy))

    iou_prox_px = 0.0
    if gap_px <= proximity_thresh_px:
        iou_prox_px = float(max(0.0, 1.0 - gap_px / max(1.0, proximity_thresh_px)))

    # 3. Physical Field Proximity (if coordinates provided)
    iou_field = 0.0
    if field_coords1 is not None and field_coords2 is not None:
        dist_inch = float(np.hypot(field_coords1[0] - field_coords2[0], field_coords1[1] - field_coords2[1]))
        if dist_inch <= proximity_thresh_inch:
            iou_field = float(max(0.0, 1.0 - dist_inch / max(1.0, proximity_thresh_inch)))

    # Synthesize emulated IoU for non-overlapping boxes
    emulated = max(iou_prox_px, iou_field)
    return float(np.clip(emulated, 0.0, 1.0))


def pairwise_emulate_iou(
    boxes1: Sequence[BoundingBox],
    boxes2: Sequence[BoundingBox],
    proximity_thresh_px: float = 160.0,
) -> np.ndarray:
    """
    Computes pairwise emulated IoU matrix of shape (N, M).
    """
    n = len(boxes1)
    m = len(boxes2)
    mat = np.zeros((n, m), dtype=np.float64)
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            mat[i, j] = emulate_iou(b1, b2, proximity_thresh_px=proximity_thresh_px)
    return mat


def fuse_bounding_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
    """
    Merges multiple bounding boxes into a single enclosing bounding box.
    """
    if not boxes:
        return BoundingBox(center_x=0.0, center_y=0.0, width=0.0, height=0.0, confidence=0.0, class_id=0)
    if len(boxes) == 1:
        return boxes[0]

    min_x = min(b.center_x - b.width * 0.5 for b in boxes)
    max_x = max(b.center_x + b.width * 0.5 for b in boxes)
    min_y = min(b.center_y - b.height * 0.5 for b in boxes)
    max_y = max(b.center_y + b.height * 0.5 for b in boxes)

    w = max_x - min_x
    h = max_y - min_y
    cx = min_x + w * 0.5
    cy = min_y + h * 0.5

    # Weighted confidence
    conf = float(np.clip(max(b.confidence for b in boxes) + 0.05 * (len(boxes) - 1), 0.0, 1.0))
    class_id = boxes[0].class_id

    return BoundingBox(
        center_x=cx,
        center_y=cy,
        width=w,
        height=h,
        confidence=conf,
        class_id=class_id,
    )


class RobotEKF:
    """
    Extended Kalman Filter for FRC Robot Tracking.

    State Vector:
        x = [X, Y, V_X, V_Y, W_px, H_px]^T in R^6
        - (X, Y): Physical field coordinates in inches.
        - (V_X, V_Y): Linear velocity on field in inches/second.
        - (W_px, H_px): Pixel bounding box width and height.

    Observation Models:
        - Image bounding box z = [u, v, w, h]^T via non-linear homography projection
          h(x) and analytical Jacobian H_j.
        - Field coordinates z = [X, Y, w, h]^T via linear measurement.
        - Multi-observation update for candidate detections within proximity,
          contracting state covariance with each corroborating detection.
    """

    def __init__(
        self,
        init_field_x: float,
        init_field_y: float,
        init_w_px: float = 80.0,
        init_h_px: float = 70.0,
        init_vx: float = 0.0,
        init_vy: float = 0.0,
        carpet_length_inch: float = 651.2,
        carpet_width_inch: float = 317.7,
        max_velocity_ips: float = 240.0,
    ) -> None:
        self.carpet_length = carpet_length_inch
        self.carpet_width = carpet_width_inch
        self.max_velocity = max_velocity_ips

        # State vector: [X, Y, Vx, Vy, W_px, H_px]
        self.x = np.array(
            [init_field_x, init_field_y, init_vx, init_vy, init_w_px, init_h_px],
            dtype=np.float64,
        )

        # State Covariance P (6x6)
        self.P = np.diag([
            25.0,  # X var (in^2)
            25.0,  # Y var (in^2)
            100.0, # Vx var ((in/s)^2)
            100.0, # Vy var ((in/s)^2)
            64.0,  # W_px var
            64.0,  # H_px var
        ]).astype(np.float64)

        # Base measurement noise covariance R (4x4)
        self.R_base = np.diag([16.0, 16.0, 25.0, 25.0]).astype(np.float64)

        # Process noise spectral density
        self.q_acc = 40.0   # in/s^2 acceleration noise
        self.q_size = 5.0   # px/s size variance

        self.last_update_count: int = 0
        self.consecutive_misses: int = 0

    @property
    def field_x(self) -> float:
        return float(self.x[0])

    @property
    def field_y(self) -> float:
        return float(self.x[1])

    @property
    def vx(self) -> float:
        return float(self.x[2])

    @property
    def vy(self) -> float:
        return float(self.x[3])

    @property
    def pixel_w(self) -> float:
        return float(max(10.0, self.x[4]))

    @property
    def pixel_h(self) -> float:
        return float(max(10.0, self.x[5]))

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def predict(self, dt: float = 1.0 / 30.0, drag_coeff: float = 0.05) -> np.ndarray:
        """
        Propagates state and covariance forward by time interval dt.
        Applies linear kinematics with velocity damping (drag).
        """
        # State transition matrix F (6x6)
        damping = max(0.5, 1.0 - drag_coeff * dt)
        F = np.eye(6, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 2] = damping
        F[3, 3] = damping

        # Predict state: x_new = F * x
        self.x = F @ self.x

        # Physical velocity magnitude clamp (mechanical limits of FRC swerve/tank drivetrains)
        v_mag = float(np.hypot(self.x[2], self.x[3]))
        if v_mag > self.max_velocity:
            scale = self.max_velocity / max(1e-6, v_mag)
            self.x[2] *= scale
            self.x[3] *= scale

        # Soft physical carpet clamp to prevent divergent extrapolation
        self.x[0] = np.clip(self.x[0], -20.0, self.carpet_length + 20.0)
        self.x[1] = np.clip(self.x[1], -20.0, self.carpet_width + 20.0)

        # Process noise covariance Q (6x6)
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.zeros((6, 6), dtype=np.float64)
        # Position / velocity blocks
        q_pos = (dt3 / 3.0) * self.q_acc
        q_pos_vel = (dt2 / 2.0) * self.q_acc
        q_vel = dt * self.q_acc

        Q[0, 0] = q_pos
        Q[0, 2] = q_pos_vel
        Q[2, 0] = q_pos_vel
        Q[2, 2] = q_vel

        Q[1, 1] = q_pos
        Q[1, 3] = q_pos_vel
        Q[3, 1] = q_pos_vel
        Q[3, 3] = q_vel

        # Box dimensions
        Q[4, 4] = self.q_size * dt
        Q[5, 5] = self.q_size * dt

        # Propagate covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q
        self.consecutive_misses += 1
        return self.x.copy()

    def _eval_h_projective(
        self,
        H_field_to_image: np.ndarray,
        ground_contact_offset_ratio: float = 0.40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates non-linear measurement function h(x) and analytical Jacobian H_j
        projecting field ground contact (X, Y) to image center (u, v) via homography.

        Returns:
            Tuple of (h_pred [4], H_j [4, 6]).
        """
        X = self.x[0]
        Y = self.x[1]
        w_px = self.pixel_w
        h_px = self.pixel_h

        # Project field point (X, Y, 1) -> (u', v', w')
        pt_f = np.array([X, Y, 1.0], dtype=np.float64)
        proj = H_field_to_image @ pt_f
        u_prime = proj[0]
        v_prime = proj[1]
        w_prime = proj[2] if abs(proj[2]) > 1e-9 else 1e-9

        # Ground contact image coords
        u_ground = u_prime / w_prime
        v_ground = v_prime / w_prime

        # Robot bbox center: center_y = v_ground - offset * h_px
        u_center = u_ground
        v_center = v_ground - ground_contact_offset_ratio * h_px

        h_pred = np.array([u_center, v_center, w_px, h_px], dtype=np.float64)

        # Analytical Jacobian H_j (4x6):
        # dh/dx = d[u_center, v_center, w_px, h_px] / d[X, Y, Vx, Vy, W_px, H_px]
        H_j = np.zeros((4, 6), dtype=np.float64)

        w_sq = w_prime * w_prime
        h00 = H_field_to_image[0, 0]
        h01 = H_field_to_image[0, 1]
        h10 = H_field_to_image[1, 0]
        h11 = H_field_to_image[1, 1]
        h20 = H_field_to_image[2, 0]
        h21 = H_field_to_image[2, 1]

        # du_center / dX = (h00 * w' - u' * h20) / w'^2
        H_j[0, 0] = (h00 * w_prime - u_prime * h20) / w_sq
        # du_center / dY = (h01 * w' - u' * h21) / w'^2
        H_j[0, 1] = (h01 * w_prime - u_prime * h21) / w_sq

        # dv_center / dX = (h10 * w' - v' * h20) / w'^2
        H_j[1, 0] = (h10 * w_prime - v_prime * h20) / w_sq
        # dv_center / dY = (h11 * w' - v' * h21) / w'^2
        H_j[1, 1] = (h11 * w_prime - v_prime * h21) / w_sq
        # dv_center / dH_px = -offset
        H_j[1, 5] = -ground_contact_offset_ratio

        # dw / dW_px = 1.0
        H_j[2, 4] = 1.0
        # dh / dH_px = 1.0
        H_j[3, 5] = 1.0

        return h_pred, H_j

    def update(
        self,
        box: BoundingBox,
        H_image_to_field: Optional[np.ndarray] = None,
        field_coords: Optional[Tuple[float, float]] = None,
        r_scale: float = 1.0,
    ) -> None:
        """
        Executes a single EKF observation update.
        """
        z_meas = np.array([box.center_x, box.center_y, box.width, box.height], dtype=np.float64)
        R = self.R_base * r_scale

        # Attempt projective measurement via homography
        updated_projective = False
        if H_image_to_field is not None:
            try:
                H_field_to_image = np.linalg.inv(H_image_to_field)
                h_pred, H_j = self._eval_h_projective(H_field_to_image)

                # Innovation y = z - h(x)
                y = z_meas - h_pred

                # Innovation covariance S = H_j * P * H_j^T + R
                S = H_j @ self.P @ H_j.T + R
                K = self.P @ H_j.T @ np.linalg.pinv(S)

                # Update state: x = x + K * y
                self.x = self.x + K @ y

                # Update covariance (Joseph form for numerical stability):
                I = np.eye(6, dtype=np.float64)
                I_KH = I - K @ H_j
                self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

                updated_projective = True
            except Exception as e:
                logger.debug("Projective EKF update failed: %s, falling back to field coords", e)

        if not updated_projective:
            # Linear field-coordinate fallback update
            fx = field_coords[0] if field_coords is not None else self.x[0]
            fy = field_coords[1] if field_coords is not None else self.x[1]
            z_field = np.array([fx, fy, box.width, box.height], dtype=np.float64)

            # Measurement matrix H (4x6): direct [X, Y, W, H]
            H = np.zeros((4, 6), dtype=np.float64)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            H[2, 4] = 1.0
            H[3, 5] = 1.0

            h_pred = H @ self.x
            y = z_field - h_pred
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.pinv(S)

            self.x = self.x + K @ y
            I = np.eye(6, dtype=np.float64)
            I_KH = I - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Physical boundary clamp
        self.x[0] = np.clip(self.x[0], 0.0, self.carpet_length)
        self.x[1] = np.clip(self.x[1], 0.0, self.carpet_width)

        # Physical velocity magnitude clamp
        v_mag = float(np.hypot(self.x[2], self.x[3]))
        if v_mag > self.max_velocity:
            scale = self.max_velocity / max(1e-6, v_mag)
            self.x[2] *= scale
            self.x[3] *= scale

        self.consecutive_misses = 0
        self.last_update_count += 1

    def update_multi(
        self,
        primary_box: BoundingBox,
        additional_boxes: Sequence[BoundingBox] = (),
        H_image_to_field: Optional[np.ndarray] = None,
        primary_field: Optional[Tuple[float, float]] = None,
        additional_fields: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> List[BoundingBox]:
        """
        Executes sequential EKF observation updates for a primary detection and all
        corroborating additional detections within proximity.

        Each additional detection contracts state covariance, reinforcing certainty.

        Returns:
            List of successfully fused additional bounding boxes.
        """
        # 1. Update with primary detection
        self.update(
            box=primary_box,
            H_image_to_field=H_image_to_field,
            field_coords=primary_field,
            r_scale=1.0,
        )

        fused_additional: List[BoundingBox] = []

        # 2. Sequential update for each additional detection within proximity
        for idx, add_box in enumerate(additional_boxes):
            add_f = additional_fields[idx] if additional_fields and idx < len(additional_fields) else None
            # Emulated IoU determines the measurement confidence weight
            emul_iou = emulate_iou(
                primary_box,
                add_box,
                field_coords1=primary_field,
                field_coords2=add_f,
            )

            # High emulated IoU -> smaller R noise (stronger reinforcement)
            # Minimum scale 0.5, maximum 3.0
            r_scale = float(np.clip(1.5 / max(0.2, emul_iou), 0.5, 3.0))

            self.update(
                box=add_box,
                H_image_to_field=H_image_to_field,
                field_coords=add_f,
                r_scale=r_scale,
            )
            fused_additional.append(add_box)

        return fused_additional

    def get_pixel_bbox(
        self,
        confidence: float = 0.85,
        H_image_to_field: Optional[np.ndarray] = None,
    ) -> BoundingBox:
        """
        Extracts current predicted/estimated pixel bounding box.
        If homography is available, projects (X, Y) to image coordinates.
        """
        w = self.pixel_w
        h = self.pixel_h

        if H_image_to_field is not None:
            try:
                H_field_to_image = np.linalg.inv(H_image_to_field)
                pt_f = np.array([self.x[0], self.x[1], 1.0], dtype=np.float64)
                proj = H_field_to_image @ pt_f
                w_p = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
                u_ground = proj[0] / w_p
                v_ground = proj[1] / w_p
                center_x = float(u_ground)
                center_y = float(v_ground - 0.40 * h)
                return BoundingBox(
                    center_x=center_x,
                    center_y=center_y,
                    width=w,
                    height=h,
                    confidence=confidence,
                    class_id=0,
                )
            except Exception:
                pass

        # Default fallback
        return BoundingBox(
            center_x=0.0,
            center_y=0.0,
            width=w,
            height=h,
            confidence=confidence,
            class_id=0,
        )
