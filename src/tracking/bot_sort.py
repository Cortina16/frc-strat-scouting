"""
BoT-SORT Multi-Object Tracking & Tracklet Bridging Engine.

Implements:
- 8-State modified Kalman filter tracking without aspect-ratio parametrization
- Camera Motion Compensation (CMC) integration
- Two-stage association:
  - Stage 1: High-confidence detection matching via dynamic fusion of IoU and 512-D ReID cosine distance
  - Stage 2: Low-confidence second-chance matching via IoU distance
- Tracklet Bridging state machine:
  - Preserves lost tracks across occlusions (e.g. 72" Alliance Hub) and broadcast camera cuts
  - Re-associates tracks upon re-emergence using ReID cosine similarity
- Field projection: maps pixel centers (x_c, y_c) to CAD ground plane (x, y) inches and calculates velocities (v_x, v_y) ips
- OCR consensus: running consensus and weighted voting for team number and alliance identification
- Conforms to TrackingActorBase and TrackingActorProtocol
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("frc.tracking.bot_sort")

from src.common.interfaces import TrackingActorBase
from src.common.types import (
    AllianceColor,
    BoundingBox,
    HomographyEstimate,
    OCRResult,
    RobotTrackletState,
    TrackState,
)
from src.tracking.cmc import apply_cmc_transform, transform_box_cmc
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.reid import (
    ReIDExtractor,
    cosine_distance,
    normalize_embedding,
)
from src.vision.homography import project_pixel_to_field_3d


def box_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    """
    Calculates pairwise Intersection over Union (IoU) between two sets of bounding boxes.
    Accelerated with PyTorch CUDA tensors on cuda:0 when available.

    Args:
        boxes_a: Array of shape (N, 4) in [x_c, y_c, w, h] format.
        boxes_b: Array of shape (M, 4) in [x_c, y_c, w, h] format.
        device: PyTorch device ('cuda:0' by default if CUDA is available).

    Returns:
        IoU matrix of shape (N, M) with values in [0.0, 1.0].
    """
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    if a.ndim == 1:
        a = a.reshape(1, 4)
    if b.ndim == 1:
        b = b.reshape(1, 4)

    use_cuda = torch.cuda.is_available()
    if use_cuda or device is not None:
        target_dev = device if device is not None else torch.device("cuda:0")
        t_a = torch.as_tensor(a, dtype=torch.float32, device=target_dev)
        t_b = torch.as_tensor(b, dtype=torch.float32, device=target_dev)

        x1_a = t_a[:, 0] - t_a[:, 2] * 0.5
        y1_a = t_a[:, 1] - t_a[:, 3] * 0.5
        x2_a = t_a[:, 0] + t_a[:, 2] * 0.5
        y2_a = t_a[:, 1] + t_a[:, 3] * 0.5
        area_a = t_a[:, 2] * t_a[:, 3]

        x1_b = t_b[:, 0] - t_b[:, 2] * 0.5
        y1_b = t_b[:, 1] - t_b[:, 3] * 0.5
        x2_b = t_b[:, 0] + t_b[:, 2] * 0.5
        y2_b = t_b[:, 1] + t_b[:, 3] * 0.5
        area_b = t_b[:, 2] * t_b[:, 3]

        inter_x1 = torch.maximum(x1_a.unsqueeze(1), x1_b.unsqueeze(0))
        inter_y1 = torch.maximum(y1_a.unsqueeze(1), y1_b.unsqueeze(0))
        inter_x2 = torch.minimum(x2_a.unsqueeze(1), x2_b.unsqueeze(0))
        inter_y2 = torch.minimum(y2_a.unsqueeze(1), y2_b.unsqueeze(0))

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area
        union_area = torch.clamp(union_area, min=1e-12)

        iou = torch.clamp(inter_area / union_area, min=0.0, max=1.0)
        return iou.cpu().numpy().astype(np.float64)
    else:
        # Convert [xc, yc, w, h] to [x1, y1, x2, y2]
        x1_a = a[:, 0] - a[:, 2] / 2.0
        y1_a = a[:, 1] - a[:, 3] / 2.0
        x2_a = a[:, 0] + a[:, 2] / 2.0
        y2_a = a[:, 1] + a[:, 3] / 2.0
        area_a = a[:, 2] * a[:, 3]

        x1_b = b[:, 0] - b[:, 2] / 2.0
        y1_b = b[:, 1] - b[:, 3] / 2.0
        x2_b = b[:, 0] + b[:, 2] / 2.0
        y2_b = b[:, 1] + b[:, 3] / 2.0
        area_b = b[:, 2] * b[:, 3]

        inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
        inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
        inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
        inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = area_a[:, None] + area_b[None, :] - inter_area
        union_area = np.maximum(union_area, 1e-12)

        iou = inter_area / union_area
        return np.clip(iou, 0.0, 1.0)


def iou_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calculates spatial IoU distance (1.0 - IoU) of shape (N, M)."""
    return 1.0 - box_iou(boxes_a, boxes_b)


def fuse_iou_reid_cost(
    iou_dist: np.ndarray,
    reid_dist: Optional[np.ndarray],
    reid_threshold: float = 0.4,
    iou_weight: float = 0.5,
    gate_cost: float = 1e5,
) -> np.ndarray:
    """
    Fuses spatial IoU distance and ReID cosine distance dynamically with strict gating.
    Accelerated with PyTorch CUDA tensors on cuda:0 when available.

    Rules:
    - If reid_dist > reid_threshold: cost is gated to gate_cost (identity mismatch).
    - If IoU == 0 (iou_dist >= 1.0): cost is gated to gate_cost (no spatial overlap).
    - Otherwise: fused_cost = iou_weight * iou_dist + (1.0 - iou_weight) * reid_dist.
    """
    if iou_dist.size == 0:
        return iou_dist.copy()

    if reid_dist is not None and reid_dist.shape == iou_dist.shape:
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            i_t = torch.as_tensor(iou_dist, dtype=torch.float32, device=dev)
            r_t = torch.as_tensor(reid_dist, dtype=torch.float32, device=dev)
            fused_t = iou_weight * i_t + (1.0 - iou_weight) * r_t
            gated_mask = (r_t > reid_threshold) | (i_t >= 1.0)
            fused_t[gated_mask] = gate_cost
            return fused_t.cpu().numpy().astype(np.float64)
        else:
            fused = iou_weight * iou_dist + (1.0 - iou_weight) * reid_dist
            gated_mask = (reid_dist > reid_threshold) | (iou_dist >= 1.0)
            fused[gated_mask] = gate_cost
            return fused
    else:
        # Fallback when ReID is unavailable: gate non-overlapping boxes
        cost = iou_dist.copy()
        cost[cost >= 1.0] = gate_cost
        return cost



def linear_assignment(
    cost_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Performs optimal bipartite matching using the Hungarian algorithm with cost gating.

    Args:
        cost_matrix: Cost array of shape (N, M).
        threshold: Maximum acceptable matching cost.

    Returns:
        Tuple of (matches [(row_idx, col_idx)], unmatched_rows, unmatched_cols).
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_rows: List[int] = list(range(cost_matrix.shape[0]))
    unmatched_cols: List[int] = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold and cost_matrix[r, c] < 1e4:
            matches.append((r, c))
            if r in unmatched_rows:
                unmatched_rows.remove(r)
            if c in unmatched_cols:
                unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols


class STrack:
    """
    Single robot tracklet state machine representation.
    """

    def __init__(
        self,
        track_id: int,
        bbox: BoundingBox,
        embedding: Optional[np.ndarray] = None,
        frame_index: int = 0,
        alpha_reid: float = 0.9,
        red_roster: Optional[Sequence[int]] = None,
        blue_roster: Optional[Sequence[int]] = None,
    ) -> None:
        self.track_id = track_id
        self.state = TrackState.TRACKED
        self.pixel_bbox = bbox
        self.confidence = bbox.confidence
        self.frame_id = frame_index
        self.start_frame = frame_index
        self.time_since_update = 0

        # Modified 8-state Kalman Filter
        init_meas = np.array([bbox.center_x, bbox.center_y, bbox.width, bbox.height], dtype=np.float64)
        self.kalman_filter = KalmanFilter(init_meas)

        # 512-D ReID embedding
        self.alpha_reid = alpha_reid
        self.smooth_embedding: Optional[np.ndarray] = None
        self.reid_embedding: Optional[np.ndarray] = None
        if embedding is not None:
            norm_emb = normalize_embedding(embedding)
            self.smooth_embedding = norm_emb.copy()
            self.reid_embedding = norm_emb.copy()

        # Identification & OCR consensus
        self.team_number: Optional[int] = None
        self.confirmed_team_number: Optional[int] = None
        self.is_team_locked: bool = False
        self.alliance: AllianceColor = AllianceColor.UNKNOWN
        self.is_alliance_locked: bool = False
        self.ocr_team_votes: Dict[int, float] = {}
        self.ocr_alliance_votes: Dict[AllianceColor, float] = {}
        self.best_ocr_conf: float = 0.0
        self.red_roster: Optional[List[int]] = [int(x) for x in red_roster] if red_roster is not None else None
        self.blue_roster: Optional[List[int]] = [int(x) for x in blue_roster] if blue_roster is not None else None
        self.mandatory_attribution: bool = False

        # Physical field coordinates and kinematics
        self.field_x_inch: float = 0.0
        self.field_y_inch: float = 0.0
        self.field_z_inch: float = 0.0
        self.center_rel_x_inch: float = -325.6
        self.center_rel_y_inch: float = -158.85
        self.center_rel_z_inch: float = 0.0
        self.depth_inch: float = 0.0
        self.robot_width_inch: float = 22.0
        self.robot_length_inch: float = 28.0
        self.robot_height_inch: float = 28.0
        self.velocity_x_ips: float = 0.0
        self.velocity_y_ips: float = 0.0
        self.heading_rad: float = 0.0
        self.has_field_coords: bool = False
        self.last_timestamp_ms: Optional[int] = None
        self.last_field_frame: int = frame_index

    def predict(self) -> None:
        """Runs Kalman filter prediction step and updates pixel bounding box center."""
        if self.state in (TrackState.TRACKED, TrackState.BRIDGED):
            self.kalman_filter.predict()
            xc, yc, w, h = self.kalman_filter.bbox
            self.pixel_bbox = BoundingBox(
                center_x=float(xc),
                center_y=float(yc),
                width=float(w),
                height=float(h),
                confidence=self.confidence,
                class_id=self.pixel_bbox.class_id,
            )

    def apply_cmc(self, affine_matrix: Optional[np.ndarray]) -> None:
        """Applies Camera Motion Compensation affine transformation."""
        if affine_matrix is not None and self.kalman_filter.mean is not None:
            self.kalman_filter.apply_cmc(affine_matrix)
            xc, yc, w, h = self.kalman_filter.bbox
            self.pixel_bbox = BoundingBox(
                center_x=float(xc),
                center_y=float(yc),
                width=float(w),
                height=float(h),
                confidence=self.confidence,
                class_id=self.pixel_bbox.class_id,
            )

    @classmethod
    def multi_predict_and_cmc(
        cls,
        tracks: Sequence[STrack],
        affine_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Batched prediction and CMC update for multiple active STrack tracks using GPU PyTorch CUDA.
        """
        active_tracks = [t for t in tracks if t.state in (TrackState.TRACKED, TrackState.BRIDGED)]
        if not active_tracks:
            return
        filters = [t.kalman_filter for t in active_tracks]
        KalmanFilter.multi_predict(filters, affine_matrix=affine_matrix)
        for t in active_tracks:
            xc, yc, w, h = t.kalman_filter.bbox
            t.pixel_bbox = BoundingBox(
                center_x=float(xc),
                center_y=float(yc),
                width=float(w),
                height=float(h),
                confidence=t.confidence,
                class_id=t.pixel_bbox.class_id,
            )


    def update(
        self,
        new_bbox: BoundingBox,
        new_embedding: Optional[np.ndarray],
        frame_index: int,
    ) -> None:
        """Updates tracklet state with a matched detection."""
        self.frame_id = frame_index
        self.time_since_update = 0
        self.confidence = new_bbox.confidence

        measurement = np.array([new_bbox.center_x, new_bbox.center_y, new_bbox.width, new_bbox.height], dtype=np.float64)
        self.kalman_filter.update(measurement, confidence=new_bbox.confidence)

        xc, yc, w, h = self.kalman_filter.bbox
        self.pixel_bbox = BoundingBox(
            center_x=float(xc),
            center_y=float(yc),
            width=float(w),
            height=float(h),
            confidence=new_bbox.confidence,
            class_id=new_bbox.class_id,
        )

        if new_embedding is not None:
            new_norm = normalize_embedding(new_embedding)
            if self.smooth_embedding is None:
                self.smooth_embedding = new_norm.copy()
            else:
                self.smooth_embedding = self.alpha_reid * self.smooth_embedding + (1.0 - self.alpha_reid) * new_norm
                self.smooth_embedding = normalize_embedding(self.smooth_embedding)
            self.reid_embedding = self.smooth_embedding.copy()

        self.state = TrackState.TRACKED

    def bridge(
        self,
        new_bbox: BoundingBox,
        new_embedding: Optional[np.ndarray],
        frame_index: int,
    ) -> None:
        """Re-activates a lost tracklet via high-confidence ReID bridging."""
        self.frame_id = frame_index
        self.time_since_update = 0
        self.confidence = new_bbox.confidence

        measurement = np.array([new_bbox.center_x, new_bbox.center_y, new_bbox.width, new_bbox.height], dtype=np.float64)
        self.kalman_filter.initiate(measurement)

        self.pixel_bbox = new_bbox

        if new_embedding is not None:
            new_norm = normalize_embedding(new_embedding)
            if self.smooth_embedding is None:
                self.smooth_embedding = new_norm.copy()
            else:
                self.smooth_embedding = 0.5 * self.smooth_embedding + 0.5 * new_norm
                self.smooth_embedding = normalize_embedding(self.smooth_embedding)
            self.reid_embedding = self.smooth_embedding.copy()

        self.state = TrackState.BRIDGED

    def update_ocr(self, ocr: Optional[OCRResult]) -> None:
        """Maintains running weighted consensus for team number and alliance with strict alliance purity."""
        if ocr is None or ocr.confidence <= 0.0:
            return

        conf = float(ocr.confidence)

        # Alliance Purity Guardrail:
        # A Red robot CANNOT have a Blue team number, and a Blue robot CANNOT have a Red team number.
        if ocr.team_number is not None:
            team_val = int(ocr.team_number)
            if self.alliance == AllianceColor.RED and self.blue_roster and team_val in self.blue_roster:
                logger.debug("Alliance Purity: rejected blue team %d on red track %d", team_val, self.track_id)
            elif self.alliance == AllianceColor.BLUE and self.red_roster and team_val in self.red_roster:
                logger.debug("Alliance Purity: rejected red team %d on blue track %d", team_val, self.track_id)
            else:
                self.ocr_team_votes[team_val] = self.ocr_team_votes.get(team_val, 0.0) + conf
                valid_votes = {
                    t: v for t, v in self.ocr_team_votes.items()
                    if not (self.alliance == AllianceColor.RED and self.blue_roster and t in self.blue_roster)
                    and not (self.alliance == AllianceColor.BLUE and self.red_roster and t in self.red_roster)
                }
                if valid_votes:
                    self.team_number = max(valid_votes, key=valid_votes.get)

        if ocr.alliance and ocr.alliance != AllianceColor.UNKNOWN and not self.is_alliance_locked:
            self.ocr_alliance_votes[ocr.alliance] = self.ocr_alliance_votes.get(ocr.alliance, 0.0) + conf
            self.alliance = max(self.ocr_alliance_votes, key=self.ocr_alliance_votes.get)

        # Infer alliance from verified team number if alliance was unknown
        if self.alliance == AllianceColor.UNKNOWN and self.team_number is not None:
            if self.red_roster and self.team_number in self.red_roster:
                self.alliance = AllianceColor.RED
            elif self.blue_roster and self.team_number in self.blue_roster:
                self.alliance = AllianceColor.BLUE

        if conf > self.best_ocr_conf:
            self.best_ocr_conf = conf

    def update_field_geometry(
        self,
        h_matrix: Optional[Union[np.ndarray, HomographyEstimate]],
        timestamp_ms: Optional[int] = None,
        frame_index: int = 0,
        target_fps: float = 60.0,
        homography_estimate: Optional[HomographyEstimate] = None,
        is_autonomous: bool = False,
    ) -> None:
        """
        Projects robot ground contact patch (bounding box bottom center) into field CAD coordinates
        (x, y) in inches on the carpet floor (Z=0), computes 3D Euclidean camera depth, and calculates velocities.
        """
        est = homography_estimate
        if isinstance(h_matrix, HomographyEstimate):
            est = h_matrix
            h_mat = est.h_matrix
        else:
            h_mat = h_matrix

        xc = self.pixel_bbox.center_x
        yc = self.pixel_bbox.center_y

        # For elevated broadcast views where 3D camera pose is solved, project the carpet contact patch (bottom center)
        # using full 3D ray back-projection; for planar homographies without camera pose, use bbox center.
        if est is not None and est.camera_center is not None and est.camera_pose_rvec is not None:
            yc_contact = yc + 0.5 * self.pixel_bbox.height
            f_cam = est.focal_length or 1250.6
            K_cam = np.array([[f_cam, 0.0, 960.0], [0.0, f_cam, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            c_3d, r_depth = project_pixel_to_field_3d(
                pixel_u=xc,
                pixel_v=yc_contact,
                elevation_inch=0.0,
                rvec=est.camera_pose_rvec,
                tvec=est.camera_pose_tvec,
                camera_matrix=K_cam,
                camera_center=est.camera_center,
                h_floor=h_mat,
            )
            new_x = float(c_3d[0])
            new_y = float(c_3d[1])
            self.depth_inch = float(r_depth)
        elif h_mat is not None:
            yc_contact = yc + (0.5 * self.pixel_bbox.height if (est is not None and est.camera_center is not None) else 0.0)
            try:
                H_inv = np.linalg.inv(h_mat)
                p_hom = np.array([xc, yc_contact, 1.0], dtype=np.float64)
                proj = np.dot(H_inv, p_hom)
                z = proj[2]
                if abs(z) < 1e-12:
                    z = 1e-12 if z >= 0 else -1e-12
                new_x = float(proj[0] / z)
                new_y = float(proj[1] / z)
            except np.linalg.LinAlgError:
                new_x, new_y = xc, yc
            self.depth_inch = float(np.linalg.norm(np.array([new_x, new_y, 0.0]) - est.camera_center)) if (est is not None and est.camera_center is not None) else 0.0
        else:
            new_x, new_y = xc, yc
            self.depth_inch = 0.0

        # Hub Physical Exclusion Zone Enforcement:
        # Red Hub at (162.8, 158.85), Blue Hub at (488.4, 158.85).
        # Solid base radius (~23.5") + robot half-width (~11") => minimum physical clearance = 34.0"
        min_hub_clearance = 33.5
        dist_red = np.hypot(new_x - 162.8, new_y - 158.85)
        dist_blue = np.hypot(new_x - 488.4, new_y - 158.85)
        if dist_red < min_hub_clearance and dist_red > 1e-3:
            scale = min_hub_clearance / dist_red
            new_x = 162.8 + (new_x - 162.8) * scale
            new_y = 158.85 + (new_y - 158.85) * scale
        elif dist_blue < min_hub_clearance and dist_blue > 1e-3:
            scale = min_hub_clearance / dist_blue
            new_x = 488.4 + (new_x - 488.4) * scale
            new_y = 158.85 + (new_y - 158.85) * scale

        # Boundary guardrails
        new_x = float(np.clip(new_x, 10.0, 641.2))
        new_y = float(np.clip(new_y, 10.0, 307.7))

        # Autonomous Phase Centerline Crossing Restriction
        # Under FRC rules, a robot cannot cross 100% over the center line (X = 325.6") during AUTO.
        # Red side: X <= 325.6", Blue side: X >= 325.6".
        if is_autonomous:
            if self.alliance == AllianceColor.RED and new_x > 335.0:
                new_x = 325.6
            elif self.alliance == AllianceColor.BLUE and new_x < 315.0:
                new_x = 325.6
            elif self.alliance == AllianceColor.UNKNOWN:
                if new_x < 315.0:
                    self.alliance = AllianceColor.RED
                elif new_x > 335.0:
                    self.alliance = AllianceColor.BLUE

        self.field_z_inch = 0.0  # Robot carpet contact elevation

        # Field-center relative coordinates (Center = 325.6, 158.85, 0.0)
        self.center_rel_x_inch = float(new_x - 325.6)
        self.center_rel_y_inch = float(new_y - 158.85)
        self.center_rel_z_inch = float(self.field_z_inch)

        # Estimate live robot dimensions from bounding box, distance, and focal length
        # Nominal FRC robot base is roughly 22 inches by 28 inches
        if (
            est is not None
            and est.focal_length is not None
            and est.focal_length > 10.0
            and self.depth_inch > 10.0
            and self.pixel_bbox is not None
        ):
            w_live = float(self.pixel_bbox.width * self.depth_inch / est.focal_length)
            h_live = float(self.pixel_bbox.height * self.depth_inch / est.focal_length)
            w_est = float(np.clip(w_live, 18.0, 36.0))
            h_est = float(np.clip(h_live, 12.0, 48.0))
            l_est = float(np.clip(28.0 * (w_est / 22.0), 20.0, 38.0))
            # Smooth dimension estimates
            self.robot_width_inch = float(0.15 * w_est + 0.85 * self.robot_width_inch)
            self.robot_height_inch = float(0.15 * h_est + 0.85 * self.robot_height_inch)
            self.robot_length_inch = float(0.15 * l_est + 0.85 * self.robot_length_inch)
        else:
            self.robot_width_inch = 22.0
            self.robot_length_inch = 28.0
            self.robot_height_inch = 28.0

        if self.has_field_coords:
            if (
                timestamp_ms is not None
                and self.last_timestamp_ms is not None
                and timestamp_ms > self.last_timestamp_ms
            ):
                dt = (timestamp_ms - self.last_timestamp_ms) / 1000.0
            elif frame_index > self.last_field_frame:
                dt = (frame_index - self.last_field_frame) / target_fps
            else:
                dt = 1.0 / target_fps

            dt = max(dt, 1e-4)
            inst_vx = (new_x - self.field_x_inch) / dt
            inst_vy = (new_y - self.field_y_inch) / dt

            # Smooth field velocities
            self.velocity_x_ips = 0.7 * inst_vx + 0.3 * self.velocity_x_ips
            self.velocity_y_ips = 0.7 * inst_vy + 0.3 * self.velocity_y_ips

            speed = np.hypot(self.velocity_x_ips, self.velocity_y_ips)
            if speed > 1.0:
                self.heading_rad = float(np.arctan2(self.velocity_y_ips, self.velocity_x_ips))
        else:
            self.velocity_x_ips = 0.0
            self.velocity_y_ips = 0.0
            self.heading_rad = 0.0
            self.has_field_coords = True

        self.field_x_inch = new_x
        self.field_y_inch = new_y
        self.last_timestamp_ms = timestamp_ms
        self.last_field_frame = frame_index

    # Alias for update_field_geometry
    update_field_coords = update_field_geometry

    def to_robot_tracklet_state(self) -> RobotTrackletState:
        """Converts internal track state to public DTO RobotTrackletState."""
        t_num = self.team_number if self.team_number is not None else self.confirmed_team_number
        # Strict Alliance Purity Guardrail
        if self.alliance == AllianceColor.RED and self.blue_roster and t_num in self.blue_roster:
            t_num = None
        elif self.alliance == AllianceColor.BLUE and self.red_roster and t_num in self.red_roster:
            t_num = None

        # Mandatory attribution fallback: ensure non-null team number if alliance roster is known
        if self.mandatory_attribution and t_num is None:
            if self.alliance == AllianceColor.RED and self.red_roster:
                t_num = self.red_roster[self.track_id % len(self.red_roster)]
            elif self.alliance == AllianceColor.BLUE and self.blue_roster:
                t_num = self.blue_roster[self.track_id % len(self.blue_roster)]

        return RobotTrackletState(
            track_id=self.track_id,
            team_number=t_num,
            alliance=self.alliance,
            pixel_bbox=self.pixel_bbox,
            field_x_inch=self.field_x_inch,
            field_y_inch=self.field_y_inch,
            velocity_x_ips=self.velocity_x_ips,
            velocity_y_ips=self.velocity_y_ips,
            heading_rad=self.heading_rad,
            track_state=self.state,
            field_z_inch=self.field_z_inch,
            depth_inch=self.depth_inch,
            center_rel_x_inch=self.center_rel_x_inch,
            center_rel_y_inch=self.center_rel_y_inch,
            center_rel_z_inch=self.center_rel_z_inch,
            robot_width_inch=self.robot_width_inch,
            robot_length_inch=self.robot_length_inch,
            robot_height_inch=self.robot_height_inch,
            reid_embedding=self.reid_embedding.copy() if self.reid_embedding is not None else None,
        )


class BoTSORT:
    """
    BoT-SORT tracking algorithm engine.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.7,
        stage2_match_thresh: float = 0.5,
        reid_thresh: float = 0.4,
        reid_bridge_thresh: float = 0.4,
        max_lost_frames: int = 300,
        max_coast_frames: int = 0,
        target_fps: float = 60.0,
        red_teams: Optional[Sequence[int]] = None,
        blue_teams: Optional[Sequence[int]] = None,
        mandatory_attribution: bool = False,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.stage2_match_thresh = stage2_match_thresh
        self.reid_thresh = reid_thresh
        self.reid_bridge_thresh = reid_bridge_thresh
        self.max_lost_frames = max_lost_frames
        self.max_coast_frames = max_coast_frames
        self.target_fps = target_fps
        self.mandatory_attribution = mandatory_attribution

        self.red_teams: Optional[List[int]] = [int(t) for t in red_teams] if red_teams is not None else None
        self.blue_teams: Optional[List[int]] = [int(t) for t in blue_teams] if blue_teams is not None else None

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self._next_id: int = 1
        self.frame_id: int = 0

    def set_roster(
        self,
        red_teams: Optional[Sequence[int]] = None,
        blue_teams: Optional[Sequence[int]] = None,
    ) -> None:
        """Sets the alliance roster for bipartite team number matching."""
        self.red_teams = [int(t) for t in red_teams] if red_teams is not None else None
        self.blue_teams = [int(t) for t in blue_teams] if blue_teams is not None else None
        for t in self.tracked_tracks + self.lost_tracks:
            t.red_roster = self.red_teams
            t.blue_roster = self.blue_teams

    def update(
        self,
        detections: Sequence[BoundingBox],
        embeddings: Optional[Sequence[Optional[np.ndarray]]],
        ocr_results: Optional[Sequence[Optional[OCRResult]]],
        homography: Optional[HomographyEstimate],
        frame_index: int,
        is_autonomous: bool = False,
    ) -> List[RobotTrackletState]:
        """
        Processes a single frame of detections and updates active robot tracks.
        """
        self.frame_id = frame_index
        camera_cut = homography.camera_cut if homography is not None else False
        affine_cmc = homography.affine_cmc if homography is not None else None

        # 1. Prediction & CMC on existing tracks via batched GPU Kalman filter
        if len(self.tracked_tracks) > 0:
            STrack.multi_predict_and_cmc(self.tracked_tracks, affine_cmc)

        for track in self.lost_tracks:
            track.time_since_update += 1

        # 2. Partition detections by confidence
        high_dets: List[BoundingBox] = []
        high_embs: List[Optional[np.ndarray]] = []
        high_indices: List[int] = []

        low_dets: List[BoundingBox] = []
        low_embs: List[Optional[np.ndarray]] = []
        low_indices: List[int] = []

        for idx, det in enumerate(detections):
            emb = embeddings[idx] if embeddings is not None and idx < len(embeddings) else None
            if det.confidence >= self.track_high_thresh:
                high_dets.append(det)
                high_embs.append(emb)
                high_indices.append(idx)
            elif det.confidence >= self.track_low_thresh:
                low_dets.append(det)
                low_embs.append(emb)
                low_indices.append(idx)

        # 3. Handle camera cut: active tracks transition to LOST immediately
        if camera_cut:
            for track in self.tracked_tracks:
                track.state = TrackState.LOST
                track.time_since_update += 1
                self.lost_tracks.append(track)
            self.tracked_tracks.clear()

        unmatched_active_tracks: List[STrack] = []
        unmatched_high_dets = high_dets
        unmatched_high_embs = high_embs
        unmatched_high_indices = high_indices

        if not camera_cut and len(self.tracked_tracks) > 0:
            # Stage 1: Dynamic fusion of high-confidence bounding box IoU and ReID cosine distance
            if len(high_dets) > 0:
                track_boxes = np.array([
                    [t.pixel_bbox.center_x, t.pixel_bbox.center_y, t.pixel_bbox.width, t.pixel_bbox.height]
                    for t in self.tracked_tracks
                ], dtype=np.float64)
                det_boxes = np.array([
                    [d.center_x, d.center_y, d.width, d.height]
                    for d in high_dets
                ], dtype=np.float64)

                iou_dist = iou_distance(track_boxes, det_boxes)

                # Compute pairwise ReID cosine distance using GPU tensor batching
                reid_dist: Optional[np.ndarray] = None
                has_embs = any(t.reid_embedding is not None for t in self.tracked_tracks) and any(e is not None for e in high_embs)
                if has_embs:
                    all_track_embs = all(t.reid_embedding is not None for t in self.tracked_tracks)
                    all_det_embs = all(e is not None for e in high_embs)
                    if all_track_embs and all_det_embs:
                        tr_embs = np.stack([t.reid_embedding for t in self.tracked_tracks])
                        dt_embs = np.stack([e for e in high_embs])
                        reid_dist = np.asarray(cosine_distance(tr_embs, dt_embs), dtype=np.float32)
                    else:
                        reid_dist = np.full((len(self.tracked_tracks), len(high_dets)), 0.5, dtype=np.float32)
                        valid_tr_idx = [i for i, t in enumerate(self.tracked_tracks) if t.reid_embedding is not None]
                        valid_dt_idx = [j for j, e in enumerate(high_embs) if e is not None]
                        if valid_tr_idx and valid_dt_idx:
                            tr_embs = np.stack([self.tracked_tracks[i].reid_embedding for i in valid_tr_idx])
                            dt_embs = np.stack([high_embs[j] for j in valid_dt_idx])
                            sub_dist = np.asarray(cosine_distance(tr_embs, dt_embs), dtype=np.float32)
                            reid_dist[np.ix_(valid_tr_idx, valid_dt_idx)] = sub_dist

                fused_cost = fuse_iou_reid_cost(
                    iou_dist,
                    reid_dist,
                    reid_threshold=self.reid_thresh,
                    iou_weight=0.5,
                )

                matches_s1, unmatched_tr_s1, unmatched_dt_s1 = linear_assignment(fused_cost, threshold=self.match_thresh)

                for t_idx, d_idx in matches_s1:
                    tr = self.tracked_tracks[t_idx]
                    dt = high_dets[d_idx]
                    em = high_embs[d_idx]
                    orig_idx = high_indices[d_idx]
                    tr.update(dt, em, frame_index)
                    if ocr_results is not None and orig_idx < len(ocr_results):
                        tr.update_ocr(ocr_results[orig_idx])

                rem_tracked = [self.tracked_tracks[i] for i in unmatched_tr_s1]
                unmatched_high_dets = [high_dets[i] for i in unmatched_dt_s1]
                unmatched_high_embs = [high_embs[i] for i in unmatched_dt_s1]
                unmatched_high_indices = [high_indices[i] for i in unmatched_dt_s1]
            else:
                rem_tracked = list(self.tracked_tracks)

            # Stage 2: Second-chance association for low-confidence detections using IoU
            if len(rem_tracked) > 0 and len(low_dets) > 0:
                track_boxes_s2 = np.array([
                    [t.pixel_bbox.center_x, t.pixel_bbox.center_y, t.pixel_bbox.width, t.pixel_bbox.height]
                    for t in rem_tracked
                ], dtype=np.float64)
                det_boxes_s2 = np.array([
                    [d.center_x, d.center_y, d.width, d.height]
                    for d in low_dets
                ], dtype=np.float64)

                iou_dist_s2 = iou_distance(track_boxes_s2, det_boxes_s2)
                matches_s2, unmatched_tr_s2, _ = linear_assignment(iou_dist_s2, threshold=self.stage2_match_thresh)

                for t_idx, d_idx in matches_s2:
                    tr = rem_tracked[t_idx]
                    dt = low_dets[d_idx]
                    em = low_embs[d_idx]
                    orig_idx = low_indices[d_idx]
                    tr.update(dt, em, frame_index)
                    if ocr_results is not None and orig_idx < len(ocr_results):
                        tr.update_ocr(ocr_results[orig_idx])

                unmatched_active_tracks = [rem_tracked[i] for i in unmatched_tr_s2]
            else:
                unmatched_active_tracks = rem_tracked

        # Stage 3: Reconnect lost tracklets upon re-emergence using ReID cosine similarity
        remaining_new_dets = unmatched_high_dets
        remaining_new_embs = unmatched_high_embs
        remaining_new_indices = unmatched_high_indices

        if len(unmatched_high_dets) > 0 and len(self.lost_tracks) > 0:
            lost_with_emb = [(idx, t) for idx, t in enumerate(self.lost_tracks) if t.reid_embedding is not None]
            dets_with_emb = [(idx, e) for idx, e in enumerate(unmatched_high_embs) if e is not None]

            if len(lost_with_emb) > 0 and len(dets_with_emb) > 0:
                lost_embs = np.stack([lost_tr.reid_embedding for _, lost_tr in lost_with_emb])
                det_embs = np.stack([det_emb for _, det_emb in dets_with_emb])
                dist_mat = np.asarray(cosine_distance(lost_embs, det_embs), dtype=np.float32)
                cost_reid = np.where(dist_mat <= self.reid_bridge_thresh, dist_mat, 1e5).astype(np.float32)

                matches_bridge, _, _ = linear_assignment(cost_reid, threshold=self.reid_bridge_thresh)


                bridged_lost_indices: List[int] = []
                bridged_det_indices: List[int] = []

                for r, c in matches_bridge:
                    l_idx, lost_track = lost_with_emb[r]
                    d_idx, det_emb = dets_with_emb[c]
                    det_box = unmatched_high_dets[d_idx]
                    orig_idx = unmatched_high_indices[d_idx]

                    # Reconnect lost tracklet
                    lost_track.bridge(det_box, det_emb, frame_index)
                    if ocr_results is not None and orig_idx < len(ocr_results):
                        lost_track.update_ocr(ocr_results[orig_idx])

                    self.tracked_tracks.append(lost_track)
                    bridged_lost_indices.append(l_idx)
                    bridged_det_indices.append(d_idx)

                # Remove bridged tracks from lost_tracks
                self.lost_tracks = [t for idx, t in enumerate(self.lost_tracks) if idx not in bridged_lost_indices]

                # Filter out consumed detections
                remaining_new_dets = [d for idx, d in enumerate(unmatched_high_dets) if idx not in bridged_det_indices]
                remaining_new_embs = [e for idx, e in enumerate(unmatched_high_embs) if idx not in bridged_det_indices]
                remaining_new_indices = [i for idx, i in enumerate(unmatched_high_indices) if idx not in bridged_det_indices]

        # Stage 4: Initialize new tracks from unmatched high-confidence detections
        for det, emb, orig_idx in zip(remaining_new_dets, remaining_new_embs, remaining_new_indices):
            if det.confidence >= self.new_track_thresh:
                new_track = STrack(
                    track_id=self._next_id,
                    bbox=det,
                    embedding=emb,
                    frame_index=frame_index,
                    red_roster=self.red_teams,
                    blue_roster=self.blue_teams,
                )
                new_track.mandatory_attribution = self.mandatory_attribution
                self._next_id += 1
                if ocr_results is not None and orig_idx < len(ocr_results):
                    new_track.update_ocr(ocr_results[orig_idx])
                self.tracked_tracks.append(new_track)

        # Stage 5: Track persistence & Kalman coasting
        # Retain unmatched tracks in active tracking with Kalman forward prediction
        # for up to `max_coast_frames` (default 15 frames / 250ms) to eliminate flickering/dropouts.
        if len(unmatched_active_tracks) > 0:
            lost_to_transition: List[STrack] = []
            for t in unmatched_active_tracks:
                t.time_since_update += 1
                if t.time_since_update > self.max_coast_frames:
                    t.state = TrackState.LOST
                    lost_to_transition.append(t)
                    self.lost_tracks.append(t)
            if lost_to_transition:
                lost_ids = {t.track_id for t in lost_to_transition}
                self.tracked_tracks = [t for t in self.tracked_tracks if t.track_id not in lost_ids]

        # Stage 6: Prune expired lost tracks older than max_lost_frames
        retained_lost: List[STrack] = []
        for t in self.lost_tracks:
            if t.time_since_update > self.max_lost_frames:
                t.state = TrackState.REMOVED
                self.removed_tracks.append(t)
            else:
                retained_lost.append(t)
        self.lost_tracks = retained_lost

        # 7. Update physical field coordinates and velocities
        h_mat = homography.h_matrix if homography is not None else None
        ts_ms = homography.timestamp_ms if homography is not None else None

        for track in self.tracked_tracks:
            track.update_field_geometry(
                h_matrix=h_mat,
                timestamp_ms=ts_ms,
                frame_index=frame_index,
                target_fps=self.target_fps,
                homography_estimate=homography,
                is_autonomous=is_autonomous,
            )

        # 8. Reconcile team assignments across alliances via bipartite matching
        self._reconcile_alliance_team_assignments()

        return [t.to_robot_tracklet_state() for t in self.tracked_tracks]

    def _reconcile_alliance_team_assignments(self) -> None:
        """
        Applies Bayesian bipartite matching via linear sum assignment to assign unique
        candidate team numbers to active tracks on each alliance.
        Guarantees that no two robots on the same alliance share the same team number,
        and enforces mandatory team attribution from the roster.
        """
        if not self.red_teams and not self.blue_teams:
            return

        for alliance, candidate_teams in [
            (AllianceColor.RED, self.red_teams),
            (AllianceColor.BLUE, self.blue_teams),
        ]:
            if not candidate_teams:
                continue

            # Check for tracks that might belong to this alliance based on team votes or field position
            for track in self.tracked_tracks:
                if track.is_alliance_locked:
                    continue
                if track.alliance == AllianceColor.UNKNOWN:
                    if track.ocr_team_votes:
                        red_v = sum(track.ocr_team_votes.get(t, 0.0) for t in (self.red_teams or []))
                        blue_v = sum(track.ocr_team_votes.get(t, 0.0) for t in (self.blue_teams or []))
                        if alliance == AllianceColor.RED and red_v > 0 and red_v >= blue_v:
                            track.alliance = AllianceColor.RED
                        elif alliance == AllianceColor.BLUE and blue_v > 0 and blue_v > red_v:
                            track.alliance = AllianceColor.BLUE
                    elif track.has_field_coords and track.field_x_inch > 0:
                        if alliance == AllianceColor.RED and track.field_x_inch < 325.6:
                            track.alliance = AllianceColor.RED
                        elif alliance == AllianceColor.BLUE and track.field_x_inch >= 325.6:
                            track.alliance = AllianceColor.BLUE

            alliance_tracks = [t for t in self.tracked_tracks if t.alliance == alliance]
            if not alliance_tracks:
                continue

            num_tracks = len(alliance_tracks)
            num_teams = len(candidate_teams)

            cost_matrix = np.zeros((num_tracks, num_teams), dtype=np.float64)
            for i, tr in enumerate(alliance_tracks):
                for j, tm in enumerate(candidate_teams):
                    cost_matrix[i, j] = -tr.ocr_team_votes.get(tm, 0.0)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_track_indices = set()

            assigned_teams: Set[int] = set()
            for r, c in zip(row_ind, col_ind):
                assigned_track_indices.add(r)
                tr = alliance_tracks[r]
                if cost_matrix[r, c] < 0.0:
                    matched_team = candidate_teams[c]
                    tr.team_number = matched_team
                    assigned_teams.add(matched_team)
                    if tr.ocr_team_votes.get(matched_team, 0.0) >= 0.50:
                        tr.alliance = alliance
                        tr.is_alliance_locked = True
                        tr.confirmed_team_number = matched_team
                        tr.is_team_locked = True
                elif tr.is_team_locked and tr.confirmed_team_number is not None and tr.confirmed_team_number not in assigned_teams:
                    tr.team_number = tr.confirmed_team_number
                    assigned_teams.add(tr.confirmed_team_number)
                else:
                    tr.team_number = None

            for i in range(num_tracks):
                if i not in assigned_track_indices:
                    tr = alliance_tracks[i]
                    if tr.is_team_locked and tr.confirmed_team_number is not None and tr.confirmed_team_number not in assigned_teams:
                        tr.team_number = tr.confirmed_team_number
                        assigned_teams.add(tr.confirmed_team_number)
                    else:
                        tr.team_number = None

            # Deductive Elimination & Mandatory Team Attribution:
            assigned_teams = {t.team_number for t in alliance_tracks if t.team_number is not None}
            unassigned_candidates = [tm for tm in candidate_teams if tm not in assigned_teams]
            unassigned_tracks = [t for t in alliance_tracks if t.team_number is None]

            if len(unassigned_candidates) == 1 and len(unassigned_tracks) == 1:
                deduced_team = unassigned_candidates[0]
                target_track = unassigned_tracks[0]
                target_track.team_number = deduced_team
                target_track.confirmed_team_number = deduced_team
                target_track.is_team_locked = True
                target_track.alliance = alliance
                target_track.is_alliance_locked = True
                target_track.ocr_team_votes[deduced_team] = max(target_track.ocr_team_votes.get(deduced_team, 0.0), 0.80)
                logger.debug("Deduced team %d for track %d on alliance %s by elimination", deduced_team, target_track.track_id, alliance.value)
            elif self.mandatory_attribution and unassigned_tracks and unassigned_candidates:
                for tr in unassigned_tracks:
                    if not unassigned_candidates:
                        break
                    # If track previously had a team number that is available, restore it!
                    if tr.confirmed_team_number in unassigned_candidates:
                        assigned_tm = tr.confirmed_team_number
                        unassigned_candidates.remove(assigned_tm)
                    else:
                        assigned_tm = unassigned_candidates.pop(0)
                    tr.team_number = assigned_tm
                    tr.confirmed_team_number = assigned_tm
                    tr.alliance = alliance
                    logger.debug("Mandatory attribution: attributed team %d to track %d on alliance %s", assigned_tm, tr.track_id, alliance.value)


class TrackingActor(TrackingActorBase):
    """
    Ray/async actor implementing multi-object tracking, CMC, and tracklet bridging.
    Strictly conforms to TrackingActorBase and TrackingActorProtocol.
    """

    def __init__(
        self,
        target_fps: float = 60.0,
        reid_extractor: Optional[ReIDExtractor] = None,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.7,
        reid_thresh: float = 0.4,
        reid_bridge_thresh: float = 0.4,
        max_lost_frames: int = 300,
        max_coast_frames: int = 0,
        red_teams: Optional[Sequence[int]] = None,
        blue_teams: Optional[Sequence[int]] = None,
        mandatory_attribution: bool = False,
    ) -> None:
        self.target_fps = target_fps
        self.reid_extractor = reid_extractor or ReIDExtractor()
        self.tracker = BoTSORT(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            match_thresh=match_thresh,
            reid_thresh=reid_thresh,
            reid_bridge_thresh=reid_bridge_thresh,
            max_lost_frames=max_lost_frames,
            max_coast_frames=max_coast_frames,
            target_fps=target_fps,
            red_teams=red_teams,
            blue_teams=blue_teams,
            mandatory_attribution=mandatory_attribution,
        )

    def set_roster(
        self,
        red_teams: Optional[Sequence[int]] = None,
        blue_teams: Optional[Sequence[int]] = None,
    ) -> None:
        """Sets the alliance roster for bipartite team number matching."""
        self.tracker.set_roster(red_teams, blue_teams)

    async def update_tracks(
        self,
        frame_index: int,
        detections: List[BoundingBox],
        ocr_results: List[OCRResult],
        homography: HomographyEstimate,
        frame_bgr: np.ndarray,
        embeddings: Optional[Sequence[np.ndarray]] = None,
        is_autonomous: Optional[bool] = None,
    ) -> List[RobotTrackletState]:
        """
        Updates Kalman filters with CMC, extracts ReID embeddings, and bridges lost tracks.

        Args:
            frame_index: Monotonically increasing video frame index.
            detections: Localized robot bounding box candidates.
            ocr_results: Super-resolved bumper OCR results matching detection candidates.
            homography: HomographyEstimate containing h_matrix and affine_cmc.
            frame_bgr: Raw video frame tensor (H, W, 3) uint8 or dummy array.
            embeddings: Optional pre-extracted 512-D normalized ReID embeddings (for testing/bypass).
            is_autonomous: Optional flag indicating autonomous period. If None, inferred from metadata.

        Returns:
            List of RobotTrackletState objects representing active robot trajectories.
        """
        # Extract embeddings if not explicitly provided
        if embeddings is None:
            if detections and frame_bgr is not None:
                embs = self.reid_extractor.extract(frame_bgr, detections)
            else:
                embs = [None] * len(detections)
        else:
            embs = list(embeddings)

        is_auto = is_autonomous if is_autonomous is not None else (
            (homography is not None and homography.timestamp_ms is not None and homography.timestamp_ms <= 15000)
            or frame_index < 900
        )

        track_states = self.tracker.update(
            detections=detections,
            embeddings=embs,
            ocr_results=ocr_results,
            homography=homography,
            frame_index=frame_index,
            is_autonomous=is_auto,
        )

        return track_states

    async def bridge_lost_tracklets(
        self,
        camera_cut: bool,
        max_bridge_gap_frames: int = 300,
    ) -> None:
        """
        Performs cosine-distance ReID matching to reconnect trajectories after broadcast cuts,
        and terminates lost tracks older than max_bridge_gap_frames.
        """
        # 1. Prune lost tracks older than max_bridge_gap_frames
        retained: List[STrack] = []
        for t in self.tracker.lost_tracks:
            if t.time_since_update > max_bridge_gap_frames:
                t.state = TrackState.REMOVED
                self.tracker.removed_tracks.append(t)
            else:
                retained.append(t)
        self.tracker.lost_tracks = retained

        # 2. If camera cut is signaled, ensure any active tracks transition to LOST
        if camera_cut and len(self.tracker.tracked_tracks) > 0:
            for t in self.tracker.tracked_tracks:
                t.state = TrackState.LOST
                t.time_since_update += 1
                self.tracker.lost_tracks.append(t)
            self.tracker.tracked_tracks.clear()
            return

        # 3. Match any active tracks to lost tracks via ReID similarity
        if len(self.tracker.tracked_tracks) > 0 and len(self.tracker.lost_tracks) > 0:
            active_with_emb = [(idx, t) for idx, t in enumerate(self.tracker.tracked_tracks) if t.reid_embedding is not None]
            lost_with_emb = [(idx, t) for idx, t in enumerate(self.tracker.lost_tracks) if t.reid_embedding is not None]

            if len(active_with_emb) > 0 and len(lost_with_emb) > 0:
                act_embs = np.stack([a_track.reid_embedding for _, a_track in active_with_emb])
                lst_embs = np.stack([l_track.reid_embedding for _, l_track in lost_with_emb])
                dist_mat = np.asarray(cosine_distance(act_embs, lst_embs), dtype=np.float32)
                cost = np.where(dist_mat <= self.tracker.reid_bridge_thresh, dist_mat, 1e5).astype(np.float32)

                matches, _, _ = linear_assignment(cost, threshold=self.tracker.reid_bridge_thresh)
                merged_lost_ids = set()
                for a_i, l_i in matches:
                    _, a_track = active_with_emb[a_i]
                    _, l_track = lost_with_emb[l_i]
                    # Reconnect: retain original track_id, team_number, alliance
                    a_track.track_id = l_track.track_id
                    if l_track.team_number is not None:
                        a_track.team_number = l_track.team_number
                    if l_track.alliance != AllianceColor.UNKNOWN:
                        a_track.alliance = l_track.alliance
                    for k, v in l_track.ocr_team_votes.items():
                        a_track.ocr_team_votes[k] = a_track.ocr_team_votes.get(k, 0.0) + v
                    a_track.state = TrackState.BRIDGED
                    merged_lost_ids.add(l_track.track_id)

                self.tracker.lost_tracks = [t for t in self.tracker.lost_tracks if t.track_id not in merged_lost_ids]

    async def update_tracks_batch(
        self,
        frames_data: Union[
            List[Tuple[int, List[BoundingBox], List[OCRResult], HomographyEstimate, np.ndarray]],
            List[int],
        ],
        detections_batch: Optional[List[List[BoundingBox]]] = None,
        ocr_batch: Optional[List[List[OCRResult]]] = None,
        homographies: Optional[List[HomographyEstimate]] = None,
        frames: Optional[List[np.ndarray]] = None,
        is_autonomous: Optional[bool] = None,
    ) -> List[List[RobotTrackletState]]:
        """
        Updates multi-object tracks for a batch of consecutive frames in a single call.

        Supports both:
        - List of (frame_index, detections, ocr_results, homography, frame_bgr) tuples
        - Batched separate list arguments (frame_indices, detections_batch, ocr_batch, homographies, frames)

        Args:
            frames_data: List of frame tuples or list of integer frame indices.
            detections_batch: Optional list of bounding box lists for each frame.
            ocr_batch: Optional list of OCRResult lists for each frame.
            homographies: Optional list of HomographyEstimates.
            frames: Optional list of BGR frame numpy arrays.
            is_autonomous: Optional autonomous flag.

        Returns:
            List of RobotTrackletState lists corresponding to each frame in the batch.
        """
        batch_results: List[List[RobotTrackletState]] = []

        if detections_batch is not None:
            # Called with separate batch lists
            frame_indices = cast(List[int], frames_data)
            for i, f_idx in enumerate(frame_indices):
                h_est = homographies[i] if homographies is not None else None
                det_boxes = detections_batch[i]
                ocr_res = ocr_batch[i] if ocr_batch is not None else []
                frame_bgr = frames[i] if frames is not None else np.zeros((1, 1, 3), dtype=np.uint8)

                if h_est is not None and h_est.camera_cut:
                    await self.bridge_lost_tracklets(camera_cut=True)

                active_tracks = await self.update_tracks(
                    frame_index=f_idx,
                    detections=det_boxes,
                    ocr_results=ocr_res,
                    homography=h_est,  # type: ignore[arg-type]
                    frame_bgr=frame_bgr,
                    is_autonomous=is_autonomous,
                )

                if h_est is not None and not h_est.camera_cut:
                    await self.bridge_lost_tracklets(camera_cut=False)

                batch_results.append(active_tracks)
        else:
            # Called with list of tuples: (frame_index, detections, ocr_results, homography, frame_bgr, [embeddings])
            for frame_item in frames_data:  # type: ignore[union-attr]
                frame_idx = frame_item[0]
                dets = frame_item[1]
                ocr = frame_item[2]
                hom = frame_item[3]
                frame_bgr = frame_item[4]
                embs = frame_item[5] if len(frame_item) > 5 else None

                states = await self.update_tracks(
                    frame_index=frame_idx,
                    detections=dets,
                    ocr_results=ocr,
                    homography=hom,
                    frame_bgr=frame_bgr,
                    embeddings=embs,
                    is_autonomous=is_autonomous,
                )
                batch_results.append(states)

        return batch_results

    async def get_active_tracks(self) -> List[RobotTrackletState]:
        """Returns the current state of all active robot trajectories."""
        return [t.to_robot_tracklet_state() for t in self.tracker.tracked_tracks]

