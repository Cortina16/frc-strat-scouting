"""
Multi-Object Tracking and Camera Motion Compensation (BoT-SORT) module.
"""

from src.tracking.ball_tracker import BallTrack, BallTracker, SingleBallKalman
from src.tracking.bot_sort import (
    BoTSORT,
    STrack,
    TrackingActor,
    box_iou,
    fuse_iou_reid_cost,
    iou_distance,
    linear_assignment,
)
from src.tracking.cmc import (
    apply_cmc_transform,
    build_cmc_matrix,
    compute_camera_motion_compensation,
    transform_box_cmc,
)
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.reid import (
    ReIDExtractor,
    cosine_distance,
    generate_synthetic_embedding,
    normalize_embedding,
)

__all__ = [
    "KalmanFilter",
    "BoTSORT",
    "TrackingActor",
    "BallTracker",
    "BallTrack",
    "SingleBallKalman",
    "STrack",
    "ReIDExtractor",
    "cosine_distance",
    "normalize_embedding",
    "generate_synthetic_embedding",
    "build_cmc_matrix",
    "apply_cmc_transform",
    "transform_box_cmc",
    "compute_camera_motion_compensation",
    "box_iou",
    "iou_distance",
    "fuse_iou_reid_cost",
    "linear_assignment",
]
