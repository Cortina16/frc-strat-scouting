"""
Vision processing modules for the FRC 2026 REBUILT scouting architecture.

Includes:
- AprilTag fiducial marker detection and canonical field tag mapping.
- Dynamic planar homography estimation and camera motion compensation (CMC).
- TensorRT YOLOv11 robot detection actor.
- Real-ESRGAN super-resolution and bumper OCR recognition actor.
"""

from src.vision.apriltag_detector import AprilTagDetector, CANONICAL_FIELD_TAGS
from src.vision.ball_detector import YellowBallDetector, detect_flying_balls_near_hubs
from src.vision.bumper_detector import (
    BumperDetection,
    BumperDetector,
    find_best_bumper_model_path,
)
from src.vision.motion_contour import (
    DynamicMotionResult,
    DynamicMotionSegmenter,
    MotionGuidedYOLOVerifier,
)
from src.vision.detector_trt import (
    RobotBumperVerificationGate,
    TensorRTDetectorActor,
    classify_bumper_fabric,
    decode_yolo_boxes,
    dynamic_waterfall_selection,
    extract_top_screen_tiles,
    letterbox_image,
    recover_missing_bumper_proposals,
    suppress_duplicate_boxes,
)
from src.vision.homography import (
    HomographyActor,
    compute_cmc_affine_detailed,
    compute_normalized_dlt,
    detect_camera_cut,
    estimate_camera_pose_pnp,
    optimize_homography_lm,
    project_pixel_to_field_3d,
    project_pixels_to_field_3d_batch,
    project_pixels_to_field_batch,
)
from src.vision.ocr_enhancer import (
    DigitTemplateBank,
    OCREnhancerActor,
    classify_alliance,
    extract_bumper_crop,
    get_template_bank,
    recognize_team_number,
    super_resolve_crop,
)
from src.vision.robot_ekf import (
    RobotEKF,
    emulate_iou,
    fuse_bounding_boxes,
    pairwise_emulate_iou,
)

__all__ = [
    # AprilTag & Homography
    "AprilTagDetector",
    "CANONICAL_FIELD_TAGS",
    "HomographyActor",
    "compute_normalized_dlt",
    "optimize_homography_lm",
    "estimate_camera_pose_pnp",
    "project_pixel_to_field_3d",
    "project_pixels_to_field_3d_batch",
    "project_pixels_to_field_batch",
    "compute_cmc_affine_detailed",
    "detect_camera_cut",
    # TensorRT YOLOv11 Detector & Ball Detector
    "TensorRTDetectorActor",
    "YellowBallDetector",
    "RobotBumperVerificationGate",
    "letterbox_image",
    "decode_yolo_boxes",
    "suppress_duplicate_boxes",
    "classify_bumper_fabric",
    "extract_top_screen_tiles",
    "dynamic_waterfall_selection",
    "recover_missing_bumper_proposals",
    # Dynamic Motion Contouring & Verification
    "DynamicMotionSegmenter",
    "DynamicMotionResult",
    "MotionGuidedYOLOVerifier",
    # Real-ESRGAN & Bumper OCR
    "OCREnhancerActor",
    "extract_bumper_crop",
    "classify_alliance",
    "super_resolve_crop",
    "recognize_team_number",
    "DigitTemplateBank",
    "get_template_bank",
    # Bumper Perception & Fusion
    "BumperDetector",
    "BumperDetection",
    "find_best_bumper_model_path",
    # Robot EKF & Proximity-Emulated IoU
    "RobotEKF",
    "emulate_iou",
    "pairwise_emulate_iou",
    "fuse_bounding_boxes",
]
