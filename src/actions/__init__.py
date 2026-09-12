"""
FRC 2026 REBUILT Action Recognition Package.

Module 06: ActionFormer Temporal Action Localization.
"""

from src.actions.action_former import (
    ACTION_CLASSES,
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
    ActionFormer,
    ActionRecognitionActor,
    BoundaryRegressionHead,
    ClassificationHead,
    LocalSelfAttentionBlock,
    compute_diou_1d,
    focal_loss,
    sinusoidal_positional_encoding_1d,
    temporal_nms,
)
from src.actions.temporal_pyramid import (
    FEATURE_DIM,
    FIELD_DIAG_INCHES,
    FIELD_LENGTH_INCHES,
    FIELD_WIDTH_INCHES,
    TemporalPyramid,
    TemporalWindowBuffer,
    TrackletFrameRecord,
    conv1d_forward,
    depthwise_conv1d_forward,
    extract_frame_features,
    extract_sequence_features,
    layer_norm_1d,
    relu,
)

__all__ = [
    "ActionFormer",
    "ActionRecognitionActor",
    "BoundaryRegressionHead",
    "ClassificationHead",
    "LocalSelfAttentionBlock",
    "compute_diou_1d",
    "focal_loss",
    "temporal_nms",
    "sinusoidal_positional_encoding_1d",
    "TemporalPyramid",
    "TemporalWindowBuffer",
    "TrackletFrameRecord",
    "extract_frame_features",
    "extract_sequence_features",
    "conv1d_forward",
    "depthwise_conv1d_forward",
    "layer_norm_1d",
    "relu",
    "ACTION_CLASSES",
    "ACTION_TO_INDEX",
    "INDEX_TO_ACTION",
    "FEATURE_DIM",
    "FIELD_LENGTH_INCHES",
    "FIELD_WIDTH_INCHES",
    "FIELD_DIAG_INCHES",
]
