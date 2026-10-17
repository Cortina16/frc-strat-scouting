"""
Strongly-typed data contracts for the 3-stage FRC 2026 REBUILT Spatial Analytics Pipeline.

Encapsulates data structures for:
- Stage 1: Spatial Calibration & Top-Down Metric Homography
- Stage 2: Game Piece (Yellow FUEL) Quantification & Proportional Mass Allocation
- Stage 3: Deterministic Robot Localization & Strict 3v3 Unit Tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import numpy as np

from src.common.types import BoundingBox


# ============================================================================
# Stage 1: Spatial Calibration & Metric Canvas Contracts
# ============================================================================

@dataclass(slots=True)
class AprilTagAnchor:
    """Detected static AprilTag fiducial with world and image coordinates."""
    tag_id: int
    image_center_px: Tuple[float, float]
    field_coords_inch: Tuple[float, float, float]
    corners_px: np.ndarray  # Shape: (4, 2)
    reprojection_error_px: float = 0.0


@dataclass(slots=True)
class FieldGeometry:
    """Ground-truth structural CAD geometry for FRC 2026 REBUILT."""
    length_inches: float = 651.2
    width_inches: float = 317.7
    guardrail_height_inches: float = 20.0
    red_hub_center_inch: Tuple[float, float] = (162.8, 158.85)
    blue_hub_center_inch: Tuple[float, float] = (488.4, 158.85)
    hub_dimensions_inch: Tuple[float, float] = (47.0, 47.0)
    carpet_scale_px_per_inch: float = 2.0  # Yields 1302 x 635 px top-down canvas


@dataclass(slots=True)
class CalibrationResult:
    """Output of Stage 1: Spatial Calibration & Homography."""
    detected_tags: List[AprilTagAnchor]
    field_geometry: FieldGeometry
    H_image_to_field: np.ndarray    # 3x3: (u, v, 1) -> (X_inch, Y_inch, 1)
    H_field_to_image: np.ndarray    # 3x3: (X_inch, Y_inch, 1) -> (u, v, 1)
    H_image_to_canvas: np.ndarray   # 3x3: (u, v, 1) -> (x_canvas, y_canvas, 1)
    canvas_shape: Tuple[int, int]   # (height, width), e.g. (635, 1302)
    reprojection_rmse_px: float
    is_cached: bool
    warped_field_plane: np.ndarray  # Top-down flattened BGR metric image
    red_hub_screen_bbox: Optional[Tuple[float, float, float, float]] = None
    blue_hub_screen_bbox: Optional[Tuple[float, float, float, float]] = None


# ============================================================================
# Stage 2: Game Piece (Yellow FUEL) Quantification Contracts
# ============================================================================

@dataclass(slots=True)
class YellowBlobCluster:
    """Individual segmented cluster of yellow game pieces (FUEL)."""
    blob_id: int
    contour_px: np.ndarray
    pixel_area: float
    percentage_total_area: float     # Percentage of total yellow pixels [0.0, 100.0]
    estimated_count: int             # Proportional mass allocation (out of max capacity)
    canvas_centroid_px: Tuple[float, float]
    field_coords_inch: Tuple[float, float]
    bounding_box_canvas: Tuple[int, int, int, int]  # (x, y, w, h)

    @property
    def field_centroid_inch(self) -> Tuple[float, float]:
        """Convenience alias for field_coords_inch."""
        return self.field_coords_inch


@dataclass(slots=True)
class GamePieceQuantificationResult:
    """Output of Stage 2: Yellow Blob Quantification & Proportional Mass Allocation."""
    total_yellow_pixel_area: float
    total_estimated_count: int       # Sum of all blob estimated counts (<= max_field_capacity)
    max_field_capacity: int = 504
    blob_clusters: List[YellowBlobCluster] = field(default_factory=list)
    foreground_mask: Optional[np.ndarray] = None
    yellow_mask: Optional[np.ndarray] = None


# ============================================================================
# Stage 3: Deterministic Robot Localization & Tracking Contracts
# ============================================================================

@dataclass(slots=True)
class RobotCandidateProposal:
    """Raw region proposal before geometric and contextual filtering."""
    proposal_id: int
    pixel_bbox: BoundingBox
    field_coords_inch: Tuple[float, float]
    aspect_ratio: float
    pixel_area: float
    is_inside_field: bool
    filter_reason: Optional[str] = None  # None if passed, reason string if rejected


@dataclass(slots=True)
class TrackedRobotUnit:
    """Authoritative match unit satisfying strict 3v3 cardinality."""
    unit_id: str                      # e.g., "RED_1", "RED_2", "RED_3", "BLUE_1", "BLUE_2", "BLUE_3"
    alliance: str                     # "RED" or "BLUE"
    pixel_bbox: BoundingBox           # Broadcast camera frame bounding box
    canvas_bbox: Tuple[int, int, int, int] # Top-down canvas bounding box (x, y, w, h)
    field_x_inch: float
    field_y_inch: float
    color_confidence: float           # [0.0, 1.0] color purity of bumper fabric
    persistence_score: float          # [0.0, 1.0] track consistency score
    is_projected: bool = False        # True if state is maintained across occlusion
    additional_detections: List[BoundingBox] = field(default_factory=list)


@dataclass(slots=True)
class RobotLocalizationResult:
    """Output of Stage 3: Deterministic Robot Localization ensuring exactly 3v3 units."""
    raw_proposals_count: int
    filtered_proposals_count: int
    candidate_proposals: List[RobotCandidateProposal] = field(default_factory=list)
    red_units: List[TrackedRobotUnit] = field(default_factory=list)   # STRICTLY len == 3
    blue_units: List[TrackedRobotUnit] = field(default_factory=list)  # STRICTLY len == 3
    all_units: List[TrackedRobotUnit] = field(default_factory=list)   # STRICTLY len == 6


# ============================================================================
# Composite Frame Pipeline Result
# ============================================================================

@dataclass(slots=True)
class SpatialPipelineResult:
    """Composite telemetry packet produced for each video frame."""
    frame_index: int
    timestamp_ms: int
    calibration: CalibrationResult
    game_pieces: GamePieceQuantificationResult
    robots: RobotLocalizationResult
    annotated_broadcast_frame: Optional[np.ndarray] = None
    annotated_top_down_plane: Optional[np.ndarray] = None
