    """
    Common domain data types and transfer objects for the FRC 2026 REBUILT scouting pipeline.
    These types bridge Ray Core actors, inference models, and serialization layers.
    """
    
    from __future__ import annotations
    
    from dataclasses import dataclass, field
    from enum import Enum, auto
    from typing import Any, Dict, List, Optional, Tuple
    
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    
    
    class AllianceColor(str, Enum):
        RED = "RED"
        BLUE = "BLUE"
        UNKNOWN = "UNKNOWN"
    
    
    class MatchStage(str, Enum):
        PRE_MATCH = "PRE_MATCH"
        AUTONOMOUS = "AUTONOMOUS"
        TELEOP = "TELEOP"
        ENDGAME = "ENDGAME"
        POST_MATCH = "POST_MATCH"
    
    
    class TrackState(str, Enum):
        TRACKED = "TRACKED"
        LOST = "LOST"
        BRIDGED = "BRIDGED"
        REMOVED = "REMOVED"
    
    
    class BallState(str, Enum):
        ON_CARPET = "ON_CARPET"
        IN_FLIGHT = "IN_FLIGHT"
        INTAKED = "INTAKED"
        SCORED = "SCORED"
        LOST = "LOST"
        REMOVED = "REMOVED"
    
    
    class ActionType(str, Enum):
        INTAKE_FUEL = "INTAKE_FUEL"
        SCORE_HUB = "SCORE_HUB"
        CLIMB_TOWER = "CLIMB_TOWER"
        DEFENSIVE_ENGAGEMENT = "DEFENSIVE_ENGAGEMENT"
        TRANSIT_NEUTRAL_ZONE = "TRANSIT_NEUTRAL_ZONE"
        OUTPOST_CORRAL_CYCLE = "OUTPOST_CORRAL_CYCLE"
        IDLE = "IDLE"
    
    
    class FieldElement(str, Enum):
        RED_HUB = "RED_HUB"
        BLUE_HUB = "BLUE_HUB"
        RED_TOWER = "RED_TOWER"
        BLUE_TOWER = "BLUE_TOWER"
        RED_OUTPOST = "RED_OUTPOST"
        BLUE_OUTPOST = "BLUE_OUTPOST"
        RED_CORRAL = "RED_CORRAL"
        BLUE_CORRAL = "BLUE_CORRAL"
        DEPOT = "DEPOT"
        BUMP = "BUMP"
        TRENCH = "TRENCH"
        NEUTRAL_ZONE = "NEUTRAL_ZONE"
    
    
    @dataclass(slots=True)
    class FrameMetadata:
        """Metadata describing a single video frame ingested from the broadcast stream."""
        frame_index: int
        pts_ms: int
        monotonic_ts_ns: int
        is_gameplay: bool
        confidence: float
        width: int
        height: int
    
    
    @dataclass(slots=True)
    class RawDecodedFrame:
        """Raw frame tensor and associated ingestion metadata."""
        metadata: FrameMetadata
        image_bgr: np.ndarray  # Shape: (H, W, 3) uint8
    
    
    @dataclass(slots=True)
    class TagDetectionResult:
        """AprilTag detection in pixel space mapped to field CAD coordinate."""
        tag_id: int
        corners_px: np.ndarray  # Shape: (4, 2) float32
        center_px: Tuple[float, float]
        field_coords_inch: Tuple[float, float, float]  # CAD ground truth (x, y, z)
        hamming: int
        reprojection_err: float
    
    
    @dataclass(slots=True)
    class HomographyEstimate:
        """Planar projective transformation matrix and optimization metrics."""
        frame_index: int
        timestamp_ms: int
        h_matrix: np.ndarray  # Shape: (3, 3) float64
        dlt_condition: float
        lm_rmse: float
        lm_iterations: int
        converged: bool
        camera_cut: bool
        affine_cmc: np.ndarray  # Shape: (2, 3) float64
    
    
    @dataclass(slots=True)
    class BoundingBox:
        """Robot bounding box representation in pixel coordinates."""
        center_x: float
        center_y: float
        width: float
        height: float
        confidence: float
        class_id: int
    
    
    @dataclass(slots=True)
    class DetectionResult:
        """Batch of bounding boxes for a specific frame."""
        frame_index: int
        timestamp_ms: int
        boxes: List[BoundingBox] = field(default_factory=list)
    
    
    @dataclass(slots=True)
    class OCRResult:
        """Super-resolved bumper OCR identification result."""
        team_number: Optional[int]
        alliance: AllianceColor
        confidence: float
        super_resolved: bool = False
    
    
    @dataclass(slots=True)
    class RobotTrackletState:
        """Kalman filter tracking state and physical coordinates for a robot."""
        track_id: int
        team_number: Optional[int]
        alliance: AllianceColor
        pixel_bbox: BoundingBox
        field_x_inch: float
        field_y_inch: float
        velocity_x_ips: float
        velocity_y_ips: float
        heading_rad: float
        track_state: TrackState
        reid_embedding: Optional[np.ndarray] = None  # Shape: (D,) float32
    
    
    @dataclass(slots=True)
    class BallCluster:
        """Represents an ambient cluster of yellow game pieces approximated from blob area."""
        cluster_id: int
        centroid_x: float
        centroid_y: float
        bbox: BoundingBox
        area_px: float
        estimated_ball_count: int
        inventory_pct: float
    
    
    @dataclass(slots=True)
    class DualZoneDetectionResult:
        """Result of dual-zone ball detection containing focal individual balls and ambient clusters."""
        individual_balls: List[BoundingBox]
        clusters: List[BallCluster]
    
    
    @dataclass(slots=True)
    class BallTrackletState:
        """Kalman tracking state and physical field coordinates for an individual game piece ball or cluster."""
        ball_id: int
        pixel_bbox: BoundingBox
        field_x_inch: float
        field_y_inch: float
        velocity_x_ips: float
        velocity_y_ips: float
        radius_px: float
        state: BallState
    occluded_by_robot_id: Optional[int] = None
    coast_frames: int = 0
    is_cluster: bool = False
    cluster_ball_count: int = 1


@dataclass(slots=True)
class ActionSegment:
    """Action localized by ActionFormer with temporal boundaries."""
    event_id: str
    track_id: int
    team_number: Optional[int]
    action_type: ActionType
    element: FieldElement
    start_time_ms: int
    end_time_ms: int
    start_frame: int
    end_frame: int
    confidence: float
    diou: float
    pyramid_level: int
    field_coords: Tuple[float, float]
    is_reconciled: bool = False
    reconciled_posterior: float = 0.0


@dataclass(slots=True)
class FieldAPIScoreState:
    """Ground-truth aggregate scoring state from official FRC API."""
    timestamp_ms: int
    match_stage: MatchStage
    red_fuel_scored: int
    blue_fuel_scored: int
    red_tower_climbs: int
    blue_tower_climbs: int


@dataclass(slots=True)
class TelemetrySyncPacket:
    """Serialized binary sync payload ready for MQTT transport."""
    topic: str
    payload_bytes: bytes
    sequence_number: int
    qos: int
    payload_size_bytes: int
