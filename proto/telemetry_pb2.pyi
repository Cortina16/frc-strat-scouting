from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AllianceColor(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ALLIANCE_UNSPECIFIED: _ClassVar[AllianceColor]
    ALLIANCE_RED: _ClassVar[AllianceColor]
    ALLIANCE_BLUE: _ClassVar[AllianceColor]

class AllianceStation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STATION_UNSPECIFIED: _ClassVar[AllianceStation]
    STATION_1: _ClassVar[AllianceStation]
    STATION_2: _ClassVar[AllianceStation]
    STATION_3: _ClassVar[AllianceStation]

class MatchStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STAGE_UNSPECIFIED: _ClassVar[MatchStage]
    STAGE_PRE_MATCH: _ClassVar[MatchStage]
    STAGE_AUTONOMOUS: _ClassVar[MatchStage]
    STAGE_TELEOP: _ClassVar[MatchStage]
    STAGE_ENDGAME: _ClassVar[MatchStage]
    STAGE_POST_MATCH: _ClassVar[MatchStage]

class TrackState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACK_STATE_UNSPECIFIED: _ClassVar[TrackState]
    TRACK_STATE_TRACKED: _ClassVar[TrackState]
    TRACK_STATE_LOST: _ClassVar[TrackState]
    TRACK_STATE_BRIDGED: _ClassVar[TrackState]
    TRACK_STATE_REMOVED: _ClassVar[TrackState]

class OcclusionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OCCLUSION_NONE: _ClassVar[OcclusionStatus]
    OCCLUSION_PARTIAL: _ClassVar[OcclusionStatus]
    OCCLUSION_FULL: _ClassVar[OcclusionStatus]

class ActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ACTION_UNSPECIFIED: _ClassVar[ActionType]
    ACTION_INTAKE_FUEL: _ClassVar[ActionType]
    ACTION_SCORE_HUB: _ClassVar[ActionType]
    ACTION_CLIMB_TOWER: _ClassVar[ActionType]
    ACTION_DEFENSIVE_ENGAGEMENT: _ClassVar[ActionType]
    ACTION_TRANSIT_NEUTRAL_ZONE: _ClassVar[ActionType]
    ACTION_OUTPOST_CORRAL_CYCLE: _ClassVar[ActionType]
    ACTION_IDLE: _ClassVar[ActionType]

class FieldElement(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ELEMENT_UNSPECIFIED: _ClassVar[FieldElement]
    ELEMENT_RED_HUB: _ClassVar[FieldElement]
    ELEMENT_BLUE_HUB: _ClassVar[FieldElement]
    ELEMENT_RED_TOWER: _ClassVar[FieldElement]
    ELEMENT_BLUE_TOWER: _ClassVar[FieldElement]
    ELEMENT_RED_OUTPOST: _ClassVar[FieldElement]
    ELEMENT_BLUE_OUTPOST: _ClassVar[FieldElement]
    ELEMENT_RED_CORRAL: _ClassVar[FieldElement]
    ELEMENT_BLUE_CORRAL: _ClassVar[FieldElement]
    ELEMENT_DEPOT: _ClassVar[FieldElement]
    ELEMENT_BUMP: _ClassVar[FieldElement]
    ELEMENT_TRENCH: _ClassVar[FieldElement]
    ELEMENT_NEUTRAL_ZONE: _ClassVar[FieldElement]

class ReconciliationStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RECON_STATUS_UNSPECIFIED: _ClassVar[ReconciliationStatus]
    RECON_STATUS_PENDING: _ClassVar[ReconciliationStatus]
    RECON_STATUS_CONFIRMED: _ClassVar[ReconciliationStatus]
    RECON_STATUS_PRUNED_FALSE_POSITIVE: _ClassVar[ReconciliationStatus]
    RECON_STATUS_ADJUSTED: _ClassVar[ReconciliationStatus]
ALLIANCE_UNSPECIFIED: AllianceColor
ALLIANCE_RED: AllianceColor
ALLIANCE_BLUE: AllianceColor
STATION_UNSPECIFIED: AllianceStation
STATION_1: AllianceStation
STATION_2: AllianceStation
STATION_3: AllianceStation
STAGE_UNSPECIFIED: MatchStage
STAGE_PRE_MATCH: MatchStage
STAGE_AUTONOMOUS: MatchStage
STAGE_TELEOP: MatchStage
STAGE_ENDGAME: MatchStage
STAGE_POST_MATCH: MatchStage
TRACK_STATE_UNSPECIFIED: TrackState
TRACK_STATE_TRACKED: TrackState
TRACK_STATE_LOST: TrackState
TRACK_STATE_BRIDGED: TrackState
TRACK_STATE_REMOVED: TrackState
OCCLUSION_NONE: OcclusionStatus
OCCLUSION_PARTIAL: OcclusionStatus
OCCLUSION_FULL: OcclusionStatus
ACTION_UNSPECIFIED: ActionType
ACTION_INTAKE_FUEL: ActionType
ACTION_SCORE_HUB: ActionType
ACTION_CLIMB_TOWER: ActionType
ACTION_DEFENSIVE_ENGAGEMENT: ActionType
ACTION_TRANSIT_NEUTRAL_ZONE: ActionType
ACTION_OUTPOST_CORRAL_CYCLE: ActionType
ACTION_IDLE: ActionType
ELEMENT_UNSPECIFIED: FieldElement
ELEMENT_RED_HUB: FieldElement
ELEMENT_BLUE_HUB: FieldElement
ELEMENT_RED_TOWER: FieldElement
ELEMENT_BLUE_TOWER: FieldElement
ELEMENT_RED_OUTPOST: FieldElement
ELEMENT_BLUE_OUTPOST: FieldElement
ELEMENT_RED_CORRAL: FieldElement
ELEMENT_BLUE_CORRAL: FieldElement
ELEMENT_DEPOT: FieldElement
ELEMENT_BUMP: FieldElement
ELEMENT_TRENCH: FieldElement
ELEMENT_NEUTRAL_ZONE: FieldElement
RECON_STATUS_UNSPECIFIED: ReconciliationStatus
RECON_STATUS_PENDING: ReconciliationStatus
RECON_STATUS_CONFIRMED: ReconciliationStatus
RECON_STATUS_PRUNED_FALSE_POSITIVE: ReconciliationStatus
RECON_STATUS_ADJUSTED: ReconciliationStatus

class Point2D(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class Vector2D(_message.Message):
    __slots__ = ("vx", "vy")
    VX_FIELD_NUMBER: _ClassVar[int]
    VY_FIELD_NUMBER: _ClassVar[int]
    vx: float
    vy: float
    def __init__(self, vx: _Optional[float] = ..., vy: _Optional[float] = ...) -> None: ...

class BoundingBox2D(_message.Message):
    __slots__ = ("center_x", "center_y", "width", "height", "confidence")
    CENTER_X_FIELD_NUMBER: _ClassVar[int]
    CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    def __init__(self, center_x: _Optional[float] = ..., center_y: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., confidence: _Optional[float] = ...) -> None: ...

class HomographyMatrix(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, elements: _Optional[_Iterable[float]] = ...) -> None: ...

class AffineTransform2D(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, elements: _Optional[_Iterable[float]] = ...) -> None: ...

class AprilTagDetection(_message.Message):
    __slots__ = ("tag_id", "image_corners", "field_position", "field_elevation_z", "reprojection_error", "hamming_distance")
    TAG_ID_FIELD_NUMBER: _ClassVar[int]
    IMAGE_CORNERS_FIELD_NUMBER: _ClassVar[int]
    FIELD_POSITION_FIELD_NUMBER: _ClassVar[int]
    FIELD_ELEVATION_Z_FIELD_NUMBER: _ClassVar[int]
    REPROJECTION_ERROR_FIELD_NUMBER: _ClassVar[int]
    HAMMING_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    tag_id: int
    image_corners: _containers.RepeatedCompositeFieldContainer[Point2D]
    field_position: Point2D
    field_elevation_z: float
    reprojection_error: float
    hamming_distance: int
    def __init__(self, tag_id: _Optional[int] = ..., image_corners: _Optional[_Iterable[_Union[Point2D, _Mapping]]] = ..., field_position: _Optional[_Union[Point2D, _Mapping]] = ..., field_elevation_z: _Optional[float] = ..., reprojection_error: _Optional[float] = ..., hamming_distance: _Optional[int] = ...) -> None: ...

class HomographyTelemetry(_message.Message):
    __slots__ = ("frame_index", "timestamp_ms", "homography", "cmc_affine", "detected_tags", "num_tags_detected", "dlt_condition_number", "lm_reprojection_rmse", "lm_iterations", "optimization_converged", "camera_cut_detected", "camera_zoom_factor")
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    CMC_AFFINE_FIELD_NUMBER: _ClassVar[int]
    DETECTED_TAGS_FIELD_NUMBER: _ClassVar[int]
    NUM_TAGS_DETECTED_FIELD_NUMBER: _ClassVar[int]
    DLT_CONDITION_NUMBER_FIELD_NUMBER: _ClassVar[int]
    LM_REPROJECTION_RMSE_FIELD_NUMBER: _ClassVar[int]
    LM_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_CONVERGED_FIELD_NUMBER: _ClassVar[int]
    CAMERA_CUT_DETECTED_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ZOOM_FACTOR_FIELD_NUMBER: _ClassVar[int]
    frame_index: int
    timestamp_ms: int
    homography: HomographyMatrix
    cmc_affine: AffineTransform2D
    detected_tags: _containers.RepeatedCompositeFieldContainer[AprilTagDetection]
    num_tags_detected: int
    dlt_condition_number: float
    lm_reprojection_rmse: float
    lm_iterations: int
    optimization_converged: bool
    camera_cut_detected: bool
    camera_zoom_factor: float
    def __init__(self, frame_index: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., homography: _Optional[_Union[HomographyMatrix, _Mapping]] = ..., cmc_affine: _Optional[_Union[AffineTransform2D, _Mapping]] = ..., detected_tags: _Optional[_Iterable[_Union[AprilTagDetection, _Mapping]]] = ..., num_tags_detected: _Optional[int] = ..., dlt_condition_number: _Optional[float] = ..., lm_reprojection_rmse: _Optional[float] = ..., lm_iterations: _Optional[int] = ..., optimization_converged: _Optional[bool] = ..., camera_cut_detected: _Optional[bool] = ..., camera_zoom_factor: _Optional[float] = ...) -> None: ...

class RobotTelemetryPoint(_message.Message):
    __slots__ = ("track_id", "team_number", "alliance", "station", "field_position", "heading_rad", "velocity", "angular_velocity_rad_s", "bbox", "track_state", "occlusion", "ocr_confidence", "ocr_super_resolved", "reid_embedding")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    TEAM_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ALLIANCE_FIELD_NUMBER: _ClassVar[int]
    STATION_FIELD_NUMBER: _ClassVar[int]
    FIELD_POSITION_FIELD_NUMBER: _ClassVar[int]
    HEADING_RAD_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    ANGULAR_VELOCITY_RAD_S_FIELD_NUMBER: _ClassVar[int]
    BBOX_FIELD_NUMBER: _ClassVar[int]
    TRACK_STATE_FIELD_NUMBER: _ClassVar[int]
    OCCLUSION_FIELD_NUMBER: _ClassVar[int]
    OCR_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    OCR_SUPER_RESOLVED_FIELD_NUMBER: _ClassVar[int]
    REID_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    track_id: int
    team_number: int
    alliance: AllianceColor
    station: AllianceStation
    field_position: Point2D
    heading_rad: float
    velocity: Vector2D
    angular_velocity_rad_s: float
    bbox: BoundingBox2D
    track_state: TrackState
    occlusion: OcclusionStatus
    ocr_confidence: float
    ocr_super_resolved: bool
    reid_embedding: bytes
    def __init__(self, track_id: _Optional[int] = ..., team_number: _Optional[int] = ..., alliance: _Optional[_Union[AllianceColor, str]] = ..., station: _Optional[_Union[AllianceStation, str]] = ..., field_position: _Optional[_Union[Point2D, _Mapping]] = ..., heading_rad: _Optional[float] = ..., velocity: _Optional[_Union[Vector2D, _Mapping]] = ..., angular_velocity_rad_s: _Optional[float] = ..., bbox: _Optional[_Union[BoundingBox2D, _Mapping]] = ..., track_state: _Optional[_Union[TrackState, str]] = ..., occlusion: _Optional[_Union[OcclusionStatus, str]] = ..., ocr_confidence: _Optional[float] = ..., ocr_super_resolved: _Optional[bool] = ..., reid_embedding: _Optional[bytes] = ...) -> None: ...

class TrajectoryFrame(_message.Message):
    __slots__ = ("frame_index", "match_time_ms", "match_remaining_seconds", "match_stage", "robots")
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    MATCH_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MATCH_REMAINING_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MATCH_STAGE_FIELD_NUMBER: _ClassVar[int]
    ROBOTS_FIELD_NUMBER: _ClassVar[int]
    frame_index: int
    match_time_ms: int
    match_remaining_seconds: float
    match_stage: MatchStage
    robots: _containers.RepeatedCompositeFieldContainer[RobotTelemetryPoint]
    def __init__(self, frame_index: _Optional[int] = ..., match_time_ms: _Optional[int] = ..., match_remaining_seconds: _Optional[float] = ..., match_stage: _Optional[_Union[MatchStage, str]] = ..., robots: _Optional[_Iterable[_Union[RobotTelemetryPoint, _Mapping]]] = ...) -> None: ...

class MatchTrajectoryBatch(_message.Message):
    __slots__ = ("event_key", "match_id", "frames")
    EVENT_KEY_FIELD_NUMBER: _ClassVar[int]
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    event_key: str
    match_id: str
    frames: _containers.RepeatedCompositeFieldContainer[TrajectoryFrame]
    def __init__(self, event_key: _Optional[str] = ..., match_id: _Optional[str] = ..., frames: _Optional[_Iterable[_Union[TrajectoryFrame, _Mapping]]] = ...) -> None: ...

class ActionFormerEvent(_message.Message):
    __slots__ = ("event_id", "track_id", "team_number", "alliance", "action_type", "primary_element", "start_time_ms", "end_time_ms", "start_frame", "end_frame", "duration_seconds", "raw_confidence", "boundary_diou", "feature_pyramid_level", "field_position_at_onset", "field_position_at_offset", "reconciliation", "reconciled_posterior", "reconciled_fuel_delta", "reconciliation_notes")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    TEAM_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ALLIANCE_FIELD_NUMBER: _ClassVar[int]
    ACTION_TYPE_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_ELEMENT_FIELD_NUMBER: _ClassVar[int]
    START_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    START_FRAME_FIELD_NUMBER: _ClassVar[int]
    END_FRAME_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    RAW_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_DIOU_FIELD_NUMBER: _ClassVar[int]
    FEATURE_PYRAMID_LEVEL_FIELD_NUMBER: _ClassVar[int]
    FIELD_POSITION_AT_ONSET_FIELD_NUMBER: _ClassVar[int]
    FIELD_POSITION_AT_OFFSET_FIELD_NUMBER: _ClassVar[int]
    RECONCILIATION_FIELD_NUMBER: _ClassVar[int]
    RECONCILED_POSTERIOR_FIELD_NUMBER: _ClassVar[int]
    RECONCILED_FUEL_DELTA_FIELD_NUMBER: _ClassVar[int]
    RECONCILIATION_NOTES_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    track_id: int
    team_number: int
    alliance: AllianceColor
    action_type: ActionType
    primary_element: FieldElement
    start_time_ms: int
    end_time_ms: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    raw_confidence: float
    boundary_diou: float
    feature_pyramid_level: int
    field_position_at_onset: Point2D
    field_position_at_offset: Point2D
    reconciliation: ReconciliationStatus
    reconciled_posterior: float
    reconciled_fuel_delta: int
    reconciliation_notes: str
    def __init__(self, event_id: _Optional[str] = ..., track_id: _Optional[int] = ..., team_number: _Optional[int] = ..., alliance: _Optional[_Union[AllianceColor, str]] = ..., action_type: _Optional[_Union[ActionType, str]] = ..., primary_element: _Optional[_Union[FieldElement, str]] = ..., start_time_ms: _Optional[int] = ..., end_time_ms: _Optional[int] = ..., start_frame: _Optional[int] = ..., end_frame: _Optional[int] = ..., duration_seconds: _Optional[float] = ..., raw_confidence: _Optional[float] = ..., boundary_diou: _Optional[float] = ..., feature_pyramid_level: _Optional[int] = ..., field_position_at_onset: _Optional[_Union[Point2D, _Mapping]] = ..., field_position_at_offset: _Optional[_Union[Point2D, _Mapping]] = ..., reconciliation: _Optional[_Union[ReconciliationStatus, str]] = ..., reconciled_posterior: _Optional[float] = ..., reconciled_fuel_delta: _Optional[int] = ..., reconciliation_notes: _Optional[str] = ...) -> None: ...

class ActionMetadataBatch(_message.Message):
    __slots__ = ("event_key", "match_id", "actions")
    EVENT_KEY_FIELD_NUMBER: _ClassVar[int]
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    event_key: str
    match_id: str
    actions: _containers.RepeatedCompositeFieldContainer[ActionFormerEvent]
    def __init__(self, event_key: _Optional[str] = ..., match_id: _Optional[str] = ..., actions: _Optional[_Iterable[_Union[ActionFormerEvent, _Mapping]]] = ...) -> None: ...

class RobotMatchAggregate(_message.Message):
    __slots__ = ("team_number", "alliance", "total_fuel_scored", "total_cycles", "avg_cycle_time_seconds", "min_cycle_time_seconds", "avg_neutral_zone_transit_time_s", "total_distance_traveled_inches", "max_velocity_inches_s", "avg_velocity_inches_s", "time_under_defensive_pressure_s", "tower_climb_success", "tower_climb_duration_s")
    TEAM_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ALLIANCE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FUEL_SCORED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_CYCLES_FIELD_NUMBER: _ClassVar[int]
    AVG_CYCLE_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MIN_CYCLE_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    AVG_NEUTRAL_ZONE_TRANSIT_TIME_S_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DISTANCE_TRAVELED_INCHES_FIELD_NUMBER: _ClassVar[int]
    MAX_VELOCITY_INCHES_S_FIELD_NUMBER: _ClassVar[int]
    AVG_VELOCITY_INCHES_S_FIELD_NUMBER: _ClassVar[int]
    TIME_UNDER_DEFENSIVE_PRESSURE_S_FIELD_NUMBER: _ClassVar[int]
    TOWER_CLIMB_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    TOWER_CLIMB_DURATION_S_FIELD_NUMBER: _ClassVar[int]
    team_number: int
    alliance: AllianceColor
    total_fuel_scored: int
    total_cycles: int
    avg_cycle_time_seconds: float
    min_cycle_time_seconds: float
    avg_neutral_zone_transit_time_s: float
    total_distance_traveled_inches: float
    max_velocity_inches_s: float
    avg_velocity_inches_s: float
    time_under_defensive_pressure_s: float
    tower_climb_success: bool
    tower_climb_duration_s: float
    def __init__(self, team_number: _Optional[int] = ..., alliance: _Optional[_Union[AllianceColor, str]] = ..., total_fuel_scored: _Optional[int] = ..., total_cycles: _Optional[int] = ..., avg_cycle_time_seconds: _Optional[float] = ..., min_cycle_time_seconds: _Optional[float] = ..., avg_neutral_zone_transit_time_s: _Optional[float] = ..., total_distance_traveled_inches: _Optional[float] = ..., max_velocity_inches_s: _Optional[float] = ..., avg_velocity_inches_s: _Optional[float] = ..., time_under_defensive_pressure_s: _Optional[float] = ..., tower_climb_success: _Optional[bool] = ..., tower_climb_duration_s: _Optional[float] = ...) -> None: ...

class ZoneTransitStatistic(_message.Message):
    __slots__ = ("team_number", "origin_zone", "destination_zone", "transit_corridor", "avg_transit_seconds", "sample_count", "defensive_impedance_ratio")
    TEAM_NUMBER_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_ZONE_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_ZONE_FIELD_NUMBER: _ClassVar[int]
    TRANSIT_CORRIDOR_FIELD_NUMBER: _ClassVar[int]
    AVG_TRANSIT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEFENSIVE_IMPEDANCE_RATIO_FIELD_NUMBER: _ClassVar[int]
    team_number: int
    origin_zone: str
    destination_zone: str
    transit_corridor: str
    avg_transit_seconds: float
    sample_count: int
    defensive_impedance_ratio: float
    def __init__(self, team_number: _Optional[int] = ..., origin_zone: _Optional[str] = ..., destination_zone: _Optional[str] = ..., transit_corridor: _Optional[str] = ..., avg_transit_seconds: _Optional[float] = ..., sample_count: _Optional[int] = ..., defensive_impedance_ratio: _Optional[float] = ...) -> None: ...

class DuckDBSyncPayload(_message.Message):
    __slots__ = ("event_key", "match_id", "sequence_number", "generated_at_epoch_ms", "payload_type", "robot_aggregates", "zone_transits", "verified_scoring_events", "compressed_parquet_chunk", "parquet_table_name", "frc_api_red_score", "frc_api_blue_score", "vision_reconciled_red_score", "vision_reconciled_blue_score", "fully_reconciled")
    class PayloadType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        PAYLOAD_TYPE_UNSPECIFIED: _ClassVar[DuckDBSyncPayload.PayloadType]
        PAYLOAD_DELTA_SUMMARY: _ClassVar[DuckDBSyncPayload.PayloadType]
        PAYLOAD_COLUMNAR_PARQUET: _ClassVar[DuckDBSyncPayload.PayloadType]
        PAYLOAD_FULL_SNAPSHOT: _ClassVar[DuckDBSyncPayload.PayloadType]
    PAYLOAD_TYPE_UNSPECIFIED: DuckDBSyncPayload.PayloadType
    PAYLOAD_DELTA_SUMMARY: DuckDBSyncPayload.PayloadType
    PAYLOAD_COLUMNAR_PARQUET: DuckDBSyncPayload.PayloadType
    PAYLOAD_FULL_SNAPSHOT: DuckDBSyncPayload.PayloadType
    EVENT_KEY_FIELD_NUMBER: _ClassVar[int]
    MATCH_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    GENERATED_AT_EPOCH_MS_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_TYPE_FIELD_NUMBER: _ClassVar[int]
    ROBOT_AGGREGATES_FIELD_NUMBER: _ClassVar[int]
    ZONE_TRANSITS_FIELD_NUMBER: _ClassVar[int]
    VERIFIED_SCORING_EVENTS_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_PARQUET_CHUNK_FIELD_NUMBER: _ClassVar[int]
    PARQUET_TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    FRC_API_RED_SCORE_FIELD_NUMBER: _ClassVar[int]
    FRC_API_BLUE_SCORE_FIELD_NUMBER: _ClassVar[int]
    VISION_RECONCILED_RED_SCORE_FIELD_NUMBER: _ClassVar[int]
    VISION_RECONCILED_BLUE_SCORE_FIELD_NUMBER: _ClassVar[int]
    FULLY_RECONCILED_FIELD_NUMBER: _ClassVar[int]
    event_key: str
    match_id: str
    sequence_number: int
    generated_at_epoch_ms: int
    payload_type: DuckDBSyncPayload.PayloadType
    robot_aggregates: _containers.RepeatedCompositeFieldContainer[RobotMatchAggregate]
    zone_transits: _containers.RepeatedCompositeFieldContainer[ZoneTransitStatistic]
    verified_scoring_events: _containers.RepeatedCompositeFieldContainer[ActionFormerEvent]
    compressed_parquet_chunk: bytes
    parquet_table_name: str
    frc_api_red_score: int
    frc_api_blue_score: int
    vision_reconciled_red_score: int
    vision_reconciled_blue_score: int
    fully_reconciled: bool
    def __init__(self, event_key: _Optional[str] = ..., match_id: _Optional[str] = ..., sequence_number: _Optional[int] = ..., generated_at_epoch_ms: _Optional[int] = ..., payload_type: _Optional[_Union[DuckDBSyncPayload.PayloadType, str]] = ..., robot_aggregates: _Optional[_Iterable[_Union[RobotMatchAggregate, _Mapping]]] = ..., zone_transits: _Optional[_Iterable[_Union[ZoneTransitStatistic, _Mapping]]] = ..., verified_scoring_events: _Optional[_Iterable[_Union[ActionFormerEvent, _Mapping]]] = ..., compressed_parquet_chunk: _Optional[bytes] = ..., parquet_table_name: _Optional[str] = ..., frc_api_red_score: _Optional[int] = ..., frc_api_blue_score: _Optional[int] = ..., vision_reconciled_red_score: _Optional[int] = ..., vision_reconciled_blue_score: _Optional[int] = ..., fully_reconciled: _Optional[bool] = ...) -> None: ...

class TelemetryEnvelope(_message.Message):
    __slots__ = ("message_id", "epoch_timestamp_ms", "schema_version", "homography", "trajectory_frame", "trajectory_batch", "action_event", "action_batch", "duckdb_sync")
    MESSAGE_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_FRAME_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_BATCH_FIELD_NUMBER: _ClassVar[int]
    ACTION_EVENT_FIELD_NUMBER: _ClassVar[int]
    ACTION_BATCH_FIELD_NUMBER: _ClassVar[int]
    DUCKDB_SYNC_FIELD_NUMBER: _ClassVar[int]
    message_id: str
    epoch_timestamp_ms: int
    schema_version: int
    homography: HomographyTelemetry
    trajectory_frame: TrajectoryFrame
    trajectory_batch: MatchTrajectoryBatch
    action_event: ActionFormerEvent
    action_batch: ActionMetadataBatch
    duckdb_sync: DuckDBSyncPayload
    def __init__(self, message_id: _Optional[str] = ..., epoch_timestamp_ms: _Optional[int] = ..., schema_version: _Optional[int] = ..., homography: _Optional[_Union[HomographyTelemetry, _Mapping]] = ..., trajectory_frame: _Optional[_Union[TrajectoryFrame, _Mapping]] = ..., trajectory_batch: _Optional[_Union[MatchTrajectoryBatch, _Mapping]] = ..., action_event: _Optional[_Union[ActionFormerEvent, _Mapping]] = ..., action_batch: _Optional[_Union[ActionMetadataBatch, _Mapping]] = ..., duckdb_sync: _Optional[_Union[DuckDBSyncPayload, _Mapping]] = ...) -> None: ...
