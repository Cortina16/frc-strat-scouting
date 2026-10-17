"""
Module 09: Ultra-Low-Bandwidth Transport (MQTT & Protobuf).

Provides:
- Protobuf binary serialization helpers for telemetry envelopes, trajectory frames,
  robot telemetry points, ActionFormer events, and DuckDB match sync payloads.
- Packet size guardrail enforcing strict < 51,200 bytes (< 50 KB ceiling) to comply
  with venue wireless constraints.
- MQTTTelemetryPublisherActor implementing TelemetryPublisherActorBase and conforming
  to TelemetryPublisherActorProtocol with Paho-MQTT v2, TLS, keepalive, automated
  exponential backoff reconnection, and mock/loopback client support.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion, MQTTErrorCode
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode

from proto import telemetry_pb2
from src.common.interfaces import (
    TelemetryPublisherActorBase,
    TelemetryPublisherActorProtocol,
)
from src.common.types import (
    ActionSegment,
    ActionType,
    AllianceColor,
    BoundingBox,
    FieldElement,
    MatchStage,
    RobotTrackletState,
    TelemetrySyncPacket,
    TrackState,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Guardrails
# ============================================================================

MAX_PAYLOAD_SIZE_BYTES: int = 51200  # 50 KB ceiling (50 * 1024 bytes)
DEFAULT_KEEP_ALIVE_SECONDS: int = 60
DEFAULT_RECONNECT_MIN_DELAY_SECONDS: float = 1.0
DEFAULT_RECONNECT_MAX_DELAY_SECONDS: float = 60.0
DEFAULT_PUBLISH_TIMEOUT_SECONDS: float = 5.0
DEFAULT_CONNECT_TIMEOUT_SECONDS: float = 5.0


class PayloadSizeExceededError(ValueError):
    """Raised when a serialized telemetry payload exceeds the 51,200 bytes budget."""

    def __init__(self, size: int, limit: int = MAX_PAYLOAD_SIZE_BYTES, message: Optional[str] = None):
        self.size = size
        self.limit = limit
        super().__init__(
            message
            or f"Telemetry payload size ({size} bytes) exceeds maximum venue wireless budget of {limit} bytes (< 50 KB ceiling)."
        )


# ============================================================================
# Enum Mapping Helpers
# ============================================================================

def to_proto_alliance(alliance: Union[AllianceColor, str, int, None]) -> telemetry_pb2.AllianceColor:
    """Map domain AllianceColor or string/int to Protobuf AllianceColor enum."""
    if alliance is None:
        return telemetry_pb2.AllianceColor.ALLIANCE_UNSPECIFIED
    if isinstance(alliance, int):
        return alliance  # type: ignore[return-value]
    val = str(getattr(alliance, "value", alliance)).upper()
    if "RED" in val:
        return telemetry_pb2.AllianceColor.ALLIANCE_RED
    if "BLUE" in val:
        return telemetry_pb2.AllianceColor.ALLIANCE_BLUE
    return telemetry_pb2.AllianceColor.ALLIANCE_UNSPECIFIED


def to_proto_match_stage(stage: Union[MatchStage, str, int, None]) -> telemetry_pb2.MatchStage:
    """Map domain MatchStage or string/int to Protobuf MatchStage enum."""
    if stage is None:
        return telemetry_pb2.MatchStage.STAGE_UNSPECIFIED
    if isinstance(stage, int):
        return stage  # type: ignore[return-value]
    val = str(getattr(stage, "value", stage)).upper()
    if "PRE" in val:
        return telemetry_pb2.MatchStage.STAGE_PRE_MATCH
    if "AUTO" in val:
        return telemetry_pb2.MatchStage.STAGE_AUTONOMOUS
    if "TELE" in val:
        return telemetry_pb2.MatchStage.STAGE_TELEOP
    if "END" in val:
        return telemetry_pb2.MatchStage.STAGE_ENDGAME
    if "POST" in val:
        return telemetry_pb2.MatchStage.STAGE_POST_MATCH
    return telemetry_pb2.MatchStage.STAGE_UNSPECIFIED


def to_proto_track_state(state: Union[TrackState, str, int, None]) -> telemetry_pb2.TrackState:
    """Map domain TrackState or string/int to Protobuf TrackState enum."""
    if state is None:
        return telemetry_pb2.TrackState.TRACK_STATE_UNSPECIFIED
    if isinstance(state, int):
        return state  # type: ignore[return-value]
    val = str(getattr(state, "value", state)).upper()
    if "TRACKED" in val:
        return telemetry_pb2.TrackState.TRACK_STATE_TRACKED
    if "LOST" in val:
        return telemetry_pb2.TrackState.TRACK_STATE_LOST
    if "BRIDGED" in val:
        return telemetry_pb2.TrackState.TRACK_STATE_BRIDGED
    if "REMOVED" in val:
        return telemetry_pb2.TrackState.TRACK_STATE_REMOVED
    return telemetry_pb2.TrackState.TRACK_STATE_UNSPECIFIED


def to_proto_action_type(action: Union[ActionType, str, int, None]) -> telemetry_pb2.ActionType:
    """Map domain ActionType or string/int to Protobuf ActionType enum."""
    if action is None:
        return telemetry_pb2.ActionType.ACTION_UNSPECIFIED
    if isinstance(action, int):
        return action  # type: ignore[return-value]
    val = str(getattr(action, "value", action)).upper()
    mapping = {
        "INTAKE_FUEL": telemetry_pb2.ActionType.ACTION_INTAKE_FUEL,
        "SCORE_HUB": telemetry_pb2.ActionType.ACTION_SCORE_HUB,
        "CLIMB_TOWER": telemetry_pb2.ActionType.ACTION_CLIMB_TOWER,
        "DEFENSIVE_ENGAGEMENT": telemetry_pb2.ActionType.ACTION_DEFENSIVE_ENGAGEMENT,
        "TRANSIT_NEUTRAL_ZONE": telemetry_pb2.ActionType.ACTION_TRANSIT_NEUTRAL_ZONE,
        "OUTPOST_CORRAL_CYCLE": telemetry_pb2.ActionType.ACTION_OUTPOST_CORRAL_CYCLE,
        "IDLE": telemetry_pb2.ActionType.ACTION_IDLE,
    }
    return mapping.get(val, telemetry_pb2.ActionType.ACTION_UNSPECIFIED)


def to_proto_field_element(elem: Union[FieldElement, str, int, None]) -> telemetry_pb2.FieldElement:
    """Map domain FieldElement or string/int to Protobuf FieldElement enum."""
    if elem is None:
        return telemetry_pb2.FieldElement.ELEMENT_UNSPECIFIED
    if isinstance(elem, int):
        return elem  # type: ignore[return-value]
    val = str(getattr(elem, "value", elem)).upper()
    mapping = {
        "RED_HUB": telemetry_pb2.FieldElement.ELEMENT_RED_HUB,
        "BLUE_HUB": telemetry_pb2.FieldElement.ELEMENT_BLUE_HUB,
        "RED_TOWER": telemetry_pb2.FieldElement.ELEMENT_RED_TOWER,
        "BLUE_TOWER": telemetry_pb2.FieldElement.ELEMENT_BLUE_TOWER,
        "RED_OUTPOST": telemetry_pb2.FieldElement.ELEMENT_RED_OUTPOST,
        "BLUE_OUTPOST": telemetry_pb2.FieldElement.ELEMENT_BLUE_OUTPOST,
        "RED_CORRAL": telemetry_pb2.FieldElement.ELEMENT_RED_CORRAL,
        "BLUE_CORRAL": telemetry_pb2.FieldElement.ELEMENT_BLUE_CORRAL,
        "DEPOT": telemetry_pb2.FieldElement.ELEMENT_DEPOT,
        "BUMP": telemetry_pb2.FieldElement.ELEMENT_BUMP,
        "TRENCH": telemetry_pb2.FieldElement.ELEMENT_TRENCH,
        "NEUTRAL_ZONE": telemetry_pb2.FieldElement.ELEMENT_NEUTRAL_ZONE,
    }
    return mapping.get(val, telemetry_pb2.FieldElement.ELEMENT_UNSPECIFIED)


def to_proto_recon_status(status: Union[str, int, bool, None]) -> telemetry_pb2.ReconciliationStatus:
    """Map reconciliation flag or string to Protobuf ReconciliationStatus enum."""
    if status is True:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_CONFIRMED
    if status is False or status is None:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_PENDING
    if isinstance(status, int):
        return status  # type: ignore[return-value]
    val = str(status).upper()
    if "CONFIRMED" in val:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_CONFIRMED
    if "PRUNED" in val or "FALSE_POSITIVE" in val:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_PRUNED_FALSE_POSITIVE
    if "ADJUSTED" in val:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_ADJUSTED
    if "PENDING" in val:
        return telemetry_pb2.ReconciliationStatus.RECON_STATUS_PENDING
    return telemetry_pb2.ReconciliationStatus.RECON_STATUS_UNSPECIFIED


def to_proto_payload_type(ptype: Union[str, int, None]) -> int:
    """Map DuckDBSyncPayload payload type string/int to Protobuf enum value."""
    if isinstance(ptype, int):
        return ptype
    if not ptype:
        return telemetry_pb2.DuckDBSyncPayload.PayloadType.PAYLOAD_DELTA_SUMMARY
    val = str(ptype).upper()
    if "PARQUET" in val or "COLUMNAR" in val:
        return telemetry_pb2.DuckDBSyncPayload.PayloadType.PAYLOAD_COLUMNAR_PARQUET
    if "FULL" in val or "SNAPSHOT" in val:
        return telemetry_pb2.DuckDBSyncPayload.PayloadType.PAYLOAD_FULL_SNAPSHOT
    return telemetry_pb2.DuckDBSyncPayload.PayloadType.PAYLOAD_DELTA_SUMMARY


# ============================================================================
# Protobuf Entity Serializer Helpers
# ============================================================================

def to_protobuf_point_2d(val: Any) -> telemetry_pb2.Point2D:
    """Convert tuple, dict, or Point2D-like object into telemetry_pb2.Point2D."""
    if isinstance(val, telemetry_pb2.Point2D):
        return val
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return telemetry_pb2.Point2D(x=float(val[0]), y=float(val[1]))
    if isinstance(val, dict):
        return telemetry_pb2.Point2D(x=float(val.get("x", 0.0)), y=float(val.get("y", 0.0)))
    if hasattr(val, "x") and hasattr(val, "y"):
        return telemetry_pb2.Point2D(x=float(val.x), y=float(val.y))
    return telemetry_pb2.Point2D(x=0.0, y=0.0)


def to_protobuf_vector_2d(val: Any) -> telemetry_pb2.Vector2D:
    """Convert tuple, dict, or Vector2D-like object into telemetry_pb2.Vector2D."""
    if isinstance(val, telemetry_pb2.Vector2D):
        return val
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return telemetry_pb2.Vector2D(vx=float(val[0]), vy=float(val[1]))
    if isinstance(val, dict):
        return telemetry_pb2.Vector2D(
            vx=float(val.get("vx", val.get("velocity_x", 0.0))),
            vy=float(val.get("vy", val.get("velocity_y", 0.0))),
        )
    if hasattr(val, "vx") and hasattr(val, "vy"):
        return telemetry_pb2.Vector2D(vx=float(val.vx), vy=float(val.vy))
    return telemetry_pb2.Vector2D(vx=0.0, vy=0.0)


def to_protobuf_bbox_2d(val: Any) -> telemetry_pb2.BoundingBox2D:
    """Convert BoundingBox dataclass or dict to telemetry_pb2.BoundingBox2D."""
    if isinstance(val, telemetry_pb2.BoundingBox2D):
        return val
    if isinstance(val, dict):
        return telemetry_pb2.BoundingBox2D(
            center_x=float(val.get("center_x", 0.0)),
            center_y=float(val.get("center_y", 0.0)),
            width=float(val.get("width", 0.0)),
            height=float(val.get("height", 0.0)),
            confidence=float(val.get("confidence", 1.0)),
        )
    if hasattr(val, "center_x"):
        return telemetry_pb2.BoundingBox2D(
            center_x=float(val.center_x),
            center_y=float(val.center_y),
            width=float(val.width),
            height=float(val.height),
            confidence=float(getattr(val, "confidence", 1.0)),
        )
    return telemetry_pb2.BoundingBox2D(center_x=0.0, center_y=0.0, width=0.0, height=0.0, confidence=1.0)


def _safe_team_number(val: Any) -> int:
    """Ensures team number fits cleanly within uint32 protobuf constraints."""
    if val is None:
        return 0
    try:
        n = int(val)
        return n if 0 <= n <= 99999 else 0
    except (ValueError, TypeError):
        return 0


def to_protobuf_robot_telemetry_point(item: Any) -> telemetry_pb2.RobotTelemetryPoint:
    """Convert RobotTrackletState or dict into telemetry_pb2.RobotTelemetryPoint."""
    if isinstance(item, telemetry_pb2.RobotTelemetryPoint):
        return item

    pt = telemetry_pb2.RobotTelemetryPoint()
    if isinstance(item, RobotTrackletState):
        pt.track_id = int(item.track_id)
        pt.team_number = _safe_team_number(item.team_number)
        pt.alliance = to_proto_alliance(item.alliance)
        pt.field_position.CopyFrom(to_protobuf_point_2d((item.field_x_inch, item.field_y_inch)))
        pt.heading_rad = float(item.heading_rad)
        pt.velocity.CopyFrom(to_protobuf_vector_2d((item.velocity_x_ips, item.velocity_y_ips)))
        if item.pixel_bbox:
            pt.bbox.CopyFrom(to_protobuf_bbox_2d(item.pixel_bbox))
        pt.track_state = to_proto_track_state(item.track_state)
        if item.reid_embedding is not None:
            if hasattr(item.reid_embedding, "tobytes"):
                pt.reid_embedding = item.reid_embedding.tobytes()
            elif isinstance(item.reid_embedding, bytes):
                pt.reid_embedding = item.reid_embedding
        return pt

    if isinstance(item, dict):
        pt.track_id = int(item.get("track_id", 0))
        pt.team_number = _safe_team_number(item.get("team_number", 0))
        pt.alliance = to_proto_alliance(item.get("alliance"))
        pt.station = int(item.get("station", 0))
        field_pos = item.get("field_position", (item.get("field_x_inch", 0.0), item.get("field_y_inch", 0.0)))
        pt.field_position.CopyFrom(to_protobuf_point_2d(field_pos))
        pt.heading_rad = float(item.get("heading_rad", 0.0))
        vel = item.get("velocity", (item.get("velocity_x_ips", 0.0), item.get("velocity_y_ips", 0.0)))
        pt.velocity.CopyFrom(to_protobuf_vector_2d(vel))
        pt.angular_velocity_rad_s = float(item.get("angular_velocity_rad_s", 0.0))
        if "bbox" in item or "pixel_bbox" in item:
            pt.bbox.CopyFrom(to_protobuf_bbox_2d(item.get("bbox") or item.get("pixel_bbox")))
        pt.track_state = to_proto_track_state(item.get("track_state"))
        pt.occlusion = int(item.get("occlusion", 0))
        pt.ocr_confidence = float(item.get("ocr_confidence", 0.0))
        pt.ocr_super_resolved = bool(item.get("ocr_super_resolved", False))
        reid = item.get("reid_embedding")
        if reid is not None:
            pt.reid_embedding = reid if isinstance(reid, bytes) else getattr(reid, "tobytes", lambda: bytes(reid))()
        return pt

    # Generic object fallback
    pt.track_id = int(getattr(item, "track_id", 0))
    pt.team_number = _safe_team_number(getattr(item, "team_number", 0))
    pt.alliance = to_proto_alliance(getattr(item, "alliance", None))
    pt.heading_rad = float(getattr(item, "heading_rad", 0.0))
    if hasattr(item, "field_x_inch"):
        pt.field_position.CopyFrom(to_protobuf_point_2d((item.field_x_inch, item.field_y_inch)))
    if hasattr(item, "velocity_x_ips"):
        pt.velocity.CopyFrom(to_protobuf_vector_2d((item.velocity_x_ips, item.velocity_y_ips)))
    if hasattr(item, "pixel_bbox") and item.pixel_bbox:
        pt.bbox.CopyFrom(to_protobuf_bbox_2d(item.pixel_bbox))
    if hasattr(item, "track_state"):
        pt.track_state = to_proto_track_state(item.track_state)
    return pt


def to_protobuf_trajectory_frame(data: Any) -> telemetry_pb2.TrajectoryFrame:
    """Convert raw frame dictionary, list of tracklets, or TrajectoryFrame to telemetry_pb2.TrajectoryFrame."""
    if isinstance(data, telemetry_pb2.TrajectoryFrame):
        return data

    frame = telemetry_pb2.TrajectoryFrame()

    # If passed a list/iterable of tracklet states
    if isinstance(data, (list, tuple)):
        for item in data:
            frame.robots.append(to_protobuf_robot_telemetry_point(item))
        return frame

    # If passed a single tracklet state
    if isinstance(data, RobotTrackletState):
        frame.robots.append(to_protobuf_robot_telemetry_point(data))
        return frame

    # If passed a dictionary
    if isinstance(data, dict):
        frame.frame_index = int(data.get("frame_index", 0))
        frame.match_time_ms = int(data.get("match_time_ms", 0))
        frame.match_remaining_seconds = float(data.get("match_remaining_seconds", 0.0))
        frame.match_stage = to_proto_match_stage(data.get("match_stage"))
        for r in data.get("robots", []):
            frame.robots.append(to_protobuf_robot_telemetry_point(r))
        return frame

    # Generic object fallback
    frame.frame_index = int(getattr(data, "frame_index", 0))
    frame.match_time_ms = int(getattr(data, "match_time_ms", 0))
    frame.match_remaining_seconds = float(getattr(data, "match_remaining_seconds", 0.0))
    frame.match_stage = to_proto_match_stage(getattr(data, "match_stage", None))
    if hasattr(data, "robots"):
        for r in getattr(data, "robots", []):
            frame.robots.append(to_protobuf_robot_telemetry_point(r))
    return frame


def to_protobuf_action_former_event(data: Any) -> telemetry_pb2.ActionFormerEvent:
    """Convert ActionSegment or dict into telemetry_pb2.ActionFormerEvent."""
    if isinstance(data, telemetry_pb2.ActionFormerEvent):
        return data

    ev = telemetry_pb2.ActionFormerEvent()
    if isinstance(data, ActionSegment):
        ev.event_id = str(data.event_id)
        ev.track_id = int(data.track_id)
        ev.team_number = _safe_team_number(data.team_number)
        ev.alliance = to_proto_alliance(getattr(data, "alliance", AllianceColor.UNKNOWN))
        ev.action_type = to_proto_action_type(data.action_type)
        ev.primary_element = to_proto_field_element(data.element)
        ev.start_time_ms = int(data.start_time_ms)
        ev.end_time_ms = int(data.end_time_ms)
        ev.start_frame = int(data.start_frame)
        ev.end_frame = int(data.end_frame)
        ev.duration_seconds = max(0.0, float(data.end_time_ms - data.start_time_ms) / 1000.0)
        ev.raw_confidence = float(data.confidence)
        ev.boundary_diou = float(data.diou)
        ev.feature_pyramid_level = int(data.pyramid_level)
        if data.field_coords:
            ev.field_position_at_onset.CopyFrom(to_protobuf_point_2d(data.field_coords))
            ev.field_position_at_offset.CopyFrom(to_protobuf_point_2d(data.field_coords))
        ev.reconciliation = to_proto_recon_status(data.is_reconciled)
        ev.reconciled_posterior = float(data.reconciled_posterior)
        return ev

    if isinstance(data, dict):
        ev.event_id = str(data.get("event_id", str(uuid.uuid4())))
        ev.track_id = int(data.get("track_id", 0))
        ev.team_number = _safe_team_number(data.get("team_number", 0))
        ev.alliance = to_proto_alliance(data.get("alliance"))
        ev.action_type = to_proto_action_type(data.get("action_type"))
        ev.primary_element = to_proto_field_element(data.get("primary_element") or data.get("element"))
        ev.start_time_ms = int(data.get("start_time_ms", 0))
        ev.end_time_ms = int(data.get("end_time_ms", 0))
        ev.start_frame = int(data.get("start_frame", 0))
        ev.end_frame = int(data.get("end_frame", 0))
        ev.duration_seconds = float(
            data.get("duration_seconds", max(0.0, float(ev.end_time_ms - ev.start_time_ms) / 1000.0))
        )
        ev.raw_confidence = float(data.get("raw_confidence", data.get("confidence", 0.0)))
        ev.boundary_diou = float(data.get("boundary_diou", data.get("diou", 0.0)))
        ev.feature_pyramid_level = int(data.get("feature_pyramid_level", data.get("pyramid_level", 0)))
        onset = data.get("field_position_at_onset", data.get("field_coords"))
        offset = data.get("field_position_at_offset", data.get("field_coords"))
        if onset:
            ev.field_position_at_onset.CopyFrom(to_protobuf_point_2d(onset))
        if offset:
            ev.field_position_at_offset.CopyFrom(to_protobuf_point_2d(offset))
        ev.reconciliation = to_proto_recon_status(data.get("reconciliation", data.get("is_reconciled")))
        ev.reconciled_posterior = float(data.get("reconciled_posterior", 0.0))
        ev.reconciled_fuel_delta = int(data.get("reconciled_fuel_delta", 0))
        ev.reconciliation_notes = str(data.get("reconciliation_notes", ""))
        return ev

    # Object fallback
    ev.event_id = str(getattr(data, "event_id", str(uuid.uuid4())))
    ev.track_id = int(getattr(data, "track_id", 0))
    ev.team_number = _safe_team_number(getattr(data, "team_number", 0))
    ev.alliance = to_proto_alliance(getattr(data, "alliance", None))
    ev.action_type = to_proto_action_type(getattr(data, "action_type", None))
    ev.primary_element = to_proto_field_element(getattr(data, "element", getattr(data, "primary_element", None)))
    ev.start_time_ms = int(getattr(data, "start_time_ms", 0))
    ev.end_time_ms = int(getattr(data, "end_time_ms", 0))
    ev.start_frame = int(getattr(data, "start_frame", 0))
    ev.end_frame = int(getattr(data, "end_frame", 0))
    ev.raw_confidence = float(getattr(data, "confidence", 0.0))
    return ev


def to_protobuf_robot_match_aggregate(data: Any) -> telemetry_pb2.RobotMatchAggregate:
    """Convert dictionary or object into telemetry_pb2.RobotMatchAggregate."""
    if isinstance(data, telemetry_pb2.RobotMatchAggregate):
        return data

    agg = telemetry_pb2.RobotMatchAggregate()
    if isinstance(data, dict):
        agg.team_number = _safe_team_number(data.get("team_number", 0))
        agg.alliance = to_proto_alliance(data.get("alliance"))
        agg.total_fuel_scored = int(data.get("total_fuel_scored", 0))
        agg.total_cycles = int(data.get("total_cycles", 0))
        agg.avg_cycle_time_seconds = float(data.get("avg_cycle_time_seconds", 0.0))
        agg.min_cycle_time_seconds = float(data.get("min_cycle_time_seconds", 0.0))
        agg.avg_neutral_zone_transit_time_s = float(data.get("avg_neutral_zone_transit_time_s", 0.0))
        agg.total_distance_traveled_inches = float(data.get("total_distance_traveled_inches", 0.0))
        agg.max_velocity_inches_s = float(data.get("max_velocity_inches_s", 0.0))
        agg.avg_velocity_inches_s = float(data.get("avg_velocity_inches_s", 0.0))
        agg.time_under_defensive_pressure_s = float(data.get("time_under_defensive_pressure_s", 0.0))
        agg.tower_climb_success = bool(data.get("tower_climb_success", False))
        agg.tower_climb_duration_s = float(data.get("tower_climb_duration_s", 0.0))
        return agg

    agg.team_number = _safe_team_number(getattr(data, "team_number", 0))
    agg.alliance = to_proto_alliance(getattr(data, "alliance", None))
    agg.total_fuel_scored = int(getattr(data, "total_fuel_scored", 0))
    agg.total_cycles = int(getattr(data, "total_cycles", 0))
    agg.avg_cycle_time_seconds = float(getattr(data, "avg_cycle_time_seconds", 0.0))
    agg.min_cycle_time_seconds = float(getattr(data, "min_cycle_time_seconds", 0.0))
    agg.avg_neutral_zone_transit_time_s = float(getattr(data, "avg_neutral_zone_transit_time_s", 0.0))
    agg.total_distance_traveled_inches = float(getattr(data, "total_distance_traveled_inches", 0.0))
    agg.max_velocity_inches_s = float(getattr(data, "max_velocity_inches_s", 0.0))
    agg.avg_velocity_inches_s = float(getattr(data, "avg_velocity_inches_s", 0.0))
    agg.time_under_defensive_pressure_s = float(getattr(data, "time_under_defensive_pressure_s", 0.0))
    agg.tower_climb_success = bool(getattr(data, "tower_climb_success", False))
    agg.tower_climb_duration_s = float(getattr(data, "tower_climb_duration_s", 0.0))
    return agg


def to_protobuf_zone_transit_statistic(data: Any) -> telemetry_pb2.ZoneTransitStatistic:
    """Convert dictionary or object into telemetry_pb2.ZoneTransitStatistic."""
    if isinstance(data, telemetry_pb2.ZoneTransitStatistic):
        return data

    z = telemetry_pb2.ZoneTransitStatistic()
    if isinstance(data, dict):
        z.team_number = _safe_team_number(data.get("team_number", 0))
        z.origin_zone = str(data.get("origin_zone", ""))
        z.destination_zone = str(data.get("destination_zone", ""))
        z.transit_corridor = str(data.get("transit_corridor", ""))
        z.avg_transit_seconds = float(data.get("avg_transit_seconds", 0.0))
        z.sample_count = int(data.get("sample_count", 0))
        z.defensive_impedance_ratio = float(data.get("defensive_impedance_ratio", 1.0))
        return z

    z.team_number = _safe_team_number(getattr(data, "team_number", 0))
    z.origin_zone = str(getattr(data, "origin_zone", ""))
    z.destination_zone = str(getattr(data, "destination_zone", ""))
    z.transit_corridor = str(getattr(data, "transit_corridor", ""))
    z.avg_transit_seconds = float(getattr(data, "avg_transit_seconds", 0.0))
    z.sample_count = int(getattr(data, "sample_count", 0))
    z.defensive_impedance_ratio = float(getattr(data, "defensive_impedance_ratio", 1.0))
    return z


def to_protobuf_duckdb_sync_payload(data: Any) -> telemetry_pb2.DuckDBSyncPayload:
    """Convert dictionary or object into telemetry_pb2.DuckDBSyncPayload."""
    if isinstance(data, telemetry_pb2.DuckDBSyncPayload):
        return data

    sync = telemetry_pb2.DuckDBSyncPayload()
    if isinstance(data, dict):
        sync.event_key = str(data.get("event_key", "2026casf"))
        sync.match_id = str(data.get("match_id", "qm1"))
        sync.sequence_number = int(data.get("sequence_number", 1))
        sync.generated_at_epoch_ms = int(data.get("generated_at_epoch_ms", int(time.time() * 1000)))
        sync.payload_type = to_proto_payload_type(data.get("payload_type"))

        for agg in data.get("robot_aggregates", []):
            sync.robot_aggregates.append(to_protobuf_robot_match_aggregate(agg))

        for z in data.get("zone_transits", []):
            sync.zone_transits.append(to_protobuf_zone_transit_statistic(z))

        for ev in data.get("verified_scoring_events", []):
            sync.verified_scoring_events.append(to_protobuf_action_former_event(ev))

        parquet_chunk = data.get("compressed_parquet_chunk")
        if parquet_chunk:
            sync.compressed_parquet_chunk = bytes(parquet_chunk)
        sync.parquet_table_name = str(data.get("parquet_table_name", "match_trajectory"))

        sync.frc_api_red_score = int(data.get("frc_api_red_score", 0))
        sync.frc_api_blue_score = int(data.get("frc_api_blue_score", 0))
        sync.vision_reconciled_red_score = int(data.get("vision_reconciled_red_score", 0))
        sync.vision_reconciled_blue_score = int(data.get("vision_reconciled_blue_score", 0))
        sync.fully_reconciled = bool(data.get("fully_reconciled", False))
        return sync

    # Object fallback
    sync.event_key = str(getattr(data, "event_key", "2026casf"))
    sync.match_id = str(getattr(data, "match_id", "qm1"))
    sync.sequence_number = int(getattr(data, "sequence_number", 1))
    sync.generated_at_epoch_ms = int(getattr(data, "generated_at_epoch_ms", int(time.time() * 1000)))
    sync.payload_type = to_proto_payload_type(getattr(data, "payload_type", None))

    for agg in getattr(data, "robot_aggregates", []):
        sync.robot_aggregates.append(to_protobuf_robot_match_aggregate(agg))
    for z in getattr(data, "zone_transits", []):
        sync.zone_transits.append(to_protobuf_zone_transit_statistic(z))
    for ev in getattr(data, "verified_scoring_events", []):
        sync.verified_scoring_events.append(to_protobuf_action_former_event(ev))

    chunk = getattr(data, "compressed_parquet_chunk", None)
    if chunk:
        sync.compressed_parquet_chunk = bytes(chunk)
    sync.parquet_table_name = str(getattr(data, "parquet_table_name", "match_trajectory"))
    sync.frc_api_red_score = int(getattr(data, "frc_api_red_score", 0))
    sync.frc_api_blue_score = int(getattr(data, "frc_api_blue_score", 0))
    sync.vision_reconciled_red_score = int(getattr(data, "vision_reconciled_red_score", 0))
    sync.vision_reconciled_blue_score = int(getattr(data, "vision_reconciled_blue_score", 0))
    sync.fully_reconciled = bool(getattr(data, "fully_reconciled", False))
    return sync


def to_protobuf_envelope(
    payload: Any,
    message_id: Optional[str] = None,
    timestamp_ms: Optional[int] = None,
    schema_version: int = 1,
) -> telemetry_pb2.TelemetryEnvelope:
    """Wraps a supported telemetry sub-message into a top-level TelemetryEnvelope."""
    if isinstance(payload, telemetry_pb2.TelemetryEnvelope):
        return payload

    envelope = telemetry_pb2.TelemetryEnvelope()
    envelope.message_id = message_id or str(uuid.uuid4())
    envelope.epoch_timestamp_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    envelope.schema_version = schema_version

    if isinstance(payload, telemetry_pb2.DuckDBSyncPayload):
        envelope.duckdb_sync.CopyFrom(payload)
    elif isinstance(payload, telemetry_pb2.TrajectoryFrame):
        envelope.trajectory_frame.CopyFrom(payload)
    elif isinstance(payload, telemetry_pb2.MatchTrajectoryBatch):
        envelope.trajectory_batch.CopyFrom(payload)
    elif isinstance(payload, telemetry_pb2.ActionFormerEvent):
        envelope.action_event.CopyFrom(payload)
    elif isinstance(payload, telemetry_pb2.ActionMetadataBatch):
        envelope.action_batch.CopyFrom(payload)
    elif isinstance(payload, telemetry_pb2.HomographyTelemetry):
        envelope.homography.CopyFrom(payload)
    elif isinstance(payload, RobotTrackletState) or (
        isinstance(payload, (list, tuple)) and payload and isinstance(payload[0], RobotTrackletState)
    ):
        envelope.trajectory_frame.CopyFrom(to_protobuf_trajectory_frame(payload))
    elif isinstance(payload, ActionSegment):
        envelope.action_event.CopyFrom(to_protobuf_action_former_event(payload))
    elif isinstance(payload, dict):
        if "robot_aggregates" in payload or "zone_transits" in payload or "frc_api_red_score" in payload:
            envelope.duckdb_sync.CopyFrom(to_protobuf_duckdb_sync_payload(payload))
        elif "robots" in payload or "frame_index" in payload:
            envelope.trajectory_frame.CopyFrom(to_protobuf_trajectory_frame(payload))
        elif "action_type" in payload:
            envelope.action_event.CopyFrom(to_protobuf_action_former_event(payload))
        else:
            raise ValueError(f"Unable to infer envelope payload type from dictionary keys: {list(payload.keys())}")
    else:
        raise TypeError(f"Unsupported payload type for TelemetryEnvelope: {type(payload)}")

    return envelope


def serialize_payload(payload: Any, wrap_envelope: bool = True) -> bytes:
    """Serializes telemetry payload to binary Protobuf bytes, optionally wrapping in TelemetryEnvelope."""
    if isinstance(payload, bytes):
        return payload

    if wrap_envelope:
        env = to_protobuf_envelope(payload)
        return env.SerializeToString()

    if hasattr(payload, "SerializeToString"):
        return payload.SerializeToString()

    # Fallback to envelope
    return to_protobuf_envelope(payload).SerializeToString()


def deserialize_envelope(payload_bytes: bytes) -> telemetry_pb2.TelemetryEnvelope:
    """Parses binary Protobuf bytes into a TelemetryEnvelope."""
    env = telemetry_pb2.TelemetryEnvelope()
    env.ParseFromString(payload_bytes)
    return env


def unpack_payload(payload_bytes: bytes) -> Any:
    """Unpacks binary Protobuf bytes, extracting inner message if wrapped in TelemetryEnvelope."""
    try:
        env = deserialize_envelope(payload_bytes)
        field_name = env.WhichOneof("payload")
        if field_name:
            return getattr(env, field_name)
        return env
    except Exception:
        # Try direct DuckDBSyncPayload
        try:
            sync = telemetry_pb2.DuckDBSyncPayload()
            sync.ParseFromString(payload_bytes)
            return sync
        except Exception:
            traj = telemetry_pb2.TrajectoryFrame()
            traj.ParseFromString(payload_bytes)
            return traj


# ============================================================================
# Packet Size Guardrail & Pruning
# ============================================================================

def enforce_payload_size_guardrail(
    payload_bytes: bytes,
    max_bytes: int = MAX_PAYLOAD_SIZE_BYTES,
) -> bytes:
    """Validates that payload byte size does not exceed max_bytes. Raises PayloadSizeExceededError if violated."""
    size = len(payload_bytes)
    if size >= max_bytes:
        logger.error(
            "Packet size guardrail violated: %d bytes exceeds %d bytes limit (< 50 KB ceiling)",
            size,
            max_bytes,
        )
        raise PayloadSizeExceededError(size=size, limit=max_bytes)
    return payload_bytes


def prune_duckdb_sync_payload(
    payload: telemetry_pb2.DuckDBSyncPayload,
    max_bytes: int = MAX_PAYLOAD_SIZE_BYTES,
    wrap_envelope: bool = True,
) -> Tuple[telemetry_pb2.DuckDBSyncPayload, bytes]:
    """Prunes optional fields (compressed Parquet chunk, then non-critical actions) to fit under max_bytes."""
    wire = serialize_payload(payload, wrap_envelope=wrap_envelope)
    if len(wire) < max_bytes:
        return payload, wire

    logger.warning(
        "DuckDB sync payload size %d bytes exceeds budget %d bytes. Commencing adaptive pruning.",
        len(wire),
        max_bytes,
    )

    # Step 1: Strip optional compressed Parquet chunk if present
    if payload.compressed_parquet_chunk:
        logger.info("Pruning compressed_parquet_chunk (%d bytes)", len(payload.compressed_parquet_chunk))
        payload.compressed_parquet_chunk = b""
        payload.payload_type = telemetry_pb2.DuckDBSyncPayload.PayloadType.PAYLOAD_DELTA_SUMMARY
        wire = serialize_payload(payload, wrap_envelope=wrap_envelope)
        if len(wire) < max_bytes:
            return payload, wire

    # Step 2: Trim non-reconciled or lower-confidence action events if still oversized
    if len(payload.verified_scoring_events) > 5 and len(wire) >= max_bytes:
        events = list(payload.verified_scoring_events)
        # Keep reconciled events first, then highest confidence
        events.sort(
            key=lambda e: (
                e.reconciliation == telemetry_pb2.ReconciliationStatus.RECON_STATUS_CONFIRMED,
                e.reconciled_posterior,
                e.raw_confidence,
            ),
            reverse=True,
        )
        low = 0
        high = len(events)
        best_count = 0
        best_wire = wire

        while low <= high:
            mid = (low + high) // 2
            del payload.verified_scoring_events[:]
            payload.verified_scoring_events.extend(events[:mid])
            candidate_wire = serialize_payload(payload, wrap_envelope=wrap_envelope)
            if len(candidate_wire) < max_bytes:
                best_count = mid
                best_wire = candidate_wire
                low = mid + 1
            else:
                high = mid - 1

        del payload.verified_scoring_events[:]
        payload.verified_scoring_events.extend(events[:best_count])
        wire = best_wire
        if len(wire) < max_bytes:
            return payload, wire

    if len(wire) >= max_bytes:
        raise PayloadSizeExceededError(
            size=len(wire),
            limit=max_bytes,
            message=f"DuckDB sync payload remains oversized ({len(wire)} bytes) after pruning optional fields.",
        )

    return payload, wire


# ============================================================================
# Mock & Loopback MQTT Client
# ============================================================================

class LoopbackMQTTClient(mqtt.Client):
    """
    Subclass of Paho MQTT Client implementing in-memory loopback transport.

    Enables 100% reliable execution of real Paho-MQTT v2 callback contracts
    in isolated CI/test environments without external daemon dependencies.
    """

    def __init__(self, client_id: str = "loopback-client", *args: Any, **kwargs: Any):
        super().__init__(CallbackAPIVersion.VERSION2, client_id=client_id, *args, **kwargs)
        self.published_messages: List[Dict[str, Any]] = []
        self._connected: bool = False
        self._mid_seq: int = 0
        self.connected_event: asyncio.Event = asyncio.Event()

    def connect(self, host: str, port: int = 1883, keepalive: int = 60, bind_address: str = "") -> int:
        self._connected = True
        self.connected_event.set()
        if self.on_connect:
            self.on_connect(self, self._userdata, {}, ReasonCode(PacketTypes.CONNACK, "Success"), None)
        return MQTTErrorCode.MQTT_ERR_SUCCESS

    def connect_async(self, host: str, port: int = 1883, keepalive: int = 60, bind_address: str = "") -> int:
        return self.connect(host, port, keepalive, bind_address)

    def reconnect(self) -> int:
        self._connected = True
        self.connected_event.set()
        if self.on_connect:
            self.on_connect(self, self._userdata, {}, ReasonCode(PacketTypes.CONNACK, "Success"), None)
        return MQTTErrorCode.MQTT_ERR_SUCCESS

    def is_connected(self) -> bool:
        return self._connected

    def loop_start(self) -> int:
        return MQTTErrorCode.MQTT_ERR_SUCCESS

    def loop_stop(self, force: bool = False) -> int:
        return MQTTErrorCode.MQTT_ERR_SUCCESS

    def publish(
        self,
        topic: str,
        payload: Any = None,
        qos: int = 0,
        retain: bool = False,
        properties: Any = None,
    ) -> mqtt.MQTTMessageInfo:
        if not self._connected:
            info = mqtt.MQTTMessageInfo(0)
            info.rc = MQTTErrorCode.MQTT_ERR_NO_CONN
            return info

        self._mid_seq += 1
        mid = self._mid_seq

        raw_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        msg_record = {
            "topic": topic,
            "payload": raw_bytes,
            "qos": qos,
            "retain": retain,
            "mid": mid,
            "timestamp": time.time(),
        }
        self.published_messages.append(msg_record)

        info = mqtt.MQTTMessageInfo(mid)
        info.rc = MQTTErrorCode.MQTT_ERR_SUCCESS
        info._set_as_published()

        if self.on_publish:
            self.on_publish(self, self._userdata, mid, ReasonCode(PacketTypes.PUBACK, "Success"), properties)

        return info

    def disconnect(self, reasoncode: Any = None, properties: Any = None) -> int:
        self._connected = False
        self.connected_event.clear()
        if self.on_disconnect:
            rc = reasoncode or ReasonCode(PacketTypes.DISCONNECT, "Normal disconnection")
            self.on_disconnect(self, self._userdata, {}, rc, properties)
        return MQTTErrorCode.MQTT_ERR_SUCCESS

    def simulate_network_disconnect(self, reason: Union[str, int, ReasonCode] = "Unspecified error") -> None:
        """Simulate unexpected network partition or broker drop."""
        self._connected = False
        self.connected_event.clear()
        if self.on_disconnect:
            if isinstance(reason, ReasonCode):
                rc = reason
            else:
                try:
                    rc = ReasonCode(PacketTypes.DISCONNECT, reason)
                except Exception:
                    rc = ReasonCode(PacketTypes.DISCONNECT, "Unspecified error")
            self.on_disconnect(self, self._userdata, {}, rc, None)

    def simulate_network_reconnect(self) -> None:
        """Simulate successful network recovery and reconnection."""
        self.reconnect()

    def get_published_messages(self) -> List[Dict[str, Any]]:
        return list(self.published_messages)

    def clear_published_messages(self) -> None:
        self.published_messages.clear()


# ============================================================================
# MQTT Telemetry Publisher Actor
# ============================================================================

class MQTTTelemetryPublisherActor(TelemetryPublisherActorBase):
    """
    MQTT Telemetry Publisher Actor for FRC scouting pipeline.

    Subclasses TelemetryPublisherActorBase and conforms to TelemetryPublisherActorProtocol.
    Manages Paho-MQTT v2 TLS/TCP client connection, keep-alive, automated exponential
    backoff reconnection, QoS 0 trajectory publishing, QoS 1 DuckDB sync publishing,
    and strict < 50 KB payload size budget enforcement.
    """

    def __init__(
        self,
        tls_enabled: bool = False,
        tls_ca_certs: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        keepalive: int = DEFAULT_KEEP_ALIVE_SECONDS,
        reconnect_min_delay: float = DEFAULT_RECONNECT_MIN_DELAY_SECONDS,
        reconnect_max_delay: float = DEFAULT_RECONNECT_MAX_DELAY_SECONDS,
        publish_timeout: float = DEFAULT_PUBLISH_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        default_wrap_envelope: bool = True,
        mock_mode: bool = False,
        auto_prune_oversized: bool = False,
    ):
        self.tls_enabled = tls_enabled
        self.tls_ca_certs = tls_ca_certs
        self.tls_certfile = tls_certfile
        self.tls_keyfile = tls_keyfile
        self.keepalive = keepalive
        self.reconnect_min_delay = reconnect_min_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.publish_timeout = publish_timeout
        self.connect_timeout = connect_timeout
        self.default_wrap_envelope = default_wrap_envelope
        self.mock_mode = mock_mode
        self.auto_prune_oversized = auto_prune_oversized

        self._client: Optional[mqtt.Client] = None
        self._is_connected: bool = False
        self._connected_event: asyncio.Event = asyncio.Event()
        self._sequence_counter: int = 0
        self._published_packets: List[TelemetrySyncPacket] = []
        self._total_bytes_published: int = 0
        self._reconnect_count: int = 0

        # Async futures for QoS 1 delivery acknowledgments
        self._pending_publishes: Dict[int, asyncio.Future[bool]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_connected(self) -> bool:
        """Returns True if the publisher currently holds an active broker connection."""
        if self._client is not None and hasattr(self._client, "is_connected"):
            return bool(self._client.is_connected() and self._is_connected)
        return self._is_connected

    @property
    def client(self) -> Optional[mqtt.Client]:
        """Returns underlying Paho-MQTT Client instance."""
        return self._client

    @property
    def published_packets(self) -> List[TelemetrySyncPacket]:
        """Returns delivery ledger of all emitted TelemetrySyncPackets."""
        return list(self._published_packets)

    @property
    def total_bytes_published(self) -> int:
        """Returns cumulative wire bytes published across all topics."""
        return self._total_bytes_published

    @property
    def reconnect_count(self) -> int:
        """Returns total number of successful reconnections observed."""
        return self._reconnect_count

    # ------------------------------------------------------------------------
    # Protocol Methods
    # ------------------------------------------------------------------------

    async def connect(
        self,
        broker_url: str,
        port: int = 1883,
        client_id: str = "scout-pub",
    ) -> bool:
        """
        Establishes persistent connection to venue MQTT broker.

        Uses Paho-MQTT v2 API with automated exponential backoff reconnection,
        keepalive, and non-blocking asynchronous connection wait.
        """
        self._loop = asyncio.get_running_loop()
        self._connected_event.clear()

        is_mock_target = (
            self.mock_mode
            or broker_url.lower() in ("loopback", "mock", "test", "mock://localhost", "localhost:mock")
        )

        if is_mock_target:
            logger.info("Initializing LoopbackMQTTClient for target %s:%d", broker_url, port)
            loopback = LoopbackMQTTClient(client_id=client_id)
            self._client = loopback
        else:
            logger.info("Initializing Paho-MQTT v2 client (ID: %s) for %s:%d", client_id, broker_url, port)
            self._client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=client_id)

        # Wire Paho v2 Callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

        # Configure automated exponential backoff
        self._client.reconnect_delay_set(
            min_delay=int(max(1.0, self.reconnect_min_delay)),
            max_delay=int(max(self.reconnect_min_delay, self.reconnect_max_delay)),
        )

        # Configure TLS if requested or standard TLS port (8883)
        if self.tls_enabled or port == 8883:
            logger.info("Configuring TLS for MQTT publisher connection")
            self._client.tls_set(
                ca_certs=self.tls_ca_certs,
                certfile=self.tls_certfile,
                keyfile=self.tls_keyfile,
            )

        # Initiate non-blocking connection
        try:
            rc = self._client.connect_async(broker_url, port=port, keepalive=self.keepalive)
            if rc != MQTTErrorCode.MQTT_ERR_SUCCESS and rc != 0:
                logger.warning("client.connect_async returned code: %s", rc)
            self._client.loop_start()
        except Exception as err:
            logger.error("Failed to initiate MQTT connection to %s:%d: %s", broker_url, port, err)
            return False

        # Wait for on_connect callback
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=self.connect_timeout)
            logger.info("Successfully established connection to MQTT broker %s:%d", broker_url, port)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "MQTT connection to %s:%d timed out after %.1f seconds",
                broker_url,
                port,
                self.connect_timeout,
            )
            # If in mock mode or loopback, force connected
            if is_mock_target:
                self._is_connected = True
                self._connected_event.set()
                return True
            return False

    async def publish_live_trajectory(
        self,
        topic: str,
        trajectory_frame: Any,
        qos: int = 0,
        wrap_envelope: Optional[bool] = None,
    ) -> bool:
        """
        Serializes and publishes live trajectory frame via Protobuf over MQTT at QoS 0.

        Accepts TrajectoryFrame, RobotTrackletState, list of tracklets, TelemetryEnvelope, or dict.
        """
        if not self.is_connected:
            logger.warning("Cannot publish trajectory: publisher is not connected")
            return False

        wrap = self.default_wrap_envelope if wrap_envelope is None else wrap_envelope
        if isinstance(trajectory_frame, bytes):
            payload_bytes = trajectory_frame
        else:
            traj_pb = to_protobuf_trajectory_frame(trajectory_frame)
            payload_bytes = serialize_payload(traj_pb, wrap_envelope=wrap)

        # Fire and forget QoS 0 publish
        self._sequence_counter += 1
        seq = self._sequence_counter

        try:
            info = self._client.publish(topic, payload=payload_bytes, qos=qos)
            if info.rc != MQTTErrorCode.MQTT_ERR_SUCCESS and info.rc != 0:
                logger.error("client.publish returned error code %s on topic %s", info.rc, topic)
                return False

            sync_packet = TelemetrySyncPacket(
                topic=topic,
                payload_bytes=payload_bytes,
                sequence_number=seq,
                qos=qos,
                payload_size_bytes=len(payload_bytes),
            )
            self._published_packets.append(sync_packet)
            self._total_bytes_published += len(payload_bytes)
            return True
        except Exception as err:
            logger.error("Exception publishing live trajectory: %s", err)
            return False

    async def publish_live_trajectories_batch(
        self,
        topic: str,
        trajectory_frames: List[Any],
        qos: int = 0,
        wrap_envelope: Optional[bool] = None,
    ) -> int:
        """Publishes a batch of live trajectory frames, returning count of successfully published packets."""
        if not self.is_connected:
            return 0
        success_count = 0
        for frame in trajectory_frames:
            ok = await self.publish_live_trajectory(topic, frame, qos=qos, wrap_envelope=wrap_envelope)
            if ok:
                success_count += 1
        return success_count


    async def publish_duckdb_sync(
        self,
        topic: str,
        sync_payload: Any,
        qos: int = 1,
        wrap_envelope: Optional[bool] = None,
        prune_if_oversized: Optional[bool] = None,
    ) -> bool:
        """
        Publishes authoritative match aggregate delta under strict < 50 KB (51,200 bytes) payload budget.

        Serializes DuckDBSyncPayload with Protobuf binary encoding and publishes at QoS 1.
        Validates packet size guardrail and raises PayloadSizeExceededError if budget exceeded.
        """
        if not self.is_connected:
            logger.warning("Cannot publish DuckDB sync: publisher is not connected")
            return False

        wrap = self.default_wrap_envelope if wrap_envelope is None else wrap_envelope
        should_prune = self.auto_prune_oversized if prune_if_oversized is None else prune_if_oversized

        if isinstance(sync_payload, bytes):
            payload_bytes = sync_payload
            enforce_payload_size_guardrail(payload_bytes, max_bytes=MAX_PAYLOAD_SIZE_BYTES)
        else:
            sync_pb = to_protobuf_duckdb_sync_payload(sync_payload)
            if should_prune:
                sync_pb, payload_bytes = prune_duckdb_sync_payload(
                    sync_pb,
                    max_bytes=MAX_PAYLOAD_SIZE_BYTES,
                    wrap_envelope=wrap,
                )
            else:
                payload_bytes = serialize_payload(sync_pb, wrap_envelope=wrap)
                enforce_payload_size_guardrail(payload_bytes, max_bytes=MAX_PAYLOAD_SIZE_BYTES)

        self._sequence_counter += 1
        seq = self._sequence_counter

        # Register pending publish future for QoS 1 ACK
        fut: Optional[asyncio.Future[bool]] = None
        current_loop = asyncio.get_running_loop()

        try:
            info = self._client.publish(topic, payload=payload_bytes, qos=qos)
            if info.rc != MQTTErrorCode.MQTT_ERR_SUCCESS and info.rc != 0:
                logger.error("client.publish returned error code %s on topic %s", info.rc, topic)
                return False

            if qos >= 1:
                mid = info.mid
                fut = current_loop.create_future()
                self._pending_publishes[mid] = fut

                # If already marked as published (e.g. LoopbackMQTTClient)
                if getattr(info, "is_published", lambda: False)():
                    if not fut.done():
                        fut.set_result(True)

                try:
                    await asyncio.wait_for(fut, timeout=self.publish_timeout)
                except asyncio.TimeoutError:
                    logger.warning("Publish ACK timeout for mid=%d on topic=%s", mid, topic)
                finally:
                    self._pending_publishes.pop(mid, None)

            sync_packet = TelemetrySyncPacket(
                topic=topic,
                payload_bytes=payload_bytes,
                sequence_number=seq,
                qos=qos,
                payload_size_bytes=len(payload_bytes),
            )
            self._published_packets.append(sync_packet)
            self._total_bytes_published += len(payload_bytes)
            return True
        except PayloadSizeExceededError:
            raise
        except Exception as err:
            logger.error("Exception publishing DuckDB sync payload: %s", err)
            return False

    async def disconnect(self) -> None:
        """Cleanly disconnects publisher and terminates network event loop."""
        self._is_connected = False
        self._connected_event.clear()

        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception as err:
                logger.warning("Exception during publisher disconnect: %s", err)
            finally:
                self._client = None
        logger.info("MQTT publisher disconnected cleanly")

    async def reconnect(self) -> bool:
        """Manually trigger client reconnection and await confirmation."""
        if self._client is None:
            return False
        self._connected_event.clear()
        try:
            self._client.reconnect()
            await asyncio.wait_for(self._connected_event.wait(), timeout=self.connect_timeout)
            return True
        except Exception as err:
            logger.error("Manual reconnect failed: %s", err)
            return False

    def simulate_network_drop(self, recovery_delay_s: float = 0.05) -> None:
        """Simulate unexpected network disconnect followed by automatic recovery for unit testing."""
        if isinstance(self._client, LoopbackMQTTClient):
            self._client.simulate_network_disconnect("Unspecified error")
            if recovery_delay_s > 0:
                if self._loop and self._loop.is_running():
                    self._loop.call_later(recovery_delay_s, self._client.simulate_network_reconnect)
                else:
                    self._client.simulate_network_reconnect()
        else:
            # Trigger disconnect callback on real client
            self._on_disconnect(self._client, None, {}, ReasonCode(PacketTypes.DISCONNECT, "Unspecified error"), None)
            if recovery_delay_s > 0:
                if self._loop and self._loop.is_running():
                    self._loop.call_later(
                        recovery_delay_s,
                        self._on_connect,
                        self._client,
                        None,
                        {},
                        ReasonCode(PacketTypes.CONNACK, "Success"),
                        None,
                    )

    # ------------------------------------------------------------------------
    # Paho-MQTT v2 Callbacks
    # ------------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, Any],
        reason_code: Any,
        properties: Any,
    ) -> None:
        is_failure = getattr(reason_code, "is_failure", False)
        rc_val = getattr(reason_code, "value", reason_code)
        is_success = (not is_failure) and (rc_val == 0 or str(reason_code).lower() == "success")
        if is_success:
            logger.info("MQTT on_connect callback received with success reason_code=%s", reason_code)
            self._is_connected = True
            self._reconnect_count += 1
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._connected_event.set)
            else:
                self._connected_event.set()
        else:
            logger.error("MQTT on_connect failed with reason_code: %s", reason_code)
            self._is_connected = False

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: Any,
        reason_code: Any,
        properties: Any,
    ) -> None:
        logger.warning("MQTT on_disconnect callback triggered with reason_code: %s", reason_code)
        self._is_connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._connected_event.clear)
        else:
            self._connected_event.clear()

    def _on_publish(
        self,
        client: mqtt.Client,
        userdata: Any,
        mid: int,
        reason_code: Any,
        properties: Any,
    ) -> None:
        if mid in self._pending_publishes:
            fut = self._pending_publishes[mid]
            if not fut.done():
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(fut.set_result, True)
                else:
                    fut.set_result(True)


# Export alias conforming to standard naming
TelemetryPublisherActor = MQTTTelemetryPublisherActor
