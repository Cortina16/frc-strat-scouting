"""Unit tests for Module 09: Ultra-Low-Bandwidth Transport (MQTT & Protobuf).

Verifies:
1. Interface Contract Validation:
   - MQTTTelemetryPublisherActor inherits TelemetryPublisherActorBase.
   - Satisfies TelemetryPublisherActorProtocol with async coroutine methods.
2. Payload Budget Verification (< 50 KB / 51,200 Bytes):
   - Full match sync payload with 6 robots, aggregates, zone transits, Parquet delta chunk,
     and 60+ action events strictly < 51,200 bytes.
   - Packet size guardrail raises PayloadSizeExceededError (ValueError) on oversized payload.
   - Adaptive pruning drops optional Parquet chunk to recover compliance.
3. Publishing & QoS Validation:
   - QoS 0 live trajectory publishing with Protobuf serialization.
   - QoS 1 authoritative DuckDB sync publishing under < 50 KB budget.
   - Full Protobuf serialization and deserialization round-trip.
4. Network Reconnection & Resynchronization:
   - Simulates carrier drop and automatic exponential backoff reconnection.
   - Verifies publisher recovers and successfully resumes data transmission.
5. Real TCP Socket Loopback Validation:
   - Tests real Paho-MQTT v2 client over local TCP ephemeral port.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Dict, List, Tuple

import pytest

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
from src.transport import (
    DEFAULT_KEEP_ALIVE_SECONDS,
    MAX_PAYLOAD_SIZE_BYTES,
    LoopbackMQTTClient,
    MQTTTelemetryPublisherActor,
    PayloadSizeExceededError,
    TelemetryPublisherActor,
    deserialize_envelope,
    enforce_payload_size_guardrail,
    prune_duckdb_sync_payload,
    serialize_payload,
    to_proto_action_type,
    to_proto_alliance,
    to_proto_field_element,
    to_proto_match_stage,
    to_proto_payload_type,
    to_proto_recon_status,
    to_proto_track_state,
    to_protobuf_action_former_event,
    to_protobuf_bbox_2d,
    to_protobuf_duckdb_sync_payload,
    to_protobuf_envelope,
    to_protobuf_point_2d,
    to_protobuf_robot_match_aggregate,
    to_protobuf_robot_telemetry_point,
    to_protobuf_trajectory_frame,
    to_protobuf_vector_2d,
    to_protobuf_zone_transit_statistic,
    unpack_payload,
)


# ============================================================================
# Helpers & Fixtures
# ============================================================================

SIX_TEAMS = [254, 1678, 1323, 971, 1114, 2056]


def _make_tracklet(
    track_id: int,
    team_number: int,
    alliance: AllianceColor,
    field_x: float = 120.0,
    field_y: float = 140.0,
) -> RobotTrackletState:
    """Creates a synthetic RobotTrackletState instance."""
    bbox = BoundingBox(
        center_x=320.0,
        center_y=240.0,
        width=80.0,
        height=60.0,
        confidence=0.95,
        class_id=0,
    )
    return RobotTrackletState(
        track_id=track_id,
        team_number=team_number,
        alliance=alliance,
        pixel_bbox=bbox,
        field_x_inch=field_x,
        field_y_inch=field_y,
        velocity_x_ips=14.5,
        velocity_y_ips=-2.3,
        heading_rad=1.57,
        track_state=TrackState.TRACKED,
        reid_embedding=b"\x01\x02\x03\x04\x05\x06\x07\x08",
    )


def _make_action(
    event_id: str,
    track_id: int,
    team_number: int,
    action_type: ActionType,
    element: FieldElement,
    start_ms: int,
    end_ms: int,
    confidence: float = 0.92,
) -> ActionSegment:
    """Creates a synthetic ActionSegment instance."""
    return ActionSegment(
        event_id=event_id,
        track_id=track_id,
        team_number=team_number,
        action_type=action_type,
        element=element,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        start_frame=int(start_ms * 30 / 1000),
        end_frame=int(end_ms * 30 / 1000),
        confidence=confidence,
        diou=0.88,
        pyramid_level=1,
        field_coords=(180.0, 150.0),
        is_reconciled=True,
        reconciled_posterior=0.97,
    )


def _build_full_match_payload_dict(action_count: int = 60, parquet_bytes_size: int = 4096) -> Dict[str, Any]:
    """Constructs full-match sync payload dictionary for all 6 robots."""
    aggregates = []
    for idx, team in enumerate(SIX_TEAMS):
        alliance = "RED" if idx < 3 else "BLUE"
        aggregates.append({
            "team_number": team,
            "alliance": alliance,
            "total_fuel_scored": 42 + idx * 5,
            "total_cycles": 14 + idx,
            "avg_cycle_time_seconds": 9.4 - idx * 0.2,
            "min_cycle_time_seconds": 6.8,
            "avg_neutral_zone_transit_time_s": 2.1,
            "total_distance_traveled_inches": 3400.0 + idx * 100,
            "max_velocity_inches_s": 180.0,
            "avg_velocity_inches_s": 85.0,
            "time_under_defensive_pressure_s": 12.5,
            "tower_climb_success": True,
            "tower_climb_duration_s": 3.8,
        })

    zone_transits = []
    corridors = ["TRENCH_WEST", "BUMP_NORTH", "CORRAL_SOUTH", "DEPOT_LANE"]
    for team in SIX_TEAMS:
        for corr in corridors:
            zone_transits.append({
                "team_number": team,
                "origin_zone": "NEUTRAL_ZONE",
                "destination_zone": "ALLIANCE_HUB",
                "transit_corridor": corr,
                "avg_transit_seconds": 2.35,
                "sample_count": 8,
                "defensive_impedance_ratio": 1.15,
            })

    actions = []
    action_types = [
        ActionType.INTAKE_FUEL,
        ActionType.SCORE_HUB,
        ActionType.TRANSIT_NEUTRAL_ZONE,
        ActionType.CLIMB_TOWER,
    ]
    elements = [FieldElement.RED_HUB, FieldElement.BLUE_HUB, FieldElement.DEPOT, FieldElement.RED_TOWER]

    for i in range(action_count):
        team = SIX_TEAMS[i % len(SIX_TEAMS)]
        act_type = action_types[i % len(action_types)]
        elem = elements[i % len(elements)]
        start_t = i * 2000
        seg = _make_action(
            event_id=f"evt_{i:04d}",
            track_id=(i % 6) + 1,
            team_number=team,
            action_type=act_type,
            element=elem,
            start_ms=start_t,
            end_ms=start_t + 1500,
        )
        actions.append(seg)

    parquet_chunk = b"PAR1" + b"\xaa" * parquet_bytes_size + b"PAR1" if parquet_bytes_size > 0 else b""

    return {
        "event_key": "2026casf",
        "match_id": "qm42",
        "sequence_number": 42,
        "generated_at_epoch_ms": 1772678400000,
        "payload_type": "PAYLOAD_DELTA_SUMMARY",
        "robot_aggregates": aggregates,
        "zone_transits": zone_transits,
        "verified_scoring_events": actions,
        "compressed_parquet_chunk": parquet_chunk,
        "parquet_table_name": "match_trajectory_qm42",
        "frc_api_red_score": 118,
        "frc_api_blue_score": 105,
        "vision_reconciled_red_score": 118,
        "vision_reconciled_blue_score": 105,
        "fully_reconciled": True,
    }


# ============================================================================
# 1. Interface Contract Validation
# ============================================================================

def test_interface_contract_inheritance() -> None:
    """Assert MQTTTelemetryPublisherActor inherits TelemetryPublisherActorBase."""
    assert issubclass(MQTTTelemetryPublisherActor, TelemetryPublisherActorBase)
    assert issubclass(TelemetryPublisherActor, TelemetryPublisherActorBase)

    actor = MQTTTelemetryPublisherActor(mock_mode=True)
    assert isinstance(actor, TelemetryPublisherActorBase)
    assert isinstance(actor, TelemetryPublisherActorProtocol)


def test_interface_contract_async_methods() -> None:
    """Verify all protocol methods are async coroutine functions."""
    methods = [
        "connect",
        "publish_live_trajectory",
        "publish_duckdb_sync",
        "disconnect",
    ]
    for method_name in methods:
        fn = getattr(MQTTTelemetryPublisherActor, method_name, None)
        assert fn is not None, f"Missing method {method_name}"
        assert inspect.iscoroutinefunction(fn), f"Method {method_name} must be an async coroutine function"


# ============================================================================
# 2. Serialization Helpers & Round-Trip Tests
# ============================================================================

def test_point_and_vector_conversion() -> None:
    """Verify 2D geometry conversions to Protobuf Point2D and Vector2D."""
    pt = to_protobuf_point_2d((12.5, 45.0))
    assert isinstance(pt, telemetry_pb2.Point2D)
    assert pt.x == pytest.approx(12.5)
    assert pt.y == pytest.approx(45.0)

    vec = to_protobuf_vector_2d({"vx": -10.0, "vy": 25.5})
    assert isinstance(vec, telemetry_pb2.Vector2D)
    assert vec.vx == pytest.approx(-10.0)
    assert vec.vy == pytest.approx(25.5)

    bbox = to_protobuf_bbox_2d(BoundingBox(10, 20, 30, 40, 0.99, 1))
    assert isinstance(bbox, telemetry_pb2.BoundingBox2D)
    assert bbox.center_x == pytest.approx(10.0)
    assert bbox.confidence == pytest.approx(0.99)


def test_robot_tracklet_to_telemetry_point_conversion() -> None:
    """Verify RobotTrackletState domain dataclass converts cleanly into RobotTelemetryPoint."""
    tracklet = _make_tracklet(track_id=1, team_number=254, alliance=AllianceColor.RED, field_x=150.0, field_y=200.0)
    pt_pb = to_protobuf_robot_telemetry_point(tracklet)

    assert isinstance(pt_pb, telemetry_pb2.RobotTelemetryPoint)
    assert pt_pb.track_id == 1
    assert pt_pb.team_number == 254
    assert pt_pb.alliance == telemetry_pb2.AllianceColor.ALLIANCE_RED
    assert pt_pb.field_position.x == pytest.approx(150.0)
    assert pt_pb.field_position.y == pytest.approx(200.0)
    assert pt_pb.velocity.vx == pytest.approx(14.5)
    assert pt_pb.reid_embedding == b"\x01\x02\x03\x04\x05\x06\x07\x08"


def test_action_segment_to_action_former_event_conversion() -> None:
    """Verify ActionSegment domain dataclass converts cleanly into ActionFormerEvent."""
    action = _make_action("act_1", 2, 1678, ActionType.SCORE_HUB, FieldElement.BLUE_HUB, 1000, 2500)
    ev_pb = to_protobuf_action_former_event(action)

    assert isinstance(ev_pb, telemetry_pb2.ActionFormerEvent)
    assert ev_pb.event_id == "act_1"
    assert ev_pb.team_number == 1678
    assert ev_pb.action_type == telemetry_pb2.ActionType.ACTION_SCORE_HUB
    assert ev_pb.primary_element == telemetry_pb2.FieldElement.ELEMENT_BLUE_HUB
    assert ev_pb.start_time_ms == 1000
    assert ev_pb.end_time_ms == 2500
    assert ev_pb.duration_seconds == pytest.approx(1.5)
    assert ev_pb.reconciliation == telemetry_pb2.ReconciliationStatus.RECON_STATUS_CONFIRMED
    assert ev_pb.reconciled_posterior == pytest.approx(0.97)


def test_envelope_serialization_and_unpack_roundtrip() -> None:
    """Verify TelemetryEnvelope wrapping, binary serialization, and clean deserialization."""
    tracklet = _make_tracklet(track_id=3, team_number=1323, alliance=AllianceColor.BLUE)
    frame = to_protobuf_trajectory_frame([tracklet])

    raw_bytes = serialize_payload(frame, wrap_envelope=True)
    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 0

    envelope = deserialize_envelope(raw_bytes)
    assert isinstance(envelope, telemetry_pb2.TelemetryEnvelope)
    assert envelope.schema_version == 1
    assert envelope.WhichOneof("payload") == "trajectory_frame"

    unpacked = unpack_payload(raw_bytes)
    assert isinstance(unpacked, telemetry_pb2.TrajectoryFrame)
    assert len(unpacked.robots) == 1
    assert unpacked.robots[0].team_number == 1323


# ============================================================================
# 3. Payload Budget Verification (< 50 KB / 51,200 Bytes)
# ============================================================================

def test_full_match_sync_payload_wire_budget() -> None:
    """
    Build a comprehensive full-match sync payload for all 6 competing robots:
    - 6 RobotMatchAggregates
    - 24 ZoneTransitStatistics
    - 60 ActionFormerEvents
    - 4 KB Parquet delta chunk
    Verify the serialized wire size is strictly < 51,200 bytes (< 50 KB ceiling).
    """
    payload_dict = _build_full_match_payload_dict(action_count=60, parquet_bytes_size=4096)
    sync_pb = to_protobuf_duckdb_sync_payload(payload_dict)

    # 1. Verify message fields
    assert len(sync_pb.robot_aggregates) == 6
    assert len(sync_pb.zone_transits) == 24
    assert len(sync_pb.verified_scoring_events) == 60
    assert sync_pb.fully_reconciled is True

    # 2. Serialize inside TelemetryEnvelope
    wire_bytes = serialize_payload(sync_pb, wrap_envelope=True)
    wire_size = len(wire_bytes)

    # 3. Strictly enforce < 51,200 bytes limit
    assert wire_size < MAX_PAYLOAD_SIZE_BYTES, (
        f"Full match sync payload ({wire_size} bytes) exceeded maximum 50 KB budget ({MAX_PAYLOAD_SIZE_BYTES} bytes)"
    )
    # Check that guardrail passes without error
    validated_bytes = enforce_payload_size_guardrail(wire_bytes, max_bytes=MAX_PAYLOAD_SIZE_BYTES)
    assert len(validated_bytes) == wire_size

    # 4. Deserialization round-trip
    unpacked_sync = unpack_payload(wire_bytes)
    assert isinstance(unpacked_sync, telemetry_pb2.DuckDBSyncPayload)
    assert len(unpacked_sync.robot_aggregates) == 6
    assert unpacked_sync.robot_aggregates[0].team_number == 254
    assert len(unpacked_sync.verified_scoring_events) == 60


def test_packet_size_guardrail_triggers_on_oversized_payload() -> None:
    """Test that supplying an oversized payload (> 51,200 bytes) triggers PayloadSizeExceededError (ValueError)."""
    # Create an oversized Parquet chunk of 60 KB
    oversized_parquet = b"\x00" * 60000
    payload_dict = _build_full_match_payload_dict(action_count=20, parquet_bytes_size=0)
    payload_dict["compressed_parquet_chunk"] = oversized_parquet

    sync_pb = to_protobuf_duckdb_sync_payload(payload_dict)
    oversized_bytes = serialize_payload(sync_pb, wrap_envelope=True)

    assert len(oversized_bytes) > MAX_PAYLOAD_SIZE_BYTES

    # Assert enforce_payload_size_guardrail raises ValueError / PayloadSizeExceededError
    with pytest.raises(ValueError) as exc_info:
        enforce_payload_size_guardrail(oversized_bytes, max_bytes=MAX_PAYLOAD_SIZE_BYTES)
    assert issubclass(exc_info.type, PayloadSizeExceededError)


def test_packet_size_guardrail_adaptive_pruning() -> None:
    """Test adaptive pruning drops optional large Parquet chunk to fit under 50 KB ceiling."""
    oversized_parquet = b"\xaa" * 55000
    payload_dict = _build_full_match_payload_dict(action_count=30, parquet_bytes_size=0)
    payload_dict["compressed_parquet_chunk"] = oversized_parquet

    sync_pb = to_protobuf_duckdb_sync_payload(payload_dict)
    raw_oversized = serialize_payload(sync_pb, wrap_envelope=True)
    assert len(raw_oversized) > MAX_PAYLOAD_SIZE_BYTES

    # Commencing adaptive pruning
    pruned_pb, pruned_bytes = prune_duckdb_sync_payload(sync_pb, max_bytes=MAX_PAYLOAD_SIZE_BYTES, wrap_envelope=True)

    # Parquet chunk pruned; now within budget
    assert len(pruned_bytes) < MAX_PAYLOAD_SIZE_BYTES
    assert pruned_pb.compressed_parquet_chunk == b""
    assert len(pruned_pb.robot_aggregates) == 6


# ============================================================================
# 4. Publishing & QoS Validation (QoS 0 and QoS 1)
# ============================================================================

@pytest.mark.asyncio
async def test_publish_live_trajectory_qos_0() -> None:
    """Test publishing live trajectory frame at QoS 0 via Loopback client."""
    publisher = MQTTTelemetryPublisherActor(mock_mode=True)
    connected = await publisher.connect("loopback", port=1883, client_id="test-pub-qos0")
    assert connected is True
    assert publisher.is_connected is True

    # Build trajectory frame with 6 robots
    robots = [_make_tracklet(i + 1, SIX_TEAMS[i], AllianceColor.RED if i < 3 else AllianceColor.BLUE) for i in range(6)]
    frame = {
        "frame_index": 120,
        "match_time_ms": 4000,
        "match_remaining_seconds": 131.0,
        "match_stage": "TELEOP",
        "robots": robots,
    }

    pub_success = await publisher.publish_live_trajectory(
        topic="frc/2026casf/qm42/trajectory",
        trajectory_frame=frame,
        qos=0,
    )
    assert pub_success is True

    packets = publisher.published_packets
    assert len(packets) == 1
    p = packets[0]
    assert p.topic == "frc/2026casf/qm42/trajectory"
    assert p.qos == 0
    assert p.sequence_number == 1
    assert p.payload_size_bytes > 0

    # Verify message payload can be unpacked
    unpacked = unpack_payload(p.payload_bytes)
    assert isinstance(unpacked, telemetry_pb2.TrajectoryFrame)
    assert unpacked.frame_index == 120
    assert unpacked.match_stage == telemetry_pb2.MatchStage.STAGE_TELEOP
    assert len(unpacked.robots) == 6

    await publisher.disconnect()
    assert publisher.is_connected is False


@pytest.mark.asyncio
async def test_publish_duckdb_sync_qos_1() -> None:
    """Test publishing authoritative DuckDB match sync payload at QoS 1."""
    publisher = MQTTTelemetryPublisherActor(mock_mode=True)
    await publisher.connect("mock", port=1883, client_id="test-pub-qos1")

    payload_dict = _build_full_match_payload_dict(action_count=50, parquet_bytes_size=2048)

    pub_success = await publisher.publish_duckdb_sync(
        topic="frc/2026casf/qm42/sync",
        sync_payload=payload_dict,
        qos=1,
    )
    assert pub_success is True

    packets = publisher.published_packets
    assert len(packets) == 1
    p = packets[0]
    assert p.topic == "frc/2026casf/qm42/sync"
    assert p.qos == 1
    assert p.payload_size_bytes < MAX_PAYLOAD_SIZE_BYTES

    # Assert round-trip payload integrity
    unpacked = unpack_payload(p.payload_bytes)
    assert isinstance(unpacked, telemetry_pb2.DuckDBSyncPayload)
    assert unpacked.match_id == "qm42"
    assert len(unpacked.robot_aggregates) == 6
    assert len(unpacked.verified_scoring_events) == 50

    await publisher.disconnect()


@pytest.mark.asyncio
async def test_publish_duckdb_sync_oversized_triggers_error() -> None:
    """Assert publish_duckdb_sync raises PayloadSizeExceededError when oversized and pruning is disabled."""
    publisher = MQTTTelemetryPublisherActor(mock_mode=True, auto_prune_oversized=False)
    await publisher.connect("loopback", port=1883)

    oversized_payload = _build_full_match_payload_dict(action_count=20, parquet_bytes_size=0)
    oversized_payload["compressed_parquet_chunk"] = b"\xff" * 60000

    with pytest.raises(PayloadSizeExceededError) as exc_info:
        await publisher.publish_duckdb_sync(
            topic="frc/sync",
            sync_payload=oversized_payload,
            qos=1,
            prune_if_oversized=False,
        )
    assert "exceeds maximum venue wireless budget" in str(exc_info.value)

    await publisher.disconnect()


# ============================================================================
# 5. Network Disconnect & Automated Reconnection
# ============================================================================

@pytest.mark.asyncio
async def test_network_disconnect_and_reconnection_recovery() -> None:
    """
    Simulate network carrier drop and recovery:
    1. Connect and publish an initial message.
    2. Simulate unexpected network partition / disconnect.
    3. Assert publisher detects disconnect and drops subsequent sends.
    4. Trigger recovery and assert publisher resynchronizes and resumes publishing.
    """
    publisher = MQTTTelemetryPublisherActor(mock_mode=True)
    await publisher.connect("loopback", port=1883, client_id="test-recon")
    assert publisher.is_connected is True

    # 1. Publish initial live trajectory
    res1 = await publisher.publish_live_trajectory(
        topic="frc/live",
        trajectory_frame={"frame_index": 1, "robots": []},
        qos=0,
    )
    assert res1 is True

    # 2. Simulate network carrier drop
    publisher.simulate_network_drop(recovery_delay_s=0.0)
    assert publisher.is_connected is False

    # 3. Publish during disconnect fails safely
    res_failed = await publisher.publish_live_trajectory(
        topic="frc/live",
        trajectory_frame={"frame_index": 2, "robots": []},
        qos=0,
    )
    assert res_failed is False

    # 4. Trigger recovery resynchronization
    reconnected = await publisher.reconnect()
    assert reconnected is True
    assert publisher.is_connected is True
    assert publisher.reconnect_count >= 1

    # 5. Resume publishing after resynchronization
    res2 = await publisher.publish_live_trajectory(
        topic="frc/live",
        trajectory_frame={"frame_index": 3, "robots": []},
        qos=0,
    )
    assert res2 is True
    assert len(publisher.published_packets) == 2

    await publisher.disconnect()


# ============================================================================
# 6. Real TCP Socket Loopback Validation with Paho-MQTT v2
# ============================================================================

class _MiniLoopbackMQTTServer:
    """Minimal in-process TCP server handling real MQTT CONNECT, PUBLISH, PUBACK, DISCONNECT."""

    def __init__(self) -> None:
        self.server: Optional[asyncio.Server] = None
        self.port: int = 0
        self.received_messages: List[Tuple[int, bytes]] = []

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while not reader.at_eof():
            try:
                b1 = await reader.read(1)
                if not b1:
                    break
                pkt_type = b1[0] >> 4

                # Read variable byte length
                multiplier = 1
                rem_len = 0
                while True:
                    b = await reader.read(1)
                    if not b:
                        break
                    digit = b[0]
                    rem_len += (digit & 127) * multiplier
                    multiplier *= 128
                    if (digit & 128) == 0:
                        break

                payload = await reader.readexactly(rem_len) if rem_len > 0 else b""

                if pkt_type == 1:  # CONNECT -> CONNACK
                    writer.write(bytes([0x20, 0x02, 0x00, 0x00]))
                    await writer.drain()
                elif pkt_type == 3:  # PUBLISH -> (PUBACK if QoS 1)
                    qos = (b1[0] & 0x06) >> 1
                    self.received_messages.append((qos, payload))
                    if qos == 1:
                        topic_len = (payload[0] << 8) | payload[1]
                        packet_id = payload[topic_len + 2 : topic_len + 4]
                        writer.write(bytes([0x40, 0x02]) + packet_id)
                        await writer.drain()
                elif pkt_type == 14:  # DISCONNECT
                    break
            except Exception:
                break
        writer.close()
        await writer.wait_closed()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()


@pytest.mark.asyncio
async def test_real_tcp_broker_roundtrip_with_paho_v2() -> None:
    """
    Test real Paho-MQTT v2 client over local TCP ephemeral port:
    - Real socket connection
    - QoS 0 publish
    - QoS 1 publish with PUBACK handshake
    """
    broker = _MiniLoopbackMQTTServer()
    await broker.start()

    publisher = MQTTTelemetryPublisherActor(connect_timeout=2.0, publish_timeout=2.0)
    try:
        connected = await publisher.connect("127.0.0.1", port=broker.port, client_id="test-tcp-pub")
        assert connected is True
        assert publisher.is_connected is True

        # Publish QoS 0
        pub0 = await publisher.publish_live_trajectory(
            topic="frc/live",
            trajectory_frame={"frame_index": 1, "robots": []},
            qos=0,
        )
        assert pub0 is True

        # Publish QoS 1 DuckDB sync
        sync_dict = _build_full_match_payload_dict(action_count=5, parquet_bytes_size=512)
        pub1 = await publisher.publish_duckdb_sync(
            topic="frc/sync",
            sync_payload=sync_dict,
            qos=1,
        )
        assert pub1 is True

        # Allow TCP packets to flush
        await asyncio.sleep(0.1)
        assert len(broker.received_messages) >= 2

        await publisher.disconnect()
        assert publisher.is_connected is False
    finally:
        await broker.stop()
