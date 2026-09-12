"""
Core ABC and Protocol base classes for Ray pipeline actors in the FRC 2026 REBUILT scouting architecture.

Adheres to:
- Ray Core async actor execution patterns (SPEC § Distributed Infrastructure)
- WorkerWithPipelining streaming design pattern
- Strongly-typed inter-actor RPC contracts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from src.common.types import (
    ActionSegment,
    BoundingBox,
    DetectionResult,
    FieldAPIScoreState,
    FrameMetadata,
    HomographyEstimate,
    OCRResult,
    RawDecodedFrame,
    RobotTrackletState,
    TagDetectionResult,
    TelemetrySyncPacket,
)


# ============================================================================
# 1. Stream Ingestion & Gating Interfaces
# ============================================================================

@runtime_checkable
class StreamIngestionActorProtocol(Protocol):
    """Protocol for CPU-bound video ingestion and ResNet-18 frame gating actors."""

    async def initialize_stream(self, stream_uri: str, target_fps: int = 60) -> bool:
        """Connects to RTMP/HLS broadcast stream and initializes decoder queues."""
        ...

    async def get_total_frames(self) -> Optional[int]:
        """Returns total frame count of the source stream/file if known."""
        ...

    async def get_next_gameplay_batch(self, batch_size: int = 8) -> List[RawDecodedFrame]:
        """Pipelined retrieval of next batch of active gameplay frames, dropping crowd/graphics."""
        ...

    async def is_stream_active(self) -> bool:
        """Returns True if the broadcast stream is currently receiving packets."""
        ...

    async def close_stream(self) -> None:
        """Gracefully closes network stream and flushes FFmpeg buffers."""
        ...


class StreamIngestionActorBase(ABC):
    """Abstract Base Class for Stream Ingestion Actors."""

    @abstractmethod
    async def initialize_stream(self, stream_uri: str, target_fps: int = 60) -> bool:
        pass

    async def get_total_frames(self) -> Optional[int]:
        """Returns total frame count of the source stream/file if known."""
        return None

    @abstractmethod
    async def get_next_gameplay_batch(self, batch_size: int = 8) -> List[RawDecodedFrame]:
        pass

    @abstractmethod
    async def is_stream_active(self) -> bool:
        pass

    @abstractmethod
    async def close_stream(self) -> None:
        pass


# ============================================================================
# 2. AprilTag Detection & Dynamic Homography Interfaces
# ============================================================================

@runtime_checkable
class HomographyActorProtocol(Protocol):
    """Protocol for AprilTag fiducial detection, DLT, and Levenberg-Marquardt homography actors."""

    async def detect_apriltags(self, frame: RawDecodedFrame) -> List[TagDetectionResult]:
        """Locates up to 32 AprilTag markers on field elements within a single frame."""
        ...

    async def estimate_homography(
        self,
        frame: RawDecodedFrame,
        detections: List[TagDetectionResult],
        prev_homography: Optional[HomographyEstimate] = None,
    ) -> HomographyEstimate:
        """Computes 3x3 homography matrix via normalized DLT + Levenberg-Marquardt optimization."""
        ...

    async def project_pixel_to_field(
        self,
        pixel_coords: Tuple[float, float],
        homography: HomographyEstimate,
    ) -> Tuple[float, float]:
        """Transforms 2D pixel coordinates (u, v) into physical field coordinates (x, y) in inches."""
        ...

    async def compute_camera_motion_compensation(
        self,
        curr_frame: np.ndarray,
        prev_frame: np.ndarray,
    ) -> np.ndarray:
        """Calculates 2x3 affine background transformation matrix for BoT-SORT CMC."""
        ...


class HomographyActorBase(ABC):
    """Abstract Base Class for Dynamic Homography Estimation."""

    @abstractmethod
    async def detect_apriltags(self, frame: RawDecodedFrame) -> List[TagDetectionResult]:
        pass

    @abstractmethod
    async def estimate_homography(
        self,
        frame: RawDecodedFrame,
        detections: List[TagDetectionResult],
        prev_homography: Optional[HomographyEstimate] = None,
    ) -> HomographyEstimate:
        pass

    @abstractmethod
    async def project_pixel_to_field(
        self,
        pixel_coords: Tuple[float, float],
        homography: HomographyEstimate,
    ) -> Tuple[float, float]:
        pass

    @abstractmethod
    async def compute_camera_motion_compensation(
        self,
        curr_frame: np.ndarray,
        prev_frame: np.ndarray,
    ) -> np.ndarray:
        pass


# ============================================================================
# 3. TensorRT Object Detection & Super-Resolution OCR Interfaces
# ============================================================================

@runtime_checkable
class DetectionActorProtocol(Protocol):
    """Protocol for GPU-bound TensorRT YOLOv11 robot detection actors."""

    async def load_engine(self, engine_path: str, precision: str = "fp16") -> None:
        """Loads and binds serialized TensorRT execution engine on target GPU."""
        ...

    async def detect_batch(self, frames: List[RawDecodedFrame]) -> List[DetectionResult]:
        """Executes batched inference returning localized robot bounding box candidates."""
        ...


class DetectionActorBase(ABC):
    """Abstract Base Class for TensorRT Object Detectors."""

    @abstractmethod
    async def load_engine(self, engine_path: str, precision: str = "fp16") -> None:
        pass

    @abstractmethod
    async def detect_batch(self, frames: List[RawDecodedFrame]) -> List[DetectionResult]:
        pass


@runtime_checkable
class OCREnhancerActorProtocol(Protocol):
    """Protocol for Real-ESRGAN super-resolution and bumper text recognition actors."""

    async def enhance_and_read_bumper(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
    ) -> OCRResult:
        """Applies Real-ESRGAN to degraded bumper crops and identifies team number and alliance."""
        ...

    async def batch_enhance_bumpers(
        self,
        crops: List[np.ndarray],
    ) -> List[OCRResult]:
        """High-throughput batched super-resolution and OCR inference on extracted crops."""
        ...


class OCREnhancerActorBase(ABC):
    """Abstract Base Class for Bumper Super-Resolution & OCR."""

    @abstractmethod
    async def enhance_and_read_bumper(
        self,
        frame_bgr: np.ndarray,
        bbox: BoundingBox,
    ) -> OCRResult:
        pass

    @abstractmethod
    async def batch_enhance_bumpers(
        self,
        crops: List[np.ndarray],
    ) -> List[OCRResult]:
        pass


# ============================================================================
# 4. Multi-Object Tracking & Tracklet Bridging Interfaces
# ============================================================================

@runtime_checkable
class TrackingActorProtocol(Protocol):
    """Protocol for BoT-SORT multi-object tracking and tracklet bridging actors."""

    async def update_tracks(
        self,
        frame_index: int,
        detections: List[BoundingBox],
        ocr_results: List[OCRResult],
        homography: HomographyEstimate,
        frame_bgr: np.ndarray,
    ) -> List[RobotTrackletState]:
        """Updates Kalman filters with CMC, extracts ReID embeddings, and bridges lost tracks."""
        ...

    async def bridge_lost_tracklets(
        self,
        camera_cut: bool,
        max_bridge_gap_frames: int = 300,
    ) -> None:
        """Performs cosine-distance ReID matching to reconnect trajectories after broadcast cuts."""
        ...

    async def get_active_tracks(self) -> List[RobotTrackletState]:
        """Returns the current state of all 6 active robot trajectories."""
        ...


class TrackingActorBase(ABC):
    """Abstract Base Class for Multi-Object Tracking."""

    @abstractmethod
    async def update_tracks(
        self,
        frame_index: int,
        detections: List[BoundingBox],
        ocr_results: List[OCRResult],
        homography: HomographyEstimate,
        frame_bgr: np.ndarray,
    ) -> List[RobotTrackletState]:
        pass

    @abstractmethod
    async def bridge_lost_tracklets(
        self,
        camera_cut: bool,
        max_bridge_gap_frames: int = 300,
    ) -> None:
        pass

    @abstractmethod
    async def get_active_tracks(self) -> List[RobotTrackletState]:
        pass


# ============================================================================
# 5. Temporal Action Recognition (ActionFormer) Interfaces
# ============================================================================

@runtime_checkable
class ActionRecognitionActorProtocol(Protocol):
    """Protocol for ActionFormer multi-scale temporal action localization actors."""

    async def ingest_tracklet_window(
        self,
        track_states: List[RobotTrackletState],
        window_size: int = 128,
    ) -> None:
        """Appends temporal feature vectors derived from tracked robot bounding boxes and dynamics."""
        ...

    async def localize_actions(
        self,
        track_id: int,
    ) -> List[ActionSegment]:
        """Runs ActionFormer local self-attention pyramid to predict action boundaries and classifications."""
        ...


class ActionRecognitionActorBase(ABC):
    """Abstract Base Class for Temporal Action Recognition."""

    @abstractmethod
    async def ingest_tracklet_window(
        self,
        track_states: List[RobotTrackletState],
        window_size: int = 128,
    ) -> None:
        pass

    @abstractmethod
    async def localize_actions(
        self,
        track_id: int,
    ) -> List[ActionSegment]:
        pass


# ============================================================================
# 6. Bayesian Ground-Truth Reconciliation Interfaces
# ============================================================================

@runtime_checkable
class ReconciliationActorProtocol(Protocol):
    """Protocol for reconciling ActionFormer events with FRC Field API ground truth."""

    async def update_field_api_score(self, score_state: FieldAPIScoreState) -> None:
        """Ingests live aggregate scoring data from official FRC Field Management System (FMS)."""
        ...

    async def reconcile_actions(
        self,
        candidate_actions: List[ActionSegment],
        robot_positions: Dict[int, Tuple[float, float]],
    ) -> List[ActionSegment]:
        """Calculates Bayesian posterior probabilities, pruning spatial false positives."""
        ...


class ReconciliationActorBase(ABC):
    """Abstract Base Class for Bayesian Reconciliation."""

    @abstractmethod
    async def update_field_api_score(self, score_state: FieldAPIScoreState) -> None:
        pass

    @abstractmethod
    async def reconcile_actions(
        self,
        candidate_actions: List[ActionSegment],
        robot_positions: Dict[int, Tuple[float, float]],
    ) -> List[ActionSegment]:
        pass


# ============================================================================
# 7. DuckDB Spatio-Temporal Storage Interfaces
# ============================================================================

@runtime_checkable
class StorageActorProtocol(Protocol):
    """Protocol for DuckDB in-process OLAP database and Parquet trajectory storage actors."""

    async def insert_trajectory_batch(self, trajectories: List[RobotTrackletState], frame_index: int) -> None:
        """Appends trajectory state batch to DuckDB memory-mapped Parquet buffer."""
        ...

    async def insert_action_events(self, actions: List[ActionSegment]) -> None:
        """Inserts reconciled action localization events into DuckDB action tables."""
        ...

    async def execute_asof_join_query(
        self,
        match_id: str,
        team_number: int,
    ) -> Dict[str, Any]:
        """Executes vectorized ASOF JOIN merging trajectories with discrete scoring events."""
        ...

    async def export_match_parquet(self, output_path: str) -> str:
        """Flushes in-memory columnar data to disk for persistent archival."""
        ...


class StorageActorBase(ABC):
    """Abstract Base Class for DuckDB Trajectory Analytics & Storage."""

    @abstractmethod
    async def insert_trajectory_batch(self, trajectories: List[RobotTrackletState], frame_index: int) -> None:
        pass

    @abstractmethod
    async def insert_action_events(self, actions: List[ActionSegment]) -> None:
        pass

    @abstractmethod
    async def execute_asof_join_query(
        self,
        match_id: str,
        team_number: int,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def export_match_parquet(self, output_path: str) -> str:
        pass


# ============================================================================
# 8. Low-Bandwidth Transport / Client Sync Interfaces
# ============================================================================

@runtime_checkable
class TelemetryPublisherActorProtocol(Protocol):
    """Protocol for Protobuf binary serialization and MQTT publisher actors (<50 KB budget)."""

    async def connect(self, broker_url: str, port: int = 1883, client_id: str = "scout-pub") -> bool:
        """Establishes persistent connection to venue MQTT broker."""
        ...

    async def publish_live_trajectory(
        self,
        topic: str,
        trajectory_frame: Any,
        qos: int = 0,
    ) -> bool:
        """Serializes and publishes live trajectory frame via Protobuf over MQTT."""
        ...

    async def publish_duckdb_sync(
        self,
        topic: str,
        sync_payload: Any,
        qos: int = 1,
    ) -> bool:
        """Publishes authoritative match aggregate delta under strict <50KB payload budget."""
        ...

    async def disconnect(self) -> None:
        """Cleanly disconnects publisher from MQTT broker."""
        ...


class TelemetryPublisherActorBase(ABC):
    """Abstract Base Class for Telemetry Publisher."""

    @abstractmethod
    async def connect(self, broker_url: str, port: int = 1883, client_id: str = "scout-pub") -> bool:
        pass

    @abstractmethod
    async def publish_live_trajectory(
        self,
        topic: str,
        trajectory_frame: Any,
        qos: int = 0,
    ) -> bool:
        pass

    @abstractmethod
    async def publish_duckdb_sync(
        self,
        topic: str,
        sync_payload: Any,
        qos: int = 1,
    ) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass


# ============================================================================
# 9. Pipeline Orchestration & Streaming DAG Interface
# ============================================================================

@runtime_checkable
class PipelineOrchestratorProtocol(Protocol):
    """Protocol for coordinating end-to-end Ray pipeline actors across cluster nodes."""

    async def start_pipeline(self, stream_uri: str, match_id: str) -> None:
        """Initializes and connects all Ray actors in the streaming DAG."""
        ...

    async def process_match_stream(self) -> None:
        """Executes streaming pipelining loop with backpressure regulation."""
        ...

    async def stop_pipeline(self) -> Dict[str, Any]:
        """Tears down stream, flushes storage, and returns match execution summary."""
        ...
