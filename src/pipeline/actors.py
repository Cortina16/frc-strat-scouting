"""
Ray Core distributed actor wrappers and invocation utilities for the FRC 2026 REBUILT scouting pipeline.

Wraps all pipeline actors with @ray.remote, assigning appropriate CPU and GPU resource
allocations. Supports transparent invocation whether running in distributed Ray mode
or direct in-process local mode.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import ray

from src.actions.action_former import ActionRecognitionActor
from src.analytics.duckdb_store import DuckDBStorageActor
from src.common.interfaces import (
    ActionRecognitionActorBase,
    ActionRecognitionActorProtocol,
    DetectionActorBase,
    DetectionActorProtocol,
    HomographyActorBase,
    HomographyActorProtocol,
    OCREnhancerActorBase,
    OCREnhancerActorProtocol,
    ReconciliationActorBase,
    ReconciliationActorProtocol,
    StorageActorBase,
    StorageActorProtocol,
    StreamIngestionActorBase,
    StreamIngestionActorProtocol,
    TelemetryPublisherActorBase,
    TelemetryPublisherActorProtocol,
    TrackingActorBase,
    TrackingActorProtocol,
)
from src.common.types import (
    ActionSegment,
    BoundingBox,
    DetectionResult,
    FieldAPIScoreState,
    HomographyEstimate,
    OCRResult,
    RawDecodedFrame,
    RobotTrackletState,
    TagDetectionResult,
    TelemetrySyncPacket,
)
from src.ingestion.stream_decoder import StreamIngestionActor
from src.reconciliation.bayesian_engine import ReconciliationActor
from src.tracking.bot_sort import TrackingActor
from src.transport.mqtt_publisher import MQTTTelemetryPublisherActor
from src.vision.detector_trt import TensorRTDetectorActor
from src.vision.homography import HomographyActor
from src.vision.ocr_enhancer import OCREnhancerActor

logger = logging.getLogger("frc.pipeline.actors")


# ============================================================================
# 1. Hardware & Resource Detection
# ============================================================================

def get_available_gpu_count() -> int:
    """
    Returns 1 if a CUDA GPU is available via PyTorch and not in a CPU test environment;
    otherwise returns 0 so Ray actors schedule immediately on CPU-only nodes.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 1
    except (ImportError, Exception):
        pass
    return 0


# ============================================================================
# 2. Remote Method Enhancements (Zero-Copy Plasma Store & Batched RPCs)
# ============================================================================

async def resolve_frame(frame: Any) -> RawDecodedFrame:
    """
    Resolves a frame whether passed as a concrete RawDecodedFrame or a
    ray.ObjectRef[RawDecodedFrame] from the Ray Plasma shared memory store.
    Awaits non-blockingly without stalling the actor event loop.
    """
    if isinstance(frame, ray.ObjectRef):
        return await frame
    return frame


async def resolve_frames_batch(frames: List[Any]) -> List[RawDecodedFrame]:
    """
    Resolves a list of frames concurrently using asyncio.gather without
    blocking the Ray worker event loop.
    """
    resolved = list(frames)
    ref_indices = [i for i, f in enumerate(frames) if isinstance(f, ray.ObjectRef)]
    if ref_indices:
        refs_to_await = [frames[i] for i in ref_indices]
        awaited_frames = await asyncio.gather(*refs_to_await)
        for idx, f in zip(ref_indices, awaited_frames):
            resolved[idx] = f
    return resolved  # type: ignore[return-value]


async def resolve_frame_bgr(frame_or_bgr: Any) -> Optional[np.ndarray]:
    """
    Extracts the uint8 BGR image array whether passed as:
    - RawDecodedFrame
    - ray.ObjectRef[RawDecodedFrame]
    - np.ndarray
    - ray.ObjectRef[np.ndarray]
    - None
    Awaits non-blockingly without thread contention.
    """
    if frame_or_bgr is None:
        return None
    if isinstance(frame_or_bgr, ray.ObjectRef):
        frame_or_bgr = await frame_or_bgr
    if hasattr(frame_or_bgr, "image_bgr"):
        return frame_or_bgr.image_bgr
    if isinstance(frame_or_bgr, np.ndarray):
        return frame_or_bgr
    return None


def actor_has_method(actor: Any, method_name: str) -> bool:
    """Safely checks if an actor (remote ActorHandle or local instance) supports a method."""
    try:
        if hasattr(actor, method_name):
            return getattr(actor, method_name, None) is not None
        return False
    except (AttributeError, Exception):
        return False


# --- StreamIngestionActor: Batched retrieval with Plasma store references ---
if not hasattr(StreamIngestionActor, "get_next_batch_with_refs"):
    async def _get_next_batch_with_refs(
        self: StreamIngestionActor, batch_size: int = 8
    ) -> Tuple[List[RawDecodedFrame], List[Any]]:
        batch: List[RawDecodedFrame] = []
        while len(batch) < batch_size:
            sub = await self.get_next_gameplay_batch(batch_size=batch_size - len(batch))
            if not sub:
                break
            batch.extend(sub)
        refs = list(self._last_ray_refs) if hasattr(self, "_last_ray_refs") else []
        return batch, refs
    StreamIngestionActor.get_next_batch_with_refs = _get_next_batch_with_refs  # type: ignore[attr-defined]


# --- HomographyActor: Zero-copy ObjectRef resolution & batched homography estimation ---
if not hasattr(HomographyActor, "_orig_detect_apriltags"):
    HomographyActor._orig_detect_apriltags = HomographyActor.detect_apriltags  # type: ignore[attr-defined]

    async def _detect_apriltags(self: HomographyActor, frame: Any) -> List[TagDetectionResult]:
        resolved = await resolve_frame(frame)
        return await self._orig_detect_apriltags(resolved)  # type: ignore[attr-defined]

    HomographyActor.detect_apriltags = _detect_apriltags  # type: ignore[assignment]

if not hasattr(HomographyActor, "_orig_estimate_homography"):
    HomographyActor._orig_estimate_homography = HomographyActor.estimate_homography  # type: ignore[attr-defined]

    async def _estimate_homography(
        self: HomographyActor,
        frame: Any,
        detections: List[TagDetectionResult],
        prev_homography: Optional[HomographyEstimate] = None,
    ) -> HomographyEstimate:
        resolved = await resolve_frame(frame)
        return await self._orig_estimate_homography(resolved, detections, prev_homography)  # type: ignore[attr-defined]

    HomographyActor.estimate_homography = _estimate_homography  # type: ignore[assignment]

if hasattr(HomographyActor, "estimate_batch"):
    if not hasattr(HomographyActor, "_orig_estimate_batch"):
        HomographyActor._orig_estimate_batch = HomographyActor.estimate_batch  # type: ignore[attr-defined]

        async def _estimate_batch(
            self: HomographyActor,
            frames: List[Any],
            tags_list: Optional[List[List[TagDetectionResult]]] = None,
        ) -> List[HomographyEstimate]:
            resolved_frames = await resolve_frames_batch(frames)
            return await self._orig_estimate_batch(resolved_frames, tags_list=tags_list)  # type: ignore[attr-defined]

        HomographyActor.estimate_batch = _estimate_batch  # type: ignore[assignment]
else:
    async def _estimate_batch(
        self: HomographyActor,
        frames: List[Any],
        tags_list: Optional[List[List[TagDetectionResult]]] = None,
    ) -> List[HomographyEstimate]:
        resolved_frames = await resolve_frames_batch(frames)
        estimates: List[HomographyEstimate] = []
        for f in resolved_frames:
            tags = await self.detect_apriltags(f)
            est = await self.estimate_homography(f, tags)
            estimates.append(est)
        return estimates

    HomographyActor.estimate_batch = _estimate_batch  # type: ignore[attr-defined]


# --- TensorRTDetectorActor: Zero-copy ObjectRef resolution ---
if not hasattr(TensorRTDetectorActor, "_orig_detect_batch"):
    TensorRTDetectorActor._orig_detect_batch = TensorRTDetectorActor.detect_batch  # type: ignore[attr-defined]

    async def _detect_batch(self: TensorRTDetectorActor, frames: List[Any]) -> List[DetectionResult]:
        resolved = await resolve_frames_batch(frames)
        return await self._orig_detect_batch(resolved)  # type: ignore[attr-defined]

    TensorRTDetectorActor.detect_batch = _detect_batch  # type: ignore[assignment]


# --- TrackingActor: Zero-copy ObjectRef resolution & batched multi-frame tracking ---
if not hasattr(TrackingActor, "_orig_update_tracks"):
    TrackingActor._orig_update_tracks = TrackingActor.update_tracks  # type: ignore[attr-defined]

    async def _update_tracks(
        self: TrackingActor,
        frame_index: int,
        detections: List[BoundingBox],
        ocr_results: List[OCRResult],
        homography: HomographyEstimate,
        frame_bgr: Any,
        embeddings: Optional[Sequence[np.ndarray]] = None,
        is_autonomous: Optional[bool] = None,
    ) -> List[RobotTrackletState]:
        resolved_bgr = await resolve_frame_bgr(frame_bgr)
        return await self._orig_update_tracks(
            frame_index=frame_index,
            detections=detections,
            ocr_results=ocr_results,
            homography=homography,
            frame_bgr=resolved_bgr,
            embeddings=embeddings,
            is_autonomous=is_autonomous,
        )  # type: ignore[attr-defined]

    TrackingActor.update_tracks = _update_tracks  # type: ignore[assignment]

if not hasattr(TrackingActor, "update_tracks_batch"):
    async def _update_tracks_batch(
        self: TrackingActor,
        frame_indices: List[int],
        detections_batch: List[List[BoundingBox]],
        ocr_batch: List[List[OCRResult]],
        homographies: List[HomographyEstimate],
        frames: List[Any],
        is_autonomous: Optional[bool] = None,
    ) -> List[List[RobotTrackletState]]:
        resolved_bgrs = await asyncio.gather(*[resolve_frame_bgr(f) for f in frames])
        batch_active_tracks: List[List[RobotTrackletState]] = []
        for i, frame_idx in enumerate(frame_indices):
            h_est = homographies[i]
            det_boxes = detections_batch[i]
            ocr_res = ocr_batch[i]
            frame_bgr = resolved_bgrs[i]

            if h_est.camera_cut:
                await self.bridge_lost_tracklets(camera_cut=True)

            active_tracks = await self.update_tracks(
                frame_index=frame_idx,
                detections=det_boxes,
                ocr_results=ocr_res,
                homography=h_est,
                frame_bgr=frame_bgr,
                is_autonomous=is_autonomous,
            )

            if not h_est.camera_cut:
                await self.bridge_lost_tracklets(camera_cut=False)

            batch_active_tracks.append(active_tracks)

        return batch_active_tracks

    TrackingActor.update_tracks_batch = _update_tracks_batch  # type: ignore[attr-defined]


# --- ActionRecognitionActor: Batched action localization ---
if not hasattr(ActionRecognitionActor, "localize_all_actions"):
    async def _localize_all_actions(
        self: ActionRecognitionActor,
        track_ids: Optional[Sequence[int]] = None,
    ) -> Dict[int, List[ActionSegment]]:
        if track_ids is None:
            track_ids = list(self._buffer.tracks.keys())
        results: Dict[int, List[ActionSegment]] = {}
        for tid in track_ids:
            results[tid] = await self.localize_actions(tid)
        return results

    ActionRecognitionActor.localize_all_actions = _localize_all_actions  # type: ignore[attr-defined]
    ActionRecognitionActor.localize_batch = _localize_all_actions  # type: ignore[attr-defined]


# --- DuckDBStorageActor: Batched multi-frame trajectory and action ingestion ---
if not hasattr(DuckDBStorageActor, "insert_multi_frame_batch"):
    async def _insert_multi_frame_batch(
        self: DuckDBStorageActor,
        frame_trajectories: List[Tuple[List[RobotTrackletState], int]],
        action_events: Optional[List[ActionSegment]] = None,
    ) -> None:
        async with self._lock:
            for trajectories, frame_index in frame_trajectories:
                if trajectories:
                    self._store.insert_trajectory_batch(
                        trajectories=trajectories,
                        frame_index=frame_index,
                        match_id=self.match_id,
                    )
            if action_events:
                self._store.insert_action_events(
                    actions=action_events,
                    match_id=self.match_id,
                )

    DuckDBStorageActor.insert_multi_frame_batch = _insert_multi_frame_batch  # type: ignore[attr-defined]

if not hasattr(DuckDBStorageActor, "insert_multi_frame_ball_batch"):
    async def _insert_multi_frame_ball_batch(
        self: DuckDBStorageActor,
        frame_balls: List[Tuple[Sequence[BallTrackletState], int]],
    ) -> None:
        async with self._lock:
            if hasattr(self._store, "insert_multi_frame_ball_batch"):
                self._store.insert_multi_frame_ball_batch(frame_balls, match_id=self.match_id)
            else:
                for balls, frame_index in frame_balls:
                    if balls:
                        self._store.insert_ball_trajectory_batch(
                            balls=balls,
                            frame_index=frame_index,
                            match_id=self.match_id,
                        )

    DuckDBStorageActor.insert_multi_frame_ball_batch = _insert_multi_frame_ball_batch  # type: ignore[attr-defined]



# --- MQTTTelemetryPublisherActor: Batched live telemetry publishing ---
if not hasattr(MQTTTelemetryPublisherActor, "publish_live_trajectories_batch"):
    async def _publish_live_trajectories_batch(
        self: MQTTTelemetryPublisherActor,
        topic: str,
        payloads: List[Any],
        qos: int = 0,
    ) -> int:
        published_count = 0
        for p in payloads:
            ok = await self.publish_live_trajectory(topic, p, qos)
            if ok:
                published_count += 1
        return published_count

    MQTTTelemetryPublisherActor.publish_live_trajectories_batch = _publish_live_trajectories_batch  # type: ignore[attr-defined]

# Attach getter methods to MQTTTelemetryPublisherActor if not defined,
# enabling remote Ray ActorHandles to retrieve published packet ledgers and status.
if not hasattr(MQTTTelemetryPublisherActor, "get_published_packets"):
    def _get_published_packets(self: MQTTTelemetryPublisherActor) -> List[TelemetrySyncPacket]:
        return list(self._published_packets)
    MQTTTelemetryPublisherActor.get_published_packets = _get_published_packets  # type: ignore[attr-defined]

if not hasattr(MQTTTelemetryPublisherActor, "get_total_bytes_published"):
    def _get_total_bytes_published(self: MQTTTelemetryPublisherActor) -> int:
        return self._total_bytes_published
    MQTTTelemetryPublisherActor.get_total_bytes_published = _get_total_bytes_published  # type: ignore[attr-defined]

if not hasattr(MQTTTelemetryPublisherActor, "get_is_connected"):
    def _get_is_connected(self: MQTTTelemetryPublisherActor) -> bool:
        return self.is_connected
    MQTTTelemetryPublisherActor.get_is_connected = _get_is_connected  # type: ignore[attr-defined]


# ============================================================================
# 3. Ray Remote Actor Declarations
# ============================================================================

_NUM_GPUS = get_available_gpu_count()

RemoteStreamIngestionActor = ray.remote(num_cpus=0)(StreamIngestionActor)
RemoteHomographyActor = ray.remote(num_cpus=0)(HomographyActor)
RemoteDetectorActor = ray.remote(num_cpus=0, num_gpus=_NUM_GPUS)(TensorRTDetectorActor)
RemoteOCREnhancerActor = ray.remote(num_cpus=0, num_gpus=_NUM_GPUS)(OCREnhancerActor)
RemoteTrackingActor = ray.remote(num_cpus=0)(TrackingActor)
RemoteActionRecognitionActor = ray.remote(num_cpus=0)(ActionRecognitionActor)
RemoteReconciliationActor = ray.remote(num_cpus=0)(ReconciliationActor)
RemoteDuckDBStorageActor = ray.remote(num_cpus=0)(DuckDBStorageActor)
RemoteTelemetryPublisherActor = ray.remote(num_cpus=0)(MQTTTelemetryPublisherActor)


# ============================================================================
# 4. Transparent Actor Invocation Utilities
# ============================================================================

async def call_actor(actor: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Transparently calls a method on an actor handle, whether it is a Ray ActorHandle
    or a direct local class instance. Awaits coroutines or Ray ObjectRefs without blocking.
    """
    method = getattr(actor, method_name)
    if hasattr(method, "remote"):
        # Ray Remote Method Call
        res = method.remote(*args, **kwargs)
        if isinstance(res, ray.ObjectRef):
            return await res
        return res
    else:
        # Local Python Method Call
        res = method(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res


def dispatch_actor_call(actor: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Dispatches an actor call asynchronously for task pipelining.
    Returns a Ray ObjectRef (if remote) or an asyncio.Task (if local coroutine).
    """
    method = getattr(actor, method_name)
    if hasattr(method, "remote"):
        return method.remote(*args, **kwargs)
    else:
        res = method(*args, **kwargs)
        if inspect.isawaitable(res):
            return asyncio.create_task(res)
        return res


async def await_result(obj: Any) -> Any:
    """Awaits a Ray ObjectRef, asyncio Task, or Coroutine, returning the concrete value."""
    if isinstance(obj, ray.ObjectRef):
        return await obj
    if inspect.isawaitable(obj):
        return await obj
    return obj


# ============================================================================
# ============================================================================
# 5. Hardware Pinning & Workload Slicing Configuration
# ============================================================================

def load_hardware_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads hardware slicing and workload pinning configuration from configs/pipeline.yaml
    or provided path. Returns a structured dictionary of cluster and pinning specs.
    """
    default_cfg: Dict[str, Any] = {
        "cluster": {
            "num_cpus": 64,
            "num_gpus": 2,
            "object_store_memory": 68719476736,
            "plasma_directory": "/dev/shm",
        },
        "hardware_pinning": {
            "gpus": {
                "detection_actor": {"device_id": 0, "cuda_visible_devices": "0", "num_gpus": 1},
                "action_recognition_actor": {"device_id": 1, "cuda_visible_devices": "1", "num_gpus": 1},
            },
            "cpu_worker_pool": {
                "stream_ingestion_actor": 8,
                "homography_actor": 8,
                "duckdb_storage_actor": 8,
                "tracking_actor": 4,
                "reconciliation_actor": 4,
                "telemetry_publisher_actor": 4,
            },
            "plasma_store": {
                "mount_path": "/dev/shm",
                "memory_bytes": 68719476736,
            },
        },
    }

    path_to_try = config_path or "configs/pipeline.yaml"
    cfg_file = Path(path_to_try)
    if cfg_file.is_file():
        try:
            import yaml
            with open(cfg_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            pipe = data.get("pipeline", {})
            if "cluster" in pipe and isinstance(pipe["cluster"], dict):
                default_cfg["cluster"].update(pipe["cluster"])
            if "hardware_pinning" in pipe and isinstance(pipe["hardware_pinning"], dict):
                hp = pipe["hardware_pinning"]
                if "gpus" in hp and isinstance(hp["gpus"], dict):
                    default_cfg["hardware_pinning"]["gpus"].update(hp["gpus"])
                if "cpu_worker_pool" in hp and isinstance(hp["cpu_worker_pool"], dict):
                    default_cfg["hardware_pinning"]["cpu_worker_pool"].update(hp["cpu_worker_pool"])
                if "plasma_store" in hp and isinstance(hp["plasma_store"], dict):
                    default_cfg["hardware_pinning"]["plasma_store"].update(hp["plasma_store"])
        except Exception as exc:
            logger.warning("Could not parse hardware config from %s: %s", path_to_try, exc)

    return default_cfg


def compute_actor_hardware_options(
    config: Optional[Dict[str, Any]] = None,
    available_cpus: Optional[int] = None,
    available_gpus: Optional[int] = None,
    force_production: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Ray .options(...) parameters for all pipeline actors based on available resources
    and configs/pipeline.yaml pinning specs.

    In production mode (available_cpus >= 36 and available_gpus >= 2, or force_production=True):
      - DetectionActor: num_gpus=1, runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}
      - ActionRecognitionActor: num_gpus=1, runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": "1"}}
      - StreamIngestionActor: num_cpus=8
      - HomographyActor: num_cpus=8
      - DuckDBStorageActor: num_cpus=8
      - TrackingActor: num_cpus=4
      - ReconciliationActor: num_cpus=4
      - TelemetryPublisherActor: num_cpus=4

    In test environments (CPU-only or constrained CPU cores < 36):
      - Gracefully scales down CPU allocations (num_cpus=0) and GPU allocations (num_gpus=0)
        so unit and integration tests schedule instantly without deadlock.
    """
    hw_cfg = config or load_hardware_config()
    cpu_pool = hw_cfg.get("hardware_pinning", {}).get("cpu_worker_pool", {})
    gpu_pool = hw_cfg.get("hardware_pinning", {}).get("gpus", {})

    det_gpu_cfg = gpu_pool.get("detection_actor", {"num_gpus": 1, "cuda_visible_devices": "0"})
    act_gpu_cfg = gpu_pool.get("action_recognition_actor", {"num_gpus": 1, "cuda_visible_devices": "1"})

    req_ingestion_cpus = int(cpu_pool.get("stream_ingestion_actor", 8))
    req_homography_cpus = int(cpu_pool.get("homography_actor", 8))
    req_storage_cpus = int(cpu_pool.get("duckdb_storage_actor", 8))
    req_tracking_cpus = int(cpu_pool.get("tracking_actor", 4))
    req_reconciliation_cpus = int(cpu_pool.get("reconciliation_actor", 4))
    req_publisher_cpus = int(cpu_pool.get("telemetry_publisher_actor", 4))

    total_req_cpus = (
        req_ingestion_cpus
        + req_homography_cpus
        + req_storage_cpus
        + req_tracking_cpus
        + req_reconciliation_cpus
        + req_publisher_cpus
    )  # 36 cores total

    cpus = available_cpus if available_cpus is not None else 0
    gpus = available_gpus if available_gpus is not None else 0

    if force_production or (cpus >= total_req_cpus and gpus >= 2):
        # Full production allocation with hardware pinning
        return {
            "detection": {
                "num_cpus": 0,
                "num_gpus": int(det_gpu_cfg.get("num_gpus", 1)),
                "runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": str(det_gpu_cfg.get("cuda_visible_devices", "0"))}},
            },
            "ocr": {
                "num_cpus": 0,
                "num_gpus": 0,
                "runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": str(det_gpu_cfg.get("cuda_visible_devices", "0"))}},
            },
            "action": {
                "num_cpus": 0,
                "num_gpus": int(act_gpu_cfg.get("num_gpus", 1)),
                "runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": str(act_gpu_cfg.get("cuda_visible_devices", "1"))}},
            },
            "ingestion": {"num_cpus": req_ingestion_cpus},
            "homography": {"num_cpus": req_homography_cpus},
            "storage": {"num_cpus": req_storage_cpus},
            "tracking": {"num_cpus": req_tracking_cpus},
            "reconciliation": {"num_cpus": req_reconciliation_cpus},
            "publisher": {"num_cpus": req_publisher_cpus},
        }

    # Test / constrained environment fallback:
    # Use zero CPU allocations so actors schedule concurrently on small clusters (e.g. 4-8 cores)
    # without hitting resource exhaustion or deadlock.
    det_gpu_alloc = 0.0
    act_gpu_alloc = 0.0
    det_runtime_env = None
    act_runtime_env = None

    if gpus >= 2:
        det_gpu_alloc = 1.0
        act_gpu_alloc = 1.0
        det_runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": str(det_gpu_cfg.get("cuda_visible_devices", "0"))}}
        act_runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": str(act_gpu_cfg.get("cuda_visible_devices", "1"))}}
    elif gpus == 1:
        det_gpu_alloc = 0.5
        act_gpu_alloc = 0.5
        det_runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}
        act_runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": "0"}}

    detection_opts: Dict[str, Any] = {"num_cpus": 0, "num_gpus": det_gpu_alloc}
    if det_runtime_env:
        detection_opts["runtime_env"] = det_runtime_env

    ocr_opts: Dict[str, Any] = {"num_cpus": 0, "num_gpus": 0.0}
    if det_runtime_env:
        ocr_opts["runtime_env"] = det_runtime_env

    action_opts: Dict[str, Any] = {"num_cpus": 0, "num_gpus": act_gpu_alloc}
    if act_runtime_env:
        action_opts["runtime_env"] = act_runtime_env

    return {
        "detection": detection_opts,
        "ocr": ocr_opts,
        "action": action_opts,
        "ingestion": {"num_cpus": 0},
        "homography": {"num_cpus": 0},
        "storage": {"num_cpus": 0},
        "tracking": {"num_cpus": 0},
        "reconciliation": {"num_cpus": 0},
        "publisher": {"num_cpus": 0},
    }


# ============================================================================
# 6. Pipeline Actors Container & Factory Functions
# ============================================================================

@dataclass
class PipelineActors:
    """Container holding handles to the 9 pipeline actors."""
    ingestion: Any
    homography: Any
    detector: Any
    ocr: Any
    tracking: Any
    action: Any
    reconciliation: Any
    storage: Any
    publisher: Any
    use_ray: bool = True
    hardware_options: Optional[Dict[str, Dict[str, Any]]] = None


def create_remote_actors(
    stream_uri: Optional[str] = None,
    match_id: str = "default_match",
    target_fps: int = 30,
    db_path: str = ":memory:",
    detector_engine_path: Optional[str] = None,
    detector_mock_mode: bool = False,
    mqtt_mock_mode: bool = False,
    homography_config_path: Optional[str] = None,
    num_gpus: Optional[int] = None,
    config_path: Optional[str] = None,
    force_production: bool = False,
    red_teams: Optional[Sequence[int]] = None,
    blue_teams: Optional[Sequence[int]] = None,
    field_bounds_screen: Optional[Tuple[float, float, float, float]] = None,
    detector_input_shape: Optional[Tuple[int, int]] = None,
    field_roi_y: Optional[Tuple[float, float]] = None,
    max_coast_frames: int = 15,
    detector_conf_threshold: Optional[float] = None,
) -> PipelineActors:
    """Instantiates all 9 pipeline actors as Ray remote actors across cluster nodes."""
    cluster_cpus = 0
    cluster_gpus = 0
    if ray.is_initialized():
        res = ray.cluster_resources()
        cluster_cpus = int(res.get("CPU", 0))
        cluster_gpus = int(res.get("GPU", 0))

    effective_gpus = num_gpus if num_gpus is not None else cluster_gpus

    actor_opts = compute_actor_hardware_options(
        config=load_hardware_config(config_path),
        available_cpus=cluster_cpus,
        available_gpus=effective_gpus,
        force_production=force_production,
    )

    det_opts = dict(actor_opts["detection"])
    ocr_opts = dict(actor_opts["ocr"])
    act_opts = dict(actor_opts["action"])

    # When Ray allocates GPUs (num_gpus > 0), Ray natively manages CUDA_VISIBLE_DEVICES per worker.
    # Avoid injecting CUDA_VISIBLE_DEVICES via runtime_env when num_gpus > 0 to prevent Ray internal IndexError.
    if det_opts.get("num_gpus", 0) > 0 and "runtime_env" in det_opts:
        det_opts = {k: v for k, v in det_opts.items() if k != "runtime_env"}
    if act_opts.get("num_gpus", 0) > 0 and "runtime_env" in act_opts:
        act_opts = {k: v for k, v in act_opts.items() if k != "runtime_env"}

    ingestion = RemoteStreamIngestionActor.options(**actor_opts["ingestion"]).remote(
        backend="auto",
        target_fps=target_fps,
    )
    homography = RemoteHomographyActor.options(**actor_opts["homography"]).remote(
        config_path=homography_config_path,
    )
    det_kwargs: Dict[str, Any] = {
        "engine_path": detector_engine_path,
        "mock_mode": detector_mock_mode,
        "generate_mock_boxes_if_empty": True,
        "input_shape": detector_input_shape or (360, 640),
        "field_bounds": field_bounds_screen,
        "field_roi_y": field_roi_y,
    }
    if detector_conf_threshold is not None:
        det_kwargs["conf_threshold"] = detector_conf_threshold
    detector = RemoteDetectorActor.options(**det_opts).remote(**det_kwargs)
    ocr = RemoteOCREnhancerActor.options(**ocr_opts).remote(
        super_res_scale=2,
        red_teams=red_teams,
        blue_teams=blue_teams,
    )
    tracking = RemoteTrackingActor.options(**actor_opts["tracking"]).remote(
        target_fps=float(target_fps),
        red_teams=red_teams,
        blue_teams=blue_teams,
        max_coast_frames=max_coast_frames,
        mandatory_attribution=bool(red_teams or blue_teams),
    )
    action = RemoteActionRecognitionActor.options(**act_opts).remote(
        fps=float(target_fps),
    )
    reconciliation = RemoteReconciliationActor.options(**actor_opts["reconciliation"]).remote()
    storage = RemoteDuckDBStorageActor.options(**actor_opts["storage"]).remote(
        db_path=db_path,
        match_id=match_id,
        fps=float(target_fps),
    )
    publisher = RemoteTelemetryPublisherActor.options(**actor_opts["publisher"]).remote(
        mock_mode=mqtt_mock_mode,
        auto_prune_oversized=True,
    )

    return PipelineActors(
        ingestion=ingestion,
        homography=homography,
        detector=detector,
        ocr=ocr,
        tracking=tracking,
        action=action,
        reconciliation=reconciliation,
        storage=storage,
        publisher=publisher,
        use_ray=True,
        hardware_options=actor_opts,
    )


def create_local_actors(
    stream_uri: Optional[str] = None,
    match_id: str = "default_match",
    target_fps: int = 30,
    db_path: str = ":memory:",
    detector_engine_path: Optional[str] = None,
    detector_mock_mode: bool = False,
    mqtt_mock_mode: bool = False,
    homography_config_path: Optional[str] = None,
    num_gpus: Optional[int] = None,
    config_path: Optional[str] = None,
    red_teams: Optional[Sequence[int]] = None,
    blue_teams: Optional[Sequence[int]] = None,
    field_bounds_screen: Optional[Tuple[float, float, float, float]] = None,
    detector_input_shape: Optional[Tuple[int, int]] = None,
    field_roi_y: Optional[Tuple[float, float]] = None,
    max_coast_frames: int = 15,
    detector_conf_threshold: Optional[float] = None,
    **kwargs: Any,
) -> PipelineActors:
    """Instantiates all 9 pipeline actors as local in-process Python instances."""
    ingestion = StreamIngestionActor(
        backend="auto",
        target_fps=target_fps,
        enable_ray_put=False,
    )
    homography = HomographyActor(
        config_path=homography_config_path,
    )
    det_kwargs = {
        "engine_path": detector_engine_path,
        "mock_mode": detector_mock_mode,
        "generate_mock_boxes_if_empty": True,
        "input_shape": detector_input_shape or (360, 640),
        "field_bounds": field_bounds_screen,
        "field_roi_y": field_roi_y,
    }
    if detector_conf_threshold is not None:
        det_kwargs["conf_threshold"] = detector_conf_threshold
    detector = TensorRTDetectorActor(**det_kwargs)
    ocr = OCREnhancerActor(
        super_res_scale=2,
        red_teams=red_teams,
        blue_teams=blue_teams,
    )
    tracking = TrackingActor(
        target_fps=float(target_fps),
        red_teams=red_teams,
        blue_teams=blue_teams,
        max_coast_frames=max_coast_frames,
        mandatory_attribution=bool(red_teams or blue_teams),
    )
    action = ActionRecognitionActor(
        fps=float(target_fps),
    )
    reconciliation = ReconciliationActor()
    storage = DuckDBStorageActor(
        db_path=db_path,
        match_id=match_id,
        fps=float(target_fps),
    )
    publisher = MQTTTelemetryPublisherActor(
        mock_mode=mqtt_mock_mode,
        auto_prune_oversized=True,
    )

    return PipelineActors(
        ingestion=ingestion,
        homography=homography,
        detector=detector,
        ocr=ocr,
        tracking=tracking,
        action=action,
        reconciliation=reconciliation,
        storage=storage,
        publisher=publisher,
        use_ray=False,
        hardware_options=None,
    )


def create_pipeline_actors(
    use_ray: bool = True,
    stream_uri: Optional[str] = None,
    match_id: str = "default_match",
    target_fps: int = 30,
    db_path: str = ":memory:",
    detector_engine_path: Optional[str] = None,
    detector_mock_mode: bool = False,
    mqtt_mock_mode: bool = False,
    homography_config_path: Optional[str] = None,
    num_gpus: Optional[int] = None,
    config_path: Optional[str] = None,
    force_production: bool = False,
    red_teams: Optional[Sequence[int]] = None,
    blue_teams: Optional[Sequence[int]] = None,
    field_bounds_screen: Optional[Tuple[float, float, float, float]] = None,
    detector_input_shape: Optional[Tuple[int, int]] = None,
    field_roi_y: Optional[Tuple[float, float]] = None,
    max_coast_frames: int = 15,
    detector_conf_threshold: Optional[float] = None,
) -> PipelineActors:
    """Factory creating PipelineActors either via Ray remote handles or local instances."""
    if use_ray:
        return create_remote_actors(
            stream_uri=stream_uri,
            match_id=match_id,
            target_fps=target_fps,
            db_path=db_path,
            detector_engine_path=detector_engine_path,
            detector_mock_mode=detector_mock_mode,
            mqtt_mock_mode=mqtt_mock_mode,
            homography_config_path=homography_config_path,
            num_gpus=num_gpus,
            config_path=config_path,
            force_production=force_production,
            red_teams=red_teams,
            blue_teams=blue_teams,
            field_bounds_screen=field_bounds_screen,
            detector_input_shape=detector_input_shape,
            field_roi_y=field_roi_y,
            max_coast_frames=max_coast_frames,
            detector_conf_threshold=detector_conf_threshold,
        )
    return create_local_actors(
        stream_uri=stream_uri,
        match_id=match_id,
        target_fps=target_fps,
        db_path=db_path,
        detector_engine_path=detector_engine_path,
        detector_mock_mode=detector_mock_mode,
        mqtt_mock_mode=mqtt_mock_mode,
        homography_config_path=homography_config_path,
        num_gpus=num_gpus,
        config_path=config_path,
        red_teams=red_teams,
        blue_teams=blue_teams,
        field_bounds_screen=field_bounds_screen,
        detector_input_shape=detector_input_shape,
        field_roi_y=field_roi_y,
        max_coast_frames=max_coast_frames,
        detector_conf_threshold=detector_conf_threshold,
    )
