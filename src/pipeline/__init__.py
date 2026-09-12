"""
Distributed streaming pipeline package for FRC 2026 REBUILT scouting architecture.

Provides Ray Core actor wrappers and the PipelineOrchestrator coordinating
stream ingestion, vision processing, multi-object tracking, action recognition,
Bayesian ground-truth reconciliation, DuckDB OLAP storage, and low-bandwidth MQTT transport.
"""

from src.pipeline.actors import (
    PipelineActors,
    RemoteActionRecognitionActor,
    RemoteDetectorActor,
    RemoteDuckDBStorageActor,
    RemoteHomographyActor,
    RemoteOCREnhancerActor,
    RemoteReconciliationActor,
    RemoteStreamIngestionActor,
    RemoteTelemetryPublisherActor,
    RemoteTrackingActor,
    actor_has_method,
    await_result,
    call_actor,
    create_local_actors,
    create_pipeline_actors,
    create_remote_actors,
    dispatch_actor_call,
    get_available_gpu_count,
    resolve_frame,
    resolve_frame_bgr,
)
from src.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator",
    "PipelineActors",
    "RemoteStreamIngestionActor",
    "RemoteHomographyActor",
    "RemoteDetectorActor",
    "RemoteOCREnhancerActor",
    "RemoteTrackingActor",
    "RemoteActionRecognitionActor",
    "RemoteReconciliationActor",
    "RemoteDuckDBStorageActor",
    "RemoteTelemetryPublisherActor",
    "create_pipeline_actors",
    "create_remote_actors",
    "create_local_actors",
    "call_actor",
    "dispatch_actor_call",
    "await_result",
    "get_available_gpu_count",
    "resolve_frame",
    "resolve_frame_bgr",
    "actor_has_method",
]
