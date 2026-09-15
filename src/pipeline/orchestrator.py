"""
Ray Core Distributed Pipeline Orchestration for the FRC 2026 REBUILT scouting architecture.

Implements PipelineOrchestrator conforming to PipelineOrchestratorProtocol:
- Distributed or in-process streaming DAG lifecycle (start_pipeline, process_match_stream, stop_pipeline)
- WorkerWithPipelining asynchronous streaming dataflow with non-blocking await
- Concurrent dispatch of homography estimation and batched YOLOv11 + Real-ESRGAN OCR
- Kalman tracking with BoT-SORT and ReID trajectory bridging
- ActionFormer multi-scale temporal action localization and Bayesian reconciliation
- In-process DuckDB columnar trajectory storage with ASOF JOIN OLAP queries
- MQTT telemetry transport enforcing strict < 50 KB payload budget
- Backpressure regulation monitoring in-flight task queue depths
"""

from __future__ import annotations

import asyncio
import collections
import logging
import os
import tempfile
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

import ray

from src.common.interfaces import PipelineOrchestratorProtocol
from src.common.types import (
    ActionSegment,
    BallTrackletState,
    BoundingBox,
    CameraPoseState,
    DetectionResult,
    FieldAPIScoreState,
    HomographyEstimate,
    OCRResult,
    RawDecodedFrame,
    RobotTrackletState,
    TelemetrySyncPacket,
)
from src.pipeline.actors import (
    PipelineActors,
    call_actor,
    create_pipeline_actors,
    dispatch_actor_call,
    load_hardware_config,
)
from src.tracking.ball_tracker import BallTracker
from src.vision.ball_detector import YellowBallDetector
from src.vision.ocr_enhancer import extract_bumper_crop

logger = logging.getLogger("frc.pipeline.orchestrator")


class PipelineOrchestrator(PipelineOrchestratorProtocol):
    """
    Coordinates distributed Ray Core actors across cluster nodes for real-time
    FRC broadcast video ingestion, multi-object tracking, action recognition,
    Bayesian ground-truth reconciliation, DuckDB storage, and MQTT telemetry transport.
    """

    def __init__(
        self,
        use_ray: bool = True,
        num_cpus: int = 8,
        num_gpus: int = 2,
        batch_size: int = 8,
        max_in_flight: int = 4,
        backpressure_threshold: int = 16,
        buffer_depth_limit: int = 16,
        db_path: str = ":memory:",
        parquet_export_dir: Optional[str] = None,
        mqtt_broker_url: str = "loopback",
        mqtt_port: int = 1883,
        mqtt_client_id: str = "scout-orchestrator",
        event_key: str = "2026casf",
        target_fps: int = 30,
        detector_engine_path: Optional[str] = None,
        detector_mock_mode: bool = False,
        mqtt_mock_mode: bool = True,
        homography_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
        actors: Optional[PipelineActors] = None,
        max_frames: Optional[int] = None,
        red_teams: Optional[Sequence[int]] = None,
        blue_teams: Optional[Sequence[int]] = None,
        field_bounds_screen: Optional[Tuple[float, float, float, float]] = None,
        detector_input_shape: Optional[Tuple[int, int]] = None,
        field_roi_y: Optional[Tuple[float, float]] = None,
        max_coast_frames: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self.use_ray = use_ray
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.batch_size = max(1, batch_size)
        self.max_in_flight = max(1, max_in_flight)
        self.max_frames = max_frames
        self.effective_max_frames: Optional[int] = max_frames
        self.backpressure_threshold = max(1, backpressure_threshold)
        self.buffer_depth_limit = max(self.max_in_flight, buffer_depth_limit)
        self.db_path = db_path
        self.parquet_export_dir = parquet_export_dir or tempfile.gettempdir()
        self.mqtt_broker_url = mqtt_broker_url
        self.mqtt_port = mqtt_port
        self.mqtt_client_id = mqtt_client_id
        self.event_key = event_key
        self.target_fps = target_fps

        self.detector_engine_path = detector_engine_path
        self.detector_mock_mode = detector_mock_mode
        self.mqtt_mock_mode = mqtt_mock_mode
        self.homography_config_path = homography_config_path
        self.config_path = config_path

        self.red_teams: Optional[List[int]] = [int(t) for t in red_teams] if red_teams is not None else None
        self.blue_teams: Optional[List[int]] = [int(t) for t in blue_teams] if blue_teams is not None else None
        self.field_bounds_screen: Optional[Tuple[float, float, float, float]] = field_bounds_screen
        self.detector_input_shape: Optional[Tuple[int, int]] = detector_input_shape
        self.field_roi_y: Optional[Tuple[float, float]] = field_roi_y
        self.max_coast_frames: Optional[int] = max_coast_frames
        self.confidence_threshold: Optional[float] = confidence_threshold

        self.ball_tracking_enabled: bool = True
        self.max_balls: int = 504
        self.ball_min_radius_px: float = 2.5
        self.ball_max_radius_px: float = 30.0
        self.ball_distance_match_thresh: float = 25.0
        self.ball_max_coast_frames: int = 20
        self._ball_detector: Optional[YellowBallDetector] = None
        self._ball_tracker: Optional[BallTracker] = None
        self._total_balls_tracked: int = 0

        self._load_pipeline_yaml_defaults()

        self.actors: Optional[PipelineActors] = actors

        # Backpressure regulation state
        self._queued_frames: int = 0
        self._detection_queue_frames: int = 0
        self._backpressure_paused: bool = False
        self._backpressure_pause_count: int = 0
        self._backpressure_resume_count: int = 0
        self._backpressure_event: asyncio.Event = asyncio.Event()
        self._allocated_gpus: int = 0
        self._active_tasks: List[asyncio.Task] = []

        # Match execution state
        self._stream_uri: Optional[str] = None
        self._match_id: str = "default_match"
        self._is_running: bool = False
        self._start_time: Optional[float] = None
        self._total_frames_processed: int = 0
        self._total_gameplay_frames: int = 0
        self._total_trajectories_inserted: int = 0
        self._seen_event_ids: Set[str] = set()
        self._all_reconciled_actions: List[ActionSegment] = []
        self._fms_red_score: int = 0
        self._fms_blue_score: int = 0
        self._reconciled_red_score: int = 0
        self._reconciled_blue_score: int = 0
        self._published_trajectory_packets: int = 0
        self._last_sync_published: bool = False

    @property
    def is_running(self) -> bool:
        """Returns True if the orchestrator is actively processing match streams."""
        return self._is_running

    @property
    def match_id(self) -> str:
        return self._match_id

    @property
    def total_frames_processed(self) -> int:
        return self._total_frames_processed

    @property
    def total_trajectories_inserted(self) -> int:
        return self._total_trajectories_inserted

    @property
    def all_reconciled_actions(self) -> List[ActionSegment]:
        return list(self._all_reconciled_actions)

    @property
    def published_trajectory_packets(self) -> int:
        return self._published_trajectory_packets

    @property
    def queued_frames(self) -> int:
        """Total number of frames currently in-flight across DAG processing stages."""
        return self._queued_frames

    @queued_frames.setter
    def queued_frames(self, value: int) -> None:
        self._queued_frames = value

    @property
    def detection_queue_frames(self) -> int:
        """Number of frames currently awaiting or undergoing detection/vision inference."""
        return self._detection_queue_frames

    @property
    def is_backpressure_paused(self) -> bool:
        """Returns True if stream ingestion is currently paused due to backpressure."""
        return self._backpressure_paused

    @property
    def backpressure_paused(self) -> bool:
        """Alias for is_backpressure_paused."""
        return self._backpressure_paused

    @property
    def backpressure_pause_count(self) -> int:
        """Number of times stream ingestion was paused due to backpressure."""
        return self._backpressure_pause_count

    @property
    def backpressure_resume_count(self) -> int:
        """Number of times stream ingestion resumed after backpressure paused."""
        return self._backpressure_resume_count

    @property
    def allocated_gpus(self) -> int:
        """Number of GPUs allocated to the Ray Core cluster session."""
        return self._allocated_gpus

    def _is_backpressure_active(self) -> bool:
        """
        Checks if detection/in-flight queue depth meets or exceeds the backpressure threshold.
        Returns True if backpressure should throttle stream ingestion.
        """
        return (
            self._queued_frames >= self.backpressure_threshold
            or self._detection_queue_frames >= self.backpressure_threshold
        )

    def _load_pipeline_yaml_defaults(self) -> None:
        """Loads default roster, field bounds, and detector input shape from pipeline YAML config."""
        path_to_try = self.config_path or "configs/pipeline.yaml"
        if not os.path.exists(path_to_try):
            return
        try:
            import yaml
            with open(path_to_try, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            pipe = cfg.get("pipeline", {})
            roster = pipe.get("roster", {})
            detection = pipe.get("detection", {})
            tracking = pipe.get("tracking", {})

            if self.red_teams is None and "red_teams" in roster:
                self.red_teams = [int(t) for t in roster["red_teams"]]
            if self.blue_teams is None and "blue_teams" in roster:
                self.blue_teams = [int(t) for t in roster["blue_teams"]]
            if self.field_bounds_screen is None and "field_bounds_screen" in detection:
                fb = detection["field_bounds_screen"]
                if isinstance(fb, (list, tuple)) and len(fb) == 4:
                    self.field_bounds_screen = (float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3]))
            if self.detector_input_shape is None and "input_shape" in detection:
                inp = detection["input_shape"]
                if isinstance(inp, (list, tuple)) and len(inp) == 2:
                    self.detector_input_shape = (int(inp[0]), int(inp[1]))
            if self.field_roi_y is None and "field_roi_y" in detection:
                fry = detection["field_roi_y"]
                if isinstance(fry, (list, tuple)) and len(fry) == 2:
                    self.field_roi_y = (float(fry[0]), float(fry[1]))
            if self.confidence_threshold is None and "confidence_threshold" in detection:
                self.confidence_threshold = float(detection["confidence_threshold"])
            if self.max_coast_frames is None and "max_coast_frames" in tracking:
                self.max_coast_frames = int(tracking["max_coast_frames"])
            if "ball_tracking" in pipe:
                bt = pipe["ball_tracking"]
                if "enabled" in bt:
                    self.ball_tracking_enabled = bool(bt["enabled"])
                if "max_balls" in bt:
                    self.max_balls = int(bt["max_balls"])
                if "min_radius_px" in bt:
                    self.ball_min_radius_px = float(bt["min_radius_px"])
                if "max_radius_px" in bt:
                    self.ball_max_radius_px = float(bt["max_radius_px"])
                if "distance_match_threshold_px" in bt:
                    self.ball_distance_match_thresh = float(bt["distance_match_threshold_px"])
                if "max_coast_frames" in bt:
                    self.ball_max_coast_frames = int(bt["max_coast_frames"])
        except Exception as exc:
            logger.debug("Could not read pipeline YAML defaults: %s", exc)

    # ========================================================================
    # PipelineOrchestratorProtocol Interface Methods
    # ========================================================================

    async def start_pipeline(self, stream_uri: str, match_id: str) -> None:
        """
        Initializes and connects all Ray actors in the streaming DAG.

        1. Initializes Ray Core cluster session with hardware check (2 GPUs or graceful 0 CPU fallback).
        2. Instantiates remote or local actor handles for the complete 9-actor topology.
        3. Initializes stream decoder on ingestion actor.
        4. Establishes persistent connection for MQTT telemetry publisher.
        """
        self._stream_uri = stream_uri
        self._match_id = match_id
        if "_" in match_id:
            self.event_key = match_id.split("_")[0]

        logger.info(
            "Starting pipeline: MatchID=%s, EventKey=%s, StreamURI=%s, Ray=%s",
            match_id,
            self.event_key,
            stream_uri,
            self.use_ray,
        )

        # 1. Initialize Ray if requested and not yet active
        target_gpus = 0
        if self.use_ray:
            import torch
            available_gpus = 0
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()

            target_gpus = min(self.num_gpus, available_gpus)

            if not ray.is_initialized():
                ray_init_kwargs = {
                    "ignore_reinit_error": True,
                    "num_cpus": self.num_cpus,
                    "num_gpus": target_gpus,
                    "include_dashboard": False,
                }
                hw_cfg = load_hardware_config(self.config_path)
                plasma_cfg = hw_cfg.get("hardware_pinning", {}).get("plasma_store", {})
                plasma_dir = plasma_cfg.get("mount_path", "/dev/shm")
                obj_store_mem = int(plasma_cfg.get("memory_bytes", 68719476736))

                if os.path.isdir(plasma_dir) and os.access(plasma_dir, os.W_OK):
                    ray_init_kwargs["_plasma_directory"] = plasma_dir
                    try:
                        st = os.statvfs(plasma_dir)
                        avail_shm = st.f_bavail * st.f_frsize
                        if obj_store_mem > avail_shm * 0.9:
                            obj_store_mem = max(int(avail_shm * 0.7), 100 * 1024 * 1024)
                    except Exception:
                        pass
                    ray_init_kwargs["object_store_memory"] = obj_store_mem

                ray.init(**ray_init_kwargs)
                logger.info("Ray Core initialized with %d GPUs", target_gpus)
        self._allocated_gpus = target_gpus

        # 2. Instantiate pipeline actors if not supplied
        if self.actors is None:
            self.actors = create_pipeline_actors(
                use_ray=self.use_ray,
                stream_uri=stream_uri,
                match_id=match_id,
                target_fps=self.target_fps,
                db_path=self.db_path,
                detector_engine_path=self.detector_engine_path,
                detector_mock_mode=self.detector_mock_mode,
                mqtt_mock_mode=self.mqtt_mock_mode,
                homography_config_path=self.homography_config_path,
                num_gpus=target_gpus,
                config_path=self.config_path,
                red_teams=self.red_teams,
                blue_teams=self.blue_teams,
                field_bounds_screen=self.field_bounds_screen,
                detector_input_shape=self.detector_input_shape,
                field_roi_y=self.field_roi_y,
                max_coast_frames=self.max_coast_frames if self.max_coast_frames is not None else 15,
                detector_conf_threshold=self.confidence_threshold,
            )
            logger.info("Created 9 pipeline actors (use_ray=%s, num_gpus=%d)", self.use_ray, target_gpus)

        # Apply field bounds, field ROI, & roster to actors
        if self.field_bounds_screen is not None:
            try:
                await call_actor(self.actors.detector, "set_field_bounds", self.field_bounds_screen)
            except Exception as exc:
                logger.debug("Could not set_field_bounds on detector actor: %s", exc)
        if self.field_roi_y is not None:
            try:
                await call_actor(self.actors.detector, "set_field_roi_y", self.field_roi_y)
            except Exception as exc:
                logger.debug("Could not set_field_roi_y on detector actor: %s", exc)
        if self.red_teams is not None or self.blue_teams is not None:
            try:
                await call_actor(self.actors.ocr, "set_roster", self.red_teams, self.blue_teams)
            except Exception as exc:
                logger.debug("Could not set_roster on ocr actor: %s", exc)
            try:
                await call_actor(self.actors.tracking, "set_roster", self.red_teams, self.blue_teams)
            except Exception as exc:
                logger.debug("Could not set_roster on tracking actor: %s", exc)

        # 3. Connect MQTT publisher
        connected = await call_actor(
            self.actors.publisher,
            "connect",
            self.mqtt_broker_url,
            self.mqtt_port,
            f"{self.mqtt_client_id}-{match_id}",
        )
        if not connected:
            logger.warning(
                "MQTT publisher connect returned False for %s:%d (proceeding in mock/loopback mode)",
                self.mqtt_broker_url,
                self.mqtt_port,
            )

        # 4. Initialize stream ingestion
        stream_ready = await call_actor(
            self.actors.ingestion,
            "initialize_stream",
            stream_uri,
            self.target_fps,
        )
        if not stream_ready:
            logger.error("StreamIngestionActor failed to initialize stream: %s", stream_uri)
            raise RuntimeError(f"Failed to initialize stream ingestion for URI: {stream_uri}")

        # Determine effective_max_frames bounded by video total frames
        total_frames = None
        try:
            total_frames = await call_actor(self.actors.ingestion, "get_total_frames")
        except Exception as exc:
            logger.debug("Could not query total_frames from ingestion actor: %s", exc)

        if total_frames is not None and total_frames > 0:
            if self.max_frames is not None:
                self.effective_max_frames = min(self.max_frames, total_frames)
            else:
                self.effective_max_frames = total_frames
        else:
            self.effective_max_frames = self.max_frames

        logger.info(
            "Effective max frames determined: %s (max_frames=%s, total_frames=%s)",
            self.effective_max_frames,
            self.max_frames,
            total_frames,
        )

        # Reset counters & flags
        self._is_running = True
        self._start_time = time.perf_counter()
        self._total_frames_processed = 0
        self._total_frames_ingested = 0
        self._total_gameplay_frames = 0
        self._total_trajectories_inserted = 0
        self._seen_event_ids.clear()
        self._all_reconciled_actions.clear()
        self._published_trajectory_packets = 0
        self._last_sync_published = False

        self._queued_frames = 0
        self._detection_queue_frames = 0
        self._backpressure_paused = False
        self._backpressure_pause_count = 0
        self._backpressure_resume_count = 0
        self._backpressure_event = asyncio.Event()

        if self.ball_tracking_enabled:
            self._ball_detector = YellowBallDetector(
                max_balls=self.max_balls,
                min_radius_px=self.ball_min_radius_px,
                max_radius_px=self.ball_max_radius_px,
                field_roi_y=self.field_roi_y or (0.12, 0.65),
                field_bounds_screen=self.field_bounds_screen or (0.02, 0.98, 0.13, 0.62),
            )
            self._ball_tracker = BallTracker(
                max_balls=self.max_balls,
                distance_match_thresh_px=self.ball_distance_match_thresh,
                max_coast_frames=self.ball_max_coast_frames,
                target_fps=float(self.target_fps),
            )

        logger.info("Pipeline started successfully for match %s", match_id)

    async def process_match_stream(self) -> None:
        """
        Executes streaming pipelining loop with WorkerWithPipelining design pattern
        and 16-frame backpressure regulation across the coordinated DAG:
        StreamIngestionActor -> (HomographyActor + DetectionActor) -> TrackingActor
        -> ActionRecognitionActor -> ReconciliationActor -> (StorageActor + TelemetryPublisherActor)
        """
        if not self._is_running or self.actors is None:
            raise RuntimeError("Pipeline must be started before process_match_stream() is called.")
        actors = self.actors

        logger.info(
            "Entering streaming dataflow loop (batch_size=%d, backpressure_threshold=%d, use_ray=%s)",
            self.batch_size,
            self.backpressure_threshold,
            self.use_ray,
        )

        # Asynchronous queues connecting adjacent DAG stages
        ingestion_to_vision: asyncio.Queue[
            Optional[Union[List[RawDecodedFrame], Tuple[List[RawDecodedFrame], Optional[List[Any]]]]]
        ] = asyncio.Queue()
        vision_to_tracking: asyncio.Queue[
            Optional[
                Tuple[
                    List[RawDecodedFrame],
                    List[HomographyEstimate],
                    List[DetectionResult],
                    List[List[OCRResult]],
                    Optional[List[Any]],
                ]
            ]
        ] = asyncio.Queue()
        tracking_to_action: asyncio.Queue[
            Optional[Tuple[List[RawDecodedFrame], List[List[RobotTrackletState]]]]
        ] = asyncio.Queue()
        action_to_reconciliation: asyncio.Queue[
            Optional[
                Tuple[
                    List[RawDecodedFrame],
                    List[List[RobotTrackletState]],
                    List[List[ActionSegment]],
                    List[Dict[int, Tuple[float, float]]],
                ]
            ]
        ] = asyncio.Queue()
        reconciliation_to_sink: asyncio.Queue[
            Optional[
                Tuple[
                    List[RawDecodedFrame],
                    List[List[RobotTrackletState]],
                    List[List[ActionSegment]],
                ]
            ]
        ] = asyncio.Queue()

        # Stage 1: Stream Ingestion Worker with 16-frame backpressure regulation
        async def stage_ingestion() -> None:
            consecutive_empty = 0
            max_empty_attempts = 5

            try:
                while self._is_running:
                    # Backpressure Regulation: If in-flight queue depth meets/exceeds threshold,
                    # pause stream ingestion until downstream stages process frames.
                    while self._is_running and self._is_backpressure_active():
                        if not self._backpressure_paused:
                            self._backpressure_paused = True
                            self._backpressure_pause_count += 1
                            self._backpressure_event.clear()
                            logger.info(
                                "Backpressure regulation: %d queued frames (detection queue: %d, threshold: %d). "
                                "Stream ingestion paused.",
                                self._queued_frames,
                                self._detection_queue_frames,
                                self.backpressure_threshold,
                            )
                        try:
                            await asyncio.wait_for(self._backpressure_event.wait(), timeout=0.05)
                        except asyncio.TimeoutError:
                            pass

                    if not self._is_running:
                        break

                    # Retrieve next gameplay batch from StreamIngestionActor
                    batch: List[RawDecodedFrame] = []
                    frame_refs: Optional[List[Any]] = None

                    if hasattr(actors.ingestion, "get_next_batch_with_refs"):
                        batch, frame_refs = await call_actor(
                            actors.ingestion,
                            "get_next_batch_with_refs",
                            self.batch_size,
                        )
                    else:
                        batch = await call_actor(
                            actors.ingestion,
                            "get_next_gameplay_batch",
                            self.batch_size,
                        )

                    effective_max_frames = getattr(self, "effective_max_frames", self.max_frames)
                    if effective_max_frames is not None:
                        remaining = effective_max_frames - self._total_frames_ingested
                        if remaining <= 0:
                            break
                        if len(batch) > remaining:
                            batch = batch[:remaining]
                            if frame_refs is not None:
                                frame_refs = frame_refs[:remaining]

                    if batch:
                        consecutive_empty = 0
                        self._queued_frames += len(batch)
                        self._detection_queue_frames += len(batch)
                        self._total_frames_ingested += len(batch)

                        # Shared memory optimization: if running under Ray, ensure frames are
                        # registered in Plasma shared memory store (/dev/shm) once.
                        if self.use_ray and ray.is_initialized():
                            if not frame_refs or len(frame_refs) != len(batch):
                                frame_refs = [
                                    f if isinstance(f, ray.ObjectRef) else ray.put(f)
                                    for f in batch
                                ]
                        else:
                            frame_refs = None

                        await ingestion_to_vision.put((batch, frame_refs))

                        # If in-flight queue reached/exceeded threshold, engage backpressure pause immediately
                        if self._is_backpressure_active() and not self._backpressure_paused:
                            self._backpressure_paused = True
                            self._backpressure_pause_count += 1
                            self._backpressure_event.clear()
                            logger.info(
                                "Backpressure regulation: %d queued frames (detection queue: %d, threshold: %d). "
                                "Stream ingestion paused.",
                                self._queued_frames,
                                self._detection_queue_frames,
                                self.backpressure_threshold,
                            )

                        if effective_max_frames is not None and self._total_frames_ingested >= effective_max_frames:
                            logger.info(
                                "Reached effective_max_frames limit (%d frames ingested). Terminating stream ingestion.",
                                self._total_frames_ingested,
                            )
                            break
                    else:
                        consecutive_empty += 1
                        is_active = await call_actor(actors.ingestion, "is_stream_active")
                        if not is_active or consecutive_empty >= max_empty_attempts:
                            logger.info("Video stream ingestion completed / EOF reached.")
                            break
                        await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                pass
            finally:
                await ingestion_to_vision.put(None)

        # Stage 2: Concurrent Batched Vision Inference (Homography + Detection/OCR)
        async def stage_vision() -> None:
            try:
                while self._is_running:
                    batch_item = await ingestion_to_vision.get()
                    if batch_item is None:
                        await vision_to_tracking.put(None)
                        break

                    if isinstance(batch_item, tuple):
                        batch, frame_refs = batch_item
                    else:
                        batch = batch_item
                        frame_refs = (
                            [f if isinstance(f, ray.ObjectRef) else ray.put(f) for f in batch]
                            if (self.use_ray and ray.is_initialized())
                            else None
                        )

                    # Concurrently dispatch Homography and Detection+OCR sub-pipelines
                    estimates, det_results, per_frame_ocr = await self._process_vision_batch(
                        batch, frame_refs=frame_refs
                    )
                    self._detection_queue_frames = max(0, self._detection_queue_frames - len(batch))
                    await vision_to_tracking.put((batch, estimates, det_results, per_frame_ocr, frame_refs))
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Exception in stage_vision: %s", exc)
                raise

        # Stage 3: Trajectory Tracking (Kalman updates + CMC + ReID bridging)
        async def stage_tracking() -> None:
            try:
                while self._is_running:
                    item = await vision_to_tracking.get()
                    if item is None:
                        await tracking_to_action.put(None)
                        break

                    if len(item) == 5:
                        batch, estimates, det_results, per_frame_ocr, frame_refs = item
                    else:
                        batch, estimates, det_results, per_frame_ocr = item
                        frame_refs = None

                    target_frames = frame_refs if (frame_refs is not None and self.use_ray) else batch

                    # Batched tracking call: 1 RPC instead of 16-24 individual RPCs
                    if hasattr(actors.tracking, "update_tracks_batch"):
                        frame_indices = [f.metadata.frame_index for f in batch]
                        detections_batch = [det.boxes for det in det_results]
                        batch_active_tracks: List[List[RobotTrackletState]] = await call_actor(
                            actors.tracking,
                            "update_tracks_batch",
                            frame_indices,
                            detections_batch,
                            per_frame_ocr,
                            estimates,
                            target_frames,
                        )
                    else:
                        batch_active_tracks = []
                        for i, frame in enumerate(batch):
                            h_est = estimates[i]
                            det = det_results[i]
                            ocr_res = per_frame_ocr[i]
                            f_item = target_frames[i] if target_frames is not None else frame.image_bgr

                            if h_est.camera_cut:
                                await call_actor(
                                    actors.tracking, "bridge_lost_tracklets", camera_cut=True
                                )

                            active_tracks: List[RobotTrackletState] = await call_actor(
                                actors.tracking,
                                "update_tracks",
                                frame.metadata.frame_index,
                                det.boxes,
                                ocr_res,
                                h_est,
                                f_item,
                            )

                            if not h_est.camera_cut:
                                await call_actor(
                                    actors.tracking, "bridge_lost_tracklets", camera_cut=False
                                )

                            batch_active_tracks.append(active_tracks)

                    # Ball Tracking: detect and track up to 504 yellow balls concurrently
                    batch_ball_tracks: List[List[BallTrackletState]] = []
                    if self.ball_tracking_enabled and self._ball_detector and self._ball_tracker:
                        for i, frame in enumerate(batch):
                            h_est = estimates[i] if i < len(estimates) else None
                            r_tracks = batch_active_tracks[i] if i < len(batch_active_tracks) else []
                            r_boxes = [t.pixel_bbox for t in r_tracks if t.pixel_bbox]
                            dual_res = self._ball_detector.detect_dual_zone(frame.image_bgr, robot_boxes=r_boxes)
                            b_states = self._ball_tracker.update(
                                detections=dual_res.individual_balls,
                                frame_index=frame.metadata.frame_index,
                                homography=h_est,
                                robot_boxes=r_boxes,
                                clusters=dual_res.clusters,
                            )
                            batch_ball_tracks.append(b_states)
                    else:
                        batch_ball_tracks = [[] for _ in batch]

                    # Camera Pose Extraction
                    batch_camera_poses: List[CameraPoseState] = []
                    for i, frame in enumerate(batch):
                        h_est = estimates[i] if i < len(estimates) else None
                        if h_est is not None:
                            c_center = tuple(float(x) for x in h_est.camera_center) if h_est.camera_center is not None else (0.0, 0.0, 0.0)
                            r_vec = tuple(float(x) for x in h_est.camera_pose_rvec.ravel()) if h_est.camera_pose_rvec is not None else (0.0, 0.0, 0.0)
                            t_vec = tuple(float(x) for x in h_est.camera_pose_tvec.ravel()) if h_est.camera_pose_tvec is not None else (0.0, 0.0, 0.0)
                            pose = CameraPoseState(
                                match_id=self.match_id,
                                frame_index=frame.metadata.frame_index,
                                timestamp_ms=frame.metadata.pts_ms,
                                camera_center=c_center,
                                rvec=r_vec,
                                tvec=t_vec,
                                focal_length=float(h_est.focal_length or 1250.6),
                                reproj_rmse=float(h_est.lm_rmse or 0.0),
                                converged=bool(h_est.converged),
                                visible_tags=getattr(h_est, "visible_tags", []),
                                tag_distances=getattr(h_est, "tag_distances", {}),
                            )
                            batch_camera_poses.append(pose)

                    await tracking_to_action.put((batch, batch_active_tracks, batch_ball_tracks, batch_camera_poses))
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Exception in stage_tracking: %s", exc)
                raise

        # Stage 4: Temporal Action Localization (ActionFormer)
        async def stage_action() -> None:
            try:
                while self._is_running:
                    item = await tracking_to_action.get()
                    if item is None:
                        await action_to_reconciliation.put(None)
                        break

                    batch = item[0]
                    batch_active_tracks = item[1]
                    batch_ball_tracks = item[2] if len(item) > 2 else [[] for _ in batch]
                    batch_camera_poses = item[3] if len(item) > 3 else []
                    batch_candidates: List[List[ActionSegment]] = [[] for _ in range(len(batch))]
                    batch_positions: List[Dict[int, Tuple[float, float]]] = []

                    all_tracks: List[RobotTrackletState] = []
                    all_pts: List[int] = []
                    all_frame_indices: List[int] = []
                    unique_tids: Set[int] = set()

                    for i, frame in enumerate(batch):
                        active_tracks = batch_active_tracks[i]
                        robot_positions = {
                            t.track_id: (t.field_x_inch, t.field_y_inch) for t in active_tracks
                        }
                        batch_positions.append(robot_positions)
                        for t in active_tracks:
                            all_tracks.append(t)
                            all_pts.append(frame.metadata.pts_ms)
                            all_frame_indices.append(frame.metadata.frame_index)
                            unique_tids.add(t.track_id)

                    if all_tracks:
                        # 1. Ingest all tracklet states across the batch in a single RPC (1 RPC)
                        await call_actor(
                            actors.action,
                            "ingest_tracklet_window",
                            all_tracks,
                            128,
                            all_pts,
                            all_frame_indices,
                        )

                        # 2. Localize actions once per unique track ID across the entire batch (1 RPC)
                        actions_by_tid: Dict[int, List[ActionSegment]] = {}
                        if hasattr(actors.action, "localize_all_actions"):
                            actions_by_tid = await call_actor(
                                actors.action, "localize_all_actions", list(unique_tids)
                            )
                        elif hasattr(actors.action, "localize_batch"):
                            actions_by_tid = await call_actor(
                                actors.action, "localize_batch", list(unique_tids)
                            )
                        else:
                            for tid in unique_tids:
                                actions_by_tid[tid] = await call_actor(
                                    actors.action, "localize_actions", tid
                                )

                        new_actions: List[ActionSegment] = []
                        for tid in unique_tids:
                            for a in actions_by_tid.get(tid, []):
                                if a.event_id not in self._seen_event_ids:
                                    self._seen_event_ids.add(a.event_id)
                                    new_actions.append(a)

                        # Distribute candidate actions to matching frames in batch
                        for a in new_actions:
                            assigned = False
                            for f_idx, frame in enumerate(batch):
                                if frame.metadata.frame_index == a.end_frame:
                                    batch_candidates[f_idx].append(a)
                                    assigned = True
                                    break
                            if not assigned:
                                for f_idx, frame in enumerate(batch):
                                    if frame.metadata.pts_ms >= a.end_time_ms:
                                        batch_candidates[f_idx].append(a)
                                        assigned = True
                                        break
                            if not assigned:
                                batch_candidates[-1].append(a)

                    await action_to_reconciliation.put(
                        (batch, batch_active_tracks, batch_candidates, batch_positions, batch_ball_tracks, batch_camera_poses)
                    )
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Exception in stage_action: %s", exc)
                raise

        # Stage 5: Bayesian Ground-Truth Reconciliation
        async def stage_reconciliation() -> None:
            try:
                while self._is_running:
                    item = await action_to_reconciliation.get()
                    if item is None:
                        await reconciliation_to_sink.put(None)
                        break

                    batch = item[0]
                    batch_active_tracks = item[1]
                    batch_candidates = item[2]
                    batch_positions = item[3]
                    batch_ball_tracks = item[4] if len(item) > 4 else [[] for _ in batch]
                    batch_camera_poses = item[5] if len(item) > 5 else []
                    batch_reconciled: List[List[ActionSegment]] = [[] for _ in range(len(batch))]

                    all_candidates: List[ActionSegment] = []
                    candidate_frame_indices: List[int] = []
                    for i in range(len(batch)):
                        for cand in batch_candidates[i]:
                            all_candidates.append(cand)
                            candidate_frame_indices.append(i)

                    if all_candidates:
                        merged_positions: Dict[int, Tuple[float, float]] = {}
                        for pos_dict in batch_positions:
                            merged_positions.update(pos_dict)

                        # Reconcile all batch candidate actions in 1 RPC
                        reconciled = await call_actor(
                            actors.reconciliation,
                            "reconcile_actions",
                            all_candidates,
                            merged_positions,
                        )
                        reconciled_list = list(reconciled)
                        self._all_reconciled_actions.extend(reconciled_list)

                        rec_by_id = {r.event_id: r for r in reconciled_list}
                        for cand, f_idx in zip(all_candidates, candidate_frame_indices):
                            if cand.event_id in rec_by_id:
                                batch_reconciled[f_idx].append(rec_by_id[cand.event_id])

                    await reconciliation_to_sink.put((batch, batch_active_tracks, batch_reconciled, batch_ball_tracks, batch_camera_poses))
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Exception in stage_reconciliation: %s", exc)
                raise

        # Stage 6: Storage (DuckDB) & Live Telemetry (MQTT) Sinks (Executed Concurrently)
        async def stage_sink() -> None:
            traj_topic = f"frc/{self.event_key}/{self._match_id}/trajectory"

            try:
                while self._is_running:
                    item = await reconciliation_to_sink.get()
                    if item is None:
                        break

                    batch = item[0]
                    batch_active_tracks = item[1]
                    batch_reconciled = item[2]
                    batch_ball_tracks = item[3] if len(item) > 3 else [[] for _ in batch]
                    batch_camera_poses = item[4] if len(item) > 4 else []

                    # Storage writer task
                    async def _write_storage() -> None:
                        all_reconciled = [r for frame_recs in batch_reconciled for r in frame_recs]
                        frame_trajectories = [
                            (batch_active_tracks[i], batch[i].metadata.frame_index)
                            for i in range(len(batch))
                            if batch_active_tracks[i]
                        ]
                        traj_count = sum(len(t) for t, _ in frame_trajectories)

                        if hasattr(actors.storage, "insert_multi_frame_batch"):
                            await call_actor(
                                actors.storage,
                                "insert_multi_frame_batch",
                                frame_trajectories,
                                all_reconciled if all_reconciled else None,
                            )
                            self._total_trajectories_inserted += traj_count
                        else:
                            if all_reconciled:
                                await call_actor(
                                    actors.storage, "insert_action_events", all_reconciled
                                )
                            for active_tracks, frame_index in frame_trajectories:
                                await call_actor(
                                    actors.storage,
                                    "insert_trajectory_batch",
                                    active_tracks,
                                    frame_index,
                                )
                                self._total_trajectories_inserted += len(active_tracks)

                        # Write ball trajectories if present
                        if hasattr(actors.storage, "insert_ball_trajectories") and batch_ball_tracks:
                            for i, frame in enumerate(batch):
                                b_states = batch_ball_tracks[i] if i < len(batch_ball_tracks) else []
                                if b_states:
                                    await call_actor(
                                        actors.storage,
                                        "insert_ball_trajectories",
                                        b_states,
                                        frame.metadata.frame_index,
                                    )

                        # Write camera poses if present
                        if hasattr(actors.storage, "insert_camera_poses") and batch_camera_poses:
                            await call_actor(
                                actors.storage,
                                "insert_camera_poses",
                                batch_camera_poses,
                            )

                    # Telemetry publisher task
                    async def _publish_telemetry() -> None:
                        payloads = []
                        for i, frame in enumerate(batch):
                            active_tracks = batch_active_tracks[i]
                            payloads.append({
                                "frame_index": frame.metadata.frame_index,
                                "match_time_ms": frame.metadata.pts_ms,
                                "match_remaining_seconds": max(
                                    0.0, 150.0 - (frame.metadata.pts_ms / 1000.0)
                                ),
                                "match_stage": "TELEOP",
                                "robots": active_tracks,
                            })

                        if hasattr(actors.publisher, "publish_live_trajectories_batch"):
                            pub_count = await call_actor(
                                actors.publisher,
                                "publish_live_trajectories_batch",
                                traj_topic,
                                payloads,
                                0,
                            )
                            self._published_trajectory_packets += pub_count
                        else:
                            for p in payloads:
                                pub_ok = await call_actor(
                                    actors.publisher,
                                    "publish_live_trajectory",
                                    traj_topic,
                                    p,
                                    0,
                                )
                                if pub_ok:
                                    self._published_trajectory_packets += 1

                    # Concurrently execute Storage and Telemetry Publishing
                    await asyncio.gather(_write_storage(), _publish_telemetry())

                    for frame in batch:
                        self._total_frames_processed += 1
                        if frame.metadata.is_gameplay:
                            self._total_gameplay_frames += 1

                    # Batch processing complete: decrement in-flight queued frames
                    self._queued_frames = max(0, self._queued_frames - len(batch))

                    # If queue dropped below 16 frames, resume stream ingestion
                    if not self._is_backpressure_active():
                        if self._backpressure_paused:
                            self._backpressure_paused = False
                            self._backpressure_resume_count += 1
                            logger.info(
                                "Backpressure released: %d queued frames (< %d threshold). "
                                "Stream ingestion resumed.",
                                self._queued_frames,
                                self.backpressure_threshold,
                            )
                        self._backpressure_event.set()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Exception in stage_sink: %s", exc)
                raise

        # Launch concurrent DAG workers
        self._active_tasks = [
            asyncio.create_task(stage_ingestion(), name="dag-ingestion"),
            asyncio.create_task(stage_vision(), name="dag-vision"),
            asyncio.create_task(stage_tracking(), name="dag-tracking"),
            asyncio.create_task(stage_action(), name="dag-action"),
            asyncio.create_task(stage_reconciliation(), name="dag-reconciliation"),
            asyncio.create_task(stage_sink(), name="dag-sink"),
        ]

        try:
            await asyncio.gather(*self._active_tasks)
        except Exception as dag_err:
            logger.error("Error in DAG pipeline execution: %s", dag_err)
            for t in self._active_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            raise
        finally:
            self._active_tasks.clear()
            self._backpressure_event.set()

        # Final sweep: query ActionFormer for any actions completed across full history
        await self._final_action_sweep()

        logger.info(
            "process_match_stream completed: %d total frames, %d trajectories, %d actions reconciled",
            self._total_frames_processed,
            self._total_trajectories_inserted,
            len(self._all_reconciled_actions),
        )

    async def stop_pipeline(self) -> Dict[str, Any]:
        """
        Tears down stream, flushes DuckDB storage to Parquet, constructs final
        DuckDBSyncPayload, publishes via MQTT (QoS 1, verified < 50 KB budget),
        disconnects publisher, and returns match execution summary dictionary.
        """
        self._is_running = False
        self._backpressure_event.set()
        for t in getattr(self, "_active_tasks", []):
            if not t.done():
                t.cancel()
        elapsed = (time.perf_counter() - self._start_time) if self._start_time else 0.0

        logger.info("Stopping pipeline for match %s (elapsed=%.2fs)", self._match_id, elapsed)

        # 1. Close stream ingestion
        if self.actors and self.actors.ingestion:
            try:
                await call_actor(self.actors.ingestion, "close_stream")
            except Exception as exc:
                logger.warning("Exception during close_stream: %s", exc)

        # 2. Flush storage to Parquet & retrieve OLAP analytics
        parquet_path = ""
        actions_parquet_path = ""
        summary_metrics: List[Dict[str, Any]] = []
        cycle_analytics: Dict[str, Any] = {}
        velocity_degradation: Dict[str, Any] = {}

        if self.actors and self.actors.storage:
            try:
                out_file = os.path.join(
                    self.parquet_export_dir,
                    f"match_{self._match_id}_trajectories.parquet",
                )
                parquet_path = await call_actor(
                    self.actors.storage,
                    "export_match_parquet",
                    out_file,
                    "robot_trajectories",
                )
            except Exception as exc:
                logger.warning("Exception exporting parquet: %s", exc)

            try:
                actions_out_file = os.path.join(
                    self.parquet_export_dir,
                    f"match_{self._match_id}_actions.parquet",
                )
                actions_parquet_path = await call_actor(
                    self.actors.storage,
                    "export_match_parquet",
                    actions_out_file,
                    "action_events",
                )
            except Exception as exc:
                logger.warning("Exception exporting actions parquet: %s", exc)

            try:
                balls_out_file = os.path.join(
                    self.parquet_export_dir,
                    f"match_{self._match_id}_balls.parquet",
                )
                balls_parquet_path = await call_actor(
                    self.actors.storage,
                    "export_match_parquet",
                    balls_out_file,
                    "ball_trajectories",
                )
            except Exception as exc:
                logger.debug("Exception exporting balls parquet: %s", exc)

            try:
                camera_out_file = os.path.join(
                    self.parquet_export_dir,
                    f"match_{self._match_id}_camera_poses.parquet",
                )
                camera_parquet_path = await call_actor(
                    self.actors.storage,
                    "export_match_parquet",
                    camera_out_file,
                    "camera_poses",
                )
            except Exception as exc:
                logger.debug("Exception exporting camera_poses parquet: %s", exc)

            try:
                summary_metrics = await call_actor(
                    self.actors.storage,
                    "get_match_summary_metrics",
                    self._match_id,
                )
                cycle_analytics = await call_actor(
                    self.actors.storage,
                    "get_cycle_analytics",
                    self._match_id,
                )
                velocity_degradation = await call_actor(
                    self.actors.storage,
                    "get_velocity_degradation",
                    self._match_id,
                )
            except Exception as exc:
                logger.warning("Exception querying DuckDB analytics: %s", exc)

        # 3. Construct final DuckDBSyncPayload
        robot_aggregates = []
        for m in summary_metrics:
            robot_aggregates.append({
                "team_number": int(m.get("team_number", 0)),
                "alliance": str(m.get("alliance", "UNKNOWN")),
                "total_fuel_scored": int(m.get("total_fuel_scored", 0)),
                "total_cycles": int(m.get("total_cycles", 0)),
                "avg_cycle_time_seconds": float(m.get("avg_cycle_time_seconds", 0.0)),
                "min_cycle_time_seconds": float(m.get("min_cycle_time_seconds", 0.0)),
                "avg_neutral_zone_transit_time_s": float(m.get("avg_neutral_zone_transit_time_s", 0.0)),
                "total_distance_traveled_inches": float(m.get("total_distance_traveled_inches", 0.0)),
                "max_velocity_inches_s": float(m.get("max_velocity_inches_s", 0.0)),
                "avg_velocity_inches_s": float(m.get("avg_velocity_inches_s", 0.0)),
                "time_under_defensive_pressure_s": float(m.get("time_under_defensive_pressure_s", 0.0)),
                "tower_climb_success": bool(m.get("tower_climb_success", False)),
                "tower_climb_duration_s": float(m.get("tower_climb_duration_s", 0.0)),
            })

        sync_payload = {
            "event_key": self.event_key,
            "match_id": self._match_id,
            "sequence_number": 1,
            "generated_at_epoch_ms": int(time.time() * 1000),
            "payload_type": "PAYLOAD_DELTA_SUMMARY",
            "robot_aggregates": robot_aggregates,
            "zone_transits": [],
            "verified_scoring_events": self._all_reconciled_actions,
            "parquet_table_name": "robot_trajectories",
            "frc_api_red_score": self._fms_red_score,
            "frc_api_blue_score": self._fms_blue_score,
            "vision_reconciled_red_score": self._reconciled_red_score,
            "vision_reconciled_blue_score": self._reconciled_blue_score,
            "fully_reconciled": True,
        }

        # 4. Publish DuckDBSyncPayload via MQTT (QoS 1, verified < 50 KB budget)
        sync_published = False
        sync_payload_bytes = 0
        if self.actors and self.actors.publisher:
            sync_topic = f"frc/{self.event_key}/{self._match_id}/sync"
            try:
                sync_published = await call_actor(
                    self.actors.publisher,
                    "publish_duckdb_sync",
                    sync_topic,
                    sync_payload,
                    1,
                )
                self._last_sync_published = bool(sync_published)
                logger.info("DuckDB sync payload published at QoS 1: %s", sync_published)
            except Exception as exc:
                logger.error("Failed to publish DuckDB sync payload: %s", exc)

            try:
                packets = await call_actor(self.actors.publisher, "get_published_packets")
                if packets:
                    sync_pkts = [p for p in packets if p.topic.endswith("/sync")]
                    if sync_pkts:
                        sync_payload_bytes = sync_pkts[-1].payload_size_bytes
                    else:
                        sync_payload_bytes = packets[-1].payload_size_bytes
            except Exception as exc:
                logger.warning("Could not retrieve published packets from publisher: %s", exc)

            if sync_payload_bytes == 0:
                try:
                    from src.transport.mqtt_publisher import serialize_payload, to_protobuf_duckdb_sync_payload
                    sync_pb = to_protobuf_duckdb_sync_payload(sync_payload)
                    sync_payload_bytes = len(serialize_payload(sync_pb, wrap_envelope=True))
                except Exception:
                    pass

            try:
                await call_actor(self.actors.publisher, "disconnect")
            except Exception as exc:
                logger.warning("Exception during publisher disconnect: %s", exc)

        fps = (self._total_frames_processed / elapsed) if elapsed > 0 else 0.0

        summary: Dict[str, Any] = {
            "match_id": self._match_id,
            "event_key": self.event_key,
            "status": "completed",
            "total_frames_processed": self._total_frames_processed,
            "total_gameplay_frames": self._total_gameplay_frames,
            "total_trajectories_inserted": self._total_trajectories_inserted,
            "total_actions_reconciled": len(self._all_reconciled_actions),
            "total_reconciled_actions": len(self._all_reconciled_actions),
            "total_balls_active": self._ball_tracker.active_track_count if self._ball_tracker else 0,
            "parquet_path": parquet_path,
            "actions_parquet_path": actions_parquet_path,
            "balls_parquet_path": locals().get("balls_parquet_path", ""),
            "elapsed_seconds": elapsed,
            "processing_fps": fps,
            "summary_metrics": summary_metrics,
            "cycle_analytics": cycle_analytics,
            "velocity_degradation": velocity_degradation,
            "sync_published": sync_published,
            "sync_payload_bytes": sync_payload_bytes,
            "published_trajectory_packets": self._published_trajectory_packets,
        }

        logger.info("Pipeline stop completed: %s", summary)
        return summary

    # ========================================================================
    # Concurrent Pipelining Internal Helpers
    # ========================================================================

    async def _process_vision_batch(
        self,
        frames: List[RawDecodedFrame],
        frame_refs: Optional[List[Any]] = None,
    ) -> Tuple[List[HomographyEstimate], List[DetectionResult], List[List[OCRResult]]]:
        """
        Concurrently dispatches:
          a) Dynamic homography estimation (AprilTag detection + DLT + LM + CMC affine)
          b) YOLOv11 robot detection & bumper crop OCR
        """
        assert self.actors is not None

        # Concurrently dispatch Path A (Homography) and Path B (Detection + OCR)
        homography_task = asyncio.create_task(
            self._process_homography_subpipeline(frames, frame_refs=frame_refs)
        )
        detection_ocr_task = asyncio.create_task(
            self._process_detection_ocr_subpipeline(frames, frame_refs=frame_refs)
        )

        estimates, (det_results, per_frame_ocr) = await asyncio.gather(
            homography_task,
            detection_ocr_task,
        )
        return estimates, det_results, per_frame_ocr

    async def _process_homography_subpipeline(
        self,
        frames: List[RawDecodedFrame],
        frame_refs: Optional[List[Any]] = None,
    ) -> List[HomographyEstimate]:
        """Processes dynamic homography across frames in temporal order."""
        assert self.actors is not None
        target_frames = frame_refs if (frame_refs is not None and self.use_ray) else frames

        # Batched homography estimation (1 RPC)
        if hasattr(self.actors.homography, "estimate_batch"):
            estimates: List[HomographyEstimate] = await call_actor(
                self.actors.homography, "estimate_batch", target_frames
            )
            return estimates

        # Graceful fallback: per-frame calls
        estimates = []
        for f in target_frames:
            tags = await call_actor(self.actors.homography, "detect_apriltags", f)
            est = await call_actor(self.actors.homography, "estimate_homography", f, tags)
            estimates.append(est)
        return estimates

    async def _process_detection_ocr_subpipeline(
        self,
        frames: List[RawDecodedFrame],
        frame_refs: Optional[List[Any]] = None,
    ) -> Tuple[List[DetectionResult], List[List[OCRResult]]]:
        """Executes batched robot detection and batched super-resolution bumper OCR."""
        assert self.actors is not None
        target_frames = frame_refs if (frame_refs is not None and self.use_ray) else frames

        # 1. Batched detection (1 RPC)
        det_results: List[DetectionResult] = await call_actor(
            self.actors.detector, "detect_batch", target_frames
        )

        # 2. Extract bumper crops from local frame image buffers (zero-copy in orchestrator)
        all_crops: List[Any] = []
        box_counts: List[int] = []

        for frame, det in zip(frames, det_results):
            frame_crops = []
            img_bgr = frame.image_bgr if hasattr(frame, "image_bgr") else None
            if det.boxes and img_bgr is not None:
                for b in det.boxes:
                    crop = extract_bumper_crop(img_bgr, b)
                    frame_crops.append(crop)
            all_crops.extend(frame_crops)
            box_counts.append(len(frame_crops))

        # 3. Batched OCR (1 RPC)
        all_ocr_results: List[OCRResult] = []
        if all_crops:
            all_ocr_results = await call_actor(
                self.actors.ocr, "batch_enhance_bumpers", all_crops
            )

        # 4. Partition OCR results per frame
        per_frame_ocr: List[List[OCRResult]] = []
        idx = 0
        for count in box_counts:
            per_frame_ocr.append(all_ocr_results[idx : idx + count])
            idx += count

        return det_results, per_frame_ocr

    async def _process_downstream_batch(
        self,
        batch: List[RawDecodedFrame],
        estimates: List[HomographyEstimate],
        det_results: List[DetectionResult],
        per_frame_ocr: List[List[OCRResult]],
        frame_refs: Optional[List[Any]] = None,
    ) -> None:
        """
        Executes chronological tracking, action localization, Bayesian reconciliation,
        DuckDB ingestion, and live trajectory MQTT publishing for a processed batch.
        """
        assert self.actors is not None
        traj_topic = f"frc/{self.event_key}/{self._match_id}/trajectory"
        target_frames = frame_refs if (frame_refs is not None and self.use_ray) else batch

        # 1. Batched tracking (1 RPC)
        if hasattr(self.actors.tracking, "update_tracks_batch"):
            frame_indices = [f.metadata.frame_index for f in batch]
            detections_batch = [det.boxes for det in det_results]
            batch_active_tracks: List[List[RobotTrackletState]] = await call_actor(
                self.actors.tracking,
                "update_tracks_batch",
                frame_indices,
                detections_batch,
                per_frame_ocr,
                estimates,
                target_frames,
            )
        else:
            batch_active_tracks = []
            for i, frame in enumerate(batch):
                h_est = estimates[i]
                det = det_results[i]
                ocr_res = per_frame_ocr[i]
                f_item = target_frames[i] if target_frames is not None else frame.image_bgr

                if h_est.camera_cut:
                    await call_actor(
                        self.actors.tracking, "bridge_lost_tracklets", camera_cut=True
                    )
                active_tracks = await call_actor(
                    self.actors.tracking,
                    "update_tracks",
                    frame.metadata.frame_index,
                    det.boxes,
                    ocr_res,
                    h_est,
                    f_item,
                )
                if not h_est.camera_cut:
                    await call_actor(
                        self.actors.tracking, "bridge_lost_tracklets", camera_cut=False
                    )
                batch_active_tracks.append(active_tracks)

        # 2. Batched Action Recognition Ingestion & Localization (2 RPCs)
        all_tracks: List[RobotTrackletState] = []
        all_pts: List[int] = []
        all_frame_indices: List[int] = []
        unique_tids: Set[int] = set()

        for i, frame in enumerate(batch):
            for t in batch_active_tracks[i]:
                all_tracks.append(t)
                all_pts.append(frame.metadata.pts_ms)
                all_frame_indices.append(frame.metadata.frame_index)
                unique_tids.add(t.track_id)

        new_candidates: List[ActionSegment] = []
        if all_tracks:
            await call_actor(
                self.actors.action,
                "ingest_tracklet_window",
                all_tracks,
                128,
                all_pts,
                all_frame_indices,
            )

            actions_by_tid: Dict[int, List[ActionSegment]] = {}
            if hasattr(self.actors.action, "localize_all_actions"):
                actions_by_tid = await call_actor(
                    self.actors.action, "localize_all_actions", list(unique_tids)
                )
            elif hasattr(self.actors.action, "localize_batch"):
                actions_by_tid = await call_actor(
                    self.actors.action, "localize_batch", list(unique_tids)
                )
            else:
                for tid in unique_tids:
                    actions_by_tid[tid] = await call_actor(
                        self.actors.action, "localize_actions", tid
                    )

            for tid in unique_tids:
                for a in actions_by_tid.get(tid, []):
                    if a.event_id not in self._seen_event_ids:
                        self._seen_event_ids.add(a.event_id)
                        new_candidates.append(a)

        # 3. Batched Bayesian Reconciliation (0 or 1 RPC)
        new_reconciled: List[ActionSegment] = []
        if new_candidates:
            merged_positions: Dict[int, Tuple[float, float]] = {}
            for tracks in batch_active_tracks:
                for t in tracks:
                    merged_positions[t.track_id] = (t.field_x_inch, t.field_y_inch)

            reconciled = await call_actor(
                self.actors.reconciliation,
                "reconcile_actions",
                new_candidates,
                merged_positions,
            )
            new_reconciled = list(reconciled)
            self._all_reconciled_actions.extend(new_reconciled)

        # 4. Batched DuckDB Storage Writer (1 RPC)
        frame_trajectories = [
            (batch_active_tracks[i], batch[i].metadata.frame_index)
            for i in range(len(batch))
            if batch_active_tracks[i]
        ]
        traj_count = sum(len(t) for t, _ in frame_trajectories)

        if hasattr(self.actors.storage, "insert_multi_frame_batch"):
            await call_actor(
                self.actors.storage,
                "insert_multi_frame_batch",
                frame_trajectories,
                new_reconciled if new_reconciled else None,
            )
            self._total_trajectories_inserted += traj_count
        else:
            if new_reconciled:
                await call_actor(
                    self.actors.storage, "insert_action_events", new_reconciled
                )
            for active_tracks, frame_index in frame_trajectories:
                await call_actor(
                    self.actors.storage,
                    "insert_trajectory_batch",
                    active_tracks,
                    frame_index,
                )
                self._total_trajectories_inserted += len(active_tracks)

        # Store camera poses if available
        if hasattr(self.actors.storage, "insert_camera_poses") and estimates:
            poses = []
            for i, frame in enumerate(batch):
                h_est = estimates[i] if i < len(estimates) else None
                if h_est is not None:
                    c_center = tuple(float(x) for x in h_est.camera_center) if h_est.camera_center is not None else (0.0, 0.0, 0.0)
                    r_vec = tuple(float(x) for x in h_est.camera_pose_rvec.ravel()) if h_est.camera_pose_rvec is not None else (0.0, 0.0, 0.0)
                    t_vec = tuple(float(x) for x in h_est.camera_pose_tvec.ravel()) if h_est.camera_pose_tvec is not None else (0.0, 0.0, 0.0)
                    poses.append(CameraPoseState(
                        match_id=self.match_id,
                        frame_index=frame.metadata.frame_index,
                        timestamp_ms=frame.metadata.pts_ms,
                        camera_center=c_center,
                        rvec=r_vec,
                        tvec=t_vec,
                        focal_length=float(h_est.focal_length or 1250.6),
                        reproj_rmse=float(h_est.lm_rmse or 0.0),
                        converged=bool(h_est.converged),
                        visible_tags=getattr(h_est, "visible_tags", []),
                        tag_distances=getattr(h_est, "tag_distances", {}),
                    ))
            if poses:
                await call_actor(self.actors.storage, "insert_camera_poses", poses)

        # 5. Batched Live Telemetry Publishing (1 RPC)
        payloads = [
            {
                "frame_index": frame.metadata.frame_index,
                "match_time_ms": frame.metadata.pts_ms,
                "match_remaining_seconds": max(0.0, 150.0 - (frame.metadata.pts_ms / 1000.0)),
                "match_stage": "TELEOP",
                "robots": batch_active_tracks[i],
            }
            for i, frame in enumerate(batch)
        ]
        if hasattr(self.actors.publisher, "publish_live_trajectories_batch"):
            pub_count = await call_actor(
                self.actors.publisher,
                "publish_live_trajectories_batch",
                traj_topic,
                payloads,
                0,
            )
            self._published_trajectory_packets += pub_count
        else:
            for p in payloads:
                pub_ok = await call_actor(
                    self.actors.publisher,
                    "publish_live_trajectory",
                    traj_topic,
                    p,
                    0,
                )
                if pub_ok:
                    self._published_trajectory_packets += 1

        for frame in batch:
            self._total_frames_processed += 1
            if frame.metadata.is_gameplay:
                self._total_gameplay_frames += 1

    async def _final_action_sweep(self) -> None:
        """Executes a final localization sweep across all active tracks when stream ends."""
        if not self.actors or not self.actors.tracking or not self.actors.action:
            return

        active_tracks: List[RobotTrackletState] = await call_actor(
            self.actors.tracking, "get_active_tracks"
        )
        if not active_tracks:
            return

        robot_positions = {t.track_id: (t.field_x_inch, t.field_y_inch) for t in active_tracks}
        new_candidates: List[ActionSegment] = []
        unique_tids = list({track.track_id for track in active_tracks})

        if hasattr(self.actors.action, "localize_all_actions"):
            actions_by_tid = await call_actor(
                self.actors.action, "localize_all_actions", unique_tids
            )
            for tid, actions in actions_by_tid.items():
                for a in actions:
                    if a.event_id not in self._seen_event_ids:
                        self._seen_event_ids.add(a.event_id)
                        new_candidates.append(a)
        elif hasattr(self.actors.action, "localize_batch"):
            actions_by_tid = await call_actor(
                self.actors.action, "localize_batch", unique_tids
            )
            for tid, actions in actions_by_tid.items():
                for a in actions:
                    if a.event_id not in self._seen_event_ids:
                        self._seen_event_ids.add(a.event_id)
                        new_candidates.append(a)
        else:
            for track in active_tracks:
                actions: List[ActionSegment] = await call_actor(
                    self.actors.action, "localize_actions", track.track_id
                )
                for a in actions:
                    if a.event_id not in self._seen_event_ids:
                        self._seen_event_ids.add(a.event_id)
                        new_candidates.append(a)

        if new_candidates and self.actors.reconciliation and self.actors.storage:
            reconciled = await call_actor(
                self.actors.reconciliation,
                "reconcile_actions",
                new_candidates,
                robot_positions,
            )
            self._all_reconciled_actions.extend(reconciled)
            await call_actor(self.actors.storage, "insert_action_events", reconciled)

    async def execute_asof_join_query(
        self,
        match_id: Optional[str] = None,
        team_number: int = 254,
    ) -> Dict[str, Any]:
        """Convenience query executing vectorized ASOF JOIN directly on storage actor."""
        target_match = match_id or self._match_id
        if self.actors and self.actors.storage:
            res = await call_actor(
                self.actors.storage,
                "execute_asof_join_query",
                target_match,
                team_number,
            )
            if isinstance(res, dict):
                return res
        return {
            "match_id": target_match,
            "team_number": team_number,
            "event_count": 0,
            "events": [],
            "query_latency_ms": 0.0,
        }
