"""
High-Performance Stream Ingestion Actor with PyAV and OpenCV Backends.

Implements WorkerWithPipelining streaming pattern for the FRC 2026 REBUILT scouting pipeline.
Decodes RTMP/HLS broadcast video streams, executes ResNet-18 frame gating every K=10 frames,
drops crowd/referee/graphic segments, and yields batches of RawDecodedFrame without blocking
the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import cv2
import numpy as np
import yaml

try:
    import ray
except ImportError:
    ray = None  # type: ignore[assignment]

# Optional PyAV import
try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False
    av = None  # type: ignore[assignment]

from src.common.interfaces import StreamIngestionActorBase
from src.common.types import FrameMetadata, RawDecodedFrame
from src.ingestion.frame_gate import FrameGate, FrameGateClassifier

logger = logging.getLogger(__name__)


class _StreamSource:
    """Internal abstraction over video capture backend (OpenCV or PyAV)."""

    def is_opened(self) -> bool:
        raise NotImplementedError

    def get_fps(self, fallback_fps: float = 60.0) -> float:
        raise NotImplementedError

    def get_dimensions(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get_total_frames(self) -> Optional[int]:
        """Returns total frame count of the video stream/file if known."""
        raise NotImplementedError

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Returns (success, frame_bgr, pts_ms)."""
        raise NotImplementedError

    def grab_only(self) -> bool:
        """Grabs next packet/frame without decompressing full image buffer."""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError


class _OpenCVStreamSource(_StreamSource):
    """OpenCV VideoCapture backend wrapping FFmpeg."""

    def __init__(self, stream_uri: str) -> None:
        self._cap = cv2.VideoCapture(stream_uri)
        # Suggest low buffer size to reduce latency on live streams
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        except Exception:
            pass

    def is_opened(self) -> bool:
        return bool(self._cap.isOpened())

    def get_fps(self, fallback_fps: float = 60.0) -> float:
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            return fallback_fps
        return fps

    def get_dimensions(self) -> Tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def get_total_frames(self) -> Optional[int]:
        try:
            count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if count is not None and not np.isnan(count) and count > 0:
                return int(count)
        except Exception:
            pass
        return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None, 0
        pts = int(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        return True, frame, pts

    def grab_only(self) -> bool:
        return bool(self._cap.grab())

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception as exc:
            logger.debug("Error releasing cv2.VideoCapture: %s", exc)


class _PyAVStreamSource(_StreamSource):
    """PyAV container demuxer and decoder backend."""

    def __init__(self, stream_uri: str) -> None:
        if not PYAV_AVAILABLE:
            raise RuntimeError("PyAV is not installed")
        self._container = av.open(stream_uri, mode="r")
        self._stream = self._container.streams.video[0]
        self._frame_generator = self._iter_frames()
        self._closed = False

    def is_opened(self) -> bool:
        return not self._closed and self._container is not None

    def get_fps(self, fallback_fps: float = 60.0) -> float:
        rate = self._stream.average_rate
        if rate is not None and float(rate) > 0:
            return float(rate)
        return fallback_fps

    def get_dimensions(self) -> Tuple[int, int]:
        return int(self._stream.width), int(self._stream.height)

    def get_total_frames(self) -> Optional[int]:
        try:
            if hasattr(self, "_stream") and self._stream is not None:
                frames = getattr(self._stream, "frames", 0)
                if frames and int(frames) > 0:
                    return int(frames)
        except Exception:
            pass
        return None

    def _iter_frames(self) -> Any:
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                yield frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        if self._closed:
            return False, None, 0
        try:
            av_frame = next(self._frame_generator)
            img_bgr = av_frame.to_ndarray(format="bgr24")
            pts_ms = int(av_frame.time * 1000) if av_frame.time is not None else 0
            return True, img_bgr, pts_ms
        except (StopIteration, av.AVError):
            return False, None, 0

    def grab_only(self) -> bool:
        if self._closed:
            return False
        try:
            # Grab without calling expensive to_ndarray conversion
            _ = next(self._frame_generator)
            return True
        except (StopIteration, av.AVError):
            return False

    def release(self) -> None:
        self._closed = True
        try:
            self._container.close()
        except Exception as exc:
            logger.debug("Error closing PyAV container: %s", exc)


def _open_stream_source(stream_uri: str, backend: str) -> Optional[_StreamSource]:
    """Factory creating the appropriate video stream source backend."""
    if backend in ("pyav", "auto") and PYAV_AVAILABLE:
        try:
            pyav_source: Optional[_StreamSource] = _PyAVStreamSource(stream_uri)
            if pyav_source is not None and pyav_source.is_opened():
                logger.info("Successfully opened stream with PyAV backend: %s", stream_uri)
                return pyav_source
        except Exception as exc:
            if backend == "pyav":
                logger.error("Failed to open stream with PyAV: %s", exc)
                return None
            logger.warning("PyAV failed (%s); falling back to OpenCV VideoCapture", exc)

    # OpenCV fallback
    try:
        cv_source: Optional[_StreamSource] = _OpenCVStreamSource(stream_uri)
        if cv_source is not None and cv_source.is_opened():
            logger.info("Successfully opened stream with OpenCV backend: %s", stream_uri)
            return cv_source
        if cv_source is not None:
            cv_source.release()
    except Exception as exc:
        logger.error("Failed to open stream with OpenCV: %s", exc)

    return None


def _worker_decode_loop(
    source: _StreamSource,
    frame_gate: FrameGate,
    queue: asyncio.Queue[Optional[RawDecodedFrame]],
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    target_fps: float,
    metrics: Dict[str, Any],
    total_frames: Optional[int] = None,
) -> None:
    """Dedicated background OS thread decoding and filtering frames."""
    frame_index = 0
    try:
        while not stop_event.is_set():
            if total_frames is not None and frame_index >= total_frames:
                break
            is_eval = frame_gate.is_evaluation_frame(frame_index)

            if is_eval:
                ret, frame_bgr, pts_ms = source.read_frame()
                if not ret or frame_bgr is None:
                    break

                if pts_ms <= 0:
                    pts_ms = int(frame_index * (1000.0 / target_fps))

                is_gameplay, conf = frame_gate.evaluate(frame_index, frame_bgr)
                metrics["total_frames_read"] += 1

                if is_gameplay:
                    meta = FrameMetadata(
                        frame_index=frame_index,
                        pts_ms=pts_ms,
                        monotonic_ts_ns=time.monotonic_ns(),
                        is_gameplay=True,
                        confidence=conf,
                        width=int(frame_bgr.shape[1]),
                        height=int(frame_bgr.shape[0]),
                    )
                    raw_frame = RawDecodedFrame(metadata=meta, image_bgr=frame_bgr)
                    if not _enqueue_item(queue, loop, raw_frame, stop_event):
                        break
                    metrics["gameplay_yielded"] += 1
            else:
                # Intervening frame
                if frame_gate.last_decision:
                    ret, frame_bgr, pts_ms = source.read_frame()
                    if not ret or frame_bgr is None:
                        break

                    if pts_ms <= 0:
                        pts_ms = int(frame_index * (1000.0 / target_fps))

                    metrics["total_frames_read"] += 1
                    meta = FrameMetadata(
                        frame_index=frame_index,
                        pts_ms=pts_ms,
                        monotonic_ts_ns=time.monotonic_ns(),
                        is_gameplay=True,
                        confidence=frame_gate.last_confidence,
                        width=int(frame_bgr.shape[1]),
                        height=int(frame_bgr.shape[0]),
                    )
                    raw_frame = RawDecodedFrame(metadata=meta, image_bgr=frame_bgr)
                    if not _enqueue_item(queue, loop, raw_frame, stop_event):
                        break
                    metrics["gameplay_yielded"] += 1
                else:
                    # Non-gameplay intervening frame: drop immediately without full decode
                    ret = source.grab_only()
                    if not ret:
                        break
                    frame_gate.record_intervening_drop()
                    metrics["total_frames_read"] += 1

            frame_index += 1
    finally:
        source.release()
        metrics["eof_reached"] = True
        metrics["active"] = False
        # Sentinel None signals end-of-stream to async consumers
        _enqueue_item(queue, loop, None, stop_event)


def _enqueue_item(
    queue: asyncio.Queue[Optional[RawDecodedFrame]],
    loop: asyncio.AbstractEventLoop,
    item: Optional[RawDecodedFrame],
    stop_event: threading.Event,
) -> bool:
    """Thread-safe enqueue to asyncio.Queue respecting backpressure and stop_event."""
    if loop.is_closed():
        return False
    try:
        fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    except Exception:
        return False

    while not stop_event.is_set():
        if loop.is_closed():
            fut.cancel()
            return False
        try:
            fut.result(timeout=0.1)
            return True
        except TimeoutError:
            continue
        except Exception:
            return False

    fut.cancel()
    return False


def _load_config_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    """Loads configuration defaults from yaml if available."""
    default_cfg: Dict[str, Any] = {
        "frame_gating_stride": 10,
        "queue_depth": 32,
        "target_fps": 60,
    }

    path_to_try = config_path or "configs/pipeline.yaml"
    cfg_file = Path(path_to_try)
    if cfg_file.is_file():
        try:
            with open(cfg_file, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            stream_cfg = data.get("pipeline", {}).get("stream", {})
            pipe_cfg = data.get("pipeline", {}).get("pipelining", {})
            if "frame_gating_stride" in stream_cfg:
                default_cfg["frame_gating_stride"] = int(stream_cfg["frame_gating_stride"])
            if "target_fps" in stream_cfg:
                default_cfg["target_fps"] = int(stream_cfg["target_fps"])
            if "queue_depth" in pipe_cfg:
                default_cfg["queue_depth"] = int(pipe_cfg["queue_depth"])
        except Exception as exc:
            logger.warning("Could not parse config from %s: %s", path_to_try, exc)

    return default_cfg


class StreamIngestionActor(StreamIngestionActorBase):
    """
    Asynchronous video stream ingestion actor implementing WorkerWithPipelining.

    Decodes 1080p60 broadcast video streams in a dedicated worker thread using
    PyAV or OpenCV FFmpeg, applies strided ResNet-18 frame gating, drops non-gameplay
    segments, and provides non-blocking pipelined batch retrieval.
    """

    def __init__(
        self,
        frame_gating_stride: Optional[int] = None,
        queue_depth: Optional[int] = None,
        classifier: Optional[
            Union[FrameGateClassifier, Callable[[np.ndarray], Tuple[bool, float]]]
        ] = None,
        backend: str = "auto",
        enable_ray_put: bool = True,
        target_fps: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> None:
        cfg = _load_config_defaults(config_path)

        self.frame_gating_stride = (
            frame_gating_stride
            if frame_gating_stride is not None
            else cfg["frame_gating_stride"]
        )
        self.queue_depth = (
            queue_depth if queue_depth is not None else cfg["queue_depth"]
        )
        self.target_fps = (
            target_fps if target_fps is not None else cfg["target_fps"]
        )
        self.backend = backend
        self.enable_ray_put = enable_ray_put

        self.frame_gate = FrameGate(
            stride=self.frame_gating_stride, classifier=classifier
        )

        self._queue: Optional[asyncio.Queue[Optional[RawDecodedFrame]]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._source: Optional[_StreamSource] = None
        self._is_active: bool = False
        self._stream_uri: Optional[str] = None
        self._last_ray_refs: List[Any] = []

        self._metrics: Dict[str, Any] = {
            "total_frames_read": 0,
            "gameplay_yielded": 0,
            "eof_reached": False,
            "active": False,
        }
        self.total_video_frames: Optional[int] = None
        self._frames_yielded_count: int = 0

    async def initialize_stream(self, stream_uri: str, target_fps: int = 60) -> bool:
        """
        Connects to RTMP/HLS broadcast stream or video file and initializes decoder queues.
        Ensures stream opening and probing does not block the asyncio event loop.
        """
        if self._is_active:
            await self.close_stream()

        self.target_fps = target_fps
        self._stream_uri = stream_uri
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.queue_depth)
        self._stop_event = threading.Event()
        self.frame_gate.reset()
        self._last_ray_refs = []
        self._frames_yielded_count = 0

        self._metrics = {
            "total_frames_read": 0,
            "gameplay_yielded": 0,
            "eof_reached": False,
            "active": False,
        }

        # Open stream source in thread to prevent blocking event loop on network I/O
        source = await asyncio.to_thread(_open_stream_source, stream_uri, self.backend)
        if source is None or not source.is_opened():
            logger.error("Failed to initialize stream source for URI: %s", stream_uri)
            return False

        self._source = source
        self.total_video_frames = source.get_total_frames()
        self._is_active = True
        self._metrics["active"] = True

        actual_fps = source.get_fps(float(target_fps))

        self._worker_thread = threading.Thread(
            target=_worker_decode_loop,
            args=(
                self._source,
                self.frame_gate,
                self._queue,
                self._loop,
                self._stop_event,
                actual_fps,
                self._metrics,
                self.total_video_frames,
            ),
            name="StreamIngestionWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info(
            "StreamIngestionActor initialized: URI=%s, FPS=%.1f, Stride=%d, QueueDepth=%d",
            stream_uri,
            actual_fps,
            self.frame_gating_stride,
            self.queue_depth,
        )
        return True

    async def get_total_frames(self) -> Optional[int]:
        """Returns total frame count of the source stream or video file, if known."""
        return self.total_video_frames

    async def get_next_gameplay_batch(
        self, batch_size: int = 8
    ) -> List[RawDecodedFrame]:
        """
        Pipelined retrieval of next batch of active gameplay frames, dropping crowd/graphics.
        Non-blocking to the asyncio event loop.
        """
        if self._queue is None:
            return []

        if self.total_video_frames is not None and self._frames_yielded_count >= self.total_video_frames:
            return []

        effective_batch_size = batch_size
        if self.total_video_frames is not None:
            remaining = self.total_video_frames - self._frames_yielded_count
            if remaining <= 0:
                return []
            effective_batch_size = min(batch_size, remaining)

        batch: List[RawDecodedFrame] = []

        while len(batch) < effective_batch_size:
            if self._queue.empty():
                if not self._is_active or self._metrics.get("eof_reached", False):
                    break

                timeout = 0.05 if len(batch) > 0 else 5.0
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
            else:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            if item is None:
                # Put sentinel back so subsequent calls also receive EOF
                try:
                    self._queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                break

            if self.total_video_frames is not None and item.metadata.frame_index >= self.total_video_frames:
                break

            batch.append(item)

        if self.total_video_frames is not None:
            remaining = self.total_video_frames - self._frames_yielded_count
            if len(batch) > remaining:
                batch = batch[:max(0, remaining)]
            self._frames_yielded_count += len(batch)
        else:
            self._frames_yielded_count += len(batch)

        # Register frames into Ray's Plasma object store if Ray is initialized
        if ray is not None and ray.is_initialized() and self.enable_ray_put:
            self._last_ray_refs = [ray.put(f) for f in batch]
        else:
            self._last_ray_refs = []

        return batch

    async def is_stream_active(self) -> bool:
        """Returns True if the broadcast stream is currently active and receiving packets."""
        return self._is_active and not self._metrics.get("eof_reached", False)

    async def close_stream(self) -> None:
        """Gracefully closes network stream and flushes FFmpeg buffers."""
        self._is_active = False
        self._metrics["active"] = False

        if self._stop_event is not None:
            self._stop_event.set()

        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass

        if self._worker_thread is not None and self._worker_thread.is_alive():
            await asyncio.to_thread(self._worker_thread.join, 2.0)
            self._worker_thread = None

        if self._source is not None:
            await asyncio.to_thread(self._source.release)
            self._source = None

        self._queue = None
        self._frames_yielded_count = 0
        logger.info("StreamIngestionActor cleanly closed stream: %s", self._stream_uri)

    async def stream_gameplay_batches(
        self, batch_size: int = 8
    ) -> AsyncIterator[List[RawDecodedFrame]]:
        """Pipelined asynchronous generator streaming batches until EOF or stream closure."""
        while True:
            batch = await self.get_next_gameplay_batch(batch_size=batch_size)
            if not batch:
                break
            yield batch

    @property
    def last_ray_refs(self) -> List[Any]:
        """Returns Ray object references for the most recently retrieved batch."""
        return self._last_ray_refs

    @property
    def stats(self) -> Dict[str, Any]:
        """Returns ingestion and frame gating diagnostic statistics."""
        return {
            "stream_uri": self._stream_uri,
            "total_video_frames": self.total_video_frames,
            "total_frames_read": self._metrics.get("total_frames_read", 0),
            "gameplay_frames_yielded": self._metrics.get("gameplay_yielded", 0),
            "frames_dropped": self.frame_gate.frames_dropped,
            "evaluations_count": self.frame_gate.total_evaluations,
            "is_active": self._is_active and not self._metrics.get("eof_reached", False),
            "eof_reached": self._metrics.get("eof_reached", False),
        }
