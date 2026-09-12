"""
Multi-Scale Temporal Feature Pyramid & Kinematic Feature Extractor for ActionFormer on GPU 1.

Implements:
- 1D Temporal Feature Extractor:
  - Extracts per-frame kinematic and spatial feature vectors from RobotTrackletState:
    normalized field coordinates (x/L, y/W), normalized velocities (vx/vmax, vy/vmax),
    speed, acceleration, bounding box dimensions and aspect ratio, distance to key
    field elements (Hubs, Towers, Depots, Trenches, Bumps), heading cosine/sine,
    and neutral zone indicator.
- Sliding window buffer:
  - Maintains sliding window kinematic and spatial feature buffers as PyTorch CUDA tensors
    on GPU 1 (cuda:1) with fallback to cuda:0 or cpu.
- Multi-Scale Temporal Feature Pyramid:
  - Generates multi-scale 1D feature representations across levels 0, 1, 2, 3
    with temporal strides [1, 2, 4, 8] via PyTorch 1D strided depthwise operations
    and torch.nn.functional.avg_pool1d on CUDA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.types import (
    AllianceColor,
    BoundingBox,
    FieldElement,
    RobotTrackletState,
    TrackState,
)

# ============================================================================
# Device Resolution Helper (Target GPU 1 / cuda:1 with automatic fallbacks)
# ============================================================================

def get_torch_device(preferred_device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Resolves the target PyTorch compute device.

    Priority:
      1. Explicit preferred_device if valid.
      2. cuda:1 (NVIDIA RTX 3090 secondary accelerator).
      3. cuda:0 (Primary accelerator fallback).
      4. cpu (Deterministic CPU fallback when CUDA is unavailable).
    """
    if preferred_device is not None:
        if isinstance(preferred_device, str):
            preferred_device = torch.device(preferred_device)
        if preferred_device.type == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            if preferred_device.index is not None and preferred_device.index >= torch.cuda.device_count():
                return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
        return preferred_device

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        elif torch.cuda.device_count() > 0:
            return torch.device("cuda:0")
    return torch.device("cpu")


# ============================================================================
# Field Geometry & Kinematic Constants (FRC 2026 REBUILT)
# ============================================================================

FIELD_LENGTH_INCHES: float = 651.2
FIELD_WIDTH_INCHES: float = 317.7
FIELD_DIAG_INCHES: float = math.sqrt(FIELD_LENGTH_INCHES**2 + FIELD_WIDTH_INCHES**2)  # ~724.62

# Key Field Element Centers (inches)
RED_HUB_POS: Tuple[float, float] = (162.8, 158.85)
BLUE_HUB_POS: Tuple[float, float] = (488.4, 158.85)
RED_TOWER_POS: Tuple[float, float] = (90.0, 158.85)
BLUE_TOWER_POS: Tuple[float, float] = (561.2, 158.85)

DEPOT_NORTH_POS: Tuple[float, float] = (325.6, 280.0)
DEPOT_SOUTH_POS: Tuple[float, float] = (325.6, 37.7)

TRENCH_NORTH_POS: Tuple[float, float] = (270.0, 262.7)
TRENCH_SOUTH_POS: Tuple[float, float] = (270.0, 15.0)

BUMPS_POS: List[Tuple[float, float]] = [
    (280.0, 80.0),
    (280.0, 237.7),
    (371.2, 80.0),
    (371.2, 237.7),
]

NEUTRAL_ZONE_X_MIN: float = 250.0
NEUTRAL_ZONE_X_MAX: float = 400.0

RED_OUTPOST_POS: Tuple[float, float] = (0.0, 158.85)
BLUE_OUTPOST_POS: Tuple[float, float] = (651.2, 158.85)

# Kinematic Normalization Constants
MAX_VELOCITY_IPS: float = 240.0       # Max robot speed (20 ft/s)
MAX_ACCELERATION_IPS2: float = 480.0  # Max robot acceleration (40 ft/s^2)
CANONICAL_IMG_WIDTH: float = 1920.0
CANONICAL_IMG_HEIGHT: float = 1080.0

FEATURE_DIM: int = 20


# ============================================================================
# 1. 1D Temporal Feature Extractor
# ============================================================================

def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def extract_frame_features(
    state: RobotTrackletState,
    prev_state: Optional[RobotTrackletState] = None,
    dt: float = 1.0 / 60.0,
) -> np.ndarray:
    """
    Extracts a 20-dimensional kinematic and spatial feature vector from a single RobotTrackletState.

    Feature Vector Layout:
      0: norm_x (x / FIELD_LENGTH_INCHES)
      1: norm_y (y / FIELD_WIDTH_INCHES)
      2: norm_vx (vx / MAX_VELOCITY_IPS)
      3: norm_vy (vy / MAX_VELOCITY_IPS)
      4: norm_speed (speed / MAX_VELOCITY_IPS)
      5: norm_accel (acceleration / MAX_ACCELERATION_IPS2)
      6: norm_bbox_w (bbox.width / CANONICAL_IMG_WIDTH)
      7: norm_bbox_h (bbox.height / CANONICAL_IMG_HEIGHT)
      8: bbox_aspect_ratio (bbox.width / bbox.height)
      9: dist_red_hub / FIELD_DIAG_INCHES
     10: dist_blue_hub / FIELD_DIAG_INCHES
     11: dist_red_tower / FIELD_DIAG_INCHES
     12: dist_blue_tower / FIELD_DIAG_INCHES
     13: dist_nearest_depot / FIELD_DIAG_INCHES
     14: dist_nearest_trench / FIELD_DIAG_INCHES
     15: dist_nearest_bump / FIELD_DIAG_INCHES
     16: heading_cos (cos(heading_rad))
     17: heading_sin (sin(heading_rad))
     18: neutral_zone_indicator (1.0 if inside [250, 400] else 0.0)
     19: track_state_val (1.0 if TRACKED, 0.5 if BRIDGED, 0.0 if LOST)
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    # 1. Normalized Field Position
    pos_x = state.field_x_inch
    pos_y = state.field_y_inch
    features[0] = float(np.clip(pos_x / FIELD_LENGTH_INCHES, 0.0, 1.0))
    features[1] = float(np.clip(pos_y / FIELD_WIDTH_INCHES, 0.0, 1.0))

    # 2. Normalized Velocities
    vx = state.velocity_x_ips
    vy = state.velocity_y_ips
    features[2] = float(np.clip(vx / MAX_VELOCITY_IPS, -1.0, 1.0))
    features[3] = float(np.clip(vy / MAX_VELOCITY_IPS, -1.0, 1.0))

    # 3. Speed
    speed = math.hypot(vx, vy)
    features[4] = float(np.clip(speed / MAX_VELOCITY_IPS, 0.0, 1.0))

    # 4. Acceleration
    if prev_state is not None and dt > 1e-6:
        prev_speed = math.hypot(prev_state.velocity_x_ips, prev_state.velocity_y_ips)
        accel = (speed - prev_speed) / dt
        features[5] = float(np.clip(accel / MAX_ACCELERATION_IPS2, -1.0, 1.0))
    else:
        features[5] = 0.0

    # 5. Bounding Box Dimensions
    bbox = state.pixel_bbox
    if bbox is not None:
        features[6] = float(np.clip(bbox.width / CANONICAL_IMG_WIDTH, 0.0, 1.0))
        features[7] = float(np.clip(bbox.height / CANONICAL_IMG_HEIGHT, 0.0, 1.0))
        h = max(bbox.height, 1.0)
        features[8] = float(np.clip(bbox.width / h, 0.1, 10.0))
    else:
        features[6] = 0.05
        features[7] = 0.05
        features[8] = 1.0

    # 6. Distances to Key Field Elements
    bot_pos = (pos_x, pos_y)
    features[9] = float(euclidean_dist(bot_pos, RED_HUB_POS) / FIELD_DIAG_INCHES)
    features[10] = float(euclidean_dist(bot_pos, BLUE_HUB_POS) / FIELD_DIAG_INCHES)
    features[11] = float(euclidean_dist(bot_pos, RED_TOWER_POS) / FIELD_DIAG_INCHES)
    features[12] = float(euclidean_dist(bot_pos, BLUE_TOWER_POS) / FIELD_DIAG_INCHES)

    min_depot = min(
        euclidean_dist(bot_pos, DEPOT_NORTH_POS),
        euclidean_dist(bot_pos, DEPOT_SOUTH_POS),
    )
    features[13] = float(min_depot / FIELD_DIAG_INCHES)

    min_trench = min(
        euclidean_dist(bot_pos, TRENCH_NORTH_POS),
        euclidean_dist(bot_pos, TRENCH_SOUTH_POS),
    )
    features[14] = float(min_trench / FIELD_DIAG_INCHES)

    min_bump = min(euclidean_dist(bot_pos, b) for b in BUMPS_POS)
    features[15] = float(min_bump / FIELD_DIAG_INCHES)

    # 7. Heading
    heading = state.heading_rad
    features[16] = float(math.cos(heading))
    features[17] = float(math.sin(heading))

    # 8. Neutral Zone Indicator
    if NEUTRAL_ZONE_X_MIN <= pos_x <= NEUTRAL_ZONE_X_MAX:
        features[18] = 1.0
    else:
        features[18] = 0.0

    # 9. Track State Confidence
    if state.track_state == TrackState.TRACKED:
        features[19] = 1.0
    elif state.track_state == TrackState.BRIDGED:
        features[19] = 0.5
    else:
        features[19] = 0.0

    return features


def extract_sequence_features(
    states: Sequence[RobotTrackletState],
    fps: float = 60.0,
) -> np.ndarray:
    """
    Extracts a (T, 20) feature matrix from an ordered temporal sequence of RobotTrackletStates.
    """
    t = len(states)
    if t == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    dt = 1.0 / max(fps, 1.0)
    matrix = np.zeros((t, FEATURE_DIM), dtype=np.float32)

    prev_state: Optional[RobotTrackletState] = None
    for i, state in enumerate(states):
        matrix[i] = extract_frame_features(state, prev_state=prev_state, dt=dt)
        prev_state = state

    return matrix


def extract_sequence_features_torch(
    states: Sequence[RobotTrackletState],
    fps: float = 60.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Extracts a (T, 20) PyTorch feature tensor directly on target device (default: GPU 1).
    """
    dev = get_torch_device(device)
    np_matrix = extract_sequence_features(states, fps=fps)
    return torch.as_tensor(np_matrix, dtype=torch.float32, device=dev)


# ============================================================================
# 2. Sliding Window Buffer per Robot Track with PyTorch CUDA Tensors
# ============================================================================

@dataclass(slots=True)
class TrackletFrameRecord:
    """Temporal frame entry in tracklet sliding window."""
    frame_index: int
    timestamp_ms: int
    state: RobotTrackletState


class TemporalWindowBuffer:
    """
    Maintains a sliding temporal window of track states per track_id (e.g., T=128 frames).
    Maintains sliding window kinematic and spatial feature buffers as PyTorch CUDA tensors
    on GPU 1 (cuda:1) with automatic fallback.
    """

    def __init__(
        self,
        default_window_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
        fps: float = 60.0,
    ) -> None:
        self.default_window_size = default_window_size
        self.device = get_torch_device(device)
        self.fps = fps
        self._buffers: Dict[int, Deque[TrackletFrameRecord]] = {}
        self._tensor_buffers: Dict[int, torch.Tensor] = {}

    def append(
        self,
        track_id: int,
        state: RobotTrackletState,
        frame_index: int,
        timestamp_ms: int,
        max_window: Optional[int] = None,
    ) -> None:
        """Appends a new track state and updates the PyTorch CUDA tensor buffer."""
        limit = max_window if max_window is not None else self.default_window_size
        prev_state = (
            self._buffers[track_id][-1].state
            if (track_id in self._buffers and len(self._buffers[track_id]) > 0)
            else None
        )

        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=limit)
        elif self._buffers[track_id].maxlen != limit:
            old_items = list(self._buffers[track_id])
            self._buffers[track_id] = deque(old_items, maxlen=limit)

        self._buffers[track_id].append(
            TrackletFrameRecord(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                state=state,
            )
        )

        dt = 1.0 / max(self.fps, 1.0)
        feat_vec = extract_frame_features(state, prev_state=prev_state, dt=dt)
        feat_t = torch.as_tensor(feat_vec, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 20)

        if track_id not in self._tensor_buffers or self._tensor_buffers[track_id].numel() == 0:
            self._tensor_buffers[track_id] = feat_t
        else:
            curr_buf = self._tensor_buffers[track_id]
            if curr_buf.shape[0] >= limit:
                self._tensor_buffers[track_id] = torch.cat([curr_buf[-(limit - 1):], feat_t], dim=0)
            else:
                self._tensor_buffers[track_id] = torch.cat([curr_buf, feat_t], dim=0)

    def get_window(self, track_id: int) -> List[TrackletFrameRecord]:
        """Returns the list of frame records in the sliding window for track_id."""
        if track_id not in self._buffers:
            return []
        return list(self._buffers[track_id])

    def get_states(self, track_id: int) -> List[RobotTrackletState]:
        """Returns the list of RobotTrackletStates in the window."""
        return [r.state for r in self.get_window(track_id)]

    def get_features_tensor(
        self,
        track_id: int,
        fps: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        Retrieves the (T, 20) PyTorch feature tensor on GPU 1 (or target device)
        along with parallel lists of frame_indices and timestamps_ms.
        """
        records = self.get_window(track_id)
        if not records:
            return torch.zeros((0, FEATURE_DIM), dtype=torch.float32, device=self.device), [], []

        frame_indices = [r.frame_index for r in records]
        timestamps_ms = [r.timestamp_ms for r in records]

        if track_id in self._tensor_buffers and self._tensor_buffers[track_id].shape[0] == len(records):
            tensor = self._tensor_buffers[track_id]
        else:
            states = [r.state for r in records]
            eff_fps = fps if fps is not None else self.fps
            feat_np = extract_sequence_features(states, fps=eff_fps)
            tensor = torch.as_tensor(feat_np, dtype=torch.float32, device=self.device)
            self._tensor_buffers[track_id] = tensor

        return tensor, frame_indices, timestamps_ms

    def get_features(
        self,
        track_id: int,
        fps: float = 60.0,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Extracts the (T, 20) feature matrix along with parallel lists of
        frame_indices and timestamps_ms as NumPy arrays.
        """
        tensor, frame_indices, timestamps_ms = self.get_features_tensor(track_id, fps=fps)
        return tensor.detach().cpu().numpy(), frame_indices, timestamps_ms

    def clear(self, track_id: Optional[int] = None) -> None:
        """Clears buffer for a specific track_id or all tracks."""
        if track_id is None:
            self._buffers.clear()
            self._tensor_buffers.clear()
        else:
            self._buffers.pop(track_id, None)
            self._tensor_buffers.pop(track_id, None)

    def track_ids(self) -> List[int]:
        """Returns all currently buffered track IDs."""
        return list(self._buffers.keys())


# ============================================================================
# 3. 1D Operators (NumPy & PyTorch Dual Compatibility)
# ============================================================================

def layer_norm_1d(
    x: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> Union[np.ndarray, torch.Tensor]:
    """Applies layer normalization along channel dimension."""
    if isinstance(x, torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def relu(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Element-wise Rectified Linear Unit."""
    if isinstance(x, torch.Tensor):
        return torch.relu(x)
    return np.maximum(0.0, x)


def conv1d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Standard 1D convolution over temporal sequence (NumPy compatibility reference).
    """
    t, c_in = x.shape
    c_out, _, k = weight.shape

    if padding > 0:
        x_pad = np.pad(x, ((padding, padding), (0, 0)), mode="constant", constant_values=0.0)
    else:
        x_pad = x

    t_pad = x_pad.shape[0]
    if t_pad < k:
        return np.zeros((1, c_out), dtype=np.float32)

    t_out = (t_pad - k) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (k, c_in))[:, 0, :, :]
    windows = windows[::stride]  # (T_out, K, C_in)

    flat_windows = windows.reshape(t_out, -1)
    flat_w = weight.transpose(0, 2, 1).reshape(c_out, -1)
    out = flat_windows @ flat_w.T + bias
    return out.astype(np.float32)


def depthwise_conv1d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Strided depthwise 1D convolution over temporal sequence (NumPy compatibility reference).
    """
    t, c = x.shape
    c_w, k = weight.shape
    if c != c_w:
        raise ValueError(f"Channel mismatch: input has {c}, weights have {c_w}")

    if padding > 0:
        x_pad = np.pad(x, ((padding, padding), (0, 0)), mode="constant", constant_values=0.0)
    else:
        x_pad = x

    t_pad = x_pad.shape[0]
    if t_pad < k:
        return np.zeros((1, c), dtype=np.float32)

    t_out = (t_pad - k) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (k, c))[:, 0, :, :]
    windows = windows[::stride]  # (T_out, K, C)

    out = np.sum(windows.transpose(0, 2, 1) * weight[None, :, :], axis=2) + bias
    return out.astype(np.float32)


# ============================================================================
# 4. PyTorch CUDA Multi-Scale Temporal Feature Pyramid
# ============================================================================

class TemporalPyramid(nn.Module):
    """
    Multi-Scale Temporal Feature Pyramid on CUDA (GPU 1).

    Constructs 1D feature maps across 4 temporal resolution levels:
    - Level 0: T (stride 1) - Fine-grained localization (fast actions: intake)
    - Level 1: T / 2 (stride 2) - Short actions
    - Level 2: T / 4 (stride 4) - Medium actions (scoring cycles)
    - Level 3: T / 8 (stride 8) - Coarse actions (prolonged events: climbing, defense)

    Uses torch.nn.functional.avg_pool1d and strided 1D depthwise operations on CUDA.
    """

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.strides = [2**i for i in range(num_levels)]  # [1, 2, 4, 8]
        self.device = get_torch_device(device)

        torch.manual_seed(seed)
        k = kernel_size
        padding = k // 2

        # 1. Level 0 Input projection: in_channels -> hidden_channels (stride 1)
        self.proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=k, padding=padding)
        std_proj = 1.0 / math.sqrt(in_channels * k)
        nn.init.uniform_(self.proj.weight, -std_proj, std_proj)
        nn.init.zeros_(self.proj.bias)
        # Calibrate projection to pass through key kinematic & spatial channels directly
        with torch.no_grad():
            for ch in range(min(in_channels, hidden_channels)):
                self.proj.weight[ch, ch, k // 2] = 1.0

        self.norm0 = nn.LayerNorm(hidden_channels)

        # 2. Downsampling operations for levels 1, 2, 3:
        # Multi-scale 1D construction with strided depthwise conv & avg_pool1d
        self.strided_convs = nn.ModuleDict()
        self.pool_convs = nn.ModuleDict()
        self.downsample_norms = nn.ModuleDict()

        std_dw = 1.0 / math.sqrt(k)
        for lvl in range(1, num_levels):
            lvl_str = str(lvl)
            # Strided depthwise 1D conv (stride 2)
            sc = nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=k,
                stride=2,
                padding=padding,
                groups=hidden_channels,
            )
            nn.init.uniform_(sc.weight, -std_dw, std_dw)
            with torch.no_grad():
                sc.weight.fill_(0.5 / k)
            nn.init.zeros_(sc.bias)
            self.strided_convs[lvl_str] = sc

            # Depthwise conv coupled with avg_pool1d
            pc = nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=k,
                stride=1,
                padding=padding,
                groups=hidden_channels,
            )
            nn.init.uniform_(pc.weight, -std_dw, std_dw)
            with torch.no_grad():
                pc.weight.fill_(0.5 / k)
            nn.init.zeros_(pc.bias)
            self.pool_convs[lvl_str] = pc

            self.downsample_norms[lvl_str] = nn.LayerNorm(hidden_channels)

        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(
        self,
        features: Union[np.ndarray, torch.Tensor],
        return_tensor: Optional[bool] = None,
    ) -> Dict[int, Union[np.ndarray, torch.Tensor]]:
        """
        Executes multi-scale temporal pyramid forward pass on CUDA.

        Args:
            features: Input 1D feature array of shape (T, FEATURE_DIM) or (B, T, FEATURE_DIM).
            return_tensor: If True, returns torch.Tensor on CUDA. If False, returns np.ndarray.
                           If None, returns torch.Tensor if features is Tensor, else np.ndarray.

        Returns:
            Dictionary mapping pyramid level (0, 1, 2, 3) to feature tensor of shape (T_lvl, C)
            or (B, T_lvl, C).
        """
        is_numpy = isinstance(features, np.ndarray)
        if return_tensor is None:
            should_return_tensor = not is_numpy
        else:
            should_return_tensor = return_tensor

        if is_numpy:
            x_t = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        else:
            x_t = features.to(self.device, dtype=torch.float32)

        is_2d = (x_t.ndim == 2)
        if is_2d:
            t_len, _ = x_t.shape
            if t_len == 0:
                empty = {
                    lvl: (
                        torch.zeros((0, self.hidden_channels), dtype=torch.float32, device=self.device)
                        if should_return_tensor
                        else np.zeros((0, self.hidden_channels), dtype=np.float32)
                    )
                    for lvl in range(self.num_levels)
                }
                return empty
            x_in = x_t.transpose(0, 1).unsqueeze(0)  # (1, D, T)
        elif x_t.ndim == 3:
            b, t_len, _ = x_t.shape
            if t_len == 0:
                empty = {
                    lvl: (
                        torch.zeros((b, 0, self.hidden_channels), dtype=torch.float32, device=self.device)
                        if should_return_tensor
                        else np.zeros((b, 0, self.hidden_channels), dtype=np.float32)
                    )
                    for lvl in range(self.num_levels)
                }
                return empty
            x_in = x_t.transpose(1, 2)  # (B, D, T)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {x_t.shape}")

        # Level 0: Input projection with stride 1
        x0 = self.proj(x_in)  # (B, C, T)
        if x0.shape[-1] > t_len:
            x0 = x0[..., :t_len]

        # LayerNorm along channel dimension: (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        x0 = self.norm0(x0.transpose(1, 2)).transpose(1, 2)
        x0 = self.relu(x0)

        # Build multi-scale temporal pyramid
        pyramid: Dict[int, torch.Tensor] = {}
        if is_2d:
            pyramid[0] = x0.squeeze(0).transpose(0, 1)  # (T, C)
        else:
            pyramid[0] = x0.transpose(1, 2)  # (B, T, C)

        curr_x = x0
        for lvl in range(1, self.num_levels):
            lvl_str = str(lvl)
            # 1. Strided depthwise 1D conv (stride 2)
            conv_strided = self.strided_convs[lvl_str](curr_x)
            # 2. Avg pool 1D downsampling (stride 2)
            conv_pooled = F.avg_pool1d(
                self.pool_convs[lvl_str](curr_x),
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )
            # Fused multi-scale representation
            next_x = conv_strided + conv_pooled
            next_x = self.downsample_norms[lvl_str](next_x.transpose(1, 2)).transpose(1, 2)
            next_x = self.relu(next_x)

            curr_x = next_x
            if is_2d:
                pyramid[lvl] = curr_x.squeeze(0).transpose(0, 1)  # (T_lvl, C)
            else:
                pyramid[lvl] = curr_x.transpose(1, 2)  # (B, T_lvl, C)

        if should_return_tensor:
            return pyramid
        return {lvl: t.detach().cpu().numpy() for lvl, t in pyramid.items()}

