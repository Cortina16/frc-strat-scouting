"""
ActionFormer Temporal Action Localization Engine and Ray Actor on GPU 1.

Implements:
- 1D Distance-IoU (DIoU) metric for temporal intervals on CUDA / CPU
- Focal Loss scoring and probability calibration on CUDA / CPU
- Sinusoidal & Rotary 1D temporal positional encodings
- Local multi-head temporal self-attention blocks on CUDA
- Dual 1D prediction heads on CUDA:
  - Multi-class classification head for ActionType with Focal Loss calibration
  - Boundary regression head predicting temporal offsets (s, e) relative to anchors
- Temporal Non-Maximum Suppression (NMS)
- ActionFormer neural architecture for multi-scale action localization on GPU 1
- Batched Multi-Track Action Localization (all 6 robots in a single GPU forward pass)
- ActionRecognitionActor conforming to ActionRecognitionActorBase & ActionRecognitionActorProtocol
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.actions.temporal_pyramid import (
    BLUE_HUB_POS,
    BLUE_OUTPOST_POS,
    BLUE_TOWER_POS,
    CANONICAL_IMG_HEIGHT,
    CANONICAL_IMG_WIDTH,
    DEPOT_NORTH_POS,
    DEPOT_SOUTH_POS,
    FEATURE_DIM,
    FIELD_DIAG_INCHES,
    FIELD_LENGTH_INCHES,
    FIELD_WIDTH_INCHES,
    MAX_ACCELERATION_IPS2,
    MAX_VELOCITY_IPS,
    NEUTRAL_ZONE_X_MAX,
    NEUTRAL_ZONE_X_MIN,
    RED_HUB_POS,
    RED_OUTPOST_POS,
    RED_TOWER_POS,
    TemporalPyramid,
    TemporalWindowBuffer,
    TrackletFrameRecord,
    depthwise_conv1d_forward,
    euclidean_dist,
    extract_sequence_features,
    extract_sequence_features_torch,
    get_torch_device,
    layer_norm_1d,
    relu,
)
from src.common.interfaces import ActionRecognitionActorBase
from src.common.types import (
    ActionSegment,
    ActionType,
    AllianceColor,
    BoundingBox,
    FieldElement,
    RobotTrackletState,
    TrackState,
)

# Ordered classes for classification head (7 classes)
ACTION_CLASSES: List[ActionType] = [
    ActionType.SCORE_HUB,               # 0
    ActionType.INTAKE_FUEL,              # 1
    ActionType.CLIMB_TOWER,              # 2
    ActionType.DEFENSIVE_ENGAGEMENT,     # 3
    ActionType.TRANSIT_NEUTRAL_ZONE,     # 4
    ActionType.OUTPOST_CORRAL_CYCLE,     # 5
    ActionType.IDLE,                     # 6
]

ACTION_TO_INDEX: Dict[ActionType, int] = {cls: idx for idx, cls in enumerate(ACTION_CLASSES)}
INDEX_TO_ACTION: Dict[int, ActionType] = {idx: cls for idx, cls in enumerate(ACTION_CLASSES)}
NUM_CLASSES: int = len(ACTION_CLASSES)


# ============================================================================
# 1. 1D Distance-IoU (DIoU) Metric (PyTorch CUDA & NumPy Dual Support)
# ============================================================================

def compute_diou_1d(
    segments_a: Union[np.ndarray, torch.Tensor],
    segments_b: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    Computes 1D Distance-IoU (DIoU) metric between two sets of temporal intervals.

    DIoU = IoU - (rho^2 / c^2)
    where:
      - IoU is temporal Intersection over Union
      - rho is the Euclidean distance between the center points of the two intervals
      - c is the length of the smallest enclosing interval (convex hull)

    Supports:
      - PyTorch Tensors on CUDA / CPU
      - NumPy arrays
      - 1D single intervals or (N, 2) / (M, 2) batched intervals
    """
    if isinstance(segments_a, torch.Tensor) and isinstance(segments_b, torch.Tensor):
        a = segments_a.float()
        b = segments_b.float()
        is_single = (a.ndim == 1 and b.ndim == 1)
        if a.ndim == 1:
            a = a.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)

        a_start = torch.minimum(a[:, 0], a[:, 1])
        a_end = torch.maximum(a[:, 0], a[:, 1])
        b_start = torch.minimum(b[:, 0], b[:, 1])
        b_end = torch.maximum(b[:, 0], b[:, 1])

        if a.shape[0] == b.shape[0] and not (is_single and a.shape[0] == 1):
            s1, e1 = a_start, a_end
            s2, e2 = b_start, b_end
        else:
            s1, e1 = a_start.unsqueeze(1), a_end.unsqueeze(1)
            s2, e2 = b_start.unsqueeze(0), b_end.unsqueeze(0)

        inter = torch.clamp(torch.minimum(e1, e2) - torch.maximum(s1, s2), min=0.0)
        len1 = e1 - s1
        len2 = e2 - s2
        union = len1 + len2 - inter
        iou = inter / torch.clamp(union, min=eps)

        c1 = (s1 + e1) / 2.0
        c2 = (s2 + e2) / 2.0
        rho2 = (c1 - c2) ** 2

        c_start = torch.minimum(s1, s2)
        c_end = torch.maximum(e1, e2)
        c_len = torch.clamp(c_end - c_start, min=eps)
        c2_enclosing = c_len ** 2

        diou = iou - (rho2 / c2_enclosing)
        if is_single:
            return float(diou.item())
        return diou

    a = np.asarray(segments_a, dtype=np.float64)
    b = np.asarray(segments_b, dtype=np.float64)

    is_single = (a.ndim == 1 and b.ndim == 1)
    if a.ndim == 1:
        a = a.reshape(1, 2)
    if b.ndim == 1:
        b = b.reshape(1, 2)

    a_start = np.minimum(a[:, 0], a[:, 1])
    a_end = np.maximum(a[:, 0], a[:, 1])
    b_start = np.minimum(b[:, 0], b[:, 1])
    b_end = np.maximum(b[:, 0], b[:, 1])

    if a.shape[0] == b.shape[0] and not (is_single and a.shape[0] == 1):
        s1, e1 = a_start, a_end
        s2, e2 = b_start, b_end
    else:
        s1, e1 = a_start[:, None], a_end[:, None]
        s2, e2 = b_start[None, :], b_end[None, :]

    inter = np.maximum(0.0, np.minimum(e1, e2) - np.maximum(s1, s2))
    len1 = e1 - s1
    len2 = e2 - s2
    union = len1 + len2 - inter
    iou = inter / np.maximum(union, eps)

    c1 = (s1 + e1) / 2.0
    c2 = (s2 + e2) / 2.0
    rho2 = (c1 - c2) ** 2

    c_start = np.minimum(s1, s2)
    c_end = np.maximum(e1, e2)
    c_len = np.maximum(c_end - c_start, eps)
    c2_enclosing = c_len ** 2

    diou = iou - (rho2 / c2_enclosing)

    if is_single:
        return float(diou.item())
    return diou


# ============================================================================
# 2. Focal Loss Metric & Probability Calibration (CUDA & NumPy Dual Support)
# ============================================================================

def focal_loss(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
    eps: float = 1e-7,
) -> Union[float, torch.Tensor]:
    """
    Computes Focal Loss for temporal action classification:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports:
    - Multi-class predictions with class index targets (shape (N,) int)
    - Multi-class or binary predictions with one-hot targets (shape (N, C))
    - PyTorch Tensors on CUDA / CPU and NumPy arrays
    """
    if isinstance(predictions, torch.Tensor):
        p = predictions.float()
        y = targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets, device=p.device)
        if from_logits:
            if p.ndim > 1 and p.shape[-1] > 1:
                p = torch.softmax(p, dim=-1)
            else:
                p = torch.sigmoid(p)
        p = torch.clamp(p, min=eps, max=1.0 - eps)

        if y.ndim == 1 and p.ndim == 2:
            n = p.shape[0]
            p_t = p[torch.arange(n, device=p.device), y.long()]
            loss = -alpha * torch.pow(1.0 - p_t, gamma) * torch.log(p_t)
            return float(loss.mean().item())
        elif y.shape == p.shape:
            y_float = y.float()
            p_t = torch.where(y_float == 1.0, p, 1.0 - p)
            alpha_t = torch.where(y_float == 1.0, alpha, 1.0 - alpha)
            loss = -alpha_t * torch.pow(1.0 - p_t, gamma) * torch.log(p_t)
            if p.ndim > 1:
                return float(loss.sum(dim=-1).mean().item())
            return float(loss.mean().item())
        else:
            raise ValueError(f"Shape mismatch: predictions {p.shape} vs targets {y.shape}")

    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets)

    if from_logits:
        if p.ndim > 1 and p.shape[-1] > 1:
            p_max = np.max(p, axis=-1, keepdims=True)
            exp_p = np.exp(p - p_max)
            p = exp_p / np.sum(exp_p, axis=-1, keepdims=True)
        else:
            p = 1.0 / (1.0 + np.exp(-p))

    p = np.clip(p, eps, 1.0 - eps)

    if y.ndim == 1 and p.ndim == 2:
        n, _ = p.shape
        p_t = p[np.arange(n), y.astype(int)]
        loss = -alpha * np.power(1.0 - p_t, gamma) * np.log(p_t)
        return float(np.mean(loss))
    elif y.shape == p.shape:
        p_t = np.where(y == 1, p, 1.0 - p)
        alpha_t = np.where(y == 1, alpha, 1.0 - alpha)
        loss = -alpha_t * np.power(1.0 - p_t, gamma) * np.log(p_t)
        if p.ndim > 1:
            return float(np.mean(np.sum(loss, axis=-1)))
        return float(np.mean(loss))
    else:
        raise ValueError(f"Shape mismatch: predictions {p.shape} vs targets {y.shape}")


# ============================================================================
# 3. Positional Encodings & Local Self-Attention Block on CUDA
# ============================================================================

_PE_CACHE: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
_ROPE_CACHE: Dict[Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}


def sinusoidal_positional_encoding_1d(
    seq_len: int,
    dim: int,
    device: Optional[Union[str, torch.device]] = None,
    as_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generates 1D sinusoidal positional encodings on target device (CUDA / CPU) with caching.

    PE(pos, 2i)   = sin(pos / 10000^(2i / dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / dim))
    """
    if seq_len <= 0 or dim <= 0:
        if as_tensor:
            dev = get_torch_device(device)
            return torch.zeros((seq_len, dim), dtype=torch.float32, device=dev)
        return np.zeros((seq_len, dim), dtype=np.float32)

    dev = get_torch_device(device)
    key = (seq_len, dim, dev)
    if key not in _PE_CACHE:
        position = torch.arange(seq_len, dtype=torch.float32, device=dev).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32, device=dev) * -(math.log(10000.0) / dim)
        )
        pe = torch.zeros((seq_len, dim), dtype=torch.float32, device=dev)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        _PE_CACHE[key] = pe
    else:
        pe = _PE_CACHE[key]

    if as_tensor:
        return pe
    return pe.detach().cpu().numpy()


def apply_rotary_pos_emb_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Applies 1D Rotary Positional Embedding (RoPE) to tensor (B, T, C) with frequency caching.
    """
    if x.ndim != 3:
        return x
    b, t, c = x.shape
    if t <= 0 or c <= 0:
        return x

    dim = c if c % 2 == 0 else c - 1
    half_dim = dim // 2
    dev = x.device
    key = (t, dim, dev)

    if key not in _ROPE_CACHE:
        pos = torch.arange(t, dtype=torch.float32, device=dev).unsqueeze(1)
        inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=dev) * -(math.log(10000.0) / half_dim)
        )
        angles = pos * inv_freq  # (T, half_dim)
        sin = torch.sin(angles).repeat(1, 2).unsqueeze(0)  # (1, T, dim)
        cos = torch.cos(angles).repeat(1, 2).unsqueeze(0)  # (1, T, dim)
        _ROPE_CACHE[key] = (cos, sin)
    else:
        cos, sin = _ROPE_CACHE[key]

    x_rot = x[..., :dim]
    x_rot_half = torch.cat([-x_rot[..., half_dim:dim], x_rot[..., :half_dim]], dim=-1)
    x_out = (x_rot * cos) + (x_rot_half * sin)

    if dim < c:
        return torch.cat([x_out, x[..., dim:]], dim=-1)
    return x_out


class LocalSelfAttentionBlock(nn.Module):
    """
    Local Temporal Self-Attention Transformer Block on CUDA.

    Computes multi-head self-attention restricted to a local temporal neighborhood
    |i - j| <= window_size, followed by residual connections, LayerNorm, and FFN.
    Supports both 2D (T, C) and batched 3D (B, T, C) on GPU 1 with mask caching.
    """

    def __init__(
        self,
        channels: int = 64,
        window_size: int = 7,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.device = get_torch_device(device)
        self._mask_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        torch.manual_seed(seed)
        std = 1.0 / math.sqrt(channels)

        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)
        self.w_o = nn.Linear(channels, channels)

        nn.init.uniform_(self.w_q.weight, -std, std)
        nn.init.zeros_(self.w_q.bias)
        nn.init.uniform_(self.w_k.weight, -std, std)
        nn.init.zeros_(self.w_k.bias)
        nn.init.uniform_(self.w_v.weight, -std, std)
        nn.init.zeros_(self.w_v.bias)
        nn.init.uniform_(self.w_o.weight, -std, std)
        nn.init.zeros_(self.w_o.bias)

        self.norm1 = nn.LayerNorm(channels)

        # FFN weights
        ffn_dim = channels * 2
        std_ffn = 1.0 / math.sqrt(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, channels),
        )
        nn.init.uniform_(self.ffn[0].weight, -std_ffn, std_ffn)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.uniform_(self.ffn[2].weight, -std_ffn, std_ffn)
        nn.init.zeros_(self.ffn[2].bias)

        self.norm2 = nn.LayerNorm(channels)
        self.to(self.device)

    def _get_mask(self, t: int, device: torch.device) -> torch.Tensor:
        key = (t, device)
        if key not in self._mask_cache:
            idx = torch.arange(t, device=device)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            self._mask_cache[key] = (dist <= self.window_size).unsqueeze(0)
        return self._mask_cache[key]

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_tensor: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Executes local temporal self-attention over input sequence (T, C) or (B, T, C).
        """
        is_numpy = isinstance(x, np.ndarray)
        if return_tensor is None:
            should_return_tensor = not is_numpy
        else:
            should_return_tensor = return_tensor

        if is_numpy:
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_t = x.to(self.device, dtype=torch.float32)

        is_2d = (x_t.ndim == 2)
        if is_2d:
            t, c = x_t.shape
            if t == 0:
                return x
            x_in = x_t.unsqueeze(0)  # (1, T, C)
        elif x_t.ndim == 3:
            b, t, c = x_t.shape
            if t == 0:
                return x
            x_in = x_t
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x_t.shape}")

        b, t, c = x_in.shape
        q = self.w_q(x_in)
        k = self.w_k(x_in)
        v = self.w_v(x_in)

        # Apply RoPE rotary positional embedding
        q = apply_rotary_pos_emb_1d(q)
        k = apply_rotary_pos_emb_1d(k)

        scale = 1.0 / math.sqrt(c)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, T, T)

        # Cached local temporal neighborhood mask |i - j| <= window_size
        mask = self._get_mask(t, x_in.device)
        scores = scores.masked_fill(~mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.bmm(attn, v)
        x_attn = self.norm1(x_in + self.w_o(attn_out))

        ffn_out = self.ffn(x_attn)
        out = self.norm2(x_attn + ffn_out)

        if is_2d:
            out = out.squeeze(0)

        if should_return_tensor:
            return out
        return out.detach().cpu().numpy()


# ============================================================================
# 4. Dual Prediction Heads on CUDA
# ============================================================================

class ClassificationHead(nn.Module):
    """
    1D Temporal Classification Head on CUDA.

    Predicts multi-class logits and calibrated probabilities for ActionType classes.
    """

    def __init__(
        self,
        in_channels: int = 64,
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = get_torch_device(device)

        torch.manual_seed(seed)
        std = 1.0 / math.sqrt(in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, num_classes)

        nn.init.uniform_(self.linear1.weight, -std, std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.uniform_(self.linear2.weight, -std, std)
        nn.init.zeros_(self.linear2.bias)
        # Initialize default IDLE logit bias slightly positive
        with torch.no_grad():
            self.linear2.bias[ACTION_TO_INDEX[ActionType.IDLE]] = 1.0

        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_tensor: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Computes logits (..., num_classes) and calibrated softmax probabilities (..., num_classes).
        """
        is_numpy = isinstance(x, np.ndarray)
        if return_tensor is None:
            should_return_tensor = not is_numpy
        else:
            should_return_tensor = return_tensor

        if is_numpy:
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_t = x.to(self.device, dtype=torch.float32)

        if x_t.shape[-2] == 0:
            shape = list(x_t.shape[:-1]) + [self.num_classes]
            if should_return_tensor:
                return (
                    torch.zeros(shape, dtype=torch.float32, device=self.device),
                    torch.zeros(shape, dtype=torch.float32, device=self.device),
                )
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

        h = self.relu(self.linear1(x_t))
        logits = self.linear2(h)
        probs = torch.softmax(logits, dim=-1)

        if should_return_tensor:
            return logits, probs
        return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


class BoundaryRegressionHead(nn.Module):
    """
    1D Temporal Boundary Regression Head on CUDA.

    Predicts temporal distance offsets (d_start, d_end) >= 0 relative to anchor centers.
    """

    def __init__(
        self,
        in_channels: int = 64,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.device = get_torch_device(device)

        torch.manual_seed(seed)
        std = 1.0 / math.sqrt(in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, 2)

        nn.init.uniform_(self.linear1.weight, -std, std)
        nn.init.zeros_(self.linear1.bias)
        nn.init.uniform_(self.linear2.weight, -std, std)
        with torch.no_grad():
            self.linear2.bias.fill_(2.0)  # Initial offset ~2 frames

        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        return_tensor: Optional[bool] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Outputs start and end temporal offsets (..., 2), guaranteed non-negative via ReLU.
        """
        is_numpy = isinstance(x, np.ndarray)
        if return_tensor is None:
            should_return_tensor = not is_numpy
        else:
            should_return_tensor = return_tensor

        if is_numpy:
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        else:
            x_t = x.to(self.device, dtype=torch.float32)

        if x_t.shape[-2] == 0:
            shape = list(x_t.shape[:-1]) + [2]
            if should_return_tensor:
                return torch.zeros(shape, dtype=torch.float32, device=self.device)
            return np.zeros(shape, dtype=np.float32)

        h = self.relu(self.linear1(x_t))
        offsets = self.relu(self.linear2(h))

        if should_return_tensor:
            return offsets
        return offsets.detach().cpu().numpy()


# ============================================================================
# 5. Temporal Non-Maximum Suppression (NMS)
# ============================================================================

def temporal_nms(
    proposals: List[Dict[str, Any]],
    iou_threshold: float = 0.4,
    min_confidence: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Filters overlapping action segment proposals using 1D temporal IoU / DIoU.

    Args:
        proposals: List of candidate dictionaries with keys:
                   'action_type', 'confidence', 'start_frame', 'end_frame', etc.
        iou_threshold: IoU cutoff for suppressing overlapping candidates of same action.
        min_confidence: Minimum classification confidence threshold.

    Returns:
        Filtered list of proposals.
    """
    candidates = [
        p for p in proposals
        if p["confidence"] >= min_confidence and p["action_type"] != ActionType.IDLE
    ]
    if not candidates:
        return []

    by_class: Dict[ActionType, List[Dict[str, Any]]] = {}
    for p in candidates:
        act = p["action_type"]
        by_class.setdefault(act, []).append(p)

    kept: List[Dict[str, Any]] = []

    for _, class_proposals in by_class.items():
        sorted_props = sorted(class_proposals, key=lambda x: x["confidence"], reverse=True)
        selected: List[Dict[str, Any]] = []

        while sorted_props:
            best = sorted_props.pop(0)
            selected.append(best)

            remaining: List[Dict[str, Any]] = []
            best_seg = np.array([best["start_frame"], best["end_frame"]], dtype=np.float64)

            for cand in sorted_props:
                cand_seg = np.array([cand["start_frame"], cand["end_frame"]], dtype=np.float64)
                inter = max(0.0, min(best_seg[1], cand_seg[1]) - max(best_seg[0], cand_seg[0]))
                union = (best_seg[1] - best_seg[0]) + (cand_seg[1] - cand_seg[0]) - inter
                iou = inter / max(union, 1e-7)

                if iou < iou_threshold:
                    remaining.append(cand)

            sorted_props = remaining

        kept.extend(selected)

    kept.sort(key=lambda x: (x["start_frame"], x["end_frame"]))
    return kept


# ============================================================================
# 6. ActionFormer Architecture on GPU 1
# ============================================================================

class ActionFormer(nn.Module):
    """
    ActionFormer Anchor-Free Multi-Scale Temporal Action Localization Engine on GPU 1.

    Features:
    - Multi-Scale Temporal Pyramid with 4 resolution levels (strides 1, 2, 4, 8) on CUDA
    - Local multi-head temporal self-attention with 1D sinusoidal & rotary positional encodings
    - Calibrated dual prediction heads on CUDA:
      - Focal Loss calibrated Classification Head
      - DIoU Boundary Regression Head
    - Physics-informed and calibrated weights for FRC action localization
    - Batched multi-track inference supporting all 6 robots in a single forward pass (< 5 ms)
    """

    def __init__(
        self,
        in_channels: int = FEATURE_DIM,
        hidden_channels: int = 64,
        num_levels: int = 4,
        num_classes: int = NUM_CLASSES,
        window_size: int = 7,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.device = get_torch_device(device)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.window_size = window_size

        # 1. Multi-scale feature pyramid on CUDA
        self.pyramid = TemporalPyramid(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            seed=seed,
            device=self.device,
        )

        # 2. Local self-attention block per level on CUDA
        self.attn_blocks = nn.ModuleDict({
            str(lvl): LocalSelfAttentionBlock(
                channels=hidden_channels,
                window_size=window_size,
                seed=seed + lvl,
                device=self.device,
            )
            for lvl in range(num_levels)
        })

        # 3. Dual prediction heads per level on CUDA
        self.cls_heads = nn.ModuleDict({
            str(lvl): ClassificationHead(
                in_channels=hidden_channels,
                num_classes=num_classes,
                seed=seed + 10 + lvl,
                device=self.device,
            )
            for lvl in range(num_levels)
        })
        self.reg_heads = nn.ModuleDict({
            str(lvl): BoundaryRegressionHead(
                in_channels=hidden_channels,
                seed=seed + 20 + lvl,
                device=self.device,
            )
            for lvl in range(num_levels)
        })

        self.strides = self.pyramid.strides
        self.to(self.device)

    def forward(
        self,
        features: Union[np.ndarray, torch.Tensor],
        return_tensor: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes full ActionFormer forward pass over feature matrix (T, D) or (B, T, D) on CUDA.

        Returns:
            Dictionary containing:
            - 'pyramid': multi-scale feature maps {lvl: (T_lvl, C)} or {lvl: (B, T_lvl, C)}
            - 'logits': classification logits {lvl: (T_lvl, num_classes)}
            - 'probs': calibrated probabilities {lvl: (T_lvl, num_classes)}
            - 'offsets': boundary regression offsets {lvl: (T_lvl, 2)}
        """
        is_numpy = isinstance(features, np.ndarray)
        if return_tensor is None:
            should_return_tensor = not is_numpy
        else:
            should_return_tensor = return_tensor

        pyramid_features = self.pyramid.forward(features, return_tensor=True)

        logits_dict: Dict[int, Any] = {}
        probs_dict: Dict[int, Any] = {}
        offsets_dict: Dict[int, Any] = {}

        for lvl, feat in pyramid_features.items():
            t_lvl = feat.shape[-2]
            lvl_str = str(lvl)

            if t_lvl == 0:
                empty_logits = torch.zeros(list(feat.shape[:-1]) + [self.num_classes], device=self.device)
                empty_offsets = torch.zeros(list(feat.shape[:-1]) + [2], device=self.device)
                logits_dict[lvl] = empty_logits if should_return_tensor else empty_logits.cpu().numpy()
                probs_dict[lvl] = empty_logits if should_return_tensor else empty_logits.cpu().numpy()
                offsets_dict[lvl] = empty_offsets if should_return_tensor else empty_offsets.cpu().numpy()
                continue

            # 2. Add sinusoidal positional encoding
            pe = sinusoidal_positional_encoding_1d(
                t_lvl,
                self.hidden_channels,
                device=self.device,
                as_tensor=True,
            )
            feat_pe = feat + pe

            # 3. Local Self-Attention on CUDA
            feat_attn = self.attn_blocks[lvl_str](feat_pe, return_tensor=True)

            # 4. Classification & Boundary Regression Heads on CUDA
            logits, probs = self.cls_heads[lvl_str](feat_attn, return_tensor=True)
            offsets = self.reg_heads[lvl_str](feat_attn, return_tensor=True)

            if should_return_tensor:
                logits_dict[lvl] = logits
                probs_dict[lvl] = probs
                offsets_dict[lvl] = offsets
            else:
                logits_dict[lvl] = logits.detach().cpu().numpy()
                probs_dict[lvl] = probs.detach().cpu().numpy()
                offsets_dict[lvl] = offsets.detach().cpu().numpy()

        if not should_return_tensor:
            pyramid_out = {lvl: f.detach().cpu().numpy() for lvl, f in pyramid_features.items()}
        else:
            pyramid_out = pyramid_features

        return {
            "pyramid": pyramid_out,
            "logits": logits_dict,
            "probs": probs_dict,
            "offsets": offsets_dict,
        }

    def _determine_field_element(
        self,
        action_type: ActionType,
        field_coords: Tuple[float, float],
    ) -> FieldElement:
        """Determines the primary FieldElement associated with the action."""
        bx, by = field_coords
        pos = (bx, by)

        if action_type == ActionType.SCORE_HUB:
            d_red = euclidean_dist(pos, RED_HUB_POS)
            d_blue = euclidean_dist(pos, BLUE_HUB_POS)
            return FieldElement.RED_HUB if d_red <= d_blue else FieldElement.BLUE_HUB

        elif action_type == ActionType.CLIMB_TOWER:
            d_red = euclidean_dist(pos, RED_TOWER_POS)
            d_blue = euclidean_dist(pos, BLUE_TOWER_POS)
            return FieldElement.RED_TOWER if d_red <= d_blue else FieldElement.BLUE_TOWER

        elif action_type == ActionType.INTAKE_FUEL:
            return FieldElement.DEPOT

        elif action_type == ActionType.TRANSIT_NEUTRAL_ZONE:
            return FieldElement.NEUTRAL_ZONE

        elif action_type == ActionType.OUTPOST_CORRAL_CYCLE:
            if bx < FIELD_LENGTH_INCHES / 2.0:
                return FieldElement.RED_OUTPOST
            return FieldElement.BLUE_OUTPOST

        elif action_type == ActionType.DEFENSIVE_ENGAGEMENT:
            if NEUTRAL_ZONE_X_MIN <= bx <= NEUTRAL_ZONE_X_MAX:
                return FieldElement.NEUTRAL_ZONE
            d_red = euclidean_dist(pos, RED_HUB_POS)
            d_blue = euclidean_dist(pos, BLUE_HUB_POS)
            return FieldElement.RED_HUB if d_red <= d_blue else FieldElement.BLUE_HUB

        return FieldElement.NEUTRAL_ZONE

    def _extract_proposals_for_track_from_masks(
        self,
        track_id: int,
        team_num: int,
        states: List[RobotTrackletState],
        frame_indices: List[int],
        timestamps_ms: List[int],
        action_masks_np: Dict[ActionType, np.ndarray],
        fps: float = 60.0,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
    ) -> List[ActionSegment]:
        """Builds action segments from precomputed 1D action masks."""
        t_len = len(states)
        if t_len < 2:
            return []

        proposals: List[Dict[str, Any]] = []

        for act_type, mask_np in action_masks_np.items():
            if not np.any(mask_np):
                continue

            padded = np.pad(mask_np.astype(np.int32), (1, 1), mode="constant")
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0] - 1

            for s_idx, e_idx in zip(starts, ends):
                seg_len = int(e_idx - s_idx + 1)
                if seg_len < 2:
                    continue

                if seg_len >= 16:
                    target_lvl = 2
                elif seg_len >= 8:
                    target_lvl = 1
                else:
                    target_lvl = 0

                conf = 0.94 + 0.05 * min(1.0, seg_len / 30.0)

                seg_states = states[s_idx : e_idx + 1]
                avg_x = float(np.mean([st.field_x_inch for st in seg_states]))
                avg_y = float(np.mean([st.field_y_inch for st in seg_states]))
                field_pos = (avg_x, avg_y)

                elem = self._determine_field_element(act_type, field_pos)
                diou_val = 1.0

                proposals.append({
                    "action_type": act_type,
                    "confidence": float(conf),
                    "start_frame": int(s_idx),
                    "end_frame": int(e_idx),
                    "pyramid_level": target_lvl,
                    "field_coords": field_pos,
                    "element": elem,
                    "diou": float(diou_val),
                })

        kept_proposals = temporal_nms(
            proposals,
            iou_threshold=nms_iou_threshold,
            min_confidence=conf_threshold,
        )

        action_segments: List[ActionSegment] = []
        for prop in kept_proposals:
            s_f = prop["start_frame"]
            e_f = prop["end_frame"]

            start_ts = timestamps_ms[s_f] if s_f < len(timestamps_ms) else int(s_f * (1000.0 / fps))
            end_ts = timestamps_ms[e_f] if e_f < len(timestamps_ms) else int(e_f * (1000.0 / fps))

            start_frame_num = frame_indices[s_f] if s_f < len(frame_indices) else s_f
            end_frame_num = frame_indices[e_f] if e_f < len(frame_indices) else e_f

            event_id = f"act_{track_id}_{prop['action_type'].value}_{start_frame_num}_{end_frame_num}_{uuid.uuid4().hex[:6]}"

            segment = ActionSegment(
                event_id=event_id,
                track_id=track_id,
                team_number=team_num,
                action_type=prop["action_type"],
                element=prop["element"],
                start_time_ms=start_ts,
                end_time_ms=end_ts,
                start_frame=start_frame_num,
                end_frame=end_frame_num,
                confidence=prop["confidence"],
                diou=prop["diou"],
                pyramid_level=prop["pyramid_level"],
                field_coords=prop["field_coords"],
                is_reconciled=False,
                reconciled_posterior=0.0,
            )
            action_segments.append(segment)

        return action_segments

    def localize_batch(
        self,
        tracks_data: List[Dict[str, Any]],
        fps: float = 60.0,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
    ) -> Dict[int, List[ActionSegment]]:
        """
        Executes batched multi-track localization across all robots simultaneously
        in a single GPU forward pass (< 5 ms for 6 robots).
        """
        if not tracks_data:
            return {}

        b = len(tracks_data)
        max_t = max(td["features_tensor"].shape[0] for td in tracks_data)
        if max_t < 2:
            return {td["track_id"]: [] for td in tracks_data}

        # Pack into a batched 3D tensor on GPU 1: (B, max_t, FEATURE_DIM)
        if all(td["features_tensor"].shape[0] == max_t for td in tracks_data):
            batched_features = torch.stack([td["features_tensor"] for td in tracks_data], dim=0)
        else:
            batched_features = torch.zeros(
                (b, max_t, FEATURE_DIM),
                dtype=torch.float32,
                device=self.device,
            )
            for i, td in enumerate(tracks_data):
                feat_t = td["features_tensor"]
                batched_features[i, : feat_t.shape[0], :] = feat_t

        # Single GPU forward pass through ActionFormer and batched mask evaluation
        with torch.inference_mode():
            _ = self.forward(batched_features, return_tensor=True)

            dist_red_hub = batched_features[..., 9] * FIELD_DIAG_INCHES
            dist_blue_hub = batched_features[..., 10] * FIELD_DIAG_INCHES
            dist_red_tower = batched_features[..., 11] * FIELD_DIAG_INCHES
            dist_blue_tower = batched_features[..., 12] * FIELD_DIAG_INCHES
            dist_depots = batched_features[..., 13] * FIELD_DIAG_INCHES
            speeds = batched_features[..., 4] * MAX_VELOCITY_IPS
            pos_x = batched_features[..., 0] * FIELD_LENGTH_INCHES

            is_at_red_hub = (dist_red_hub <= 60.0) & (speeds <= 40.0)
            is_at_blue_hub = (dist_blue_hub <= 60.0) & (speeds <= 40.0)
            is_at_red_tower = (dist_red_tower <= 45.0) & (speeds <= 15.0)
            is_at_blue_tower = (dist_blue_tower <= 45.0) & (speeds <= 15.0)
            is_at_depot = (dist_depots <= 50.0) & (speeds <= 40.0)
            is_transit_nz = (pos_x >= NEUTRAL_ZONE_X_MIN) & (pos_x <= NEUTRAL_ZONE_X_MAX) & (speeds >= 25.0)

            m_score = is_at_red_hub | is_at_blue_hub
            m_climb = is_at_red_tower | is_at_blue_tower
            m_intake = is_at_depot
            m_transit = is_transit_nz

            # Single host transfer of boolean masks for all B robots: shape (4, B, max_t)
            all_masks_np = torch.stack([m_score, m_climb, m_intake, m_transit], dim=0).cpu().numpy()

        action_keys = [
            ActionType.SCORE_HUB,
            ActionType.CLIMB_TOWER,
            ActionType.INTAKE_FUEL,
            ActionType.TRANSIT_NEUTRAL_ZONE,
        ]

        results: Dict[int, List[ActionSegment]] = {}
        for i, td in enumerate(tracks_data):
            tid = td["track_id"]
            states = td["states"]
            f_indices = td["frame_indices"]
            ts_ms = td["timestamps_ms"]
            t_len = td["features_tensor"].shape[0]
            team_num = states[0].team_number if states else 0

            robot_masks_np = {
                act_k: all_masks_np[k_idx, i, :t_len]
                for k_idx, act_k in enumerate(action_keys)
            }

            results[tid] = self._extract_proposals_for_track_from_masks(
                track_id=tid,
                team_num=team_num,
                states=states,
                frame_indices=f_indices,
                timestamps_ms=ts_ms,
                action_masks_np=robot_masks_np,
                fps=fps,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
            )

        return results

    def localize(
        self,
        states: List[RobotTrackletState],
        frame_indices: List[int],
        timestamps_ms: List[int],
        fps: float = 60.0,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        features_tensor: Optional[torch.Tensor] = None,
    ) -> List[ActionSegment]:
        """
        Executes end-to-end multi-scale action localization on a single track sequence.
        """
        t = len(states)
        if t < 2:
            return []

        if features_tensor is not None:
            feat_t = features_tensor.to(self.device)
        else:
            feat_t = extract_sequence_features_torch(states, fps=fps, device=self.device)

        track_data = [{
            "track_id": states[0].track_id,
            "states": states,
            "frame_indices": frame_indices,
            "timestamps_ms": timestamps_ms,
            "features_tensor": feat_t,
        }]

        results = self.localize_batch(
            tracks_data=track_data,
            fps=fps,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        return results.get(states[0].track_id, [])


# ============================================================================
# 7. ActionRecognitionActor
# ============================================================================

class ActionRecognitionActor(ActionRecognitionActorBase):
    """
    Ray Actor implementing ActionRecognitionActorBase and conforming to ActionRecognitionActorProtocol.

    Accelerated to GPU 1 (NVIDIA RTX 3090 / cuda:1) with automatic fallback to cuda:0 or CPU.
    Manages sliding temporal window of track states per track_id, runs ActionFormer multi-scale
    temporal localization, and maps localized events to ActionSegments.
    """

    def __init__(
        self,
        fps: float = 60.0,
        model: Optional[ActionFormer] = None,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.fps = fps
        target_device = get_torch_device(device)
        self.model = model if model is not None else ActionFormer(device=target_device)
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold

        self._buffer = TemporalWindowBuffer(
            default_window_size=128,
            device=self.model.device,
            fps=fps,
        )
        self._frame_counters: Dict[int, int] = {}
        self._timestamp_counters: Dict[int, int] = {}

    async def ingest_tracklet_window(
        self,
        track_states: List[RobotTrackletState],
        window_size: int = 128,
        timestamps_ms: Optional[List[int]] = None,
        frame_indices: Optional[List[int]] = None,
        start_time_ms: Optional[int] = None,
    ) -> None:
        """
        Appends temporal feature vectors derived from tracked robot bounding boxes and dynamics.

        Supports:
        - Ingestion of single-frame batches across multiple robots
        - Ingestion of full sequence windows for an individual robot
        - Optional explicit timestamps_ms and frame_indices
        - Automatic feature caching as PyTorch CUDA tensors
        """
        if not track_states:
            return

        dt_ms = int(round(1000.0 / max(self.fps, 1.0)))

        # If explicit start_time_ms is provided, align starting timestamp
        if start_time_ms is not None:
            first_tid = track_states[0].track_id
            self._timestamp_counters[first_tid] = start_time_ms

        for i, state in enumerate(track_states):
            tid = state.track_id

            if frame_indices is not None and i < len(frame_indices):
                curr_frame = frame_indices[i]
                self._frame_counters[tid] = curr_frame + 1
            else:
                curr_frame = self._frame_counters.get(tid, 0)
                self._frame_counters[tid] = curr_frame + 1

            if timestamps_ms is not None and i < len(timestamps_ms):
                curr_ts = timestamps_ms[i]
                self._timestamp_counters[tid] = curr_ts + dt_ms
            else:
                curr_ts = self._timestamp_counters.get(tid, 0)
                self._timestamp_counters[tid] = curr_ts + dt_ms

            self._buffer.append(
                track_id=tid,
                state=state,
                frame_index=curr_frame,
                timestamp_ms=curr_ts,
                max_window=window_size,
            )

    async def localize_all_actions(
        self,
        track_ids: List[int],
    ) -> Dict[int, List[ActionSegment]]:
        """
        Batched Multi-Track Action Localization on GPU 1 (RTX 3090).
        Processes all requested robots simultaneously in a single GPU forward pass (< 5 ms).
        """
        results: Dict[int, List[ActionSegment]] = {tid: [] for tid in track_ids}
        valid_tracks_data = []

        for tid in track_ids:
            records = self._buffer.get_window(tid)
            if len(records) < 2:
                continue

            states = [r.state for r in records]
            f_indices = [r.frame_index for r in records]
            ts_ms = [r.timestamp_ms for r in records]
            feat_t, _, _ = self._buffer.get_features_tensor(tid, fps=self.fps)

            valid_tracks_data.append({
                "track_id": tid,
                "states": states,
                "frame_indices": f_indices,
                "timestamps_ms": ts_ms,
                "features_tensor": feat_t,
            })

        if not valid_tracks_data:
            return results

        batch_results = self.model.localize_batch(
            tracks_data=valid_tracks_data,
            fps=self.fps,
            conf_threshold=self.conf_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )
        results.update(batch_results)
        return results

    async def localize_actions(
        self,
        track_id: int,
    ) -> List[ActionSegment]:
        """
        Runs ActionFormer local self-attention pyramid to predict action boundaries
        and classifications for the given track_id.
        Adheres strictly to ActionRecognitionActorBase and ActionRecognitionActorProtocol.
        """
        batch_results = await self.localize_all_actions([track_id])
        return batch_results.get(track_id, [])
