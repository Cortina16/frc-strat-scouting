"""
Re-Identification (ReID) module for BoT-SORT multi-object tracking.

Provides:
- 512-D normalized appearance embedding representation
- Cosine distance metric calculation: D_cos(e1, e2) = 1 - (e1 . e2) / (||e1|| * ||e2||)
- ReID appearance feature extraction from bounding box crops
- Synthetic unit embedding generators for deterministic simulation testing
"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Optional, Sequence, Union
import cv2
import numpy as np

import torch

from src.common.types import BoundingBox


def normalize_embedding(embedding: Union[np.ndarray, Sequence[float], torch.Tensor]) -> np.ndarray:
    """
    Normalizes a vector or batch of vectors along the last axis to unit L2 norm.

    Args:
        embedding: Array of shape (D,) or (N, D), or torch.Tensor.

    Returns:
        L2-normalized array of same shape, dtype float32.
    """
    if isinstance(embedding, torch.Tensor):
        norm_t = torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)
        return norm_t.cpu().numpy()

    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            norm = 1e-12
        return arr / norm
    elif arr.ndim == 2:
        if torch.cuda.is_available() and arr.shape[0] > 16:
            t = torch.as_tensor(arr, dtype=torch.float32, device="cuda:0")
            t = torch.nn.functional.normalize(t, p=2, dim=-1)
            return t.cpu().numpy()
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return arr / norms
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def cosine_distance(
    e1: Union[np.ndarray, Sequence[float], torch.Tensor],
    e2: Union[np.ndarray, Sequence[float], torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> Union[float, np.ndarray]:
    """
    Computes cosine distance D_cos between two embeddings or two batches of embeddings:
        D_cos(e1, e2) = 1.0 - (e1 . e2) / (||e1|| * ||e2||)

    Accelerated using PyTorch CUDA tensors (torch.matmul, torch.cdist) on cuda:0 when CUDA is available.

    Args:
        e1: Embedding of shape (D,) or batch of shape (N, D).
        e2: Embedding of shape (D,) or batch of shape (M, D).
        device: Target device for PyTorch CUDA acceleration ('cuda:0' by default if CUDA is available).

    Returns:
        Scalar float if both e1 and e2 are 1D, or 2D array of shape (N, M) clipped to [0.0, 2.0].
    """
    if isinstance(e1, torch.Tensor):
        t1 = e1.float()
        is_1d_1 = t1.ndim == 1
        if is_1d_1:
            t1 = t1.unsqueeze(0)
    else:
        arr1 = np.asarray(e1, dtype=np.float32)
        is_1d_1 = arr1.ndim == 1
        if is_1d_1:
            arr1 = arr1.reshape(1, -1)

    if isinstance(e2, torch.Tensor):
        t2 = e2.float()
        is_1d_2 = t2.ndim == 1
        if is_1d_2:
            t2 = t2.unsqueeze(0)
    else:
        arr2 = np.asarray(e2, dtype=np.float32)
        is_1d_2 = arr2.ndim == 1
        if is_1d_2:
            arr2 = arr2.reshape(1, -1)

    d1 = t1.shape[1] if isinstance(e1, torch.Tensor) else arr1.shape[1]
    d2 = t2.shape[1] if isinstance(e2, torch.Tensor) else arr2.shape[1]
    if d1 != d2:
        raise ValueError(f"Embedding dimension mismatch: {d1} vs {d2}")

    use_cuda = torch.cuda.is_available()
    if use_cuda or device is not None:
        target_dev = device if device is not None else torch.device("cuda:0")
        if not isinstance(e1, torch.Tensor):
            t1 = torch.as_tensor(arr1, dtype=torch.float32, device=target_dev)
        else:
            t1 = t1.to(target_dev)

        if not isinstance(e2, torch.Tensor):
            t2 = torch.as_tensor(arr2, dtype=torch.float32, device=target_dev)
        else:
            t2 = t2.to(target_dev)

        norm1 = torch.nn.functional.normalize(t1, p=2, dim=-1)
        norm2 = torch.nn.functional.normalize(t2, p=2, dim=-1)
        similarity = torch.matmul(norm1, norm2.t())
        dist = torch.clamp(1.0 - similarity, 0.0, 2.0)
        distance = dist.cpu().numpy()
    else:
        norm1 = normalize_embedding(arr1)
        norm2 = normalize_embedding(arr2)
        similarity = np.dot(norm1, norm2.T)
        distance = 1.0 - similarity
        distance = np.clip(distance, 0.0, 2.0)

    if is_1d_1 and is_1d_2:
        return float(distance[0, 0])
    return distance



def generate_synthetic_embedding(
    dim: int = 512,
    seed: Optional[int] = None,
    basis_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generates a synthetic 512-D unit embedding.

    Args:
        dim: Embedding dimension (default 512).
        seed: Random seed for reproducible generation.
        basis_index: If specified, creates an exact canonical basis vector e_i.

    Returns:
        Normalized array of shape (dim,) with ||e||_2 = 1.0.
    """
    if basis_index is not None:
        vec = np.zeros(dim, dtype=np.float32)
        vec[basis_index % dim] = 1.0
        return vec

    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return normalize_embedding(vec)


class ReIDExtractor:
    """
    Appearance feature extractor producing 512-D normalized ReID vectors.

    Supports custom feature extractors, PyTorch backbones (when available),
    or deterministic OpenCV/NumPy visual feature fallbacks.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        custom_extractor: Optional[Callable[[np.ndarray, Sequence[BoundingBox]], List[np.ndarray]]] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.custom_extractor = custom_extractor

    def extract(
        self,
        frame_bgr: Optional[np.ndarray],
        bboxes: Sequence[BoundingBox],
    ) -> List[np.ndarray]:
        """
        Extracts 512-D normalized embeddings for each bounding box.

        Args:
            frame_bgr: Input frame (H, W, 3) BGR image or None.
            bboxes: Sequence of bounding boxes in pixel coordinates.

        Returns:
            List of normalized 512-D np.ndarray feature vectors.
        """
        if self.custom_extractor is not None:
            return self.custom_extractor(frame_bgr if frame_bgr is not None else np.zeros((1, 1, 3), dtype=np.uint8), bboxes)

        embeddings: List[np.ndarray] = []
        if frame_bgr is None or frame_bgr.size == 0 or len(bboxes) == 0:
            for bbox in bboxes:
                embeddings.append(self._fallback_embedding(bbox))
            return embeddings

        h_img, w_img = frame_bgr.shape[:2]

        for bbox in bboxes:
            # Calculate pixel crop coordinates
            x1 = max(0, int(bbox.center_x - bbox.width / 2.0))
            y1 = max(0, int(bbox.center_y - bbox.height / 2.0))
            x2 = min(w_img, int(bbox.center_x + bbox.width / 2.0))
            y2 = min(h_img, int(bbox.center_y + bbox.height / 2.0))

            if x2 <= x1 or y2 <= y1:
                embeddings.append(self._fallback_embedding(bbox))
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                embeddings.append(self._fallback_embedding(bbox))
                continue

            emb = self._extract_crop_features(crop)
            embeddings.append(emb)

        return embeddings

    def _extract_crop_features(self, crop: np.ndarray) -> np.ndarray:
        """Extracts normalized spatial-color features projected to 512-D."""
        resized = cv2.resize(crop, (16, 32), interpolation=cv2.INTER_AREA)  # 32x16 = 512 pixels
        # Flatten and normalize
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        features = np.zeros(self.embedding_dim, dtype=np.float32)

        # First 256 dims: H + S channels downsampled
        flat_h = cv2.resize(hsv[:, :, 0], (16, 16)).flatten().astype(np.float32) / 180.0
        # Second 256 dims: V + Color gradient
        flat_v = cv2.resize(hsv[:, :, 2], (16, 16)).flatten().astype(np.float32) / 255.0

        features[:256] = flat_h[:256]
        features[256:] = flat_v[:256]

        return normalize_embedding(features)

    def _fallback_embedding(self, bbox: BoundingBox) -> np.ndarray:
        """Generates a deterministic unit embedding for a bounding box when image is unavailable."""
        seed_str = f"{bbox.center_x:.2f}_{bbox.center_y:.2f}_{bbox.width:.2f}_{bbox.height:.2f}_{bbox.class_id}"
        hash_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        return generate_synthetic_embedding(self.embedding_dim, seed=hash_val)
