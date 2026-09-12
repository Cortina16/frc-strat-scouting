"""
ResNet-18 Frame Gate Classifier and Strided Frame Gating Policy.

Evaluates broadcast stream video frames to distinguish wide-angle field action
(green carpet / gameplay) from non-gameplay intervals (crowd shots, referee
discussions, score reveals, text banners, and static).

Supports:
- PyTorch ResNet-18 model weights (when PyTorch is installed).
- OpenCV DNN engine for ONNX models.
- Fast, robust color-histogram & field carpet heuristic fallback.
- Configurable frame gating stride (default: every 10th frame).
- Immediate dropping of non-gameplay intervening frames to conserve compute.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional PyTorch & Torchvision imports
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tv_models = None  # type: ignore[assignment]
    tv_transforms = None  # type: ignore[assignment]


def heuristic_field_classifier(
    image_bgr: np.ndarray,
    min_green_ratio: float = 0.15,
) -> Tuple[bool, float]:
    """
    Fast, robust heuristic classifier for wide-angle FRC field view.

    Identifies both the characteristic green carpet (e.g. REBUILT synthetic arena)
    and standard FRC grey Shaw carpet with yellow FUEL game pieces and red/blue
    alliance elements, while differentiating them from crowd shots, referee graphics,
    score banners, and high-frequency video noise/static.

    Returns:
        (is_gameplay, confidence): Tuple of bool and float in [0.0, 1.0].
    """
    if image_bgr is None or image_bgr.size == 0:
        return False, 0.0

    # Downsample large frames to ~320x180 thumbnail for sub-millisecond evaluation
    h, w = image_bgr.shape[:2]
    if w > 320 or h > 180:
        thumb = cv2.resize(image_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    else:
        thumb = image_bgr

    # Convert to HSV color space and split channels
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(thumb)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # -------------------------------------------------------------------------
    # 1. Green carpet arena check (synthetic test clips & green field variants)
    # -------------------------------------------------------------------------
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    green_ratio = float(np.count_nonzero(green_mask) / green_mask.size)

    dom_mask = (g.astype(np.int16) > r.astype(np.int16) + 15) & (
        g.astype(np.int16) > b.astype(np.int16) + 15
    )
    dom_ratio = float(np.count_nonzero(dom_mask) / dom_mask.size)

    green_peakiness = 0.0
    if green_ratio > 0.02:
        hist = cv2.calcHist([hsv], [0], green_mask, [180], [0, 180])
        green_peakiness = float(hist.max() / (hist.sum() + 1e-6))

    is_green_noise = (lap_var > 20000.0 and dom_ratio < 0.5) or (
        green_peakiness < 0.08 and green_ratio < 0.4
    )
    is_green_gameplay = bool(
        (green_ratio >= min_green_ratio) and (dom_ratio >= 0.10) and not is_green_noise
    )

    if is_green_gameplay:
        confidence = float(min(1.0, 0.5 + (green_ratio / 0.5) * 0.5))
        return True, confidence

    # -------------------------------------------------------------------------
    # 2. Grey carpet arena check (Official FRC Shaw grey carpet & broadcast)
    # -------------------------------------------------------------------------
    diff_rg = np.abs(r.astype(np.int16) - g.astype(np.int16))
    diff_gb = np.abs(g.astype(np.int16) - b.astype(np.int16))
    diff_rb = np.abs(r.astype(np.int16) - b.astype(np.int16))

    # Neutral grey carpet: low saturation, moderate value, channel balance
    grey_carpet_mask = (
        (hsv[:, :, 1] < 60)
        & (hsv[:, :, 2] >= 35)
        & (hsv[:, :, 2] <= 195)
        & (diff_rg < 28)
        & (diff_gb < 28)
        & (diff_rb < 28)
    )
    grey_ratio = float(np.count_nonzero(grey_carpet_mask) / grey_carpet_mask.size)

    # Yellow game pieces (FUEL balls, Power Cells, Notes): H in [15, 38], S > 60, V > 60
    yellow_mask = (
        (hsv[:, :, 0] >= 15)
        & (hsv[:, :, 0] <= 38)
        & (hsv[:, :, 1] > 60)
        & (hsv[:, :, 2] > 60)
    )
    yellow_ratio = float(np.count_nonzero(yellow_mask) / yellow_mask.size)

    # Alliance markers/bumpers: Blue [95, 135] and Red [0, 12] or [168, 180]
    blue_mask = (
        (hsv[:, :, 0] >= 95)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] > 45)
        & (hsv[:, :, 2] > 40)
    )
    red_mask = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
        & (hsv[:, :, 1] > 45)
        & (hsv[:, :, 2] > 40)
    )
    blue_ratio = float(np.count_nonzero(blue_mask) / blue_mask.size)
    red_ratio = float(np.count_nonzero(red_mask) / red_mask.size)

    # Structural edges (perimeter borders, hub/truss structures, robot outlines)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(np.count_nonzero(edges) / edges.size)

    # Rejection of high-frequency noise or solid featureless banners
    is_noise_or_static = (lap_var > 20000.0) or (lap_var < 50.0 and edge_ratio < 0.01)

    has_carpet = (0.18 <= grey_ratio <= 0.75)
    has_alliance_elements = (
        (blue_ratio >= 0.03 and red_ratio >= 0.02)
        or (yellow_ratio >= 0.01 and (blue_ratio >= 0.02 or red_ratio >= 0.02))
    )
    has_structure = (edge_ratio >= 0.04) and (lap_var >= 150.0)

    is_grey_gameplay = bool(
        has_carpet and has_alliance_elements and has_structure and not is_noise_or_static
    )

    if is_grey_gameplay:
        element_score = min(1.0, (yellow_ratio + blue_ratio + red_ratio) / 0.20)
        carpet_score = min(1.0, grey_ratio / 0.35)
        confidence = float(min(1.0, 0.6 + 0.2 * carpet_score + 0.2 * element_score))
        return True, confidence

    # Non-gameplay frame: compute high rejection confidence (>0.80)
    diff = max(0.0, min_green_ratio - green_ratio)
    confidence = float(min(1.0, 0.85 + (diff / min_green_ratio) * 0.15))
    return False, confidence


class FrameGateClassifier:
    """
    Classifier for wide-angle gameplay vs non-gameplay broadcast frames.

    Supports:
    1. Custom callable classifier: (image_bgr) -> (is_gameplay, confidence)
    2. PyTorch ResNet-18 classifier (if PyTorch is installed and weights provided)
    3. OpenCV DNN engine (for ONNX models)
    4. Heuristic color-histogram and carpet dominance fallback
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        custom_classifier: Optional[
            Callable[[np.ndarray], Tuple[bool, float]]
        ] = None,
        method: str = "auto",
        min_green_ratio: float = 0.15,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.custom_classifier = custom_classifier
        self.method = method
        self.min_green_ratio = min_green_ratio

        self._torch_model: Optional[Any] = None
        self._torch_transform: Optional[Any] = None
        self._cv_net: Optional[cv2.dnn.Net] = None
        self._backend: str = "heuristic"

        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        if self.custom_classifier is not None:
            self._backend = "custom"
            return

        # Attempt OpenCV DNN if model_path is ONNX
        if self.model_path and self.model_path.endswith((".onnx", ".pb")):
            try:
                self._cv_net = cv2.dnn.readNet(self.model_path)
                self._backend = "opencv_dnn"
                logger.info("Initialized OpenCV DNN model from %s", self.model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load OpenCV DNN model from %s: %s", self.model_path, exc)

        # Attempt PyTorch ResNet-18 if torch is available and requested / weights provided
        if TORCH_AVAILABLE and (
            self.method in ("resnet18", "torch") or self.model_path is not None
        ):
            try:
                self._torch_model = self._build_resnet18()
                self._backend = "resnet18"
                logger.info("Initialized PyTorch ResNet-18 frame gate")
                return
            except Exception as exc:
                logger.warning("Failed to initialize PyTorch ResNet-18: %s", exc)

        # Default fallback to heuristic
        self._backend = "heuristic"
        logger.info("Using robust color-histogram heuristic for frame gating")

    def _build_resnet18(self) -> Any:
        """Constructs a ResNet-18 binary classifier head."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

        # Create ResNet-18 model
        model = tv_models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)  # Class 0: non_gameplay, Class 1: gameplay

        if self.model_path:
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)

        model.to(self.device)
        model.eval()

        self._torch_transform = tv_transforms.Compose(
            [
                tv_transforms.ToPILImage(),
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model

    def classify(self, image_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Classifies an input BGR image as gameplay (wide field) or non-gameplay.

        Returns:
            (is_gameplay, confidence): Tuple of bool and float in [0.0, 1.0].
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, 0.0

        if self.custom_classifier is not None:
            return self.custom_classifier(image_bgr)

        if self._backend == "resnet18" and self._torch_model is not None:
            return self._classify_torch(image_bgr)

        if self._backend == "opencv_dnn" and self._cv_net is not None:
            return self._classify_cv_dnn(image_bgr)

        return heuristic_field_classifier(
            image_bgr, min_green_ratio=self.min_green_ratio
        )

    def _classify_torch(self, image_bgr: np.ndarray) -> Tuple[bool, float]:
        if (
            not TORCH_AVAILABLE
            or torch is None
            or self._torch_model is None
            or self._torch_transform is None
        ):
            return heuristic_field_classifier(
                image_bgr, min_green_ratio=self.min_green_ratio
            )
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._torch_transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._torch_model(tensor)
            probs = torch.softmax(logits, dim=-1)
            gameplay_prob = float(probs[0, 1].item())
            is_gameplay = bool(gameplay_prob >= self.confidence_threshold)
            conf = gameplay_prob if is_gameplay else (1.0 - gameplay_prob)
            return is_gameplay, float(conf)

    def _classify_cv_dnn(self, image_bgr: np.ndarray) -> Tuple[bool, float]:
        if self._cv_net is None:
            return heuristic_field_classifier(
                image_bgr, min_green_ratio=self.min_green_ratio
            )
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(104.0, 117.0, 123.0),
            swapRB=True,
            crop=False,
        )
        self._cv_net.setInput(blob)
        out = self._cv_net.forward()
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / np.sum(exp_out)
        gameplay_prob = (
            float(probs[0, 1]) if probs.shape[-1] >= 2 else float(probs[0, 0])
        )
        is_gameplay = bool(gameplay_prob >= self.confidence_threshold)
        conf = gameplay_prob if is_gameplay else (1.0 - gameplay_prob)
        return is_gameplay, float(conf)

    @property
    def backend(self) -> str:
        return self._backend


class FrameGate:
    """
    Strided frame gating policy.

    Evaluates the classifier every `stride` frames (default 10).
    Maintains the classification decision across intervening frames.
    When a frame is classified as non-gameplay, all intervening frames are
    dropped immediately without further processing.
    """

    def __init__(
        self,
        stride: int = 10,
        classifier: Optional[
            Union[FrameGateClassifier, Callable[[np.ndarray], Tuple[bool, float]]]
        ] = None,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = stride

        if classifier is None:
            self.classifier = FrameGateClassifier()
        elif isinstance(classifier, FrameGateClassifier):
            self.classifier = classifier
        else:
            self.classifier = FrameGateClassifier(custom_classifier=classifier)

        self._last_decision: bool = False
        self._last_confidence: float = 0.0
        self._has_decision: bool = False

        # Metrics
        self._total_evaluations: int = 0
        self._frames_accepted: int = 0
        self._frames_dropped: int = 0

    def is_evaluation_frame(self, frame_index: int) -> bool:
        """Returns True if the given frame_index requires classifier evaluation."""
        return not self._has_decision or (frame_index % self.stride == 0)

    def evaluate(
        self,
        frame_index: int,
        image_bgr: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Evaluates the frame gate for the specified frame_index and image.

        If `frame_index` is an evaluation frame, runs classification and updates
        internal state. Otherwise, applies the decision of the current stride window.

        Returns:
            (is_gameplay, confidence): Decision for this frame.
        """
        if self.is_evaluation_frame(frame_index):
            is_gameplay, conf = self.classifier.classify(image_bgr)
            self._last_decision = is_gameplay
            self._last_confidence = conf
            self._has_decision = True
            self._total_evaluations += 1
        else:
            is_gameplay = self._last_decision
            conf = self._last_confidence

        if is_gameplay:
            self._frames_accepted += 1
        else:
            self._frames_dropped += 1

        return is_gameplay, conf

    def record_intervening_drop(self) -> None:
        """Records an intervening frame dropped without decoding the full image tensor."""
        self._frames_dropped += 1

    def reset(self) -> None:
        """Resets gate state and counters."""
        self._last_decision = False
        self._last_confidence = 0.0
        self._has_decision = False
        self._total_evaluations = 0
        self._frames_accepted = 0
        self._frames_dropped = 0

    @property
    def last_decision(self) -> bool:
        return self._last_decision

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def total_evaluations(self) -> int:
        return self._total_evaluations

    @property
    def frames_accepted(self) -> int:
        return self._frames_accepted

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped
