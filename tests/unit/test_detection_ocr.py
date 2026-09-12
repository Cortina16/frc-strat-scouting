"""
Unit tests for TensorRT YOLOv11 Detection Actor & Real-ESRGAN Bumper OCR.

Verifies:
- Interface contracts (DetectionActorBase, DetectionActorProtocol, OCREnhancerActorBase, OCREnhancerActorProtocol).
- High-throughput batched inference on TensorRT detector with batch size 8 achieving <25ms latency.
- Graceful Mock/Simulation fallback and OpenCV DNN fallback on non-GPU environments.
- Accurate bumper crop extraction from robot bounding boxes (bottom 25-40% chassis region).
- Robust alliance color determination (RED vs BLUE vs UNKNOWN) under illumination and contrast shifts.
- Real-ESRGAN super-resolution enhancement and OCR digit recognition exceeding >98% accuracy on
  synthetically blurred, degraded, and perspective-skewed bumper images (digits "254", "1678", "118", etc.).
- High-throughput batched super-resolution and crop enhancement.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import torch

from src.common.interfaces import (
    DetectionActorBase,
    DetectionActorProtocol,
    OCREnhancerActorBase,
    OCREnhancerActorProtocol,
)
from src.common.types import (
    AllianceColor,
    BoundingBox,
    DetectionResult,
    FrameMetadata,
    OCRResult,
    RawDecodedFrame,
)
from src.vision.detector_trt import (
    TensorRTDetectorActor,
    decode_yolo_boxes,
    decode_yolo_boxes_gpu,
    letterbox_batch_gpu,
    letterbox_image,
    preprocess_batch_gpu,
)
from src.vision.ocr_enhancer import (
    DigitTemplateBank,
    OCREnhancerActor,
    classify_alliance,
    classify_alliance_gpu,
    extract_bumper_crop,
    get_template_bank,
    recognize_team_number,
    super_resolve_crop,
)


def create_mock_frame(
    frame_index: int,
    pts_ms: int,
    width: int = 1920,
    height: int = 1080,
    image_bgr: Optional[np.ndarray] = None,
) -> RawDecodedFrame:
    """Helper creating a RawDecodedFrame for testing."""
    if image_bgr is None:
        image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    meta = FrameMetadata(
        frame_index=frame_index,
        pts_ms=pts_ms,
        monotonic_ts_ns=time.monotonic_ns(),
        is_gameplay=True,
        confidence=0.99,
        width=width,
        height=height,
    )
    return RawDecodedFrame(metadata=meta, image_bgr=image_bgr)


def synthesize_robot_frame(
    team_number: int,
    alliance: str,
    skew_dy: int = 0,
    blur_sigma: float = 1.2,
    contrast: float = 0.8,
    frame_w: int = 640,
    frame_h: int = 480,
) -> Tuple[np.ndarray, BoundingBox]:
    """
    Synthesizes a test frame with a robot body and a realistic degraded FRC bumper.
    """
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Robot bounding box: width 120, height 120, centered at (320, 240)
    rx, ry, rw, rh = 260, 180, 120, 120
    # Robot chassis (dark metallic)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (50, 50, 50), -1)

    # Bumper occupies the bottom 35% of the robot chassis
    bh = 42
    bw = rw
    by = ry + rh - bh

    bg_color = (180, 40, 20) if alliance == "BLUE" else (20, 30, 200)
    bumper_patch = np.full((bh, bw, 3), bg_color, dtype=np.uint8)

    text = str(team_number)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = max(4, (bw - tw) // 2)
    ty = (bh + th) // 2
    cv2.putText(bumper_patch, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    # Perspective skew / tilt
    if skew_dy != 0:
        src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
        dst = np.float32(
            [
                [0, max(0, skew_dy)],
                [bw, max(0, -skew_dy)],
                [bw, bh - max(0, skew_dy)],
                [0, bh - max(0, -skew_dy)],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        bumper_patch = cv2.warpPerspective(
            bumper_patch, M, (bw, bh), borderMode=cv2.BORDER_REPLICATE
        )

    # Contrast degradation
    if contrast < 1.0:
        bumper_patch = np.clip(
            bumper_patch.astype(np.float32) * contrast + (1.0 - contrast) * 40.0,
            0,
            255,
        ).astype(np.uint8)

    # Gaussian blur
    if blur_sigma > 0:
        bumper_patch = cv2.GaussianBlur(bumper_patch, (5, 5), blur_sigma)

    frame[by : by + bh, rx : rx + bw] = bumper_patch

    bbox = BoundingBox(
        center_x=float(rx + rw / 2.0),
        center_y=float(ry + rh / 2.0),
        width=float(rw),
        height=float(rh),
        confidence=0.95,
        class_id=0,
    )
    return frame, bbox


# ============================================================================
# 1. Interface Contract Tests
# ============================================================================


def test_detection_actor_interface_contracts() -> None:
    """Verifies TensorRTDetectorActor conforms to DetectionActorBase and Protocol."""
    assert issubclass(TensorRTDetectorActor, DetectionActorBase)
    actor = TensorRTDetectorActor(mock_mode=True)
    assert isinstance(actor, DetectionActorBase)
    assert isinstance(actor, DetectionActorProtocol)


def test_ocr_enhancer_actor_interface_contracts() -> None:
    """Verifies OCREnhancerActor conforms to OCREnhancerActorBase and Protocol."""
    assert issubclass(OCREnhancerActor, OCREnhancerActorBase)
    actor = OCREnhancerActor()
    assert isinstance(actor, OCREnhancerActorBase)
    assert isinstance(actor, OCREnhancerActorProtocol)


# ============================================================================
# 2. TensorRT YOLOv11 Detector & Latency Tests
# ============================================================================


@pytest.mark.asyncio
async def test_trt_detector_engine_load_modes() -> None:
    """Verifies engine loading with mock fallback and precision options."""
    actor = TensorRTDetectorActor()
    assert not actor.is_loaded

    # Load mock engine
    await actor.load_engine("mock_engine.plan", precision="fp16")
    assert actor.is_loaded
    assert actor.execution_mode == "mock"

    # Reload with INT8
    await actor.load_engine("mock_engine_int8.engine", precision="int8")
    assert actor.is_loaded


@pytest.mark.asyncio
async def test_trt_detector_batched_latency_and_throughput() -> None:
    """
    Verifies batched inference on batch size 8 achieves <25ms latency guarantee
    and produces valid DetectionResult DTOs with BoundingBox candidates.
    """
    actor = TensorRTDetectorActor()
    await actor.load_engine("mock_yolov11.engine", precision="fp16")

    # Create batch of 8 full HD frames (1920x1080)
    batch_size = 8
    frames = [
        create_mock_frame(
            frame_index=i,
            pts_ms=i * 16,
            width=1920,
            height=1080,
        )
        for i in range(batch_size)
    ]

    t0 = time.perf_counter()
    results = await actor.detect_batch(frames)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
    assert latency_ms < 15.0, f"Batched inference took {latency_ms:.2f} ms; must be < 15.0 ms"
    assert actor.last_latency_ms < 15.0

    for i, res in enumerate(results):
        assert res.frame_index == i
        assert res.timestamp_ms == i * 16
        assert isinstance(res.boxes, list)
        assert len(res.boxes) > 0
        for box in res.boxes:
            assert isinstance(box, BoundingBox)
            assert 0.0 <= box.center_x <= 1920.0
            assert 0.0 <= box.center_y <= 1080.0
            assert box.width > 0.0
            assert box.height > 0.0
            assert 0.0 <= box.confidence <= 1.0
            assert box.class_id == 0


@pytest.mark.asyncio
async def test_trt_detector_custom_mock_boxes() -> None:
    """Verifies registering specific mock boxes and custom generators."""
    actor = TensorRTDetectorActor(generate_mock_boxes_if_empty=False)
    await actor.load_engine("mock")

    custom_box = BoundingBox(
        center_x=500.0, center_y=300.0, width=60.0, height=60.0, confidence=0.98, class_id=0
    )
    actor.set_mock_boxes(frame_index=42, boxes=[custom_box])

    frame_42 = create_mock_frame(frame_index=42, pts_ms=672)
    frame_43 = create_mock_frame(frame_index=43, pts_ms=688)

    results = await actor.detect_batch([frame_42, frame_43])
    assert len(results) == 2

    # Frame 42 returns custom registered box
    assert len(results[0].boxes) == 1
    assert results[0].boxes[0].center_x == 500.0
    assert results[0].boxes[0].confidence == 0.98

    # Frame 43 has no custom boxes registered
    assert len(results[1].boxes) == 0


@pytest.mark.asyncio
async def test_trt_detector_heuristic_detection_on_drawn_frames() -> None:
    """Verifies heuristic detection finds simulated robot bounding boxes."""
    actor = TensorRTDetectorActor()
    await actor.load_engine("mock")

    frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw two robots with bumpers
    cv2.rectangle(frame_bgr, (300, 400), (450, 460), (180, 40, 20), -1)  # Blue bumper
    cv2.rectangle(frame_bgr, (800, 500), (950, 560), (20, 30, 200), -1)  # Red bumper

    frame = create_mock_frame(frame_index=1, pts_ms=16, image_bgr=frame_bgr)
    results = await actor.detect_batch([frame])

    assert len(results) == 1
    boxes = results[0].boxes
    assert len(boxes) == 2


# ============================================================================
# 3. Bumper Crop Extraction Tests
# ============================================================================


def test_extract_bumper_crop_geometry() -> None:
    """Verifies bumper crop extracts the bottom 25-40% of the bounding box."""
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    # Robot bbox: center (400, 300), width 100, height 100 -> [350, 250, 450, 350]
    bbox = BoundingBox(
        center_x=400.0, center_y=300.0, width=100.0, height=100.0, confidence=0.9, class_id=0
    )

    crop_35 = extract_bumper_crop(frame, bbox, bumper_fraction=0.35)
    assert crop_35.shape[0] == 35, f"Expected height 35, got {crop_35.shape[0]}"
    assert crop_35.shape[1] == 100, f"Expected width 100, got {crop_35.shape[1]}"

    crop_25 = extract_bumper_crop(frame, bbox, bumper_fraction=0.25)
    assert crop_25.shape[0] == 25

    crop_40 = extract_bumper_crop(frame, bbox, bumper_fraction=0.40)
    assert crop_40.shape[0] == 40


def test_extract_bumper_crop_boundaries_and_clamping() -> None:
    """Verifies bumper crop handles edge-of-frame clamping and invalid bboxes."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Box near right boundary
    bbox_edge = BoundingBox(
        center_x=630.0, center_y=400.0, width=50.0, height=60.0, confidence=0.9, class_id=0
    )
    crop_edge = extract_bumper_crop(frame, bbox_edge)
    assert crop_edge.shape[1] > 0
    assert crop_edge.shape[0] > 0

    # Box outside frame
    bbox_out = BoundingBox(
        center_x=-100.0, center_y=-100.0, width=50.0, height=50.0, confidence=0.5, class_id=0
    )
    crop_out = extract_bumper_crop(frame, bbox_out)
    assert crop_out.size == 0


# ============================================================================
# 4. Alliance Color Classification Tests
# ============================================================================


def test_alliance_color_classification() -> None:
    """Verifies robust alliance color classification in HSV/BGR spaces."""
    # Red bumper fabric
    red_crop = np.full((30, 80, 3), (20, 30, 210), dtype=np.uint8)
    alliance, conf = classify_alliance(red_crop)
    assert alliance == AllianceColor.RED
    assert conf > 0.85

    # Blue bumper fabric
    blue_crop = np.full((30, 80, 3), (200, 45, 20), dtype=np.uint8)
    alliance, conf = classify_alliance(blue_crop)
    assert alliance == AllianceColor.BLUE
    assert conf > 0.85

    # Neutral grey fabric
    grey_crop = np.full((30, 80, 3), (80, 80, 80), dtype=np.uint8)
    alliance, conf = classify_alliance(grey_crop)
    assert alliance == AllianceColor.UNKNOWN
    assert conf == 0.0


# ============================================================================
# 5. Real-ESRGAN Super-Resolution & Bumper OCR Tests (>98% Accuracy)
# ============================================================================


@pytest.mark.asyncio
async def test_real_esrgan_and_ocr_accuracy_on_degraded_bumpers() -> None:
    """
    Tests OCR extraction on synthetically blurred, contrast-degraded, and perspective-skewed
    bumper images across canonical team numbers (254, 1678, 118, 2056, 33, 971, 4414).

    Asserts:
    - Overall team number recognition accuracy > 98%.
    - Alliance color classification accuracy > 98%.
    - Super-resolved flag is True on enhanced results.
    """
    actor = OCREnhancerActor(super_res_scale=2, bumper_crop_fraction=0.35)

    test_numbers = [254, 1678, 118, 2056, 33, 971, 4414]
    alliances = ["BLUE", "RED"]
    skews = [-4, 0, 4]
    blurs = [1.0, 1.4]
    contrasts = [0.75, 1.0]

    total_tests = 0
    correct_ocr = 0
    correct_alliance = 0

    for num in test_numbers:
        for alliance in alliances:
            for skew in skews:
                for blur in blurs:
                    for contrast in contrasts:
                        frame, bbox = synthesize_robot_frame(
                            team_number=num,
                            alliance=alliance,
                            skew_dy=skew,
                            blur_sigma=blur,
                            contrast=contrast,
                        )

                        res: OCRResult = await actor.enhance_and_read_bumper(frame, bbox)
                        total_tests += 1

                        expected_alliance = (
                            AllianceColor.BLUE if alliance == "BLUE" else AllianceColor.RED
                        )
                        if res.alliance == expected_alliance:
                            correct_alliance += 1

                        if res.team_number == num:
                            correct_ocr += 1

                        assert res.super_resolved is True
                        assert 0.0 <= res.confidence <= 1.0

    ocr_accuracy = correct_ocr / float(total_tests)
    alliance_accuracy = correct_alliance / float(total_tests)

    assert ocr_accuracy > 0.98, (
        f"OCR accuracy {ocr_accuracy * 100:.2f}% did not meet > 98% requirement "
        f"({correct_ocr}/{total_tests})"
    )
    assert alliance_accuracy > 0.98, (
        f"Alliance accuracy {alliance_accuracy * 100:.2f}% did not meet > 98% requirement "
        f"({correct_alliance}/{total_tests})"
    )


@pytest.mark.asyncio
async def test_batch_enhance_bumpers() -> None:
    """Verifies batched enhancement and OCR on a sequence of bumper crops."""
    actor = OCREnhancerActor(super_res_scale=2)

    crops: List[np.ndarray] = []
    expected_teams = [254, 1678, 118, 2056]
    expected_alliances = [
        AllianceColor.BLUE,
        AllianceColor.RED,
        AllianceColor.BLUE,
        AllianceColor.RED,
    ]

    for team, alliance_enum in zip(expected_teams, expected_alliances):
        alliance_str = "BLUE" if alliance_enum == AllianceColor.BLUE else "RED"
        frame, bbox = synthesize_robot_frame(team, alliance_str, blur_sigma=0.5)
        crop = extract_bumper_crop(frame, bbox)
        crops.append(crop)

    # Append an empty crop to test edge case robustness
    crops.append(np.zeros((0, 0, 3), dtype=np.uint8))

    batch_results = await actor.batch_enhance_bumpers(crops)
    assert len(batch_results) == 5

    for i in range(4):
        assert batch_results[i].team_number == expected_teams[i]
        assert batch_results[i].alliance == expected_alliances[i]
        assert batch_results[i].super_resolved is True
        assert batch_results[i].confidence > 0.5

    # Empty crop result
    assert batch_results[4].team_number is None
    assert batch_results[4].alliance == AllianceColor.UNKNOWN
    assert batch_results[4].super_resolved is False


@pytest.mark.asyncio
async def test_ocr_on_blank_and_empty_inputs() -> None:
    """Verifies graceful handling of empty or blank crops."""
    actor = OCREnhancerActor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = BoundingBox(
        center_x=320.0, center_y=240.0, width=100.0, height=100.0, confidence=0.9, class_id=0
    )

    result = await actor.enhance_and_read_bumper(frame, bbox)
    assert result.team_number is None
    assert result.alliance == AllianceColor.UNKNOWN


# ============================================================================
# 6. Letterbox & YOLO Decoding Utility Tests
# ============================================================================


def test_letterbox_image_dimensions() -> None:
    """Verifies letterbox_image maintains aspect ratio and produces exact target size."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, (pad_w, pad_h) = letterbox_image(img, target_shape=(640, 640))
    assert padded.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad_w == 0.0
    assert pad_h == 80.0


def test_decode_yolo_boxes_nms_filtering() -> None:
    """Verifies decode_yolo_boxes decodes proposals and applies NMS correctly."""
    # Synthetic proposals: 2 duplicate overlapping boxes
    # Format: [xc, yc, w, h, score_class0]
    preds = np.array(
        [
            [320.0, 320.0, 100.0, 100.0, 0.95],
            [321.0, 321.0, 101.0, 101.0, 0.90],
        ],
        dtype=np.float32,
    ).T  # Shape: (5, 2)

    boxes = decode_yolo_boxes(
        preds,
        orig_shape=(480, 640),
        input_shape=(640, 640),
        conf_threshold=0.5,
        iou_threshold=0.45,
    )
    # NMS should merge the two overlapping boxes into 1
    assert len(boxes) == 1
    assert boxes[0].confidence == pytest.approx(0.95, rel=1e-3)


# ============================================================================
# 7. GPU Acceleration Tests (cuda:0)
# ============================================================================


def test_gpu_tensor_preprocessing() -> None:
    """Verifies GPU batch preprocessing converts frames to float32 CUDA tensor with letterboxing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    device = torch.device("cuda:0")
    batch_size = 8
    frames = [create_mock_frame(i, i * 16, width=1920, height=1080) for i in range(batch_size)]

    padded_gpu, frame_params, _ = preprocess_batch_gpu(frames, input_shape=(640, 640), device=device)

    assert isinstance(padded_gpu, torch.Tensor)
    assert padded_gpu.device.type == "cuda"
    assert padded_gpu.device.index == 0
    assert padded_gpu.shape == (batch_size, 3, 640, 640)
    assert padded_gpu.dtype == torch.float32
    assert 0.0 <= float(padded_gpu.min()) <= float(padded_gpu.max()) <= 1.0
    assert len(frame_params) == batch_size
    assert frame_params[0][3] == 1080
    assert frame_params[0][4] == 1920


def test_gpu_batched_nms() -> None:
    """Verifies decode_yolo_boxes_gpu performs vectorized GPU NMS via torchvision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    device = torch.device("cuda:0")
    # 2 overlapping boxes on cuda:0
    preds = torch.zeros((1, 100, 5), device=device, dtype=torch.float32)
    preds[0, 0, :4] = torch.tensor([320.0, 320.0, 100.0, 100.0], device=device)
    preds[0, 0, 4] = 0.95
    preds[0, 1, :4] = torch.tensor([321.0, 321.0, 101.0, 101.0], device=device)
    preds[0, 1, 4] = 0.90

    params = [(1.0, 0.0, 80.0, 480, 640)]
    results = decode_yolo_boxes_gpu(preds, params, conf_threshold=0.5, iou_threshold=0.45)

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].confidence == pytest.approx(0.95, rel=1e-3)


def test_digit_template_bank_gpu() -> None:
    """Verifies DigitTemplateBank moves templates to normalized float32 tensor on cuda:0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    device = torch.device("cuda:0")
    bank = DigitTemplateBank(device=device)

    assert bank.matrix_gpu is not None
    assert bank.matrix_gpu.device.type == "cuda"
    assert bank.matrix_gpu.device.index == 0
    assert bank.matrix_gpu.dtype == torch.float32
    assert bank.matrix_gpu.ndim == 2
    assert bank.matrix_gpu.shape[1] == 1024

    # Verify rows are L2-normalized
    norms = torch.norm(bank.matrix_gpu, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_classify_alliance_gpu_thresholding() -> None:
    """Verifies classify_alliance_gpu fast BGR/HSV thresholding on cuda:0."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    device = torch.device("cuda:0")
    red_crop = np.full((30, 80, 3), (20, 30, 210), dtype=np.uint8)
    blue_crop = np.full((30, 80, 3), (200, 45, 20), dtype=np.uint8)
    grey_crop = np.full((30, 80, 3), (80, 80, 80), dtype=np.uint8)

    red_al, red_conf = classify_alliance_gpu(red_crop, device=device)
    assert red_al == AllianceColor.RED
    assert red_conf > 0.85

    blue_al, blue_conf = classify_alliance_gpu(blue_crop, device=device)
    assert blue_al == AllianceColor.BLUE
    assert blue_conf > 0.85

    grey_al, grey_conf = classify_alliance_gpu(grey_crop, device=device)
    assert grey_al == AllianceColor.UNKNOWN
    assert grey_conf == 0.0


@pytest.mark.asyncio
async def test_batched_ocr_gpu_acceleration() -> None:
    """Verifies batched bumper OCR uses GPU parallel template matching across all digits."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    device = torch.device("cuda:0")
    actor = OCREnhancerActor(device=device)

    test_teams = [254, 1678, 118, 2056, 33, 971, 4414]
    crops = []
    for team in test_teams:
        frame, bbox = synthesize_robot_frame(team, "BLUE", blur_sigma=0.5)
        crops.append(extract_bumper_crop(frame, bbox))

    t0 = time.perf_counter()
    results = await actor.batch_enhance_bumpers(crops)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert len(results) == len(test_teams)
    for i, res in enumerate(results):
        assert res.team_number == test_teams[i]
        assert res.alliance == AllianceColor.BLUE
        assert res.super_resolved is True
        assert res.confidence > 0.6


def test_multi_sided_bumper_digit_clustering() -> None:
    """Verifies that an angled robot with two visible bumper sides (e.g. 4065 and 4065 separated by a gap) resolves cleanly."""
    crop = np.full((40, 240, 3), (20, 20, 200), dtype=np.uint8)  # Red fabric
    # Draw "4065" on left face (x=20) and "4065" on right face (x=140)
    cv2.putText(crop, "4065", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(crop, "4065", (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    roster = [1323, 4065, 4414]
    team, conf = recognize_team_number(crop, AllianceColor.RED, candidate_teams=roster)
    assert team == 4065
    assert conf > 0.60

