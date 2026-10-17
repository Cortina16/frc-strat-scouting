"""
Match Video Training & Fine-Tuning Pipeline for FRC 2026 REBUILT.

Extracts keyframes from raw match footage (e.g. output.mp4), generates high-confidence
pseudo-ground-truth bounding box labels for Robots and FUEL game pieces using the multi-spectral
RobotBumperVerificationGate and colorimetric ball detection, formats a standard YOLO dataset,
and fine-tunes a YOLO model on GPU (NVIDIA RTX 3090, cuda:0).

Usage:
    python scripts/train_on_match_video.py --video output.mp4 --num-frames 800 --epochs 15 --device 0
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.common.types import BoundingBox
from src.ingestion.frame_gate import heuristic_field_classifier
from src.vision.detector_trt import RobotBumperVerificationGate

logger = logging.getLogger("frc.train")


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def detect_balls_colorimetric(
    frame_bgr: np.ndarray,
    min_radius: float = 2.5,
    max_radius: float = 25.0,
    max_balls: int = 150,
) -> List[Tuple[float, float, float, float]]:
    """
    Colorimetric detection of yellow FUEL foam balls.
    Returns list of [xc_norm, yc_norm, w_norm, h_norm].
    """
    h_img, w_img = frame_bgr.shape[:2]
    # Restrict to field region (avoid scoreboard)
    roi_y1 = int(0.12 * h_img)
    roi_y2 = int(0.95 * h_img)
    roi = frame_bgr[roi_y1:roi_y2, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Yellow foam ball HSV mask: H in [20, 38], S >= 90, V >= 110
    yellow_mask = (h_ch >= 20) & (h_ch <= 38) & (s_ch >= 90) & (v_ch >= 110)
    mask_u8 = yellow_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ball_boxes: List[Tuple[float, float, float, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * (min_radius ** 2) * 0.4:
            continue
        ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
        if min_radius <= radius <= max_radius:
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                if circularity >= 0.45:
                    global_cy = cy + roi_y1
                    box_w = 2.0 * radius
                    box_h = 2.0 * radius
                    xc_norm = float(cx / w_img)
                    yc_norm = float(global_cy / h_img)
                    w_norm = float(box_w / w_img)
                    h_norm = float(box_h / h_img)
                    ball_boxes.append((xc_norm, yc_norm, w_norm, h_norm))

    # Keep top max_balls
    return ball_boxes[:max_balls]


def extract_pseudo_labeled_dataset(
    video_path: str,
    output_dataset_dir: str,
    base_model_path: str = "data/models/medium_robo_v3.pt",
    num_samples: int = 800,
    val_split: float = 0.20,
    seed: int = 42,
    device: str = "0",
) -> str:
    """
    Extracts keyframes from video_path, generates high-confidence pseudo-labels
    for Robot (0) and Fuel (1), and saves a YOLO dataset.
    """
    random.seed(seed)
    np.random.seed(seed)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        "Source video: %s (%dx%d @ %.2f FPS, %d total frames, %.1fs duration)",
        video_path, w_img, h_img, fps, total_frames, total_frames / fps,
    )

    # Initialize teacher detector and verification gate
    logger.info("Initializing teacher model (%s) on device %s...", base_model_path, device)
    teacher_model = YOLO(base_model_path)
    gate = RobotBumperVerificationGate(
        min_bumper_score=0.10,
        min_superstructure_score=0.06,
        max_robots_total=6,
        max_robots_per_alliance=3,
        min_aspect_ratio=0.40,
        max_aspect_ratio=2.60,
    )

    # Prepare dataset directories
    dataset_path = Path(output_dataset_dir)
    img_train_dir = dataset_path / "images" / "train"
    img_val_dir = dataset_path / "images" / "val"
    lbl_train_dir = dataset_path / "labels" / "train"
    lbl_val_dir = dataset_path / "labels" / "val"

    for p in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Sample frame indices across active gameplay (skip first 300 and last 1200 frames)
    start_frame = min(400, total_frames // 10)
    end_frame = max(start_frame + 100, total_frames - 1200)
    available_indices = list(range(start_frame, end_frame, max(1, (end_frame - start_frame) // (num_samples * 2))))
    random.shuffle(available_indices)
    sampled_indices = sorted(available_indices[:num_samples])

    logger.info("Selected %d candidate frames across frames %d..%d", len(sampled_indices), start_frame, end_frame)

    processed_count = 0
    robot_count = 0
    ball_count = 0

    t_start = time.perf_counter()

    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Check field classifier
        is_field, field_conf = heuristic_field_classifier(frame)
        if not is_field:
            continue

        # 1. Run teacher model for candidate robot boxes
        yolo_res = teacher_model.predict(
            source=frame,
            conf=0.15,
            iou=0.28,
            device=device,
            verbose=False,
            imgsz=640,
        )

        candidate_boxes: List[BoundingBox] = []
        if len(yolo_res) > 0 and yolo_res[0].boxes is not None:
            boxes_data = yolo_res[0].boxes
            for b in boxes_data:
                cls_id = int(b.cls[0].item())
                if cls_id == 0:  # Robot class
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].item())
                    bw = float(xyxy[2] - xyxy[0])
                    bh = float(xyxy[3] - xyxy[1])
                    bcx = float(xyxy[0] + bw / 2.0)
                    bcy = float(xyxy[1] + bh / 2.0)
                    candidate_boxes.append(BoundingBox(
                        center_x=bcx,
                        center_y=bcy,
                        width=bw,
                        height=bh,
                        confidence=conf,
                        class_id=0,
                    ))

        # 2. Filter candidate boxes via multi-spectral verification gate
        verified_robots = gate.verify_candidates(
            frame_bgr=frame,
            candidates=candidate_boxes,
            is_autonomous=(idx < 1300),  # Approx first 15-20s of gameplay
        )

        # 3. Detect yellow FUEL balls
        detected_balls = detect_balls_colorimetric(frame)

        # Skip frames with zero objects
        if not verified_robots and not detected_balls:
            continue

        # Determine train vs val split
        is_val = (random.random() < val_split)
        split_img_dir = img_val_dir if is_val else img_train_dir
        split_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

        base_name = f"frame_{idx:06d}"
        img_dest = split_img_dir / f"{base_name}.jpg"
        lbl_dest = split_lbl_dir / f"{base_name}.txt"

        # Save image (JPEG quality 95)
        cv2.imwrite(str(img_dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write labels: class 0 for Robot, class 1 for Fuel
        with open(lbl_dest, "w", encoding="utf-8") as f_lbl:
            for r_box in verified_robots:
                xc_norm = float(r_box.center_x / w_img)
                yc_norm = float(r_box.center_y / h_img)
                w_norm = float(r_box.width / w_img)
                h_norm = float(r_box.height / h_img)
                f_lbl.write(f"0 {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                robot_count += 1

            for (b_xc, b_yc, b_w, b_h) in detected_balls:
                f_lbl.write(f"1 {b_xc:.6f} {b_yc:.6f} {b_w:.6f} {b_h:.6f}\n")
                ball_count += 1

        processed_count += 1
        if processed_count % 100 == 0:
            elapsed = time.perf_counter() - t_start
            logger.info("Processed %d frames (%.1f FPS) -> %d robots, %d balls",
                        processed_count, processed_count / elapsed, robot_count, ball_count)

    cap.release()

    # Generate dataset.yaml
    dataset_yaml_path = dataset_path / "dataset.yaml"
    dataset_meta = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "Robot",
            1: "Fuel",
        },
    }
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_meta, f, sort_keys=False)

    total_time = time.perf_counter() - t_start
    logger.info(
        "Dataset generation complete in %.1fs: %d images generated (%d robots, %d balls) at %s",
        total_time, processed_count, robot_count, ball_count, dataset_yaml_path,
    )
    return str(dataset_yaml_path)


def fine_tune_yolo_model(
    dataset_yaml_path: str,
    base_model_path: str = "data/models/medium_robo_v3.pt",
    output_model_path: str = "data/models/einstein_raw_v1.pt",
    epochs: int = 15,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project_dir: str = "runs/train",
    name: str = "einstein_raw_v1",
) -> str:
    """
    Fine-tunes YOLO model on the generated dataset and exports the weights.
    """
    logger.info("Loading base weights: %s", base_model_path)
    model = YOLO(base_model_path)

    logger.info(
        "Beginning fine-tuning: epochs=%d, batch=%d, imgsz=%d, device=%s",
        epochs, batch_size, imgsz, device,
    )

    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project_dir,
        name=name,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=2,
        val=True,
        save=True,
        plots=False,
        exist_ok=True,
    )

    # Locate best weights from training run
    best_weights_path = Path(project_dir) / name / "weights" / "best.pt"
    if not best_weights_path.exists():
        last_weights = Path(project_dir) / name / "weights" / "last.pt"
        best_weights_path = last_weights if last_weights.exists() else Path(base_model_path)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    shutil.copy(str(best_weights_path), output_model_path)
    logger.info("Successfully exported fine-tuned model to: %s", output_model_path)

    return output_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pseudo-labeled dataset and train YOLO on match video.")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Path to raw broadcast video file.")
    parser.add_argument("--dataset-dir", type=str, default="data/datasets/einstein_raw", help="Output dataset directory.")
    parser.add_argument("--base-model", type=str, default="data/models/medium_robo_v3.pt", help="Base model weights.")
    parser.add_argument("--output-model", type=str, default="data/models/einstein_raw_v1.pt", help="Exported model path.")
    parser.add_argument("--num-frames", "-n", type=int, default=600, help="Number of frames to extract and label.")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for training.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip dataset extraction if dataset exists.")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick smoke test without full training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    yaml_path = os.path.join(args.dataset_dir, "dataset.yaml")
    if not args.skip_extract or not os.path.exists(yaml_path):
        num_frames = 50 if args.dry_run else args.num_frames
        yaml_path = extract_pseudo_labeled_dataset(
            video_path=args.video,
            output_dataset_dir=args.dataset_dir,
            base_model_path=args.base_model,
            num_samples=num_frames,
            device=args.device,
        )

    epochs = 1 if args.dry_run else args.epochs
    fine_tune_yolo_model(
        dataset_yaml_path=yaml_path,
        base_model_path=args.base_model,
        output_model_path=args.output_model,
        epochs=epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
