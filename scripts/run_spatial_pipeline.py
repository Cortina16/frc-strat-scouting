#!/usr/bin/env python3
"""
CLI Runner for the 3-Stage FRC 2026 REBUILT Spatial Analytics Pipeline.

Executes:
1. Spatial Calibration & Top-Down Metric Homography
2. Game Piece (Yellow FUEL) Quantification & Proportional Mass Allocation (504 max)
3. Deterministic Robot Localization & Strict 3v3 Tracking

Usage Examples:
    # 1. Run on synthetic demo match video
    python scripts/run_spatial_pipeline.py --demo --max-frames 120 --output reports/spatial_pipeline_demo.mp4

    # 2. Run on real broadcast video (output.mp4)
    python scripts/run_spatial_pipeline.py --video output.mp4 --start 300 --max-frames 200 --output reports/spatial_pipeline_match.mp4
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.spatial_pipeline import SpatialAnalyticsPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frc.run_spatial_pipeline")


def create_demo_match_video(output_path: str, num_frames: int = 120, fps: float = 30.0) -> str:
    """Generates synthetic broadcast footage with AprilTags, moving robots, and yellow FUEL balls."""
    logger.info("Generating synthetic demo match footage (%d frames)...", num_frames)
    w, h = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Pre-generate AprilTag markers
    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_images = {
        tid: cv2.cvtColor(cv2.aruco.generateImageMarker(dict_tag, tid, 60), cv2.COLOR_GRAY2BGR)
        for tid in [2, 5, 21, 18]  # Hub tags: 2, 5 (Red Hub), 21, 18 (Blue Hub)
    }

    np.random.seed(42)
    # 15 random stationary yellow ball clusters
    ball_clusters = [
        (int(np.random.randint(400, 1500)), int(np.random.randint(350, 750)), int(np.random.randint(8, 22)))
        for _ in range(15)
    ]

    for i in range(num_frames):
        # Field green carpet
        frame = np.full((h, w, 3), (35, 110, 35), dtype=np.uint8)

        # White tape lines
        cv2.line(frame, (200, 780), (1720, 780), (240, 240, 240), 3)
        cv2.line(frame, (450, 250), (1470, 250), (240, 240, 240), 3)
        cv2.line(frame, (960, 250), (960, 780), (240, 240, 240), 3)  # Center line

        # Alliance Hub Markers (AprilTags)
        # Red Hub area (left): tags 2 and 5
        frame[320:380, 480:540] = tag_images[2]
        frame[440:500, 480:540] = tag_images[5]
        # Blue Hub area (right): tags 21 and 18
        frame[320:380, 1380:1440] = tag_images[21]
        frame[440:500, 1380:1440] = tag_images[18]

        # Draw Yellow FUEL Game Pieces (Clusters on Carpet)
        for bx, by, r in ball_clusters:
            # Slight jitter simulating dynamic interaction
            jx = bx + int(np.sin(i * 0.1 + r) * 2)
            jy = by + int(np.cos(i * 0.1 + r) * 2)
            cv2.circle(frame, (jx, jy), r, (20, 225, 245), -1)

        # 3 Red Robots (Moving left to center)
        for r_idx in range(3):
            rx = int(320 + r_idx * 100 + np.sin(i * 0.08 + r_idx) * 60)
            ry = int(450 + r_idx * 90 + np.cos(i * 0.05 + r_idx) * 30)
            # Robot chassis
            cv2.rectangle(frame, (rx - 45, ry - 35), (rx + 45, ry + 35), (40, 40, 40), -1)
            # Red Bumper Base (lower 35%)
            cv2.rectangle(frame, (rx - 45, ry + 12), (rx + 45, ry + 35), (25, 25, 220), -1)
            cv2.putText(frame, f"R{r_idx+1}", (rx - 12, ry + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 3 Blue Robots (Moving right to center)
        for b_idx in range(3):
            bx = int(1550 - b_idx * 100 - np.sin(i * 0.08 + b_idx) * 60)
            by = int(450 + b_idx * 90 + np.sin(i * 0.06 + b_idx) * 30)
            # Robot chassis
            cv2.rectangle(frame, (bx - 45, by - 35), (bx + 45, by + 35), (40, 40, 40), -1)
            # Blue Bumper Base (lower 35%)
            cv2.rectangle(frame, (bx - 45, by + 12), (bx + 45, by + 35), (220, 45, 20), -1)
            cv2.putText(frame, f"B{b_idx+1}", (bx - 12, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Add 1 Human Referee Outside Guardrails (standing upright, high aspect ratio)
        ref_x = 120
        ref_y = 650
        cv2.rectangle(frame, (ref_x - 12, ref_y - 60), (ref_x + 12, ref_y + 60), (220, 220, 220), -1)
        # Black stripes
        for s in range(-50, 50, 15):
            cv2.rectangle(frame, (ref_x - 12, ref_y + s), (ref_x + 12, ref_y + s + 7), (10, 10, 10), -1)

        out.write(frame)

    out.release()
    logger.info("Demo video generated at: %s", output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FRC 2026 3-Stage Spatial Analytics Pipeline")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Input broadcast video file")
    parser.add_argument("--output", "-o", type=str, default="reports/spatial_pipeline_annotated.mp4", help="Output annotated video")
    parser.add_argument("--start", "-s", type=int, default=0, help="Starting frame number")
    parser.add_argument("--max-frames", "-f", type=int, default=200, help="Maximum frames to process")
    parser.add_argument("--stride", type=int, default=1, help="Frame step stride")
    parser.add_argument("--demo", action="store_true", help="Generate and run on synthetic demo video")
    parser.add_argument("--layout", type=str, choices=["broadcast", "topdown", "side_by_side"], default="side_by_side", help="Output video layout")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)

    input_video = args.video
    if args.demo or not os.path.isfile(input_video):
        demo_path = "reports/demo_input_fixture.mp4"
        input_video = create_demo_match_video(demo_path, num_frames=max(args.max_frames, 90))

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        logger.error("Could not open input video: %s", input_video)
        return 1

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info("Opened video '%s': %d frames at %.2f FPS", input_video, total_video_frames, fps)

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # Initialize Spatial Pipeline
    pipeline = SpatialAnalyticsPipeline(
        scale_px_per_inch=2.0,
        max_fuel_capacity=504,
        enable_cmc=True,
    )

    out_writer: Optional[cv2.VideoWriter] = None
    processed_count = 0
    t0 = time.perf_counter()

    sample_snapshot_saved = False
    snapshot_path = "reports/spatial_pipeline_snapshot.jpg"

    while cap.isOpened() and processed_count < args.max_frames:
        curr_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break

        if args.stride > 1 and (curr_frame_idx % args.stride != 0):
            continue

        ts_ms = int(curr_frame_idx * (1000.0 / fps))
        res = pipeline.process_frame(
            frame_bgr=frame,
            frame_index=curr_frame_idx,
            timestamp_ms=ts_ms,
            render_visualization=True,
        )

        vis_b = res.annotated_broadcast_frame
        vis_t = res.annotated_top_down_plane

        # Compose output frame based on requested layout
        if args.layout == "broadcast" and vis_b is not None:
            comp_frame = vis_b
        elif args.layout == "topdown" and vis_t is not None:
            comp_frame = vis_t
        else:
            # Side-by-side or picture-in-picture
            # Top-down is 1302 x 635, broadcast is 1920 x 1080
            # Resize broadcast to 1280 x 720 and top-down to 1280 x 624, stacked vertically or side-by-side
            b_resized = cv2.resize(vis_b, (960, 540)) if vis_b is not None else np.zeros((540, 960, 3), dtype=np.uint8)
            t_resized = cv2.resize(vis_t, (960, 540)) if vis_t is not None else np.zeros((540, 960, 3), dtype=np.uint8)
            comp_frame = np.hstack([b_resized, t_resized])

        # Initialize video writer on first frame
        if out_writer is None:
            h_out, w_out = comp_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.output, fourcc, fps / args.stride, (w_out, h_out))

        out_writer.write(comp_frame)

        if not sample_snapshot_saved:
            cv2.imwrite(snapshot_path, comp_frame)
            sample_snapshot_saved = True

        processed_count += 1
        if processed_count % 30 == 0:
            elapsed = time.perf_counter() - t0
            fps_proc = processed_count / max(1e-5, elapsed)
            logger.info(
                "Processed %d/%d frames (%.1f FPS) | Tags: %d | Fuel: %d/%d | Units: %d Red / %d Blue",
                processed_count, args.max_frames, fps_proc,
                len(res.calibration.detected_tags),
                res.game_pieces.total_estimated_count, res.game_pieces.max_field_capacity,
                len(res.robots.red_units), len(res.robots.blue_units),
            )

    cap.release()
    if out_writer is not None:
        out_writer.release()

    total_time = time.perf_counter() - t0
    logger.info(
        "Finished processing %d frames in %.2f s (%.1f FPS). Output video saved: %s",
        processed_count, total_time, processed_count / max(1e-5, total_time), args.output,
    )
    if sample_snapshot_saved:
        logger.info("Sample visualizer snapshot saved: %s", snapshot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
