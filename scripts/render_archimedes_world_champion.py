#!/usr/bin/env python3
"""Render Video Visualizing Vision Perception, Hub Scoring, and Real-Time Predictions

Executes visualizer on the real 2026 World Champion Archimedes division match,
rendering an annotated video displaying:
1. What the model is actively looking at:
   - 6-Robot detections with alliance colors, IDs, team numbers, and velocity vectors.
   - Real-time Possession State Machine (PSM) states (Intaking, Scoring, Transit, Idle).
   - Yellow FUEL ball detections and field Hub catchment zones.
   - Hub Scoring Intersections during score increments.
2. What the model is actively predicting to happen:
   - Real-Time XGBoost Win Probability meter (Red % vs Blue %).
   - 10,000 Monte Carlo rollouts projected final score with 95% Confidence Intervals.
   - Strategic recommendation (Cooperative Passing vs Direct Scoring, +delta-W).
   - Ground-truth comparison against the real 2026 World Champion match winner.

Usage:
    python scripts/render_archimedes_world_champion.py --video output.mp4 --output output_analytics_prediction_archemdies.mp4 --frames 600
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.tba_client import MatchMetadata
from src.analytics.visualizer import AnalyticsPredictionVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frc.render_archimedes")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render 2026 World Champion Archimedes Visualizer Video")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Input match video")
    parser.add_argument("--output", "-o", type=str, default="output_analytics_prediction_archemdies.mp4", help="Output annotated video")
    parser.add_argument("--weights", "-w", type=str, default="yolo11x_robot_best.pt", help="YOLO model weights")
    parser.add_argument("--frames", "-f", type=int, default=700, help="Frames to render")
    parser.add_argument("--start", "-s", type=int, default=400, help="Starting frame index")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride (process every Nth frame)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        logger.error("Input video file not found: %s", args.video)
        return 1

    # Real 2026 World Champion Archimedes Finals Match metadata:
    # Blue Alliance (World Champions): 5940 (BREAD), 9470 (Ctrl-Alt-Defeat), 3006 (Coogs of Calculus)
    # Red Alliance (Championship Finalists): 1323 (MadTown), 4065 (Nerds of the North), 4414 (HighTide)
    match_meta = MatchMetadata(
        match_key="2026arc_f1m1",
        event_key="2026arc",
        comp_level="f",
        match_number=1,
        red_teams=["1323", "4065", "4414"],
        blue_teams=["5940", "9470", "3006"],
        red_score=114,
        blue_score=126,
        winning_alliance="blue",  # 2026 World Champions
        score_breakdown={
            "red": {"autoPoints": 34, "teleopPoints": 62, "endgamePoints": 20, "fuelScored": 48},
            "blue": {"autoPoints": 38, "teleopPoints": 68, "endgamePoints": 20, "fuelScored": 54},
        },
        local_video_path=os.path.abspath(args.video),
    )

    logger.info("Initializing visualizer with YOLO weights: %s...", args.weights)
    visualizer = AnalyticsPredictionVisualizer(
        model_weights=args.weights,
        predictor_weights="weights/xgboost_match_predictor.json",
    )

    logger.info("Beginning video render for %s (%s vs %s)...", match_meta.match_key, match_meta.red_teams, match_meta.blue_teams)
    res = visualizer.render_prediction_video(
        input_video_path=args.video,
        output_video_path=args.output,
        match_metadata=match_meta,
        max_frames=args.frames,
        start_frame=args.start,
        frame_stride=args.stride,
    )

    # Save a keyframe snapshot image for inspection
    snapshot_path = "reports/archimedes_prediction_snapshot.jpg"
    os.makedirs("reports", exist_ok=True)
    import cv2
    cap = cv2.VideoCapture(args.output)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, res["frames_rendered"] - 1))
        ret, snap_frame = cap.read()
        if ret and snap_frame is not None:
            cv2.imwrite(snapshot_path, snap_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info("Saved keyframe snapshot to %s", snapshot_path)
        cap.release()

    print("\n" + "=" * 65)
    print(f"Render Complete: {res['output_video_path']}")
    print(f"File Size: {res['file_size_mb']:.2f} MB")
    print(f"Frames Rendered: {res['frames_rendered']}")
    print(f"Confirmed Fuel Scored Events: {res['confirmed_scored_balls']}")
    print(f"Match: {match_meta.match_key} (World Champion: {match_meta.winning_alliance.upper()})")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
