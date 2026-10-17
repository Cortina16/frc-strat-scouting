#!/usr/bin/env python3
"""CLI Entrypoint for FRC Video Analytics, Model Training, and Real Winner Validation.

Executes the end-to-end analytics workflow implementing analytics-implementation.md:
1. Harvests match schedules, video URLs, and outcomes from The Blue Alliance (TBA) API.
2. Watches match videos with YOLOv11x robot tracking and yellow FUEL ball detection.
3. Detects confirmed Hub scoring by intersecting ball detections with Hub boundaries when points increase.
4. Derives 12 spatial, kinematic, and tactical metrics across 5 match phases (90-dim feature vector).
5. Trains an XGBoost classifier and runs 10,000 Monte Carlo match rollouts.
6. Validates predicted outputs against real ground-truth match winners from TBA.

Usage:
    python scripts/run_analytics_flow.py --event 2024cmptx --max-matches 10 --video output.mp4
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

from src.analytics import AnalyticsFlowOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frc.cli.analytics_flow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FRC Video Analytics & Match Outcome Validation Flow")
    parser.add_argument("--event", "-e", type=str, default="2024cmptx", help="The Blue Alliance event key (e.g. 2024cmptx)")
    parser.add_argument("--match", "-M", "--match-key", type=str, default=None, help="Target specific match key (e.g. 2026casf_qm1)")
    parser.add_argument("--tba-key", "-k", type=str, default=None, help="The Blue Alliance API v3 key (or env TBA_API_KEY)")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Default fallback local video file")
    parser.add_argument("--weights", "-w", type=str, default="yolo11x_robot_best.pt", help="YOLO model weights path")
    parser.add_argument("--bumper-weights", type=str, default="weights/bumper_best.pt", help="Bumper YOLO model weights path")
    parser.add_argument("--no-bumper", action="store_true", default=False, help="Disable bumper model detection")
    parser.add_argument("--max-matches", "-m", type=int, default=10, help="Maximum matches to process")
    parser.add_argument("--max-frames", "-f", type=int, default=2000, help="Maximum frames to watch per video (0 or -1 for full video)")
    parser.add_argument("--cv-backend", type=str, choices=["spatial", "fast"], default="spatial", help="CV detection backend")
    parser.add_argument("--render-video", action="store_true", default=False, help="Render annotated prediction video")
    parser.add_argument("--video-output", type=str, default="reports/analytics_prediction_match.mp4", help="Annotated video output path")
    parser.add_argument("--layout", type=str, choices=["side_by_side", "broadcast"], default="side_by_side", help="Visualizer video layout")
    parser.add_argument("--render-frames", type=int, default=400, help="Maximum frames to render in visualizer video")
    parser.add_argument("--output-report", "-o", type=str, default="reports/validation_report.md", help="Path to save markdown report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Initializing Analytics Flow Orchestrator for event %s (backend: %s)...", args.event, args.cv_backend)

    orchestrator = AnalyticsFlowOrchestrator(
        tba_api_key=args.tba_key,
        model_weights_path=args.weights,
        bumper_model_path=args.bumper_weights,
        enable_bumper_model=not args.no_bumper,
        default_video_fallback=args.video,
        cv_backend=args.cv_backend,
    )

    try:
        flow_result = orchestrator.run_flow(
            event_key=args.event,
            match_key=args.match,
            max_matches=args.max_matches,
            max_frames_per_video=args.max_frames,
            render_video=args.render_video,
            video_output_path=args.video_output,
            render_layout=args.layout,
            render_frames=args.render_frames,
        )

        report_md = flow_result.validation_report.summary_markdown()
        print("\n" + "=" * 60)
        print(report_md)
        print("=" * 60 + "\n")

        # Save report
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Saved validation report to %s", out_path.resolve())

        if flow_result.rendered_video_path:
            logger.info("Annotated Prediction Video rendered to: %s", flow_result.rendered_video_path)
            print(f"[PREDICTION VIDEO GENERATED]: {flow_result.rendered_video_path}\n")

        return 0
    except Exception as e:
        logger.exception("Analytics Flow failed with error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
