"""
FRC 2026 REBUILT: Automated Video Scouting Pipeline CLI Entrypoint.

Provides a unified command-line interface to execute the video scouting pipeline
either in distributed Ray Core mode (production) or local in-process mode (development).

Usage Examples:
    # 1. Run local in-process pipeline on a video file
    python -m src.main --stream videos/match_42.mp4 --match-id 2026casf_qm42 --no-ray

    # 2. Run distributed Ray supercluster pipeline with hardware slicing
    python -m src.main --stream rtmp://broadcast.venue/live/field1 --match-id 2026casf_qm42 --use-ray

    # 3. Run with synthetic fixture generation (for immediate verification)
    python -m src.main --demo --match-id 2026casf_demo
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.pipeline.orchestrator import PipelineOrchestrator
from src.transport.mqtt_publisher import MAX_PAYLOAD_SIZE_BYTES

logger = logging.getLogger("frc.scouting.main")


def setup_logging(level: str = "INFO") -> None:
    """Configures structured terminal logging with timestamp and component names."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_demo_synthetic_video(output_path: str, num_frames: int = 120, fps: float = 60.0) -> str:
    """Generates a temporary synthetic broadcast video with AprilTags and moving robot bumpers."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_images = {
        tid: cv2.cvtColor(cv2.aruco.generateImageMarker(dict_tag, tid, 60), cv2.COLOR_GRAY2BGR)
        for tid in [5, 6, 7, 8]
    }

    for i in range(num_frames):
        frame = np.full((480, 640, 3), (35, 120, 35), dtype=np.uint8)
        cv2.line(frame, (30, 240), (610, 240), (255, 255, 255), 3)

        # 4 perimeter AprilTags
        frame[50:110, 50:110] = tag_images[5]
        frame[50:110, 530:590] = tag_images[6]
        frame[370:430, 530:590] = tag_images[7]
        frame[370:430, 50:110] = tag_images[8]

        # Robot 1 (Red 254)
        rx1 = int(120 + (i / float(num_frames)) * 180)
        ry1 = int(180 + np.sin(i * 0.1) * 20)
        cv2.rectangle(frame, (rx1, ry1), (rx1 + 70, ry1 + 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (rx1, ry1 + 45), (rx1 + 70, ry1 + 70), (20, 30, 200), -1)
        cv2.putText(frame, "254", (rx1 + 10, ry1 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Robot 2 (Blue 1678)
        rx2 = int(450 - (i / float(num_frames)) * 180)
        ry2 = int(280 + np.cos(i * 0.1) * 20)
        cv2.rectangle(frame, (rx2, ry2), (rx2 + 70, ry2 + 70), (50, 50, 50), -1)
        cv2.rectangle(frame, (rx2, ry2 + 45), (rx2 + 70, ry2 + 70), (180, 40, 20), -1)
        cv2.putText(frame, "1678", (rx2 + 5, ry2 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    return output_path


async def run_pipeline(
    stream_uri: str,
    match_id: str,
    use_ray: bool = True,
    num_cpus: int = 64,
    num_gpus: int = 2,
    batch_size: int = 8,
    backpressure_threshold: int = 16,
    mqtt_host: str = "localhost",
    mqtt_port: int = 1883,
    mqtt_mock_mode: bool = False,
    export_dir: Optional[str] = None,
    config_path: str = "configs/pipeline.yaml",
    max_frames: Optional[int] = None,
    detector_engine_path: Optional[str] = None,
    red_teams: Optional[Sequence[int]] = None,
    blue_teams: Optional[Sequence[int]] = None,
    render_video: bool = False,
) -> Dict[str, Any]:
    """
    Executes the end-to-end scouting pipeline lifecycle.
    """
    logger.info("Initializing FRC 2026 Scouting Pipeline Orchestrator...")
    logger.info("Target Match ID: %s", match_id)
    logger.info("Stream URI: %s", stream_uri)
    logger.info("Execution Mode: %s", "Distributed Ray Core" if use_ray else "Local In-Process")
    logger.info("MQTT Broker: %s:%d (mock=%s)", mqtt_host, mqtt_port, mqtt_mock_mode)
    if detector_engine_path:
        logger.info("Detector Weights: %s", detector_engine_path)
    if max_frames:
        logger.info("Max Frames Limit: %d", max_frames)
    if red_teams:
        logger.info("Red Alliance Roster: %s", list(red_teams))
    if blue_teams:
        logger.info("Blue Alliance Roster: %s", list(blue_teams))

    orchestrator = PipelineOrchestrator(
        use_ray=use_ray,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        batch_size=batch_size,
        backpressure_threshold=backpressure_threshold,
        mqtt_broker_url=mqtt_host,
        mqtt_port=mqtt_port,
        mqtt_mock_mode=mqtt_mock_mode,
        parquet_export_dir=export_dir,
        config_path=config_path,
        max_frames=max_frames,
        detector_engine_path=detector_engine_path,
        red_teams=red_teams,
        blue_teams=blue_teams,
    )

    t_start = time.perf_counter()

    try:
        # Step 1: Start pipeline and actors
        logger.info("[1/3] Starting pipeline DAG and connecting transport...")
        await orchestrator.start_pipeline(stream_uri=stream_uri, match_id=match_id)

        # Step 2: Stream and process video frames
        logger.info("[2/3] Processing video stream dataflow with pipelining...")
        await orchestrator.process_match_stream()

        # Step 3: Flush analytics, publish sync payload, and teardown
        logger.info("[3/3] Finalizing match analytics and publishing sync payload...")
        summary = await orchestrator.stop_pipeline()

    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        logger.warning("Pipeline terminating or interrupted (%s): stopping gracefully...", exc)
        summary = await orchestrator.stop_pipeline()

    total_time = time.perf_counter() - t_start
    summary["wall_clock_time_s"] = total_time

    # Pretty-print match results
    print("\n" + "=" * 78)
    print(f" MATCH SCOUTING EXECUTION SUMMARY: {match_id}")
    print("=" * 78)
    print(f" Status:                     {summary.get('status', 'N/A').upper()}")
    print(f" Total Frames Ingested:      {summary.get('total_frames_processed', 0):,}")
    print(f" Trajectories Stored:        {summary.get('total_trajectories_inserted', 0):,}")
    reconciled_actions = summary.get("total_reconciled_actions", summary.get("total_actions_reconciled", 0))
    print(f" Reconciled Action Events:   {reconciled_actions:,}")
    print(f" Live Trajectory Packets:    {summary.get('published_trajectory_packets', 0):,}")
    print(f" Match Sync Dispatched:      {summary.get('sync_published', False)}")
    sync_bytes = summary.get("sync_payload_bytes", 0)
    print(f" Sync Wire Size:             {sync_bytes:,} bytes ({sync_bytes / 1024:.2f} KB / max 50 KB)")
    proc_fps = summary.get("processing_fps", 0.0)
    print(f" Processing FPS:             {proc_fps:.2f} FPS")
    print(f" Total Turnaround Latency:   {total_time:.2f} seconds")
    print("=" * 78)

    # Execute sample ASOF join query for telemetry inspection on top active teams
    sorted_metrics = sorted(
        summary.get("summary_metrics", []),
        key=lambda m: (m.get("reconciled_scores_count", 0), m.get("total_trajectory_points", 0)),
        reverse=True,
    )
    teams_to_query = [int(m["team_number"]) for m in sorted_metrics if m.get("team_number")]
    if not teams_to_query:
        teams_to_query = [254]
    elif 254 not in teams_to_query[:4]:
        teams_to_query = teams_to_query[:3] + [254]
    else:
        teams_to_query = teams_to_query[:4]

    for t_num in teams_to_query:
        try:
            sample_asof = await orchestrator.execute_asof_join_query(match_id=match_id, team_number=t_num)
            print(f"\n[DuckDB OLAP] Sample ASOF Join Query for Team {t_num}:")
            print(f"  Query Latency: {sample_asof.get('query_latency_ms', 0.0):.2f} ms")
            print(f"  Matched Events: {sample_asof.get('event_count', 0)}")
        except Exception as q_err:
            logger.warning("Could not execute sample ASOF query for team %s: %s", t_num, q_err)

    if render_video and os.path.exists(stream_uri):
        try:
            logger.info("\n[Annotated Video] Rendering match video overlay for %s...", match_id)
            from src.render import render_annotated_video
            out_v_name = f"annotated_match_{match_id}.mp4"
            out_v_path = os.path.join(export_dir or "data/parquet", out_v_name)
            target_traj = os.path.join(export_dir or "data/parquet", f"match_{match_id}_trajectories.parquet")
            traj_input = target_traj if os.path.exists(target_traj) else (export_dir or "data/parquet")
            render_annotated_video(
                video_path=stream_uri,
                parquet_path=traj_input,
                output_path=out_v_path,
                max_frames=max_frames,
            )
            print(f"[Annotated Video] Successfully exported: {out_v_path}")
            summary["annotated_video_path"] = out_v_path
        except Exception as r_err:
            logger.warning("Could not render annotated video: %s", r_err)

    return summary


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="FRC 2026 REBUILT Automated Video Scouting Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stream", "-s",
        type=str,
        default=None,
        help="Path to video file (.mp4, .mkv) or live stream URI (rtmp://, rtsp://, http://).",
    )
    parser.add_argument(
        "--match-id", "-m",
        type=str,
        default="2026casf_qm01",
        help="Official FRC match identifier (e.g., 2026casf_qm42).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a synthetic 120-frame broadcast video fixture and run end-to-end.",
    )
    parser.add_argument(
        "--use-ray",
        dest="use_ray",
        action="store_true",
        default=True,
        help="Execute on distributed Ray Core actors across CPU and GPU hardware slices.",
    )
    parser.add_argument(
        "--no-ray",
        dest="use_ray",
        action="store_false",
        help="Execute in local single-process mode without starting a Ray cluster.",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=64,
        help="Number of CPU cores allocated for Ray actors across Threadripper pool.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs allocated for Ray actors (GPU 0: Detection, GPU 1: ActionFormer).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for pipelined GPU inference and frame decoding.",
    )
    parser.add_argument(
        "--backpressure",
        type=int,
        default=16,
        help="Maximum in-flight detection queue depth before pausing stream ingestion.",
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default="localhost",
        help="Hostname or IP of the venue MQTT broker.",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="Port of the venue MQTT broker.",
    )
    parser.add_argument(
        "--mqtt-mock",
        action="store_true",
        help="Run MQTT telemetry publisher in mock/loopback mode.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="data/parquet",
        help="Directory to save persistent match Parquet archive files.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to pipeline configuration YAML file.",
    )
    parser.add_argument(
        "--detector-weights", "--model",
        type=str,
        default="data/models/medium_robo_v3.pt" if os.path.isfile("data/models/medium_robo_v3.pt") else None,
        help="Path to FRC robot detection model weights (.pt, .onnx, .engine).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum gameplay frames to process before finalizing (None = entire stream).",
    )
    parser.add_argument(
        "--red-teams",
        type=str,
        default=None,
        help="Comma-separated list of Red alliance team numbers (e.g. 1323,4065,4414).",
    )
    parser.add_argument(
        "--blue-teams",
        type=str,
        default=None,
        help="Comma-separated list of Blue alliance team numbers (e.g. 868,2046,2910).",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Automatically render annotated video overlaying robot tracking and balls upon completion.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(args)


def main(args: Optional[list[str]] = None) -> None:
    """CLI main entrypoint."""
    raw_args = list(sys.argv[1:]) if args is None else list(args)
    if raw_args and raw_args[0] == "render":
        from src.render import main as render_main
        sys.exit(render_main(raw_args[1:]))
    if raw_args and raw_args[0] == "web":
        from src.server.__main__ import main as server_main
        sys.exit(server_main(raw_args[1:]))

    args_parsed = parse_args(raw_args)
    setup_logging(args_parsed.log_level)

    temp_video_path: Optional[str] = None
    detector_weights = args_parsed.detector_weights

    red_teams = [int(x.strip()) for x in args_parsed.red_teams.split(",") if x.strip()] if args_parsed.red_teams else None
    blue_teams = [int(x.strip()) for x in args_parsed.blue_teams.split(",") if x.strip()] if args_parsed.blue_teams else None

    if args_parsed.demo:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            temp_video_path = tf.name
        logger.info("Generating synthetic 120-frame broadcast video at: %s", temp_video_path)
        stream_uri = create_demo_synthetic_video(temp_video_path, num_frames=120, fps=60.0)
        if not any(a in raw_args for a in ("--detector-weights", "--model")):
            detector_weights = None
    elif args_parsed.stream:
        stream_uri = args_parsed.stream
    else:
        logger.error("Must provide either --stream <path_or_url> or --demo flag. See --help.")
        sys.exit(1)

    try:
        asyncio.run(
            run_pipeline(
                stream_uri=stream_uri,
                match_id=args_parsed.match_id,
                use_ray=args_parsed.use_ray,
                num_cpus=args_parsed.num_cpus,
                num_gpus=args_parsed.num_gpus,
                batch_size=args_parsed.batch_size,
                backpressure_threshold=args_parsed.backpressure,
                mqtt_host=args_parsed.mqtt_host,
                mqtt_port=args_parsed.mqtt_port,
                mqtt_mock_mode=args_parsed.mqtt_mock,
                export_dir=args_parsed.export_dir,
                config_path=args_parsed.config,
                max_frames=args_parsed.max_frames,
                detector_engine_path=detector_weights,
                red_teams=red_teams,
                blue_teams=blue_teams,
                render_video=args_parsed.render_video,
            )
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
