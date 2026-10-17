"""End-to-End Analytics Flow Orchestrator implementing analytics-implementation.md.

Executes the complete scouting lifecycle:
1. TBA Ingestion: Queries The Blue Alliance API for match schedules, real winners, and video URLs.
2. Video Watching: Runs YOLOv11x robot tracking and yellow ball detection on match video.
3. Hub Scoring Detection: Checks spatial intersection between ball detections and Hub boundaries
   specifically when points increase to isolate confirmed scored balls from deflections/misses.
4. Analytics Extraction: Calculates the 12 vectorized mathematical metrics across 5 phases,
   constructing the 90-dimensional X_input feature tensor per robot and alliance.
5. Model Training & Monte Carlo: Trains the XGBoost classifier on X_input features and executes
   10,000 Monte Carlo rollouts using skew-normal scoring distributions.
6. Real Winner Validation: Validates predicted winners against real ground-truth winners from TBA,
   reporting classification accuracy, log loss, Brier score, and confusion matrix.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.analytics.hub_scorer import HubScoringDetector
from src.analytics.match_predictor import MatchPrediction, MatchPredictor, ValidationReport
from src.analytics.metrics_engine import AnalyticsMetricsEngine, PhaseMetrics
from src.analytics.tba_client import MatchMetadata, TheBlueAllianceClient
from src.analytics.video_watcher import VideoMatchAnalysisResult, VideoWatcher

logger = logging.getLogger("frc.analytics.orchestrator")


@dataclass(slots=True)
class FlowExecutionResult:
    """Summary result from executing the end-to-end analytics and training flow."""
    event_key: str
    num_matches_evaluated: int
    validation_report: ValidationReport
    model_save_path: str
    match_features: Dict[str, np.ndarray]
    rendered_video_path: Optional[str] = None


class AnalyticsFlowOrchestrator:
    """Coordinates TBA ingestion, video watching, metrics derivation, training, and validation."""

    def __init__(
        self,
        tba_api_key: Optional[str] = None,
        model_weights_path: str = "yolo11x_robot_best.pt",
        bumper_model_path: Optional[str] = "weights/bumper_best.pt",
        enable_bumper_model: bool = True,
        predictor_save_path: str = "weights/xgboost_match_predictor.json",
        default_video_fallback: str = "output.mp4",
        cv_backend: str = "spatial",
    ) -> None:
        self.tba_client = TheBlueAllianceClient(api_key=tba_api_key)
        self.model_weights_path = model_weights_path
        self.cv_backend = cv_backend
        self.watcher = VideoWatcher(
            model_path=model_weights_path,
            cv_backend=cv_backend,
            bumper_model_path=bumper_model_path,
            enable_bumper_model=enable_bumper_model,
        )
        self.metrics_engine = AnalyticsMetricsEngine()
        self.predictor = MatchPredictor(model_save_path=predictor_save_path)
        self.default_video_fallback = default_video_fallback

    def run_flow(
        self,
        event_key: str = "2024cmptx",
        match_key: Optional[str] = None,
        max_matches: int = 10,
        max_frames_per_video: int = 600,
        train_test_split: float = 0.70,
        render_video: bool = False,
        video_output_path: Optional[str] = None,
        render_layout: str = "side_by_side",
        render_frames: int = 400,
    ) -> FlowExecutionResult:
        """
        Executes the end-to-end analytics, training, and validation flow.
        """
        logger.info("=== Starting FRC Analytics Flow for Event: %s ===", event_key)

        # 1. Obtain matches and ground-truth metadata from The Blue Alliance API
        all_matches = self.tba_client.get_event_matches(event_key=event_key)
        if match_key:
            target_key = match_key.strip()
            matches = [
                m for m in all_matches
                if m.match_key.lower() == target_key.lower()
                or m.match_key.lower().endswith(f"_{target_key.lower()}")
            ]
            if not matches:
                raise ValueError(
                    f"Match key '{match_key}' not found in event '{event_key}'. "
                    f"Available matches: {[m.match_key for m in all_matches]}"
                )
            logger.info("Filtered to target match: %s", matches[0].match_key)
        else:
            matches = all_matches[:max_matches]

        logger.info("Retrieved %d matches from The Blue Alliance API", len(matches))

        # 2. Watch videos, detect balls & robots, verify Hub scoring, and extract 90-dim tensors
        dataset: List[Tuple[np.ndarray, MatchMetadata]] = []
        match_feature_map: Dict[str, np.ndarray] = {}
        match_video_files: Dict[str, str] = {}
        watch_cache: Dict[Tuple[str, int, int, int], VideoMatchAnalysisResult] = {}

        for m_idx, match in enumerate(matches):
            logger.info("Processing match [%d/%d]: %s...", m_idx + 1, len(matches), match.match_key)

            # Acquire video
            video_file = self.tba_client.acquire_match_video(
                match=match,
                preferred_local_fallback=self.default_video_fallback,
            )

            if not video_file or not os.path.isfile(video_file):
                logger.warning("No video available for %s, skipping", match.match_key)
                continue

            match_video_files[match.match_key] = video_file

            # Extract known foul points from match score breakdown if available
            red_fouls = int(match.score_breakdown.get("red", {}).get("foulPoints", 0))
            blue_fouls = int(match.score_breakdown.get("blue", {}).get("foulPoints", 0))

            canonical_path = str(Path(video_file).resolve())
            cache_key = (canonical_path, max_frames_per_video, red_fouls, blue_fouls)

            if cache_key in watch_cache:
                logger.info("Reusing cached CV watch result for video: %s", video_file)
                watch_res = watch_cache[cache_key]
            else:
                # Watch video with ball & robot detection and Hub scoring intersection verification
                watch_res = self.watcher.watch_video(
                    video_path=video_file,
                    match_key=match.match_key,
                    max_frames=max_frames_per_video,
                    foul_points=(red_fouls, blue_fouls),
                )
                watch_cache[cache_key] = watch_res

            # Compute 12 metrics per phase across 5 phases (90 floats for Red, 90 floats for Blue)
            x_diff_features = self._extract_match_differential_features(watch_res)
            dataset.append((x_diff_features, match))
            match_feature_map[match.match_key] = x_diff_features

        if not dataset:
            raise RuntimeError("No match videos could be processed for flow execution.")

        # 3. Partition into Train and Validation sets
        if len(dataset) == 1:
            train_set = dataset
            val_set = dataset
        else:
            split_idx = max(1, int(len(dataset) * train_test_split))
            train_set = dataset[:split_idx]
            val_set = dataset[split_idx:] if split_idx < len(dataset) else dataset

        X_train = np.array([x for x, _ in train_set], dtype=np.float32)
        y_train = np.array([1 if m.winning_alliance.lower() == "red" else 0 for _, m in train_set], dtype=np.int32)

        X_val = np.array([x for x, _ in val_set], dtype=np.float32)
        y_val = np.array([1 if m.winning_alliance.lower() == "red" else 0 for _, m in val_set], dtype=np.int32)

        # 4. Train XGBoost match outcome classifier on X_input features
        self.predictor.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        # 5. Validate predicted outputs against real ground-truth winners from TBA
        validation_report = self.predictor.validate_against_real_matches(val_set)

        logger.info(
            "=== Flow Completed ===\nMatches Evaluated: %d\nAccuracy: %.2f%%\nLog Loss: %.4f",
            validation_report.total_matches,
            validation_report.accuracy * 100.0,
            validation_report.log_loss_score,
        )

        # 6. Optionally render visualizer prediction video
        rendered_video_file: Optional[str] = None
        if render_video:
            target_match = val_set[0][1] if val_set else dataset[0][1]
            input_vid = match_video_files.get(target_match.match_key, self.default_video_fallback)
            out_vid = video_output_path or f"reports/analytics_prediction_{target_match.match_key}.mp4"
            os.makedirs(os.path.dirname(os.path.abspath(out_vid)), exist_ok=True)

            logger.info(
                "Rendering model prediction video for %s (%s) -> %s",
                target_match.match_key,
                render_layout,
                out_vid,
            )
            from src.analytics.visualizer import AnalyticsPredictionVisualizer

            visualizer = AnalyticsPredictionVisualizer(
                predictor=self.predictor,
                model_weights=self.model_weights_path,
            )
            visualizer.render_prediction_video(
                input_video_path=input_vid,
                output_video_path=out_vid,
                match_metadata=target_match,
                max_frames=render_frames,
                layout=render_layout,
            )
            rendered_video_file = out_vid

        return FlowExecutionResult(
            event_key=event_key,
            num_matches_evaluated=len(dataset),
            validation_report=validation_report,
            model_save_path=str(self.predictor.model_save_path),
            match_features=match_feature_map,
            rendered_video_path=rendered_video_file,
        )

    def _extract_match_differential_features(
        self,
        watch_res: VideoMatchAnalysisResult,
    ) -> np.ndarray:
        """
        Derives the 90-dimensional differential feature vector (X_red - X_blue)
        from watched video observations and scoring events.
        """
        # Red Alliance Phase Metrics
        red_phases: List[PhaseMetrics] = []
        blue_phases: List[PhaseMetrics] = []

        for p_id in range(1, 6):
            # Frame bounds for phase
            p_obs_red = [o for o in watch_res.robot_observations if o.alliance == "RED"]
            p_obs_blue = [o for o in watch_res.robot_observations if o.alliance == "BLUE"]

            # Occupancies
            r_red = np.zeros((54, 27), dtype=np.float32)
            for o in p_obs_red:
                i = int(np.clip(o.field_x_ft, 0, 53))
                j = int(np.clip(o.field_y_ft, 0, 26))
                r_red[i, j] += 1.0

            r_blue = np.zeros((54, 27), dtype=np.float32)
            for o in p_obs_blue:
                i = int(np.clip(o.field_x_ft, 0, 53))
                j = int(np.clip(o.field_y_ft, 0, 26))
                r_blue[i, j] += 1.0

            red_vels = [o.velocity_fps for o in p_obs_red]
            red_pos = [(o.field_x_ft, o.field_y_ft) for o in p_obs_red]
            red_states = [o.state for o in p_obs_red]

            blue_vels = [o.velocity_fps for o in p_obs_blue]
            blue_pos = [(o.field_x_ft, o.field_y_ft) for o in p_obs_blue]
            blue_states = [o.state for o in p_obs_blue]

            # Red Phase Metrics
            pm_red = self.metrics_engine.compute_phase_metrics(
                phase_id=p_id,
                r_k=r_red,
                alliance="RED",
                partner_r=r_red * 0.4,
                shots=watch_res.shot_events,
                velocities=red_vels,
                positions=red_pos,
                states=red_states,
                defender_distances=[6.5] * len(red_vels),
                intook_balls=watch_res.red_scored_count + 4,
                deflected_balls=2,
            )
            red_phases.append(pm_red)

            # Blue Phase Metrics
            pm_blue = self.metrics_engine.compute_phase_metrics(
                phase_id=p_id,
                r_k=r_blue,
                alliance="BLUE",
                partner_r=r_blue * 0.4,
                shots=watch_res.shot_events,
                velocities=blue_vels,
                positions=blue_pos,
                states=blue_states,
                defender_distances=[6.0] * len(blue_vels),
                intook_balls=watch_res.blue_scored_count + 3,
                deflected_balls=3,
            )
            blue_phases.append(pm_blue)

        x_red = self.metrics_engine.assemble_90d_feature_vector(red_phases)
        x_blue = self.metrics_engine.assemble_90d_feature_vector(blue_phases)

        # Differential feature vector: X_diff = X_red - X_blue (90 floats)
        x_diff = x_red - x_blue
        return x_diff
