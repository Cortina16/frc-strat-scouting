"""Match Data Provider for 3D Match Digital Twin.

Provides unified, high-performance in-memory and DuckDB access to match trajectories,
ball dynamics, localized action events, and field CAD specifications.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import yaml

logger = logging.getLogger(__name__)

# Locate project root directory relative to this file: src/server/data_provider.py -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class MatchDataProvider:
    """Telemetry and field CAD data provider supporting sub-millisecond in-memory scrubbing."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/parquet",
        config_path: Union[str, Path] = "configs/field_2026_rebuilt.yaml",
    ) -> None:
        raw_data_dir = Path(data_dir)
        if not raw_data_dir.is_absolute():
            candidate = PROJECT_ROOT / raw_data_dir
            self.data_dir = candidate if candidate.exists() else raw_data_dir.resolve()
        else:
            self.data_dir = raw_data_dir

        raw_cfg = Path(config_path)
        if not raw_cfg.is_absolute():
            candidate = PROJECT_ROOT / raw_cfg
            self.config_path = candidate if candidate.exists() else raw_cfg.resolve()
        else:
            self.config_path = raw_cfg

        self._con = duckdb.connect()
        self._frame_cache: Dict[str, Dict[str, Any]] = {}
        self._summary_cache: Dict[str, Dict[str, Any]] = {}
        self._cad_cache: Dict[str, Dict[str, Any]] = {}

    def _get_parquet_path(self, match_id: str, layer: str) -> Path:
        """Returns the path to a parquet file for a given match and layer."""
        cand1 = self.data_dir / f"match_{match_id}_{layer}.parquet"
        if cand1.exists():
            return cand1
        layer_map = {
            "trajectories": "robot_trajectories.parquet",
            "balls": "ball_trajectories.parquet",
            "actions": "action_events.parquet",
            "camera_poses": "camera_poses.parquet",
        }
        cand2 = self.data_dir / match_id / layer_map.get(layer, f"{layer}.parquet")
        if cand2.exists():
            return cand2
        cand3 = self.data_dir / layer_map.get(layer, f"{layer}.parquet")
        if cand3.exists():
            return cand3
        return cand1

    def get_available_matches(self) -> List[Dict[str, Any]]:
        """Scans data/parquet/ for match files matching match_*_trajectories.parquet.

        Returns a list of matches with match_id, frame counts, and available parquet layers.
        """
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            return []

        pattern = re.compile(r"^match_(.+)_trajectories\.parquet$")
        matches: List[Dict[str, Any]] = []

        for entry in self.data_dir.iterdir():
            if not entry.is_file():
                continue
            m = pattern.match(entry.name)
            if not m:
                continue

            match_id = m.group(1)

            # Discover all available layers for this match
            prefix = f"match_{match_id}_"
            layers: List[str] = []
            for file_path in self.data_dir.glob(f"{prefix}*.parquet"):
                suffix = file_path.name[len(prefix):-len(".parquet")]
                layers.append(suffix)
            layers.sort()

            # Query frame counts and frame bounds from trajectories
            traj_path = str(entry).replace("'", "''")
            try:
                res = self._con.execute(
                    f"SELECT COUNT(DISTINCT frame_index), MIN(frame_index), MAX(frame_index) "
                    f"FROM read_parquet('{traj_path}')"
                ).fetchone()
                frame_count = int(res[0]) if res and res[0] is not None else 0
                start_frame = int(res[1]) if res and res[1] is not None else 0
                end_frame = int(res[2]) if res and res[2] is not None else 0
            except Exception as e:
                logger.warning("Failed to query frame count for match %s: %s", match_id, e)
                frame_count = 0
                start_frame = 0
                end_frame = 0

            matches.append({
                "match_id": match_id,
                "frame_count": frame_count,
                "total_frames": frame_count,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "layers": layers,
                "available_layers": layers,
            })

        matches.sort(key=lambda item: item["match_id"])
        return matches

    def get_field_cad(
        self,
        config_path: str = "configs/field_2026_rebuilt.yaml",
    ) -> Dict[str, Any]:
        """Loads configs/field_2026_rebuilt.yaml and returns field dimensions, elements, and 32 AprilTags.

        Field dimensions: length 651.2, width 317.7, guardrail height 20.0 inches.
        Elements: Red Hub, Blue Hub, Towers, Depots, Trenches.
        AprilTags: All 32 AprilTags with (id, x, y, z, roll, pitch, yaw).
        """
        cfg_key = str(config_path)
        if cfg_key in self._cad_cache:
            return self._cad_cache[cfg_key]

        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cand = PROJECT_ROOT / cfg_file
            target_path = cand if cand.exists() else self.config_path
        else:
            target_path = cfg_file

        if not target_path.exists():
            raise FileNotFoundError(f"Field configuration file not found at {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        field_info = data.get("field", {})
        raw_elements = data.get("elements", {})
        raw_tags = data.get("apriltags", {}).get("tags", [])

        length = float(field_info.get("length_inches", 651.2))
        width = float(field_info.get("width_inches", 317.7))
        guardrail_height = float(field_info.get("guardrail_height_inches", 20.0))
        carpet_thickness = float(field_info.get("carpet_thickness_inches", 0.125))

        elements: Dict[str, Any] = {
            "red_hub": raw_elements.get("red_hub"),
            "blue_hub": raw_elements.get("blue_hub"),
            "towers": raw_elements.get("towers"),
            "depots": raw_elements.get("depots"),
            "trenches": raw_elements.get("trenches"),
            "bumps": raw_elements.get("bumps"),
            "Red Hub": raw_elements.get("red_hub"),
            "Blue Hub": raw_elements.get("blue_hub"),
            "Towers": raw_elements.get("towers"),
            "Depots": raw_elements.get("depots"),
            "Trenches": raw_elements.get("trenches"),
        }

        apriltags: List[Dict[str, Any]] = [
            {
                "id": int(t["id"]),
                "x": float(t["x"]),
                "y": float(t["y"]),
                "z": float(t["z"]),
                "roll": float(t.get("roll", 0.0)),
                "pitch": float(t.get("pitch", 0.0)),
                "yaw": float(t.get("yaw", 0.0)),
            }
            for t in raw_tags
        ]

        result: Dict[str, Any] = {
            "field": {
                "name": field_info.get("name", "FRC 2026 REBUILT"),
                "length": length,
                "width": width,
                "guardrail_height": guardrail_height,
                "length_inches": length,
                "width_inches": width,
                "guardrail_height_inches": guardrail_height,
                "carpet_thickness_inches": carpet_thickness,
            },
            "dimensions": {
                "length": length,
                "width": width,
                "guardrail_height": guardrail_height,
                "length_inches": length,
                "width_inches": width,
                "guardrail_height_inches": guardrail_height,
            },
            "length": length,
            "width": width,
            "guardrail_height": guardrail_height,
            "length_inches": length,
            "width_inches": width,
            "guardrail_height_inches": guardrail_height,
            "elements": elements,
            "apriltags": apriltags,
            "tag_count": len(apriltags),
            "family": data.get("apriltags", {}).get("family", "tag36h11"),
            "tag_size_inches": float(data.get("apriltags", {}).get("size_inches", 8.125)),
            "fuel": data.get("fuel", {
                "diameter_inches": 5.91,
                "radius_inches": 2.955,
                "mass_grams": 140.0,
                "carpet_center_elevation_inches": 2.955,
            }),
        }

        self._cad_cache[cfg_key] = result
        return result

    def get_match_summary(self, match_id: str) -> Dict[str, Any]:
        """Reads DuckDB parquet for the match, returns total frames, frame rate, duration, teams, and action summary."""
        if match_id in self._summary_cache:
            return self._summary_cache[match_id]

        traj_file = self._get_parquet_path(match_id, "trajectories")
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectories parquet not found for match '{match_id}' at {traj_file}")

        escaped_traj = str(traj_file).replace("'", "''")

        # 1. Trajectory aggregations (frames, teams)
        traj_query = f"""
        SELECT 
            COUNT(DISTINCT frame_index) AS total_frames,
            MIN(frame_index) AS start_frame,
            MAX(frame_index) AS end_frame
        FROM read_parquet('{escaped_traj}')
        """
        row = self._con.execute(traj_query).fetchone()
        total_frames = int(row[0]) if row and row[0] is not None else 0
        start_frame = int(row[1]) if row and row[1] is not None else 0
        end_frame = int(row[2]) if row and row[2] is not None else 0

        frame_rate = 60.0
        duration_seconds = round(float(total_frames) / frame_rate, 3) if frame_rate > 0 else 0.0

        # Query red and blue alliance teams
        teams_query = f"""
        SELECT DISTINCT alliance, team_number
        FROM read_parquet('{escaped_traj}')
        WHERE team_number > 0
        ORDER BY team_number ASC
        """
        team_rows = self._con.execute(teams_query).fetchall()
        red_teams = sorted({int(r[1]) for r in team_rows if str(r[0]).upper() == "RED"})
        blue_teams = sorted({int(r[1]) for r in team_rows if str(r[0]).upper() == "BLUE"})
        all_teams = sorted(set(red_teams).union(blue_teams))

        # 2. Action event summary counts
        actions_file = self._get_parquet_path(match_id, "actions")
        action_counts: Dict[str, int] = {}
        total_actions = 0

        if actions_file.exists():
            escaped_actions = str(actions_file).replace("'", "''")
            action_query = f"""
            SELECT action_type, COUNT(*) AS cnt
            FROM read_parquet('{escaped_actions}')
            GROUP BY action_type
            ORDER BY action_type ASC
            """
            for a_row in self._con.execute(action_query).fetchall():
                act_type = str(a_row[0])
                count = int(a_row[1])
                action_counts[act_type] = count
                total_actions += count

        summary: Dict[str, Any] = {
            "match_id": match_id,
            "total_frames": total_frames,
            "frame_rate": frame_rate,
            "duration_seconds": duration_seconds,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "red_teams": red_teams,
            "blue_teams": blue_teams,
            "teams": all_teams,
            "action_counts": action_counts,
            "action_summary": action_counts,
            "action_event_counts": action_counts,
            "action_event_summary": action_counts,
            "total_actions": total_actions,
            "action_events_count": total_actions,
        }

        self._summary_cache[match_id] = summary
        return summary

    def get_match_events(self, match_id: str) -> List[Dict[str, Any]]:
        """Reads match_{match_id}_actions.parquet.

        Returns list of reconciled events with:
        event_id, track_id, team_number, action_type, element, start_frame, end_frame,
        start_time_ms, end_time_ms, duration_s, confidence.
        """
        actions_file = self._get_parquet_path(match_id, "actions")
        if not actions_file.exists():
            raise FileNotFoundError(f"Actions parquet not found for match '{match_id}' at {actions_file}")

        escaped_actions = str(actions_file).replace("'", "''")

        # Check if is_reconciled column exists in schema
        cols = [
            c[0]
            for c in self._con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_actions}')").fetchall()
        ]
        has_reconciled = "is_reconciled" in cols

        where_clause = "WHERE is_reconciled = true" if has_reconciled else ""

        query = f"""
        SELECT 
            event_id,
            track_id,
            team_number,
            action_type,
            element,
            start_frame,
            end_frame,
            start_time_ms,
            end_time_ms,
            duration_s,
            confidence
        FROM read_parquet('{escaped_actions}')
        {where_clause}
        ORDER BY start_frame ASC, start_time_ms ASC
        """

        rows = self._con.execute(query).fetchall()
        events: List[Dict[str, Any]] = [
            {
                "event_id": str(r[0]),
                "track_id": int(r[1]),
                "team_number": int(r[2]),
                "action_type": str(r[3]),
                "element": str(r[4]),
                "start_frame": int(r[5]),
                "end_frame": int(r[6]),
                "start_time_ms": int(r[7]),
                "end_time_ms": int(r[8]),
                "duration_s": float(r[9]),
                "confidence": float(r[10]),
            }
            for r in rows
        ]
        return events

    def _ensure_frames_cached(self, match_id: str) -> None:
        """Loads and indexes trajectories and ball states in memory for match_id."""
        if match_id in self._frame_cache:
            return

        traj_file = self._get_parquet_path(match_id, "trajectories")
        if not traj_file.exists():
            raise FileNotFoundError(f"Trajectories parquet not found for match '{match_id}' at {traj_file}")

        escaped_traj = str(traj_file).replace("'", "''")

        # Inspect trajectory columns
        traj_cols = [
            c[0]
            for c in self._con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_traj}')").fetchall()
        ]
        fz_col = "field_z_inch" if "field_z_inch" in traj_cols else "0.0 AS field_z_inch"
        cr_x_col = "center_rel_x_inch" if "center_rel_x_inch" in traj_cols else "(field_x_inch - 325.6) AS center_rel_x_inch"
        cr_y_col = "center_rel_y_inch" if "center_rel_y_inch" in traj_cols else "(field_y_inch - 158.85) AS center_rel_y_inch"
        cr_z_col = "center_rel_z_inch" if "center_rel_z_inch" in traj_cols else "0.0 AS center_rel_z_inch"
        depth_col = "depth_inch" if "depth_inch" in traj_cols else "0.0 AS depth_inch"
        rw_col = "robot_width_inch" if "robot_width_inch" in traj_cols else "22.0 AS robot_width_inch"
        rl_col = "robot_length_inch" if "robot_length_inch" in traj_cols else "28.0 AS robot_length_inch"
        rh_col = "robot_height_inch" if "robot_height_inch" in traj_cols else "28.0 AS robot_height_inch"

        # Fetch all trajectories
        traj_sql = f"""
        SELECT 
            frame_index,
            timestamp_ms,
            track_id,
            team_number,
            alliance,
            field_x_inch,
            field_y_inch,
            heading_rad,
            speed_ips,
            in_defense,
            {fz_col},
            {cr_x_col},
            {cr_y_col},
            {cr_z_col},
            {depth_col},
            {rw_col},
            {rl_col},
            {rh_col}
        FROM read_parquet('{escaped_traj}')
        ORDER BY frame_index ASC, track_id ASC
        """
        traj_rows = self._con.execute(traj_sql).fetchall()

        traj_by_frame = defaultdict(list)
        frame_timestamps: Dict[int, int] = {}

        for r in traj_rows:
            f_idx = int(r[0])
            frame_timestamps[f_idx] = int(r[1])
            traj_by_frame[f_idx].append({
                "track_id": int(r[2]),
                "team_number": int(r[3]),
                "alliance": str(r[4]),
                "field_x_inch": float(r[5]),
                "field_y_inch": float(r[6]),
                "heading_rad": float(r[7]),
                "speed_ips": float(r[8]),
                "in_defense": bool(r[9]),
                "field_z_inch": float(r[10] or 0.0),
                "center_rel_x_inch": float(r[11] if r[11] is not None else (r[5] - 325.6)),
                "center_rel_y_inch": float(r[12] if r[12] is not None else (r[6] - 158.85)),
                "center_rel_z_inch": float(r[13] or 0.0),
                "depth_inch": float(r[14] or 0.0),
                "robot_width_inch": float(r[15] or 22.0),
                "robot_length_inch": float(r[16] or 28.0),
                "robot_height_inch": float(r[17] or 28.0),
            })

        # Fetch balls if present
        balls_file = self._get_parquet_path(match_id, "balls")
        balls_by_frame = defaultdict(list)

        if balls_file.exists():
            escaped_balls = str(balls_file).replace("'", "''")
            b_cols = [
                c[0]
                for c in self._con.execute(f"DESCRIBE SELECT * FROM read_parquet('{escaped_balls}')").fetchall()
            ]
            shooter_col = "shooter_team_number" if "shooter_team_number" in b_cols else "NULL AS shooter_team_number"
            bfz_col = "field_z_inch" if "field_z_inch" in b_cols else "2.955 AS field_z_inch"
            bcr_x_col = "center_rel_x_inch" if "center_rel_x_inch" in b_cols else "(field_x_inch - 325.6) AS center_rel_x_inch"
            bcr_y_col = "center_rel_y_inch" if "center_rel_y_inch" in b_cols else "(field_y_inch - 158.85) AS center_rel_y_inch"
            bcr_z_col = "center_rel_z_inch" if "center_rel_z_inch" in b_cols else "2.955 AS center_rel_z_inch"
            b_depth_col = "depth_inch" if "depth_inch" in b_cols else "0.0 AS depth_inch"

            balls_sql = f"""
            SELECT 
                frame_index,
                timestamp_ms,
                ball_id,
                field_x_inch,
                field_y_inch,
                velocity_x_ips,
                velocity_y_ips,
                ball_state,
                is_cluster,
                cluster_ball_count,
                {shooter_col},
                {bfz_col},
                {bcr_x_col},
                {bcr_y_col},
                {bcr_z_col},
                {b_depth_col}
            FROM read_parquet('{escaped_balls}')
            ORDER BY frame_index ASC, ball_id ASC
            """
            ball_rows = self._con.execute(balls_sql).fetchall()
            for b in ball_rows:
                f_idx = int(b[0])
                if f_idx not in frame_timestamps:
                    frame_timestamps[f_idx] = int(b[1])
                balls_by_frame[f_idx].append({
                    "ball_id": int(b[2]),
                    "field_x_inch": float(b[3]),
                    "field_y_inch": float(b[4]),
                    "velocity_x_ips": float(b[5]),
                    "velocity_y_ips": float(b[6]),
                    "ball_state": str(b[7]),
                    "is_cluster": bool(b[8]),
                    "cluster_ball_count": int(b[9]),
                    "shooter_team_number": int(b[10]) if b[10] is not None else None,
                    "field_z_inch": float(b[11] if b[11] is not None else 2.955),
                    "center_rel_x_inch": float(b[12] if b[12] is not None else (b[3] - 325.6)),
                    "center_rel_y_inch": float(b[13] if b[13] is not None else (b[4] - 158.85)),
                    "center_rel_z_inch": float(b[14] if b[14] is not None else 2.955),
                    "depth_inch": float(b[15] or 0.0),
                })

        # Fetch camera poses if present
        camera_file = self._get_parquet_path(match_id, "camera_poses")
        camera_by_frame: Dict[int, Any] = {}
        if camera_file.exists():
            escaped_cam = str(camera_file).replace("'", "''")
            cam_sql = f"""
            SELECT 
                frame_index,
                camera_x_inch,
                camera_y_inch,
                camera_z_inch,
                focal_length,
                reproj_rmse,
                pnp_converged,
                visible_tags,
                tag_distances
            FROM read_parquet('{escaped_cam}')
            """
            try:
                cam_rows = self._con.execute(cam_sql).fetchall()
                for c in cam_rows:
                    f_idx = int(c[0])
                    vis_tags = []
                    tag_dists = {}
                    try:
                        if c[7]:
                            vis_tags = json.loads(c[7])
                    except Exception:
                        pass
                    try:
                        if c[8]:
                            tag_dists = json.loads(c[8])
                    except Exception:
                        pass
                    camera_by_frame[f_idx] = {
                        "camera_center": [float(c[1]), float(c[2]), float(c[3])],
                        "focal_length": float(c[4]),
                        "reproj_rmse": float(c[5]),
                        "pnp_converged": bool(c[6]),
                        "visible_tags": vis_tags,
                        "tag_distances": tag_dists,
                    }
            except Exception as e:
                logger.debug("Failed reading camera_poses: %s", e)

        all_frame_indices = sorted(frame_timestamps.keys())
        frames_list: List[Dict[str, Any]] = [
            {
                "frame_index": f,
                "timestamp_ms": frame_timestamps[f],
                "robots": traj_by_frame.get(f, []),
                "balls": balls_by_frame.get(f, []),
                "camera_pose": camera_by_frame.get(f, None),
            }
            for f in all_frame_indices
        ]

        self._frame_cache[match_id] = {
            "frames": frames_list,
            "frame_map": {f["frame_index"]: f for f in frames_list},
        }

    def get_match_frames(
        self,
        match_id: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fast query over trajectories and balls parquet. Caches in memory for instant scrubbing.

        Returns:
            Dict containing:
            - frame_index: int (first frame index in range)
            - timestamp_ms: int (first frame timestamp in range)
            - robots: List of {track_id, team_number, alliance, field_x_inch, field_y_inch, heading_rad, speed_ips, in_defense}
            - balls: List of {ball_id, field_x_inch, field_y_inch, velocity_x_ips, velocity_y_ips, ball_state, is_cluster, cluster_ball_count, shooter_team_number}
            - frames: List of all matched frames in the queried slice
            - total_frames: int
        """
        self._ensure_frames_cached(match_id)
        cache_entry = self._frame_cache[match_id]
        all_frames: List[Dict[str, Any]] = cache_entry["frames"]

        # Filter by start_frame and end_frame
        filtered = all_frames
        if start_frame is not None:
            filtered = [f for f in filtered if f["frame_index"] >= start_frame]
        if end_frame is not None:
            filtered = [f for f in filtered if f["frame_index"] <= end_frame]

        if limit is not None and limit >= 0:
            filtered = filtered[:limit]

        first_frame = filtered[0] if filtered else {}

        return {
            "match_id": match_id,
            "frame_index": first_frame.get("frame_index"),
            "timestamp_ms": first_frame.get("timestamp_ms"),
            "robots": first_frame.get("robots", []),
            "balls": first_frame.get("balls", []),
            "frames": filtered,
            "total_frames": len(filtered),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

    def clear_cache(self, match_id: Optional[str] = None) -> None:
        """Clears in-memory caches, optionally for a specific match_id."""
        if match_id:
            self._frame_cache.pop(match_id, None)
            self._summary_cache.pop(match_id, None)
        else:
            self._frame_cache.clear()
            self._summary_cache.clear()
            self._cad_cache.clear()
