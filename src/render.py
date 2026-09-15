"""
Video Annotation Rendering Module for FRC 2026 REBUILT Scouting Pipeline.

Provides high-performance video annotation overlaying robot detections, tracking IDs,
team numbers, alliance colors, kinematics, field coordinates, action recognition events,
and telemetry HUD from DuckDB Parquet stores.

Usage:
    python -m src.render --video sample.mp4 --parquet data/parquet/match_2026cmptx_f1m3_trajectories.parquet --output output_annotated.mp4
    python -m src.main render -v sample.mp4 -p data/parquet/ --max-frames 150
"""

from __future__ import annotations

import argparse
import collections
import glob
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import duckdb
import numpy as np

logger = logging.getLogger("frc.scouting.render")

# Alliance BGR colors
ALLIANCE_COLORS = {
    "RED": (30, 30, 225),       # Vibrant Red
    "BLUE": (225, 115, 25),     # Vibrant Blue
    "UNKNOWN": (180, 180, 180), # Neutral Silver Gray
}


def resolve_parquet_paths(parquet_arg: str) -> Tuple[str, Optional[str]]:
    """
    Resolves trajectories and optional action events Parquet file paths.

    Args:
        parquet_arg: Path to a Parquet file or a directory containing Parquet exports.

    Returns:
        Tuple of (trajectories_parquet_path, actions_parquet_path).
    """
    if not os.path.exists(parquet_arg):
        raise FileNotFoundError(f"Parquet path does not exist: {parquet_arg}")

    traj_path: Optional[str] = None
    actions_path: Optional[str] = None

    if os.path.isdir(parquet_arg):
        # Search directory for trajectories parquet
        traj_candidates = glob.glob(os.path.join(parquet_arg, "*trajectories*.parquet"))
        if not traj_candidates:
            traj_candidates = glob.glob(os.path.join(parquet_arg, "robot_trajectories.parquet"))
        if not traj_candidates:
            # Fallback to any parquet file in dir
            all_parquets = glob.glob(os.path.join(parquet_arg, "*.parquet"))
            if all_parquets:
                traj_candidates = [all_parquets[0]]

        if traj_candidates:
            traj_candidates.sort(key=os.path.getmtime, reverse=True)
            traj_path = traj_candidates[0]

        # Check for companion actions file corresponding to selected trajectories file
        if traj_path and "_trajectories.parquet" in traj_path:
            companion_act = traj_path.replace("_trajectories.parquet", "_actions.parquet")
            if os.path.isfile(companion_act):
                actions_path = companion_act

        if not actions_path:
            # Search directory for actions parquet
            act_candidates = glob.glob(os.path.join(parquet_arg, "*actions*.parquet"))
            if not act_candidates:
                act_candidates = glob.glob(os.path.join(parquet_arg, "action_events.parquet"))
            if act_candidates:
                act_candidates.sort(key=os.path.getmtime, reverse=True)
                actions_path = act_candidates[0]

    elif os.path.isfile(parquet_arg):
        traj_path = parquet_arg
        dir_name = os.path.dirname(os.path.abspath(parquet_arg))

        # Check if companion actions file exists
        if "_trajectories.parquet" in parquet_arg:
            candidate_act = parquet_arg.replace("_trajectories.parquet", "_actions.parquet")
            if os.path.isfile(candidate_act):
                actions_path = candidate_act
        if not actions_path:
            alt_act = os.path.join(dir_name, "action_events.parquet")
            if os.path.isfile(alt_act):
                actions_path = alt_act

    if not traj_path or not os.path.isfile(traj_path):
        raise FileNotFoundError(f"Could not locate valid trajectories Parquet file from: {parquet_arg}")

    logger.info("Resolved Parquet paths: Trajectories=%s, Actions=%s", traj_path, actions_path)
    return traj_path, actions_path


def load_parquet_data(
    traj_path: str,
    actions_path: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[int, Dict[str, Any]]], str]:
    """
    Loads trajectories and actions from Parquet files via DuckDB.

    Returns:
        Tuple of (
            trajectories_by_frame: {frame_index: [robot_records]},
            actions_by_frame_and_track: {frame_index: {track_id: action_record}},
            match_id: str
        )
    """
    con = duckdb.connect()
    match_id = "unknown_match"

    # 1. Query Trajectories
    sql_traj = f"""
    SELECT 
        match_id,
        frame_index,
        timestamp_ms,
        track_id,
        team_number,
        alliance,
        pixel_center_x,
        pixel_center_y,
        pixel_width,
        pixel_height,
        pixel_confidence,
        field_x_inch,
        field_y_inch,
        velocity_x_ips,
        velocity_y_ips,
        speed_ips,
        heading_rad,
        track_state,
        in_defense
    FROM read_parquet('{traj_path}')
    """
    if max_frames is not None:
        sql_traj += f" WHERE frame_index < {max_frames}"
    sql_traj += " ORDER BY frame_index, track_id"

    traj_cursor = con.execute(sql_traj)
    cols = [d[0] for d in traj_cursor.description]
    rows = traj_cursor.fetchall()

    trajectories_by_frame: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        rec = dict(zip(cols, r))
        f_idx = int(rec["frame_index"])
        trajectories_by_frame[f_idx].append(rec)
        if match_id == "unknown_match" and rec.get("match_id"):
            match_id = str(rec["match_id"])

    # Fallback match_id from filename if still unknown
    if match_id == "unknown_match":
        base_name = os.path.basename(traj_path)
        if base_name.startswith("match_"):
            parts = base_name.split("_")
            if len(parts) >= 3:
                match_id = f"{parts[1]}_{parts[2]}"

    # 2. Query Action Events (if present)
    actions_by_frame_and_track: Dict[int, Dict[int, Dict[str, Any]]] = collections.defaultdict(dict)
    if actions_path and os.path.isfile(actions_path):
        try:
            sql_act = f"""
            SELECT
                match_id,
                track_id,
                team_number,
                action_type,
                element,
                start_frame,
                end_frame,
                confidence
            FROM read_parquet('{actions_path}')
            """
            act_cursor = con.execute(sql_act)
            act_cols = [d[0] for d in act_cursor.description]
            act_rows = act_cursor.fetchall()

            for ar in act_rows:
                act_rec = dict(zip(act_cols, ar))
                s_frame = int(act_rec.get("start_frame") or 0)
                e_frame = int(act_rec.get("end_frame") or s_frame)
                t_id = int(act_rec.get("track_id") or 0)
                for f in range(s_frame, e_frame + 1):
                    if max_frames is not None and f >= max_frames:
                        break
                    actions_by_frame_and_track[f][t_id] = act_rec
            logger.info("Loaded %d action event records from %s", len(act_rows), actions_path)
        except Exception as exc:
            logger.warning("Could not read action events from %s: %s", actions_path, exc)

    con.close()
    logger.info("Loaded %d trajectory records across %d frames (Match ID: %s)", len(rows), len(trajectories_by_frame), match_id)
    return trajectories_by_frame, actions_by_frame_and_track, match_id


def load_ball_parquet_data(
    balls_path: str,
    max_frames: Optional[int] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Loads ball trajectories from Parquet via DuckDB."""
    if not balls_path or not os.path.isfile(balls_path):
        return {}
    con = duckdb.connect()
    balls_by_frame: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    try:
        where_clause = f" WHERE frame_index < {max_frames}" if max_frames is not None else ""
        sql_balls = f"SELECT * FROM read_parquet('{balls_path}'){where_clause} ORDER BY frame_index, ball_id"

        cursor = con.execute(sql_balls)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        for r in rows:
            rec = dict(zip(cols, r))
            balls_by_frame[int(rec["frame_index"])].append(rec)
        logger.info("Loaded %d ball trajectory records across %d frames", len(rows), len(balls_by_frame))
    except Exception as exc:
        logger.debug("Could not load ball parquet data: %s", exc)
    finally:
        con.close()
    return balls_by_frame


def draw_ball_annotations(
    frame: np.ndarray,
    ball_records: List[Dict[str, Any]],
) -> None:
    """Renders sleek antialiased glowing circles for individual balls and capsules for ambient clusters."""
    h_img, w_img = frame.shape[:2]
    for b in ball_records:
        cx = int(round(b["pixel_center_x"]))
        cy = int(round(b["pixel_center_y"]))
        is_cluster = bool(b.get("is_cluster", False))

        if is_cluster:
            # Ambient carpet cluster: render subtle rounded boundary capsule and count badge
            cluster_count = int(b.get("cluster_ball_count", 1))
            pct = (cluster_count / 504.0) * 100.0
            r = min(40, max(8, int(round(b.get("pixel_radius", 10.0)))))

            # Subtle amber/gold contour ring
            cv2.circle(frame, (cx, cy), r, (0, 200, 240), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (0, 165, 255), -1)

            # Badge text
            badge_text = f"~{cluster_count}b ({pct:.1f}%)"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            bx1 = max(0, min(w_img - tw - 8, cx - tw // 2 - 3))
            by1 = max(0, min(h_img - th - 6, cy - r - th - 4))
            cv2.rectangle(frame, (bx1, by1), (bx1 + tw + 6, by1 + th + 4), (20, 20, 25), -1)
            cv2.putText(
                frame,
                badge_text,
                (bx1 + 3, by1 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 220, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        else:
            # Focal Zone: discrete individual ball
            r = min(14, max(3, int(round(b.get("pixel_radius", 5.0)))))
            state = str(b.get("ball_state", "ON_CARPET"))

            fz = float(b.get("field_z_inch", 2.955) or 2.955)
            if state == "IN_FLIGHT" or fz > 4.5:
                # Cyan for in-flight projectile balls
                color = (255, 255, 0)
                thickness = 2
                cv2.circle(frame, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"Z:{fz:.0f}\"",
                    (cx + r + 2, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            elif state == "INTAKED":
                # Vivid orange for intaked / robot-occluded balls
                color = (0, 140, 255)
                thickness = 1
                cv2.circle(frame, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)
            elif state == "LOST":
                # Amber for momentarily coasting occluded balls
                color = (0, 190, 230)
                thickness = 1
                cv2.circle(frame, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)
            else:
                # Bright yellow-gold for carpet balls
                color = (0, 230, 255)
                thickness = 1
                cv2.circle(frame, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)


def draw_hud(
    frame: np.ndarray,
    match_id: str,
    frame_index: int,
    total_frames: int,
    pts_ms: int,
    active_robots_count: int,
    active_balls_count: Optional[int] = None,
    focal_balls_count: Optional[int] = None,
    ambient_balls_count: Optional[int] = None,
) -> None:
    """Draws top-left HUD overlay with match status, frame index, PTS, and robot count."""
    h_img, w_img = frame.shape[:2]
    pts_sec = pts_ms / 1000.0

    hud_lines = [
        f"Match: {match_id}",
        f"Frame: {frame_index:,} / {total_frames:,}",
        f"PTS:   {pts_ms} ms ({pts_sec:.2f}s)",
        f"Robots: {active_robots_count} active",
    ]
    if active_balls_count is not None and active_balls_count > 0:
        if focal_balls_count is not None and ambient_balls_count is not None:
            hud_lines.append(f"Balls:  {active_balls_count} / 504 ({focal_balls_count} @ robots, {ambient_balls_count} in piles)")
        else:
            hud_lines.append(f"Balls:  {active_balls_count} / 504 tracked")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.44
    thickness = 1
    line_height = 18
    pad_x = 10
    pad_y = 8

    max_w = 0
    for line in hud_lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if tw > max_w:
            max_w = tw

    hud_w = max_w + pad_x * 2
    hud_h = len(hud_lines) * line_height + pad_y * 2

    # Draw semi-transparent HUD background
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + hud_w, 12 + hud_h), (14, 14, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Draw HUD accent border
    cv2.rectangle(frame, (12, 12), (12 + hud_w, 12 + hud_h), (70, 70, 80), 1)
    cv2.line(frame, (12, 12), (12, 12 + hud_h), (0, 210, 255), 3)

    # Render HUD text
    for i, line in enumerate(hud_lines):
        y = 12 + pad_y + (i + 1) * line_height - 4
        # Accent for Match header
        color = (0, 220, 255) if i == 0 else (240, 240, 240)
        cv2.putText(frame, line, (12 + pad_x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_robot_annotation(
    frame: np.ndarray,
    robot: Dict[str, Any],
    action: Optional[Dict[str, Any]] = None,
    draw_field_coords: bool = False,
) -> None:
    """Draws bounding box with alliance colors and high-contrast label pill."""
    h_img, w_img = frame.shape[:2]

    cx = float(robot.get("pixel_center_x", 0.0))
    cy = float(robot.get("pixel_center_y", 0.0))
    w = float(robot.get("pixel_width", 0.0))
    h = float(robot.get("pixel_height", 0.0))

    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    # Clamp coordinates to frame boundaries
    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    alliance = str(robot.get("alliance", "UNKNOWN")).upper()
    box_color = ALLIANCE_COLORS.get(alliance, ALLIANCE_COLORS["UNKNOWN"])

    # 1. Draw Main Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 2. Draw Corner Accents for high-tech HUD look
    corner_len = max(5, min(14, int(min(w, h) * 0.25)))
    c_thick = 3
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, c_thick)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, c_thick)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, c_thick)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, c_thick)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, c_thick)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, c_thick)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, c_thick)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, c_thick)

    # 3. Construct Label Lines
    team_num = robot.get("team_number")
    team_str = f"Team {team_num}" if (team_num is not None and team_num > 0) else f"Track #{robot.get('track_id', '?')}"
    conf = float(robot.get("pixel_confidence", 0.0) or 0.0)
    line1 = f"{team_str} [{alliance}] {conf:.2f}"

    speed = float(robot.get("speed_ips", 0.0) or 0.0)
    in_defense = bool(robot.get("in_defense", False))
    line2 = f"Speed: {speed:.1f} ips"
    if in_defense:
        line2 += " | DEFENSE"

    label_lines = [line1, line2]

    # Action if matched
    if action and action.get("action_type"):
        act_name = str(action["action_type"]).replace("_", " ")
        label_lines.append(f"Action: {act_name}")

    # Field coordinates if requested
    if draw_field_coords:
        fx = robot.get("field_x_inch")
        fy = robot.get("field_y_inch")
        if fx is not None and fy is not None:
            label_lines.append(f"Field: ({float(fx):.1f}, {float(fy):.1f}) in")

    # 4. Measure Label Dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness = 1
    line_height = 15
    pad_x = 7
    pad_y = 5

    max_line_w = 0
    for l_text in label_lines:
        (tw, th), _ = cv2.getTextSize(l_text, font, font_scale, thickness)
        if tw > max_line_w:
            max_line_w = tw

    pill_w = max_line_w + pad_x * 2 + 5
    pill_h = len(label_lines) * line_height + pad_y * 2

    # 5. Position Label Pill (Prefer above bounding box; fallback to below or inside)
    pill_x1 = max(2, min(w_img - pill_w - 2, x1))
    pill_x2 = pill_x1 + pill_w

    if y1 - pill_h - 4 >= 0:
        pill_y1 = y1 - pill_h - 4
        pill_y2 = y1 - 4
    elif y2 + 4 + pill_h <= h_img:
        pill_y1 = y2 + 4
        pill_y2 = y2 + 4 + pill_h
    else:
        pill_y1 = max(2, y1 + 4)
        pill_y2 = min(h_img - 2, pill_y1 + pill_h)

    # Draw solid high-contrast background pill
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (18, 18, 22), -1)
    # Left alliance indicator bar
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x1 + 4, pill_y2), box_color, -1)
    # Thin border
    cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (55, 55, 65), 1)

    # Render Text Lines
    for idx, l_text in enumerate(label_lines):
        ty = pill_y1 + pad_y + (idx + 1) * line_height - 4
        if idx == 0:
            text_color = (255, 255, 255)
        elif "DEFENSE" in l_text:
            text_color = (30, 180, 255)  # Orange accent for defense
        elif "Action:" in l_text:
            text_color = (80, 240, 120)  # Green accent for active action
        else:
            text_color = (220, 220, 220)

        cv2.putText(frame, l_text, (pill_x1 + pad_x + 3, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)


def render_annotated_video(
    video_path: str,
    parquet_path: str,
    output_path: str = "output_annotated.mp4",
    max_frames: Optional[int] = None,
    fps: Optional[float] = None,
    draw_field_coords: bool = False,
) -> str:
    """
    Renders an annotated video by overlaying bounding boxes, kinematics, actions,
    and HUD from DuckDB Parquet stores onto the source video.

    Args:
        video_path: Path to original source video.
        parquet_path: Path to trajectories parquet file or directory.
        output_path: Path for annotated output video.
        max_frames: Optional frame processing upper limit.
        fps: Optional output FPS (default inherits source video FPS).
        draw_field_coords: If True, renders physical field coordinates (inches).

    Returns:
        Absolute path to rendered output video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Resolve Parquet files and load data
    traj_file, act_file = resolve_parquet_paths(parquet_path)
    trajectories_by_frame, actions_by_frame_and_track, match_id = load_parquet_data(
        traj_path=traj_file,
        actions_path=act_file,
        max_frames=max_frames,
    )

    balls_file = None
    if traj_file and "_trajectories.parquet" in traj_file:
        companion_balls = traj_file.replace("_trajectories.parquet", "_balls.parquet")
        if os.path.isfile(companion_balls):
            balls_file = companion_balls

    if not balls_file:
        if os.path.isdir(parquet_path):
            cands = glob.glob(os.path.join(parquet_path, "*balls*.parquet"))
            if cands:
                cands.sort(key=os.path.getmtime, reverse=True)
                balls_file = cands[0]
        elif os.path.isfile(parquet_arg := parquet_path):
            dir_name = os.path.dirname(os.path.abspath(parquet_arg))
            c = os.path.join(dir_name, "ball_trajectories.parquet")
            if os.path.isfile(c):
                balls_file = c

    balls_by_frame = load_ball_parquet_data(balls_file, max_frames=max_frames) if balls_file else {}

    # 2. Open Source Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0.0 or np.isnan(video_fps):
        video_fps = 30.0

    output_fps = float(fps) if fps is not None and fps > 0 else video_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Upper limit bound
    effective_limit = total_video_frames if total_video_frames > 0 else None
    if max_frames is not None:
        effective_limit = min(effective_limit, max_frames) if effective_limit is not None else max_frames

    logger.info(
        "Rendering setup: Video=%s (%dx%d @ %.2f FPS, %d frames), EffectiveLimit=%s, Output=%s",
        video_path,
        width,
        height,
        output_fps,
        total_video_frames,
        effective_limit,
        output_path,
    )

    # 3. Setup VideoWriter
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create VideoWriter for output path: {output_path}")

    frames_rendered = 0
    t_start = time.perf_counter()

    try:
        frame_index = 0
        while cap.isOpened():
            if effective_limit is not None and frame_index >= effective_limit:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            pts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if pts_ms <= 0:
                pts_ms = int(frame_index * (1000.0 / output_fps))

            robots = trajectories_by_frame.get(frame_index, [])
            frame_actions = actions_by_frame_and_track.get(frame_index, {})
            balls = balls_by_frame.get(frame_index, [])

            # Draw ball annotations first so robot boxes render on top cleanly
            if balls:
                draw_ball_annotations(frame=frame, ball_records=balls)

            # Draw robot bounding boxes and labels
            for robot in robots:
                t_id = int(robot.get("track_id") or 0)
                matched_action = frame_actions.get(t_id)
                draw_robot_annotation(
                    frame=frame,
                    robot=robot,
                    action=matched_action,
                    draw_field_coords=draw_field_coords,
                )

            focal_cnt = sum(1 for b in balls if not b.get("is_cluster", False)) if balls else 0
            ambient_cnt = sum(int(b.get("cluster_ball_count", 1)) for b in balls if b.get("is_cluster", False)) if balls else 0
            tot_balls = focal_cnt + ambient_cnt if balls else None

            # Draw top-left HUD overlay
            draw_hud(
                frame=frame,
                match_id=match_id,
                frame_index=frame_index,
                total_frames=total_video_frames,
                pts_ms=pts_ms,
                active_robots_count=len(robots),
                active_balls_count=tot_balls,
                focal_balls_count=focal_cnt if focal_cnt > 0 else None,
                ambient_balls_count=ambient_cnt if ambient_cnt > 0 else None,
            )

            writer.write(frame)
            frames_rendered += 1
            frame_index += 1

            if frames_rendered % 150 == 0:
                elapsed = time.perf_counter() - t_start
                cur_fps = frames_rendered / elapsed if elapsed > 0 else 0.0
                pct = (frames_rendered / effective_limit * 100.0) if effective_limit else 0.0
                logger.info("Rendering progress: %d frames (%.1f%%, %.1f FPS)", frames_rendered, pct, cur_fps)

    finally:
        cap.release()
        writer.release()

    total_time = time.perf_counter() - t_start
    final_fps = frames_rendered / total_time if total_time > 0 else 0.0
    abs_output = os.path.abspath(output_path)
    file_size_mb = os.path.getsize(abs_output) / (1024 * 1024) if os.path.isfile(abs_output) else 0.0

    print("\n" + "=" * 70)
    print(" VIDEO ANNOTATION RENDERING COMPLETE")
    print("=" * 70)
    print(f" Source Video:      {video_path}")
    print(f" Output Video:      {abs_output} ({file_size_mb:.2f} MB)")
    print(f" Total Frames:      {frames_rendered:,}")
    print(f" Render Speed:      {final_fps:.1f} FPS")
    print(f" Time Elapsed:      {total_time:.2f} seconds")
    print("=" * 70 + "\n")

    return abs_output


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the render CLI."""
    parser = argparse.ArgumentParser(
        description="FRC 2026 REBUILT: Video Annotation Rendering Command",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to original source video (e.g. sample.mp4).",
    )
    parser.add_argument(
        "--parquet", "-p",
        type=str,
        required=True,
        help="Path to Parquet file or directory containing trajectories (e.g. data/parquet/match_2026cmptx_f1m3_trajectories.parquet).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_annotated.mp4",
        help="Path for annotated output video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick rendering/verification.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for output video (default matches input video).",
    )
    parser.add_argument(
        "--draw-field-coords",
        action="store_true",
        default=False,
        help="Display physical field coordinates (inches) on robot labels.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint for video annotation rendering."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    logging.basicConfig(
        level=getattr(logging, parsed_args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        render_annotated_video(
            video_path=parsed_args.video,
            parquet_path=parsed_args.parquet,
            output_path=parsed_args.output,
            max_frames=parsed_args.max_frames,
            fps=parsed_args.fps,
            draw_field_coords=parsed_args.draw_field_coords,
        )
        return 0
    except Exception as exc:
        logger.error("Video annotation rendering failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
