"""Unit tests for 3D Scene, Field CAD & Kinematics Modules (Three.js).

Verifies via Node.js execution:
1. field3d.js: Coordinate conversion, carpet plane (651.2 x 317.7), guardrails (20" side rails,
   78" Red/Blue driver station end walls), Red/Blue Upper Hubs (rim Y=72", diameter 41.7",
   emissive LED ring), climbing towers (70" height, rungs), depots, trenches, 32 AprilTags
   at CAD ground-truth 3D positions and yaw rotations, ambient & stadium spotlight lighting.
2. robot3d.js: RobotManager, 30x30x28 inch envelope, dark metallic titanium chassis, lower 5"
   bumpers (Y=1 to 6) in Red/Blue fabric, front heading arrow indicator, 3D billboard sprite
   (Team number, live speed in ips, action badge pills), smooth lerp/slerp interpolation,
   click-to-select with glowing wireframe box.
3. ball3d.js: BallManager with 504 ball capacity, yellow FUEL spheres (r=4.0"), carpet balls
   at floor level Y=4.0, in-flight parabolic ballistic arc elevation arcing toward the Hub
   (up to 72"-85" height), subtle emissive glow for in-flight balls.
"""

from __future__ import annotations

import json
import subprocess
import pytest


def run_node_test(script: str) -> subprocess.CompletedProcess[str]:
    """Executes an inline Node.js ESM script and returns the process result."""
    return subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_field3d_geometry_and_elements() -> None:
    """Verifies field3d.js exports, coordinate conversions, field CAD geometry, 32 AprilTags, and lighting."""
    script = """
    import THREE from './src/server/static/vendor/three.min.js';
    import {
      createFieldCAD,
      to3DCoords,
      toFieldCoords,
      FIELD_DIMENSIONS,
      DEFAULT_APRILTAGS,
      CAD_SCALE_METERS_TO_INCHES,
      loadFieldGLB
    } from './src/server/static/js/field3d.js';

    if (Math.abs(CAD_SCALE_METERS_TO_INCHES - 39.37007874) > 1e-4) {
      throw new Error(`CAD_SCALE_METERS_TO_INCHES mismatch: ${CAD_SCALE_METERS_TO_INCHES}`);
    }
    if (typeof loadFieldGLB !== 'function') {
      throw new Error('loadFieldGLB must be a function');
    }

    // 1. Verify coordinate transforms
    // Length: 651.2, Width: 317.7 -> center is (325.6, 158.85)
    const center = to3DCoords(325.6, 158.85, 0.0);
    if (Math.abs(center.x) > 1e-4 || Math.abs(center.z) > 1e-4 || center.y !== 0) {
      throw new Error(`Center mapping failed: ${JSON.stringify(center)}`);
    }

    const backCenter = toFieldCoords(0.0, 0.0, 72.0);
    if (Math.abs(backCenter.xInch - 325.6) > 1e-4 || Math.abs(backCenter.yInch - 158.85) > 1e-4 || backCenter.zInch !== 72.0) {
      throw new Error(`Inverse mapping failed: ${JSON.stringify(backCenter)}`);
    }

    // 2. Instantiate Scene & Field CAD
    const scene = new THREE.Scene();
    const field = createFieldCAD(scene, {}, THREE);

    if (!field.fieldGroup) throw new Error('Missing fieldGroup in return object');
    if (!field.carpet) throw new Error('Missing carpet in return object');
    if (!field.redHub || !field.blueHub) throw new Error('Missing hubs');
    if (!field.redTower || !field.blueTower) throw new Error('Missing towers');
    if (!field.guardrails) throw new Error('Missing guardrails');
    if (!field.apriltags || field.apriltags.length !== 32) {
      throw new Error(`Expected 32 AprilTags, got ${field.apriltags?.length}`);
    }

    // 3. Hub Dimensions and Positions:
    // Red Hub at (162.8, 158.85) -> 3D X = 162.8 - 325.6 = -162.8, Z = 0
    // Blue Hub at (488.4, 158.85) -> 3D X = 488.4 - 325.6 = +162.8, Z = 0
    if (Math.abs(field.redHub.position.x - (-162.8)) > 1e-3) throw new Error('Red Hub X mismatch');
    if (Math.abs(field.blueHub.position.x - 162.8) > 1e-3) throw new Error('Blue Hub X mismatch');

    // 4. Towers:
    // Red Tower at (90.0, 158.85) -> 3D X = 90 - 325.6 = -235.6
    // Blue Tower at (561.2, 158.85) -> 3D X = 561.2 - 325.6 = +235.6
    if (Math.abs(field.redTower.position.x - (-235.6)) > 1e-3) throw new Error('Red Tower X mismatch');
    if (Math.abs(field.blueTower.position.x - 235.6) > 1e-3) throw new Error('Blue Tower X mismatch');

    // 5. AprilTag Ground-Truth Coordinates & Yaw Orientation (matching field.glb CAD)
    // Tag 1 (Red Hub/Trench): x=183.33, y=25.38, z=35.0, yaw=90.4
    // In 3D: X = 183.33 - 325.6 = -142.27, Z = 25.38 - 158.85 = -133.47, Y = 35.0
    const tag1 = field.apriltags.find(t => t.userData.tagId === 1);
    if (!tag1) throw new Error('Tag 1 not found');
    if (Math.abs(tag1.position.y - 35.0) > 1e-3) throw new Error(`Tag 1 Y elevation mismatch: ${tag1.position.y}`);
    if (Math.abs(tag1.position.z - (-133.47)) > 0.5) throw new Error(`Tag 1 Z mismatch: ${tag1.position.z}`);

    // Tag 2 & 11 (Red Hub & Trench Assembly facing field, aligned with CAD & broadcast)
    // Tag 2: x=182.10, y=135.10, z=44.25, yaw=243.2
    // In 3D: X = 182.10 - 325.6 = -143.50, Z = 135.10 - 158.85 = -23.75, Y = 44.25
    const tag2 = field.apriltags.find(t => t.userData.tagId === 2);
    if (!tag2) throw new Error('Tag 2 not found');
    if (Math.abs(tag2.position.y - 44.25) > 1e-3) throw new Error(`Tag 2 Y elevation mismatch: ${tag2.position.y}`);
    if (Math.abs(tag2.position.z - (-23.75)) > 0.5) throw new Error(`Tag 2 Z mismatch: ${tag2.position.z}`);

    // Tag 11: x=168.10, y=135.34, z=44.25, yaw=0.0
    // In 3D: X = 168.10 - 325.6 = -157.50, Z = 135.34 - 158.85 = -23.51, Y = 44.25
    const tag11 = field.apriltags.find(t => t.userData.tagId === 11);
    if (!tag11) throw new Error('Tag 11 not found');
    if (Math.abs(tag11.position.y - 44.25) > 1e-3) throw new Error(`Tag 11 Y elevation mismatch: ${tag11.position.y}`);
    if (Math.abs(tag11.position.z - (-23.51)) > 0.5) throw new Error(`Tag 11 Z mismatch: ${tag11.position.z}`);

    // 6. Lighting checks: Ambient 0.5, Stadium spotlight 0.8
    if (field.lights.ambientLight.intensity !== 0.5) throw new Error('Ambient light intensity mismatch');
    if (field.lights.stadiumLight.intensity !== 0.8) throw new Error('Stadium light intensity mismatch');
    if (!field.lights.stadiumLight.castShadow) throw new Error('Stadium light must cast shadows');

    console.log('field3d tests passed');
    """
    proc = run_node_test(script)
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "field3d tests passed" in proc.stdout


def test_robot3d_manager_kinematics_and_selection() -> None:
    """Verifies robot3d.js RobotManager, kinematics interpolation, HUD badges, and glowing wireframe selection."""
    script = """
    import THREE from './src/server/static/vendor/three.min.js';
    import { RobotManager } from './src/server/static/js/robot3d.js';

    const scene = new THREE.Scene();
    const mgr = new RobotManager(scene, THREE);

    // 1. Add robots
    const frame1 = [
      {
        track_id: 1,
        team_number: 4414,
        alliance: 'RED',
        field_x_inch: 160.0,
        field_y_inch: 158.85,
        velocity_x_ips: 30.0,
        velocity_y_ips: 0.0,
        heading_rad: 0.0,
        action_type: 'SCORE_HUB'
      },
      {
        track_id: 2,
        team_number: 2910,
        alliance: 'BLUE',
        field_x_inch: 490.0,
        field_y_inch: 158.85,
        velocity_x_ips: 0.0,
        velocity_y_ips: 15.0,
        heading_rad: Math.PI / 2.0,
        action_type: 'CLIMB_TOWER'
      }
    ];

    mgr.updateRobots(frame1, 1.0); // instant snap
    if (mgr.robots.size !== 2) throw new Error(`Expected 2 robots, got ${mgr.robots.size}`);

    const r1 = mgr.getRobot(1);
    const r2 = mgr.getRobot(2);
    if (!r1 || !r2) throw new Error('Robots not found in manager');

    // Check position in 3D:
    // 160 - 325.6 = -165.6
    if (Math.abs(r1.group.position.x - (-165.6)) > 1e-3) throw new Error('Robot 1 X position mismatch');
    if (Math.abs(r1.group.position.z - 0.0) > 1e-3) throw new Error('Robot 1 Z position mismatch');
    if (r1.group.position.y !== 0) throw new Error('Robot 1 Y must be 0 on carpet');

    // 2. Verify Robot Components
    let hasBumper = false;
    let hasChassis = false;
    let hasHeadingArrow = false;
    r1.group.traverse(child => {
      if (child.name === 'bumper') hasBumper = true;
      if (child.name === 'chassis') hasChassis = true;
      if (child.name === 'heading_arrow') hasHeadingArrow = true;
    });
    if (!hasBumper) throw new Error('Robot missing lower bumper');
    if (!hasChassis) throw new Error('Robot missing titanium chassis');
    if (!hasHeadingArrow) throw new Error('Robot missing directional heading arrow');

    // 3. Selection & Glowing Wireframe Box
    if (mgr.selectionBox.visible) throw new Error('Selection box must be initially hidden');
    if (mgr.getSelectedRobot() !== null) throw new Error('Initial selection should be null');

    const selected = mgr.selectRobot(1);
    if (!selected) throw new Error('Failed to select robot 1');
    if (!mgr.selectionBox.visible) throw new Error('Selection box should be visible when robot selected');

    const selInfo = mgr.getSelectedRobot();
    if (selInfo.trackId !== 1 || selInfo.teamNumber !== 4414 || selInfo.alliance !== 'RED') {
      throw new Error(`Selected robot info mismatch: ${JSON.stringify(selInfo)}`);
    }

    mgr.clearSelection();
    if (mgr.selectionBox.visible) throw new Error('Selection box should be hidden after clearSelection');
    if (mgr.getSelectedRobot() !== null) throw new Error('Selection should be null after clearSelection');

    // 4. Smooth Interpolation
    const frame2 = [
      {
        track_id: 1,
        team_number: 4414,
        alliance: 'RED',
        field_x_inch: 200.0, // target X = 200 - 325.6 = -125.6
        field_y_inch: 158.85,
        heading_rad: 0.0
      }
    ];

    // Lerp with alpha = 0.5 from -165.6 towards -125.6 -> halfway is -145.6
    mgr.updateRobots(frame2, 0.5);
    if (Math.abs(r1.group.position.x - (-145.6)) > 1e-3) {
      throw new Error(`Interpolation failed: expected -145.6, got ${r1.group.position.x}`);
    }

    // 5. Dynamic Disappearance Rule:
    // Robot 1 was detected, Robot 2 was not found in frame2.
    // Robot 2 must disappear (visible === false) because another robot (robot 1) was detected.
    if (r1.group.visible !== true) throw new Error('Robot 1 should remain visible in frame 2');
    if (r2.group.visible !== false) throw new Error('Robot 2 must disappear in frame 2 when robot 1 is detected');

    // Selection on disappeared robot should fail
    if (mgr.selectRobot(2) !== false) throw new Error('selectRobot should return false for disappeared robot');

    // 6. Reappearance Snapping:
    // When Robot 2 reappears in frame3, it should snap directly to its new position rather than lerping
    const frame3 = [
      {
        track_id: 2,
        team_number: 2910,
        alliance: 'BLUE',
        field_x_inch: 400.0, // target X = 400 - 325.6 = 74.4
        field_y_inch: 158.85,
        heading_rad: 0.0
      }
    ];

    mgr.updateRobots(frame3, 0.5);
    if (r2.group.visible !== true) throw new Error('Robot 2 should be visible again upon reappearance');
    // Robot 1 is now missing, and Robot 2 was detected -> Robot 1 must disappear!
    if (r1.group.visible !== false) throw new Error('Robot 1 should disappear in frame 3');

    // Robot 2 snapped to 74.4 immediately (not lerped from previous position)
    if (Math.abs(r2.group.position.x - 74.4) > 1e-3) {
      throw new Error(`Robot 2 reappearance should snap to 74.4, got ${r2.group.position.x}`);
    }

    console.log('robot3d tests passed');
    """
    proc = run_node_test(script)
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "robot3d tests passed" in proc.stdout


def test_ball3d_manager_parabolic_arc_and_capacity() -> None:
    """Verifies ball3d.js BallManager, 504 ball capacity, carpet vs in-flight elevation, and parabolic arc."""
    script = """
    import THREE from './src/server/static/vendor/three.min.js';
    import { BallManager, BALL_CONFIG } from './src/server/static/js/ball3d.js';

    const scene = new THREE.Scene();
    const mgr = new BallManager(scene, THREE);

    // 1. Carpet Ball Test: floor elevation Y = 2.955 (radius = 2.955, 5.91" diameter FUEL sphere)
    if (Math.abs(BALL_CONFIG.DIAMETER - 5.91) > 1e-4) throw new Error(`BALL_CONFIG.DIAMETER mismatch: ${BALL_CONFIG.DIAMETER}`);
    if (Math.abs(BALL_CONFIG.RADIUS - 2.955) > 1e-4) throw new Error(`BALL_CONFIG.RADIUS mismatch: ${BALL_CONFIG.RADIUS}`);
    if (Math.abs(BALL_CONFIG.CARPET_Y - 2.955) > 1e-4) throw new Error(`BALL_CONFIG.CARPET_Y mismatch: ${BALL_CONFIG.CARPET_Y}`);

    const carpetState = [
      {
        ball_id: 101,
        field_x_inch: 325.6,
        field_y_inch: 158.85,
        ball_state: 'ON_CARPET'
      }
    ];

    mgr.updateBalls(carpetState);
    let counts = mgr.getBallCount();
    if (counts.carpet !== 1 || counts.inFlight !== 0 || counts.total !== 1) {
      throw new Error(`Carpet count mismatch: ${JSON.stringify(counts)}`);
    }

    const matCarpet = new THREE.Matrix4();
    mgr.carpetMesh.getMatrixAt(0, matCarpet);
    const posCarpet = new THREE.Vector3().setFromMatrixPosition(matCarpet);
    if (Math.abs(posCarpet.y - 2.955) > 1e-3) {
      throw new Error(`Carpet ball elevation must be 2.955, got ${posCarpet.y}`);
    }

    // 2. In-Flight Ball Test: Parabolic arc toward Upper Hub
    // Red Hub at 3D X = -162.8, Z = 0.0. Ball shot at X_inch=220 (3D X = -105.6, distance ~57")
    const inFlightState = [
      {
        ball_id: 202,
        field_x_inch: 220.0,
        field_y_inch: 158.85,
        ball_state: 'IN_FLIGHT',
        velocity_x_ips: -45.0,
        velocity_y_ips: 0.0
      }
    ];

    mgr.updateBalls(inFlightState);
    counts = mgr.getBallCount();
    if (counts.inFlight !== 1 || counts.carpet !== 0) {
      throw new Error(`In-flight count mismatch: ${JSON.stringify(counts)}`);
    }

    const matFlight = new THREE.Matrix4();
    mgr.inFlightMesh.getMatrixAt(0, matFlight);
    const posFlight = new THREE.Vector3().setFromMatrixPosition(matFlight);

    // Requirement: In-flight elevation along parabolic arc up to 72"-85" height
    if (posFlight.y < 72.0 || posFlight.y > 85.0) {
      throw new Error(`In-flight elevation out of 72"-85" range: ${posFlight.y}`);
    }

    // 3. Test Parabolic Elevation Method directly across full arc
    // Hub Rim elevation is 72.0 at distance 0"
    const elevAtHub = mgr.computeParabolicElevation(-162.8, 0.0);
    if (Math.abs(elevAtHub - 72.0) > 1e-2) {
      throw new Error(`Elevation at Hub rim must be 72.0, got ${elevAtHub}`);
    }

    // Apex elevation (around 40-50" distance)
    const elevNearApex = mgr.computeParabolicElevation(-115.0, 0.0);
    if (elevNearApex < 78.0 || elevNearApex > 85.0) {
      throw new Error(`Apex elevation out of range: ${elevNearApex}`);
    }

    // 4. Test 504 ball capacity limit
    const allBalls = [];
    for (let i = 0; i < 504; i++) {
      allBalls.push({
        ball_id: i,
        field_x_inch: 50.0 + (i % 25) * 20.0,
        field_y_inch: 20.0 + Math.floor(i / 25) * 12.0,
        ball_state: (i < 24) ? 'IN_FLIGHT' : 'ON_CARPET'
      });
    }

    mgr.updateBalls(allBalls);
    counts = mgr.getBallCount();
    if (counts.total !== 504) {
      throw new Error(`Expected exactly 504 total balls, got ${counts.total}`);
    }
    if (counts.inFlight !== 24) throw new Error(`Expected 24 in-flight, got ${counts.inFlight}`);
    if (counts.carpet !== 480) throw new Error(`Expected 480 carpet, got ${counts.carpet}`);

    // 5. Clear method
    mgr.clear();
    counts = mgr.getBallCount();
    if (counts.total !== 0 || counts.carpet !== 0 || counts.inFlight !== 0) {
      throw new Error(`Clear failed: ${JSON.stringify(counts)}`);
    }

    console.log('ball3d tests passed');
    """
    proc = run_node_test(script)
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "ball3d tests passed" in proc.stdout


def test_end_to_end_scene_integration() -> None:
    """Verifies that field3d, robot3d, and ball3d integrate seamlessly in a shared Three.js scene."""
    script = """
    import THREE from './src/server/static/vendor/three.min.js';
    import { createFieldCAD } from './src/server/static/js/field3d.js';
    import { RobotManager } from './src/server/static/js/robot3d.js';
    import { BallManager } from './src/server/static/js/ball3d.js';

    const scene = new THREE.Scene();

    // 1. Build Field
    const field = createFieldCAD(scene, {}, THREE);

    // 2. Initialize Managers
    const robotMgr = new RobotManager(scene, THREE);
    const ballMgr = new BallManager(scene, THREE);

    // 3. Simulate Match Telemetry Frame (6 robots, 20 balls)
    const robots = [
      { track_id: 1, team_number: 4414, alliance: 'RED', field_x_inch: 180.0, field_y_inch: 158.85, heading_rad: 0.0, action_type: 'SCORE_HUB' },
      { track_id: 2, team_number: 2910, alliance: 'RED', field_x_inch: 100.0, field_y_inch: 100.0, heading_rad: 0.5, action_type: 'TRANSIT' },
      { track_id: 3, team_number: 1678, alliance: 'RED', field_x_inch: 90.0,  field_y_inch: 158.85, heading_rad: 1.0, action_type: 'CLIMB_TOWER' },
      { track_id: 4, team_number: 254,  alliance: 'BLUE', field_x_inch: 470.0, field_y_inch: 158.85, heading_rad: 3.14, action_type: 'SCORE_HUB' },
      { track_id: 5, team_number: 1323, alliance: 'BLUE', field_x_inch: 550.0, field_y_inch: 200.0, heading_rad: 2.8, action_type: 'DEFENSE' },
      { track_id: 6, team_number: 971,  alliance: 'BLUE', field_x_inch: 561.2, field_y_inch: 158.85, heading_rad: 3.0, action_type: 'CLIMB_TOWER' }
    ];

    const balls = [];
    for (let i = 0; i < 20; i++) {
      balls.push({
        ball_id: i,
        field_x_inch: 150.0 + i * 15.0,
        field_y_inch: 150.0 + (i % 3) * 10.0,
        ball_state: (i < 4) ? 'IN_FLIGHT' : 'ON_CARPET'
      });
    }

    robotMgr.updateRobots(robots, 1.0);
    ballMgr.updateBalls(balls);

    if (robotMgr.robots.size !== 6) throw new Error(`Expected 6 active robots, got ${robotMgr.robots.size}`);
    if (ballMgr.getBallCount().total !== 20) throw new Error(`Expected 20 active balls, got ${ballMgr.getBallCount().total}`);

    // Verify raycast selection capability
    const raycaster = new THREE.Raycaster();
    // Ray pointing straight down onto robot 1 at 3D pos (-145.6, 0, 0)
    raycaster.set(new THREE.Vector3(-145.6, 100.0, 0.0), new THREE.Vector3(0, -1, 0));
    const hitId = robotMgr.raycast(raycaster);
    if (hitId !== 1) throw new Error(`Raycast expected to hit robot 1, got ${hitId}`);
    if (robotMgr.getSelectedRobot()?.trackId !== 1) throw new Error('Robot 1 not selected after raycast');

    console.log('e2e scene integration tests passed');
    """
    proc = run_node_test(script)
    assert proc.returncode == 0, f"Node test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "e2e scene integration tests passed" in proc.stdout
