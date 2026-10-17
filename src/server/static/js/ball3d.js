/**
 * FRC 2026 REBUILT: 3D Yellow FUEL Ball Visualization
 * Three.js 3D Game Piece Module
 *
 * Manages up to 504 yellow FUEL spheres (r = 4.0 inches) using InstancedMesh
 * for 60+ FPS rendering performance. Supports carpet balls at floor elevation (Y=4.0),
 * parabolic ballistic trajectories arcing toward Red/Blue Upper Hubs (up to 72-85" height),
 * and subtle emissive glow for in-flight game pieces.
 */

import { to3DCoords, FIELD_DIMENSIONS } from './field3d.js';

function getTHREE(customThree) {
  if (customThree) return customThree;
  if (typeof THREE !== 'undefined') return THREE;
  if (typeof window !== 'undefined' && window.THREE) return window.THREE;
  if (typeof globalThis !== 'undefined' && globalThis.THREE) return globalThis.THREE;
  if (typeof global !== 'undefined' && global.THREE) return global.THREE;
  throw new Error('THREE is not defined. Ensure three.min.js is included or pass customThree instance.');
}

export const BALL_CONFIG = {
  DIAMETER: 5.91,        // Official FRC 2026 FUEL ball diameter (150 mm)
  RADIUS: 2.955,         // 2.955 inch radius (5.91" diameter FUEL sphere)
  MAX_BALLS: 504,        // Official FRC 2026 match total
  MAX_IN_FLIGHT: 128,    // High-capacity in-flight allocation
  CARPET_Y: 2.955,       // Floor level elevation (radius = 2.955)
  HUB_RIM_Y: 72.0,       // Upper Hub rim elevation
  HUB_APEX_Y: 83.0,      // Parabolic flight apex (72" to 85" height)
  SHOOTER_LAUNCH_Y: 22.0,// Typical robot shooter exit height
  COLOR_YELLOW: 0xfacc15,// Vibrant yellow (#facc15)
  COLOR_EMISSIVE: 0xfde047 // Subtle emissive glow (#fde047)
};

/**
 * BallManager orchestrates high-performance InstancedMesh rendering for all 504 FUEL balls,
 * computing real-time parabolic ballistic arcs for in-flight shots toward Upper Hubs.
 */
export class BallManager {
  /**
   * @param {THREE.Scene|THREE.Group} scene - Three.js target scene or parent group
   * @param {object} [customThree=null] - Optional THREE namespace override
   */
  constructor(scene, customThree = null) {
    this.THREE = getTHREE(customThree);
    if (!scene) {
      throw new Error('BallManager requires a valid Three.js Scene or Group instance.');
    }
    this.scene = scene;

    // Parent group for all ball instances
    this.group = new this.THREE.Group();
    this.group.name = 'fuel_balls_manager';
    this.scene.add(this.group);

    // Dummy Object3D used for computing instance transformation matrices
    this.dummy = new this.THREE.Object3D();

    // Cache of in-flight tracking state across frames
    this.inFlightTracker = new Map();

    // Cache of active balls metadata
    this.activeCarpetCount = 0;
    this.activeInFlightCount = 0;
    this.lastBallStates = [];

    // Initialize InstancedMeshes
    this._initInstancedMeshes();
  }

  /**
   * Initializes dual InstancedMeshes:
   * 1. Carpet Balls: Vibrant yellow (#facc15, roughness 0.3, metalness 0.1)
   * 2. In-Flight Balls: Vibrant yellow with subtle emissive glow (#fde047)
   * @private
   */
  _initInstancedMeshes() {
    const THREE = this.THREE;

    // Sphere geometry (r = 4.0 inches, 16 segments)
    this.sphereGeom = new THREE.SphereGeometry(BALL_CONFIG.RADIUS, 16, 16);

    // 1. Carpet Ball Material (Standard vibrant yellow, no emissive)
    this.carpetMat = new THREE.MeshStandardMaterial({
      color: BALL_CONFIG.COLOR_YELLOW,
      roughness: 0.3,
      metalness: 0.1
    });

    // 2. In-Flight Ball Material (Subtle emissive glow arcing toward Hub)
    this.inFlightMat = new THREE.MeshStandardMaterial({
      color: BALL_CONFIG.COLOR_YELLOW,
      emissive: BALL_CONFIG.COLOR_EMISSIVE,
      emissiveIntensity: 0.65,
      roughness: 0.2,
      metalness: 0.1
    });

    // InstancedMesh for Carpet Balls (up to 504 balls)
    this.carpetMesh = new THREE.InstancedMesh(
      this.sphereGeom,
      this.carpetMat,
      BALL_CONFIG.MAX_BALLS
    );
    this.carpetMesh.name = 'carpet_fuel_balls';
    this.carpetMesh.castShadow = true;
    this.carpetMesh.receiveShadow = true;
    this.carpetMesh.count = 0; // Initially 0 active

    // InstancedMesh for In-Flight Glowing Balls (up to 128 balls)
    this.inFlightMesh = new THREE.InstancedMesh(
      this.sphereGeom,
      this.inFlightMat,
      BALL_CONFIG.MAX_IN_FLIGHT
    );
    this.inFlightMesh.name = 'in_flight_fuel_balls';
    this.inFlightMesh.castShadow = true;
    this.inFlightMesh.count = 0; // Initially 0 active

    // Zero-initialize all instance matrices
    const zeroMat = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < BALL_CONFIG.MAX_BALLS; i++) {
      this.carpetMesh.setMatrixAt(i, zeroMat);
    }
    for (let i = 0; i < BALL_CONFIG.MAX_IN_FLIGHT; i++) {
      this.inFlightMesh.setMatrixAt(i, zeroMat);
    }
    this.carpetMesh.instanceMatrix.needsUpdate = true;
    this.inFlightMesh.instanceMatrix.needsUpdate = true;

    this.group.add(this.carpetMesh);
    this.group.add(this.inFlightMesh);
  }

  /**
   * Computes 3D elevation Y along a parabolic ballistic arc arcing toward the Upper Hub.
   * Trajectory rises from shooter height (~22") up to an apex of 72"-85" elevation,
   * descending cleanly into the Upper Hub rim at Y=72.0".
   *
   * @param {number} x3D - Three.js X coordinate
   * @param {number} z3D - Three.js Z coordinate
   * @param {object} [options={}] - Kinematics options (velocity, explicit elevation, flight progress)
   * @returns {number} 3D elevation Y in inches
   */
  computeParabolicElevation(x3D, z3D, options = {}) {
    // 1. Explicit elevation override if provided by telemetry/tracker
    if (options.elevation !== undefined && options.elevation !== null) {
      return Math.max(BALL_CONFIG.CARPET_Y, options.elevation);
    }
    if (options.field_z_inch !== undefined && options.field_z_inch !== null && options.field_z_inch > 4.0) {
      return options.field_z_inch;
    }

    // 2. Identify target Upper Hub
    // Red Hub at X = -162.8, Z = 0.0; Blue Hub at X = +162.8, Z = 0.0
    const redHubX = -162.8;
    const blueHubX = 162.8;
    const targetHubX = (x3D < 0) ? redHubX : blueHubX;
    const targetHubZ = 0.0;

    // Horizontal distance to target Hub opening
    const distToHub = Math.hypot(x3D - targetHubX, z3D - targetHubZ);

    // 3. If normalized flight progress t in [0, 1] is provided:
    if (options.flightProgress !== undefined && options.flightProgress !== null) {
      const t = Math.max(0.0, Math.min(1.0, options.flightProgress));
      // Parabolic equation with apex at t=0.6, apex height=83", rim=72", launch=22"
      const y = BALL_CONFIG.SHOOTER_LAUNCH_Y +
                (BALL_CONFIG.HUB_RIM_Y - BALL_CONFIG.SHOOTER_LAUNCH_Y) * t +
                4.0 * (BALL_CONFIG.HUB_APEX_Y - (BALL_CONFIG.SHOOTER_LAUNCH_Y + BALL_CONFIG.HUB_RIM_Y) / 2.0) * t * (1.0 - t);
      return Math.min(BALL_CONFIG.HUB_APEX_Y, Math.max(BALL_CONFIG.CARPET_Y, y));
    }

    // 4. Parabolic arc parameterized by distance from Hub:
    // Typical shooting range: 140" from Hub down to 0" (Hub rim)
    const maxShotDist = 140.0;
    if (distToHub > maxShotDist + 20.0) {
      // Outside typical shot window: return launch elevation or carpet
      return BALL_CONFIG.SHOOTER_LAUNCH_Y;
    }

    // Normalized progress s: s = 0 at distance 140", s = 1.0 at Hub center (0")
    const s = Math.max(0.0, Math.min(1.0, 1.0 - (distToHub / maxShotDist)));

    // Quadratic ballistic arc: passes through (0, 22"), apex (0.7, ~83"), and (1.0, 72")
    // Y(s) = 22.0 + 172.0*s - 122.0*s^2
    const yElev = 22.0 + 172.0 * s - 122.0 * s * s;

    return Math.min(BALL_CONFIG.HUB_APEX_Y, Math.max(BALL_CONFIG.CARPET_Y, yElev));
  }

  /**
   * Updates all 504 ball positions from state telemetry.
   *
   * @param {Array<object>|object} ballStates - Array or dictionary of ball state records
   */
  updateBalls(ballStates) {
    if (!ballStates) return;

    const list = Array.isArray(ballStates)
      ? ballStates
      : Object.values(ballStates);

    this.lastBallStates = list;

    let carpetIdx = 0;
    let inFlightIdx = 0;

    for (const state of list) {
      // Extract coordinates (field inches) and convert to Three.js coordinates
      let p3dX = 0.0;
      let p3dZ = 0.0;
      if (state.center_rel_x_inch !== undefined && state.center_rel_y_inch !== undefined) {
        p3dX = state.center_rel_x_inch;
        p3dZ = state.center_rel_y_inch;
      } else {
        const fieldX = state.field_x_inch !== undefined ? state.field_x_inch : (state.x || 325.6);
        const fieldY = state.field_y_inch !== undefined ? state.field_y_inch : (state.y || 158.85);
        const p3d = to3DCoords(fieldX, fieldY, 0.0);
        p3dX = p3d.x;
        p3dZ = p3d.z;
      }

      // Determine state: 'ON_CARPET' | 'IN_FLIGHT' | 'SCORED' | 'INTAKED' | 'LOST'
      const rawState = (state.ball_state || state.state || 'ON_CARPET').toUpperCase();
      const isExplicitElevation = (state.field_z_inch > 6.0 || state.elevation > 6.0 || state.elevation_inch > 6.0);
      const isInFlight = rawState === 'IN_FLIGHT' || rawState === 'SCORED' || isExplicitElevation;

      // Handle clustered ball piles (e.g. Depots or Corrals)
      const isCluster = Boolean(state.is_cluster);
      const count = (isCluster && state.cluster_ball_count > 1)
        ? Math.min(16, state.cluster_ball_count)
        : 1;

      for (let c = 0; c < count; c++) {
        let x = p3dX;
        let z = p3dZ;

        // Cluster radial dispersal pattern
        if (isCluster && count > 1 && c > 0) {
          const angle = (c / (count - 1)) * Math.PI * 2.0;
          const dist = 5.5 + (c % 2) * 4.0;
          x += Math.cos(angle) * dist;
          z += Math.sin(angle) * dist;
        }

        if (isInFlight) {
          if (inFlightIdx >= BALL_CONFIG.MAX_IN_FLIGHT) continue;

          // Compute 3D elevation Y along parabolic arc toward Hub
          const elevY = this.computeParabolicElevation(x, z, {
            elevation: state.elevation || state.elevation_inch,
            field_z_inch: state.field_z_inch,
            flightProgress: state.flight_progress,
            velocity_x_ips: state.velocity_x_ips,
            velocity_y_ips: state.velocity_y_ips
          });

          this.dummy.position.set(x, elevY, z);
          this.dummy.scale.set(1.0, 1.0, 1.0);
          this.dummy.updateMatrix();

          this.inFlightMesh.setMatrixAt(inFlightIdx, this.dummy.matrix);
          inFlightIdx++;
        } else {
          if (carpetIdx >= BALL_CONFIG.MAX_BALLS) continue;

          // Carpet balls sit squarely on floor carpet (Y = 4.0)
          this.dummy.position.set(x, BALL_CONFIG.CARPET_Y, z);
          this.dummy.scale.set(1.0, 1.0, 1.0);
          this.dummy.updateMatrix();

          this.carpetMesh.setMatrixAt(carpetIdx, this.dummy.matrix);
          carpetIdx++;
        }
      }
    }

    // Zero out unused matrices in InstancedMeshes
    const zeroMat = new this.THREE.Matrix4().makeScale(0, 0, 0);

    for (let i = carpetIdx; i < this.activeCarpetCount; i++) {
      this.carpetMesh.setMatrixAt(i, zeroMat);
    }
    for (let i = inFlightIdx; i < this.activeInFlightCount; i++) {
      this.inFlightMesh.setMatrixAt(i, zeroMat);
    }

    // Set active instance counts
    this.carpetMesh.count = carpetIdx;
    this.inFlightMesh.count = inFlightIdx;

    this.activeCarpetCount = carpetIdx;
    this.activeInFlightCount = inFlightIdx;

    this.carpetMesh.instanceMatrix.needsUpdate = true;
    this.inFlightMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Resets and hides all active ball instances.
   */
  clear() {
    const zeroMat = new this.THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < BALL_CONFIG.MAX_BALLS; i++) {
      this.carpetMesh.setMatrixAt(i, zeroMat);
    }
    for (let i = 0; i < BALL_CONFIG.MAX_IN_FLIGHT; i++) {
      this.inFlightMesh.setMatrixAt(i, zeroMat);
    }
    this.carpetMesh.count = 0;
    this.inFlightMesh.count = 0;
    this.activeCarpetCount = 0;
    this.activeInFlightCount = 0;
    this.carpetMesh.instanceMatrix.needsUpdate = true;
    this.inFlightMesh.instanceMatrix.needsUpdate = true;
    this.inFlightTracker.clear();
  }

  /**
   * Returns current count of active balls.
   *
   * @returns {{carpet: number, inFlight: number, total: number}}
   */
  getBallCount() {
    return {
      carpet: this.activeCarpetCount,
      inFlight: this.activeInFlightCount,
      total: this.activeCarpetCount + this.activeInFlightCount
    };
  }

  /**
   * Disposes all InstancedMeshes, geometries, and materials.
   */
  dispose() {
    this.clear();
    this.sphereGeom.dispose();
    this.carpetMat.dispose();
    this.inFlightMat.dispose();
    this.scene.remove(this.group);
  }
}

export default BallManager;
