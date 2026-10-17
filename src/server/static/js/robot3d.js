/**
 * FRC 2026 REBUILT: 3D Robot Kinematics & Tracking Visualization
 * Three.js 3D Robot Module
 *
 * Manages 3D robot models, swerve drive chassis, alliance bumpers,
 * heading directional indicators, 3D billboard HUD sprites,
 * smooth frame interpolation, and click-to-select glowing wireframes.
 */

import { to3DCoords } from './field3d.js';

function getTHREE(customThree) {
  if (customThree) return customThree;
  if (typeof THREE !== 'undefined') return THREE;
  if (typeof window !== 'undefined' && window.THREE) return window.THREE;
  if (typeof globalThis !== 'undefined' && globalThis.THREE) return globalThis.THREE;
  if (typeof global !== 'undefined' && global.THREE) return global.THREE;
  throw new Error('THREE is not defined. Ensure three.min.js is included or pass customThree instance.');
}

// Action badge styling specifications
const ACTION_BADGES = {
  SCORE_HUB: { label: 'SCORE HUB', bg: '#15803d', text: '#dcfce7', pill: '#22c55e' },
  CLIMB_TOWER: { label: 'CLIMB TOWER', bg: '#7e22ce', text: '#f3e8ff', pill: '#a855f7' },
  DEFENSIVE_ENGAGEMENT: { label: 'DEFENSE', bg: '#c2410c', text: '#ffedd5', pill: '#f97316' },
  DEFENSE: { label: 'DEFENSE', bg: '#c2410c', text: '#ffedd5', pill: '#f97316' },
  INTAKE_FUEL: { label: 'INTAKE FUEL', bg: '#b45309', text: '#fef3c7', pill: '#eab308' },
  TRANSIT_NEUTRAL_ZONE: { label: 'TRANSIT', bg: '#0e7490', text: '#cffafe', pill: '#06b6d4' },
  TRANSIT: { label: 'TRANSIT', bg: '#0e7490', text: '#cffafe', pill: '#06b6d4' },
  OUTPOST_CORRAL_CYCLE: { label: 'OUTPOST', bg: '#1d4ed8', text: '#dbeafe', pill: '#3b82f6' },
  IDLE: { label: 'IDLE', bg: '#334155', text: '#f1f5f9', pill: '#64748b' }
};

/**
 * RobotManager oversees all 3D robot instances on the field,
 * managing kinematics interpolation, HUD badges, and interactive selection.
 */
export class RobotManager {
  /**
   * @param {THREE.Scene|THREE.Group} scene - Three.js scene or parent group
   * @param {object} [customThree=null] - Optional THREE namespace override
   */
  constructor(scene, customThree = null) {
    this.THREE = getTHREE(customThree);
    if (!scene) {
      throw new Error('RobotManager requires a valid Three.js Scene or Group instance.');
    }
    this.scene = scene;

    // Container for all robot 3D objects
    this.container = new this.THREE.Group();
    this.container.name = 'robots_container';
    this.scene.add(this.container);

    // Map: trackId (number|string) -> RobotEntry
    this.robots = new Map();

    // Selection tracking
    this.selectedTrackId = null;

    // Shared reusable geometries & materials
    this._initSharedAssets();

    // Selection highlight wireframe box
    this.selectionBox = this._createSelectionBox();
    this.container.add(this.selectionBox);
  }

  /**
   * Initializes shared assets to optimize GPU memory.
   * @private
   */
  _initSharedAssets() {
    const THREE = this.THREE;

    // Robot dimensions: 30" x 30" base, 28" overall height
    // Lower 5 inches (Y=1 to Y=6) are bumpers (30 x 5 x 30)
    this.bumperGeom = new THREE.BoxGeometry(30.0, 5.0, 30.0);
    this.bumperGeom.translate(0, 3.5, 0); // Y: 1.0 to 6.0

    // Main titanium chassis (26" x 14" x 26")
    this.chassisGeom = new THREE.BoxGeometry(26.0, 14.0, 26.0);
    this.chassisGeom.translate(0, 10.0, 0); // Y: 3.0 to 17.0

    // Elevator / upper upright structure tubes rising to 28"
    this.mastGeom = new THREE.CylinderGeometry(1.25, 1.25, 18.0, 8);
    this.mastGeom.translate(0, 19.0, 0); // Y: 10.0 to 28.0

    // Mast cross-brace at top
    this.mastTopGeom = new THREE.BoxGeometry(3.0, 1.5, 20.0);
    this.mastTopGeom.translate(0, 27.5, 0);

    // Front directional arrow geometry (on top deck pointing +X)
    const arrowShape = new THREE.Shape();
    arrowShape.moveTo(0, 8);
    arrowShape.lineTo(6, -6);
    arrowShape.lineTo(2, -4);
    arrowShape.lineTo(2, -10);
    arrowShape.lineTo(-2, -10);
    arrowShape.lineTo(-2, -4);
    arrowShape.lineTo(-6, -6);
    arrowShape.closePath();

    this.arrowGeom = new THREE.ShapeGeometry(arrowShape);
    this.arrowGeom.rotateX(-Math.PI / 2); // Lay flat on XZ
    this.arrowGeom.rotateY(-Math.PI / 2); // Point towards +X
    this.arrowGeom.scale(0.8, 0.8, 0.8);
    this.arrowGeom.translate(3.0, 17.15, 0); // Top of chassis deck

    // Front bumper directional chevron indicator
    this.frontAccentGeom = new THREE.BoxGeometry(0.5, 3.0, 12.0);
    this.frontAccentGeom.translate(15.2, 3.5, 0); // Mounted on front face of bumper (+X)

    // Chassis materials
    this.chassisMat = new THREE.MeshStandardMaterial({
      color: 0x242933,
      metalness: 0.85,
      roughness: 0.25
    });

    this.mastMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.9,
      roughness: 0.2
    });

    this.redBumperMat = new THREE.MeshStandardMaterial({
      color: 0xdc2626,
      roughness: 0.7,
      metalness: 0.05
    });

    this.blueBumperMat = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      roughness: 0.7,
      metalness: 0.05
    });

    this.grayBumperMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.7,
      metalness: 0.05
    });

    // Heading arrow material
    this.arrowMat = new THREE.MeshBasicMaterial({
      color: 0xfef08a,
      side: THREE.DoubleSide
    });

    this.accentMat = new THREE.MeshStandardMaterial({
      color: 0xfacc15,
      emissive: 0xfacc15,
      emissiveIntensity: 0.8,
      roughness: 0.3
    });
  }

  /**
   * Constructs the selection highlight box (glowing wireframe around selected robot).
   * @private
   */
  _createSelectionBox() {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.name = 'selection_highlight';
    group.visible = false;

    // Outer wireframe box (32 x 30 x 32 inches) surrounding 30x28x30 robot
    const boxGeom = new THREE.BoxGeometry(32.0, 30.0, 32.0);
    const edgesGeom = new THREE.EdgesGeometry(boxGeom);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      linewidth: 2
    });
    const wireframe = new THREE.LineSegments(edgesGeom, lineMat);
    wireframe.position.y = 15.0; // Y: 0 to 30
    group.add(wireframe);

    // Subtle translucent cyan glow volume
    const fillMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const fillMesh = new THREE.Mesh(boxGeom, fillMat);
    fillMesh.position.y = 15.0;
    group.add(fillMesh);

    return group;
  }

  /**
   * Creates a dynamic canvas texture for the 3D billboard sprite above a robot.
   * Renders:
   *   - Team Number (e.g. 'Team 4414', 'Team 2910')
   *   - Live Speed (e.g. '34.2 ips')
   *   - Action Badge (e.g. 'SCORE_HUB' green pill, 'CLIMB_TOWER' purple pill, 'DEFENSE' orange pill)
   *
   * @private
   */
  _createBillboardSprite(teamNumber, trackId, alliance = 'RED') {
    const THREE = this.THREE;

    if (typeof document === 'undefined') {
      // Headless / non-browser fallback: return mock sprite
      const mockMat = new THREE.SpriteMaterial({ color: 0xffffff });
      const sprite = new THREE.Sprite(mockMat);
      sprite.position.set(0, 38.0, 0);
      sprite.scale.set(24.0, 12.0, 1.0);
      return { sprite, update: () => {} };
    }

    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;

    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(0, 38.0, 0); // 10" above 28" robot top
    sprite.scale.set(24.0, 12.0, 1.0);

    let lastSignature = '';

    const update = (teamNum, speedIps, actionType, curAlliance) => {
      const teamText = (teamNum && teamNum > 0) ? `Team ${teamNum}` : `Robot #${trackId}`;
      const speedText = `${(speedIps || 0.0).toFixed(1)} ips`;
      const actionKey = (actionType || 'IDLE').toUpperCase();
      const badgeConfig = ACTION_BADGES[actionKey] || ACTION_BADGES.IDLE;
      const signature = `${teamText}|${speedText}|${badgeConfig.label}|${curAlliance}`;

      if (signature === lastSignature) {
        return; // Skip re-drawing if unchanged
      }
      lastSignature = signature;

      // Clear canvas
      ctx.clearRect(0, 0, 512, 256);

      // Card Background (Frosted Glass Card)
      const isRed = (curAlliance || alliance).toUpperCase() === 'RED';
      const allianceColor = isRed ? '#ef4444' : '#3b82f6';

      // Rounded container box
      const r = 24;
      const x = 16, y = 16, w = 480, h = 224;
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();

      ctx.fillStyle = 'rgba(15, 23, 42, 0.90)'; // Dark slate
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.stroke();

      // Top Alliance Banner Stripe
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + 8);
      ctx.lineTo(x, y + 8);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
      ctx.fillStyle = allianceColor;
      ctx.fill();

      // 1. Team Number Heading
      ctx.font = 'bold 38px "Segoe UI", Roboto, system-ui, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(teamText, 36, 68);

      // 2. Live Speed Indicator
      ctx.font = 'bold 26px monospace';
      ctx.fillStyle = '#38bdf8'; // Cyan
      ctx.textAlign = 'right';
      ctx.fillText(`⚡ ${speedText}`, 470, 68);

      // Divider line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(36, 112);
      ctx.lineTo(476, 112);
      ctx.stroke();

      // 3. Action Badge Pill (e.g. SCORE_HUB green pill, CLIMB_TOWER purple pill)
      const pillW = 440;
      const pillH = 72;
      const pillX = 36;
      const pillY = 136;
      const pillR = 18;

      ctx.beginPath();
      ctx.moveTo(pillX + pillR, pillY);
      ctx.lineTo(pillX + pillW - pillR, pillY);
      ctx.quadraticCurveTo(pillX + pillW, pillY, pillX + pillW, pillY + pillR);
      ctx.lineTo(pillX + pillW, pillY + pillH - pillR);
      ctx.quadraticCurveTo(pillX + pillW, pillY + pillH, pillX + pillW - pillR, pillY + pillH);
      ctx.lineTo(pillX + pillR, pillY + pillH);
      ctx.quadraticCurveTo(pillX, pillY + pillH, pillX, pillY + pillH - pillR);
      ctx.lineTo(pillX, pillY + pillR);
      ctx.quadraticCurveTo(pillX, pillY, pillX + pillR, pillY);
      ctx.closePath();

      ctx.fillStyle = badgeConfig.bg;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = badgeConfig.pill;
      ctx.stroke();

      // Badge Text
      ctx.font = 'bold 28px "Segoe UI", Roboto, system-ui, sans-serif';
      ctx.fillStyle = badgeConfig.text;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(badgeConfig.label, pillX + pillW / 2.0, pillY + pillH / 2.0);

      texture.needsUpdate = true;
    };

    // Initial render
    update(teamNumber, 0.0, 'IDLE', alliance);

    return { sprite, update };
  }

  /**
   * Constructs a complete 3D robot mesh group for a track ID.
   *
   * @param {number|string} trackId - Unique robot track ID
   * @param {number|null} teamNumber - Team number (e.g. 4414, 2910)
   * @param {string} alliance - 'RED' | 'BLUE' | 'UNKNOWN'
   * @returns {object} RobotEntry containing group, meshes, and updater
   * @private
   */
  _createRobotMesh(trackId, teamNumber, alliance) {
    const THREE = this.THREE;
    const group = new THREE.Group();
    group.name = `robot_${trackId}`;

    const isRed = (alliance || 'RED').toUpperCase() === 'RED';
    const bumperMat = isRed ? this.redBumperMat : this.blueBumperMat;

    // 1. Lower Bumper (30" x 30" x 5", Y: 1 to 6)
    const bumperMesh = new THREE.Mesh(this.bumperGeom, bumperMat);
    bumperMesh.name = 'bumper';
    bumperMesh.castShadow = true;
    bumperMesh.receiveShadow = true;
    group.add(bumperMesh);

    // 2. Front Bumper Accent Line (+X face)
    const frontAccent = new THREE.Mesh(this.frontAccentGeom, this.accentMat);
    frontAccent.name = 'front_accent';
    group.add(frontAccent);

    // 3. Main Dark Metallic Titanium Chassis
    const chassisMesh = new THREE.Mesh(this.chassisGeom, this.chassisMat);
    chassisMesh.name = 'chassis';
    chassisMesh.castShadow = true;
    chassisMesh.receiveShadow = true;
    group.add(chassisMesh);

    // 4. Elevator / Superstructure Upright Mast Tubes rising to 28"
    const mastLeft = new THREE.Mesh(this.mastGeom, this.mastMat);
    mastLeft.position.set(-4.0, 0, -10.0);
    mastLeft.castShadow = true;
    group.add(mastLeft);

    const mastRight = new THREE.Mesh(this.mastGeom, this.mastMat);
    mastRight.position.set(-4.0, 0, 10.0);
    mastRight.castShadow = true;
    group.add(mastRight);

    const mastTop = new THREE.Mesh(this.mastTopGeom, this.mastMat);
    mastTop.position.set(-4.0, 0, 0);
    group.add(mastTop);

    // 5. Directional Heading Arrow on Chassis Top Deck
    const arrowMesh = new THREE.Mesh(this.arrowGeom, this.arrowMat);
    arrowMesh.name = 'heading_arrow';
    group.add(arrowMesh);

    // 6. 3D Billboard Sprite above Robot (Y = 38")
    const { sprite, update: updateBillboard } = this._createBillboardSprite(teamNumber, trackId, alliance);
    group.add(sprite);

    // Store metadata in userData for raycasting / inspection
    group.userData = {
      trackId,
      teamNumber,
      alliance,
      isRobot: true
    };

    return {
      trackId,
      teamNumber,
      alliance,
      group,
      bumperMesh,
      sprite,
      updateBillboard,
      targetPosition: new THREE.Vector3(),
      targetQuaternion: new THREE.Quaternion(),
      currentSpeed: 0.0,
      currentAction: 'IDLE',
      lastSeenMs: Date.now()
    };
  }

  /**
   * Updates all active robots from state telemetry with smooth interpolation.
   *
   * @param {Array<object>|object} robotStates - Array of robot state dictionaries
   * @param {number} [interpolationAlpha=1.0] - Lerp/slerp factor (0.0 to 1.0; 1.0 = instant snap)
   */
  updateRobots(robotStates, interpolationAlpha = 1.0) {
    if (!robotStates) return;

    const statesList = Array.isArray(robotStates)
      ? robotStates
      : Object.values(robotStates);

    const now = Date.now();
    const activeTrackIds = new Set();
    const THREE = this.THREE;

    for (const state of statesList) {
      const trackId = state.track_id !== undefined ? state.track_id : state.trackId;
      if (trackId === undefined || trackId === null) continue;

      activeTrackIds.add(trackId);
      let robot = this.robots.get(trackId);

      const teamNumber = state.team_number !== undefined ? state.team_number : state.teamNumber;
      const alliance = (state.alliance || 'RED').toUpperCase();

      let wasHidden = false;

      // Create new robot mesh if encountering this track ID for the first time
      if (!robot) {
        robot = this._createRobotMesh(trackId, teamNumber, alliance);
        this.container.add(robot.group);
        this.robots.set(trackId, robot);
        wasHidden = true;
      } else {
        wasHidden = !robot.group.visible;
        // Restore visibility now that it is detected in this frame
        robot.group.visible = true;

        // Update alliance material if changed
        if (state.alliance && state.alliance !== robot.alliance) {
          robot.alliance = alliance;
          robot.bumperMesh.material = (alliance === 'RED')
            ? this.redBumperMat
            : (alliance === 'BLUE' ? this.blueBumperMat : this.grayBumperMat);
        }
        if (teamNumber && teamNumber !== robot.teamNumber) {
          robot.teamNumber = teamNumber;
          robot.group.userData.teamNumber = teamNumber;
        }
      }

      robot.lastSeenMs = now;

      // Extract field coordinates (inches)
      let posX = 0.0;
      let posZ = 0.0;
      let posY = (state.field_z_inch !== undefined && state.field_z_inch !== null) ? state.field_z_inch : (state.z || 0.0);

      if (state.center_rel_x_inch !== undefined && state.center_rel_y_inch !== undefined) {
        // Direct field-center relative coordinates matching Three.js world coordinates
        posX = state.center_rel_x_inch;
        posZ = state.center_rel_y_inch;
      } else {
        const fieldX = state.field_x_inch !== undefined ? state.field_x_inch : (state.x || 325.6);
        const fieldY = state.field_y_inch !== undefined ? state.field_y_inch : (state.y || 158.85);
        const p3d = to3DCoords(fieldX, fieldY, posY);
        posX = p3d.x;
        posZ = p3d.z;
      }
      robot.targetPosition.set(posX, posY, posZ);

      // Live sizing: scale robot 3D model according to live/nominal dimensions (Nominal: 22" W x 28" L x 28" H)
      const robotWidth = state.robot_width_inch || 22.0;
      const robotLength = state.robot_length_inch || 28.0;
      const robotHeight = state.robot_height_inch || 28.0;
      // Default template is 30" L x 28" H x 30" W
      robot.group.scale.set(robotLength / 30.0, robotHeight / 28.0, robotWidth / 30.0);

      // Heading rotation around vertical Y axis:
      // In local coordinates, robot front is +X.
      // Heading angle 0 rad = +X, PI/2 = +Z, PI = -X.
      // Three.js Y-rotation formula: rotY = -headingRad
      const headingRad = state.heading_rad !== undefined ? state.heading_rad : (state.heading || 0.0);
      robot.targetQuaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), -headingRad);

      // Smooth Position & Orientation Interpolation:
      // If robot just reappeared after being hidden, snap immediately to prevent cross-field lerping!
      const alpha = wasHidden ? 1.0 : Math.min(1.0, Math.max(0.01, interpolationAlpha));
      if (alpha >= 0.99) {
        robot.group.position.copy(robot.targetPosition);
        robot.group.quaternion.copy(robot.targetQuaternion);
      } else {
        robot.group.position.lerp(robot.targetPosition, alpha);
        robot.group.quaternion.slerp(robot.targetQuaternion, alpha);
      }

      // Compute speed
      let speedIps = state.speed_ips;
      if (speedIps === undefined) {
        const vx = state.velocity_x_ips !== undefined ? state.velocity_x_ips : (state.vx || 0.0);
        const vy = state.velocity_y_ips !== undefined ? state.velocity_y_ips : (state.vy || 0.0);
        speedIps = Math.hypot(vx, vy);
      }
      robot.currentSpeed = speedIps;

      // Current action badge
      const actionType = state.action_type || state.action || (state.in_defense ? 'DEFENSE' : 'IDLE');
      robot.currentAction = actionType;

      // Update Billboard Sprite HUD
      robot.updateBillboard(robot.teamNumber, speedIps, actionType, robot.alliance);
    }

    // Dynamic Visibility Rule:
    // When a specific bot is no longer found on screen in this frame,
    // rather than causing it to stand still on the web viewer, make it disappear
    // if another bot is detected.
    const hasAnyDetections = activeTrackIds.size > 0;
    if (hasAnyDetections) {
      for (const [trackId, robot] of this.robots) {
        if (!activeTrackIds.has(trackId)) {
          robot.group.visible = false;
        }
      }
    }

    // Update Selection Highlight Box Position if a robot is selected
    if (this.selectedTrackId !== null) {
      const selectedRobot = this.robots.get(this.selectedTrackId);
      if (selectedRobot && selectedRobot.group.visible) {
        this.selectionBox.visible = true;
        this.selectionBox.position.copy(selectedRobot.group.position);
        this.selectionBox.quaternion.copy(selectedRobot.group.quaternion);
      } else {
        this.selectionBox.visible = false;
      }
    }
  }

  /**
   * Selects a robot by its tracking ID, enabling the glowing wireframe highlight box.
   *
   * @param {number|string} trackId - Target robot tracking ID
   * @returns {boolean} True if robot was found and selected
   */
  selectRobot(trackId) {
    if (this.robots.has(trackId)) {
      const robot = this.robots.get(trackId);
      if (!robot.group.visible) {
        return false;
      }
      this.selectedTrackId = trackId;
      this.selectionBox.visible = true;
      this.selectionBox.position.copy(robot.group.position);
      this.selectionBox.quaternion.copy(robot.group.quaternion);
      return true;
    }
    return false;
  }

  /**
   * Returns details for the currently selected robot.
   *
   * @returns {object|null} Selected robot state and mesh reference or null
   */
  getSelectedRobot() {
    if (this.selectedTrackId === null) return null;
    const robot = this.robots.get(this.selectedTrackId);
    if (!robot) return null;

    return {
      trackId: robot.trackId,
      teamNumber: robot.teamNumber,
      alliance: robot.alliance,
      position: robot.group.position.clone(),
      speed: robot.currentSpeed,
      action: robot.currentAction,
      group: robot.group
    };
  }

  /**
   * Clears current robot selection, hiding the wireframe box.
   */
  clearSelection() {
    this.selectedTrackId = null;
    this.selectionBox.visible = false;
  }

  /**
   * Performs raycasting against all active robot bumper and chassis meshes.
   * Useful for mouse/touch click-to-select interactions.
   *
   * @param {THREE.Raycaster} raycaster - Configured Three.js Raycaster
   * @returns {number|string|null} Intersected robot trackId or null
   */
  raycast(raycaster) {
    if (!raycaster) return null;

    // Ensure all world matrices are up-to-date
    this.container.updateMatrixWorld(true);

    const interactiveMeshes = [];
    this.robots.forEach((robot) => {
      if (robot.group.visible) {
        robot.group.traverse((child) => {
          if (child.isMesh && (child.name === 'bumper' || child.name === 'chassis')) {
            interactiveMeshes.push(child);
          }
        });
      }
    });

    const intersects = raycaster.intersectObjects(interactiveMeshes, false);
    if (intersects.length > 0) {
      let parent = intersects[0].object.parent;
      while (parent && !parent.userData?.isRobot && parent !== this.container) {
        parent = parent.parent;
      }
      if (parent && parent.userData?.isRobot) {
        const hitId = parent.userData.trackId;
        this.selectRobot(hitId);
        return hitId;
      }
    }
    return null;
  }

  /**
   * Retrieves a robot entry by track ID.
   *
   * @param {number|string} trackId
   * @returns {object|undefined}
   */
  getRobot(trackId) {
    return this.robots.get(trackId);
  }

  /**
   * Removes a robot by track ID from scene and cache.
   *
   * @param {number|string} trackId
   */
  removeRobot(trackId) {
    const robot = this.robots.get(trackId);
    if (robot) {
      if (this.selectedTrackId === trackId) {
        this.clearSelection();
      }
      this.container.remove(robot.group);
      this.robots.delete(trackId);
    }
  }

  /**
   * Disposes all robot geometries, materials, and textures.
   */
  dispose() {
    this.clearSelection();
    this.robots.forEach((robot) => {
      this.container.remove(robot.group);
    });
    this.robots.clear();
    this.scene.remove(this.container);
  }
}

export default RobotManager;
