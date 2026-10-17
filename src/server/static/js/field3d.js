/**
 * FRC 2026 REBUILT: 3D Field CAD & Kinematics Environment
 * Three.js 3D Field Module
 *
 * Coordinate System:
 * 1 unit = 1 inch, origin (0, 0, 0) at center of field carpet:
 *   X_3D = X_inch - 325.6  (Length: -325.6 to +325.6, total 651.2")
 *   Z_3D = Y_inch - 158.85 (Width: -158.85 to +158.85, total 317.7")
 *   Y_3D = Z_inch          (Height above floor: 0.0 is carpet floor, 72.0 is Upper Hub rim)
 */

// Global/import resolver for THREE
function getTHREE(customThree) {
  if (customThree) return customThree;
  if (typeof THREE !== 'undefined') return THREE;
  if (typeof window !== 'undefined' && window.THREE) return window.THREE;
  if (typeof globalThis !== 'undefined' && globalThis.THREE) return globalThis.THREE;
  if (typeof global !== 'undefined' && global.THREE) return global.THREE;
  throw new Error('THREE is not defined. Ensure three.min.js is included or pass customThree instance.');
}

// Field Dimensions (inches)
export const FIELD_DIMENSIONS = {
  LENGTH: 651.2,
  WIDTH: 317.7,
  HALF_LENGTH: 325.6,
  HALF_WIDTH: 158.85,
  AUTO_LINE_OFFSET: 205.6, // X = ±205.6 in 3D (X_inch = 120.0 and 531.2)
  GUARDRAIL_HEIGHT: 20.0,
  ENDWALL_HEIGHT: 78.0,
  HUB_HEIGHT: 72.0,
  HUB_HEX_DIAMETER: 41.7,
  HUB_RED_X_INCH: 162.8,
  HUB_BLUE_X_INCH: 488.4,
  TOWER_RED_X_INCH: 90.0,
  TOWER_BLUE_X_INCH: 561.2,
  TOWER_HEIGHT: 70.0,
  TOWER_RUNG_ELEVATION: 65.0,
  TAG_SIZE: 8.125
};

/**
 * Converts 2D/3D field coordinates (inches) to Three.js world coordinates.
 *
 * @param {number} xInch - Field length coordinate (0 to 651.2)
 * @param {number} yInch - Field width coordinate (0 to 317.7)
 * @param {number} [zInch=0.0] - Field elevation coordinate (0 is floor)
 * @returns {{x: number, y: number, z: number}} Coordinates in Three.js space
 */
export function to3DCoords(xInch, yInch, zInch = 0.0) {
  return {
    x: xInch - FIELD_DIMENSIONS.HALF_LENGTH,
    y: zInch,
    z: yInch - FIELD_DIMENSIONS.HALF_WIDTH
  };
}

/**
 * Converts Three.js world coordinates back to field inches.
 *
 * @param {number} x3D - Three.js X
 * @param {number} z3D - Three.js Z
 * @param {number} [y3D=0.0] - Three.js Y (elevation)
 * @returns {{xInch: number, yInch: number, zInch: number}}
 */
export function toFieldCoords(x3D, z3D, y3D = 0.0) {
  return {
    xInch: x3D + FIELD_DIMENSIONS.HALF_LENGTH,
    yInch: z3D + FIELD_DIMENSIONS.HALF_WIDTH,
    zInch: y3D
  };
}

// Default 32 AprilTags ground truth if fieldConfig is omitted or incomplete (matching field.glb CAD)
export const DEFAULT_APRILTAGS = [
  // Red Hub & Trench Assembly (Tags 1-12)
  { id:  1, x: 183.33, y:  25.38, z: 35.00, roll: 0.0, pitch: 0.0, yaw:  90.4 },
  { id:  2, x: 182.10, y: 135.10, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 243.2 },
  { id:  3, x: 205.85, y: 144.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 151.4 },
  { id:  4, x: 205.62, y: 158.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 151.4 },
  { id:  5, x: 182.10, y: 182.60, z: 44.25, roll: 0.0, pitch: 0.0, yaw:   0.0 },
  { id:  6, x: 183.33, y: 292.32, z: 35.00, roll: 0.0, pitch: 0.0, yaw: 184.5 },
  { id:  7, x: 180.64, y: 292.32, z: 35.00, roll: 0.0, pitch: 0.0, yaw: 184.5 },
  { id:  8, x: 168.10, y: 182.36, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 180.0 },
  { id:  9, x: 158.34, y: 172.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 176.3 },
  { id: 10, x: 158.58, y: 158.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 176.3 },
  { id: 11, x: 168.10, y: 135.34, z: 44.25, roll: 0.0, pitch: 0.0, yaw:   0.0 },
  { id: 12, x: 180.64, y:  25.38, z: 35.00, roll: 0.0, pitch: 0.0, yaw:  89.9 },
  // Red Driver Station Wall (Tags 13-16)
  { id: 13, x:   0.05, y:  26.23, z: 21.75, roll: 0.0, pitch: 0.0, yaw:  98.4 },
  { id: 14, x:   0.28, y:  43.22, z: 21.75, roll: 0.0, pitch: 0.0, yaw:  98.4 },
  { id: 15, x:   0.06, y: 147.48, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 270.0 },
  { id: 16, x:   0.06, y: 164.48, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 270.0 },
  // Blue Hub & Trench Assembly (Tags 17-28)
  { id: 17, x: 467.64, y: 292.32, z: 35.00, roll: 0.0, pitch: 0.0, yaw:  61.9 },
  { id: 18, x: 469.10, y: 182.60, z: 44.25, roll: 0.0, pitch: 0.0, yaw:   0.0 },
  { id: 19, x: 445.35, y: 172.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 331.4 },
  { id: 20, x: 445.58, y: 158.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 331.4 },
  { id: 21, x: 469.10, y: 135.10, z: 44.25, roll: 0.0, pitch: 0.0, yaw:   0.0 },
  { id: 22, x: 467.87, y:  25.38, z: 35.00, roll: 0.0, pitch: 0.0, yaw:  85.7 },
  { id: 23, x: 470.33, y:  25.38, z: 35.00, roll: 0.0, pitch: 0.0, yaw: 263.2 },
  { id: 24, x: 483.10, y: 135.34, z: 44.25, roll: 0.0, pitch: 0.0, yaw:   0.0 },
  { id: 25, x: 492.62, y: 144.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 356.3 },
  { id: 26, x: 492.86, y: 158.85, z: 44.25, roll: 0.0, pitch: 0.0, yaw: 356.3 },
  { id: 27, x: 483.10, y: 182.36, z: 44.25, roll: 0.0, pitch: 0.0, yaw:  62.1 },
  { id: 28, x: 470.33, y: 292.32, z: 35.00, roll: 0.0, pitch: 0.0, yaw:  61.9 },
  // Blue Driver Station Wall (Tags 29-32)
  { id: 29, x: 651.15, y: 291.47, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 349.0 },
  { id: 30, x: 650.92, y: 274.48, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 349.0 },
  { id: 31, x: 651.14, y: 170.22, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 210.7 },
  { id: 32, x: 650.90, y: 153.22, z: 21.75, roll: 0.0, pitch: 0.0, yaw: 210.7 }
];

/**
 * Procedurally generates the 2D canvas texture for the FRC 2026 REBUILT field carpet.
 * Includes dark grey/green canvas grain, white border tape, center line, auto starting lines,
 * and subtle Red & Blue alliance zone color washes.
 */
function createCarpetTexture(THREE) {
  if (typeof document === 'undefined') {
    return null;
  }

  const canvasWidth = 2048;
  const canvasHeight = Math.round(2048 * (FIELD_DIMENSIONS.WIDTH / FIELD_DIMENSIONS.LENGTH)); // ~1000px
  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const scaleX = canvasWidth / FIELD_DIMENSIONS.LENGTH;
  const scaleY = canvasHeight / FIELD_DIMENSIONS.WIDTH;

  // 1. Base Carpet: Dark charcoal green (#1a251e)
  ctx.fillStyle = '#1c2822';
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);

  // Subtle carpet weave noise
  ctx.fillStyle = 'rgba(255, 255, 255, 0.015)';
  for (let i = 0; i < 30000; i++) {
    const rx = Math.random() * canvasWidth;
    const ry = Math.random() * canvasHeight;
    ctx.fillRect(rx, ry, 1, 1);
  }

  // 2. Alliance Zone Tints
  // Red Zone: X_3D from -325.6 to -205.6 => X_inch from 0 to 120.0
  const redZoneWidthPx = 120.0 * scaleX;
  const redGradient = ctx.createLinearGradient(0, 0, redZoneWidthPx, 0);
  redGradient.addColorStop(0, 'rgba(239, 68, 68, 0.22)');
  redGradient.addColorStop(0.85, 'rgba(239, 68, 68, 0.14)');
  redGradient.addColorStop(1, 'rgba(239, 68, 68, 0.04)');
  ctx.fillStyle = redGradient;
  ctx.fillRect(0, 0, redZoneWidthPx, canvasHeight);

  // Blue Zone: X_3D from +205.6 to +325.6 => X_inch from 531.2 to 651.2
  const blueZoneStartPx = 531.2 * scaleX;
  const blueZoneWidthPx = canvasWidth - blueZoneStartPx;
  const blueGradient = ctx.createLinearGradient(canvasWidth, 0, blueZoneStartPx, 0);
  blueGradient.addColorStop(0, 'rgba(59, 130, 246, 0.22)');
  blueGradient.addColorStop(0.85, 'rgba(59, 130, 246, 0.14)');
  blueGradient.addColorStop(1, 'rgba(59, 130, 246, 0.04)');
  ctx.fillStyle = blueGradient;
  ctx.fillRect(blueZoneStartPx, 0, blueZoneWidthPx, canvasHeight);

  // 3. Neutral Zone Subtle Highlight
  ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
  ctx.fillRect(redZoneWidthPx, 0, blueZoneStartPx - redZoneWidthPx, canvasHeight);

  // 4. Field Lines (2-inch white tape)
  const tapeWidthPx = Math.max(2, Math.round(2.0 * scaleX));
  ctx.fillStyle = '#f3f4f6';

  // Perimeter border tape (2" inset)
  ctx.lineWidth = tapeWidthPx;
  ctx.strokeStyle = '#f3f4f6';
  ctx.strokeRect(tapeWidthPx / 2, tapeWidthPx / 2, canvasWidth - tapeWidthPx, canvasHeight - tapeWidthPx);

  // Center line (X_3D = 0 => X_inch = 325.6)
  const centerX = Math.round(FIELD_DIMENSIONS.HALF_LENGTH * scaleX);
  ctx.fillRect(centerX - tapeWidthPx / 2, 0, tapeWidthPx, canvasHeight);

  // Red Auto Starting Line (X_3D = -205.6 => X_inch = 120.0)
  const redAutoX = Math.round(120.0 * scaleX);
  ctx.fillRect(redAutoX - tapeWidthPx / 2, 0, tapeWidthPx, canvasHeight);

  // Blue Auto Starting Line (X_3D = +205.6 => X_inch = 531.2)
  const blueAutoX = Math.round(531.2 * scaleX);
  ctx.fillRect(blueAutoX - tapeWidthPx / 2, 0, tapeWidthPx, canvasHeight);

  // Subtle Alliance text labels on carpet
  ctx.font = 'bold 36px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = 'rgba(239, 68, 68, 0.35)';
  ctx.fillText('RED ALLIANCE', redZoneWidthPx * 0.5, canvasHeight * 0.5);

  ctx.fillStyle = 'rgba(59, 130, 246, 0.35)';
  ctx.fillText('BLUE ALLIANCE', blueZoneStartPx + blueZoneWidthPx * 0.5, canvasHeight * 0.5);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.anisotropy = 16;
  return texture;
}

/**
 * Creates an AprilTag canvas texture for a specific tag ID.
 * Features a black outer border, white fiducial margin, high-contrast interior matrix,
 * and clear tag ID numeral.
 */
function createAprilTagTexture(THREE, tagId) {
  if (typeof document === 'undefined') {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  // Solid black outer border (family tag36h11 spec)
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, 128, 128);

  // White quiet zone margin
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(16, 16, 96, 96);

  // Black interior frame
  ctx.fillStyle = '#000000';
  ctx.fillRect(26, 26, 76, 76);

  // Seeded pseudo-fiducial bits pattern for high visual fidelity
  let seed = (tagId * 2654435761) >>> 0;
  const nextBit = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return (seed >>> 30) & 1;
  };

  const gridSize = 6;
  const cellSize = 54 / gridSize;
  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      if (nextBit()) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(37 + col * cellSize, 37 + row * cellSize, cellSize, cellSize);
      }
    }
  }

  // Tag ID label badge in bottom corner
  ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
  ctx.fillRect(30, 96, 68, 18);
  ctx.fillStyle = '#facc15';
  ctx.font = 'bold 13px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`#${tagId}`, 64, 105);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

/**
 * Builds the Upper Hub assembly (cone, catch basin, pedestal, emissive alliance LED ring).
 *
 * @param {object} THREE - Three.js namespace
 * @param {string} alliance - 'RED' | 'BLUE'
 * @param {number} x3D - Center X in Three.js coordinates
 * @param {number} z3D - Center Z in Three.js coordinates
 * @returns {THREE.Group} Upper Hub 3D Object Group
 */
function createUpperHub(THREE, alliance, x3D, z3D) {
  const hubGroup = new THREE.Group();
  hubGroup.name = `${alliance.toLowerCase()}_hub`;
  hubGroup.position.set(x3D, 0, z3D);

  const isRed = alliance.toUpperCase() === 'RED';
  const allianceHex = isRed ? 0xef4444 : 0x3b82f6;
  const emissiveHex = isRed ? 0xff2222 : 0x2277ff;

  // 1. Base Column / Sturdy Central Pedestal
  // Carpet (Y=0) up to catch basin base (Y=38)
  const baseGeom = new THREE.CylinderGeometry(10.0, 14.0, 38.0, 16);
  const baseMat = new THREE.MeshStandardMaterial({
    color: 0x27272a,
    metalness: 0.85,
    roughness: 0.25
  });
  const baseMesh = new THREE.Mesh(baseGeom, baseMat);
  baseMesh.position.y = 19.0;
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  hubGroup.add(baseMesh);

  // Heavy steel ground anchoring flange
  const flangeGeom = new THREE.CylinderGeometry(24.0, 26.0, 2.0, 24);
  const flangeMesh = new THREE.Mesh(flangeGeom, baseMat);
  flangeMesh.position.y = 1.0;
  flangeMesh.receiveShadow = true;
  hubGroup.add(flangeMesh);

  // 2. Catch Basin Cylinder (Y=38 to Y=52)
  const basinGeom = new THREE.CylinderGeometry(23.5, 20.0, 14.0, 12, 1, true);
  const basinMat = new THREE.MeshStandardMaterial({
    color: 0x3f3f46,
    metalness: 0.6,
    roughness: 0.35,
    side: THREE.DoubleSide
  });
  const basinMesh = new THREE.Mesh(basinGeom, basinMat);
  basinMesh.position.y = 45.0;
  basinMesh.castShadow = true;
  hubGroup.add(basinMesh);

  // 3. Hexagonal Cone / Funnel (Y=52 to Y=72, rim at Y=72")
  // Opening diameter 41.7" => radius = 20.85"
  const rimRadius = FIELD_DIMENSIONS.HUB_HEX_DIAMETER / 2.0; // 20.85"
  const throatRadius = 12.5; // lower funnel neck
  const coneHeight = 20.0;
  const funnelGeom = new THREE.CylinderGeometry(rimRadius, throatRadius, coneHeight, 6, 1, true);
  const funnelMat = new THREE.MeshStandardMaterial({
    color: 0x18181b,
    metalness: 0.7,
    roughness: 0.3,
    side: THREE.DoubleSide
  });
  const funnelMesh = new THREE.Mesh(funnelGeom, funnelMat);
  funnelMesh.position.y = 62.0; // center of 52 to 72
  funnelMesh.castShadow = true;
  hubGroup.add(funnelMesh);

  // 4. Emissive LED Alliance Ring around Top Rim (Y = 72")
  const torusGeom = new THREE.TorusGeometry(rimRadius, 0.9, 16, 6);
  torusGeom.rotateX(Math.PI / 2);
  const ledMat = new THREE.MeshStandardMaterial({
    color: allianceHex,
    emissive: emissiveHex,
    emissiveIntensity: 1.8,
    roughness: 0.15,
    metalness: 0.1
  });
  const ledRing = new THREE.Mesh(torusGeom, ledMat);
  ledRing.position.y = FIELD_DIMENSIONS.HUB_HEIGHT;
  hubGroup.add(ledRing);

  // Point light providing localized alliance glow around the Hub opening
  const hubLight = new THREE.PointLight(emissiveHex, 0.9, 150, 1.5);
  hubLight.position.set(0, FIELD_DIMENSIONS.HUB_HEIGHT + 2, 0);
  hubGroup.add(hubLight);

  return hubGroup;
}

/**
 * Builds an Alliance Climbing Tower (4-column truss rising to 70" with climbing rungs).
 *
 * @param {object} THREE - Three.js namespace
 * @param {string} alliance - 'RED' | 'BLUE'
 * @param {number} x3D - Center X in Three.js coordinates
 * @param {number} z3D - Center Z in Three.js coordinates
 * @returns {THREE.Group} Tower 3D Object Group
 */
function createTower(THREE, alliance, x3D, z3D) {
  const towerGroup = new THREE.Group();
  towerGroup.name = `${alliance.toLowerCase()}_tower`;
  towerGroup.position.set(x3D, 0, z3D);

  const isRed = alliance.toUpperCase() === 'RED';
  const allianceHex = isRed ? 0xef4444 : 0x3b82f6;

  const trussMat = new THREE.MeshStandardMaterial({
    color: 0x71717a,
    metalness: 0.85,
    roughness: 0.25
  });
  const rungMat = new THREE.MeshStandardMaterial({
    color: 0xe4e4e7,
    metalness: 0.9,
    roughness: 0.15
  });

  const towerHalfW = 18.0; // 36" wide base
  const towerHalfD = 18.0;
  const towerHeight = FIELD_DIMENSIONS.TOWER_HEIGHT; // 70"

  // 4 Vertical Corner Truss Columns
  const columnGeom = new THREE.CylinderGeometry(1.5, 1.5, towerHeight, 12);
  const corners = [
    [-towerHalfW, -towerHalfD],
    [towerHalfW, -towerHalfD],
    [towerHalfW, towerHalfD],
    [-towerHalfW, towerHalfD]
  ];

  corners.forEach(([cx, cz]) => {
    const colMesh = new THREE.Mesh(columnGeom, trussMat);
    colMesh.position.set(cx, towerHeight / 2.0, cz);
    colMesh.castShadow = true;
    towerGroup.add(colMesh);
  });

  // Climbing Horizontal Rungs across side faces
  const rungElevations = [28.0, 48.0, FIELD_DIMENSIONS.TOWER_RUNG_ELEVATION];
  const rungGeomX = new THREE.CylinderGeometry(1.0, 1.0, towerHalfW * 2, 12);
  rungGeomX.rotateZ(Math.PI / 2);

  const rungGeomZ = new THREE.CylinderGeometry(1.0, 1.0, towerHalfD * 2, 12);
  rungGeomZ.rotateX(Math.PI / 2);

  rungElevations.forEach((elev) => {
    // Front and Back rungs
    const frontRung = new THREE.Mesh(rungGeomX, rungMat);
    frontRung.position.set(0, elev, towerHalfD);
    frontRung.castShadow = true;
    towerGroup.add(frontRung);

    const backRung = new THREE.Mesh(rungGeomX, rungMat);
    backRung.position.set(0, elev, -towerHalfD);
    backRung.castShadow = true;
    towerGroup.add(backRung);

    // Left and Right rungs
    const leftRung = new THREE.Mesh(rungGeomZ, rungMat);
    leftRung.position.set(-towerHalfW, elev, 0);
    leftRung.castShadow = true;
    towerGroup.add(leftRung);

    const rightRung = new THREE.Mesh(rungGeomZ, rungMat);
    rightRung.position.set(towerHalfW, elev, 0);
    rightRung.castShadow = true;
    towerGroup.add(rightRung);
  });

  // Alliance Cap / Banner on top of tower
  const capGeom = new THREE.BoxGeometry(towerHalfW * 2 + 4, 3.0, towerHalfD * 2 + 4);
  const capMat = new THREE.MeshStandardMaterial({
    color: allianceHex,
    emissive: allianceHex,
    emissiveIntensity: 0.4,
    metalness: 0.5,
    roughness: 0.3
  });
  const capMesh = new THREE.Mesh(capGeom, capMat);
  capMesh.position.set(0, towerHeight + 1.5, 0);
  towerGroup.add(capMesh);

  return towerGroup;
}

/**
 * Builds Perimeter Guardrails and Alliance Driver Station End Walls.
 * Side rails: 20" transparent polycarbonate panels + aluminum posts.
 * End walls: 78" height with Red (#ef4444) and Blue (#3b82f6) translucent acrylic panels.
 */
function createPerimeter(THREE) {
  const perimeterGroup = new THREE.Group();
  perimeterGroup.name = 'field_perimeter';

  const halfL = FIELD_DIMENSIONS.HALF_LENGTH; // 325.6
  const halfW = FIELD_DIMENSIONS.HALF_WIDTH;  // 158.85
  const railH = FIELD_DIMENSIONS.GUARDRAIL_HEIGHT; // 20.0
  const wallH = FIELD_DIMENSIONS.ENDWALL_HEIGHT;   // 78.0

  // Materials
  const polycarbMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    transparent: true,
    opacity: 0.35,
    roughness: 0.1,
    metalness: 0.15,
    side: THREE.DoubleSide
  });

  const aluminumMat = new THREE.MeshStandardMaterial({
    color: 0xd1d5db,
    metalness: 0.85,
    roughness: 0.25
  });

  const redWallMat = new THREE.MeshStandardMaterial({
    color: 0xef4444,
    emissive: 0xef4444,
    emissiveIntensity: 0.15,
    transparent: true,
    opacity: 0.48,
    roughness: 0.2,
    metalness: 0.1,
    side: THREE.DoubleSide
  });

  const blueWallMat = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    emissive: 0x3b82f6,
    emissiveIntensity: 0.15,
    transparent: true,
    opacity: 0.48,
    roughness: 0.2,
    metalness: 0.1,
    side: THREE.DoubleSide
  });

  // --- 1. Long Side Guardrails (North: +Z, South: -Z) ---
  const sideLength = FIELD_DIMENSIONS.LENGTH;
  const sideGeom = new THREE.BoxGeometry(sideLength, railH, 0.75);

  // South side rail (-Z)
  const southRail = new THREE.Mesh(sideGeom, polycarbMat);
  southRail.position.set(0, railH / 2.0, -halfW);
  perimeterGroup.add(southRail);

  // North side rail (+Z)
  const northRail = new THREE.Mesh(sideGeom, polycarbMat);
  northRail.position.set(0, railH / 2.0, halfW);
  perimeterGroup.add(northRail);

  // Top Aluminum Extrusions for Side Guardrails
  const sideTopGeom = new THREE.BoxGeometry(sideLength, 1.5, 1.5);
  const southTop = new THREE.Mesh(sideTopGeom, aluminumMat);
  southTop.position.set(0, railH + 0.75, -halfW);
  perimeterGroup.add(southTop);

  const northTop = new THREE.Mesh(sideTopGeom, aluminumMat);
  northTop.position.set(0, railH + 0.75, halfW);
  perimeterGroup.add(northTop);

  // Aluminum Posts along Sides (every ~46.5 inches)
  const postGeom = new THREE.BoxGeometry(1.75, railH + 1.5, 1.75);
  const postCount = 15;
  for (let i = 0; i <= postCount; i++) {
    const postX = -halfL + i * (sideLength / postCount);

    const postSouth = new THREE.Mesh(postGeom, aluminumMat);
    postSouth.position.set(postX, (railH + 1.5) / 2.0, -halfW);
    postSouth.castShadow = true;
    perimeterGroup.add(postSouth);

    const postNorth = new THREE.Mesh(postGeom, aluminumMat);
    postNorth.position.set(postX, (railH + 1.5) / 2.0, halfW);
    postNorth.castShadow = true;
    perimeterGroup.add(postNorth);
  }

  // --- 2. Alliance Driver Station End Walls (78" height) ---
  const wallWidth = FIELD_DIMENSIONS.WIDTH;
  const endWallGeom = new THREE.BoxGeometry(1.0, wallH, wallWidth);

  // Red Alliance End Wall (X = -325.6)
  const redWall = new THREE.Mesh(endWallGeom, redWallMat);
  redWall.position.set(-halfL, wallH / 2.0, 0);
  redWall.receiveShadow = true;
  perimeterGroup.add(redWall);

  // Blue Alliance End Wall (X = +325.6)
  const blueWall = new THREE.Mesh(endWallGeom, blueWallMat);
  blueWall.position.set(halfL, wallH / 2.0, 0);
  blueWall.receiveShadow = true;
  perimeterGroup.add(blueWall);

  // Top and Bottom Framing on End Walls
  const endFrameGeom = new THREE.BoxGeometry(2.0, 2.0, wallWidth);

  const redTopFrame = new THREE.Mesh(endFrameGeom, aluminumMat);
  redTopFrame.position.set(-halfL, wallH + 1.0, 0);
  perimeterGroup.add(redTopFrame);

  const blueTopFrame = new THREE.Mesh(endFrameGeom, aluminumMat);
  blueTopFrame.position.set(halfL, wallH + 1.0, 0);
  perimeterGroup.add(blueTopFrame);

  // Driver Station Vertical Dividers (Stations 1, 2, 3)
  const dividerGeom = new THREE.BoxGeometry(2.0, wallH, 2.0);
  const stationDivs = [-halfW / 3.0, halfW / 3.0];
  stationDivs.forEach((divZ) => {
    const redDiv = new THREE.Mesh(dividerGeom, aluminumMat);
    redDiv.position.set(-halfL, wallH / 2.0, divZ);
    perimeterGroup.add(redDiv);

    const blueDiv = new THREE.Mesh(dividerGeom, aluminumMat);
    blueDiv.position.set(halfL, wallH / 2.0, divZ);
    perimeterGroup.add(blueDiv);
  });

  return perimeterGroup;
}

/**
 * Builds Depots, Mid-field Trenches, and Bumps based on field topology.
 */
function createDepotsAndTrenches(THREE, fieldConfig = {}) {
  const group = new THREE.Group();
  group.name = 'depots_and_trenches';

  const depotMat = new THREE.MeshStandardMaterial({
    color: 0x475569,
    metalness: 0.6,
    roughness: 0.4
  });
  const trenchMat = new THREE.MeshStandardMaterial({
    color: 0x334155,
    metalness: 0.8,
    roughness: 0.25
  });
  const bumpMat = new THREE.MeshStandardMaterial({
    color: 0x52525b,
    roughness: 0.6,
    metalness: 0.3
  });

  // 1. Depots: North (325.6, 280.0) & South (325.6, 37.7) in field inches
  // Size: [36.0, 24.0]
  const depotsConfig = fieldConfig?.elements?.depots || [
    { name: 'depot_north', center: [325.6, 280.0], size: [36.0, 24.0] },
    { name: 'depot_south', center: [325.6, 37.7],  size: [36.0, 24.0] }
  ];

  depotsConfig.forEach((depot) => {
    const p3d = to3DCoords(depot.center[0], depot.center[1], 0);
    const w = depot.size[0];
    const d = depot.size[1];

    // Ramp/curb staging zone
    const depotGeom = new THREE.BoxGeometry(w, 3.5, d);
    const depotMesh = new THREE.Mesh(depotGeom, depotMat);
    depotMesh.position.set(p3d.x, 1.75, p3d.z);
    depotMesh.receiveShadow = true;
    group.add(depotMesh);

    // Accent boundary tape
    const borderGeom = new THREE.BoxGeometry(w + 1.0, 0.5, d + 1.0);
    const borderMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    const borderMesh = new THREE.Mesh(borderGeom, borderMat);
    borderMesh.position.set(p3d.x, 3.8, p3d.z);
    group.add(borderMesh);
  });

  // 2. Mid-field Trenches along sides (Y_inch = 15.0 and 262.7)
  // Trench dimensions from config: [270.0, 15.0, 110.0, 40.0]
  const trenchesConfig = fieldConfig?.elements?.trenches || [
    [270.0, 15.0, 110.0, 40.0],
    [270.0, 262.7, 110.0, 40.0]
  ];

  trenchesConfig.forEach(([xInch, yInch, length, width]) => {
    const p3d = to3DCoords(xInch, yInch, 0);

    // Overhead clearance beam structure at 28" elevation
    const beamGeom = new THREE.BoxGeometry(length, 3.0, 4.0);
    const beamMesh = new THREE.Mesh(beamGeom, trenchMat);
    beamMesh.position.set(p3d.x, 28.0, p3d.z);
    beamMesh.castShadow = true;
    group.add(beamMesh);

    // Support legs on both sides
    const legGeom = new THREE.BoxGeometry(3.0, 28.0, 3.0);
    const leg1 = new THREE.Mesh(legGeom, trenchMat);
    leg1.position.set(p3d.x - length / 2.0 + 2.0, 14.0, p3d.z);
    leg1.castShadow = true;
    group.add(leg1);

    const leg2 = new THREE.Mesh(legGeom, trenchMat);
    leg2.position.set(p3d.x + length / 2.0 - 2.0, 14.0, p3d.z);
    leg2.castShadow = true;
    group.add(leg2);
  });

  // 3. Speed Bumps
  const bumpsConfig = fieldConfig?.elements?.bumps || [
    [280.0, 80.0, 30.0, 10.0],
    [280.0, 237.7, 30.0, 10.0],
    [371.2, 80.0, 30.0, 10.0],
    [371.2, 237.7, 30.0, 10.0]
  ];

  bumpsConfig.forEach(([bx, by, bw, bd]) => {
    const bp = to3DCoords(bx, by, 0);
    const bumpGeom = new THREE.CylinderGeometry(bd / 2.0, bd / 2.0, bw, 12);
    bumpGeom.rotateZ(Math.PI / 2);
    const bumpMesh = new THREE.Mesh(bumpGeom, bumpMat);
    bumpMesh.position.set(bp.x, 1.25, bp.z);
    bumpMesh.scale.set(1.0, 0.25, 1.0); // flatten to ramp bump
    bumpMesh.receiveShadow = true;
    group.add(bumpMesh);
  });

  return group;
}

/**
 * Dynamically constructs all 32 AprilTag fiducial meshes from fieldConfig.
 * Rendered at exact ground-truth 3D positions (X, Y, Z) and yaw angles with
 * high-contrast black border and white fiducial pattern.
 */
function createAprilTags(THREE, fieldConfig = {}) {
  const tagsGroup = new THREE.Group();
  tagsGroup.name = 'apriltags';

  const tagsList = Array.isArray(fieldConfig?.apriltags)
    ? fieldConfig.apriltags
    : (fieldConfig?.apriltags?.tags || DEFAULT_APRILTAGS);
  const tagSize = fieldConfig?.tag_size_inches || fieldConfig?.apriltags?.size_inches || FIELD_DIMENSIONS.TAG_SIZE; // 8.125"

  const tagGeom = new THREE.PlaneGeometry(tagSize, tagSize);

  // Cache materials by ID to minimize texture memory
  const tagMeshes = [];

  tagsList.forEach((tag) => {
    const p3d = to3DCoords(tag.x, tag.y, tag.z);
    const yawRad = THREE.MathUtils.degToRad(tag.yaw || 0.0);

    const texture = createAprilTagTexture(THREE, tag.id);
    let tagMat;
    if (texture) {
      tagMat = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide
      });
    } else {
      // High contrast fallback for non-canvas environments
      tagMat = new THREE.MeshBasicMaterial({
        color: (tag.id % 2 === 0) ? 0x000000 : 0xffffff,
        side: THREE.DoubleSide
      });
    }

    const tagMesh = new THREE.Mesh(tagGeom, tagMat);
    tagMesh.name = `apriltag_${tag.id}`;

    // Tag yaw orientation:
    // yaw=0 faces +X; yaw=90 faces +Z; yaw=180 faces -X; yaw=270 faces -Z
    // In Three.js, PlaneGeometry normal is (0,0,1).
    // Rotating around Y by (PI/2 - yawRad) turns normal to [cos(yaw), 0, sin(yaw)].
    const rotY = Math.PI / 2.0 - yawRad;
    tagMesh.rotation.y = rotY;

    // Slight offset along normal to eliminate z-fighting on mounting surfaces
    const normalX = Math.cos(yawRad);
    const normalZ = Math.sin(yawRad);
    const offset = 0.15; // 0.15" surface standoff

    tagMesh.position.set(
      p3d.x + normalX * offset,
      p3d.y,
      p3d.z + normalZ * offset
    );

    tagMesh.userData = {
      tagId: tag.id,
      fieldCoords: { x: tag.x, y: tag.y, z: tag.z },
      yaw: tag.yaw
    };

    tagsGroup.add(tagMesh);
    tagMeshes.push(tagMesh);
  });

  return { tagsGroup, tagMeshes };
}

/**
 * Creates stadium lighting:
 * - Ambient light (intensity 0.5)
 * - Directional stadium spotlight (intensity 0.8) casting soft shadows.
 * - Balanced opposing fill light to eliminate dark spots.
 */
function createLighting(THREE) {
  const lightsGroup = new THREE.Group();
  lightsGroup.name = 'stadium_lighting';

  // 1. Ambient Light (intensity 0.5)
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  lightsGroup.add(ambientLight);

  // 2. Primary Directional Stadium Spotlight (intensity 0.8) with soft shadows
  const stadiumLight = new THREE.DirectionalLight(0xffffff, 0.8);
  stadiumLight.position.set(220, 360, 180);
  stadiumLight.castShadow = true;

  // Shadow Map Configuration
  stadiumLight.shadow.mapSize.width = 2048;
  stadiumLight.shadow.mapSize.height = 2048;
  stadiumLight.shadow.camera.near = 50;
  stadiumLight.shadow.camera.far = 800;
  stadiumLight.shadow.camera.left = -400;
  stadiumLight.shadow.camera.right = 400;
  stadiumLight.shadow.camera.top = 220;
  stadiumLight.shadow.camera.bottom = -220;
  stadiumLight.shadow.bias = -0.0005;

  lightsGroup.add(stadiumLight);

  // 3. Secondary Complementary Fill Light (intensity 0.35)
  const fillLight = new THREE.DirectionalLight(0xe2e8f0, 0.35);
  fillLight.position.set(-220, 320, -180);
  lightsGroup.add(fillLight);

  return { lightsGroup, ambientLight, stadiumLight, fillLight };
}

/**
 * Main Field CAD Factory function.
 * Constructs and adds all FRC 2026 REBUILT field elements to the given scene.
 *
 * @param {THREE.Scene|THREE.Group} scene - The Three.js target scene or parent group
 * @param {object} [fieldConfig={}] - Parsed YAML field configuration object
 * @param {object} [customThree=null] - Optional THREE instance override
 * @returns {object} Reference dictionary to all created field elements
 */
export function createFieldCAD(scene, fieldConfig = {}, customThree = null) {
  const THREE = getTHREE(customThree);
  if (!scene) {
    throw new Error('createFieldCAD requires a valid Three.js Scene or Group instance.');
  }

  const fieldGroup = new THREE.Group();
  fieldGroup.name = 'frc2026_field';

  // 1. Carpet Plane (651.2 x 317.7 inches)
  const carpetGeom = new THREE.PlaneGeometry(FIELD_DIMENSIONS.LENGTH, FIELD_DIMENSIONS.WIDTH);
  carpetGeom.rotateX(-Math.PI / 2); // Lay flat on XZ plane at Y=0

  const carpetTex = createCarpetTexture(THREE);
  let carpetMat;
  if (carpetTex) {
    carpetMat = new THREE.MeshStandardMaterial({
      map: carpetTex,
      roughness: 0.85,
      metalness: 0.05
    });
  } else {
    carpetMat = new THREE.MeshStandardMaterial({
      color: 0x1c2822,
      roughness: 0.85,
      metalness: 0.05
    });
  }

  const carpetMesh = new THREE.Mesh(carpetGeom, carpetMat);
  carpetMesh.name = 'carpet_floor';
  carpetMesh.position.set(0, 0, 0);
  carpetMesh.receiveShadow = true;
  fieldGroup.add(carpetMesh);

  // 2. Perimeter Guardrails & Driver Station End Walls
  const guardrails = createPerimeter(THREE);
  fieldGroup.add(guardrails);

  // 3. Upper Hubs: Red at (162.8, 158.85), Blue at (488.4, 158.85)
  const redHubP = to3DCoords(FIELD_DIMENSIONS.HUB_RED_X_INCH, FIELD_DIMENSIONS.HALF_WIDTH, 0);
  const blueHubP = to3DCoords(FIELD_DIMENSIONS.HUB_BLUE_X_INCH, FIELD_DIMENSIONS.HALF_WIDTH, 0);

  const redHub = createUpperHub(THREE, 'RED', redHubP.x, redHubP.z);
  const blueHub = createUpperHub(THREE, 'BLUE', blueHubP.x, blueHubP.z);
  fieldGroup.add(redHub);
  fieldGroup.add(blueHub);

  // 4. Towers: Red at (90.0, 158.85), Blue at (561.2, 158.85)
  const redTowerP = to3DCoords(FIELD_DIMENSIONS.TOWER_RED_X_INCH, FIELD_DIMENSIONS.HALF_WIDTH, 0);
  const blueTowerP = to3DCoords(FIELD_DIMENSIONS.TOWER_BLUE_X_INCH, FIELD_DIMENSIONS.HALF_WIDTH, 0);

  const redTower = createTower(THREE, 'RED', redTowerP.x, redTowerP.z);
  const blueTower = createTower(THREE, 'BLUE', blueTowerP.x, blueTowerP.z);
  fieldGroup.add(redTower);
  fieldGroup.add(blueTower);

  // 5. Depots, Trenches & Bumps
  const depotsAndTrenches = createDepotsAndTrenches(THREE, fieldConfig);
  fieldGroup.add(depotsAndTrenches);

  // 6. 32 Dynamic AprilTags
  const { tagsGroup, tagMeshes } = createAprilTags(THREE, fieldConfig);
  fieldGroup.add(tagsGroup);

  // 7. Lighting
  const { lightsGroup, ambientLight, stadiumLight, fillLight } = createLighting(THREE);
  fieldGroup.add(lightsGroup);

  // Add field group to parent scene
  scene.add(fieldGroup);

  return {
    fieldGroup,
    carpet: carpetMesh,
    guardrails,
    redHub,
    blueHub,
    redTower,
    blueTower,
    depotsAndTrenches,
    apriltags: tagMeshes,
    lights: {
      ambientLight,
      stadiumLight,
      fillLight
    }
  };
}

export const CAD_SCALE_METERS_TO_INCHES = 39.37007874;

/**
 * Asynchronously loads the official 3D CAD model 'field.glb' (FE-2026: REBUILT Playing Field).
 *
 * Scales the model from meters to inches (39.37007874) and aligns CAD Z-up
 * with Three.js Y-up (rotation.x = -Math.PI / 2).
 *
 * @param {THREE.Scene|THREE.Group} scene - Three.js target scene or parent group
 * @param {string} [modelUrl='/static/models/field.glb'] - URL path to field.glb
 * @param {function} [onProgress=null] - Optional progress callback (percent: number)
 * @param {object} [customThree=null] - Optional THREE namespace
 * @returns {Promise<THREE.Group|null>} The loaded and transformed CAD model group
 */
export async function loadFieldGLB(scene, modelUrl = '/static/models/field.glb', onProgress = null, customThree = null) {
  const THREE = getTHREE(customThree);
  if (!scene) {
    throw new Error('loadFieldGLB requires a valid Three.js Scene or Group instance.');
  }

  if (typeof THREE.GLTFLoader === 'undefined') {
    console.warn('THREE.GLTFLoader is not defined. Skipping field.glb load.');
    return null;
  }

  const loader = new THREE.GLTFLoader();

  return new Promise((resolve, reject) => {
    loader.load(
      modelUrl,
      (gltf) => {
        const model = gltf.scene;
        model.name = 'cad_field_model';

        // 1. Scale from CAD meters to Three.js inches (1m = 39.37007874 in)
        model.scale.set(
          CAD_SCALE_METERS_TO_INCHES,
          CAD_SCALE_METERS_TO_INCHES,
          CAD_SCALE_METERS_TO_INCHES
        );

        // 2. Rotate CAD Z-up to Three.js Y-up
        // Rotation around X by -90 deg aligns CAD (X, Y, Z_up) to Three.js (X, Y_up, Z)
        model.rotation.x = -Math.PI / 2;

        // 3. Configure shadows & PBR material properties on all child meshes
        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach((mat) => {
                  mat.roughness = Math.max(mat.roughness !== undefined ? mat.roughness : 0.5, 0.4);
                  mat.metalness = Math.min(mat.metalness !== undefined ? mat.metalness : 0.1, 0.6);
                });
              } else {
                child.material.roughness = Math.max(child.material.roughness !== undefined ? child.material.roughness : 0.5, 0.4);
                child.material.metalness = Math.min(child.material.metalness !== undefined ? child.material.metalness : 0.1, 0.6);
              }
            }
          }
        });

        scene.add(model);
        resolve(model);
      },
      (xhr) => {
        if (onProgress && xhr.total > 0) {
          const percent = Math.min(100, Math.round((xhr.loaded / xhr.total) * 100));
          onProgress(percent);
        }
      },
      (error) => {
        console.error('Error loading field.glb:', error);
        reject(error);
      }
    );
  });
}

export default createFieldCAD;

