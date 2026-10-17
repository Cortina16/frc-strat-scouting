/**
 * FRC 2026 REBUILT — 3D Match Digital Twin
 * Main Application, Playback Engine, Timeline Scrubber & Camera Controller
 */

import { createFieldCAD, loadFieldGLB, to3DCoords, FIELD_DIMENSIONS } from './field3d.js';
import { RobotManager } from './robot3d.js';
import { BallManager } from './ball3d.js';

(function () {
  'use strict';

  // =========================================================================
  // Math & Interpolation Utilities
  // =========================================================================
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function lerpAngle(a, b, t) {
    let diff = (b - a) % (2 * Math.PI);
    if (diff < -Math.PI) diff += 2 * Math.PI;
    if (diff > Math.PI) diff -= 2 * Math.PI;
    return a + diff * t;
  }

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) seconds = 0;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    const secWhole = Math.floor(secs).toString().padStart(2, '0');
    const secCent = Math.floor((secs % 1) * 100).toString().padStart(2, '0');
    return `${mins}:${secWhole}.${secCent}`;
  }

  // =========================================================================
  // Application State & Controller
  // =========================================================================
  class DigitalTwinApp {
    constructor() {
      this.container = document.getElementById('viewport-container');
      this.canvas = document.getElementById('viewport-canvas');

      // Playback State
      this.matches = [];
      this.activeMatchId = null;
      this.matchSummary = null;
      this.matchEvents = [];
      this.framesMap = new Map(); // frame_index -> frameData
      this.frameIndices = [];     // sorted list of frame integers
      this.minFrame = 0;
      this.maxFrame = 0;
      this.totalFrames = 0;
      this.matchFps = 60.0;
      this.currentFrameFloat = 0;
      this.isPlaying = false;
      this.playbackSpeed = 1.0;
      this.isScrubbing = false;

      // Camera Presets & Transition
      this.activePreset = 'free'; // 'free', 'topdown', 'broadcast', 'redwall', 'bluewall', 'follow'
      this.cameraTransition = null;

      // Selection & Telemetry
      this.selectedTrackId = null;
      this.selectedRobotData = null;

      // Performance & Timing
      this.lastTimestamp = performance.now();
      this.currentFPS = 60.0;

      // DOM Elements
      this.dom = {
        loadingOverlay: document.getElementById('loading-overlay'),
        loadingStatus: document.getElementById('loading-status'),
        matchSelect: document.getElementById('match-select'),
        fpsCounter: document.getElementById('fps-counter'),
        frameCounter: document.getElementById('frame-counter'),
        statusIndicator: document.getElementById('status-indicator'),
        statusText: document.getElementById('status-text'),

        // Camera Preset Buttons
        btnPresetFree: document.getElementById('preset-free'),
        btnPresetTopdown: document.getElementById('preset-topdown'),
        btnPresetBroadcast: document.getElementById('preset-broadcast'),
        btnPresetRedwall: document.getElementById('preset-redwall'),
        btnPresetBluewall: document.getElementById('preset-bluewall'),
        btnPresetFollow: document.getElementById('preset-follow'),
        btnHelp: document.getElementById('btn-help'),

        // Robot Inspector Drawer
        robotInspector: document.getElementById('robot-inspector'),
        inspectorBanner: document.getElementById('inspector-banner'),
        inspectorTeamNumber: document.getElementById('inspector-team-number'),
        inspectorAllianceBadge: document.getElementById('inspector-alliance-badge'),
        inspectorTrackId: document.getElementById('inspector-track-id'),
        inspectorActionBadge: document.getElementById('inspector-action-badge'),
        inspectorSpeedIps: document.getElementById('inspector-speed-ips'),
        inspectorSpeedMph: document.getElementById('inspector-speed-mph'),
        inspectorCoords: document.getElementById('inspector-coords'),
        inspectorCoordsM: document.getElementById('inspector-coords-m'),
        inspectorRelCoords: document.getElementById('inspector-rel-coords'),
        inspectorRelCoordsM: document.getElementById('inspector-rel-coords-m'),
        inspectorDimensions: document.getElementById('inspector-dimensions'),
        inspectorDimensionsM: document.getElementById('inspector-dimensions-m'),
        inspectorDepth: document.getElementById('inspector-depth'),
        inspectorDepthM: document.getElementById('inspector-depth-m'),
        inspectorHeadingDeg: document.getElementById('inspector-heading-deg'),
        inspectorHeadingRad: document.getElementById('inspector-heading-rad'),
        inspectorDefenseState: document.getElementById('inspector-defense-state'),
        inspectorTrackState: document.getElementById('inspector-track-state'),
        btnFocusRobot: document.getElementById('btn-focus-robot'),
        btnDeselectRobot: document.getElementById('btn-deselect-robot'),
        btnCloseInspector: document.getElementById('btn-close-inspector'),

        // Bottom Playback Controls
        timelineSlider: document.getElementById('timeline-slider'),
        timelineFill: document.getElementById('timeline-fill'),
        timelineMarkers: document.getElementById('timeline-markers'),
        timelineTooltip: document.getElementById('timeline-tooltip'),
        btnPlayPause: document.getElementById('btn-play-pause'),
        playIcon: document.getElementById('play-icon'),
        pauseIcon: document.getElementById('pause-icon'),
        btnStepBack10: document.getElementById('btn-step-back-10'),
        btnStepBack1: document.getElementById('btn-step-back-1'),
        btnStepFwd1: document.getElementById('btn-step-fwd-1'),
        btnStepFwd10: document.getElementById('btn-step-fwd-10'),
        speedPills: document.querySelectorAll('.speed-pill'),
        readoutFrame: document.getElementById('readout-frame'),
        readoutTime: document.getElementById('readout-time'),
        readoutPeriod: document.getElementById('readout-period'),

        // Modal
        shortcutsModal: document.getElementById('shortcuts-modal'),
        btnCloseModal: document.getElementById('btn-close-modal')
      };

      this.initThree();
      this.bindEvents();
      this.loadMatchList();
      this.animate(performance.now());
    }

    // =======================================================================
    // Three.js Scene, Camera, Lights & Controls Setup
    // =======================================================================
    initThree() {
      // Scene
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x090a10);
      this.scene.fog = new THREE.FogExp2(0x090a10, 0.00065);

      // Camera
      const aspect = window.innerWidth / window.innerHeight;
      this.camera = new THREE.PerspectiveCamera(50, aspect, 1.0, 3500);
      this.camera.position.set(0, 280, 380);

      // Renderer
      this.renderer = new THREE.WebGLRenderer({
        canvas: this.canvas,
        antialias: true,
        powerPreference: 'high-performance'
      });
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

      // OrbitControls with smooth damping
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxPolarAngle = Math.PI / 2 + 0.05; // Do not go below floor
      this.controls.minDistance = 20;
      this.controls.maxDistance = 1500;
      this.controls.target.set(0, 20, 0);

      // When user starts interacting with controls, deactivate preset highlights
      this.controls.addEventListener('start', () => {
        if (this.activePreset !== 'free') {
          this.setActivePreset('free', false);
        }
      });

      // Subtle base scene lights (field CAD module adds stadium spotlight and fill lights)
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
      this.scene.add(ambientLight);

      // Raycaster for 3D Robot Clicking
      this.raycaster = new THREE.Raycaster();
      this.mouse = new THREE.Vector2();

      // Managers
      this.robotManager = new RobotManager(this.scene);
      this.ballManager = new BallManager(this.scene);
    }

    // =======================================================================
    // Camera Presets & Transitions
    // =======================================================================
    setActivePreset(presetName, shouldTransition = true) {
      this.activePreset = presetName;

      // Update button styles
      const presetButtons = [
        { name: 'free', el: this.dom.btnPresetFree },
        { name: 'topdown', el: this.dom.btnPresetTopdown },
        { name: 'broadcast', el: this.dom.btnPresetBroadcast },
        { name: 'redwall', el: this.dom.btnPresetRedwall },
        { name: 'bluewall', el: this.dom.btnPresetBluewall },
        { name: 'follow', el: this.dom.btnPresetFollow }
      ];

      for (const p of presetButtons) {
        if (p.el) {
          if (p.name === presetName) {
            p.el.classList.add('active');
          } else {
            p.el.classList.remove('active');
          }
        }
      }

      if (!shouldTransition) return;

      switch (presetName) {
        case 'free':
          this.transitionCamera(new THREE.Vector3(0, 280, 380), new THREE.Vector3(0, 20, 0), 800);
          break;
        case 'topdown':
          // Bird's eye field overview
          this.transitionCamera(new THREE.Vector3(0, 560, 0.1), new THREE.Vector3(0, 0, 0), 900);
          break;
        case 'broadcast':
          // Elevated sideline view replicating match video
          this.transitionCamera(new THREE.Vector3(0, 240, 320), new THREE.Vector3(0, 25, 0), 800);
          break;
        case 'redwall':
          // Red Alliance Driver Station: Looking from -X towards field
          this.transitionCamera(new THREE.Vector3(-370, 80, 0), new THREE.Vector3(120, 30, 0), 850);
          break;
        case 'bluewall':
          // Blue Alliance Driver Station: Looking from +X towards field
          this.transitionCamera(new THREE.Vector3(370, 80, 0), new THREE.Vector3(-120, 30, 0), 850);
          break;
        case 'follow':
          // Locks camera to selected robot cube
          if (!this.selectedTrackId && this.frameIndices.length > 0) {
            const curFrame = this.framesMap.get(Math.floor(this.currentFrameFloat));
            if (curFrame && curFrame.robots && curFrame.robots.length > 0) {
              this.selectRobot(curFrame.robots[0].track_id);
            }
          }
          break;
      }
    }

    transitionCamera(targetPos, targetLookAt, duration = 800) {
      this.cameraTransition = {
        startPos: this.camera.position.clone(),
        endPos: targetPos.clone(),
        startLookAt: this.controls.target.clone(),
        endLookAt: targetLookAt.clone(),
        startTime: performance.now(),
        duration: duration
      };
    }

    // =======================================================================
    // Data Loading & API Connectivity
    // =======================================================================
    async loadMatchList() {
      try {
        const resp = await fetch('/api/matches');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        this.matches = Array.isArray(data) ? data : (data.matches || []);

        this.dom.matchSelect.innerHTML = '';
        if (this.matches.length === 0) {
          const opt = document.createElement('option');
          opt.value = "2026cmptx_f1m3";
          opt.textContent = "2026cmptx_f1m3 (Default)";
          this.dom.matchSelect.appendChild(opt);
          this.loadMatch("2026cmptx_f1m3");
        } else {
          for (const m of this.matches) {
            const opt = document.createElement('option');
            const mId = m.match_id || m;
            opt.value = mId;
            opt.textContent = `${mId} (${m.frame_count || 1200}f)`;
            this.dom.matchSelect.appendChild(opt);
          }
          const defaultMatch = this.matches[0].match_id || this.matches[0];
          this.dom.matchSelect.value = defaultMatch;
          this.loadMatch(defaultMatch);
        }
      } catch (err) {
        console.warn("Could not fetch /api/matches, using default match id:", err);
        this.dom.matchSelect.innerHTML = '<option value="2026cmptx_f1m3">2026cmptx_f1m3</option>';
        this.loadMatch("2026cmptx_f1m3");
      }
    }

    async loadMatch(matchId) {
      this.activeMatchId = matchId;
      this.isPlaying = false;
      this.updatePlayPauseUI();

      this.dom.loadingOverlay.classList.remove('hidden');
      this.dom.loadingStatus.textContent = `Loading Match ${matchId}...`;

      try {
        // 1. Load Field CAD and call createFieldCAD(scene, fieldConfig)
        this.dom.loadingStatus.textContent = "Constructing 3D Field Environment...";
        const fieldResp = await fetch(`/api/match/${matchId}/field`).catch(() => null);
        let fieldConfig = null;
        if (fieldResp && fieldResp.ok) {
          fieldConfig = await fieldResp.json();
        }

        // Remove previous field elements if any
        const prevField = this.scene.getObjectByName('frc2026_field');
        if (prevField) {
          this.scene.remove(prevField);
        }
        const prevCad = this.scene.getObjectByName('cad_field_model');
        if (prevCad) {
          this.scene.remove(prevCad);
        }

        const fieldCAD = createFieldCAD(this.scene, fieldConfig);

        // Asynchronously load the official CAD 3D model 'field.glb'
        this.dom.loadingStatus.textContent = "Loading 3D CAD Model (field.glb)...";
        try {
          const cadModel = await loadFieldGLB(
            this.scene,
            '/static/models/field.glb',
            (pct) => {
              this.dom.loadingStatus.textContent = `Loading 3D CAD Model (field.glb)... ${pct}%`;
            }
          );
          if (cadModel) {
            // Hide procedural primitives since official CAD model is active
            if (fieldCAD.carpet) fieldCAD.carpet.visible = false;
            if (fieldCAD.guardrails) fieldCAD.guardrails.visible = false;
            if (fieldCAD.redHub) fieldCAD.redHub.visible = false;
            if (fieldCAD.blueHub) fieldCAD.blueHub.visible = false;
            if (fieldCAD.redTower) fieldCAD.redTower.visible = false;
            if (fieldCAD.blueTower) fieldCAD.blueTower.visible = false;
            if (fieldCAD.depotsAndTrenches) fieldCAD.depotsAndTrenches.visible = false;
          }
        } catch (cadErr) {
          console.warn("Could not load field.glb, using procedural CAD fallback:", cadErr);
        }

        // 2. Load Match Summary
        this.dom.loadingStatus.textContent = "Fetching Match Summary & Roster...";
        const summaryResp = await fetch(`/api/match/${matchId}/summary`).catch(() => null);
        if (summaryResp && summaryResp.ok) {
          this.matchSummary = await summaryResp.json();
          this.matchFps = this.matchSummary.frame_rate || 60.0;
        }

        // 3. Load Match Events (ActionFormer)
        this.dom.loadingStatus.textContent = "Loading ActionFormer Events...";
        const eventsResp = await fetch(`/api/match/${matchId}/events`).catch(() => null);
        if (eventsResp && eventsResp.ok) {
          const evData = await eventsResp.json();
          this.matchEvents = Array.isArray(evData) ? evData : (evData.events || []);
        } else {
          this.matchEvents = [];
        }

        // 4. Load Match Telemetry Frames
        this.dom.loadingStatus.textContent = "Loading Trajectory & Ball Telemetry...";
        const framesResp = await fetch(`/api/match/${matchId}/frames`);
        if (!framesResp.ok) throw new Error(`Failed to load frames: HTTP ${framesResp.status}`);
        const framesRaw = await framesResp.json();

        this.processFramesData(framesRaw);

        // Populate Scrubber Markers
        this.renderTimelineMarkers();

        // Seek to initial frame
        this.currentFrameFloat = this.minFrame;
        this.seekFrame(this.minFrame);

        // Hide overlay
        this.dom.loadingOverlay.classList.add('hidden');
        this.dom.statusText.textContent = "READY";
      } catch (err) {
        console.error("Error loading match data:", err);
        this.dom.loadingStatus.textContent = `Error loading match: ${err.message}. Retrying...`;
        setTimeout(() => this.dom.loadingOverlay.classList.add('hidden'), 2000);
      }
    }

    processFramesData(raw) {
      this.framesMap.clear();

      let frameDict = {};
      if (raw.frames && typeof raw.frames === 'object' && !Array.isArray(raw.frames)) {
        frameDict = raw.frames;
      } else if (Array.isArray(raw.frames)) {
        for (const f of raw.frames) {
          frameDict[f.frame_index] = f;
        }
      } else if (Array.isArray(raw)) {
        for (const f of raw) {
          frameDict[f.frame_index] = f;
        }
      } else if (typeof raw === 'object') {
        frameDict = raw;
      }

      this.frameIndices = Object.keys(frameDict).map(Number).sort((a, b) => a - b);
      for (const idx of this.frameIndices) {
        this.framesMap.set(idx, frameDict[idx]);
      }

      if (this.frameIndices.length > 0) {
        this.minFrame = this.frameIndices[0];
        this.maxFrame = this.frameIndices[this.frameIndices.length - 1];
        this.totalFrames = this.frameIndices.length;
      } else {
        this.minFrame = 0;
        this.maxFrame = 1200;
        this.totalFrames = 1200;
      }

      this.dom.timelineSlider.min = this.minFrame;
      this.dom.timelineSlider.max = this.maxFrame;
      this.dom.timelineSlider.value = this.minFrame;
    }

    // =======================================================================
    // Timeline Markers & Event Tooltips
    // =======================================================================
    renderTimelineMarkers() {
      const container = this.dom.timelineMarkers;
      container.innerHTML = '';
      if (!this.matchEvents || this.matchEvents.length === 0) return;

      const span = this.maxFrame - this.minFrame;
      if (span <= 0) return;

      for (const ev of this.matchEvents) {
        const frame = ev.start_frame !== undefined ? ev.start_frame : ev.frame_index;
        if (frame === undefined || frame < this.minFrame || frame > this.maxFrame) continue;

        const pct = ((frame - this.minFrame) / span) * 100;
        const marker = document.createElement('div');
        marker.className = 'event-marker';
        marker.style.left = `${pct}%`;

        // Color coding for ActionFormer events:
        // - Green dots for SCORE_HUB
        // - Purple dots for CLIMB_TOWER
        // - Orange dots for DEFENSE
        const actionType = (ev.action_type || '').toUpperCase();
        if (actionType.includes('SCORE') || actionType.includes('HUB')) {
          marker.classList.add('score-hub');
        } else if (actionType.includes('CLIMB') || actionType.includes('TOWER')) {
          marker.classList.add('climb-tower');
        } else if (actionType.includes('DEFENSE') || actionType.includes('DEFENSIVE')) {
          marker.classList.add('defense');
        } else {
          marker.classList.add('transit');
        }

        // Hover tooltip displaying event name and team
        marker.addEventListener('mouseenter', () => {
          const teamText = ev.team_number ? `Team ${ev.team_number}` : (ev.track_id ? `Track #${ev.track_id}` : '');
          const confText = ev.confidence ? ` [${Math.round(ev.confidence * 100)}%]` : '';
          this.dom.timelineTooltip.textContent = `${actionType} — ${teamText}${confText} (f:${frame})`;
          this.dom.timelineTooltip.style.left = `${pct}%`;
          this.dom.timelineTooltip.classList.add('visible');
        });

        marker.addEventListener('mouseleave', () => {
          this.dom.timelineTooltip.classList.remove('visible');
        });

        // Clicking any marker seeks directly to that frame!
        marker.addEventListener('click', (e) => {
          e.stopPropagation();
          this.seekFrame(frame);
          if (ev.track_id !== undefined) {
            this.selectRobot(ev.track_id);
          }
        });

        container.appendChild(marker);
      }
    }

    // =======================================================================
    // Playback Engine & Frame Interpolation
    // =======================================================================
    togglePlayPause() {
      this.isPlaying = !this.isPlaying;
      this.updatePlayPauseUI();
    }

    updatePlayPauseUI() {
      if (this.isPlaying) {
        this.dom.playIcon.style.display = 'none';
        this.dom.pauseIcon.style.display = 'block';
        this.dom.statusText.textContent = "PLAYING";
      } else {
        this.dom.playIcon.style.display = 'block';
        this.dom.pauseIcon.style.display = 'none';
        this.dom.statusText.textContent = "PAUSED";
      }
    }

    seekFrame(frameIndex) {
      this.currentFrameFloat = Math.max(this.minFrame, Math.min(this.maxFrame, frameIndex));
      this.dom.timelineSlider.value = this.currentFrameFloat;
      this.updatePlaybackVisuals(0);
    }

    stepFrames(delta) {
      this.seekFrame(this.currentFrameFloat + delta);
    }

    setPlaybackSpeed(speed) {
      this.playbackSpeed = speed;
      this.dom.speedPills.forEach(pill => {
        if (parseFloat(pill.dataset.speed) === speed) {
          pill.classList.add('active');
        } else {
          pill.classList.remove('active');
        }
      });
    }

    updatePlaybackVisuals(alpha = 0.0) {
      const curFloor = Math.floor(this.currentFrameFloat);
      const nextCeil = Math.min(this.maxFrame, curFloor + 1);

      // Scrubber progress fill
      const span = this.maxFrame - this.minFrame;
      const progressPct = span > 0 ? ((this.currentFrameFloat - this.minFrame) / span) * 100 : 0;
      this.dom.timelineFill.style.width = `${progressPct}%`;
      this.dom.timelineSlider.value = this.currentFrameFloat;

      // Readouts
      this.dom.readoutFrame.textContent = `${curFloor} / ${this.maxFrame}`;
      this.dom.frameCounter.textContent = `Frame: ${curFloor} / ${this.maxFrame}`;

      const elapsedSec = (this.currentFrameFloat - this.minFrame) / this.matchFps;
      this.dom.readoutTime.textContent = formatTime(elapsedSec);

      // Autonomous vs Teleop period (First 15s is Autonomous in FRC)
      if (elapsedSec <= 15.0) {
        this.dom.readoutPeriod.textContent = "Autonomous";
        this.dom.readoutPeriod.classList.add('auto');
      } else {
        this.dom.readoutPeriod.textContent = "Teleop";
        this.dom.readoutPeriod.classList.remove('auto');
      }

      // Robot & Ball Telemetry Interpolation
      const frame0 = this.framesMap.get(curFloor);
      const frame1 = this.framesMap.get(nextCeil) || frame0;

      if (frame0 && frame0.robots) {
        const interpolatedRobots = this.interpolateRobots(frame0.robots, frame1 ? frame1.robots : frame0.robots, alpha);
        this.robotManager.updateRobots(interpolatedRobots, alpha);

        // Update selected robot telemetry in inspector drawer
        if (this.selectedTrackId !== null) {
          const selected = interpolatedRobots.find(r => r.track_id === this.selectedTrackId);
          if (selected) {
            this.updateInspectorTelemetry(selected);
          } else {
            // Selected robot disappeared / occluded on screen
            this.dom.inspectorActionBadge.textContent = 'OCCLUDED / OUT OF VIEW';
            this.dom.inspectorActionBadge.className = 'inspector-badge idle';
            this.dom.inspectorTrackState.textContent = 'OCCLUDED';
          }
        }
      }

      if (frame0 && frame0.balls) {
        this.ballManager.updateBalls(frame0.balls);
      }
    }

    interpolateRobots(robots0, robots1, alpha) {
      if (!robots1 || alpha === 0) return robots0 || [];
      if (!robots0) return robots1 || [];

      const map1 = new Map();
      for (const r of robots1) {
        map1.set(r.track_id, r);
      }

      const result = [];
      const seen = new Set();

      for (const r0 of robots0) {
        seen.add(r0.track_id);
        const r1 = map1.get(r0.track_id);
        if (r1) {
          const crx0 = r0.center_rel_x_inch !== undefined ? r0.center_rel_x_inch : (r0.field_x_inch - 325.6);
          const crx1 = r1.center_rel_x_inch !== undefined ? r1.center_rel_x_inch : (r1.field_x_inch - 325.6);
          const cry0 = r0.center_rel_y_inch !== undefined ? r0.center_rel_y_inch : (r0.field_y_inch - 158.85);
          const cry1 = r1.center_rel_y_inch !== undefined ? r1.center_rel_y_inch : (r1.field_y_inch - 158.85);
          const crz0 = r0.center_rel_z_inch !== undefined ? r0.center_rel_z_inch : (r0.field_z_inch || 0.0);
          const crz1 = r1.center_rel_z_inch !== undefined ? r1.center_rel_z_inch : (r1.field_z_inch || 0.0);
          const rw0 = r0.robot_width_inch || 22.0;
          const rw1 = r1.robot_width_inch || rw0;
          const rl0 = r0.robot_length_inch || 28.0;
          const rl1 = r1.robot_length_inch || rl0;
          const rh0 = r0.robot_height_inch || 28.0;
          const rh1 = r1.robot_height_inch || rh0;
          const d0 = r0.depth_inch || 0.0;
          const d1 = r1.depth_inch || d0;

          result.push({
            track_id: r0.track_id,
            team_number: r0.team_number || r1.team_number,
            alliance: r0.alliance || r1.alliance,
            field_x_inch: lerp(r0.field_x_inch, r1.field_x_inch, alpha),
            field_y_inch: lerp(r0.field_y_inch, r1.field_y_inch, alpha),
            field_z_inch: lerp(r0.field_z_inch || 0.0, r1.field_z_inch || 0.0, alpha),
            center_rel_x_inch: lerp(crx0, crx1, alpha),
            center_rel_y_inch: lerp(cry0, cry1, alpha),
            center_rel_z_inch: lerp(crz0, crz1, alpha),
            depth_inch: lerp(d0, d1, alpha),
            robot_width_inch: lerp(rw0, rw1, alpha),
            robot_length_inch: lerp(rl0, rl1, alpha),
            robot_height_inch: lerp(rh0, rh1, alpha),
            heading_rad: lerpAngle(r0.heading_rad || 0, r1.heading_rad || 0, alpha),
            speed_ips: lerp(r0.speed_ips || 0, r1.speed_ips || 0, alpha),
            in_defense: r0.in_defense,
            track_state: r0.track_state
          });
        } else {
          // Robot disappeared in next frame: retain only before midpoint
          if (alpha < 0.5) {
            result.push(r0);
          }
        }
      }

      // Robot appeared in next frame: show when past midpoint
      for (const r1 of robots1) {
        if (!seen.has(r1.track_id) && alpha >= 0.5) {
          result.push(r1);
        }
      }

      return result;
    }

    // =======================================================================
    // Robot Selection & Inspector Drawer
    // =======================================================================
    selectRobot(trackId) {
      this.selectedTrackId = trackId;
      this.robotManager.selectRobot(trackId);

      // Find current state
      const curFloor = Math.floor(this.currentFrameFloat);
      const curFrame = this.framesMap.get(curFloor);
      let robotState = null;
      if (curFrame && curFrame.robots) {
        robotState = curFrame.robots.find(r => r.track_id === trackId);
      }

      this.openInspector(robotState || { track_id: trackId });
    }

    clearSelection() {
      this.selectedTrackId = null;
      this.robotManager.clearSelection();
      this.closeInspector();
      if (this.activePreset === 'follow') {
        this.setActivePreset('free');
      }
    }

    openInspector(robot) {
      const d = this.dom;
      d.robotInspector.classList.add('open');

      const isBlue = robot.alliance && robot.alliance.toLowerCase() === 'blue';
      const allianceClass = isBlue ? 'blue' : 'red';
      const allianceName = isBlue ? 'BLUE' : 'RED';

      d.inspectorBanner.className = `inspector-banner alliance-${allianceClass}`;
      d.inspectorTeamNumber.textContent = robot.team_number ? `TEAM ${robot.team_number}` : `ROBOT #${robot.track_id}`;
      d.inspectorAllianceBadge.textContent = allianceName;
      d.inspectorAllianceBadge.className = `alliance-tag ${allianceClass}`;
      d.inspectorTrackId.textContent = `Track ID #${robot.track_id}`;

      this.updateInspectorTelemetry(robot);
    }

    closeInspector() {
      this.dom.robotInspector.classList.remove('open');
    }

    updateInspectorTelemetry(robot) {
      const d = this.dom;
      const speedIps = robot.speed_ips || 0;
      const speedMph = speedIps * 0.0568182; // 1 ips = 0.0568182 mph
      d.inspectorSpeedIps.textContent = `${speedIps.toFixed(1)} ips`;
      d.inspectorSpeedMph.textContent = `${speedMph.toFixed(2)} mph`;

      const fx = robot.field_x_inch || 0;
      const fy = robot.field_y_inch || 0;
      const fz = robot.field_z_inch !== undefined ? robot.field_z_inch : 0.0;
      const depth = robot.depth_inch || 0.0;

      if (fz !== 0.0) {
        d.inspectorCoords.textContent = `${fx.toFixed(1)}", ${fy.toFixed(1)}", ${fz.toFixed(1)}"`;
        d.inspectorCoordsM.textContent = `${(fx * 0.0254).toFixed(2)}m, ${(fy * 0.0254).toFixed(2)}m, ${(fz * 0.0254).toFixed(2)}m`;
      } else {
        d.inspectorCoords.textContent = `${fx.toFixed(1)}", ${fy.toFixed(1)}"`;
        d.inspectorCoordsM.textContent = `${(fx * 0.0254).toFixed(2)}m, ${(fy * 0.0254).toFixed(2)}m`;
      }

      // Center-relative coordinates: X_rel = X - 325.6, Y_rel = Y - 158.85
      const relX = robot.center_rel_x_inch !== undefined ? robot.center_rel_x_inch : (fx - 325.6);
      const relY = robot.center_rel_y_inch !== undefined ? robot.center_rel_y_inch : (fy - 158.85);
      const relZ = robot.center_rel_z_inch !== undefined ? robot.center_rel_z_inch : fz;

      if (d.inspectorRelCoords) {
        if (relZ !== 0.0) {
          d.inspectorRelCoords.textContent = `${relX.toFixed(1)}", ${relY.toFixed(1)}", ${relZ.toFixed(1)}"`;
          if (d.inspectorRelCoordsM) d.inspectorRelCoordsM.textContent = `${(relX * 0.0254).toFixed(2)}m, ${(relY * 0.0254).toFixed(2)}m, ${(relZ * 0.0254).toFixed(2)}m`;
        } else {
          d.inspectorRelCoords.textContent = `${relX.toFixed(1)}", ${relY.toFixed(1)}"`;
          if (d.inspectorRelCoordsM) d.inspectorRelCoordsM.textContent = `${(relX * 0.0254).toFixed(2)}m, ${(relY * 0.0254).toFixed(2)}m`;
        }
      }

      // Robot Dimensions (nominal 22"x28"x28" or live optical sizing)
      const rW = robot.robot_width_inch || 22.0;
      const rL = robot.robot_length_inch || 28.0;
      const rH = robot.robot_height_inch || 28.0;

      if (d.inspectorDimensions) {
        d.inspectorDimensions.textContent = `${rW.toFixed(1)}" × ${rL.toFixed(1)}" × ${rH.toFixed(1)}"`;
        if (d.inspectorDimensionsM) {
          d.inspectorDimensionsM.textContent = `${(rW * 0.0254).toFixed(2)}m × ${(rL * 0.0254).toFixed(2)}m × ${(rH * 0.0254).toFixed(2)}m`;
        }
      }

      if (d.inspectorDepth) {
        if (depth > 0.0) {
          d.inspectorDepth.textContent = `${depth.toFixed(1)}"`;
          if (d.inspectorDepthM) d.inspectorDepthM.textContent = `${(depth * 0.0254).toFixed(2)}m (Euclidean)`;
        } else {
          d.inspectorDepth.textContent = `--`;
          if (d.inspectorDepthM) d.inspectorDepthM.textContent = `3D Euclidean`;
        }
      }

      const heading = robot.heading_rad || 0;
      const deg = ((heading * 180 / Math.PI) % 360 + 360) % 360;
      d.inspectorHeadingDeg.textContent = `${deg.toFixed(1)}°`;
      d.inspectorHeadingRad.textContent = `${heading.toFixed(2)} rad`;

      d.inspectorDefenseState.textContent = robot.in_defense ? "Active Engaging" : "Normal";
      d.inspectorTrackState.textContent = robot.track_state || "TRACKED";

      // Match Current Action
      const curFrame = Math.floor(this.currentFrameFloat);
      const activeEv = this.matchEvents.find(ev =>
        ev.track_id === robot.track_id &&
        curFrame >= (ev.start_frame || 0) &&
        curFrame <= (ev.end_frame || ev.start_frame || 0)
      );

      if (activeEv) {
        d.inspectorActionBadge.textContent = activeEv.action_type;
        const act = activeEv.action_type.toLowerCase();
        if (act.includes('score')) {
          d.inspectorActionBadge.className = 'action-badge score-hub';
        } else if (act.includes('climb')) {
          d.inspectorActionBadge.className = 'action-badge climb-tower';
        } else if (act.includes('defense')) {
          d.inspectorActionBadge.className = 'action-badge defense';
        } else {
          d.inspectorActionBadge.className = 'action-badge transit';
        }
      } else if (robot.in_defense) {
        d.inspectorActionBadge.textContent = "DEFENSE";
        d.inspectorActionBadge.className = 'action-badge defense';
      } else if (speedIps > 5.0) {
        d.inspectorActionBadge.textContent = "TRANSIT";
        d.inspectorActionBadge.className = 'action-badge transit';
      } else {
        d.inspectorActionBadge.textContent = "IDLE";
        d.inspectorActionBadge.className = 'action-badge';
      }
    }

    focusSelectedRobot() {
      if (this.selectedTrackId === null) return;
      const curFloor = Math.floor(this.currentFrameFloat);
      const curFrame = this.framesMap.get(curFloor);
      if (!curFrame || !curFrame.robots) return;

      const robot = curFrame.robots.find(r => r.track_id === this.selectedTrackId);
      if (!robot) return;

      const p3d = to3DCoords(robot.field_x_inch, robot.field_y_inch, 14);
      const camTarget = new THREE.Vector3(
        p3d.x,
        p3d.y + 75,
        p3d.z + 110
      );

      this.transitionCamera(camTarget, new THREE.Vector3(p3d.x, p3d.y, p3d.z), 750);
    }

    // =======================================================================
    // Event Listeners & Keyboard Bindings
    // =======================================================================
    bindEvents() {
      // Window Resize
      window.addEventListener('resize', () => {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
      });

      // Match Selector Dropdown
      this.dom.matchSelect.addEventListener('change', (e) => {
        this.loadMatch(e.target.value);
      });

      // Camera Presets
      this.dom.btnPresetFree.addEventListener('click', () => this.setActivePreset('free'));
      this.dom.btnPresetTopdown.addEventListener('click', () => this.setActivePreset('topdown'));
      this.dom.btnPresetBroadcast.addEventListener('click', () => this.setActivePreset('broadcast'));
      this.dom.btnPresetRedwall.addEventListener('click', () => this.setActivePreset('redwall'));
      this.dom.btnPresetBluewall.addEventListener('click', () => this.setActivePreset('bluewall'));
      this.dom.btnPresetFollow.addEventListener('click', () => this.setActivePreset('follow'));

      // Help Modal
      this.dom.btnHelp.addEventListener('click', () => {
        this.dom.shortcutsModal.classList.add('open');
      });
      this.dom.btnCloseModal.addEventListener('click', () => {
        this.dom.shortcutsModal.classList.remove('open');
      });
      this.dom.shortcutsModal.addEventListener('click', (e) => {
        if (e.target === this.dom.shortcutsModal) {
          this.dom.shortcutsModal.classList.remove('open');
        }
      });

      // Inspector Drawer Buttons
      this.dom.btnCloseInspector.addEventListener('click', () => this.closeInspector());
      this.dom.btnDeselectRobot.addEventListener('click', () => this.clearSelection());
      this.dom.btnFocusRobot.addEventListener('click', () => this.focusSelectedRobot());

      // Playback Controls
      this.dom.btnPlayPause.addEventListener('click', () => this.togglePlayPause());
      this.dom.btnStepBack10.addEventListener('click', () => this.stepFrames(-10));
      this.dom.btnStepBack1.addEventListener('click', () => this.stepFrames(-1));
      this.dom.btnStepFwd1.addEventListener('click', () => this.stepFrames(1));
      this.dom.btnStepFwd10.addEventListener('click', () => this.stepFrames(10));

      // Speed Selector Pills
      this.dom.speedPills.forEach(pill => {
        pill.addEventListener('click', () => {
          this.setPlaybackSpeed(parseFloat(pill.dataset.speed));
        });
      });

      // Timeline Scrubber Input
      this.dom.timelineSlider.addEventListener('input', (e) => {
        this.isScrubbing = true;
        this.seekFrame(parseFloat(e.target.value));
      });
      this.dom.timelineSlider.addEventListener('change', () => {
        this.isScrubbing = false;
      });

      // Raycast 3D Click Selection on Canvas
      this.canvas.addEventListener('pointerdown', (e) => {
        // Only consider left click
        if (e.button !== 0) return;

        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        let hitId = null;
        if (typeof this.robotManager.raycast === 'function') {
          hitId = this.robotManager.raycast(this.raycaster);
        }

        if (hitId !== null && hitId !== undefined) {
          this.selectRobot(hitId);
        } else {
          // Deselect on background click
          this.clearSelection();
        }
      });

      // Global Keyboard Shortcuts
      window.addEventListener('keydown', (e) => {
        // Ignore typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        switch (e.code) {
          case 'Space':
            e.preventDefault();
            this.togglePlayPause();
            break;

          case 'ArrowLeft':
            e.preventDefault();
            this.stepFrames(e.shiftKey ? -10 : -1);
            break;

          case 'ArrowRight':
            e.preventDefault();
            this.stepFrames(e.shiftKey ? 10 : 1);
            break;

          case 'KeyR':
            e.preventDefault();
            this.setActivePreset('free');
            break;

          case 'KeyT':
            e.preventDefault();
            this.setActivePreset('topdown');
            break;

          case 'Digit1':
          case 'Digit2':
          case 'Digit3':
          case 'Digit4':
          case 'Digit5':
          case 'Digit6': {
            e.preventDefault();
            const robotIndex = parseInt(e.code.replace('Digit', '')) - 1;
            this.selectRobotByIndex(robotIndex);
            break;
          }

          case 'Escape':
            e.preventDefault();
            if (this.dom.shortcutsModal.classList.contains('open')) {
              this.dom.shortcutsModal.classList.remove('open');
            } else {
              this.clearSelection();
            }
            break;

          case 'Slash':
            if (e.shiftKey) { // '?' key
              e.preventDefault();
              this.dom.shortcutsModal.classList.toggle('open');
            }
            break;
        }
      });
    }

    selectRobotByIndex(index) {
      const curFloor = Math.floor(this.currentFrameFloat);
      const curFrame = this.framesMap.get(curFloor);
      if (curFrame && curFrame.robots && curFrame.robots[index]) {
        this.selectRobot(curFrame.robots[index].track_id);
        this.focusSelectedRobot();
      }
    }

    // =======================================================================
    // Render Loop & Animation Updates
    // =======================================================================
    animate(timestamp) {
      requestAnimationFrame((t) => this.animate(t));

      const dt = Math.min(0.1, (timestamp - this.lastTimestamp) / 1000);
      this.lastTimestamp = timestamp;

      // FPS calculation
      if (dt > 0) {
        const instantFps = 1 / dt;
        this.currentFPS = this.currentFPS * 0.9 + instantFps * 0.1;
        this.dom.fpsCounter.textContent = `${this.currentFPS.toFixed(1)} FPS`;
      }

      // Camera Transition Animation (smooth slerp/cubic lerp)
      if (this.cameraTransition) {
        const elapsed = timestamp - this.cameraTransition.startTime;
        const t = Math.min(1.0, elapsed / this.cameraTransition.duration);
        // Cubic ease-out: 1 - (1 - t)^3
        const ease = 1 - Math.pow(1 - t, 3);

        this.camera.position.lerpVectors(this.cameraTransition.startPos, this.cameraTransition.endPos, ease);
        this.controls.target.lerpVectors(this.cameraTransition.startLookAt, this.cameraTransition.endLookAt, ease);
        this.controls.update();

        if (t >= 1.0) {
          this.cameraTransition = null;
        }
      }

      // Follow-Robot Mode: smoothly track selected robot cube
      if (this.activePreset === 'follow' && this.selectedTrackId !== null) {
        const curFloor = Math.floor(this.currentFrameFloat);
        const curFrame = this.framesMap.get(curFloor);
        if (curFrame && curFrame.robots) {
          const r = curFrame.robots.find(rob => rob.track_id === this.selectedTrackId);
          if (r) {
            const p3d = to3DCoords(r.field_x_inch, r.field_y_inch, 14);
            const heading = r.heading_rad || 0;
            const camDist = 95;
            const camHeight = 55;
            // Robot local front is +X, heading rotation is around Y
            const offset = new THREE.Vector3(
              -Math.cos(heading) * camDist,
              camHeight,
              -Math.sin(heading) * camDist
            );
            const targetCamPos = new THREE.Vector3(p3d.x + offset.x, p3d.y + offset.y, p3d.z + offset.z);
            const robotCenter = new THREE.Vector3(p3d.x, p3d.y, p3d.z);

            this.camera.position.lerp(targetCamPos, 0.08);
            this.controls.target.lerp(robotCenter, 0.08);
            this.controls.update();
          }
        }
      } else if (!this.cameraTransition) {
        this.controls.update();
      }

      // Advance Playback Engine
      if (this.isPlaying && !this.isScrubbing && this.totalFrames > 0) {
        this.currentFrameFloat += this.matchFps * this.playbackSpeed * dt;
        if (this.currentFrameFloat >= this.maxFrame) {
          this.currentFrameFloat = this.minFrame; // Loop replay
        }
      }

      // Fractional alpha for 60 FPS subframe interpolation
      const floorFrame = Math.floor(this.currentFrameFloat);
      const alpha = this.currentFrameFloat - floorFrame;
      this.updatePlaybackVisuals(alpha);

      // Render Three.js Scene
      this.renderer.render(this.scene, this.camera);
    }
  }

  // Initialize Application once DOM is ready
  window.addEventListener('DOMContentLoaded', () => {
    window.app = new DigitalTwinApp();
  });

})();
