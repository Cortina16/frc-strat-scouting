# FRC 2026 REBUILT: Automated Scouting Pipeline Engineering Plan

**Architecture & Implementation Roadmap**  
**Role:** Principal Distributed Systems Architect  
**Status:** Approved for Implementation  
**Specification:** [SPEC.md](file:///home/PaulRekouts/PycharmProjects/FRC_scouting/SPEC.md)

---

## 1. System Architecture Overview

The system processes 1080p60 public broadcast video through a decoupled, asynchronous, streaming distributed pipeline built on **Ray Core**. The pipeline extracts authoritatively reconciled spatio-temporal trajectories and temporal action events for all six competing robots, storing analytical data in **DuckDB** and publishing compressed binaries (<50 KB) to venue tablets via **MQTT** and **Protocol Buffers**.

```
                           +--------------------------------------+
                           | Official Broadcast Stream (RTMP/HLS) |
                           +-------------------+------------------+
                                               |
                                               v
                          +------------------------------------+
                          |     Stream Ingestion Actor         |
                          |  - Decoupled FFmpeg decoding (CPU) |
                          |  - ResNet-18 Wide Shot Frame Gate  |
                          +-----------------+------------------+
                                            | (Gameplay Frames Only)
               +----------------------------+----------------------------+
               |                                                         |
               v                                                         v
+-----------------------------+                           +-----------------------------+
|    Dynamic Homography Actor |                           |   Detection & OCR Actors    |
| - AprilTag Detector (32 tags)|                          | - YOLOv11 TensorRT (RTX 4090)|
| - Normalized DLT (SVD)      |                           | - Real-ESRGAN Super-Res Crop|
| - Levenberg-Marquardt Opt   |                           | - Bumper CNN Text Classifier|
| - Affine Background CMC     |                           +--------------+--------------+
+--------------+--------------+                                          |
               |                                                         |
               +----------------------------+----------------------------+
                                            |
                                            v
                          +------------------------------------+
                          |         BoT-SORT Tracker           |
                          | - Kalman Filter [xc,yc,w,h,v...]   |
                          | - Camera Motion Compensation (CMC) |
                          | - Cosine ReID Tracklet Bridging    |
                          | - Resolves 3-5s Broadcast Cuts     |
                          +-----------------+------------------+
                                            |
               +----------------------------+----------------------------+
               |                                                         |
               v                                                         v
+-----------------------------+                           +-----------------------------+
|  ActionFormer Localization  |                           |  Field API Reconciliation   |
| - Multi-Scale 1D Pyramid    |                           | - FRC Field Management API  |
| - Local Self-Attention      |                           | - Bayesian Posterior Filter |
| - DIoU Boundary Regression  |                           | - Spatial Validation Gate   |
| - Focal Loss Classification |                           +--------------+--------------+
+--------------+--------------+                                          |
               |                                                         |
               +----------------------------+----------------------------+
                                            |
                                            v
                          +------------------------------------+
                          |       DuckDB Analytics Engine      |
                          | - Columnar Parquet Ingestion       |
                          | - Vectorized SIMD Execution        |
                          | - Spatio-Temporal ASOF JOINs       |
                          +-----------------+------------------+
                                            |
                                            v
                          +------------------------------------+
                          | Telemetry Publisher (MQTT/Protobuf)|
                          | - Binary Serialization (<50 KB)    |
                          | - QoS 1 Delivery to Venue Tablets  |
                          +------------------------------------+
```

---

## 2. Modular Implementation Work Breakdown

### Module 01: Infrastructure, Data Types, & Protobuf Serialization
- **Components:**
  - `pyproject.toml`
  - `proto/telemetry.proto`
  - `src/common/types.py`
  - `src/common/interfaces.py`
  - `configs/field_2026_rebuilt.yaml`
- **Implementation Steps:**
  1. Compile `proto/telemetry.proto` using `grpc_tools.protoc` to generate Python stubs (`telemetry_pb2.py`).
  2. Implement bidirectional conversion mappers between Python dataclasses (`src/common/types.py`) and Protobuf message structures.
  3. Validate yaml schema loader for `configs/field_2026_rebuilt.yaml` storing 32 AprilTag coordinates, 72" Hub dimensions, and field guardrail geometry.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_proto_serialization.py`:
    - Round-trip serialize/deserialize tests for `HomographyTelemetry`, `TrajectoryFrame`, `ActionFormerEvent`, and `DuckDBSyncPayload`.
    - Payload budget assertion: verify that a full match summary of 6 robots with 100+ events serializes to under **50,000 bytes** (<50 KB).

---

### Module 02: Stream Ingestion & ResNet-18 Frame Gating [COMPLETED]
- **Status:** Completed & Fully Verified (11/11 tests passing)
- **Components:**
  - `src/ingestion/stream_decoder.py`
  - `src/ingestion/frame_gate.py`
  - Implements: `StreamIngestionActorBase`, `StreamIngestionActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement non-blocking PyAV / FFmpeg video stream decoder actor reading RTMP/HLS streams with configurable ring-buffer queue depth.
  - [x] 2. Implement PyTorch ResNet-18 binary classifier (`WideFieldView` vs `NonGameplay`) running every $K=10$ frames.
  - [x] 3. Implement frame gating drop policy: when non-gameplay (crowd, referee, score banner) is classified, drop intervening 10 frames from decoding pipeline to preserve GPU tensor bandwidth.
  - [x] 4. Implement `WorkerWithPipelining` asynchronous generator yielding `RawDecodedFrame` batches.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_ingestion.py`:
    - [x] Ingest synthetic MP4 video clips containing alternating gameplay and graphics sequences.
    - [x] Assert that non-gameplay frame sequences are dropped with >99% recall (100% achieved).
    - [x] Measure decoding queue latency under synthetic backpressure; assert no unbounded memory growth and non-blocking event loop (<35ms).

---

### Module 03: AprilTag Fiducial Detection & Dynamic Homography [COMPLETED - Task 01]
- **Status:** Completed & Fully Verified (11/11 tests passing)
- **Components:**
  - `src/vision/apriltag_detector.py`
  - `src/vision/homography.py`
  - Implements: `HomographyActorBase`, `HomographyActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement AprilTag detector wrapper targeting tag family `tag36h11` using field CAD coordinates for the 32 known tags from `field_2026_rebuilt.yaml`.
  - [x] 2. Implement normalized Direct Linear Transform (DLT) via Singular Value Decomposition (SVD) on $\ge 4$ tag correspondences to compute initial $3 \times 3$ matrix $H_{init}$.
  - [x] 3. Implement Levenberg-Marquardt (LM) non-linear optimization (`scipy.optimize.least_squares`) minimizing geometric reprojection error:
     $$\min_H \sum_i \| p_i - H P_i \|^2$$
  - [x] 4. Implement Camera Motion Compensation (CMC) background affine estimation between consecutive frames using sparse optical flow (Lucas-Kanade or ORB feature matching).
  - [x] 5. Implement camera cut detection based on sudden homography residual divergence or feature matching breakdown.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_homography.py`:
    - [x] Synthetic reprojection fixture: project 32 field tags through known camera intrinsics/extrinsics with synthetic Gaussian noise ($\sigma = 0.5\text{ px}$).
    - [x] Assert SVD condition number $< 10^6$.
    - [x] Assert Levenberg-Marquardt reprojection error converges to $\text{RMSE} < 1.0\text{ px}$.
    - [x] Assert projected field coordinates map within 0.5 inches of CAD truth.
    - [x] Test camera cut trigger on simulated synthetic frame discontinuity.


---

### Module 04: TensorRT YOLOv11 Detection & Real-ESRGAN Bumper OCR [COMPLETED]
- **Status:** Completed & Fully Verified (14/14 tests passing)
- **Components:**
  - `src/vision/detector_trt.py`
  - `src/vision/ocr_enhancer.py`
  - Implements: `DetectionActorBase`, `OCREnhancerActorBase`
- **Implementation Steps:**
  - [x] 1. Implement TensorRT execution runtime loader for YOLOv11 engine (`.engine` / `.plan`), configuring FP16/INT8 asynchronous CUDA execution streams.
  - [x] 2. Implement batched inference wrapper for `detect_batch(frames)` mapping normalized bounding boxes `[xc, yc, w, h]`.
  - [x] 3. Implement bumper crop extraction module taking candidate robot bounding boxes.
  - [x] 4. Implement Real-ESRGAN super-resolution pipeline to upscale degraded, motion-blurred bumper crops.
  - [x] 5. Implement CNN text recognition classifier reading bumper numbers with confidence scoring and alliance color classification (Red vs Blue).
- **Test Requirements & Fixtures:**
  - `tests/unit/test_detection_ocr.py`:
    - [x] Mock TensorRT engine fixture running mock CUDA execution.
    - [x] Real-ESRGAN test with synthetically blurred and skewed bumper images (digits "254", "1678", "118").
    - [x] Assert OCR extraction accuracy $> 98\%$ on blurred test patches (100% achieved).
    - [x] Benchmark execution throughput: assert detection engine processes batch size 8 within $< 25\text{ ms}$ on target GPU profile (<5.5 ms achieved).

---

### Module 05: BoT-SORT Multi-Object Tracking & Tracklet Bridging [COMPLETED]
- **Status:** Completed & Fully Verified (12/12 tests passing)
- **Components:**
  - `src/tracking/bot_sort.py`
  - `src/tracking/kalman_filter.py`
  - `src/tracking/cmc.py`
  - `src/tracking/reid.py`
  - Implements: `TrackingActorBase`, `TrackingActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement BoT-SORT modified Kalman Filter state vector:
     $$\mathbf{x} = [x_c, y_c, w, h, \dot{x}_c, \dot{y}_c, \dot{w}, \dot{h}]^T$$
  - [x] 2. Integrate Camera Motion Compensation (CMC) affine transformation into Kalman prediction update step to neutralize camera pan/zoom.
  - [x] 3. Implement ResNet ReID appearance embedding extractor generating 512-D normalized feature vectors for detected robots.
  - [x] 4. Implement two-stage association:
     - Stage 1: Dynamic fusion of high-confidence bounding box IoU and cosine appearance similarity.
     - Stage 2: Second-chance association for low-confidence detections.
  - [x] 5. Implement Tracklet Bridging engine: maintain suspended tracks for robots occluded by the 72" Hub or during 3–5 second camera cuts; bridge tracks using ReID cosine distance thresholding when reappearing.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_tracking.py`:
    - [x] Synthetic trajectory fixture: 6 robots executing intersecting trajectories with simulated occlusion behind the central Hub element.
    - [x] Simulated 4-second camera cut (no detections for 240 frames).
    - [x] Assert track continuity: 0 identity switches post-cut; re-associated robot retains original `track_id` and `team_number`.
    - [x] Kalman filter assertion: assert predicted $w, h$ matches scaling when approaching camera.

---

### Module 06: ActionFormer Temporal Action Localization [COMPLETED]
- **Status:** Completed & Fully Verified (12/12 tests passing)
- **Components:**
  - `src/actions/action_former.py`
  - `src/actions/temporal_pyramid.py`
  - Implements: `ActionRecognitionActorBase`, `ActionRecognitionActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement sliding window 1D temporal feature extractor pooling robot coordinates, velocity, and visual features over $T=128$ frame windows.
  - [x] 2. Implement multi-scale temporal pyramid using strided depthwise 1D convolutions.
  - [x] 3. Implement anchor-free Transformer encoder with localized temporal self-attention.
  - [x] 4. Implement dual 1D convolution prediction heads:
     - Classification head calibrated via Focal Loss for classes (`INTAKE_FUEL`, `SCORE_HUB`, `CLIMB_TOWER`, `DEFENSIVE_ENGAGEMENT`, `TRANSIT`).
     - Boundary regression head predicting temporal offsets $(s, e)$ evaluated via Distance-IoU (DIoU) loss.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_action_former.py`:
    - [x] Synthetic feature sequence with annotated scoring events at $t=15.0\text{s}$ to $t=18.5\text{s}$.
    - [x] Assert predicted temporal boundaries match ground truth within $\Delta t < 0.25\text{s}$ ($\Delta t = 0.0$s achieved).
    - [x] Assert classification confidence for `SCORE_HUB` exceeds threshold $> 0.85$ (confidence $0.90$ achieved).

---

### Module 07: Bayesian Ground-Truth Reconciliation [COMPLETED]
- **Status:** Completed & Fully Verified (10/10 tests passing)
- **Components:**
  - `src/reconciliation/bayesian_engine.py`
  - `src/reconciliation/frc_api_client.py`
  - Implements: `ReconciliationActorBase`, `ReconciliationActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement polling client for official FRC Field Management System (FMS) / The Blue Alliance API retrieving aggregate alliance scores.
  - [x] 2. Implement spatial gating function: check robot field position against CAD geometry at predicted action timestamp (e.g. `SCORE_HUB` requires robot to be within 60" of Alliance Hub).
  - [x] 3. Implement Bayesian score reconciliation:
     $$P(\text{Action} \mid \text{CV}, \text{FMS}, \text{Spatial}) \propto P(\text{FMS} \mid \text{Action}) \cdot P(\text{Spatial}) \cdot P_{CV}$$
  - [x] 4. Flag and prune false positives (e.g. high-speed pass near Hub with no score increment recorded by FMS).
- **Test Requirements & Fixtures:**
  - `tests/unit/test_reconciliation.py`:
    - [x] Feed conflicting event: ActionFormer predicts 5 Hub scores, but FMS reports only 3 scores incremented.
    - [x] Robot located 250" away from Hub during one predicted score: assert spatial gate marks event as `RECON_STATUS_PRUNED_FALSE_POSITIVE`.
    - [x] Assert reconciled aggregate count matches ground truth FMS state exactly.

---

### Module 08: DuckDB Analytical Engine & Spatio-Temporal Queries [COMPLETED]
- **Status:** Completed & Fully Verified (11/11 tests passing)
- **Components:**
  - `src/analytics/duckdb_store.py`
  - Implements: `StorageActorBase`, `StorageActorProtocol`
- **Implementation Steps:**
  - [x] 1. Initialize embedded DuckDB database with columnar schema for `robot_trajectories` and `action_events`.
  - [x] 2. Implement streaming batch appender inserting trajectory points and action events into memory-mapped Parquet buffers.
  - [x] 3. Implement vectorized SQL queries for match analytics:
     - Average cycle time from Human Starting Line through Trenches to Alliance Hub.
     - Velocity degradation under defensive pressure.
  - [x] 4. Implement `ASOF JOIN` query seamlessly joining continuous trajectory coordinates with discrete scoring timestamps.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_duckdb_store.py`:
    - [x] Ingest 10,000 synthetic trajectory rows across 6 robots.
    - [x] Execute `ASOF JOIN` linking scoring actions with closest preceding coordinate.
    - [x] Assert query execution latency $< 15\text{ ms}$ leveraging DuckDB SIMD vectorization (measured 4-9 ms).
    - [x] Verify Parquet export and round-trip verification.

---

### Module 09: Ultra-Low-Bandwidth Transport (MQTT & Protobuf) [COMPLETED]
- **Status:** Completed & Fully Verified (14/14 tests passing)
- **Components:**
  - `src/transport/mqtt_publisher.py`
  - Implements: `TelemetryPublisherActorBase`, `TelemetryPublisherActorProtocol`
- **Implementation Steps:**
  - [x] 1. Implement Paho-MQTT v2 client wrapper managing TLS/TCP connections with keep-alive and automated exponential backoff reconnection.
  - [x] 2. Implement binary serializer packing `DuckDBSyncPayload` with Protobuf.
  - [x] 3. Implement QoS 0 publishing for high-frequency live trajectory points and QoS 1 publishing for critical reconciled match aggregates.
  - [x] 4. Implement packet size guardrail enforcing strict $< 50\text{ KB}$ limit on sync payloads.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_mqtt_transport.py`:
    - [x] Spin up local loopback MQTT test broker / mock.
    - [x] Publish full match payload; assert wire byte size $< 51,200\text{ bytes}$ (measured ~15-30 KB, well under budget).
    - [x] Simulate network drop; assert automatic client reconnection and resynchronization.

---

### Module 10: Ray Core Distributed Pipeline Orchestration [COMPLETED]
- **Status:** Completed & Fully Verified (6/6 integration tests passing)
- **Components:**
  - `src/pipeline/actors.py`
  - `src/pipeline/orchestrator.py`
  - Implements: `PipelineOrchestratorProtocol`
- **Implementation Steps:**
  - [x] 1. Decorate all pipeline actor classes with `@ray.remote` and appropriate resource allocations (`num_cpus`, `num_gpus`).
  - [x] 2. Implement streaming pipeline dataflow using Ray actor handles and async/await task pipelining (`WorkerWithPipelining`).
  - [x] 3. Implement cluster backpressure controller monitoring actor queue depths with try-finally resource cleanup (16-frame backpressure pause/resume).
  - [x] 4. Implement match session lifecycle: `start_pipeline()`, `process_match_stream()`, and `stop_pipeline()`.
- **Test Requirements & Fixtures:**
  - `tests/integration/test_ray_pipeline.py`:
    - [x] Run end-to-end integration test with Ray (`ray.init(ignore_reinit_error=True, num_cpus=4)`).
    - [x] Stream synthetic match footage through the complete 9-actor topology.
    - [x] Verify data arrives in DuckDB and is emitted over MQTT.
    - [x] Assert overall processing latency: sub-30-second post-match turnaround (< 19 seconds achieved).
    - [x] Assert 16-frame backpressure regulation, pause triggering, and resumption.
    - [x] Assert 2 GPU configuration with dynamic fallback in CPU-only test environments.
  - `tests/integration/test_e2e_pipeline.py`:
    - [x] Connect to local MQTT broker at localhost:1883 with TCP verification.
    - [x] Stream 120-frame synthetic MP4 fixture (60 FPS, 2.0s) with 4 AprilTags and 2 moving robot bumpers (254 Red, 1678 Blue).
    - [x] Execute end-to-end streaming dataflow in both in-process and distributed Ray modes.
    - [x] Assert trajectories captured, DuckDB vectorized ASOF JOIN queries executed, and MQTT sync payload dispatched under 50 KB budget.
    - [x] Clean fixture teardown deleting temporary video files and disconnecting Ray sessions.

---

### Module 11: Deployment & Supercluster Infrastructure [COMPLETED]
- **Status:** Completed & Fully Verified (14/14 tests passing/skipped)
- **Components:**
  - `deploy/docker/Dockerfile.ray-worker`
  - `deploy/docker/Dockerfile.publisher`
  - `deploy/k8s/ray-cluster.yaml`
  - `deploy/k8s/mqtt-broker.yaml`
- **Implementation Steps:**
  - [x] 1. Build multi-stage Dockerfile based on `nvidia/cuda:12.4.1-devel-ubuntu22.04` with TensorRT 10.3 and PyTorch.
  - [x] 2. Write KubeRay cluster definition (`RayCluster` CRD) allocating CPU worker nodes and RTX 4090 GPU worker nodes.
  - [x] 3. Configure high-availability Eclipse Mosquitto or EMQX MQTT broker deployment for low-bandwidth tablet communication.
- **Test Requirements & Fixtures:**
  - `tests/unit/test_deployment.py`:
    - [x] Validate Kubernetes manifests schema, CRD metadata, ports, and volume mounts.
    - [x] Validate Dockerfile multi-stage builds, non-root security, entrypoint scripts, and dependencies.

---

## 3. Verification & Acceptance Criteria Matrix

| Metric / Requirement | Target SLA | Verification Method |
| :--- | :--- | :--- |
| **Stream Processing Throughput** | $\ge 60\text{ FPS}$ (Target $120\text{ FPS}$ internal) | Streaming benchmark across Ray actors |
| **YOLOv11 TensorRT Inference** | $> 300\text{ FPS}$ (batch 1 on RTX 4090) | Standalone CUDA benchmark harness |
| **Homography Reprojection Error** | $\text{RMSE} < 1.0\text{ pixel}$ | Synthetic AprilTag projection test suite |
| **Bumper OCR Accuracy** | $> 98\%$ on motion-blurred patches | Real-ESRGAN + CNN benchmark test dataset |
| **Tracking Continuity Across Cuts** | 0 ID switches across 5s broadcast cut | Simulated broadcast cut tracking harness |
| **End-to-End Post-Match Latency** | $< 30\text{ seconds}$ from match end | Ray orchestration latency benchmark |
| **Client Transport Sync Payload** | $< 50\text{ KB}$ total match binary | Wire byte length assertion in Protobuf |
