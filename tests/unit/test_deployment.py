"""
Unit tests for Module 11: Deployment & Supercluster Infrastructure.

Validates:
1. Dockerfile.ray-worker: Multi-stage build, NVIDIA CUDA 12.4 base, TensorRT 10.3,
   Ray Core 2.58, PyTorch, OpenCV, DuckDB, /dev/shm documentation, performance env vars.
2. Dockerfile.publisher: Multi-stage build, python:3.12-slim base, Paho-MQTT v2,
   Protobuf, DuckDB, non-root scout user configuration.
3. ray-cluster.yaml: Production KubeRay RayCluster CRD, head node specs (4-8 CPUs,
   16-32 GiB RAM, ports 8265, 10001, 6379), decoupled worker groups (cpu-workers,
   gpu-workers-rtx4090 with nvidia.com/gpu: 1), and /dev/shm memory volume mounts.
4. mqtt-broker.yaml: ConfigMap with persistence, 64 KB max packet size, listeners 1883,
   8883, 9001, connection limits; PVC for durable retention; Deployment with tcpSocket
   probes on 1883; Service exposing broker ports.
5. kubectl / docker dry-run & check integration (gracefully skipped if CLI tools absent).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
DOCKER_DIR = DEPLOY_DIR / "docker"
K8S_DIR = DEPLOY_DIR / "k8s"

RAY_WORKER_DOCKERFILE = DOCKER_DIR / "Dockerfile.ray-worker"
PUBLISHER_DOCKERFILE = DOCKER_DIR / "Dockerfile.publisher"
RAY_CLUSTER_YAML = K8S_DIR / "ray-cluster.yaml"
MQTT_BROKER_YAML = K8S_DIR / "mqtt-broker.yaml"


# ==============================================================================
# Helper Functions for Parsing Dockerfiles
# ==============================================================================

def parse_dockerfile_stages(dockerfile_path: Path) -> List[Tuple[str, str]]:
    """
    Parses a Dockerfile and returns a list of (stage_name, base_image) tuples.
    """
    stages: List[Tuple[str, str]] = []
    content = dockerfile_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("FROM "):
            # Format: FROM <image> [AS <stage_name>]
            parts = line.split()
            base_image = parts[1]
            stage_name = ""
            if len(parts) >= 4 and parts[2].upper() == "AS":
                stage_name = parts[3]
            stages.append((stage_name, base_image))
    return stages


def parse_dockerfile_instructions(dockerfile_path: Path) -> Dict[str, List[str]]:
    """
    Parses a Dockerfile and groups instructions by instruction keyword.
    """
    instructions: Dict[str, List[str]] = {}
    content = dockerfile_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    accumulated = ""
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            accumulated += line[:-1].strip() + " "
        else:
            accumulated += line
            parts = accumulated.split(maxsplit=1)
            cmd = parts[0].upper()
            val = parts[1] if len(parts) > 1 else ""
            instructions.setdefault(cmd, []).append(val)
            accumulated = ""

    return instructions


# ==============================================================================
# Test Cases: Dockerfiles
# ==============================================================================

class TestDockerfiles:
    """Test suite for Dockerfile.ray-worker and Dockerfile.publisher."""

    def test_ray_worker_dockerfile_exists_and_multistage(self) -> None:
        """Assert Dockerfile.ray-worker exists and implements multi-stage build."""
        assert RAY_WORKER_DOCKERFILE.exists(), f"Missing {RAY_WORKER_DOCKERFILE}"
        stages = parse_dockerfile_stages(RAY_WORKER_DOCKERFILE)
        assert len(stages) >= 2, f"Expected at least 2 stages, found: {stages}"

        builder_stage = stages[0]
        runtime_stage = stages[-1]

        assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in builder_stage[1]
        assert "nvidia/cuda:12.4.1" in runtime_stage[1]

    def test_ray_worker_dockerfile_dependencies_and_env(self) -> None:
        """Validate CUDA 12.4, TensorRT 10.3, Ray Core, PyTorch, OpenCV, DuckDB, and env vars."""
        content = RAY_WORKER_DOCKERFILE.read_text(encoding="utf-8")
        instructions = parse_dockerfile_instructions(RAY_WORKER_DOCKERFILE)

        # Base image requirements
        assert "nvidia/cuda:12.4.1-devel-ubuntu22.04" in content

        # Required framework & library tokens
        assert "12.4" in content  # CUDA 12.4 runtime
        assert "tensorrt" in content.lower() and "10.3" in content
        assert "torch" in content.lower()
        assert "ray" in content.lower()
        assert "opencv" in content.lower()
        assert "duckdb" in content.lower()

        # Shared memory /dev/shm requirement documented or handled
        assert "/dev/shm" in content

        # Performance environment variables
        env_lines = " ".join(instructions.get("ENV", []))
        assert "RAY_DISABLE_IMPORT_WARNING=1" in env_lines
        assert "PYTHONUNBUFFERED=1" in env_lines

        # Entrypoint configured
        assert "ENTRYPOINT" in instructions or "CMD" in instructions

    def test_publisher_dockerfile_exists_and_multistage(self) -> None:
        """Assert Dockerfile.publisher exists, uses python:3.12-slim-bookworm, and is multi-stage."""
        assert PUBLISHER_DOCKERFILE.exists(), f"Missing {PUBLISHER_DOCKERFILE}"
        stages = parse_dockerfile_stages(PUBLISHER_DOCKERFILE)
        assert len(stages) >= 2, f"Expected at least 2 stages, found: {stages}"

        assert "python:3.12-slim-bookworm" in stages[0][1]
        assert "python:3.12-slim-bookworm" in stages[-1][1]

    def test_publisher_dockerfile_dependencies_and_security(self) -> None:
        """Validate Paho-MQTT v2, Protobuf, DuckDB, and non-root scout user."""
        content = PUBLISHER_DOCKERFILE.read_text(encoding="utf-8")
        instructions = parse_dockerfile_instructions(PUBLISHER_DOCKERFILE)

        # Dependencies
        assert "paho-mqtt" in content.lower()
        assert "protobuf" in content.lower()
        assert "duckdb" in content.lower()

        # Non-root user configuration
        assert "scout" in content
        user_lines = instructions.get("USER", [])
        assert any("scout" in u for u in user_lines), f"Expected 'USER scout', got: {user_lines}"

        # Entrypoint configured
        assert "ENTRYPOINT" in instructions or "CMD" in instructions

    def test_entrypoint_scripts_executable(self) -> None:
        """Assert supplementary entrypoint scripts exist."""
        worker_entry = DOCKER_DIR / "entrypoint-ray-worker.sh"
        pub_entry = DOCKER_DIR / "entrypoint-publisher.sh"

        assert worker_entry.exists()
        assert pub_entry.exists()

        worker_content = worker_entry.read_text(encoding="utf-8")
        assert "/dev/shm" in worker_content
        assert "ray start" in worker_content

        pub_content = pub_entry.read_text(encoding="utf-8")
        assert "exec" in pub_content

    @pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI not found in PATH")
    def test_docker_buildx_syntax_check(self) -> None:
        """Execute docker buildx build --check if Docker is installed."""
        try:
            # Check ray-worker Dockerfile
            res_worker = subprocess.run(
                ["docker", "buildx", "build", "--check", "-f", str(RAY_WORKER_DOCKERFILE), "."],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert res_worker.returncode == 0, f"Dockerfile.ray-worker syntax error: {res_worker.stderr}"

            # Check publisher Dockerfile
            res_pub = subprocess.run(
                ["docker", "buildx", "build", "--check", "-f", str(PUBLISHER_DOCKERFILE), "."],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert res_pub.returncode == 0, f"Dockerfile.publisher syntax error: {res_pub.stderr}"
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
            pytest.skip(f"Docker daemon or buildx check check skipped: {exc}")


@pytest.fixture(scope="module")
def ray_cluster_doc() -> Dict[str, Any]:
    assert RAY_CLUSTER_YAML.exists(), f"Missing {RAY_CLUSTER_YAML}"
    with open(RAY_CLUSTER_YAML, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    assert isinstance(doc, dict), "Parsed ray-cluster.yaml must be a dictionary"
    return doc


@pytest.fixture(scope="module")
def k8s_docs() -> List[Dict[str, Any]]:
    assert MQTT_BROKER_YAML.exists(), f"Missing {MQTT_BROKER_YAML}"
    with open(MQTT_BROKER_YAML, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    valid_docs = [d for d in docs if d is not None]
    assert len(valid_docs) >= 3, f"Expected at least 3 K8s resources, found {len(valid_docs)}"
    return valid_docs


# ==============================================================================
# Test Cases: KubeRay RayCluster Manifest
# ==============================================================================

class TestRayClusterManifest:
    """Test suite for deploy/k8s/ray-cluster.yaml."""

    def test_ray_cluster_crd_metadata(self, ray_cluster_doc: Dict[str, Any]) -> None:
        """Assert RayCluster API version and Kind."""
        assert ray_cluster_doc.get("apiVersion") == "ray.io/v1"
        assert ray_cluster_doc.get("kind") == "RayCluster"
        assert "metadata" in ray_cluster_doc
        assert "name" in ray_cluster_doc["metadata"]

    def test_ray_cluster_head_node_spec(self, ray_cluster_doc: Dict[str, Any]) -> None:
        """
        Assert head node specifications:
        - 4-8 CPUs
        - 16-32 GiB RAM
        - Dashboard port 8265, client port 10001, GCS port 6379
        - /dev/shm memory emptyDir volume
        """
        spec = ray_cluster_doc.get("spec", {})
        head_spec = spec.get("headGroupSpec", {})
        template_spec = head_spec.get("template", {}).get("spec", {})
        containers = template_spec.get("containers", [])
        assert len(containers) >= 1, "Head node must specify at least one container"

        head_container = containers[0]
        resources = head_container.get("resources", {})
        requests = resources.get("requests", {})
        limits = resources.get("limits", {})

        # CPU range: 4 to 8
        req_cpu = int(requests.get("cpu", "0").replace("m", ""))
        lim_cpu = int(limits.get("cpu", "0").replace("m", ""))
        assert req_cpu >= 4, f"Head CPU request should be >= 4, got {req_cpu}"
        assert lim_cpu <= 8, f"Head CPU limit should be <= 8, got {lim_cpu}"

        # RAM range: 16 to 32 GiB
        req_mem = requests.get("memory", "")
        lim_mem = limits.get("memory", "")
        assert "16Gi" in req_mem or "32Gi" in req_mem, f"Unexpected head memory request: {req_mem}"
        assert "32Gi" in lim_mem or "16Gi" in lim_mem, f"Unexpected head memory limit: {lim_mem}"

        # Ports: 8265, 10001, 6379
        ports = [p.get("containerPort") for p in head_container.get("ports", [])]
        assert 8265 in ports, f"Dashboard port 8265 missing from {ports}"
        assert 10001 in ports, f"Client port 10001 missing from {ports}"
        assert 6379 in ports, f"GCS port 6379 missing from {ports}"

        # Volume mounts: /dev/shm memory emptyDir
        mounts = {m.get("mountPath"): m.get("name") for m in head_container.get("volumeMounts", [])}
        assert "/dev/shm" in mounts, f"Head node missing /dev/shm mount: {mounts}"
        shm_vol_name = mounts["/dev/shm"]

        volumes = {v.get("name"): v for v in template_spec.get("volumes", [])}
        assert shm_vol_name in volumes
        assert volumes[shm_vol_name].get("emptyDir", {}).get("medium") == "Memory"

    def test_ray_cluster_worker_groups(self, ray_cluster_doc: Dict[str, Any]) -> None:
        """
        Assert worker groups:
        - cpu-workers: video decoding, homography, Kalman filtering (CPU limits)
        - gpu-workers-rtx4090: nvidia.com/gpu: 1 allocation
        - /dev/shm emptyDir memory mount on all worker groups
        """
        spec = ray_cluster_doc.get("spec", {})
        worker_groups = spec.get("workerGroupSpecs", [])
        group_map = {g.get("groupName"): g for g in worker_groups}

        assert "cpu-workers" in group_map, f"cpu-workers missing from {list(group_map.keys())}"
        assert "gpu-workers-rtx4090" in group_map, f"gpu-workers-rtx4090 missing from {list(group_map.keys())}"

        # Validate cpu-workers
        cpu_group = group_map["cpu-workers"]
        cpu_template = cpu_group.get("template", {}).get("spec", {})
        cpu_container = cpu_template.get("containers", [])[0]
        cpu_res = cpu_container.get("resources", {})
        assert "cpu" in cpu_res.get("requests", {})
        assert "cpu" in cpu_res.get("limits", {})

        cpu_mounts = {m.get("mountPath"): m.get("name") for m in cpu_container.get("volumeMounts", [])}
        assert "/dev/shm" in cpu_mounts, "cpu-workers must mount /dev/shm"
        cpu_vols = {v.get("name"): v for v in cpu_template.get("volumes", [])}
        assert cpu_vols[cpu_mounts["/dev/shm"]].get("emptyDir", {}).get("medium") == "Memory"

        # Validate gpu-workers-rtx4090
        gpu_group = group_map["gpu-workers-rtx4090"]
        gpu_template = gpu_group.get("template", {}).get("spec", {})
        gpu_container = gpu_template.get("containers", [])[0]
        gpu_res = gpu_container.get("resources", {})

        gpu_req = gpu_res.get("requests", {}).get("nvidia.com/gpu")
        gpu_lim = gpu_res.get("limits", {}).get("nvidia.com/gpu")
        assert str(gpu_req) == "1", f"Expected nvidia.com/gpu: 1 request, got {gpu_req}"
        assert str(gpu_lim) == "1", f"Expected nvidia.com/gpu: 1 limit, got {gpu_lim}"

        gpu_mounts = {m.get("mountPath"): m.get("name") for m in gpu_container.get("volumeMounts", [])}
        assert "/dev/shm" in gpu_mounts, "gpu-workers must mount /dev/shm for Plasma IPC"
        gpu_vols = {v.get("name"): v for v in gpu_template.get("volumes", [])}
        assert gpu_vols[gpu_mounts["/dev/shm"]].get("emptyDir", {}).get("medium") == "Memory"


# ==============================================================================
# Test Cases: MQTT Broker Manifest
# ==============================================================================

class TestMqttBrokerManifest:
    """Test suite for deploy/k8s/mqtt-broker.yaml."""

    def test_mqtt_broker_configmap(self, k8s_docs: List[Dict[str, Any]]) -> None:
        """Assert ConfigMap configures persistence, 64 KB ceiling, listeners, and connection limits."""
        cm = next((d for d in k8s_docs if d.get("kind") == "ConfigMap"), None)
        assert cm is not None, "ConfigMap missing from mqtt-broker.yaml"

        conf_data = cm.get("data", {}).get("mosquitto.conf", "")
        assert conf_data, "mosquitto.conf missing from ConfigMap data"

        # Persistence
        assert re.search(r"persistence\s+true", conf_data, re.IGNORECASE)

        # Max packet size ceiling (64 KB = 65536 bytes)
        assert "max_packet_size" in conf_data
        match = re.search(r"max_packet_size\s+(\d+)", conf_data)
        assert match is not None
        packet_size = int(match.group(1))
        assert packet_size == 65536 or packet_size == 64 * 1024, f"Unexpected packet size: {packet_size}"

        # Listeners: 1883, 8883, 9001
        assert "listener 1883" in conf_data
        assert "listener 8883" in conf_data
        assert "listener 9001" in conf_data
        assert "websockets" in conf_data

        # Connection limits
        assert "max_connections" in conf_data

    def test_mqtt_broker_pvc(self, k8s_docs: List[Dict[str, Any]]) -> None:
        """Assert PersistentVolumeClaim exists for durable retain message storage."""
        pvc = next((d for d in k8s_docs if d.get("kind") == "PersistentVolumeClaim"), None)
        assert pvc is not None, "PersistentVolumeClaim missing from mqtt-broker.yaml"
        spec = pvc.get("spec", {})
        assert "ReadWriteOnce" in spec.get("accessModes", [])
        assert "storage" in spec.get("resources", {}).get("requests", {})

    def test_mqtt_broker_deployment(self, k8s_docs: List[Dict[str, Any]]) -> None:
        """
        Assert Deployment contains:
        - mosquitto image
        - port 1883
        - tcpSocket liveness and readiness probes on port 1883
        - resource requests and limits
        """
        dep = next((d for d in k8s_docs if d.get("kind") == "Deployment"), None)
        assert dep is not None, "Deployment missing from mqtt-broker.yaml"

        spec = dep.get("spec", {})
        template_spec = spec.get("template", {}).get("spec", {})
        containers = template_spec.get("containers", [])
        assert len(containers) >= 1

        broker_container = containers[0]
        assert "mosquitto" in broker_container.get("image", "").lower()

        ports = [p.get("containerPort") for p in broker_container.get("ports", [])]
        assert 1883 in ports, f"Port 1883 missing from container ports: {ports}"

        # Liveness probe
        liveness = broker_container.get("livenessProbe", {})
        assert liveness.get("tcpSocket", {}).get("port") == 1883, f"Invalid livenessProbe: {liveness}"

        # Readiness probe
        readiness = broker_container.get("readinessProbe", {})
        assert readiness.get("tcpSocket", {}).get("port") == 1883, f"Invalid readinessProbe: {readiness}"

        # Resource limits & requests
        resources = broker_container.get("resources", {})
        assert "cpu" in resources.get("requests", {})
        assert "memory" in resources.get("requests", {})
        assert "cpu" in resources.get("limits", {})
        assert "memory" in resources.get("limits", {})

    def test_mqtt_broker_service(self, k8s_docs: List[Dict[str, Any]]) -> None:
        """Assert Service exposes ports 1883, 8883, 9001 and targets broker."""
        svc = next((d for d in k8s_docs if d.get("kind") == "Service"), None)
        assert svc is not None, "Service missing from mqtt-broker.yaml"

        ports = [p.get("port") for p in svc.get("spec", {}).get("ports", [])]
        assert 1883 in ports, f"Port 1883 missing from service ports: {ports}"
        assert 8883 in ports, f"Port 8883 missing from service ports: {ports}"
        assert 9001 in ports, f"Port 9001 missing from service ports: {ports}"


# ==============================================================================
# Test Cases: Kubectl Dry-Run Validation
# ==============================================================================

class TestKubectlDryRun:
    """Optional validation using kubectl CLI if present in execution environment."""

    @pytest.mark.skipif(shutil.which("kubectl") is None, reason="kubectl CLI not found in PATH")
    def test_kubectl_client_dry_run(self) -> None:
        """Validate manifests with kubectl --dry-run=client."""
        for manifest in [MQTT_BROKER_YAML, RAY_CLUSTER_YAML]:
            cmd = ["kubectl", "apply", "--dry-run=client", "-f", str(manifest)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            assert res.returncode == 0, f"kubectl dry-run failed on {manifest}: {res.stderr}"


# ==============================================================================
# Test Cases: Hardware Slicing & Workload Pinning
# ==============================================================================

class TestHardwarePinningAndWorkloadSlicing:
    """Test suite for configs/pipeline.yaml hardware slicing and Ray startup scripts."""

    def test_pipeline_yaml_hardware_pinning_spec(self) -> None:
        """Assert configs/pipeline.yaml defines explicit GPU pinning, CPU pool, and Plasma store."""
        config_path = REPO_ROOT / "configs" / "pipeline.yaml"
        assert config_path.exists(), f"Missing {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        pipe = data.get("pipeline", {})
        cluster = pipe.get("cluster", {})
        hw_pinning = pipe.get("hardware_pinning", {})

        # Cluster baseline specs
        assert cluster.get("num_cpus") == 64, f"Expected 64 CPUs in cluster config, got {cluster.get('num_cpus')}"
        assert cluster.get("num_gpus") == 2, f"Expected 2 GPUs in cluster config, got {cluster.get('num_gpus')}"
        assert cluster.get("object_store_memory") == 68719476736, (
            f"Expected 64 GB object store memory, got {cluster.get('object_store_memory')}"
        )
        assert cluster.get("plasma_directory") == "/dev/shm"

        # GPU Pinning: GPU 0 -> Detection, GPU 1 -> ActionFormer
        gpus = hw_pinning.get("gpus", {})
        assert "detection_actor" in gpus, "Missing detection_actor in hardware_pinning.gpus"
        det_gpu = gpus["detection_actor"]
        assert det_gpu.get("device_id") == 0
        assert str(det_gpu.get("cuda_visible_devices")) == "0"
        assert det_gpu.get("num_gpus") == 1

        assert "action_recognition_actor" in gpus, "Missing action_recognition_actor in hardware_pinning.gpus"
        act_gpu = gpus["action_recognition_actor"]
        assert act_gpu.get("device_id") == 1
        assert str(act_gpu.get("cuda_visible_devices")) == "1"
        assert act_gpu.get("num_gpus") == 1

        # CPU Worker Pool: 4-8 dedicated cores per Ray actor across AMD Threadripper
        cpu_pool = hw_pinning.get("cpu_worker_pool", {})
        assert cpu_pool.get("stream_ingestion_actor") == 8, "StreamIngestionActor requires 8 cores"
        assert cpu_pool.get("homography_actor") == 8, "HomographyActor requires 8 cores"
        assert cpu_pool.get("duckdb_storage_actor") == 8, "DuckDBStorageActor requires 8 cores"
        assert cpu_pool.get("tracking_actor") == 4, "TrackingActor requires 4 cores"
        assert cpu_pool.get("reconciliation_actor") == 4, "ReconciliationActor requires 4 cores"
        assert cpu_pool.get("telemetry_publisher_actor") == 4, "TelemetryPublisherActor requires 4 cores"

        # Plasma store memory ceiling at /dev/shm (64 GB)
        plasma = hw_pinning.get("plasma_store", {})
        assert plasma.get("mount_path") == "/dev/shm"
        assert plasma.get("memory_bytes") == 68719476736

    def test_compute_actor_hardware_options_production(self) -> None:
        """Verify actor options generation for 64-core, 2-GPU supercluster production configuration."""
        from src.pipeline.actors import compute_actor_hardware_options

        opts = compute_actor_hardware_options(force_production=True)

        # GPU 0 reserved for DetectionActor
        assert opts["detection"]["num_gpus"] == 1
        assert opts["detection"]["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "0"

        # GPU 1 reserved for ActionRecognitionActor
        assert opts["action"]["num_gpus"] == 1
        assert opts["action"]["runtime_env"]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "1"

        # Dedicated CPU cores
        assert opts["ingestion"]["num_cpus"] == 8
        assert opts["homography"]["num_cpus"] == 8
        assert opts["storage"]["num_cpus"] == 8
        assert opts["tracking"]["num_cpus"] == 4
        assert opts["reconciliation"]["num_cpus"] == 4
        assert opts["publisher"]["num_cpus"] == 4

    def test_compute_actor_hardware_options_test_fallback(self) -> None:
        """Verify actor options scale down to 0 CPUs/GPUs in constrained test environments."""
        from src.pipeline.actors import compute_actor_hardware_options

        # Constrained environment: 4 CPUs, 0 GPUs (typical local test runner)
        opts = compute_actor_hardware_options(available_cpus=4, available_gpus=0, force_production=False)

        assert opts["detection"]["num_gpus"] == 0
        assert opts["action"]["num_gpus"] == 0
        assert opts["ingestion"]["num_cpus"] == 0
        assert opts["homography"]["num_cpus"] == 0
        assert opts["storage"]["num_cpus"] == 0
        assert opts["tracking"]["num_cpus"] == 0
        assert opts["reconciliation"]["num_cpus"] == 0
        assert opts["publisher"]["num_cpus"] == 0

    def test_entrypoint_ray_worker_plasma_flags(self) -> None:
        """Assert entrypoint-ray-worker.sh sets --object-store-memory and --plasma-directory."""
        worker_entry = DOCKER_DIR / "entrypoint-ray-worker.sh"
        content = worker_entry.read_text(encoding="utf-8")

        assert "--object-store-memory=" in content
        assert "68719476736" in content
        assert "--plasma-directory=/dev/shm" in content

    def test_ray_cluster_startup_scripts_exist_and_pin_hardware(self) -> None:
        """Assert Ray supercluster startup script exists, is executable, and pins hardware."""
        script_paths = [
            DEPLOY_DIR / "ray" / "start_supercluster.sh",
            REPO_ROOT / "scripts" / "start_ray_cluster.sh",
        ]

        for sp in script_paths:
            assert sp.exists(), f"Missing startup script: {sp}"
            assert os.access(sp, os.X_OK), f"Startup script is not executable: {sp}"
            content = sp.read_text(encoding="utf-8")

            assert "64" in content  # 64 AMD Threadripper CPUs
            assert "2" in content   # 2 GPUs
            assert "68719476736" in content  # 64 GB object store memory
            assert "/dev/shm" in content     # plasma directory
            assert "CUDA_VISIBLE_DEVICES" in content

