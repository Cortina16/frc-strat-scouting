#!/usr/bin/env bash
set -e

# ==============================================================================
# FRC 2026 REBUILT: Ray Supercluster Head Node Startup Script
# ==============================================================================
# Starts Ray Core cluster with explicit hardware slicing & workload pinning:
# - 64 AMD Threadripper CPU threads (--num-cpus=64)
# - 2 NVIDIA GPUs (--num-gpus=2, GPU 0: Detection, GPU 1: ActionFormer)
# - 64 GB Plasma object store mounted directly in /dev/shm (--object-store-memory=68719476736)
# - Plasma directory at /dev/shm (--plasma-directory=/dev/shm)
# - Configures CUDA_VISIBLE_DEVICES="0,1"
# ==============================================================================

NUM_CPUS=${RAY_NUM_CPUS:-64}
NUM_GPUS=${RAY_NUM_GPUS:-2}
OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-68719476736} # 64 GB
PLASMA_DIR=${RAY_PLASMA_DIRECTORY:-"/dev/shm"}
DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
PORT=${RAY_PORT:-6379}

# Pin visible GPUs for cluster head
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

echo "=============================================================================="
echo "Starting FRC 2026 Ray Supercluster with Hardware Slicing"
echo "CPUs: ${NUM_CPUS}, GPUs: ${NUM_GPUS}, CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Plasma Directory: ${PLASMA_DIR}, Object Store Memory: ${OBJECT_STORE_MEMORY} bytes"
echo "=============================================================================="

exec ray start --head \
    --port="${PORT}" \
    --num-cpus="${NUM_CPUS}" \
    --num-gpus="${NUM_GPUS}" \
    --object-store-memory="${OBJECT_STORE_MEMORY}" \
    --plasma-directory="${PLASMA_DIR}" \
    --dashboard-host="0.0.0.0" \
    --dashboard-port="${DASHBOARD_PORT}" \
    "$@"
