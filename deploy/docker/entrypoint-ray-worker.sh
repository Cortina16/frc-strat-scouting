#!/usr/bin/env bash
set -e

# ==============================================================================
# FRC 2026 REBUILT: Ray Worker Node Entrypoint Script
# ==============================================================================
# This script initializes the Ray worker process and attaches it to the KubeRay
# supercluster head node. It validates shared memory (/dev/shm) allocation to
# safeguard Plasma object store zero-copy IPC at 60-120 FPS.
# ==============================================================================

echo "[Ray-Worker] Initializing FRC Scouting Ray Worker Node..."

# 1. Inspect and validate /dev/shm shared memory configuration
if [ -d "/dev/shm" ]; then
    SHM_SIZE_KB=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $2}' || echo 0)
    SHM_SIZE_MB=$((SHM_SIZE_KB / 1024))
    echo "[Ray-Worker] Detected /dev/shm shared memory size: ${SHM_SIZE_MB} MB"
    if [ "$SHM_SIZE_MB" -lt 2048 ]; then
        echo "[Ray-Worker] WARNING: /dev/shm is under 2048 MB (${SHM_SIZE_MB} MB). Plasma zero-copy IPC may thrash or trigger OOM on high-res broadcast frames!"
    else
        echo "[Ray-Worker] /dev/shm shared memory allocation is adequate for Plasma IPC."
    fi
else
    echo "[Ray-Worker] WARNING: /dev/shm is not available as a directory!"
fi

# 2. Check NVIDIA GPU presence and print hardware specs if available
if command -v nvidia-smi &> /dev/null; then
    echo "[Ray-Worker] NVIDIA GPU accelerator detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    echo "[Ray-Worker] Running in CPU worker mode (no nvidia-smi detected in current namespace)."
fi

# 3. Determine Ray Head address from environment variable or CLI argument
RAY_HEAD="${RAY_HEAD_ADDRESS:-$1}"
OBJECT_STORE_MEM="${RAY_OBJECT_STORE_MEMORY:-68719476736}"

if [ -n "$RAY_HEAD" ] && [ "$RAY_HEAD" != "--block" ]; then
    if [ "$1" = "$RAY_HEAD" ]; then
        shift || true
    fi
    echo "[Ray-Worker] Connecting worker node to Ray Head at: ${RAY_HEAD}"
    exec ray start --address="${RAY_HEAD}" --object-store-memory="${OBJECT_STORE_MEM}" --plasma-directory=/dev/shm --block "$@"
else
    echo "[Ray-Worker] Executing ray start with default / provided parameters: $@"
    exec ray start --object-store-memory="${OBJECT_STORE_MEM}" --plasma-directory=/dev/shm "$@"
fi
