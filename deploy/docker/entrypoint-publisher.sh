#!/usr/bin/env bash
set -e

# ==============================================================================
# FRC 2026 REBUILT: Telemetry Publisher Entrypoint Script
# ==============================================================================
# Minimal, secure container entrypoint executing as non-root user 'scout'.
# Sets up broker environment and forwards execution to Python MQTT publisher.
# ==============================================================================

echo "[Publisher] Starting FRC Telemetry Publisher as user $(whoami)..."
echo "[Publisher] Broker Target: ${MQTT_BROKER_HOST:-mosquitto-broker}:${MQTT_BROKER_PORT:-1883}"
echo "[Publisher] Client ID: ${MQTT_CLIENT_ID:-scout-pub}"

exec "$@"
