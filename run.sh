#!/usr/bin/env bash
# run.sh — Activate venv and start SignFuture server
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec python web_bridge.py
