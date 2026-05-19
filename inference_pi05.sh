#!/bin/bash
# Launcher for pi0.5 real-Franka inference.
#
# Prereqs in this conda env:
#   pip install pyrealsense2 websockets msgpack numpy pillow dm-tree tree
#   pip install -e /home/hez2/code/openpi/packages/openpi-client
#
# Get RealSense serials from:  rs-enumerate-devices -s
# Then fill EXTERIOR_SERIAL / WRIST_SERIAL below.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/katefgroup/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Use the same env you normally run frankapy from.
conda activate franka

EXTERIOR_SERIAL="${EXTERIOR_SERIAL:-838212071165}"   # D435  (head)
WRIST_SERIAL="${WRIST_SERIAL:-040322071615}"         # D435i (wrist)
REMOTE_HOST="${REMOTE_HOST:-0.0.0.0}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

python "$SCRIPT_DIR/inference_pi05.py" \
    --exterior-serial "$EXTERIOR_SERIAL" \
    --wrist-serial "$WRIST_SERIAL" \
    --remote-host "$REMOTE_HOST" \
    --remote-port "$REMOTE_PORT" \
    --chunk-steps 5 \
    --control-hz 10 \
    "$@"
