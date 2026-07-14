#!/usr/bin/env bash
# Generate alternative routes for a set of nuPlan scenario tokens.
#
# Uses this package's own venv (see README for setup). Requires a mounted nuPlan dataset:
#   export NUPLAN_DATA_ROOT=/scratch/nuplan/dataset/nuplan-v1.1/splits/val
#   export NUPLAN_MAPS_ROOT=/scratch/nuplan/dataset/maps
#
# Usage:
#   ./scripts/generate_alternative_routes.sh <tokens_file> <output.jsonl>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "${SCRIPT_DIR}")"
PYTHON="${PKG_DIR}/.venv/bin/python"

TOKENS_FILE="${1:?Usage: $0 <tokens_file> <output.jsonl>}"
OUTPUT="${2:?Usage: $0 <tokens_file> <output.jsonl>}"

"${PYTHON}" -m nucontrol.cli \
    --tokens-file "${TOKENS_FILE}" \
    --output "${OUTPUT}" \
    --max-alternatives 3 \
    --verbose
