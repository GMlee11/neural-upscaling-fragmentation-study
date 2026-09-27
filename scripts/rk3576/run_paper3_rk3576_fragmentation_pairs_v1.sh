#!/usr/bin/env bash
set -euo pipefail

# This command remains blocked by the checked-in ACCESS_REQUIRED contract.
# It is the exact future physical entry point after independent authorization.
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONDONTWRITEBYTECODE=1
cd "${ROOT}"
exec "${PYTHON_BIN}" -m src.deployment.benchmark_rk3576_fragmentation_pairs \
  --contract configs/paper3_rk3576_fragmentation_pairs_v1.yaml \
  --catalog results/models/sparse_refinement_rknn_v2/catalog.json \
  --output results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json \
  --require-status AUTHORIZED_FOR_ACCESS \
  --confirm-exclusive-board \
  --confirm-ac-power \
  --confirm-no-other-workloads \
  --reviewer-signoff "${PAPER3_REVIEWER_SIGNOFF:?independent reviewer signoff is required}"
