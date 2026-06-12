#!/usr/bin/env bash
# Generic CardioNet Slurm wrapper.
#
# Submit through scripts/cluster_submit.py so resources and exported variables
# are generated from config/cluster.yaml.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"

if [[ -n "${CARDIONET_ACTIVATE_COMMAND:-}" ]]; then
  eval "${CARDIONET_ACTIVATE_COMMAND}"
fi

command=(python scripts/cluster_run_stage.py --stage "${CARDIONET_STAGE}")

if [[ -n "${CARDIONET_MANIFEST:-}" ]]; then
  command+=(--manifest "${CARDIONET_MANIFEST}")
fi

if [[ -n "${CARDIONET_CONFIG:-}" ]]; then
  command+=(--config "${CARDIONET_CONFIG}")
fi

"${command[@]}"
