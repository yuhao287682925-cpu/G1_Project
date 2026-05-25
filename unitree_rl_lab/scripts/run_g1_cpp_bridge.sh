#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SIM_BUILD_DIR="${ROOT_DIR}/unitree_mujoco/simulate/build"
CTRL_BUILD_DIR="${DEPLOY_DIR}/deploy/robots/g1_29dof/build"

SIM_BIN="${SIM_BUILD_DIR}/unitree_mujoco"
CTRL_BIN="${CTRL_BUILD_DIR}/g1_ctrl"

usage() {
  cat <<'EOF'
Usage:
  run_g1_cpp_bridge.sh sim
  run_g1_cpp_bridge.sh ctrl [-n <network_if>]
  run_g1_cpp_bridge.sh vkb [--domain-id 0 --interface lo]
  run_g1_cpp_bridge.sh check

Examples:
  ./unitree_deploy/scripts/run_g1_cpp_bridge.sh sim
  ./unitree_deploy/scripts/run_g1_cpp_bridge.sh ctrl -n lo
  ./unitree_deploy/scripts/run_g1_cpp_bridge.sh vkb
  ./unitree_deploy/scripts/run_g1_cpp_bridge.sh check
EOF
}

check_bins() {
  local ok=1
  if [[ ! -x "${SIM_BIN}" ]]; then
    echo "[ERROR] Missing simulator binary: ${SIM_BIN}"
    ok=0
  fi
  if [[ ! -x "${CTRL_BIN}" ]]; then
    echo "[ERROR] Missing controller binary: ${CTRL_BIN}"
    ok=0
  fi
  if [[ "${ok}" -eq 0 ]]; then
    echo
    echo "Build with:"
    echo "  cd ${SIM_BUILD_DIR} && cmake .. && make -j\$(nproc)"
    echo "  cd ${CTRL_BUILD_DIR} && cmake .. && make -j\$(nproc)"
    exit 1
  fi
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  sim)
    check_bins
    cd "${SIM_BUILD_DIR}"
    exec ./unitree_mujoco "$@"
    ;;
  ctrl)
    check_bins
    cd "${CTRL_BUILD_DIR}"
    exec ./g1_ctrl "$@"
    ;;
  vkb)
    exec python3 "${DEPLOY_DIR}/scripts/virtual_keyboard_publisher.py" "$@"
    ;;
  check)
    check_bins
    echo "[OK] Found simulator:  ${SIM_BIN}"
    echo "[OK] Found controller: ${CTRL_BIN}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
