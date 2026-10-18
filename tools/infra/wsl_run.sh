#!/bin/bash
# Local Windows/WSL2 workaround ONLY (the WSL2 kernel can hang CUDA processes; native Linux does not need this script).
# Runs a python job in WSL under the stability protocol: VM age >= 300 s, one CUDA warm-up per VM boot, job timeout, kernel-hang check.
# usage: wsl_run.sh <timeout_s> <log_file> [ENV=VAL ...] -- <python script and args>   (paths as seen from WSL)
# env: FLUOROVLA_WSL_PY (python of the physics venv), FLUOROVLA_WSL_DISTRO (default Ubuntu-22.04), FLUOROVLA_WSL_CWD (working directory)
# exit codes: 0 ok, 124 timeout, 97 kernel hang detected, 98 warm-up failed
export MSYS_NO_PATHCONV=1
DISTRO=${FLUOROVLA_WSL_DISTRO:-Ubuntu-22.04}
PY=${FLUOROVLA_WSL_PY:?set FLUOROVLA_WSL_PY}
CWD=${FLUOROVLA_WSL_CWD:?set FLUOROVLA_WSL_CWD}
TMO=$1; LOG=$2; shift 2
ENVS=()
while [ "$1" != "--" ]; do ENVS+=("$1"); shift; done; shift
up() { wsl -d "$DISTRO" -e sh -c 'cut -d. -f1 /proc/uptime' 2>/dev/null | tr -d '\0' | grep -E '^[0-9]+$' | head -1; }
hang() { timeout 40 wsl -d "$DISTRO" -e sh -c 'dmesg 2>/dev/null | grep -cE "blocked for|Blocked by coredump"' 2>/dev/null | tr -d '\0' | grep -E '^[0-9]+$' | head -1; }
u=$(up); u=${u:-0}
[ "$u" -lt 300 ] && { echo "$(date +%T) VM uptime ${u}s -> waiting $((300-u))s" | tee -a "$LOG"; sleep $((300-u)); }
warm=$(wsl -d "$DISTRO" -e sh -c 'cat /tmp/fluorovla_warm_ok 2>/dev/null' | tr -d '\0')
if [ "$warm" != "ok" ]; then
  timeout 240 wsl -d "$DISTRO" -e "$PY" -c "import warp as wp; wp.init(); a=wp.zeros(1<<20, dtype=float, device='cuda:0'); wp.synchronize(); print('warp warmup ok')" >> "$LOG" 2>&1
  rc=$?; h=$(hang)
  if [ "$rc" != "0" ] || [ "${h:-1}" != "0" ]; then echo "$(date +%T) WARMUP FAILED rc=$rc hang_msgs=$h" | tee -a "$LOG"; exit 98; fi
  wsl -d "$DISTRO" -e sh -c 'echo ok > /tmp/fluorovla_warm_ok'
  echo "$(date +%T) warm-up ok" | tee -a "$LOG"
fi
echo "$(date +%T) RUN: ${ENVS[*]} $*" | tee -a "$LOG"
timeout "$TMO" wsl -d "$DISTRO" --cd "$CWD" -e env "${ENVS[@]}" "$PY" "$@" >> "$LOG" 2>&1
rc=$?
h=$(hang)
echo "$(date +%T) job rc=$rc kernel_hung_msgs=${h:-?}" | tee -a "$LOG"
[ "${h:-0}" != "0" ] && exit 97
exit $rc
