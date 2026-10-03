#!/usr/bin/env bash
set -u

LEVEL=${1:?usage: gate1_fixed.sh LEVEL [RUNS]}
RUNS=${RUNS:-100}
WARMUP=${WARMUP:-20}
HERE=$(cd "$(dirname "$0")" && pwd)
OWNER=${SUDO_USER:-zen}
CARD=${CARD:-/sys/class/drm/card4/device}
HWMON=${HWMON:-/sys/class/drm/card4/device/hwmon/hwmon7}
STATE_FILE=${STATE_FILE:-/tmp/poc_clock_state.env}
export CARD HWMON STATE_FILE

if [ "$(id -u)" != "0" ]; then
    echo "error: run with sudo: sudo bash $0 $LEVEL" >&2
    exit 1
fi

restore() {
    echo "=== restoring performance level ==="
    bash "$HERE/poc_clock_state.sh" restore
}
trap restore EXIT
trap 'exit 130' INT TERM

bash "$HERE/poc_clock_state.sh" save
bash "$HERE/poc_clock_state.sh" set "$LEVEL"

run_case() {
    local tag=$1 ka=$2
    echo "=== $tag (cp_keepalive=$ka, level=$LEVEL) ==="
    sudo -u "$OWNER" env HWMON="$HWMON" CARDDIR="$CARD" \
        RUNS="$RUNS" WARMUP="$WARMUP" \
        bash "$HERE/run_gate1.sh" "$tag" "$ka"
}

run_case fixed_cp_off_1 0
run_case fixed_cp_on_1 1
run_case fixed_cp_on_2 1
run_case fixed_cp_off_2 0

echo "=== all fixed-state cases done ==="
