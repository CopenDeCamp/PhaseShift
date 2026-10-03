#!/usr/bin/env bash
set -u

CARD=${CARD:-/sys/class/drm/card2/device}
STATE_FILE=${STATE_FILE:-/tmp/poc_clock_state.env}
LEVEL_FILE="$CARD/power_dpm_force_performance_level"
DPM_STATE_FILE="$CARD/power_dpm_state"

require_root() {
    if [ "$(id -u)" != "0" ]; then
        echo "error: must run as root" >&2
        exit 1
    fi
}

save_state() {
    local level dpm_state
    level=$(cat "$LEVEL_FILE")
    dpm_state=$(cat "$DPM_STATE_FILE" 2>/dev/null || echo unknown)
    {
        echo "CARD=$CARD"
        echo "LEVEL=$level"
        echo "DPM_STATE=$dpm_state"
    } > "$STATE_FILE"
    echo "saved:"
    echo "  card=$CARD"
    echo "  power_dpm_force_performance_level=$level"
    echo "  power_dpm_state=$dpm_state"
}

set_level() {
    local level=$1
    if [ ! -f "$STATE_FILE" ]; then
        echo "error: run 'save' first" >&2
        exit 1
    fi
    printf '%s\n' "$level" > "$LEVEL_FILE" || {
        echo "error: write failed: $LEVEL_FILE" >&2
        echo "valid levels: auto low high manual profile_standard profile_min_sclk profile_min_mclk profile_peak" >&2
        exit 1
    }
    echo "set: power_dpm_force_performance_level=$level"
    echo "readback: $(cat "$LEVEL_FILE")"
}

restore_state() {
    if [ ! -f "$STATE_FILE" ]; then
        echo "error: no saved state at $STATE_FILE" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    . "$STATE_FILE"
    printf '%s\n' "$LEVEL" > "$LEVEL_FILE"
    sleep 1
    echo "restore:"
    echo "  power_dpm_force_performance_level=$(cat "$LEVEL_FILE")"
    echo "  power_dpm_state=$(cat "$DPM_STATE_FILE" 2>/dev/null || echo unknown)"
}

status() {
    echo "card=$CARD"
    echo "power_dpm_force_performance_level=$(cat "$LEVEL_FILE" 2>/dev/null)"
    echo "power_dpm_state=$(cat "$DPM_STATE_FILE" 2>/dev/null)"
    echo "state_file=$STATE_FILE"
    cat "$STATE_FILE" 2>/dev/null || true
}

require_root
case "${1:-}" in
    save) save_state ;;
    set) set_level "${2:?usage: set LEVEL}" ;;
    restore) restore_state ;;
    status) status ;;
    *) echo "usage: poc_clock_state.sh save|set LEVEL|restore|status" >&2; exit 2 ;;
esac
