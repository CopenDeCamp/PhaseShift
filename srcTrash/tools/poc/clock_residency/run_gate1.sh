#!/usr/bin/env bash
set -u

TAG=${1:?usage: run_gate1.sh TAG CP_KEEPALIVE(0|1)}
KA=${2:?usage: run_gate1.sh TAG CP_KEEPALIVE(0|1)}

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
BENCH=${BENCH:-/opt/zen/wk/PhaseNonShift/build/phaseshift-bench}
MODEL=${MODEL:-/opt/zen/wk/PhaseNonShift/models/Qwen3.8-27B-PSQ}
DEVICE=${DEVICE:-3}
RUNS=${RUNS:-100}
WARMUP=${WARMUP:-20}
PROMPT=${PROMPT:-2048}
PROMPT_CHUNK=${PROMPT_CHUNK:-0}
ARENA_GIB=${ARENA_GIB:-24}
PAGE_TOKENS=${PAGE_TOKENS:-16}
OUTDIR=${OUTDIR:-$ROOT/artifacts/poc_clock_residency/gate1}
HWMON=${HWMON:-/sys/class/drm/card4/device/hwmon/hwmon7}
CARDDIR=${CARDDIR:-/sys/class/drm/card4/device}
GAPKA=${GAPKA:-0}

mkdir -p "$OUTDIR"

if [ "$KA" = "1" ]; then
    unset PHASESHIFT_KEEPALIVE
else
    export PHASESHIFT_KEEPALIVE=0
fi

TELEM="$OUTDIR/${TAG}_telemetry.csv"
STOPFLAG="$OUTDIR/.stop_${TAG}"
rm -f "$STOPFLAG"
printf 't,sclk_hz,mclk_hz,power_uw,temp_mc,gpu_busy,mem_busy\n' > "$TELEM"
(
    while [ ! -e "$STOPFLAG" ]; do
        printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s.%N)" \
            "$(cat "$HWMON/freq1_input" 2>/dev/null)" \
            "$(cat "$HWMON/freq2_input" 2>/dev/null)" \
            "$(cat "$HWMON/power1_average" 2>/dev/null)" \
            "$(cat "$HWMON/temp1_input" 2>/dev/null)" \
            "$(cat "$CARDDIR/gpu_busy_percent" 2>/dev/null)" \
            "$(cat "$CARDDIR/mem_busy_percent" 2>/dev/null)" >> "$TELEM"
        sleep 0.05
    done
) &
SAMPID=$!

EXTRA=()
if [ "$GAPKA" != "0" ]; then EXTRA=(--gap-keepalive "$GAPKA"); fi
if [ "$PROMPT_CHUNK" != "0" ]; then EXTRA+=(--prefill-chunk "$PROMPT_CHUNK"); fi

echo "tag=$TAG cp_keepalive=$KA keepalive_env=${PHASESHIFT_KEEPALIVE:-unset} gap_keepalive=$GAPKA"
"$BENCH" pp --model-dir "$MODEL" --prompt-tokens "$PROMPT" --mode forward \
    --page-tokens "$PAGE_TOKENS" --arena-gib "$ARENA_GIB" \
    --runs "$RUNS" --warmup "$WARMUP" --device "$DEVICE" "${EXTRA[@]}" \
    > "$OUTDIR/${TAG}.txt" 2>&1
BENCHRC=$?

touch "$STOPFLAG"
wait "$SAMPID" 2>/dev/null
rm -f "$STOPFLAG"

grep -E "gpu_ms_median|gpu_ms_min|gpu_ms_p10|gpu_ms_p90|gpu_ms_stddev|gpu_tokens_per_sec|wall_ms_median" \
    "$OUTDIR/${TAG}.txt"
exit "$BENCHRC"
