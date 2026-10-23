#!/usr/bin/env bash
# Interleaved A/B of several GPU-plugin builds on one model.
#
#   plugin_ab.sh <model-dir> <rounds> <plugin.so> [<plugin.so> ...]
#
# The runtime loads the GPU plugin from the directory of libopenvino.so (bin/intel64/Release); OV_PLUGIN_PATH and
# LD_LIBRARY_PATH do NOT redirect it. So each candidate is copied over the plugin in that directory for its
# measurement and the original is restored on exit. Rounds are interleaved (A B A B ...) because the GPU drifts by
# ~1 ms over several minutes; judge by the mean over rounds, with the std next to it.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"
MODEL="${1:?usage: plugin_ab.sh <model-dir> <rounds> <plugin.so>...}"; ROUNDS="${2:?rounds}"; shift 2
TARGET="$Q3_OV_BIN/libopenvino_intel_gpu_plugin.so"
BACKUP="$(mktemp)"; cp "$TARGET" "$BACKUP"
trap 'cp "$BACKUP" "$TARGET"; rm -f "$BACKUP"' EXIT
declare -A SUM COUNT
for r in $(seq 1 "$ROUNDS"); do
  for so in "$@"; do
    cp "$so" "$TARGET"
    line="$("$HERE/ttft_mean.sh" "$MODEL" 6)"
    echo "round $r $(basename "$so") $line"
    mean="${line%% *}"
    SUM[$so]=$(awk -v a="${SUM[$so]:-0}" -v b="$mean" 'BEGIN{print a+b}'); COUNT[$so]=$(( ${COUNT[$so]:-0} + 1 ))
  done
done
echo "--- mean over rounds"
for so in "$@"; do awk -v n="$(basename "$so")" -v s="${SUM[$so]}" -v c="${COUNT[$so]}" 'BEGIN{printf "%-32s %.2f ms\n", n, s/c}'; done
