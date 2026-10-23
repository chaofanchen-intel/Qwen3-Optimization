#!/usr/bin/env bash
# Mean/std of cold-prefill TTFT (warm-up excluded) from the C++ driver; one line "mean std n=N".
#
#   ttft_mean.sh <model-dir> [measured_iters=8]        TOK=<input tokens, default 1000>
#
# Prompt permutation stays ON (model_bench default), so every iteration is a real prefill. Reports the arithmetic
# mean with the population std, never the p50.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"
MODEL="${1:?usage: ttft_mean.sh <model-dir> [measured_iters]}"
ITERS="${2:-8}"
timeout 300 "$Q3_CPP_BENCH" llm -m "$MODEL" -d GPU -n "$ITERS" -ic 1 --input-tokens "${TOK:-1000}" --no-monitor 2>&1 \
  | grep -E '^\s*\[[0-9]+\]\[P0\]' | sed -E 's/.*TTFT=([0-9.]+)ms.*/\1/' \
  | awk '{a[NR]=$1; s+=$1} END{m=s/NR; for(i=1;i<=NR;i++) v+=(a[i]-m)^2; printf "%.2f %.2f n=%d\n", m, sqrt(v/NR), NR}'
