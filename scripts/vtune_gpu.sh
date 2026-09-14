#!/usr/bin/env bash
# VTune GPU analysis of a Qwen3 cold prefill on the PA backend.
#
#   vtune_gpu.sh <model-dir> <tag> [mode] [iters] [tokens]
#
# mode: overview (default) | global-memory-accesses | compute-extended |
#       instruction-count | source-analysis
#
# The workload is the real ov::genai::LLMPipeline run (ATTENTION_BACKEND=PA,
# the library default) with -ic 1, so every measured iteration is a cold
# prefill and nothing but prefill is sampled. Prompt permutation stays ON so
# PA's prefix cache cannot serve the prompt without prefilling.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/Qwen3-Optimization/scripts/env.sh"
# shellcheck disable=SC1091
source /opt/intel/oneapi/vtune/latest/vtune-vars.sh >/dev/null

MODEL="${1:?usage: vtune_gpu.sh <model-dir> <tag> [mode] [iters] [tokens]}"
TAG="${2:?missing tag}"
MODE="${3:-overview}"
ITERS="${4:-5}"
TOKENS="${5:-1000}"

OUT="$ROOT/Qwen3-Optimization/results/hotspot/${TAG}_${MODE}"
rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"

# collect-memory-bandwidth needs the PAX PMU-reservation driver for uncore
# events; it is not loaded here, and enabling it aborts the whole collection.
# Set VTUNE_MEM_BW=1 once that driver is installed.
KNOBS=()
if [[ "${VTUNE_MEM_BW:-0}" == "1" ]]; then
    KNOBS+=(-knob collect-memory-bandwidth=true)
fi
if [[ "$MODE" == "source-analysis" ]]; then
    KNOBS=(-knob gpu-profiling-mode=source-analysis)
else
    KNOBS+=(-knob characterization-mode="$MODE")
fi

set -x
vtune -collect gpu-hotspots "${KNOBS[@]}" -r "$OUT" -- \
    "$Q3_CPP_BENCH" llm -m "$MODEL" -d GPU -n "$ITERS" -ic 1 \
    --input-tokens "$TOKENS" --no-monitor
set +x

echo
echo "=============== summary: $TAG / $MODE ==============="
vtune -report summary -r "$OUT" 2>/dev/null | head -80
echo
echo "=============== hottest compute tasks ==============="
vtune -report hotspots -group-by computing-task -r "$OUT" 2>/dev/null | head -40
echo
echo "result dir: $OUT"
