#!/usr/bin/env bash

# Shared runtime contract for every Qwen3-Optimization script.
Q3_OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q3_REPO_ROOT="$(cd "$Q3_OPT_ROOT/.." && pwd)"
Q3_PYTHON_ENV="${Q3_PYTHON_ENV:-$HOME/envs/qwen_env}"
Q3_PYTHON="$Q3_PYTHON_ENV/bin/python"
Q3_OPTIMUM_CLI="$Q3_PYTHON_ENV/bin/optimum-cli"
Q3_OV_ROOT="${Q3_OV_ROOT:-$Q3_REPO_ROOT/benchmark/openvino}"
Q3_OV_INSTALL="$Q3_OV_ROOT/install"
Q3_LLM_BENCH="$Q3_OV_ROOT.genai/tools/llm_bench/benchmark.py"
Q3_CPP_BENCH="$Q3_REPO_ROOT/model_bench/build/model_bench"

export Q3_OPT_ROOT Q3_REPO_ROOT Q3_PYTHON_ENV Q3_PYTHON Q3_OPTIMUM_CLI
export Q3_OV_ROOT Q3_OV_INSTALL Q3_LLM_BENCH Q3_CPP_BENCH
export PYTHONPATH="$Q3_OV_ROOT.genai/tools/llm_bench:$Q3_OV_INSTALL/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$Q3_OV_INSTALL/runtime/lib/intel64:$Q3_OV_INSTALL/runtime/3rdparty/tbb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

q3_require_environment() {
    local missing=0
    for required in "$Q3_PYTHON" "$Q3_OPTIMUM_CLI" "$Q3_LLM_BENCH" "$Q3_CPP_BENCH"; do
        if [[ ! -e "$required" ]]; then
            printf 'missing required path: %s\n' "$required" >&2
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        return 1
    fi
    "$Q3_PYTHON" - <<'PY'
import openvino
import openvino_genai
assert openvino.__version__.startswith("2026.3.1"), openvino.__version__
print(f"OpenVINO={openvino.__version__}")
print(f"OpenVINO-GenAI={openvino_genai.__version__}")
PY
}