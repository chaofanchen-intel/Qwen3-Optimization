#!/usr/bin/env bash

# Shared runtime contract for every Qwen3-Optimization script.
Q3_OPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q3_REPO_ROOT="$(cd "$Q3_OPT_ROOT/.." && pwd)"
Q3_PYTHON_ENV="${Q3_PYTHON_ENV:-$HOME/envs/qwen_env}"
Q3_PYTHON="$Q3_PYTHON_ENV/bin/python"
Q3_OPTIMUM_CLI="$Q3_PYTHON_ENV/bin/optimum-cli"
Q3_OV_ROOT="${Q3_OV_ROOT:-$Q3_REPO_ROOT/benchmark/openvino}"
Q3_OV_INSTALL="$Q3_OV_ROOT/install"
Q3_OV_BIN="$Q3_OV_ROOT/bin/intel64/Release"
Q3_LLM_BENCH="$Q3_OV_ROOT.genai/tools/llm_bench/benchmark.py"
Q3_CPP_BENCH="$Q3_REPO_ROOT/model_bench/build/model_bench"

export Q3_OPT_ROOT Q3_REPO_ROOT Q3_PYTHON_ENV Q3_PYTHON Q3_OPTIMUM_CLI
export Q3_OV_ROOT Q3_OV_INSTALL Q3_OV_BIN Q3_LLM_BENCH Q3_CPP_BENCH
export PYTHONPATH="$Q3_OV_ROOT.genai/tools/llm_bench:$Q3_OV_INSTALL/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$Q3_OV_BIN/libopenvino_intel_gpu_plugin.so" ]]; then
    export OV_PLUGIN_PATH="$Q3_OV_BIN${OV_PLUGIN_PATH:+:$OV_PLUGIN_PATH}"
    export LD_LIBRARY_PATH="$Q3_OV_BIN:$Q3_OV_INSTALL/runtime/lib/intel64:$Q3_OV_INSTALL/runtime/3rdparty/tbb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$Q3_OV_INSTALL/runtime/lib/intel64:$Q3_OV_INSTALL/runtime/3rdparty/tbb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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
import os
import openvino
import openvino_genai
core = openvino.Core()
gpu_version = core.get_versions("GPU")["GPU"]
assert openvino.__version__.startswith("2026.4.1"), openvino.__version__
assert openvino_genai.__version__.startswith("2026.4.1.0"), openvino_genai.__version__
assert "2026.4.1-" in str(gpu_version), gpu_version
plugin_paths = {
    line.rstrip().split()[-1]
    for line in open("/proc/self/maps")
    if "libopenvino_intel_gpu_plugin.so" in line
}
assert any(path.startswith(os.environ["Q3_OV_BIN"]) for path in plugin_paths), plugin_paths
print(f"OpenVINO={openvino.__version__}")
print(f"OpenVINO-GenAI={openvino_genai.__version__}")
print(f"GPU plugin={gpu_version}")
PY
}