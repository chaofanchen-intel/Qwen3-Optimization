#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

SOURCE_DIR="$Q3_OPT_ROOT/models/source/Qwen3-0.6B"
OUTPUT_DIR="$Q3_OPT_ROOT/models/converted"
LOG_DIR="$Q3_OPT_ROOT/results/conversion"
FORCE=0
SKIP_DOWNLOAD=0

usage() {
    cat <<'EOF'
Usage: bash scripts/convert_models.sh [all|download|fp16|fp32|int8|int4-g128|int4-channelwise] [--force] [--skip-download]

The default target is all. All conversions use ~/envs/qwen_env/bin/optimum-cli.
EOF
}

target="all"
for arg in "$@"; do
    case "$arg" in
        all|download|fp16|fp32|int8|int4-g128|int4-channelwise) target="$arg" ;;
        --force) FORCE=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$arg" >&2; usage >&2; exit 2 ;;
    esac
done

mkdir -p "$SOURCE_DIR" "$OUTPUT_DIR" "$LOG_DIR"
q3_require_environment

download_source() {
    if [[ "$SKIP_DOWNLOAD" -eq 1 ]]; then
        return
    fi
    if [[ -f "$SOURCE_DIR/config.json" && -f "$SOURCE_DIR/model.safetensors.index.json" ]]; then
        printf '[skip] source checkpoint already exists: %s\n' "$SOURCE_DIR"
        return
    fi
    printf '[download] Qwen/Qwen3-0.6B -> %s\n' "$SOURCE_DIR"
    "$Q3_PYTHON" - "$SOURCE_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=sys.argv[1],
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
PY
}

convert_one() {
    local name="$1"
    shift
    local output="$OUTPUT_DIR/$name"
    local log="$LOG_DIR/${name}.log"
    local command=("$Q3_OPTIMUM_CLI" export openvino -m "$SOURCE_DIR" "$@" "$output")
    if [[ "$FORCE" -eq 0 && -f "$output/openvino_model.xml" && -f "$output/openvino_model.bin" ]]; then
        printf '[skip] %s already exists: %s\n' "$name" "$output"
        return
    fi
    rm -rf "$output"
    mkdir -p "$output"
    printf '[convert] %s\n' "$name"
    printf 'command: '
    printf '%q ' "${command[@]}"
    printf '\n' | tee "$log"
    "${command[@]}" 2>&1 | tee -a "$log"
}

download_source

case "$target" in
    all)
        convert_one Qwen3-0.6B-FP16 --task text-generation-with-past --weight-format fp16 --trust-remote-code
        convert_one Qwen3-0.6B-FP32 --task text-generation-with-past --trust-remote-code
        convert_one Qwen3-0.6B-INT8 --task text-generation-with-past --weight-format int8 --trust-remote-code
        convert_one Qwen3-0.6B-INT4-g128 --task text-generation-with-past --weight-format int4 --trust-remote-code
        convert_one Qwen3-0.6B-INT4-channelwise --task text-generation-with-past --weight-format int4 --sym --group-size -1 --ratio 1 --trust-remote-code
        ;;
    fp16)
        convert_one Qwen3-0.6B-FP16 --task text-generation-with-past --weight-format fp16 --trust-remote-code
        ;;
    fp32)
        convert_one Qwen3-0.6B-FP32 --task text-generation-with-past --trust-remote-code
        ;;
    int8)
        convert_one Qwen3-0.6B-INT8 --task text-generation-with-past --weight-format int8 --trust-remote-code
        ;;
    int4-g128)
        convert_one Qwen3-0.6B-INT4-g128 --task text-generation-with-past --weight-format int4 --trust-remote-code
        ;;
    int4-channelwise)
        convert_one Qwen3-0.6B-INT4-channelwise --task text-generation-with-past --weight-format int4 --sym --group-size -1 --ratio 1 --trust-remote-code
        ;;
    download)
        ;;
esac

printf '\nConversion outputs:\n'
find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort