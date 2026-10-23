#!/usr/bin/env bash
# Keep this archive in step with the canonical patch directory.
#
#   sync_from_canonical.sh            # verify only: fail if the archive has drifted
#   sync_from_canonical.sh --update   # re-copy from canonical and refresh MANIFEST.sha256
#
# The patches are developed and delivered from
# ../../benchmark/openvino/qwen3_patch/ (that is the copy Intel receives, and the
# one `run_matrix.py --patch-compare` applies). This directory keeps a
# byte-identical archive next to the measurements that justify it, so the
# experiment workspace stays self-contained.
#
# Because that means two copies exist, this script is the guard: verify mode is
# cheap, and CI or a pre-delivery check should run it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="$(cd "$HERE/../../benchmark/openvino/qwen3_patch" && pwd)"

declare -A FILES=(
    [opt1_rms_block_access.patch]=opt1_rms_block_access.patch
    [opt2_dynamic_quantize_reuse.patch]=opt2_dynamic_quantize_reuse.patch
    [opt3_fuse_rope_into_rms.patch]=opt3_fuse_rope_into_rms.patch
    [opt4_pa_kv_cache_vector_update.patch]=opt4_pa_kv_cache_vector_update.patch
    [opt5_sdpa_micro_xe3_igpu_tile.patch]=opt5_sdpa_micro_xe3_igpu_tile.patch
    [opt6_fuse_uncompressed_qkv_fcs.patch]=opt6_fuse_uncompressed_qkv_fcs.patch
    [README.md]=engineering_notes.md
)

if [[ "${1:-}" == "--update" ]]; then
    for src in "${!FILES[@]}"; do
        cp "$CANONICAL/$src" "$HERE/${FILES[$src]}"
        printf 'updated %s <- %s\n' "${FILES[$src]}" "$src"
    done
    (cd "$HERE" && sha256sum ./*.patch engineering_notes.md > MANIFEST.sha256)
    echo "MANIFEST.sha256 refreshed"
    exit 0
fi

rc=0
for src in "${!FILES[@]}"; do
    dst="$HERE/${FILES[$src]}"
    if ! cmp -s "$CANONICAL/$src" "$dst"; then
        printf 'DRIFT: %s differs from %s\n' "$dst" "$CANONICAL/$src" >&2
        rc=1
    fi
done
(cd "$HERE" && sha256sum -c MANIFEST.sha256 --quiet) || rc=1

if [[ "$rc" -eq 0 ]]; then
    echo "archive matches canonical ($CANONICAL)"
else
    echo "run '$0 --update' to resync" >&2
fi
exit "$rc"
