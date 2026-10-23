# Qwen3 GPU-plugin patches — archive and index

Intel GPU plugin patches that reduce **cold-start prefill / TTFT** for dense
Qwen3-0.6B and Qwen3-1.7B. The current rebenchmark focuses on FP16 and INT8,
with INT4 `group_size=128` and channel-wise variants also measured.

This directory is an **archive** kept next to the measurements that justify it,
so `Qwen3-Optimization/` is self-contained. The patches are developed and
delivered from
[`../../benchmark/openvino/qwen3_patch/`](../../benchmark/openvino/qwen3_patch/) —
that is the canonical copy, the one Intel receives, and the one
`scripts/run_matrix.py --patch-compare` applies.

The performance tables retained in `engineering_notes.md` are historical
2026.3.1 TTFT data. Current 2026.4.1 full-generation measurements for each
cumulative patch stage are in
[`../docs/performance_2026.4.1.md`](../docs/performance_2026.4.1.md).

Two copies means they can drift, so there is a guard:

```bash
./sync_from_canonical.sh            # verify (fails on drift); safe to run in CI
./sync_from_canonical.sh --update   # re-copy from canonical, refresh MANIFEST.sha256
```

## Contents

| File | What it is |
|---|---|
| `opt1_rms_block_access.patch` | opt-1 source patch |
| `opt2_dynamic_quantize_reuse.patch` | opt-2 source patch |
| `opt3_fuse_rope_into_rms.patch` | opt-3 source patch |
| `opt4_pa_kv_cache_vector_update.patch` | opt-4 source patch (FP16 KV-cache update with vector stores; bit-exact) |
| `opt5_sdpa_micro_xe3_igpu_tile.patch` | opt-5 source patch (`sdpa_micro` tile for integrated Xe3; **not bit-exact**, see the notes) |
| `opt6_fuse_uncompressed_qkv_fcs.patch` | opt-6 source patch (Q/K/V fusion for uncompressed FCs, with unit tests; bit-exact) |
| `engineering_notes.md` | Snapshot of the canonical patch README: full rationale, per-kernel VTune numbers, correctness argument, negative results, open items |
| `MANIFEST.sha256` | Hashes of everything above |
| `sync_from_canonical.sh` | Drift guard / resync |

Related documents in this workspace:

| Document | Contents |
|---|---|
| [`../docs/optimizations.md`](../docs/optimizations.md) | Experiment log: how the hotspots were found, what was tried, what was rejected and why |
| [`../docs/results.md`](../docs/results.md) | Historical 2026.3.1 TTFT report |
| [`../docs/performance_2026.4.1.md`](../docs/performance_2026.4.1.md) | Current 2026.4.1 full-generation rebenchmark |
| [`../results/`](../results/) | Raw per-run data with provenance (`metadata.json` records source commit, patch sha256, patch state, plugin binary hash) |

## Base and apply order

Generated against OpenVINO 2026.4.1 and validated against the pristine
`2026.4_base` checkout. Apply in order to a pristine tree:

```bash
cd <openvino-root>
git apply <this-dir>/opt1_rms_block_access.patch
git apply <this-dir>/opt2_dynamic_quantize_reuse.patch
git apply <this-dir>/opt3_fuse_rope_into_rms.patch   # after opt1: shares rms_gpu_bfyx_opt.cl
git apply <this-dir>/opt4_pa_kv_cache_vector_update.patch
git apply <this-dir>/opt5_sdpa_micro_xe3_igpu_tile.patch
git apply <this-dir>/opt6_fuse_uncompressed_qkv_fcs.patch
cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
```

Load the built plugin from `<openvino-root>/bin/intel64/Release` with
`OV_PLUGIN_PATH`; do not copy it into the shared `install/` tree.

opt-2 and opt-4..6 are independent; opt-3 touches the same kernel file as opt-1. Only opt-3
has a runtime switch: `OV_DISABLE_ROPE_RMS_FUSION=1` turns the fusion off
without rebuilding (used for A/B).

Together they change 18 files (including the new unit tests of opt-6), +635/−13 lines.

## What each patch does

### opt-1 — restore vectorized access in the RMS kernel

Two independent guard-expression defects in `rms_gpu_bfyx_opt.cl` forced scalar
loads/stores where sub-group block transfers were intended:

1. `USE_BLOCK_WRITE` was written `(A) & 0xF == 0`. `==` binds tighter than `&` in
   C, so it evaluated to `A & 0` — **always false**, making the block-write path
   dead code for every tensor shape in the plugin.
2. The block-read path required `workers_per_data > SUB_GROUP_SIZE`. Qwen3
   normalizes per attention head (`head_dim = 128`), which yields
   `workers_per_data == 16 == SUB_GROUP_SIZE` exactly, so `q_norm`/`k_norm`
   always took the scalar fallback. `>=` is correct.

### opt-2 — reuse one DynamicQuantize across FCs sharing an activation

`gate_proj` and `up_proj` are two FCs reading the *same*
`post_attention_layernorm` output, but each got its own activation
quantization — 28 redundant full read+write passes. Reuse one when the
quantization config matches field-for-field. DynamicQuantize node count on the
PA graph: 141 → 113. FP32 is unaffected, as it must be.

### opt-3 — fold the RoPE into the RMS epilogue

The largest of the three. Qwen3 normalizes per head and then immediately rotates
the same `[tokens, heads, 128]` tensor, so the intermediate tensor exists only to
hand the value between two kernels that both sit at the DRAM ceiling.

It folds cleanly because at `head_dim = 128` the RMS kernel uses one sub-group
per row with `SUBGROUP_BLOCK_SIZE == items_num == 8`, so `BLOCK_READ8` leaves
`data[j]` holding element `j*16 + lane` and RotateHalf's partner element `j+4`
lands **in the same lane** — no cross-lane shuffle, no extra barrier. The 56
`rope_opt` dispatches disappear entirely.

### opt-4 — vector stores for the paged KV-cache update

For a full 16-token block of an uncompressed cache the reference kernel wrote the transposed key cache with one
2-byte scalar store per lane per token (20 GB/s). The patch transposes the block in registers and writes every key
channel's 16 tokens as one 32-byte store, copies value rows with sub-group block writes, and splits key and value over
two sub-groups. `pa_kv_cache_update_ref` 7.30 -> 5.50 ms on Qwen3-0.6B FP16. Compressed (INT8/INT4) caches, partial
blocks and decode keep the old code.

### opt-5 — `sdpa_micro` tile for integrated Xe3

The PA-prefill tile for head size 128 comes from Ponte Vecchio (128 queries x 256 keys, 32 sub-groups per work group) and
leaves a 4-Xe-core GPU with ~128 coarse work groups. 32 queries x 128 keys on 8 sub-groups (4x more work groups) cuts
`sdpa_micro` 17.2 -> 12.8 ms. Applied only for xe3 + integrated + paged-attention prefill + head size 128. **Not
bit-exact** (128-key tile = different online-softmax accumulation order); a bit-exact 256-key variant keeps ~60% of the
gain, see the patch README.

### opt-6 — fuse Q/K/V of uncompressed models

New `FullyConnectedUncompressedHorizontalFusion`: three bias-free, constant-weight `FullyConnected` nodes on one
activation become one FC over concatenated weights plus an in-place-crop `VariadicSplit`, the same rewrite the existing
pass does for compressed FCs. FP16 -1.8 ms on 0.6B; byte-identical output.

## Correctness

| Patch | Status | Evidence |
|---|---|---|
| opt-1 | Bit-exact | Same elements accumulated in the same per-lane order; only the fetch instruction changes. 39/39 RMS GPU unit tests pass. |
| opt-2 | Bit-exact | Graph-level dedup of an identical computation; the decode-time f16 passthrough is preserved. |
| opt-3 | Bit-exact | 4 prompts x 128 greedy tokens reproduce the **unpatched plugin's output md5** on Qwen3-0.6B INT4-g128 (`0f7fb0eaec35ce0f5a094e93cdaaad73`) and Qwen3-1.7B INT4 channel-wise (`22e5ea17ec95a94b11975930c397d189`). 39/39 RMS and 12/12 RoPE GPU unit tests pass. |
| opt-4 | Bit-exact | `scripts/md5_check.py` (6 prompts x 96 greedy tokens) byte-identical to opt-1+2+3 on 0.6B FP16/INT8/INT4-g128/INT4-cw and 1.7B INT8/INT4-g128; 225 paged-attention/sdpa/kv-reorder unit tests: 221 pass, 4 skipped, 0 fail. |
| opt-5 | **Not bit-exact** | Identical on 0.6B FP16; greedy near-tie flips on the long prompts of INT8 (2/6) and INT4 (3/6) models. The 256-key variant is byte-identical on 0.6B INT8/INT4/INT4-cw and 1.7B INT8. Same unit tests pass. |
| opt-6 | Bit-exact | Byte-identical to opt-1..5 on 0.6B FP16/FP32/INT8/INT4-g128/INT4-cw and 1.7B FP16/INT8/INT4-g128; 5 new unit tests, 16/16 horizontal-fusion tests pass. |

## Results

Cold-start prefill TTFT, 1000 input tokens, GPU, batch 1, greedy, `-ic 1`,
prompt permutation ON, arithmetic mean of 10 measured iterations after one
warm-up. Each patch was measured against the previous stage with the
baseline-first harness (measure, apply, rebuild, measure).

| Model | Variant | opt-1 | opt-2 | opt-3 |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | −2.80% | +0.26% | −5.73% |
| Qwen3-0.6B | INT8 | −3.96% | −0.93% | −7.49% |
| Qwen3-0.6B | **INT4 g128** | **−2.91%** | **−1.46%** | **−6.02%** |
| Qwen3-0.6B | INT4 channel-wise | −4.04% | −1.93% | −7.88% |
| Qwen3-0.6B | FP16 | +1.49% | −2.54% | −10.72% |
| Qwen3-1.7B | INT8 | noisy | −2.22% | −2.88% |
| Qwen3-1.7B | FP16 | +1.01% | −0.59% | −6.47% |
| Qwen3-1.7B | **INT4 g128** | **−1.37%** | **−0.86%** | **−2.98%** |
| Qwen3-1.7B | INT4 channel-wise | −2.49% | −2.64% | −4.01% |

Cumulative against pristine 2026.3.1 in the historical formal all-precision stack run
[`opt123_all_precisions_fp16_20260915`](../results/opt123_all_precisions_fp16_20260915/),
in milliseconds:

| Model | Variant | pristine | opt-1+2+3 | total |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 146.32 | 134.80 | −7.9% |
| Qwen3-0.6B | INT8 | 129.00 | 115.28 | −10.6% |
| Qwen3-0.6B | **INT4 g128** | **139.96** | **127.09** | **−9.2%** |
| Qwen3-0.6B | INT4 channel-wise | 114.35 | 100.78 | −11.9% |
| Qwen3-0.6B | FP16 | 146.30 | 135.29 | −7.5% |
| Qwen3-1.7B | FP32* | 572.31 | 384.86 | −32.8% |
| Qwen3-1.7B | FP16* | 361.66 | 427.14 | +18.1% |
| Qwen3-1.7B | INT8 | 272.08 | 256.26 | −5.8% |
| Qwen3-1.7B | **INT4 g128** | **277.17** | **259.71** | **−6.3%** |
| Qwen3-1.7B | INT4 channel-wise | 219.19 | 202.84 | −7.5% |

`*` Qwen3-1.7B FP32 and FP16 are excluded from patch-effect claims because
their sequential arms are noisy. The other rows are the complete three-patch
result for every stable precision.

The formal DQ=0 run is
[`opt123_dq0_0p6b_fp16_20260915`](../results/opt123_dq0_0p6b_fp16_20260915/). It applies
`DYNAMIC_QUANTIZATION_GROUP_SIZE=0` to all five 0.6B precisions:

| Model | Variant | DQ=0 pristine | DQ=0 + opt-1+2+3 | Patch delta | DQ=0 vs default after patches |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 149.23 | 138.58 | −7.13% | +2.80% |
| Qwen3-0.6B | INT8 | 134.74 | 124.37 | −7.69% | +7.89% |
| Qwen3-0.6B | **INT4 g128** | **127.97** | **116.24** | **−9.17%** | **−8.54%** |
| Qwen3-0.6B | INT4 channel-wise | 125.40 | 113.80 | −9.24% | +12.92% |
| Qwen3-0.6B | FP16 | 149.73 | 141.88 | −5.24% | +4.87% |

Only INT4 g128 benefits from DQ=0. Its final 116.24 ms is **−16.95%** versus
the default pristine 139.96 ms. See
[`../docs/optimizations.md`](../docs/optimizations.md) §7 for why it is a
per-deployment setting rather than a patch.

## Reproducing

```bash
cd /home/chaofan/model_benchmark
source Qwen3-Optimization/scripts/env.sh

# measure pristine versus the complete default patch stack, with provenance
python3 Qwen3-Optimization/scripts/run_matrix.py --patch-compare \
   --run-id my_run --iterations 10

# per-kernel attribution on the real PagedAttention pipeline
./Qwen3-Optimization/scripts/vtune_gpu.sh \
    Qwen3-Optimization/models/converted/Qwen3-0.6B-INT4-g128 my_tag overview 5 1000
```

Two measurement rules that this workspace learned the hard way, both documented
in [`../docs/optimizations.md`](../docs/optimizations.md):

- Never measure TTFT with `--disable-prompt-permutation`: PA's prefix cache turns
  every repeat into a cache lookup and understates TTFT by 3–5x.
- Judge A/B by the **arithmetic mean**, not p50. One rejected change read as
  neutral by p50 and only showed up as a consistent regression in the mean.
