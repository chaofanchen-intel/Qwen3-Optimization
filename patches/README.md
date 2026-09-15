# Qwen3 GPU-plugin patches — archive and index

Intel GPU plugin patches that reduce **cold-start prefill / TTFT** for dense
Qwen3-0.6B and Qwen3-1.7B, with INT4 `group_size=128` as the primary target and
INT4 per-output-channel as the secondary one.

This directory is an **archive** kept next to the measurements that justify it,
so `Qwen3-Optimization/` is self-contained. The patches are developed and
delivered from
[`../../benchmark/openvino/qwen3_patch/`](../../benchmark/openvino/qwen3_patch/) —
that is the canonical copy, the one Intel receives, and the one
`scripts/run_matrix.py --patch-compare` applies.

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
| `engineering_notes.md` | Snapshot of the canonical patch README: full rationale, per-kernel VTune numbers, correctness argument, negative results, open items |
| `MANIFEST.sha256` | Hashes of everything above |
| `sync_from_canonical.sh` | Drift guard / resync |

Related documents in this workspace:

| Document | Contents |
|---|---|
| [`../docs/optimizations.md`](../docs/optimizations.md) | Experiment log: how the hotspots were found, what was tried, what was rejected and why |
| [`../docs/results.md`](../docs/results.md) | Canonical TTFT report, per-patch and cumulative, arithmetic means |
| [`../results/`](../results/) | Raw per-run data with provenance (`metadata.json` records source commit, patch sha256, patch state, plugin binary hash) |

## Base and apply order

Generated against this repository's OpenVINO checkout at release tag `2026.3.1`
(`759c5a6ab8c066af5f4bc5ebd04643706012a37d`). Apply in order to a pristine tree:

```bash
cd <openvino-root>
git apply <this-dir>/opt1_rms_block_access.patch
git apply <this-dir>/opt2_dynamic_quantize_reuse.patch
git apply <this-dir>/opt3_fuse_rope_into_rms.patch   # after opt1: shares rms_gpu_bfyx_opt.cl
cmake --build <build-dir> --target openvino_intel_gpu_plugin -j
cp <build-root>/bin/intel64/Release/libopenvino_intel_gpu_plugin.so \
   <install-root>/runtime/lib/intel64/
```

opt-2 is independent; opt-3 touches the same kernel file as opt-1. Only opt-3
has a runtime switch: `OV_DISABLE_ROPE_RMS_FUSION=1` turns the fusion off
without rebuilding (used for A/B).

Together they change 11 files, +364/−13 lines.

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

## Correctness

| Patch | Status | Evidence |
|---|---|---|
| opt-1 | Bit-exact | Same elements accumulated in the same per-lane order; only the fetch instruction changes. 39/39 RMS GPU unit tests pass. |
| opt-2 | Bit-exact | Graph-level dedup of an identical computation; the decode-time f16 passthrough is preserved. |
| opt-3 | Bit-exact | 4 prompts x 128 greedy tokens reproduce the **unpatched plugin's output md5** on Qwen3-0.6B INT4-g128 (`0f7fb0eaec35ce0f5a094e93cdaaad73`) and Qwen3-1.7B INT4 channel-wise (`22e5ea17ec95a94b11975930c397d189`). 39/39 RMS and 12/12 RoPE GPU unit tests pass. |

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

Cumulative against pristine 2026.3.1 from the formal all-precision stack run
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
