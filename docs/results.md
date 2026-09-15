# Qwen3 Benchmark Results

This is the canonical Qwen3 performance report. All headline latency and
throughput values use the arithmetic mean over measured iterations. Prompt
permutation is always ON, so PA prefix caching cannot turn a measured prefill
into a cache lookup.

## Protocol

- GPU, batch size 1, greedy decoding, 1000 input tokens.
- One warm-up iteration and 10 measured iterations.
- OpenVINO and OpenVINO GenAI 2026.3.1.
- `--prefill-only` uses `-ic 1`, so TTFT is cold prefill time.
- `--patch-compare` measures the pristine baseline first, then applies the
  default opt-1 -> opt-2 -> opt-3 patch stack, rebuilds, and measures the
  optimized arm.
- `--runtime-config` supplies extra C++ plugin properties; the DQ=0 run uses
  `configs/qwen3_int4_g128_dq0.json`.
- The matrix includes FP32, FP16, INT8, INT4-g128, and INT4 channel-wise for
  both Qwen3-0.6B and Qwen3-1.7B.
- The runner records the source commit, patch state, source diff hash, plugin
  binary hash, exact commands, raw reports, and arithmetic mean statistics.

## Opt-1 Cold Prefill

Run: [`cold_ttft_patch_compare_20260913`](../results/cold_ttft_patch_compare_20260913/)

The baseline arm was measured before applying `opt1_rms_block_access.patch`.
The optimized arm was measured after applying the patch and rebuilding
`openvino_intel_gpu_plugin`. Values below are TTFT arithmetic means in
milliseconds:

| Model | Variant | Baseline mean | Opt-1 mean | Delta | Baseline std | Opt-1 std |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 148.63 | 144.47 | -2.80% | 1.01 | 0.81 |
| Qwen3-0.6B | INT8 | 132.35 | 127.11 | -3.96% | 0.99 | 0.12 |
| Qwen3-0.6B | **INT4 g128** | **141.70** | **137.57** | **-2.91%** | 0.79 | 0.50 |
| Qwen3-0.6B | INT4 channel-wise | 116.93 | 112.21 | -4.04% | 0.77 | 0.35 |
| Qwen3-0.6B | FP16 | 146.82 | 149.01 | +1.49% | 0.53 | 1.80 |
| Qwen3-1.7B | FP32* | 479.72 | 404.45 | -15.69% | 1.65 | 20.00 |
| Qwen3-1.7B | FP16* | 365.20 | 368.89 | +1.01% | 1.27 | 3.61 |
| Qwen3-1.7B | INT8* | 276.76 | 279.22 | +0.89% | 2.40 | 7.29 |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **275.51** | **-1.37%** | 1.32 | 0.73 |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 217.87 | -2.49% | 1.75 | 1.50 |

`*` The 1.7B FP32 and INT8 optimized arms are noisy in this sequential
baseline-first run and are not used as patch-effect claims. The stable rows
show the measured opt-1 improvement.

Machine-readable comparison: [comparison.csv](../results/cold_ttft_patch_compare_20260913/comparison.csv),
[comparison.json](../results/cold_ttft_patch_compare_20260913/comparison.json),
and [metadata.json](../results/cold_ttft_patch_compare_20260913/metadata.json).

The FP16 rows come from the two-model supplement
[`opt1_fp16_20260915`](../results/opt1_fp16_20260915/); the 1.7B FP16 row is
365.20 -> 368.89 ms (+1.01%). Its optimized-arm std is 3.61 ms, so it is
recorded but not treated as a stable patch-effect claim.

## Opt-2 Cold Prefill

Run: [`opt2_dq_reuse_20260913`](../results/opt2_dq_reuse_20260913/)

Baseline arm is opt-1 applied; optimized arm adds
`opt2_dynamic_quantize_reuse.patch`. TTFT arithmetic means in milliseconds:

| Model | Variant | Baseline mean | Opt-2 mean | Delta | Baseline std | Opt-2 std |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 143.40 | 143.78 | +0.26% | 0.36 | 0.36 |
| Qwen3-0.6B | INT8 | 125.97 | 124.81 | -0.93% | 0.56 | 0.21 |
| Qwen3-0.6B | **INT4 g128** | **138.26** | **136.24** | **-1.46%** | 0.51 | 0.43 |
| Qwen3-0.6B | INT4 channel-wise | 112.11 | 109.95 | -1.93% | 0.29 | 0.16 |
| Qwen3-0.6B | FP16 | 153.87 | 149.97 | -2.54% | 7.63 | 2.30 |
| Qwen3-1.7B | FP32* | 553.85 | 359.78 | -35.04% | 14.37 | 0.84 |
| Qwen3-1.7B | FP16* | 375.62 | 373.39 | -0.59% | 9.32 | 10.92 |
| Qwen3-1.7B | INT8 | 270.66 | 264.66 | -2.22% | 0.84 | 0.75 |
| Qwen3-1.7B | **INT4 g128** | **272.95** | **270.59** | **-0.86%** | 0.74 | 1.08 |
| Qwen3-1.7B | INT4 channel-wise | 217.47 | 211.74 | -2.64% | 0.69 | 0.47 |

FP32 is unaffected, as it must be: an FP32 graph contains no `DynamicQuantize`.

The FP16 rows come from [`opt2_fp16_20260915`](../results/opt2_fp16_20260915/)
and are noisy, especially for 1.7B (375.62 -> 373.39 ms, std 9.32/10.92 ms).

## Opt-3 Cold Prefill

Run: [`opt3_rope_rms_20260915`](../results/opt3_rope_rms_20260915/)

Baseline arm is opt-1 + opt-2 applied; optimized arm adds
`opt3_fuse_rope_into_rms.patch`. TTFT arithmetic means in milliseconds:

| Model | Variant | Baseline mean | Opt-3 mean | Delta | Baseline std | Opt-3 std |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 143.10 | 134.91 | -5.73% | 0.39 | 0.29 |
| Qwen3-0.6B | INT8 | 124.25 | 114.95 | -7.49% | 0.21 | 0.26 |
| Qwen3-0.6B | **INT4 g128** | **136.22** | **128.01** | **-6.02%** | 0.34 | 0.55 |
| Qwen3-0.6B | INT4 channel-wise | 109.78 | 101.14 | -7.88% | 0.32 | 0.70 |
| Qwen3-0.6B | FP16 | 152.02 | 135.73 | -10.72% | 7.80 | 0.38 |
| Qwen3-1.7B | FP32* | 432.40 | 381.96 | -11.66% | 2.05 | 25.01 |
| Qwen3-1.7B | FP16* | 377.76 | 353.33 | -6.47% | 10.54 | 1.00 |
| Qwen3-1.7B | INT8 | 263.57 | 255.98 | -2.88% | 0.45 | 0.51 |
| Qwen3-1.7B | **INT4 g128** | **268.09** | **260.12** | **-2.98%** | 0.25 | 0.72 |
| Qwen3-1.7B | INT4 channel-wise | 210.45 | 202.02 | -4.01% | 0.18 | 0.40 |

Opt-3 is bit-exact: 4 prompts x 128 greedy tokens reproduce the unpatched
plugin's output md5 on both Qwen3-0.6B INT4-g128 and Qwen3-1.7B channel-wise.

`*` Qwen3-1.7B FP32 and FP16 are excluded from stable patch-effect claims in
these sequential stage runs because at least one arm is noisy.

The FP16 rows come from [`opt3_fp16_20260915`](../results/opt3_fp16_20260915/);
the 1.7B FP16 result is 377.76 -> 353.33 ms (-6.47%) with a noisy baseline
std of 10.54 ms.

## Opt-1+2+3 Full Stack

Run: [`opt123_all_precisions_fp16_20260915`](../results/opt123_all_precisions_fp16_20260915/)

This is the formal all-precision comparison requested for the complete patch
stack. The baseline arm has all three patches unapplied; the optimized arm has
all three patches applied. Both arms were rebuilt and measured by the unified
runner. TTFT arithmetic means in milliseconds:

| Model | Variant | Pristine mean | Opt-1+2+3 mean | Delta | Pristine std | Opt-1+2+3 std |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 146.32 | 134.80 | -7.87% | 0.15 | 0.29 |
| Qwen3-0.6B | INT8 | 129.00 | 115.28 | -10.63% | 0.08 | 0.15 |
| Qwen3-0.6B | **INT4 g128** | **139.96** | **127.09** | **-9.20%** | 0.33 | 0.26 |
| Qwen3-0.6B | INT4 channel-wise | 114.35 | 100.78 | -11.87% | 0.24 | 0.28 |
| Qwen3-0.6B | FP16 | 146.30 | 135.29 | -7.53% | 0.38 | 0.28 |
| Qwen3-1.7B | FP32* | 572.31 | 384.86 | -32.75% | 26.59 | 32.84 |
| Qwen3-1.7B | INT8 | 272.08 | 256.26 | -5.81% | 0.20 | 0.30 |
| Qwen3-1.7B | **INT4 g128** | **277.17** | **259.71** | **-6.30%** | 0.54 | 0.59 |
| Qwen3-1.7B | INT4 channel-wise | 219.19 | 202.84 | -7.46% | 0.34 | 0.35 |
| Qwen3-1.7B | FP16* | 361.66 | 427.14 | +18.10% | 0.76 | 13.47 |

`*` The 1.7B FP32 baseline and optimized arms are both noisy and are excluded
from the patch-effect claim. The 1.7B FP16 optimized arm is also noisy and
regresses in this sequential run, so it is recorded but excluded from the
patch-effect claim. The other 1.7B rows and all 0.6B rows are the complete
three-patch performance result for every precision.

Machine-readable comparison: [comparison.csv](../results/opt123_all_precisions_fp16_20260915/comparison.csv),
[comparison.json](../results/opt123_all_precisions_fp16_20260915/comparison.json),
and [metadata.json](../results/opt123_all_precisions_fp16_20260915/metadata.json).

## Cumulative

The cumulative numbers below come from the formal full-stack run above, not by
adding percentages from separate experiments:

| Model | Variant | Pristine mean | Opt-1+2+3 mean | Total |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 146.32 | 134.80 | -7.9% |
| Qwen3-0.6B | INT8 | 129.00 | 115.28 | -10.6% |
| Qwen3-0.6B | **INT4 g128** | **139.96** | **127.09** | **-9.2%** |
| Qwen3-0.6B | INT4 channel-wise | 114.35 | 100.78 | -11.9% |
| Qwen3-0.6B | FP16 | 146.30 | 135.29 | -7.5% |
| Qwen3-1.7B | FP32* | 572.31 | 384.86 | -32.8% |
| Qwen3-1.7B | INT8 | 272.08 | 256.26 | -5.8% |
| Qwen3-1.7B | **INT4 g128** | **277.17** | **259.71** | **-6.3%** |
| Qwen3-1.7B | INT4 channel-wise | 219.19 | 202.84 | -7.5% |
| Qwen3-1.7B | FP16* | 361.66 | 427.14 | +18.1% |

`*` 1.7B FP32 and FP16 are excluded from optimization claims because their
sequential arms are noisy.

## 0.6B DQ=0

Run: [`opt123_dq0_0p6b_fp16_20260915`](../results/opt123_dq0_0p6b_fp16_20260915/)

This run applies `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` to all five 0.6B
precisions and compares pristine versus the complete three-patch stack. The
setting is useful only for INT4 g128; the other precision formats become slower
relative to their default runtime configuration.

| Model | Variant | DQ=0 pristine mean | DQ=0 + Opt-1+2+3 mean | Patch delta | DQ=0 vs default after patches |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 149.23 | 138.58 | -7.13% | +2.80% |
| Qwen3-0.6B | INT8 | 134.74 | 124.37 | -7.69% | +7.89% |
| Qwen3-0.6B | **INT4 g128** | **127.97** | **116.24** | **-9.17%** | **-8.54%** |
| Qwen3-0.6B | INT4 channel-wise | 125.40 | 113.80 | -9.24% | +12.92% |
| Qwen3-0.6B | FP16 | 149.73 | 141.88 | -5.24% | +4.87% |

The `DQ=0 vs default after patches` column compares against the corresponding
Opt-1+2+3 means in the full-stack run. For INT4 g128, the final 116.24 ms is
**-16.95%** versus the default pristine 139.96 ms. This is a runtime setting,
not another source patch, and should not be generalized to the other precisions.

Machine-readable comparison: [comparison.csv](../results/opt123_dq0_0p6b_fp16_20260915/comparison.csv),
[comparison.json](../results/opt123_dq0_0p6b_fp16_20260915/comparison.json),
and [metadata.json](../results/opt123_dq0_0p6b_fp16_20260915/metadata.json).

## Commands

All modes use the same runner:

```bash
cd /home/chaofan/model_benchmark/Qwen3-Optimization

# Full 1000 -> 512 generation matrix, Python and C++.
python scripts/run_matrix.py --run-id qwen3_full_YYYYMMDD --iterations 10
python scripts/summarize_results.py results/qwen3_full_YYYYMMDD

# Cold prefill only.
python scripts/run_matrix.py --prefill-only --run-id qwen3_ttft_YYYYMMDD --iterations 10

# Source patch comparison: baseline first, optimized second.
python scripts/run_matrix.py --patch-compare --run-id qwen3_patch_YYYYMMDD --iterations 10

# 0.6B DQ=0 plus the same three-patch comparison.
python scripts/run_matrix.py --patch-compare \
  --runtime-config configs/qwen3_int4_g128_dq0.json \
  --models qwen3_0p6b_fp32 qwen3_0p6b_fp16 qwen3_0p6b_int8 qwen3_0p6b_int4_g128 qwen3_0p6b_int4_channelwise \
  --run-id qwen3_dq0_0p6b_fp16_YYYYMMDD --iterations 10
```

There is no prefix-cache benchmark mode in the canonical runner. Do not add
`--disable-prompt-permutation` to a performance command.

Raw per-iteration reports retain the runner's diagnostic fields, but all
headline tables and optimization decisions in this repository use arithmetic
means.
