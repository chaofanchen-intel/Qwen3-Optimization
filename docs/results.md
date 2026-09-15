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
- `--patch-compare` measures the unpatched baseline first, then applies the
  source patch, rebuilds, and measures the optimized arm.
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
| Qwen3-1.7B | FP32* | 479.72 | 404.45 | -15.69% | 1.65 | 20.00 |
| Qwen3-1.7B | INT8* | 276.76 | 279.22 | +0.89% | 2.40 | 7.29 |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **275.51** | **-1.37%** | 1.32 | 0.73 |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 217.87 | -2.49% | 1.75 | 1.50 |

`*` The 1.7B FP32 and INT8 optimized arms are noisy in this sequential
baseline-first run and are not used as patch-effect claims. The stable rows
show the measured opt-1 improvement.

Machine-readable comparison: [comparison.csv](../results/cold_ttft_patch_compare_20260913/comparison.csv),
[comparison.json](../results/cold_ttft_patch_compare_20260913/comparison.json),
and [metadata.json](../results/cold_ttft_patch_compare_20260913/metadata.json).

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
| Qwen3-1.7B | FP32* | 553.85 | 359.78 | -35.04% | 14.37 | 0.84 |
| Qwen3-1.7B | INT8 | 270.66 | 264.66 | -2.22% | 0.84 | 0.75 |
| Qwen3-1.7B | **INT4 g128** | **272.95** | **270.59** | **-0.86%** | 0.74 | 1.08 |
| Qwen3-1.7B | INT4 channel-wise | 217.47 | 211.74 | -2.64% | 0.69 | 0.47 |

FP32 is unaffected, as it must be: an FP32 graph contains no `DynamicQuantize`.

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
| Qwen3-1.7B | FP32* | 432.40 | 381.96 | -11.66% | 2.05 | 25.01 |
| Qwen3-1.7B | INT8 | 263.57 | 255.98 | -2.88% | 0.45 | 0.51 |
| Qwen3-1.7B | **INT4 g128** | **268.09** | **260.12** | **-2.98%** | 0.25 | 0.72 |
| Qwen3-1.7B | INT4 channel-wise | 210.45 | 202.02 | -4.01% | 0.18 | 0.40 |

Opt-3 is bit-exact: 4 prompts x 128 greedy tokens reproduce the unpatched
plugin's output md5 on both Qwen3-0.6B INT4-g128 and Qwen3-1.7B channel-wise.

`*` Qwen3-1.7B FP32 is excluded from every patch-effect claim: its run-to-run
std reaches 14-25 ms.

## Cumulative

Pristine 2026.3.1 versus all three patches, TTFT means in milliseconds:

| Model | Variant | Pristine | Opt-1+2+3 | Total |
|---|---|---:|---:|---:|
| Qwen3-0.6B | **INT4 g128** | **141.70** | **127.94** | **-9.7%** |
| Qwen3-0.6B | INT4 channel-wise | 116.93 | 101.09 | -13.5% |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **260.21** | **-6.8%** |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 202.02 | -9.6% |

Adding the `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` runtime setting (Qwen3-0.6B
INT4-g128 only, see
[`optimizations.md`](optimizations.md#7-runtime-setting-qwen3-06b-int4-g128-only))
takes that configuration to **116.90 ms**, i.e. **-17.5%** against pristine.

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
```

There is no prefix-cache benchmark mode in the canonical runner. Do not add
`--disable-prompt-permutation` to a performance command.

Raw per-iteration reports retain the runner's diagnostic fields, but all
headline tables and optimization decisions in this repository use arithmetic
means.
