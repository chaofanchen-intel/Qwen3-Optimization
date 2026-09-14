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
