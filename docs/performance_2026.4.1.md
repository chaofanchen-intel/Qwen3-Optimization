# Qwen3 performance rebenchmark on OpenVINO 2026.4.1

## Full-generation patch sweep

This is the current performance report for the rebased patches. It supersedes
the 2026.3.1 measurements for current patch comparisons; those results remain
historical in [`results.md`](results.md).

### Runtime and protocol

- OpenVINO checkout: branch `2026.4_base`, commit
  `f525bcc4092e5b241d6298f4d4bc9d4ccfd7febf`.
- Runtime: OpenVINO `2026.4.1-22989-f525bcc4092-2026.4_base`;
  OpenVINO GenAI `2026.4.1.0-3409-eadf6ddbbe6-2026.4_base`.
- Python packages: Optimum Intel 2.2.0, NNCF 3.4.0, Transformers 5.2.0,
  Torch 2.13.0. The C++ `model_bench` binary was rebuilt and links the
  OpenVINO 2026.4.1 / GenAI 2026.4.1.0 (`.2641`) libraries.
- Device: Intel Graphics iGPU, OpenVINO `GPU` device; prompt permutation ON;
  batch size 1; resource monitoring OFF.
- Prompt: `benchmark/tests/prompts/qwen3_1k.jsonl`, 1000 input tokens.
  Requested output is 512 tokens; all runs report 512 generated tokens.
- One warm-up and 10 measured iterations per model/backend; headline figures
  are arithmetic means. Standard deviations are population standard
  deviations across the measured iterations.
- Both Python OpenVINO GenAI and C++ `model_bench` were used. FP32 was excluded
  from this rebenchmark as requested.
- The source was compiled at each cumulative state: pristine, opt-1,
  opt-1+opt-2, and opt-1+opt-2+opt-3. opt-3 follows opt-1 because both modify
  the RMS kernel.

The source-built GPU plugins were staged for later testing under
[`../results/runs/patch_sweep_full_2026.4.1_20261022/plugins/`](../results/runs/patch_sweep_full_2026.4.1_20261022/plugins/);
the build did not run `cmake --install` or overwrite `install/`.

### Qwen3-0.6B TTFT (primary)

TTFT is first-token latency from the 1000-to-512-token generation runs.
Values are mean +/- standard deviation in milliseconds.

| Variant | Backend | Pristine | opt-1 | opt-1+opt-2 | opt-1+opt-2+opt-3 | Full vs. pristine | opt-1 / opt-2 / opt-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | Python | 149.94 +/- 2.44 | 148.05 +/- 2.14 | 147.28 +/- 0.62 | 138.11 +/- 2.23 | -7.89% | -1.26% / -0.52% / -6.22% |
| FP16 | C++ | 149.82 +/- 0.61 | 146.67 +/- 0.84 | 147.55 +/- 0.95 | 137.24 +/- 1.39 | -8.40% | -2.10% / +0.60% / -6.99% |
| INT8 | Python | 127.21 +/- 0.60 | 123.88 +/- 0.54 | 123.62 +/- 0.62 | 114.93 +/- 0.59 | -9.65% | -2.62% / -0.21% / -7.03% |
| INT8 | C++ | 126.70 +/- 0.45 | 123.51 +/- 0.61 | 124.01 +/- 0.53 | 114.86 +/- 0.23 | -9.35% | -2.51% / +0.40% / -7.38% |
| INT4 g128 | Python | 138.29 +/- 0.21 | 135.63 +/- 0.34 | 135.99 +/- 0.30 | 127.37 +/- 0.35 | -7.89% | -1.92% / +0.26% / -6.33% |
| INT4 g128 | C++ | 138.53 +/- 0.39 | 135.61 +/- 0.29 | 135.27 +/- 0.23 | 126.82 +/- 0.38 | -8.46% | -2.11% / -0.25% / -6.25% |
| INT4 channel-wise | Python | 111.33 +/- 0.39 | 108.98 +/- 0.44 | 108.83 +/- 0.39 | 99.92 +/- 0.54 | -10.25% | -2.10% / -0.14% / -8.19% |
| INT4 channel-wise | C++ | 111.17 +/- 0.37 | 109.25 +/- 0.46 | 108.97 +/- 0.54 | 100.27 +/- 0.38 | -9.80% | -1.72% / -0.26% / -7.99% |

The complete stack reduces Qwen3-0.6B TTFT by 7.9-10.3% across the four
precisions and both drivers. opt-1 contributes about 1.7-2.6%; opt-2 is small
and often within measurement noise; opt-3 is the largest incremental change at
about 6.2-8.2%.

### Qwen3-1.7B TTFT

| Variant | Backend | Pristine | opt-1 | opt-1+opt-2 | opt-1+opt-2+opt-3 | Full vs. pristine | opt-1 / opt-2 / opt-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| FP16 | Python | 365.22 +/- 2.59 | 371.29 +/- 15.23 | 362.84 +/- 0.55 | 375.41 +/- 3.41 | +2.79% | +1.66% / -2.28% / +3.47% |
| FP16 | C++ | 365.96 +/- 4.17 | 382.43 +/- 28.82 | 360.72 +/- 1.68 | 354.02 +/- 3.48 | -3.26% | +4.50% / -5.68% / -1.86% |
| INT8 | Python | 268.42 +/- 0.74 | 265.60 +/- 0.94 | 265.46 +/- 1.03 | 256.38 +/- 1.34 | -4.49% | -1.05% / -0.05% / -3.42% |
| INT8 | C++ | 268.83 +/- 0.66 | 265.89 +/- 0.91 | 265.78 +/- 0.82 | 256.13 +/- 0.72 | -4.72% | -1.09% / -0.04% / -3.63% |
| INT4 g128 | Python | 273.92 +/- 1.73 | 271.00 +/- 1.02 | 270.59 +/- 1.66 | 261.04 +/- 0.66 | -4.70% | -1.07% / -0.15% / -3.53% |
| INT4 g128 | C++ | 274.32 +/- 1.73 | 271.31 +/- 0.95 | 271.10 +/- 0.80 | 261.51 +/- 0.74 | -4.67% | -1.10% / -0.08% / -3.54% |
| INT4 channel-wise | Python | 216.96 +/- 0.92 | 214.21 +/- 0.74 | 213.40 +/- 0.92 | 204.01 +/- 0.67 | -5.97% | -1.27% / -0.38% / -4.40% |
| INT4 channel-wise | C++ | 217.54 +/- 0.58 | 214.41 +/- 1.12 | 213.02 +/- 0.83 | 203.74 +/- 0.28 | -6.34% | -1.44% / -0.65% / -4.36% |

For 1.7B INT8 and both INT4 variants, the complete stack improves TTFT by
4.5-6.3% with consistent direction across drivers. The FP16 runs are noisy:
opt-1 has high variability and the final Python/C++ deltas disagree, so no
stable FP16 patch-effect claim is made for 1.7B.

### Full-generation TPOT, throughput, and E2E

Baseline-to-full-stack arithmetic means; E2E is seconds for the requested
512-token generation. Full per-stage means and standard deviations for these
metrics are in [`stage_metrics.csv`](../results/runs/patch_sweep_full_2026.4.1_20261022/stage_metrics.csv);
per-patch changes are in [`stage_deltas.csv`](../results/runs/patch_sweep_full_2026.4.1_20261022/stage_deltas.csv).

| Model | Backend | TPOT ms (pristine -> full) | Total TPS (pristine -> full) | E2E s (pristine -> full) |
|---|---|---:|---:|---:|
| Qwen3-0.6B FP16 | Python | 33.32 -> 33.11 (-0.62%) | 29.81 -> 30.01 (+0.68%) | 17.18 -> 17.06 (-0.68%) |
| Qwen3-0.6B FP16 | C++ | 33.32 -> 33.05 (-0.82%) | 29.81 -> 30.08 (+0.89%) | 17.18 -> 17.02 (-0.88%) |
| Qwen3-0.6B INT8 | Python | 18.18 -> 17.82 (-1.94%) | 54.38 -> 55.51 (+2.08%) | 9.42 -> 9.22 (-2.04%) |
| Qwen3-0.6B INT8 | C++ | 18.16 -> 17.93 (-1.25%) | 54.43 -> 55.18 (+1.37%) | 9.41 -> 9.28 (-1.35%) |
| Qwen3-0.6B INT4 g128 | Python | 13.11 -> 12.95 (-1.23%) | 74.89 -> 75.93 (+1.38%) | 6.84 -> 6.74 (-1.36%) |
| Qwen3-0.6B INT4 g128 | C++ | 13.25 -> 13.04 (-1.55%) | 74.12 -> 75.39 (+1.72%) | 6.91 -> 6.79 (-1.69%) |
| Qwen3-0.6B INT4 channel-wise | Python | 12.78 -> 12.63 (-1.16%) | 77.10 -> 78.12 (+1.33%) | 6.64 -> 6.55 (-1.31%) |
| Qwen3-0.6B INT4 channel-wise | C++ | 12.83 -> 12.63 (-1.54%) | 76.77 -> 78.08 (+1.71%) | 6.67 -> 6.56 (-1.68%) |
| Qwen3-1.7B FP16 | Python | 86.33 -> 87.44 (+1.29%) | 11.51 -> 11.36 (-1.33%) | 44.48 -> 45.08 (+1.35%) |
| Qwen3-1.7B FP16 | C++ | 86.11 -> 85.73 (-0.44%) | 11.54 -> 11.59 (+0.47%) | 44.37 -> 44.16 (-0.47%) |
| Qwen3-1.7B INT8 | Python | 44.52 -> 44.26 (-0.57%) | 22.24 -> 22.38 (+0.62%) | 23.02 -> 22.88 (-0.61%) |
| Qwen3-1.7B INT8 | C++ | 44.83 -> 44.49 (-0.75%) | 22.09 -> 22.27 (+0.80%) | 23.18 -> 22.99 (-0.80%) |
| Qwen3-1.7B INT4 g128 | Python | 28.99 -> 28.78 (-0.73%) | 33.93 -> 34.21 (+0.81%) | 15.09 -> 14.97 (-0.80%) |
| Qwen3-1.7B INT4 g128 | C++ | 29.18 -> 28.87 (-1.09%) | 33.71 -> 34.10 (+1.17%) | 15.19 -> 15.01 (-1.15%) |
| Qwen3-1.7B INT4 channel-wise | Python | 27.83 -> 27.61 (-0.78%) | 35.46 -> 35.77 (+0.86%) | 14.44 -> 14.31 (-0.86%) |
| Qwen3-1.7B INT4 channel-wise | C++ | 27.97 -> 27.78 (-0.67%) | 35.29 -> 35.56 (+0.76%) | 14.51 -> 14.40 (-0.76%) |

Decode and end-to-end gains are smaller than the TTFT improvements, as expected
for patches that primarily reduce prefill work. The 1.7B FP16 full-generation
results also show the same driver-to-driver variability seen in its TTFT row.

### Compiled plugin artifacts and raw data

| Stage | Applied patch prefix | Plugin SHA-256 |
|---|---:|---|
| pristine | 0 | `6d9803b3f711f494be1bd230ccf41ed0463cdb9f619c5a4ad7a49bbe1e888e3f` |
| opt-1 | 1 | `fd258a328d84e4c5eca0c24c88794630c30a0e9ccd0a322a7b3daa0fe795ec50` |
| opt-1 + opt-2 | 2 | `9df4cc44386be412223090cd8bc9ace7b8f582d3f0a74204e74069993a8dc5f4` |
| opt-1 + opt-2 + opt-3 | 3 | `6c16a6ea5e9f387a2b69001ba8044b8c826af5dbafa682eb3f126184a1b40afe` |

The [run directory](../results/runs/patch_sweep_full_2026.4.1_20261022/)
contains all 4 x 16 raw runs, per-stage summaries, build logs, staged `.so`
files, and [`manifest.json`](../results/runs/patch_sweep_full_2026.4.1_20261022/manifest.json)
with the source commit, patch hashes, runtime versions, protocol, and C++
linkage. After the sweep, the source patch state was restored to pristine and
the `bin/` plugin was rebuilt; its hash matches the staged pristine artifact.

This is a performance sweep, not a numerical-equivalence or broad model-quality
validation. Existing correctness evidence remains in the patch engineering
notes.

## Earlier focused TTFT sweep (2026-10-21)

This smaller, prior experiment used only Qwen3-1.7B INT4 channel-wise and four
generated tokens. It used OpenVINO/GenAI Python bindings 2026.4.0 / 2026.4.0.0
with the source-built OpenVINO 2026.4.1 GPU plugin; it did not use the C++
`model_bench` driver. It remains a useful supplemental check but is superseded
by the full-generation matrix above.

| Patch state | TTFT mean +/- std (ms) | Change vs. previous stage | Change vs. pristine |
|---|---:|---:|---:|
| Pristine 2026.4.1 | 216.53 +/- 3.62 | — | — |
| opt-1 | 211.74 +/- 0.26 | -2.21% | -2.21% |
| opt-1 + opt-2 | 210.49 +/- 0.68 | -0.59% | -2.79% |
| opt-1 + opt-2 + opt-3 | 201.53 +/- 0.19 | -4.26% | -6.92% |

Raw reports for this earlier sweep are under
[`results/runs/patch_sweep_2026.4.1_20261021/`](../results/runs/patch_sweep_2026.4.1_20261021/).
