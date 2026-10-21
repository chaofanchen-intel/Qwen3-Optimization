# Qwen3 patch sweep on OpenVINO 2026.4.1

## Base and protocol

- Patch source base: OpenVINO release tag `2026.4.1`, commit
  `e213a14725769a6a2bf86fc86650aa78a9c05d3e`.
- Model: `Qwen3-1.7B-INT4` (channel-wise), GPU, `ATTENTION_BACKEND=PA`.
- Prompt: 1000 input tokens; prompt permutation ON.
- Generation: one warm-up plus five measured iterations, four output tokens.
- Metric: TTFT arithmetic mean and standard deviation, milliseconds.
- Driver: Python `LLMPipeline` benchmark at
  `benchmark/tests/benchmark_qwen3.py`; the legacy C++ `model_bench` binary was
  not used in this sweep.
- Patches were applied cumulatively and the full GPU plugin was linked after
  each stage.
- The local Python bindings were OpenVINO 2026.4.0 / GenAI 2026.4.0.0; the
  loaded GPU plugin was built from the OpenVINO 2026.4.1 source tree. The
  existing `install/` directory was not modified.

## Results

| Patch state | TTFT mean ± std (ms) | Change vs. previous stage | Change vs. pristine |
|---|---:|---:|---:|
| Pristine 2026.4.1 | 216.53 ± 3.62 | — | — |
| opt-1 | 211.74 ± 0.26 | -2.21% | -2.21% |
| opt-1 + opt-2 | 210.49 ± 0.68 | -0.59% | -2.79% |
| opt-1 + opt-2 + opt-3 | 201.53 ± 0.19 | -4.26% | -6.92% |

The five measured runs for each stage used the same model, prompt length,
runtime configuration, and GPU. The opt-2 incremental change is small relative
to measurement noise; opt-3 provides the largest incremental TTFT reduction
in this run.

## Raw results

Raw JSON reports for each stage are stored under
[`results/runs/patch_sweep_2026.4.1_20261021/`](../results/runs/patch_sweep_2026.4.1_20261021/).
The earlier 2026.3.1 report in [`results.md`](results.md) remains historical.

This is a performance sweep, not a numerical-equivalence or broad model-quality
validation. The existing correctness evidence is preserved in the historical
reports.
