# Qwen3 Optimization Benchmark

This directory is the reproducible workspace for Qwen3 generation performance
experiments. The first protocol is a matched `1000` input-token / `512`
generated-token benchmark on `GPU`, executed by both the OpenVINO GenAI Python
benchmark and the C++ `model_bench` runner.

The Python environment is intentionally fixed to:

```text
~/envs/qwen_env
OpenVINO 2026.3.1
OpenVINO GenAI 2026.3.1
Optimum Intel 2.1.0
NNCF 3.2.0
Transformers 5.2.0
Torch 2.13.0
```

## Layout

```text
Qwen3-Optimization/
├── configs/                 # versioned experiment matrix and protocol knobs
├── docs/                    # protocol, interpretation, and generated results
├── inputs/                  # prompt and future accuracy-analysis inputs
├── models/
│   ├── source/              # downloaded source checkpoints, not committed
│   └── converted/           # 0.6B artifacts produced by convert_models.sh
├── results/
│   ├── runs/                # immutable raw CSV/JSON/logs per run id
│   └── latest -> runs/...   # convenience symlink made by the runner
└── scripts/
    ├── env.sh               # qwen_env and local OpenVINO runtime contract
    ├── convert_models.sh    # download and convert missing 0.6B variants
    ├── run_matrix.py        # run Python and/or C++ for the same matrix
    └── summarize_results.py # normalize, compare, and render Markdown/CSV
```

Existing 1.7B artifacts remain in `../benchmark/models` and are referenced by
the matrix rather than copied. This keeps the experiment directory suitable
for later hotspot profiling, router accuracy evaluation, and optimization
work without duplicating large model files.

## Precision mapping

| Matrix label | Model directory | Conversion meaning |
|---|---|---|
| FP32 | `Qwen3-*-FP32` | floating-point OpenVINO IR |
| INT8 | `Qwen3-*-INT8` | Optimum default weight-only INT8; no explicit group size/symmetry/ratio |
| INT4-g128 | `Qwen3-*-INT4-g128` | Optimum default INT4, `group-size=128`, asymmetric |
| INT4-channelwise | `Qwen3-*-INT4-channelwise` | `--weight-format int4 --sym --group-size -1 --ratio 1` |

The existing 1.7B `Qwen3-1.7B-INT4` metadata was inspected and matches the
requested channel-wise case exactly: `group_size=-1`, `mode=int4_sym`, and
`ratio=1.0`. The older `Qwen3-1.7B-int4-ov` is not used because it is an old
`group_size=128`, `ratio=0.8` artifact.

## Quick start

Run all missing 0.6B conversions first:

```bash
cd /home/chaofan/model_benchmark/Qwen3-Optimization
bash scripts/convert_models.sh all
```

Run a short one-iteration smoke matrix:

```bash
python scripts/run_matrix.py --smoke
python scripts/summarize_results.py results/latest
```

Run the formal matrix after conversions complete:

```bash
python scripts/run_matrix.py --run-id gpu_1k_512_20260911 --iterations 10
python scripts/summarize_results.py results/gpu_1k_512_20260911
```

The runner invokes both backends by default. Use `--backends python` or
`--backends cpp` when repeating only one side. Every run stores the exact
command, environment metadata, stdout/stderr log, detailed per-iteration
report, and average report. Iteration zero is warm-up and is excluded from
the reported means by both tools.

## Fairness controls

- Both tools read the same JSONL prompt at
  `../benchmark/tests/prompts/qwen3_1k.jsonl`; the actual OpenVINO tokenizer
  measures it as exactly 1000 tokens for both Qwen3 sizes.
- Both use `GPU`, batch size 1, greedy generation, 512 requested new tokens,
  one warm-up, and prompt permutation disabled.
- Hardware monitoring is disabled during latency measurements to avoid adding
  sampling overhead. Memory and utilization can be collected in a separate
  diagnostic run.
- Raw samples are retained. Summary statistics are arithmetic means over
  measured iterations; p50/p90/p95 and standard deviation are also retained
  where available.
- Python and C++ are compared on the same model directory and device. A
  positive `Python minus C++` percentage means Python is slower for that
  metric; negative means Python is faster.

## Hotspot follow-up

The performance matrix deliberately does not enable per-operation profiling.
For a selected model/device, use the C++ runner after the matrix:

```bash
/home/chaofan/model_benchmark/model_bench/build/model_bench llm \
  -m /home/chaofan/model_benchmark/benchmark/models/Qwen3-1.7B-INT4 \
  -d GPU -n 3 -ic 128 -pf ../benchmark/tests/prompts/qwen3_1k.jsonl -pi 0 \
  --disable-prompt-permutation --no-monitor --profile \
  --profile-iters 10 --profile-csv results/hotspot/qwen3_1p7b_channelwise_gpu.csv
```

Keep hotspot reports under `results/hotspot/`; future router accuracy assets
belong under `inputs/router/` and `docs/router/`, rather than mixing them with
latency summaries.
