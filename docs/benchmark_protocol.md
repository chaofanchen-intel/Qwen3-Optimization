# Benchmark Protocol

## Objective

Measure Qwen3-1.7B and Qwen3-0.6B text generation at a fixed 1000-token
prompt and 512 requested output tokens for FP32, INT8, default INT4 g128, and
the requested INT4 symmetric channel-wise variant. Run every case through both
OpenVINO GenAI Python and the C++ `model_bench` implementation on the same
device.

## Environment

All Python commands must use `~/envs/qwen_env/bin/python`. The local OpenVINO
tree is `../benchmark/openvino/install`, and the C++ executable is
`../model_bench/build/model_bench`. `scripts/env.sh` makes these paths explicit
and rejects a Python environment that does not import OpenVINO 2026.3.1.

## Matrix

The matrix has eight model artifacts:

- Qwen3-1.7B: FP32, INT8, INT4 g128, INT4 channel-wise.
- Qwen3-0.6B: FP32, INT8, INT4 g128, INT4 channel-wise.

The primary device is `GPU`. The prompt is the existing
`benchmark/tests/prompts/qwen3_1k.jsonl`, whose actual tokenizer length is
1000 for both model sizes. Batch size is 1. Warm-up is one iteration and the
formal run uses ten measured iterations. Prompt permutation is disabled in
both tools so the same raw token sequence is used on every measured iteration.

## Conversion commands

The 0.6B source is downloaded at the pinned revision recorded in
`scripts/convert_models.sh`. The exact commands are:

```bash
optimum-cli export openvino -m models/source/Qwen3-0.6B \
  --task text-generation-with-past --trust-remote-code \
  models/converted/Qwen3-0.6B-FP32

optimum-cli export openvino -m models/source/Qwen3-0.6B \
  --task text-generation-with-past --weight-format int8 \
  --trust-remote-code \
  models/converted/Qwen3-0.6B-INT8

optimum-cli export openvino -m models/source/Qwen3-0.6B \
  --task text-generation-with-past --weight-format int4 --trust-remote-code \
  models/converted/Qwen3-0.6B-INT4-g128

optimum-cli export openvino -m models/source/Qwen3-0.6B \
  --task text-generation-with-past --weight-format int4 --sym --group-size -1 --ratio 1 \
  --trust-remote-code models/converted/Qwen3-0.6B-INT4-channelwise
```

For the default INT4 command, omission of `--group-size` intentionally tests
Optimum's default g128 behavior in the installed 2026.3.1-compatible stack.
The INT8 command likewise omits `--group-size`, `--sym`, and `--ratio`; its
quantization settings are entirely Optimum defaults apart from selecting
`--weight-format int8`.
The conversion logs are stored under `results/conversion/`.

## Metrics

Both reports retain raw measured iterations and aggregate metrics. The main
comparison uses arithmetic means, with standard deviation and p50/p90/p95
also rendered when available:

- TTFT: time to first generated token.
- TPOT: average time per generated token after the first token.
- First infer / other infer: model inference time split by prefill and decode.
- Total TPS: generated tokens divided by end-to-end generation time.
- Decode TPS: reciprocal of TPOT.
- Prefill, decode, E2E, tokenization, and detokenization latency.
- Compile time, output token count, result fingerprint, and raw timestamps.

Memory and utilization monitoring is disabled in the latency matrix. Run a
separate diagnostic pass when those measurements are needed because sampling
can perturb latency.

## Commands

```bash
cd /home/chaofan/model_benchmark/Qwen3-Optimization
bash scripts/convert_models.sh all
python scripts/run_matrix.py --run-id gpu_1k_512_YYYYMMDD --iterations 10
python scripts/summarize_results.py results/gpu_1k_512_YYYYMMDD
```

For a small validation:

```bash
python scripts/run_matrix.py --smoke --models qwen3_0p6b_int4_g128
python scripts/summarize_results.py results/latest
```
