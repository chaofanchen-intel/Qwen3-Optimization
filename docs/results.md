# Qwen3 1k-512 Results

## Formal Run

The completed formal run is [`gpu_1k_512_20260911_default_int8`](../results/gpu_1k_512_20260911_default_int8/).
It contains 8 model variants x 2 backends, with one warm-up and 10 measured
iterations per case. Every case used the same 1000-token JSONL prompt, 512
requested output tokens, batch size 1, GPU, greedy decoding, and disabled
prompt permutation.

The full generated report is [`comparison.md`](../results/gpu_1k_512_20260911_default_int8/comparison.md).
It contains mean, population standard deviation, p50, p90, p95, min, and max
for every recorded metric. The normalized machine-readable files are
[`summary.csv`](../results/gpu_1k_512_20260911_default_int8/summary.csv) and
[`summary.json`](../results/gpu_1k_512_20260911_default_int8/summary.json).
Raw per-iteration reports and command logs are under
[`raw/`](../results/gpu_1k_512_20260911_default_int8/raw/), with run metadata in
[`metadata.json`](../results/gpu_1k_512_20260911_default_int8/metadata.json).

Environment: `~/envs/qwen_env`, OpenVINO 2026.3.1, OpenVINO GenAI 2026.3.1,
Optimum Intel 2.1.0, NNCF 3.2.0, Transformers 5.2.0, and Torch 2.13.0.

## Main Metrics

Values are arithmetic means over the 10 measured iterations. `TPS` is total
generation throughput, `decode TPS` is the reciprocal of TPOT, and `E2E` is
the full generation time for the 512 output tokens. The detailed report linked
above contains the percentile columns.

| Model | Variant | Backend | TTFT ms | TPOT ms/token | TPS token/s | Decode TPS | E2E ms | Compile s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | C++ | 38.16 | 33.70 | 29.66 | 29.67 | 17261.06 | 4.27 |
| Qwen3-0.6B | FP32 | Python | 38.65 | 33.72 | 29.64 | 29.65 | 17271.80 | 4.57 |
| Qwen3-0.6B | INT8 default | C++ | 40.08 | 18.76 | 53.19 | 53.32 | 9625.79 | 3.88 |
| Qwen3-0.6B | INT8 default | Python | 41.48 | 18.65 | 53.50 | 53.63 | 9570.49 | 4.08 |
| Qwen3-0.6B | INT4 g128 default | C++ | 39.19 | 13.75 | 72.47 | 72.75 | 7064.78 | 4.03 |
| Qwen3-0.6B | INT4 g128 default | Python | 40.16 | 13.76 | 72.41 | 72.68 | 7071.36 | 4.29 |
| Qwen3-0.6B | INT4 channel-wise sym | C++ | 34.63 | 13.49 | 73.93 | 74.17 | 6927.67 | 3.38 |
| Qwen3-0.6B | INT4 channel-wise sym | Python | 35.12 | 13.36 | 74.62 | 74.86 | 6861.21 | 3.54 |
| Qwen3-1.7B | FP32 | C++ | 91.67 | 86.41 | 11.57 | 11.57 | 44247.44 | 28.96 |
| Qwen3-1.7B | FP32 | Python | 96.29 | 88.61 | 11.28 | 11.29 | 45381.96 | 28.56 |
| Qwen3-1.7B | INT8 default | C++ | 52.29 | 45.17 | 22.13 | 22.14 | 23133.74 | 4.63 |
| Qwen3-1.7B | INT8 default | Python | 52.71 | 45.02 | 22.21 | 22.21 | 23056.67 | 22.38 |
| Qwen3-1.7B | INT4 g128 default | C++ | 44.98 | 29.57 | 33.79 | 33.82 | 15154.68 | 4.54 |
| Qwen3-1.7B | INT4 g128 default | Python | 46.32 | 29.51 | 33.85 | 33.89 | 15126.62 | 10.77 |
| Qwen3-1.7B | INT4 channel-wise sym | C++ | 40.82 | 28.46 | 35.11 | 35.14 | 14584.74 | 3.83 |
| Qwen3-1.7B | INT4 channel-wise sym | Python | 41.97 | 28.46 | 35.11 | 35.14 | 14584.56 | 8.64 |

## Python Versus C++

The delta is `(Python / C++) - 1`. Positive latency delta means Python is
slower. Positive throughput delta means Python is faster.

| Model variant | TTFT delta | TPOT delta | TPS delta | E2E delta |
|---|---:|---:|---:|---:|
| Qwen3-0.6B FP32 | +1.27% | +0.06% | -0.06% | +0.06% |
| Qwen3-0.6B INT8 default | +3.50% | -0.58% | +0.58% | -0.57% |
| Qwen3-0.6B INT4 g128 default | +2.48% | +0.09% | -0.09% | +0.09% |
| Qwen3-0.6B INT4 channel-wise sym | +1.42% | -0.96% | +0.94% | -0.96% |
| Qwen3-1.7B FP32 | +5.04% | +2.54% | -2.50% | +2.56% |
| Qwen3-1.7B INT8 default | +0.79% | -0.33% | +0.33% | -0.33% |
| Qwen3-1.7B INT4 g128 default | +2.98% | -0.19% | +0.19% | -0.19% |
| Qwen3-1.7B INT4 channel-wise sym | +2.83% | -0.00% | +0.00% | -0.00% |

### Interpretation

- For decode, Python and C++ are effectively equivalent on the quantized
	models. The largest E2E difference is under 1% for every INT8/INT4 case.
- The visible difference is mainly prefill/TTFT: Python is 0.79% to 3.50%
	slower on quantized cases and 5.04% slower for 1.7B FP32.
- Python-side tokenization and detokenization are higher than C++ in the raw
	reports, but they are small compared with 512-token generation time. The
	full per-stage values are in `comparison.md`.
- The 1.7B FP32 result is the only case with a material overall backend gap:
	Python E2E is 2.56% slower and total TPS is 2.50% lower.

## Precision Conclusions

Using the Python means for a direct application-facing comparison:

- **Qwen3-0.6B:** channel-wise INT4 reaches 74.62 TPS and 35.12 ms TTFT.
	It reduces E2E by 60.28% and raises TPS by 151.73% versus FP32. It is
	about 3.0% faster in E2E than default INT4 g128.
- **Qwen3-1.7B:** channel-wise INT4 reaches 35.11 TPS and 41.97 ms TTFT.
	It reduces E2E by 67.86% and raises TPS by 211.16% versus FP32. It is
	about 3.6% faster in E2E than default INT4 g128, and its TPOT is 3.56%
	lower than g128.
- **INT8:** sits between FP32 and INT4 for both model sizes. It improves
	Python E2E by 44.59% for 0.6B and 49.19% for 1.7B versus FP32, while its
	throughput gains are 80.47% and 96.83% respectively.
- **Channel-wise versus g128:** channel-wise is faster in this GPU workload
	for both sizes. The observed advantage is small for 0.6B and clearer for
	1.7B, so accuracy evaluation should decide whether the speed gain justifies
	the quantization choice.

## Measurement Notes

- The formal latency run disabled CPU/GPU/NPU monitoring to avoid sampler
	overhead. Memory and utilization are therefore not part of this matrix.
- The 0.6B INT8 artifact was re-exported with the default command:
	`--task text-generation-with-past --weight-format int8 --trust-remote-code`.
	No explicit `--group-size`, `--sym`, or `--ratio` was supplied. Its actual
	NNCF metadata is `mode=int8_asym`, `group_size=-1`, and `ratio=1.0`.
- Both tools used the same OpenVINO 2026.3.1 runtime and the same model IR
	directories. The runner recorded every exact command in `metadata.json`.
- Do not hand-edit generated `summary.csv`, `summary.json`, or `comparison.md`.
	Re-run `scripts/summarize_results.py` from the retained raw JSON reports.
