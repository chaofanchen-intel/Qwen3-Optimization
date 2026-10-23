# Qwen3 Prefill/TTFT Optimization

> Sections 1-10 are the historical 2026.3.1 INT4-g128 experiments. The active
> patches are based on OpenVINO 2026.4.1; see
> [`performance_2026.4.1.md`](performance_2026.4.1.md) for the current
> baseline and full-generation measurements, and **section 11** below for the
> FP16/INT8 follow-up that added opt-4, opt-5 and opt-6.

The current rebenchmark focuses on Qwen3-0.6B FP16 and INT8, and includes
Qwen3-0.6B/Qwen3-1.7B INT4 `group_size=128` and channel-wise variants. The
optimization history below retains the original INT4-focused 2026.3.1 studies.

Patches live in
[`../../benchmark/openvino/qwen3_patch/`](../../benchmark/openvino/qwen3_patch/)
(archived alongside this workspace in [`../patches/`](../patches/))
with the full engineering write-up in its
[`README.md`](../../benchmark/openvino/qwen3_patch/README.md). This page is the
experiment log: how the hotspots were found, what was tried, and what the
numbers are.

## 1. Fixing the measurement first

The pre-existing TTFT numbers were unusable — they were prefix-cache hits, not
prefills, and understated real TTFT by 3-5x. The historical 2026.3.1
baseline-first comparison is [`results.md`](results.md); the current 2026.4.1
full-generation measurements are in
[`performance_2026.4.1.md`](performance_2026.4.1.md).

Two further methodology corrections were needed before any optimization work:

**The profiler was profiling a graph that never runs.** `model_bench --profile`
drove the stateful SDPA graph via raw `ov::Core`, but `ov::genai::LLMPipeline`
defaults to `ATTENTION_BACKEND=PA`. The two graphs are materially different: the
PA graph has **no `Transpose` and no KV-cache `Concat` at all**, which together
were 18% of the SDPA profile. Optimizing them would have been wasted work.

`model_bench` now has a `--profile-graph pa|sdpa|both` mode (`pa` is the
default) that rebuilds what `ContinuousBatchingPipeline` builds
(`SDPAToPagedAttention` + gather-before-lm_head), allocates the KV blocks
through the compiled model's `RemoteContext`, and profiles a single-sequence
prefill.

**OpenVINO's per-op counters under-report GPU time by about 2x.** They summed to
66 ms/prefill where TTFT was 140.6 ms, which initially looked like ~50%
host-side overhead. VTune shows the true GPU compute-task time is
**142.3 ms/prefill** — prefill is essentially 100% GPU-bound. A TTFT-vs-prompt-length
sweep agrees: TTFT is linear in token count with an intercept of only ~4 ms.

| N tokens | 128 | 256 | 512 | 1000 | 2000 |
|---|---:|---:|---:|---:|---:|
| TTFT ms | 23.9 | 38.5 | 73.4 | 139.9 | 295.4 |

VTune is therefore the authority for attribution; the OV counters are only
useful for relative ranking.

## 2. Where prefill time goes

Intel VTune `gpu-hotspots`, real PA pipeline, Qwen3-0.6B INT4-g128, 1000-token
prefill, 32-XVE Pantherlake iGPU. See
[`../results/hotspot/`](../results/hotspot/).

| Kernel family | ms/prefill | % |
|---|---:|---:|
| `gemm_kernel` (oneDNN FC, incl. lm_head) | 78.39 | 55.1% |
| `sdpa_micro` prefill | 17.05 | 12.0% |
| `rms_gpu_bfyx_opt` | 16.32 | 11.5% |
| `dynamic_quantize_gpu_opt` | 12.44 | 8.7% |
| `rope_opt` | 9.73 | 6.8% |
| `pa_kv_cache_update_ref` | 7.22 | 5.1% |

Every kernel is 53-72% XVE-stalled: this workload is memory-latency-bound
throughout, not compute-throughput-bound.

**The whole INT4-g128 penalty is in the GEMMs.** Comparing against the
channel-wise model kernel by kernel, the FC GEMMs differ by +26.5 ms and
everything else by +1.7 ms combined:

| | INT4 g128 | INT4 channel-wise | delta |
|---|---:|---:|---:|
| all FC GEMMs | 78.39 | 51.92 | **+26.47** |
| everything else | 63.94 | 62.28 | +1.66 |

VTune instruction counts give the reason: for identical GEMM math, oneDNN's
grouped path issues **2.2x the ALU0 instructions** (3633M vs 1658M) and 23% more
XMX instructions, and picks a different work decomposition entirely
(`GWS = N*8 x 2` vs `256 x 8`). That is the per-128-group scale *and* zero-point
applied inside the inner loop. It sits in `thirdparty/onednn_gpu`, not in the
plugin.

## 3. Shipped: opt-1, vectorized RMS memory access

Two independent guard-expression defects in `rms_gpu_bfyx_opt.cl` forced scalar
loads/stores where sub-group block transfers were intended:

1. `USE_BLOCK_WRITE` was written `(A) & 0xF == 0`. `==` binds tighter than `&`
   in C, so it evaluated to `A & 0` — **always false**. The block-write path was
   dead code for every tensor shape in the plugin.
2. The block-read path required `workers_per_data > SUB_GROUP_SIZE`. Qwen3
   applies RMSNorm per attention head (`head_dim = 128`), which yields
   `workers_per_data == 16 == SUB_GROUP_SIZE` exactly, so `q_norm` and `k_norm`
   always took the scalar fallback. One sub-group can use block reads perfectly
   well; `>=` is correct.

Bit-exact: lane `l` accumulates elements `l, l+16, l+32, ...` in the same order
on both paths, so only the fetch instruction changes. All 39 GPU RMS unit tests
pass.

Effect (VTune, ms/prefill): `q_norm` 8.08 → 6.37, `k_norm` 4.06 → 3.25,
layernorms 4.19 → 3.64; RMS total 16.32 → 13.26.

### Result

Cold-start TTFT, 1000 tokens, arithmetic mean of 10 iterations. The baseline
arm was measured before applying opt-1; the optimized arm was measured after
rebuilding with opt-1. Full provenance is in
[`cold_ttft_patch_compare_20260913`](../results/cold_ttft_patch_compare_20260913/).

| Model | Variant | baseline mean | opt-1 mean | delta | baseline std | opt-1 std |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 148.63 | 144.47 | −2.80% | 1.01 | 0.81 |
| Qwen3-0.6B | INT8 | 132.35 | 127.11 | −3.96% | 0.99 | 0.12 |
| Qwen3-0.6B | **INT4 g128** | **141.70** | **137.57** | **−2.91%** | 0.79 | 0.50 |
| Qwen3-0.6B | INT4 channel-wise | 116.93 | 112.21 | −4.04% | 0.77 | 0.35 |
| Qwen3-0.6B | FP16 | 146.82 | 149.01 | +1.49% | 0.53 | 1.80 |
| Qwen3-1.7B | FP32* | 479.72 | 404.45 | −15.69% | 1.65 | 20.00 |
| Qwen3-1.7B | FP16* | 365.20 | 368.89 | +1.01% | 1.27 | 3.61 |
| Qwen3-1.7B | INT8* | 276.76 | 279.22 | +0.89% | 2.40 | 7.29 |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **275.51** | **−1.37%** | 1.32 | 0.73 |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 217.87 | −2.49% | 1.75 | 1.50 |

`*` 1.7B FP32, FP16, and INT8 are noisy in this sequential baseline-first run
and are not used as patch-effect claims.

The FP16 rows come from [`opt1_fp16_20260915`](../results/opt1_fp16_20260915/).

## 4. Shipped: opt-2, reuse one DynamicQuantize across FCs sharing an activation

The plugin inserts an activation `DynamicQuantize` per compressed FC. Qwen3's
`q/k/v` are already horizontally fused into one FC, but `gate_proj` and
`up_proj` are two FCs reading the *same* `post_attention_layernorm` output, so
the identical quantization ran twice — 28 redundant full read+write passes over
a `[tokens, hidden]` activation.

Reuse one DQ when the quantization config matches field-for-field, and relax an
assertion that assumed `users == outputs` (with shared consumers the ratio is
`outputs * num_consumers`). DQ node count on the PA graph: 141 → 113.

Run [`opt2_dq_reuse_20260913`](../results/opt2_dq_reuse_20260913/), baseline is
opt-1 applied:

| Model | Variant | baseline | opt-2 | delta |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 143.40 | 143.78 | +0.26% |
| Qwen3-0.6B | INT8 | 125.97 | 124.81 | −0.93% |
| Qwen3-0.6B | **INT4 g128** | **138.26** | **136.24** | **−1.46%** |
| Qwen3-0.6B | INT4 channel-wise | 112.11 | 109.95 | −1.93% |
| Qwen3-0.6B | FP16 | 153.87 | 149.97 | −2.54% |
| Qwen3-1.7B | INT8 | 270.66 | 264.66 | −2.22% |
| Qwen3-1.7B | FP16* | 375.62 | 373.39 | −0.59% |
| Qwen3-1.7B | **INT4 g128** | **272.95** | **270.59** | **−0.86%** |
| Qwen3-1.7B | INT4 channel-wise | 217.47 | 211.74 | −2.64% |

FP32 is unaffected, as it must be — an FP32 graph has no `DynamicQuantize`.

The FP16 rows come from [`opt2_fp16_20260915`](../results/opt2_fp16_20260915/)
and are noisy, especially for 1.7B.

## 5. Shipped: opt-3, fold the RoPE into the RMS epilogue

The largest structural win. Qwen3 normalizes per attention head and then
immediately rotates the same `[tokens, heads, 128]` tensor:

```
q_proj -> reshape -> q_norm (RMS) -> reshape -> RoPE -> PagedAttention
```

Both kernels are pure memory traffic at the DRAM ceiling, so the intermediate
tensor is written and read back purely to hand the value between them. Folding
the RoPE into the RMS epilogue removes it.

**The enabler.** `rms_gpu_bfyx_opt` sizes its work group so each work item
handles ~8 elements, so `head_dim = 128` gives `LWS == 16` (one sub-group) with
`SUBGROUP_BLOCK_SIZE == items_num == 8`. `BLOCK_READ8` then leaves `data[j]`
holding element `r = j*16 + lane`, and since `HALF_ROTARY_NDIMS == 4*16`,
RotateHalf's partner of element `j` is element `j+4` **in the same lane**. No
cross-lane shuffle, no extra barrier, and cos/sin arrive with one block read
each.

Two details that had to be right: the pass must walk through the rank-fixing
`Reshape` between the ops, and the token index is `row_index / head_count` where
the head count is the product of every axis between the batch and the normalized
one — Qwen3 puts heads in **Y** (`[tokens, 1, 16, 128]`), not F. Getting the
latter wrong made every token read the wrong cos/sin row and produced fluent
garbage, which is why the correctness gate below is an output comparison and not
just a smoke test.

**Correctness.** 4 prompts x 128 greedy tokens give a **byte-identical md5** to
the unfused plugin on both Qwen3-0.6B INT4-g128 and Qwen3-1.7B INT4
channel-wise. All 39 RMS and 12 RoPE GPU unit tests pass.

Run [`opt3_rope_rms_20260915`](../results/opt3_rope_rms_20260915/), baseline is
opt-1+opt-2 applied:

| Model | Variant | baseline | opt-3 | delta |
|---|---|---:|---:|---:|
| Qwen3-0.6B | FP32 | 143.10 | 134.91 | −5.73% |
| Qwen3-0.6B | INT8 | 124.25 | 114.95 | −7.49% |
| Qwen3-0.6B | **INT4 g128** | **136.22** | **128.01** | **−6.02%** |
| Qwen3-0.6B | INT4 channel-wise | 109.78 | 101.14 | −7.88% |
| Qwen3-0.6B | FP16 | 152.02 | 135.73 | −10.72% |
| Qwen3-1.7B | INT8 | 263.57 | 255.98 | −2.88% |
| Qwen3-1.7B | FP16* | 377.76 | 353.33 | −6.47% |
| Qwen3-1.7B | **INT4 g128** | **268.09** | **260.12** | **−2.98%** |
| Qwen3-1.7B | INT4 channel-wise | 210.45 | 202.02 | −4.01% |

The 56 `rope_opt` dispatches disappear entirely.

The FP16 rows come from [`opt3_fp16_20260915`](../results/opt3_fp16_20260915/);
the 1.7B FP16 baseline is noisy.

## 6. Cumulative result

The formal full-stack run is
[`opt123_all_precisions_fp16_20260915`](../results/opt123_all_precisions_fp16_20260915/).
It measures pristine 2026.3.1 first, then applies opt-1, opt-2, and opt-3 as
one stack, rebuilds, and measures every precision:

| Model | Variant | pristine mean | opt-1+2+3 mean | total |
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

`*` 1.7B FP32 and FP16 are excluded from patch-effect claims because their
sequential arms are noisy. The other rows are the complete three-patch result
for every stable precision.
The stepwise tables above remain useful for attribution; this table is the
authoritative end-to-end three-patch comparison.

## 7. Runtime setting: Qwen3-0.6B INT4-g128 only

The formal DQ=0 run is
[`opt123_dq0_0p6b_fp16_20260915`](../results/opt123_dq0_0p6b_fp16_20260915/).
It covers all five 0.6B precisions with the complete three-patch stack:

| Model | Variant | DQ=0 pristine | DQ=0 + opt-1+2+3 | Patch delta | DQ=0 vs default after patches |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | FP32 | 149.23 | 138.58 | −7.13% | +2.80% |
| Qwen3-0.6B | INT8 | 134.74 | 124.37 | −7.69% | +7.89% |
| Qwen3-0.6B | **INT4 g128** | **127.97** | **116.24** | **−9.17%** | **−8.54%** |
| Qwen3-0.6B | INT4 channel-wise | 125.40 | 113.80 | −9.24% | +12.92% |
| Qwen3-0.6B | FP16 | 149.73 | 141.88 | −5.24% | +4.87% |

Only INT4 g128 benefits from DQ=0. Its final 116.24 ms is **−16.95%** versus
the default pristine 139.96 ms. The other three formats are slower with the
setting, so it remains a per-model deployment option rather than a plugin
heuristic.

This is not a no-op: `execution_config.cpp:349` rewrites an *unset* `0` into
per-token dynamic quantization on any immad-capable GPU. Setting it explicitly
marks it user-set and actually disables it.

Earlier VTune attribution explains the INT4-g128 result: activation quantization
costs a full extra pass over every activation tensor but barely accelerates a
GEMM that is bound by *weight* dequantization. The kernel figures below are the
representative 0.6B INT4-g128 attribution run; the cross-precision numbers
above are the formal current comparison:

| | default | `=0` |
|---|---:|---:|
| `gemm_kernel` | 78.39 ms | 80.29 ms (+1.90) |
| `dynamic_quantize` | 12.44 ms | **0** |
| total | 141.49 ms | 131.27 ms |

**It is not shipped as a patch.** In the formal all-three-patch run, DQ=0 is
`+3.78 ms` on 0.6B FP32, `+9.09 ms` on INT8, `−10.85 ms` on INT4-g128, and
`+13.02 ms` on channel-wise INT4. The sign flips with quantization scheme, so
it remains a per-model deployment option rather than a plugin heuristic. Use it
for Qwen3-0.6B INT4-g128:

```cpp
ov::genai::LLMPipeline pipe(model_dir, "GPU",
                            ov::AnyMap{{"DYNAMIC_QUANTIZATION_GROUP_SIZE", 0}});
```

The principled fix that would make this automatic is item 2 in §9, which opt-3's
technique cannot reach.

## 8. Tried and rejected

| Idea | Result | Why |
|---|---|---|
| Horizontally fuse `gate_proj`+`up_proj` on the oneDNN path | **+9.07 ms (+6.4%)**, 3/3 reps | The fused FC needs a `VariadicSplit` on the innermost axis, which cannot be an in-place crop: a 12 MB copy per layer (~11 ms) exceeds the saving. Only pays with the cldnn-only SwiGLU post-fusion, itself disabled below 128 EUs. The upstream guard is right. |
| Route grouped-INT4 FCs to the cldnn OCL FC kernel | **45 515 ms** (~320x slower) | The OCL dynamic-shape FC path is not viable for this shape/type. |
| `KV_CACHE_PRECISION=f16` / `u8` | ±1-2 ms, noise | `pa_kv_cache_update` is layout-bound, not precision-bound; prefill writes the cache once and reads it once. |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` 32/64/128/256/1024 | +4 to +55 ms | 32/64 mismatch the g128 weight groups; ≥128 is uniformly worse than per-token. |
| Reverse `rope_opt`'s GWS so the contiguous rotary index is the fastest axis (upstream does this already, but only for `vec_size == 1`, excluding every f16 LLM) | +0.4 to +1.0 ms, consistent across 3 models | RoPE is not coalescing-limited. **This one read as neutral by p50 and only showed up as a regression in the mean** — which is why these comparisons use the mean. |
| Right-size the RMS kernel's SLM buffer (sized in work items, over-allocating by `SUB_GROUP_SIZE`; ~4 KB per work group for a dynamic shape, to reduce one value) | VTune: peak occupancy 40% → 100%, SLM cap removed — **runtime unchanged** (6.33 → 6.32 ms) | A real 16x over-allocation and a real resource cap, but these kernels are bandwidth-bound, not occupancy-bound. Worth reporting upstream; not shipped because it is not measurable here. |
| Pack several 128-wide rows per RMS work group (16 000 single-sub-group work groups, 512 B each) | −0.2 ms, noise | Same conclusion: bandwidth, not work-group count, is the limiter. |
| Backport `qwen3.5_patch/opt7_fuse_rms_into_dynamic_quantize.patch` | applies cleanly after opt-2, folds **nothing** | It requires 128-element activation quantization groups; this systolic-capable GPU defaults to per-token, so its guard never passes. |

## 9. Remaining headroom

Ranked by measured size on Qwen3-0.6B INT4-g128, after opt-1/2/3.

1. **Grouped-INT4 GEMM — 26.5 ms (≈20%).** Inside `thirdparty/onednn_gpu`. The
   single largest g128-specific item by far, and the reason g128 prefill is
   slower than INT8.
2. **The two remaining `DynamicQuantize` producers — ~10 ms.** opt-3's technique
   does not reach them: `down_proj`'s DQ (6.43 ms, the largest) consumes the
   SwiGLU epilogue of a oneDNN GEMM and `o_proj`'s (3.83 ms) consumes
   PagedAttention's output. Neither is a cldnn kernel whose epilogue we can
   extend, and a GEMM epilogue cannot compute a per-token scale because the row
   is split across work groups.
3. **`sdpa_micro` prefill — 17.2 ms (13%).** oneDNN micro-kernel, untouched.
4. **`pa_kv_cache_update_ref` — 7.4 ms at ~20 GB/s**, 76% stalled, 51%
   occupancy, still the reference kernel; layout-bound.
5. **`q_norm` at ~36 GB/s vs the layernorm's ~64 GB/s.** Two follow-ups tried,
   both neutral (§8), so the residual is per-work-group launch cost for 128-wide
   rows rather than occupancy.

## 10. Tooling added

| Path | Purpose |
|---|---|
| [`../scripts/run_matrix.py`](../scripts/run_matrix.py) | Single entry point for full generation, prefill-only, and baseline-first patch comparison |
| [`../scripts/vtune_gpu.sh`](../scripts/vtune_gpu.sh) | VTune `gpu-hotspots` on the real PA pipeline |
| `model_bench --profile-graph pa` | Per-op profiling of the PagedAttention graph |

VTune note: `collect-memory-bandwidth` needs the PAX PMU-reservation driver,
which is not installed here, and enabling it aborts the collection. GPU metrics
come from the `xe` driver's observation interface
(`/proc/sys/dev/xe/observation_paranoid = 0`) and work without it.

## 11. FP16/INT8 follow-up on 2026.4.1: opt-4, opt-5, opt-6 (2026-10-22/23)

Goal: push the TTFT of Qwen3-0.6B FP16 and INT8 further after opt-1..3. Method as before
(VTune `gpu-hotspots` on the real PA pipeline, mean +/- std, md5 gate); the new part is that for
the first time the kernels were also compared against **hardware ceilings**, which turned out to
decide most of the outcome. Full write-up with the numbers:
[`qwen3_patch/README.md`](../../benchmark/openvino/qwen3_patch/README.md).

### 11.1 Where FP16 / INT8 stand after opt-1..3

| Kernel family | FP16 ms | INT8 ms |
|---|---:|---:|
| `gemm_kernel` (oneDNN FC, incl. `lm_head` 7.4 / 3.7) | 99.12 (73%) | 66.54 (58%) |
| `sdpa_micro` prefill | 17.25 | 17.08 |
| `rms_gpu_bfyx_opt` (with fused RoPE) | 11.79 | 14.20 |
| `dynamic_quantize_gpu_opt` | - | 11.87 |
| `pa_kv_cache_update_ref` | 7.30 | 5.48 |
| total (TTFT 133.4 / 114.5) | 135.67 | 115.44 |

Note that the FP16 graph differs from the INT4 one: its Q/K/V are three separate GEMMs (N = 2048,
1024, 1024) and its KV cache is uncompressed; the INT8 graph has the fused QKV and an i8 KV cache.

### 11.2 What shipped

| Patch | Effect on 0.6B (C++ / Python, stage runs) |
|---|---|
| opt-4 vectorized full-block KV-cache update (FP16 cache only, bit-exact) | FP16 -1.1% / -1.4%, kernel 7.30 -> 5.50 ms |
| opt-5 `sdpa_micro` tile 32x128 on 8 sub-groups for integrated Xe3, PA prefill, head 128 (not bit-exact; 256-key bit-exact variant documented) | FP16 -2.5% / -2.7%, INT8 -3.1% / -3.6%, INT4 -3.1% to -4.4%; `sdpa_micro` 17.2 -> 12.8 ms; 2000/4000-token prompts -11.5/-31.5 ms |
| opt-6 fuse Q/K/V of uncompressed FCs (bit-exact, FP16 only) | FP16 127.7 -> 125.9 ms (interleaved A/B) |
| **Total vs opt-1+2+3** | **FP16 -4.5 to -5.2%, INT8 -3.3 to -3.8%, INT4 g128 -2.7 to -3.0%, INT4 cw -3.6 to -3.8%**; 1.7B -1.1 to -2.5% |

### 11.3 What the hardware says about the rest

These are the facts that stopped further kernel work; each is reproducible with
[`scripts/tools/gemm_probe/`](../scripts/tools/gemm_probe/).

1. **The GPU sees 42 GB/s of DRAM read bandwidth and ~27 GB/s write** (60 GB/s copy), and a 2-6 MB
   buffer is not read faster than a 512 MB one: there is no cache between kernels worth the name.
   RMS (218 us for the 8 MB `q_norm` moves), the KV update, `dynamic_quantize` and the 311 MB
   `lm_head` GEMV (7.4 ms = 42 GB/s) are all *at* this floor. Only removing traffic helps, which is why
   opt-3 (a fusion) worked and RMS micro-tuning did not.
2. **XMX throughput is data dependent.** The same f16 GEMM runs at 17-19 TFLOPS with constant-ish
   operands and at 9.3-13 TFLOPS with random ones (also a 4096^3 GEMM); clocks stay at 2450 MHz and no
   throttle reason is set. The model's GEMMs run at ~9.6 TFLOPS and `sdpa_micro` at ~9 TFLOPS, so ~85%
   of the FP16 prefill is bounded by this. Corollary: a GEMM microbenchmark on constant data (and
   `benchdnn`'s default fill) overestimates the model by ~1.7x - the first probe here did, and cost an
   afternoon of looking for a gap that did not exist.
3. **oneDNN's GEMM tile selection is already right**: of ~30 strategies per problem the selector's pick was
   the fastest (FP16 q-proj 11.9 vs 11.0 TFLOPS runner-up).
4. **Binary post-ops are expensive.** On INT8 1024x1024x3072: no post-op 360 us, swish 377 us, binary
   mul/add/sum 543 us. A post-op that reads a second [M, N] tensor is not prefetched, so the read latency
   is exposed at the end of each tile: ~180 us per layer for gate x up, ~65 and ~55 us for the two residual
   adds, 4-8 ms per prefill. It lives inside the oneDNN GEMM generator (open item 1).

### 11.4 Measurement traps found this time

- **The runtime loads the GPU plugin from the directory of `libopenvino.so` (here `bin/intel64/Release`).**
  `OV_PLUGIN_PATH` and `LD_LIBRARY_PATH` do not redirect it. An A/B that "swapped" plugins through those
  variables compared the same binary with itself (and produced a plausible-looking -0.4 ms for opt-6 whose
  real, interleaved value is -1.8 ms). `scripts/tools/plugin_ab.sh` copies each candidate over the plugin and restores it.
- Verify which plugin is mapped (`/proc/self/maps`) before believing an A/B or an md5 comparison.
- The OV per-op counter of a `PagedAttentionExtension` node does not include every kernel of the node; do not use
  it to attribute the KV update (use VTune).
- Sequential stage runs drift by ~1-1.5 ms over minutes; interleave, and repeat the baseline stage.

### 11.5 Negative results of this round

| Idea | Result |
|---|---|
| Enable the oneDNN gated-MLP fusion (`FuseGatedMLP`) | FP16 +14% (152.2 vs 133.4 ms), INT8 +44% |
| 4-way key/value split of the KV update | no change |
| Vector path for the INT8 by-channel cache | TTFT unchanged (traffic-bound at 5.5 ms) |
| GEMM strategy sweep / forced C prefetch | default already best / prefetch variants hang kernel creation |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` 256..per-token for INT8 | -1.4 to -3.1 ms, but the plugin clamps INT8 to 128-element groups on purpose (accuracy) - runtime option only |

