# Qwen3 Prefill/TTFT Optimization

Target: reduce cold-start TTFT of Qwen3-0.6B and Qwen3-1.7B on the Intel iGPU,
with INT4 `group_size=128` as the primary case and INT4 per-output-channel as
the secondary one.

Patches live in
[`../../benchmark/openvino/qwen3_patch/`](../../benchmark/openvino/qwen3_patch/)
(archived alongside this workspace in [`../patches/`](../patches/))
with the full engineering write-up in its
[`README.md`](../../benchmark/openvino/qwen3_patch/README.md). This page is the
experiment log: how the hotspots were found, what was tried, and what the
numbers are.

## 1. Fixing the measurement first

The pre-existing TTFT numbers were unusable — they were prefix-cache hits, not
prefills, and understated real TTFT by 3-5x. The canonical baseline-first
comparison is [`results.md`](results.md), backed by
[`cold_ttft_patch_compare_20260913`](../results/cold_ttft_patch_compare_20260913/).

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
| Qwen3-1.7B | FP32* | 479.72 | 404.45 | −15.69% | 1.65 | 20.00 |
| Qwen3-1.7B | INT8* | 276.76 | 279.22 | +0.89% | 2.40 | 7.29 |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **275.51** | **−1.37%** | 1.32 | 0.73 |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 217.87 | −2.49% | 1.75 | 1.50 |

`*` 1.7B FP32 and INT8 are noisy in this sequential baseline-first run and
are not used as patch-effect claims.

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
| Qwen3-1.7B | INT8 | 270.66 | 264.66 | −2.22% |
| Qwen3-1.7B | **INT4 g128** | **272.95** | **270.59** | **−0.86%** |
| Qwen3-1.7B | INT4 channel-wise | 217.47 | 211.74 | −2.64% |

FP32 is unaffected, as it must be — an FP32 graph has no `DynamicQuantize`.

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
| Qwen3-1.7B | INT8 | 263.57 | 255.98 | −2.88% |
| Qwen3-1.7B | **INT4 g128** | **268.09** | **260.12** | **−2.98%** |
| Qwen3-1.7B | INT4 channel-wise | 210.45 | 202.02 | −4.01% |

The 56 `rope_opt` dispatches disappear entirely.

## 6. Cumulative result

| Model | Variant | pristine | opt-1+2+3 | + `GS=0` | total |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | **INT4 g128** | **141.70** | **127.94** | **116.90** | **−17.5%** |
| Qwen3-0.6B | INT4 channel-wise | 116.93 | 101.09 | — | −13.5% |
| Qwen3-1.7B | **INT4 g128** | **279.33** | **260.21** | — | **−6.8%** |
| Qwen3-1.7B | INT4 channel-wise | 223.43 | 202.02 | — | −9.6% |

`GS=0` is the runtime setting in §7 and applies to Qwen3-0.6B INT4-g128 only.

## 7. Runtime setting: Qwen3-0.6B INT4-g128 only

`DYNAMIC_QUANTIZATION_GROUP_SIZE=0` is worth a further **−11.0 ms (−8.6%)** on
Qwen3-0.6B INT4-g128 on top of all three patches (127.94 → 116.90 ms), it also
improves decode slightly (TPOT 13.95 → 13.53 ms/token), and it *improves*
accuracy because activations stay f16.

This is not a no-op: `execution_config.cpp:349` rewrites an *unset* `0` into
per-token dynamic quantization on any immad-capable GPU. Setting it explicitly
marks it user-set and actually disables it.

VTune explains it exactly — activation quantization costs a full extra pass over
every activation tensor but barely accelerates a GEMM that is bound by *weight*
dequantization:

| | default | `=0` |
|---|---:|---:|
| `gemm_kernel` | 78.39 ms | 80.29 ms (+1.90) |
| `dynamic_quantize` | 12.44 ms | **0** |
| total | 141.49 ms | 131.27 ms |

**It is not shipped as a patch.** The sign of the effect flips with model size
and quantization scheme (+12.7 ms on 0.6B channel-wise, +4.6 ms on 1.7B g128,
+56.8 ms on 1.7B channel-wise), and a couple of favourable data points do not
justify a plugin heuristic. Use it as a per-deployment setting where the model
is Qwen3-0.6B INT4-g128:

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
