# GEMM / memory probes

Small standalone programs used to decide whether the GEMMs of the Qwen3 prefill have headroom, without going through
the model. Build against the oneDNN that the plugin itself uses (`src/plugins/intel_gpu/thirdparty/onednn_gpu`), in a
separate build directory so the OpenVINO tree is not touched:

```bash
SRC=<openvino>/src/plugins/intel_gpu/thirdparty/onednn_gpu
cmake -G Ninja -S $SRC -B /tmp/dnnl_dev -DCMAKE_BUILD_TYPE=Release -DONEDNN_CPU_RUNTIME=NONE -DONEDNN_GPU_RUNTIME=OCL \
      -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_GRAPH=OFF -DONEDNN_ENABLE_PRIMITIVE="MATMUL;REORDER" \
      -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_ENABLE_PRIMITIVE_GPU_ISA="XE2;XE3;XE3P;XEHPC" \
      -DONEDNN_DEV_MODE=ON            # only needed for gemm_strategy_sweep.py (GEMM_KERNEL override)
ninja -C /tmp/dnnl_dev dnnl
g++ -O2 -std=c++17 -DCL_TARGET_OPENCL_VERSION=300 fp16_matmul_probe.cpp -I/tmp/dnnl_dev/include -I$SRC/include \
    /tmp/dnnl_dev/src/libdnnl.a -lOpenCL -ldl -lpthread -o fp16_matmul_probe
```

What they established on Pantherlake-P (32 XVE, 2.45 GHz), 0.6B shapes, M = 1008:

| Probe | Result |
|---|---|
| `fp16_matmul_probe`, constant-ish data | 17-19 TFLOPS (close to the ~20 TFLOPS XMX peak) |
| `fp16_matmul_probe`, random data | **9.3-13 TFLOPS**, including a 4096^3 GEMM; clocks stay at 2450 MHz and no throttle reason is set — XMX throughput is data dependent. The model-internal GEMMs run at ~9.6 TFLOPS, i.e. at this limit |
| `gemm_strategy_sweep.py` | oneDNN's pick is the fastest of its ~30 candidate strategies (q-proj 11.9 TFLOPS vs 11.0 for the runner-up) |
| `int8_dynquant_matmul_probe` | 15-20 TOPS; symmetric weights change nothing; activation group 1024 instead of 128 gives +10% on the K=1024 GEMMs only |
| post-op cost (INT8, 1024x1024x3072) | none 360 us, swish 377 us, **binary mul/add/sum 543 us**: a post-op that reads a second [M, N] tensor is not prefetched and its latency is exposed (+182 us for the 6 MB gate/up operand, ~+55-65 us for the 2 MB residual adds) |
| `dram_read_bw` | 42 GB/s read ceiling, ~27 GB/s write, ~60 GB/s combined copy |
