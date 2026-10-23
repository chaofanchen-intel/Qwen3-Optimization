// Standalone oneDNN GPU matmul probe for the Qwen3 prefill shapes (FP16 weights, weights transposed "acb" like the
// plugin's FC). See README.md in this directory for how to build it against the plugin's own oneDNN.
//
//   fp16_matmul_probe M K N [none|add|swiglu] [reps=50] [weight_copies=1]
//
// Operands are filled with random half-precision values on purpose: with constant-ish data the same GEMM reports
// 17-19 TFLOPS, with random data 9-13 TFLOPS (XMX power is data dependent), which is what the model sees.
// `weight_copies` rotates through that many distinct weight buffers to take weight cache residency out of the picture.
#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <CL/cl.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <string>
using namespace dnnl;
static uint16_t rnd16(){ static uint32_t x=12345; x=x*1664525u+1013904223u; uint32_t sign=(x>>31)<<15; uint32_t e=14+((x>>16)&3)-1; uint32_t m=(x>>3)&0x3ff; return sign|(e<<10)|m; }
int main(int argc, char** argv) {
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    std::string po = argc > 4 ? argv[4] : "none";
    int reps = argc > 5 ? atoi(argv[5]) : 50;
    engine eng(engine::kind::gpu, 0);
    cl_command_queue q; 
    stream s(eng, stream::flags::in_order);
    q = ocl_interop::get_command_queue(s);
    memory::desc sd({1, M, K}, memory::data_type::f16, memory::format_tag::abc);
    memory::desc wd({1, K, N}, memory::data_type::f16, memory::format_tag::acb);
    memory::desc dd({1, M, N}, memory::data_type::f16, memory::format_tag::abc);
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    post_ops ops;
    if (po == "add") ops.append_binary(algorithm::binary_add, dd);
    if (po == "swiglu") { ops.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f); ops.append_binary(algorithm::binary_mul, dd); }
    attr.set_post_ops(ops);
    matmul::primitive_desc pd(eng, sd, wd, dd, attr);
    matmul mm(pd);
    int NB = argc > 6 ? atoi(argv[6]) : 1;
    memory src(sd, eng), dst(dd, eng), aux(dd, eng), scr(pd.scratchpad_desc(), eng);
    std::vector<memory> weis; for (int b = 0; b < NB; b++) weis.emplace_back(wd, eng);
    memory wei = weis[0];
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scr}};
    if (po != "none") args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(po == "add" ? 0 : 1) | DNNL_ARG_SRC_1] = aux;
    // fill with small values
    for (auto* m : {&src, &aux, &weis[0]}) { auto* p = (uint16_t*)m->map_data(); size_t n = m->get_desc().get_size() / 2; for (size_t i = 0; i < n; i++) p[i] = rnd16(); m->unmap_data(p); }
    for (int b = 1; b < NB; b++) { auto* p = (uint16_t*)weis[b].map_data(); size_t n = wd.get_size()/2; for (size_t i = 0; i < n; i++) p[i] = rnd16(); weis[b].unmap_data(p);} 
    for (int i = 0; i < 5; i++) mm.execute(s, args);
    s.wait();
    double best = 1e9;
    for (int trial = 0; trial < 5; trial++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) { args[DNNL_ARG_WEIGHTS] = weis[i % NB]; mm.execute(s, args); }
        s.wait();
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        if (ms < best) best = ms;
    }
    printf("M=%d K=%d N=%d po=%s : %.1f us  %.2f TFLOPS  impl=%s\n", M, K, N, po.c_str(), best * 1000, 2.0 * M * N * K / best / 1e9, pd.impl_info_str());
}
