// Same probe for the INT8 FC the plugin builds on this GPU: s8 activations with group-wise scales (+ precomputed
// reductions), u8 weights with per-channel scale and zero point, f16 destination. SYM=1 drops the zero point,
// GS=<n> changes the activation group size (default 128).
//
//   [SYM=1] [GS=128] int8_dynquant_matmul_probe M K N [none|relu|swish|mul|add|sum|swiglu] [reps=50] [weight_copies=1]
//
// Used to isolate the cost of each GEMM post-op: swish is ~free, but any binary post-op (add/mul/sum) that reads a
// second [M, N] tensor costs +50% of the GEMM on [1024 x 1024 x 3072].
#include <dnnl.hpp>
#include <dnnl_ocl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
using namespace dnnl;
static uint32_t rs=777; static uint32_t rn(){ rs=rs*1664525u+1013904223u; return rs; }
static uint16_t rnd16(){ uint32_t x=rn(); uint32_t sign=(x>>31)<<15; uint32_t e=14+((x>>16)&3)-1; uint32_t m=(x>>3)&0x3ff; return sign|(e<<10)|m; }
int main(int argc, char** argv) {
    int M = atoi(argv[1]), K = atoi(argv[2]), N = atoi(argv[3]);
    std::string po = argc > 4 ? argv[4] : "none";
    int reps = argc > 5 ? atoi(argv[5]) : 50;
    int NB = argc > 6 ? atoi(argv[6]) : 1;
    int G = 128; 
    engine eng(engine::kind::gpu, 0);
    stream s(eng, stream::flags::in_order);
    using dt = memory::data_type; bool sym = getenv("SYM") != nullptr; int GS = getenv("GS") ? atoi(getenv("GS")) : 128; G = GS;
    memory::desc sd({1, M, K}, dt::s8, memory::format_tag::abc);
    memory::desc wd({1, K, N}, sym ? dt::s8 : dt::u8, memory::format_tag::acb);
    memory::desc dd({1, M, N}, dt::f16, memory::format_tag::abc);
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_scales(DNNL_ARG_SRC, 7, {1, G}, dt::f16);
    attr.set_scales(DNNL_ARG_WEIGHTS, 4, {}, dt::f16);
    if (!sym) { attr.set_zero_points(DNNL_ARG_WEIGHTS, 4, {}, dt::u8);
    attr.set_precomputed_reductions(DNNL_ARG_SRC, 7, {1, G}, dt::s32); }
    post_ops ops;
    if (po == "add") ops.append_binary(algorithm::binary_add, dd);
    if (po == "swish") { ops.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f); }
    if (po == "sum") ops.append_sum(1.0f);
    if (po == "mul") ops.append_binary(algorithm::binary_mul, dd);
    if (po == "relu") ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
    if (po == "swiglu") { ops.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f); ops.append_binary(algorithm::binary_mul, dd); }
    attr.set_post_ops(ops);
    matmul::primitive_desc pd(eng, sd, wd, dd, attr);
    matmul mm(pd);
    memory::desc ssd({1, M, K / G}, dt::f16, memory::format_tag::abc);
    memory::desc srd({1, M, K / G}, dt::s32, memory::format_tag::abc);
    memory::desc wsd({N}, dt::f16, memory::format_tag::a);
    memory::desc wzd({N}, dt::u8, memory::format_tag::a);
    memory src(sd, eng), dst(dd, eng), aux(dd, eng), scr(pd.scratchpad_desc(), eng), ss(ssd, eng), sr(srd, eng), ws(wsd, eng), wz(wzd, eng);
    std::vector<memory> weis; for (int b = 0; b < NB; b++) weis.emplace_back(wd, eng);
    auto fill8 = [&](memory& m){ auto* p=(uint8_t*)m.map_data(); size_t n=m.get_desc().get_size(); for(size_t i=0;i<n;i++) p[i]=rn()>>24; m.unmap_data(p); };
    auto fill16 = [&](memory& m){ auto* p=(uint16_t*)m.map_data(); size_t n=m.get_desc().get_size()/2; for(size_t i=0;i<n;i++) p[i]=rnd16(); m.unmap_data(p); };
    fill8(src); for (auto& w : weis) fill8(w); fill16(aux); fill16(ws); fill8(wz);
    { auto* p=(int32_t*)sr.map_data(); for(size_t i=0;i<(size_t)M*K/G;i++) p[i]=(int32_t)(rn()%1000); sr.unmap_data(p);} 
    { auto* p=(uint16_t*)ss.map_data(); for(size_t i=0;i<(size_t)M*K/G;i++) p[i]=0x2000+(rn()&0xff); ss.unmap_data(p);} 
    std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, weis[0]}, {DNNL_ARG_DST, dst}, {DNNL_ARG_SCRATCHPAD, scr},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ss}, {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, ws}};
    if (!sym) { args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = wz; args[DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | DNNL_ARG_SRC] = sr; }
    if (po == "add" || po == "mul" || po == "swiglu") args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(po == "swiglu" ? 1 : 0) | DNNL_ARG_SRC_1] = aux;
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
    printf("I8 M=%d K=%d N=%d po=%s : %.1f us  %.2f TOPS\n", M, K, N, po.c_str(), best * 1000, 2.0 * M * N * K / best / 1e9);
}
