#!/usr/bin/env python3
"""Sweep the sdpa_micro prefill tile configuration and report cold-prefill TTFT per candidate.

Needs a plugin built with sdpa_config_env_override.patch (the OV_SDPA_CFG hook). Each candidate is
(m_kq, n_kq, m_vs, n_vs, wg_m_kq, wg_n_kq, wg_m_vs, wg_n_vs) as in sdpa_config_t; valid candidates satisfy
  m_vs * wg_m_vs == head_size,  n_vs * wg_n_vs == n_kq * wg_n_kq,  wg_m_kq * wg_n_kq == wg_m_vs * wg_n_vs.
Invalid or unsupported ones fail to compile and print FAIL. The reference config is re-measured every 10 candidates
because the GPU drifts by ~1 ms over a sweep. Report the mean (std next to it), never the p50.

    source scripts/env.sh
    python3 scripts/tools/sdpa_config_sweep.py models/converted/Qwen3-0.6B-FP16 [head_size=128] [reference=16,64,32,16,16,2,4,8]
"""
import os
import re
import statistics
import subprocess
import sys

model = sys.argv[1]
head = int(sys.argv[2]) if len(sys.argv) > 2 else 128
reference = sys.argv[3] if len(sys.argv) > 3 else "16,64,32,16,16,2,4,8"
bench = os.environ["Q3_CPP_BENCH"]


def ttft(cfg, iters=6):
    out = subprocess.run([bench, "llm", "-m", model, "-d", "GPU", "-n", str(iters), "-ic", "1", "--input-tokens", "1000", "--no-monitor"],
                         env={**os.environ, "OV_SDPA_CFG": cfg}, capture_output=True, text=True).stdout
    vals = [float(v) for v in re.findall(r"^\s*\[\d+\]\[P0\].*?TTFT=([\d.]+)ms", out, re.M)]
    return f"{statistics.mean(vals):.2f} +/- {statistics.pstdev(vals):.2f}" if vals else "FAIL"


candidates = []
for sgs in (4, 8, 16):
    for umv in (16, 32, 64):
        wmv = head // umv
        if wmv > sgs or sgs % wmv:
            continue
        wnv = sgs // wmv
        for unv in (16, 32):
            tile_q = unv * wnv
            for wnk in (1, 2, 4):
                if sgs % wnk or tile_q % wnk:
                    continue
                unk = tile_q // wnk
                if unk not in (16, 32, 64):
                    continue
                for umk in (16, 32):
                    candidates.append((umk, unk, umv, unv, sgs // wnk, wnk, wmv, wnv))

print(f"{len(candidates)} candidates; reference {reference}: {ttft(reference)}", flush=True)
for i, c in enumerate(dict.fromkeys(candidates)):
    cfg = ",".join(map(str, c))
    print(cfg, ttft(cfg), flush=True)
    if i % 10 == 9:
        print("reference", ttft(reference), flush=True)
