#!/usr/bin/env python3
"""Numerical-equivalence gate for plugin patches: greedy-decode a fixed set of prompts on the PA backend and print
one md5 per generated text.

Run it once per plugin build and diff the output; a patch that claims to be bit-exact must reproduce every line.
Three of the prompts are long (~370, ~800 and ~1700 tokens, the first two sharing a prefix) so full and partial KV-cache
blocks, the prefix cache and a multi-tile sdpa prefill are exercised.

    source scripts/env.sh
    "$Q3_PYTHON" scripts/md5_check.py models/converted/Qwen3-0.6B-FP16 [new_tokens=96]
    # reference build: the runtime loads the GPU plugin from the directory of libopenvino.so (bin/intel64/Release);
    # OV_PLUGIN_PATH / LD_LIBRARY_PATH do not redirect it. Copy each build over bin/.../libopenvino_intel_gpu_plugin.so,
    # run, and compare the digests (scripts/tools/plugin_ab.sh does the swapping for timing runs).

"""
import sys, hashlib, openvino_genai as ov_genai
model = sys.argv[1]
nt = int(sys.argv[2]) if len(sys.argv) > 2 else 96
base = ("The history of computing spans many decades, from early mechanical calculators to modern neural networks. "
        "Engineers designed vacuum tube machines, then transistors, then integrated circuits, each shrinking cost and size. ")
prompts = [
    "Explain in detail why the sky appears blue during the day and red at sunset.",
    "Write a short Python function that merges two sorted lists, and explain its complexity.",
    "Translate to French: The quick brown fox jumps over the lazy dog near the river bank.",
    (base * 40)[:3600] + "\nSummarize the passage above in three sentences.",
    (base * 13)[:1500] + " Question: what was the main theme? Answer:",
    (base * 90)[:8000] + "\nWhat is the very first word of the passage? Answer:",   # ~1700 tokens
]
pipe = ov_genai.LLMPipeline(model, "GPU")
cfg = ov_genai.GenerationConfig(); cfg.max_new_tokens = nt; cfg.do_sample = False
for i, p in enumerate(prompts):
    out = pipe.generate(p, cfg)
    t = str(out)
    print(i, hashlib.md5(t.encode()).hexdigest(), repr(t[:50]))
