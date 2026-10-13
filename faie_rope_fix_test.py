"""Corrected rope-style test vs the oracle.

Why this exists: llama.cpp b10069 maps LLM_ARCH_GLM_DSA -> LLAMA_ROPE_TYPE_NORM
(llama-model.cpp, the rope-type switch), i.e. INTERLEAVED pairs (2i, 2i+1) = FAIE
style 1. FAIE has been defaulting to style 0 (NeoX). faie_sweep.py toggled the
style but (a) warmed expert residency only at the default setting and (b) never
gated on requested==executed, so its per-variant rows are contaminated
(trap #3 in the handoff).

This test re-warms to 0 misses PER STYLE and gates every run on 100% residency.
"""
import sys, ctypes, io
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

ORACLE_TOP1 = 12089                      # " Paris"
ORACLE_TOP5 = {12089, 5990, 9621, 10003, 1429}
PROMPT = "The capital of France is"

reader = GGUFReader(MODEL, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))

lib.set_rope_style.argtypes = [ctypes.c_int]
lib.set_score_scale.argtypes = [ctypes.c_float]
lib.set_router_scale.argtypes = [ctypes.c_float]
lib.set_bisect.argtypes = [ctypes.c_int, ctypes.c_int]
lib.set_bisect(0, 0)
lib.set_score_scale(ctypes.c_float(1.0))
lib.set_router_scale(ctypes.c_float(2.5))

IDS = tok.encode(PROMPT, limit=32)
print(f"prompt ids = {IDS}")
assert IDS == [785, 6722, 315, 9621, 374], "tokenizer mismatch vs live oracle!"

# first forward to trigger expert registration
eng.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()


def run_gated(label):
    """Warm to 0 misses, then a gated run. Returns logits or None if gate fails."""
    warmed = False
    for it in range(25):
        eng.reset_context()
        for t in IDS:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        paged, dropped = eng.page_in_misses()
        if paged == 0 and dropped == 0:
            warmed = True
            break
    lib.reset_expert_counters()
    eng.reset_context()
    for t in IDS:
        lg = eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    req = lib.get_experts_requested()
    exe = lib.get_experts_executed()
    misses = eng.page_in_misses()
    ok = warmed and req == exe and misses[0] == 0
    print(f"[{label}] warm_iters_ok={warmed} requested={req} executed={exe} "
          f"post_misses={misses} residency={'100%' if req==exe else 'PARTIAL'}")
    if not ok:
        print(f"[{label}] GATE FAILED — result below is not trustworthy")
    return lg.squeeze().float()


results = {}
for style, name in [(1, "gptj/NORM (llama.cpp glm-dsa)"), (0, "neox (old default)")]:
    lib.set_rope_style(style)
    lg = run_gated(name)
    rank = int((lg > lg[ORACLE_TOP1]).sum())
    top5 = torch.topk(lg, 5).indices.tolist()
    ov = len(ORACLE_TOP5 & set(top5))
    txt = [tok.decode([i]).strip()[:14] for i in top5]
    results[name] = (rank, ov, txt)
    print(f"[{name}] rank(Paris) = {rank:,} of 154,880   overlap = {ov}/5   top5 = {txt}\n")

print("=" * 80)
for name, (rank, ov, txt) in results.items():
    print(f"{name:36s} rank {rank:>7,}  overlap {ov}/5  {txt}")
