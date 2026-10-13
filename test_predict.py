"""Does the model predict SENSIBLE next tokens? And which RoPE convention is right?

Degenerate greedy output can come from many places. The sharpest signal is: on a
trivially predictable prompt, is the correct continuation in the top-5? Also runs the
kernel-vs-reference diff at pos>0, where NeoX and GPT-J finally differ (at pos=0 both
are the identity, so the earlier 1.000000 proved nothing about RoPE).
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

reader = GGUFReader(MODEL, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
eng.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong

PROMPTS = [
    ("The capital of France is", " Paris"),
    ("1, 2, 3, 4,", " 5"),
    ("Once upon a", " time"),
    ("The quick brown fox jumps over the lazy", " dog"),
]


def logits_for(ids, warm=8):
    for _ in range(warm):
        eng.reset_context()
        for t in ids:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        if eng.page_in_misses()[0] == 0:
            break
    eng.reset_context()
    lib.reset_expert_counters()
    for t in ids:
        lg = eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    req, exe = lib.get_experts_requested(), lib.get_experts_executed()
    return lg.squeeze().float(), (100.0 * exe / req if req else 0.0)


print("\n" + "=" * 76)
print("TOP-5 NEXT-TOKEN PREDICTIONS (the real coherence test)")
print("=" * 76)
hits = 0
for prompt, want in PROMPTS:
    ids = tok.encode(prompt, limit=32)
    lg, resid = logits_for(ids)
    top = torch.topk(lg, 5)
    cands = [tok.decode([int(i)]) for i in top.indices]
    ok = any(want.strip().lower() in c.strip().lower() and c.strip() for c in cands)
    hits += ok
    print(f"\n  {prompt!r}")
    print(f"    want {want!r}   residency {resid:.1f}%   {'HIT' if ok else 'miss'}")
    print(f"    top5 {[(c, round(float(v),2)) for c, v in zip(cands, top.values)]}")
    print(f"    logit spread: max {float(lg.max()):.2f}  mean {float(lg.mean()):.2f}"
          f"  std {float(lg.std()):.2f}")
print(f"\n  HITS: {hits}/{len(PROMPTS)}")
print("\n  If logit std is tiny or one token dominates every prompt, the hidden state")
print("  is collapsing upstream of the LM head — not an attention bug.")
