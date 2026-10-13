"""Sweep the attention score scale and watch entropy + predictions together.

Hypothesis: attention mean-pools because scaled scores span ~0.03-0.64 when a
selective softmax needs 1-5. If raising the multiplier drops entropy AND starts
producing sensible next tokens, the scale was the bug. If entropy drops but the
predictions stay junk, the scale is a symptom and the real fault is upstream.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, N_HEADS, MAX_SEQ
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
lib.set_score_scale.argtypes = [ctypes.c_float]

TESTS = [("The capital of France is", "paris"),
         ("Once upon a", "time"),
         ("The quick brown fox jumps over the lazy", "dog"),
         ("1, 2, 3, 4,", "5")]
ENC = [(tok.encode(p, limit=32), w) for p, w in TESTS]

for ids, _ in ENC:
    for _ in range(8):
        eng.reset_context()
        for t in ids:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        if eng.page_in_misses()[0] == 0:
            break
print(f"resident {len(eng._expert_cache):,}\n")

print(f"{'xscale':>7} {'entropy':>8} {'uniform':>8} {'hits':>5}   predictions")
print("-" * 104)
for xs in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    lib.set_score_scale(ctypes.c_float(xs))
    hits, ents, preds = 0, [], []
    for ids, want in ENC:
        eng.reset_context()
        for t in ids:
            lg = eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        n_ctx = eng.pos
        w = eng._score_buf.view(N_HEADS, MAX_SEQ)[:, :max(n_ctx, 1)].float()
        ents.append(float(-(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1).mean()))
        top = [tok.decode([int(i)]).strip().lower() for i in torch.topk(lg.squeeze().float(), 5).indices]
        if any(want in c for c in top if c):
            hits += 1
        preds.append(top[0])
    uni = float(np.log(max(eng.pos, 2)))
    print(f"{xs:>7.1f} {np.mean(ents):>8.4f} {uni:>8.4f} {hits:>5}   {preds}")

lib.set_score_scale(ctypes.c_float(1.0))
print("\nEntropy should fall as xscale rises. The question is whether HITS follow.")
