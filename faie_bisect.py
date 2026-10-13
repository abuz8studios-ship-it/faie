"""THE BISECT — which component inflates the residual stream 760x?

Known: ||x|| goes 0.574 -> 435 over 78 layers, and disabling ALL routed experts
changes it by 3%. So the culprit is attention or the shared expert. This nulls each
one independently and re-traces, and also reports the PER-LAYER contribution of each
component so the exact layer where it blows up is visible.

  attn_delta = ||x_after_attn - x_before||     (attention's contribution)
  ffn_delta  = ||x_after_ffn  - x_after_attn|| (shared + routed experts)
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, dequant_q2k_fast, HIDDEN_DIM
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
lib.set_bisect.argtypes = [ctypes.c_int, ctypes.c_int]

IDS = tok.encode("The capital of France is", limit=32)
for _ in range(8):
    eng.reset_context()
    for t in IDS:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    if eng.page_in_misses()[0] == 0:
        break
print(f"resident {len(eng._expert_cache):,}\n")


def trace(detail=False):
    """Per-layer attention and FFN contributions for the final token."""
    eng.reset_context()
    for t in IDS[:-1]:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    vec = dequant_q2k_fast(bytes(eng.emb_data[IDS[-1]]), eng.emb_hidden)
    eng._x_buf.copy_(torch.from_numpy(vec.copy()))
    pos = eng.pos
    rows = []
    for li in range(78):
        b = eng._x_buf.clone()
        eng._mla_attention(eng._x_buf, li, pos)
        lib.sync_device(); torch.cuda.synchronize()
        a = eng._x_buf.clone()
        if li < 3:
            eng._dense_ffn_flat(eng._x_buf, li)
        else:
            eng._moe_ffn_warm(eng._x_buf, li)
        lib.sync_device(); torch.cuda.synchronize()
        rows.append((li, float(b.norm()), float((a - b).norm()),
                     float((eng._x_buf - a).norm()), float(eng._x_buf.norm())))
    if detail:
        print(f"    {'L':>3} {'|x| in':>9} {'attn +':>9} {'ffn +':>9} {'|x| out':>9}")
        for li, bn, ad, fd, on in rows:
            if li < 14 or li % 16 == 0 or li > 75:
                print(f"    {li:>3} {bn:>9.3f} {ad:>9.3f} {fd:>9.3f} {on:>9.3f}")
    eng.cmv.rms_norm(eng._x_buf, eng._output_norm_f32, eng._norm_buf, HIDDEN_DIM)
    lg = F.linear(eng._norm_buf.half().unsqueeze(0), eng._out_w).squeeze().float()
    top = [tok.decode([int(i)]).strip() for i in torch.topk(lg, 5).indices]
    return rows, top


CONFIGS = [
    (0, 0, "baseline (all on)"),
    (1, 0, "shared expert OFF"),
    (0, 1, "attention OFF"),
    (1, 1, "both OFF (routed experts only)"),
]

print(f"{'config':<34} {'|x| L5':>9} {'L12':>9} {'L40':>9} {'L77':>9}   top5")
print("-" * 108)
results = {}
for ns, na, name in CONFIGS:
    lib.set_bisect(ns, na)
    rows, top = trace()
    results[name] = rows
    print(f"{name:<34} {rows[5][4]:>9.3f} {rows[12][4]:>9.3f} {rows[40][4]:>9.3f}"
          f" {rows[77][4]:>9.3f}   {top}")

print("\n=== PER-LAYER DETAIL, baseline ===")
lib.set_bisect(0, 0)
trace(detail=True)

# Where does attention's contribution first exceed the incoming residual?
rows = results["baseline (all on)"]
print("\n=== DIAGNOSIS ===")
bad = [(li, bn, ad) for li, bn, ad, fd, on in rows if ad > bn * 0.5]
if bad:
    li, bn, ad = bad[0]
    print(f"  attention contribution first exceeds 50% of the residual at L{li}:"
          f" |x|={bn:.3f} but attn adds {ad:.3f}")
else:
    print("  attention never dominates the residual")
tot_a = sum(r[2] for r in rows)
tot_f = sum(r[3] for r in rows)
print(f"  summed contributions over 78 layers: attention {tot_a:.1f}   ffn {tot_f:.1f}")
lib.set_bisect(0, 0)
