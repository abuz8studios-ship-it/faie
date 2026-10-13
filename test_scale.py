"""A/B the router weight scale against residual-norm growth and prediction quality.

Symptom: two unrelated prompts produce nearly the same top-5, and ||x|| grows 760x
across 78 layers. That is what it looks like when each layer adds a large
input-independent vector that drowns the token signal. If the expert scale is too
big, dropping it should flatten the growth curve AND change the predictions.
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
lib.set_router_scale.argtypes = [ctypes.c_float]
lib.get_router_scale.restype = ctypes.c_float

P1 = tok.encode("The capital of France is", limit=32)
P2 = tok.encode("The quick brown fox jumps over the lazy", limit=32)


def warm(ids, n=8):
    for _ in range(n):
        eng.reset_context()
        for t in ids:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        if eng.page_in_misses()[0] == 0:
            break


def run(ids):
    eng.reset_context()
    for t in ids:
        lg = eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    return lg.squeeze().float()


def trace_norm(ids):
    """Norm of the residual after every layer, for the final token."""
    eng.reset_context()
    for t in ids[:-1]:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    vec = dequant_q2k_fast(bytes(eng.emb_data[ids[-1]]), eng.emb_hidden)
    eng._x_buf.copy_(torch.from_numpy(vec.copy()))
    pos, norms = eng.pos, []
    for li in range(78):
        eng._mla_attention(eng._x_buf, li, pos)
        if li < 3:
            eng._dense_ffn_flat(eng._x_buf, li)
        else:
            eng._moe_ffn_warm(eng._x_buf, li)
        lib.sync_device(); torch.cuda.synchronize()
        norms.append(float(eng._x_buf.norm()))
    return norms


warm(P1); warm(P2)
print(f"resident {len(eng._expert_cache):,}\n")
print(f"{'scale':>6} {'|x| L5':>9} {'L20':>9} {'L40':>9} {'L77':>9}  {'top5 prompt A':<34} {'top5 prompt B'}")
print("-" * 118)
for s in (0.0, 1.0, 2.5):
    lib.set_router_scale(ctypes.c_float(s))
    n = trace_norm(P1)
    a = [tok.decode([int(i)]).strip() for i in torch.topk(run(P1), 5).indices]
    b = [tok.decode([int(i)]).strip() for i in torch.topk(run(P2), 5).indices]
    same = len(set(a) & set(b))
    print(f"{s:>6.1f} {n[5]:>9.2f} {n[20]:>9.2f} {n[40]:>9.2f} {n[77]:>9.2f}  "
          f"{str(a):<34} {b}   overlap {same}/5")
print("\nscale 0.0 disables routed experts entirely (shared expert + attention only).")
print("If predictions barely move between 0.0 and 2.5, the routed experts are not")
print("driving the output and the bug is elsewhere.")
