"""Do the cached KV latents actually DIFFER between positions?

Chain of evidence: making the softmax selective (32x score scale) changed nothing,
and disabling attention entirely changed the output but not its quality. That is what
you would see if every cached position held nearly the SAME latent — then attending
sharply and mean-pooling give the same answer, and the score scale is irrelevant.

This measures pairwise cosine between c_cache rows across positions. Near 1.0 means
attention has nothing to choose between, and the bug is in the kv_a path, not the
softmax.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, KV_LORA_RANK, QK_ROPE_DIM, HIDDEN_DIM
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

# Deliberately heterogeneous prompt so the positions SHOULD differ a lot.
IDS = tok.encode("Paris zebra 7 quantum ocean the France capital", limit=32)
print(f"prompt tokens: {len(IDS)}  ids {IDS}\n")
for _ in range(8):
    eng.reset_context()
    for t in IDS:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    if eng.page_in_misses()[0] == 0:
        break

eng.reset_context()
for t in IDS:
    eng.forward_single_token(t)
lib.sync_device(); torch.cuda.synchronize()
T = eng.pos
print(f"cached positions = {T}\n")


def pairwise(mat):
    m = mat / (mat.norm(dim=-1, keepdim=True) + 1e-12)
    g = m @ m.T
    iu = torch.triu_indices(g.shape[0], g.shape[0], offset=1)
    v = g[iu[0], iu[1]]
    return float(v.mean()), float(v.min()), float(v.max())


print(f"{'L':>3} | {'c latent: mean':>14} {'min':>8} {'max':>8} | "
      f"{'k_rope: mean':>13} {'min':>8} {'max':>8} | {'|c| spread':>11}")
print("-" * 92)
for L in (0, 4, 8, 12, 24, 40, 60, 77):
    C = eng._c_cache[L][: T * KV_LORA_RANK].view(T, KV_LORA_RANK).float()
    K = eng._kr_cache[L][: T * QK_ROPE_DIM].view(T, QK_ROPE_DIM).float()
    cm, cn, cx = pairwise(C)
    km, kn, kx = pairwise(K)
    norms = C.norm(dim=-1)
    spread = float(norms.std() / (norms.mean() + 1e-12))
    print(f"{L:>3} | {cm:>14.5f} {cn:>8.5f} {cx:>8.5f} | "
          f"{km:>13.5f} {kn:>8.5f} {kx:>8.5f} | {spread:>11.5f}")

print("\nIf 'c latent mean' is ~0.99+, every position holds the same vector and")
print("attention CANNOT discriminate — the fault is upstream in the kv_a path.")
print("Healthy values for unrelated tokens would be roughly 0.0-0.6.")

# Same probe on the residual stream itself, to see if positions differ at all.
print("\n=== residual x per position (recomputed) ===")
xs = []
eng.reset_context()
from faie_generate import dequant_q2k_fast
for t in IDS:
    eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    xs.append(eng._x_buf.clone())
X = torch.stack(xs).float()
m, n, x = pairwise(X)
print(f"  final hidden state across {len(IDS)} positions: mean cos {m:.5f} "
      f"min {n:.5f} max {x:.5f}")
print("  (if ~1.0, every token produces the same hidden state -> nothing downstream")
print("   can distinguish them, and that is the whole bug)")
