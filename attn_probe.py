"""Attention output tracks ||x|| at a constant ~30%. In a pre-norm block it should
be scale-INDEPENDENT, because rms_norm strips ||x|| before the projections.

Two things to check:
  1. Is the softmax actually attending, or degenerate (uniform / all mass on one slot)?
  2. Feed the SAME layer two inputs differing only by a scalar factor. If attention
     output scales with the factor, something bypasses the norm. If it is invariant,
     the growth is the model's own design and the bug is elsewhere.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import (FAIEInference, dequant_q2k_fast, HIDDEN_DIM, N_HEADS,
                           KV_LORA_RANK, QK_ROPE_DIM, MAX_SEQ)
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
lib.set_bisect(0, 0)

IDS = tok.encode("The capital of France is the city of", limit=32)
for _ in range(8):
    eng.reset_context()
    for t in IDS:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    if eng.page_in_misses()[0] == 0:
        break

# Build a real context so there are several positions to attend over.
eng.reset_context()
for t in IDS[:-1]:
    eng.forward_single_token(t)
lib.sync_device(); torch.cuda.synchronize()
pos = eng.pos
print(f"context positions = {pos + 1}  (prompt {len(IDS)} tokens)\n")

vec = dequant_q2k_fast(bytes(eng.emb_data[IDS[-1]]), eng.emb_hidden)
x_base = torch.from_numpy(vec.copy()).cuda()

print("=== 1. ATTENTION DISTRIBUTION (is the softmax doing anything?) ===")
print(f"  {'L':>3} {'ctx':>4} {'max w':>8} {'min w':>8} {'entropy':>8} {'uniform':>8}  verdict")
for li in (0, 4, 8, 12, 30, 60, 77):
    eng._x_buf.copy_(x_base)
    eng._mla_attention(eng._x_buf, li, pos)
    lib.sync_device(); torch.cuda.synchronize()
    n_ctx = pos + 1
    w = eng._score_buf.view(N_HEADS, MAX_SEQ)[:, :n_ctx].float()
    ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1).mean().item()
    uni = float(np.log(n_ctx))
    verdict = ("DEGENERATE one-hot" if ent < 0.15 * uni else
               "near-UNIFORM (not selecting)" if ent > 0.95 * uni else "selective")
    print(f"  {li:>3} {n_ctx:>4} {float(w.max()):>8.4f} {float(w.min()):>8.4f}"
          f" {ent:>8.4f} {uni:>8.4f}  {verdict}")

print("\n=== 2. SCALE INVARIANCE (pre-norm attention must ignore ||x||) ===")
print(f"  {'L':>3} {'|x|':>10} {'|attn out|':>12} {'ratio to |x|':>13}")
for li in (4, 8, 12, 40, 77):
    row = []
    for k in (1.0, 4.0, 16.0):
        eng._x_buf.copy_(x_base * k)
        before = eng._x_buf.clone()
        eng._mla_attention(eng._x_buf, li, pos)
        lib.sync_device(); torch.cuda.synchronize()
        d = float((eng._x_buf - before).norm())
        row.append((float(before.norm()), d))
    for bn, d in row:
        print(f"  {li:>3} {bn:>10.3f} {d:>12.4f} {d/bn:>13.4f}")
    inv = row[0][1] > 0 and abs(row[2][1] / row[0][1] - 1.0) < 0.25
    print(f"      -> {'SCALE-INVARIANT (correct pre-norm)' if inv else 'SCALES WITH |x| — norm is being bypassed'}"
          f"   (16x input -> {row[2][1]/max(row[0][1],1e-9):.2f}x output)\n")
