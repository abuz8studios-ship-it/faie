"""Trace the residual stream layer by layer to localize where it goes wrong.

Logits are varied and non-collapsed but wrong, so something is subtly off rather than
catastrophically broken. A healthy transformer grows ||x|| smoothly. A layer that
multiplies or annihilates the norm is the culprit.

Also isolates embedding + LM head: feeding a token's own embedding straight to the
head should rank that token (or close neighbours) highly in most models.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, HIDDEN_DIM
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

# ---------- A. embedding + LM head in isolation ----------
print("\n=== A. EMBEDDING -> LM HEAD (no layers) ===")
print("   a token's own embedding should rank itself / near-neighbours highly")
from faie_generate import dequant_q2k_fast
for probe in ["Paris", " the", " dog"]:
    ids = tok.encode(probe, limit=1)
    if not ids:
        continue
    t = ids[0]
    row = bytes(eng.emb_data[t])
    vec = dequant_q2k_fast(row, eng.emb_hidden)
    x = torch.from_numpy(vec.copy()).to("cuda:0")
    eng.cmv.rms_norm(x, eng._output_norm_f32, eng._norm_buf, HIDDEN_DIM)
    lg = F.linear(eng._norm_buf.half().unsqueeze(0), eng._out_w).squeeze().float()
    top = torch.topk(lg, 5)
    rank = int((lg > lg[t]).sum())
    print(f"  {probe!r} (id {t}) -> self-rank {rank}"
          f"  top5 {[tok.decode([int(i)]) for i in top.indices]}")

# ---------- B. residual norm per layer ----------
print("\n=== B. RESIDUAL NORM THROUGH 78 LAYERS ===")
ids = tok.encode("The capital of France is", limit=32)
for _ in range(8):
    eng.reset_context()
    for t in ids:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    if eng.page_in_misses()[0] == 0:
        break

eng.reset_context()
for t in ids[:-1]:
    eng.forward_single_token(t)
lib.sync_device(); torch.cuda.synchronize()

# manual final-token pass with instrumentation
row = bytes(eng.emb_data[ids[-1]])
vec = dequant_q2k_fast(row, eng.emb_hidden)
eng._x_buf.copy_(torch.from_numpy(vec.copy()))
pos = eng.pos
print(f"  embed          |x| = {float(eng._x_buf.norm()):8.3f}")
prev = float(eng._x_buf.norm())
for li in range(78):
    before = float(eng._x_buf.norm())
    eng._mla_attention(eng._x_buf, li, pos)
    lib.sync_device(); torch.cuda.synchronize()
    after_attn = float(eng._x_buf.norm())
    if li < 3:
        eng._dense_ffn_flat(eng._x_buf, li)
    else:
        eng._moe_ffn_warm(eng._x_buf, li)
    lib.sync_device(); torch.cuda.synchronize()
    after_ffn = float(eng._x_buf.norm())
    if li < 6 or li % 12 == 0 or li > 74:
        print(f"  L{li:<3d} attn {before:9.3f} -> {after_attn:9.3f}"
              f"   ffn -> {after_ffn:9.3f}"
              f"   (x{after_ffn/max(before,1e-9):.3f})")
    if not np.isfinite(after_ffn):
        print(f"  !! non-finite at layer {li}")
        break
print(f"  final          |x| = {float(eng._x_buf.norm()):8.3f}")
eng.cmv.rms_norm(eng._x_buf, eng._output_norm_f32, eng._norm_buf, HIDDEN_DIM)
lg = F.linear(eng._norm_buf.half().unsqueeze(0), eng._out_w).squeeze().float()
top = torch.topk(lg, 5)
print(f"  top5 {[tok.decode([int(i)]) for i in top.indices]}")
