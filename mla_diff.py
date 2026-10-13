"""Numerically diff the MLA CUDA kernel against a PyTorch reference, layer 0.

Guessing at RoPE conventions is how you burn hours. This computes the same layer two
ways from the same input and reports cosine at each stage, so the first stage that
diverges IS the bug.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import (FAIEInference, dequant_any, HIDDEN_DIM, N_HEADS,
                           QK_NOPE_DIM, QK_ROPE_DIM, KV_LORA_RANK, V_HEAD_DIM,
                           Q_LORA_RANK, ROPE_BASE, MAX_SEQ)

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p
L = 0
DEV = "cuda:0"


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


print("Loading...")
reader = GGUFReader(MODEL, "r")
lib = load_kernel()
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)

# --- reference weights in float32 on GPU -----------------------------------
def W(name):
    t = eng._get_tensor(f"blk.{L}.{name}.weight")
    a = dequant_any(t)                       # (…, out, in)
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV).float()

w_qa   = W("attn_q_a")            # [2048, 6144]
w_qb   = W("attn_q_b")            # [16384, 2048]
w_kva  = W("attn_kv_a_mqa")       # [576, 6144]
w_kb   = W("attn_k_b")            # [64, 512, 192]
w_vb   = W("attn_v_b")            # [64, 256, 512]
w_out  = W("attn_output")         # [6144, 16384]
n_att  = eng.attn_norms[L].float()
n_qa   = eng.attn_q_norms[L].float()
n_kv   = eng.attn_kv_norms[L].float()
print(f"  qa{list(w_qa.shape)} qb{list(w_qb.shape)} kva{list(w_kva.shape)} "
      f"kb{list(w_kb.shape)} vb{list(w_vb.shape)} out{list(w_out.shape)}")

EPS = 1e-5
def rms(x, w):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w


def reference(x, pos, c_hist, kr_hist, rope="neox"):
    h = rms(x, n_att)
    qa = rms(w_qa @ h, n_qa)
    q  = (w_qb @ qa).view(N_HEADS, QK_NOPE_DIM + QK_ROPE_DIM)
    q_nope, q_rope = q[:, :QK_NOPE_DIM], q[:, QK_NOPE_DIM:].clone()
    kv = w_kva @ h
    c, k_rope = rms(kv[:KV_LORA_RANK], n_kv), kv[KV_LORA_RANK:].clone()

    d = QK_ROPE_DIM
    idx = torch.arange(d // 2, device=DEV, dtype=torch.float32)
    inv = ROPE_BASE ** (-(2 * idx) / d)
    th = pos * inv
    cs, sn = torch.cos(th), torch.sin(th)

    def rope_apply(v):                       # v: [..., d]
        if rope == "neox":
            a, b = v[..., :d // 2], v[..., d // 2:]
            return torch.cat([a * cs - b * sn, a * sn + b * cs], dim=-1)
        else:                                # gptj / interleaved pairs
            a, b = v[..., 0::2], v[..., 1::2]
            ra, rb = a * cs - b * sn, a * sn + b * cs
            o = torch.empty_like(v)
            o[..., 0::2], o[..., 1::2] = ra, rb
            return o

    q_rope = rope_apply(q_rope)
    k_rope = rope_apply(k_rope)

    C  = torch.cat([c_hist,  c.unsqueeze(0)],      0) if c_hist  is not None else c.unsqueeze(0)
    KR = torch.cat([kr_hist, k_rope.unsqueeze(0)], 0) if kr_hist is not None else k_rope.unsqueeze(0)

    qc = torch.einsum("hoi,hi->ho", w_kb, q_nope)             # [64,512] absorbed
    s  = qc @ C.T + q_rope @ KR.T                             # [64, T]
    s  = s / (QK_NOPE_DIM + QK_ROPE_DIM) ** 0.5
    a  = torch.softmax(s, dim=-1)
    oc = a @ C                                                # [64,512]
    v  = torch.einsum("hoi,hi->ho", w_vb, oc)                 # [64,256]
    o  = w_out @ v.reshape(-1)
    return x + o, C, KR, dict(q=q, c=c, k_rope=k_rope, qc=qc, s=s, a=a, oc=oc, v=v, o=o)


torch.manual_seed(0)
x0 = (torch.randn(HIDDEN_DIM, device=DEV) * 0.02).float()

print("\n=== single token, pos=0 ===")
for style in ("neox", "gptj"):
    xr, _, _, ref = reference(x0.clone(), 0, None, None, rope=style)

    xk = x0.clone()
    eng.reset_context()
    eng._mla_attention(xk, L, 0)
    lib.sync_device(); torch.cuda.synchronize()

    print(f"  rope={style:5s}  attn-out cos = {cos(xr - x0, xk - x0):+.6f}"
          f"   |ref|={float((xr-x0).norm()):.4f}  |ker|={float((xk-x0).norm()):.4f}")

# Stage-by-stage against the kernel's own scratch buffers (neox path)
xk = x0.clone()
eng.reset_context()
eng._mla_attention(xk, L, 0)
lib.sync_device(); torch.cuda.synchronize()
_, _, _, ref = reference(x0.clone(), 0, None, None, rope="neox")

k_q  = eng._qb_buf[: N_HEADS * (QK_NOPE_DIM + QK_ROPE_DIM)].view(N_HEADS, -1)
k_c  = eng._c_cache[L][:KV_LORA_RANK]
k_kr = eng._kr_cache[L][:QK_ROPE_DIM]
k_qc = eng._qc_buf[: N_HEADS * KV_LORA_RANK].view(N_HEADS, KV_LORA_RANK)
k_oc = eng._oc_buf[: N_HEADS * KV_LORA_RANK].view(N_HEADS, KV_LORA_RANK)
k_v  = eng._head_buf[: N_HEADS * V_HEAD_DIM].view(N_HEADS, V_HEAD_DIM)

print("\n=== stage-by-stage (neox) ===")
print(f"  q  (pre-rope nope part) cos = {cos(ref['q'][:, :QK_NOPE_DIM], k_q[:, :QK_NOPE_DIM]):+.6f}")
print(f"  q_rope (post-rope)      cos = {cos(ref['q'][:, QK_NOPE_DIM:], k_q[:, QK_NOPE_DIM:]):+.6f}")
print(f"  c latent                cos = {cos(ref['c'], k_c):+.6f}")
print(f"  k_rope (post-rope)      cos = {cos(ref['k_rope'], k_kr):+.6f}")
print(f"  qc absorbed             cos = {cos(ref['qc'], k_qc):+.6f}")
print(f"  oc weighted latent      cos = {cos(ref['oc'], k_oc):+.6f}")
print(f"  v decompressed          cos = {cos(ref['v'], k_v):+.6f}")
