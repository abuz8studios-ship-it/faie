"""THE LAST UNVERIFIED PIECE: MLA with real history (pos > 0).

Everything else now checks out at cos 1.000000 — matvec, single expert, whole MoE
layer, router 8/8, tokenizer, embeddings, and MLA *at pos=0*. But pos=0 has an empty
cache: it never exercises the KV cache, positional RoPE, or causal accumulation.
At pos=0 RoPE is also the identity, so that test proved nothing about position.

This drives 6 sequential positions through both the kernel and a PyTorch reference,
keeping their histories in lockstep, and reports cosine per position. The first
position that diverges is the bug.
"""
import sys, ctypes, io
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import (FAIEInference, dequant_any, HIDDEN_DIM, N_HEADS,
                           QK_NOPE_DIM, QK_ROPE_DIM, KV_LORA_RANK, V_HEAD_DIM,
                           ROPE_BASE)

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p
DEV = "cuda:0"
L = 0            # layer 0 keeps it clean: no MoE, no expert residency confound
NPOS = 6
EPS = 1e-5


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


reader = GGUFReader(MODEL, "r")
lib = load_kernel()
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
lib.set_rope_style.argtypes = [ctypes.c_int]
lib.set_score_scale.argtypes = [ctypes.c_float]


def W(nm):
    t = eng._get_tensor(f"blk.{L}.{nm}.weight")
    return torch.from_numpy(np.ascontiguousarray(dequant_any(t))).to(DEV).float()


w_qa, w_qb = W("attn_q_a"), W("attn_q_b")
w_kva, w_kb, w_vb, w_out = W("attn_kv_a_mqa"), W("attn_k_b"), W("attn_v_b"), W("attn_output")
n_att = eng.attn_norms[L].float()
n_qa = eng.attn_q_norms[L].float()
n_kv = eng.attn_kv_norms[L].float()


def rms(v, ww):
    return v / torch.sqrt((v * v).mean(-1, keepdim=True) + EPS) * ww


def ref_step(x, pos, C, KR, style):
    h = rms(x, n_att)
    q = (w_qb @ rms(w_qa @ h, n_qa)).view(N_HEADS, QK_NOPE_DIM + QK_ROPE_DIM)
    q_nope, q_rope = q[:, :QK_NOPE_DIM], q[:, QK_NOPE_DIM:].clone()
    kv = w_kva @ h
    c = rms(kv[:KV_LORA_RANK], n_kv)
    k_rope = kv[KV_LORA_RANK:].clone()

    d = QK_ROPE_DIM
    i = torch.arange(d // 2, device=DEV, dtype=torch.float32)
    th = pos * (ROPE_BASE ** (-(2 * i) / d))
    cs, sn = torch.cos(th), torch.sin(th)

    def rope(v):
        if style == 0:                              # neox: i pairs with i+d/2
            a, b = v[..., :d // 2], v[..., d // 2:]
            return torch.cat([a * cs - b * sn, a * sn + b * cs], -1)
        a, b = v[..., 0::2], v[..., 1::2]           # gptj: 2i pairs with 2i+1
        ra, rb = a * cs - b * sn, a * sn + b * cs
        o = torch.empty_like(v); o[..., 0::2], o[..., 1::2] = ra, rb
        return o

    q_rope, k_rope = rope(q_rope), rope(k_rope)
    C = torch.cat([C, c.unsqueeze(0)], 0) if C is not None else c.unsqueeze(0)
    KR = torch.cat([KR, k_rope.unsqueeze(0)], 0) if KR is not None else k_rope.unsqueeze(0)

    qc = torch.einsum("hoi,hi->ho", w_kb, q_nope)
    s = (qc @ C.T + q_rope @ KR.T) / (QK_NOPE_DIM + QK_ROPE_DIM) ** 0.5
    a = torch.softmax(s, -1)
    oc = a @ C
    v = torch.einsum("hoi,hi->ho", w_vb, oc)
    return x + w_out @ v.reshape(-1), C, KR, a


torch.manual_seed(0)
XS = [(torch.randn(HIDDEN_DIM, device=DEV) * 0.05).float() for _ in range(NPOS)]

for style, name in ((0, "neox"), (1, "gptj")):
    lib.set_rope_style(style)
    lib.set_score_scale(ctypes.c_float(1.0))
    eng.reset_context()
    eng._c_cache[L].zero_(); eng._kr_cache[L].zero_()
    C = KR = None
    print(f"\n=== rope={name} ===")
    print(f"  {'pos':>3} {'cos(attn delta)':>17} {'|ref|':>9} {'|ker|':>9} {'ratio':>8}  attn-entropy")
    for p in range(NPOS):
        x = XS[p]
        xr, C, KR, a = ref_step(x.clone(), p, C, KR, style)
        eng._x_buf.copy_(x)
        eng._mla_attention(eng._x_buf, L, p)
        lib.sync_device(); torch.cuda.synchronize()
        dk = eng._x_buf - x
        dr = xr - x
        ent = float(-(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1).mean())
        print(f"  {p:>3} {cos(dr, dk):>17.6f} {float(dr.norm()):>9.4f} {float(dk.norm()):>9.4f}"
              f" {float(dk.norm()/(dr.norm()+1e-12)):>8.4f}  {ent:.4f} / {np.log(p+1):.4f}")

lib.set_rope_style(0)
print("\npos=0 must be 1.000000 (already known). The FIRST pos>0 that drops is the bug.")
