"""Attention is mean-pooling: entropy 2.069 vs uniform 2.079. The scores have no
discriminative spread. score = (qc . c + q_rope . k_rope) / sqrt(256), so split it
and see which half is dead, and what spread each half actually has.

For a softmax to select, the SCALED scores need a spread of several units. If the
spread is ~0.01 the result is uniform no matter how correct the math is.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import (FAIEInference, dequant_any, dequant_q2k_fast, HIDDEN_DIM,
                           N_HEADS, QK_NOPE_DIM, QK_ROPE_DIM, KV_LORA_RANK,
                           V_HEAD_DIM, ROPE_BASE)
from expert_locality import GGUFTokenizer

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
DEV = "cuda:0"
EPS = 1e-5

reader = GGUFReader(MODEL, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)


def rms(x, w):
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w


def probe(L, n_ctx=8):
    def W(nm):
        t = eng._get_tensor(f"blk.{L}.{nm}.weight")
        return torch.from_numpy(np.ascontiguousarray(dequant_any(t))).to(DEV).float()
    w_qa, w_qb = W("attn_q_a"), W("attn_q_b")
    w_kva, w_kb = W("attn_kv_a_mqa"), W("attn_k_b")
    n_att, n_qa = eng.attn_norms[L].float(), eng.attn_q_norms[L].float()
    n_kv = eng.attn_kv_norms[L].float()

    # a few different "token" hidden states, as a stand-in for real context
    torch.manual_seed(0)
    xs = [torch.randn(HIDDEN_DIM, device=DEV) * 0.3 for _ in range(n_ctx)]

    C, KR = [], []
    for x in xs:
        h = rms(x, n_att)
        kv = w_kva @ h
        C.append(rms(kv[:KV_LORA_RANK], n_kv))
        KR.append(kv[KV_LORA_RANK:])
    C, KR = torch.stack(C), torch.stack(KR)

    h = rms(xs[-1], n_att)
    q = (w_qb @ rms(w_qa @ h, n_qa)).view(N_HEADS, QK_NOPE_DIM + QK_ROPE_DIM)
    q_nope, q_rope = q[:, :QK_NOPE_DIM], q[:, QK_NOPE_DIM:]
    qc = torch.einsum("hoi,hi->ho", w_kb, q_nope)

    s_nope = qc @ C.T
    s_rope = q_rope @ KR.T
    scale = 1.0 / (QK_NOPE_DIM + QK_ROPE_DIM) ** 0.5
    tot = (s_nope + s_rope) * scale
    p = torch.softmax(tot, -1)
    ent = float(-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1).mean())
    return dict(
        qc=float(qc.norm(dim=-1).mean()), c=float(C.norm(dim=-1).mean()),
        qr=float(q_rope.norm(dim=-1).mean()), kr=float(KR.norm(dim=-1).mean()),
        nope_spread=float(s_nope.std(dim=-1).mean()),
        rope_spread=float(s_rope.std(dim=-1).mean()),
        scaled_spread=float(tot.std(dim=-1).mean()),
        ent=ent, uni=float(np.log(n_ctx)))


print(f"{'L':>3} {'|qc|':>9} {'|c|':>8} {'|q_rope|':>9} {'|k_rope|':>9}"
      f" {'nope sd':>9} {'rope sd':>9} {'scaled sd':>10} {'entropy':>8} {'uniform':>8}")
print("-" * 100)
for L in (0, 4, 8, 12, 30, 60, 77):
    r = probe(L)
    print(f"{L:>3} {r['qc']:>9.3f} {r['c']:>8.3f} {r['qr']:>9.3f} {r['kr']:>9.3f}"
          f" {r['nope_spread']:>9.3f} {r['rope_spread']:>9.3f} {r['scaled_spread']:>10.4f}"
          f" {r['ent']:>8.4f} {r['uni']:>8.4f}")

print("\nA selective softmax needs scaled-score spread of roughly 1-5.")
print("If 'scaled sd' is ~0.01-0.1, attention mean-pools no matter how correct")
print("the arithmetic is — and the culprit is whichever of nope/rope has no spread.")
