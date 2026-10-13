"""Which k_b / v_b orientation makes attention actually SELECT?

Kernel and reference agree at cos 1.000000 everywhere, and BOTH mean-pool
(entropy 1.7907 vs uniform 1.7918). So the shared assumption is wrong, not the code.

Strong prior that needs no oracle: trained attention is selective. Sweep the
plausible interpretations of attn_k_b [192,512,64] and attn_v_b [512,256,64] and
score each by how far below the uniform ceiling its entropy sits. Uses REAL token
hidden states, not random vectors, because random inputs have no structure to attend
to and would look flat under any orientation.
"""
import sys, ctypes, io, itertools
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
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

# Real token embeddings — structured input, so selectivity is meaningful.
IDS = tok.encode("The capital of France is Paris and the capital of Japan is", limit=32)
X = torch.stack([torch.from_numpy(np.asarray(
    dequant_q2k_fast(bytes(eng.emb_data[t]), eng.emb_hidden), dtype=np.float32).copy())
    for t in IDS]).to(DEV)
T = X.shape[0]
print(f"{T} real token embeddings, ids={IDS}\n")


def rms(v, w):
    return v / torch.sqrt((v * v).mean(-1, keepdim=True) + EPS) * w


def probe(L):
    def W(nm):
        t = eng._get_tensor(f"blk.{L}.{nm}.weight")
        return torch.from_numpy(np.ascontiguousarray(dequant_any(t))).to(DEV).float()
    w_qa, w_qb, w_kva = W("attn_q_a"), W("attn_q_b"), W("attn_kv_a_mqa")
    w_kb = W("attn_k_b")                       # numpy view: [64, 512, 192]
    n_att, n_qa, n_kv = (eng.attn_norms[L].float(), eng.attn_q_norms[L].float(),
                         eng.attn_kv_norms[L].float())

    H = rms(X, n_att)                          # [T, 6144]
    Q = (H @ w_qa.T)
    Q = rms(Q, n_qa) @ w_qb.T                  # [T, 16384]
    Q = Q.view(T, N_HEADS, QK_NOPE_DIM + QK_ROPE_DIM)
    KV = H @ w_kva.T                           # [T, 576]
    C = rms(KV[:, :KV_LORA_RANK], n_kv)        # [T, 512]
    KRp = KV[:, KV_LORA_RANK:]                 # [T, 64]

    d = QK_ROPE_DIM
    i = torch.arange(d // 2, device=DEV, dtype=torch.float32)
    pos = torch.arange(T, device=DEV, dtype=torch.float32).unsqueeze(1)
    th = pos * (ROPE_BASE ** (-(2 * i) / d))
    cs, sn = torch.cos(th), torch.sin(th)

    def rope(v):                               # neox, v: [T, ..., 64]
        a, b = v[..., :d // 2], v[..., d // 2:]
        c2 = cs.view(T, *([1] * (v.dim() - 2)), d // 2)
        s2 = sn.view(T, *([1] * (v.dim() - 2)), d // 2)
        return torch.cat([a * c2 - b * s2, a * s2 + b * c2], -1)

    Qr = rope(Q[:, :, QK_NOPE_DIM:])
    Qn = Q[:, :, :QK_NOPE_DIM]
    KR = rope(KRp)

    scale = 1.0 / (QK_NOPE_DIM + QK_ROPE_DIM) ** 0.5
    out = {}
    # k_b memory is [h][a][b] with a=512, b=192 under my current read.
    VARIANTS = {
        "A: qc = kb[h] @ q_nope   (in=192,out=512)  <-- current":
            lambda: torch.einsum("hoi,thi->tho", w_kb, Qn),
        "B: qc = kb[h].T @ q_nope (in=512,out=192 then dot k_nope)":
            None,   # handled specially below
    }
    qcA = torch.einsum("hoi,thi->tho", w_kb, Qn)          # [T,H,512]
    sA = torch.einsum("tho,so->ths", qcA, C)              # query t, key s
    # Variant B: treat kb[h] as [512->192] decompressing c into k_nope, then dot
    Kn = torch.einsum("hoi,so->shi", w_kb, C)             # [T_keys, H, 192]
    sB = torch.einsum("thi,shi->ths", Qn, Kn)

    for tag, s_nope in (("A absorb (current)", sA), ("B decompress-k", sB)):
        s = (s_nope + torch.einsum("thd,sd->ths", Qr, KR)) * scale
        mask = torch.tril(torch.ones(T, T, device=DEV)).bool()
        s = s.masked_fill(~mask.unsqueeze(1), -1e30)
        a = torch.softmax(s, -1)
        # entropy of the LAST query row, vs uniform ceiling over T keys
        ent = float(-(a[-1].clamp_min(1e-12) * a[-1].clamp_min(1e-12).log()).sum(-1).mean())
        sd = float(s[-1][:, :T].std())
        out[tag] = (ent, np.log(T), sd)
    return out


print(f"{'layer':>6} {'variant':>22} {'entropy':>9} {'uniform':>9} {'gap':>8} {'score sd':>9}")
print("-" * 74)
for L in (0, 8, 20, 40, 60, 77):
    r = probe(L)
    for tag, (ent, uni, sd) in r.items():
        print(f"{L:>6} {tag:>22} {ent:>9.4f} {uni:>9.4f} {uni-ent:>8.4f} {sd:>9.4f}")
    print()
print("A larger 'gap' = more selective. If B is consistently far more selective than A,")
print("the absorption orientation is the bug.")
