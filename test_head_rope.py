"""Two remaining suspects: the Q6_K LM head, and the RoPE convention.

HEAD: the output matrix gets the same structure test the embedding passed. If the
Q6_K dequant were wrong there would be no semantic structure in its rows.

ROPE: at pos=0 NeoX and GPT-J are both the identity, so the earlier cos=1.000000
proved nothing. This drives real positions and asks which convention actually makes
the model predict better.
"""
import sys, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, dequant_q2k_fast
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

def cs(a, b):
    a, b = a.float(), b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

# ---------- 1. LM head structure (same probe the embedding passed) ----------
print("=== 1. LM HEAD (Q6_K) SEMANTIC STRUCTURE ===")
W = eng._out_w                                   # [vocab, hidden] fp16 on GPU
def hrow(w):
    return W[tok.encode(w)[0]].float()
PAIRS = [((" dog", " cat"), "rel"), ((" king", " queen"), "rel"),
         ((" Monday", " Tuesday"), "rel"), ((" dog", " economics"), "unrel"),
         ((" king", " tomato"), "unrel"), ((" Monday", " helium"), "unrel")]
rel, unr = [], []
for (a, b), k in PAIRS:
    c = cs(hrow(a), hrow(b))
    (rel if k == "rel" else unr).append(c)
    print(f"  {a!r:11s} vs {b!r:11s} cos={c:+.4f}  [{k}]")
mr, mu = np.mean(rel), np.mean(unr)
print(f"  related {mr:+.4f}  unrelated {mu:+.4f}  separation {mr-mu:+.4f}  "
      f"{'HEAD OK' if mr-mu > 0.05 else 'HEAD SUSPECT'}")

# ---------- 2. head vs embedding alignment ----------
print("\n=== 2. HEAD ROW vs EMBEDDING ROW (untied, but should correlate) ===")
for w in [" dog", " the", "Paris"]:
    t = tok.encode(w)[0]
    e = torch.from_numpy(np.asarray(
        dequant_q2k_fast(bytes(eng.emb_data[t]), 6144), dtype=np.float32).copy()).cuda()
    print(f"  {w!r:8s} cos(emb, head) = {cs(e, W[t].float()):+.4f}")

# ---------- 3. ROPE A/B at real positions ----------
print("\n=== 3. ROPE CONVENTION A/B (needs pos>0 to differ) ===")
PROMPTS = [("The capital of France is", "paris"), ("Once upon a", "time"),
           ("1, 2, 3, 4,", "5"), ("The quick brown fox jumps over the lazy", "dog")]

def score_all():
    tot, hit = 0, 0
    for p, want in PROMPTS:
        ids = tok.encode(p, limit=32)
        for _ in range(8):
            eng.reset_context()
            for t in ids:
                eng.forward_single_token(t)
            lib.sync_device(); torch.cuda.synchronize()
            if eng.page_in_misses()[0] == 0:
                break
        eng.reset_context()
        for t in ids:
            lg = eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        lg = lg.squeeze().float()
        top = torch.topk(lg, 10)
        cands = [tok.decode([int(i)]).strip().lower() for i in top.indices]
        ok = any(want in c for c in cands if c)
        hit += ok; tot += 1
        print(f"    {p[:34]!r:36s} -> {cands[:5]}  {'HIT' if ok else ''}")
    return hit, tot

lib.set_rope_style.argtypes = [ctypes.c_int]
for style, name in ((0, "neox (i, i+d/2)"), (1, "gptj (2i, 2i+1)")):
    lib.set_rope_style(style)
    print(f"  --- rope = {name} ---")
    h, t = score_all()
    print(f"    hits {h}/{t}")
