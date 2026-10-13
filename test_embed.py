"""Is the Q2_K embedding dequant correct, and is the tokenizer emitting real ids?

Test A (embedding -> LM head) was a weak probe because GLM has UNTIED embeddings.
This tests the embedding table on its own terms: related tokens must be closer than
unrelated ones. If the dequant were scrambled there would be no structure at all.
Also round-trips the tokenizer, since a wrong id invalidates every other test.
"""
import sys
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference, dequant_q2k_fast
from expert_locality import GGUFTokenizer

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"

reader = GGUFReader(MODEL, "r")
tok = GGUFTokenizer(reader)

# ---------- tokenizer round-trip ----------
print("=== TOKENIZER ROUND-TRIP ===")
for s in ["Paris", " the", " dog", "The capital of France is", "بسم الله", "def add("]:
    ids = tok.encode(s)
    back = tok.decode(ids)
    print(f"  {s!r:32s} -> {len(ids):>2} ids {ids[:8]}{'...' if len(ids)>8 else ''}"
          f"  round-trip {'OK ' if back == s else 'MISMATCH'} {back!r}")

# ---------- embedding table ----------
emb_t = None
for t in reader.tensors:
    if t.name == "token_embd.weight":
        emb_t = t
        break
print(f"\n=== EMBEDDING TABLE ===")
print(f"  {emb_t.name} shape={[int(s) for s in emb_t.shape]} type={emb_t.tensor_type}"
      f" data={emb_t.data.shape}")

D = 6144
cache = {}
def emb(word):
    ids = tok.encode(word)
    t = ids[0]
    if t not in cache:
        v = dequant_q2k_fast(bytes(emb_t.data[t]), D)
        cache[t] = torch.from_numpy(np.asarray(v, dtype=np.float32).copy())
    return cache[t], t

def cs(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

v, t = emb(" dog")
print(f"  ' dog' id={t}  |v|={float(v.norm()):.4f}  mean={float(v.mean()):+.5f}"
      f"  std={float(v.std()):.5f}  min={float(v.min()):+.3f}  max={float(v.max()):+.3f}"
      f"  zeros={int((v==0).sum())}/{D}")

PAIRS = [
    ((" dog", " cat"),        "related"),
    ((" king", " queen"),     "related"),
    ((" Monday", " Tuesday"), "related"),
    ((" red", " blue"),       "related"),
    ((" dog", " economics"),  "unrelated"),
    ((" king", " tomato"),    "unrelated"),
    ((" Monday", " helium"),  "unrelated"),
]
print(f"\n=== SEMANTIC STRUCTURE (related should beat unrelated) ===")
rel, unr = [], []
for (a, b), kind in PAIRS:
    va, _ = emb(a); vb, _ = emb(b)
    c = cs(va, vb)
    (rel if kind == "related" else unr).append(c)
    print(f"  {a!r:12s} vs {b!r:12s}  cos = {c:+.4f}   [{kind}]")
mr, mu = np.mean(rel), np.mean(unr)
print(f"\n  mean related  : {mr:+.4f}")
print(f"  mean unrelated: {mu:+.4f}")
print(f"  separation    : {mr - mu:+.4f}   "
      f"{'STRUCTURE PRESENT — dequant is sane' if mr - mu > 0.05 else 'NO STRUCTURE — dequant is likely WRONG'}")
