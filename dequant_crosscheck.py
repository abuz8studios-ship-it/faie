"""Independent dequant verification: FAIE's own dequant routines vs gguf.quants.dequantize.

Every prior kernel-vs-PyTorch check used FAIE's dequant on both sides, so a dequant
bug would be invisible. gguf.quants is an independent implementation.
"""
import sys, io
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from faie_generate import resolve_qtype, dequant_any, dequant_q6k
from fast_dequant import dequant_q2k_fast, dequant_q3k_fast

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
reader = GGUFReader(MODEL, "r")
tensors = {t.name: t for t in reader.tensors}

NAMES = [
    "blk.8.attn_q_a.weight",      # Q2_K
    "blk.8.attn_q_b.weight",      # Q2_K
    "blk.8.attn_kv_a_mqa.weight", # Q2_K
    "blk.8.attn_output.weight",   # Q3_K
    "blk.8.attn_k_b.weight",      # Q4_0
    "blk.8.attn_v_b.weight",      # Q2_K
    "blk.0.attn_q_b.weight",      # Q2_K, layer 0
    "blk.0.ffn_gate.weight",      # dense layer FFN
    "blk.0.ffn_down.weight",
]

for name in NAMES:
    t = tensors.get(name)
    if t is None:
        print(f"{name}: MISSING")
        continue
    qt = resolve_qtype(t.tensor_type)
    ne = [int(s) for s in t.shape]
    n_elem = int(np.prod(ne))
    # independent reference
    ref = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)[:n_elem]
    # FAIE's dequant
    mine = dequant_any(t).reshape(-1)[:n_elem]
    d = np.abs(ref - mine)
    denom = np.abs(ref).max() + 1e-30
    cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine) + 1e-30))
    print(f"{name:32s} {qt:5s} ne={ne}  maxabs={d.max():.3e}  rel={d.max()/denom:.3e}  cos={cos:.8f}")

# output head Q6_K — check a slice (full is 3.8 GB f32)
t = tensors["output.weight"]
qt = resolve_qtype(t.tensor_type)
rows = 2048
row_bytes = t.data.shape[-1] if t.data.ndim > 1 else None
sl = np.ascontiguousarray(t.data[:rows])
ref = dequantize(sl, t.tensor_type).astype(np.float32).reshape(-1)
mine = dequant_q6k(sl.tobytes(), ref.size)
d = np.abs(ref - np.asarray(mine, dtype=np.float32))
cos = float(np.dot(ref, mine) / (np.linalg.norm(ref) * np.linalg.norm(mine) + 1e-30))
print(f"{'output.weight[:2048 rows]':32s} {qt:5s}            maxabs={d.max():.3e}  cos={cos:.8f}")

# token_embd Q2_K — one row semantic + numeric
t = tensors["token_embd.weight"]
sl = np.ascontiguousarray(t.data[12089])  # " Paris"
ref = dequantize(sl.reshape(1, -1), t.tensor_type).astype(np.float32).reshape(-1)
mine = dequant_q2k_fast(sl.tobytes(), ref.size)
d = np.abs(ref - np.asarray(mine, dtype=np.float32))
print(f"{'token_embd[12089]':32s} Q2_K             maxabs={d.max():.3e}")
print("done")
