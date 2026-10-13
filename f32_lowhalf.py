"""Are some F32 tensors actually f16 bit-patterns shifted into the high halfword?

blk.0.attn_kv_a_norm reads 0.00915 as f32, but its high 16 bits read as f16 give
1.021 — and its low 16 bits are ZERO. output_norm reads a normal 1.21. If the
low-16-bits-zero pattern is ~100% for some tensors and ~0% for others, those tensors
were written with f16 values placed at bf16 positions, and every consumer of this
GGUF (including llama.cpp) sees weights ~110x too small.

Decisive stat per tensor: fraction of elements whose low 16 bits are zero.
Random f32 data has essentially 0% (chance ~1/65536).
"""
import sys
import numpy as np
from gguf import GGUFReader

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
reader = GGUFReader(MODEL, "r")

def stats(t):
    n = int(np.prod([int(s) for s in t.shape]))
    raw = np.ascontiguousarray(t.data).tobytes()[: n * 4]
    u = np.frombuffer(raw, dtype=np.uint32)
    f = np.frombuffer(raw, dtype=np.float32)
    lo_zero = float((u & 0xFFFF == 0).mean())
    hi_as_f16 = np.frombuffer((u >> 16).astype(np.uint16).tobytes(), dtype=np.float16)
    return n, lo_zero, float(np.sqrt((f * f).mean())), float(np.sqrt((hi_as_f16.astype(np.float32) ** 2).mean()))

TARGETS = []
for L in (0, 12, 40, 77):
    for nm in ("attn_norm", "attn_q_a_norm", "attn_kv_a_norm", "ffn_norm"):
        TARGETS.append(f"blk.{L}.{nm}.weight")
TARGETS += ["output_norm.weight"]

by_name = {t.name: t for t in reader.tensors}
print(f"{'tensor':>32} {'n':>6} {'low16==0':>10} {'rms as f32':>12} {'rms hi-as-f16':>14}  verdict")
print("-" * 104)
for nm in TARGETS:
    t = by_name.get(nm)
    if t is None or str(t.tensor_type) not in ("0", "GGMLQuantizationType.F32"):
        continue
    n, lz, r32, r16 = stats(t)
    v = "F16-IN-HIGH-HALF" if lz > 0.98 else ("mixed" if lz > 0.02 else "genuine F32")
    print(f"{nm:>32} {n:>6} {lz*100:>9.2f}% {r32:>12.5f} {r16:>14.5f}  {v}")

# Scan every F32 tensor in the file for the pattern.
print("\n=== WHOLE-FILE SCAN of F32 tensors ===")
tot = bad = 0
worst = []
for t in reader.tensors:
    if str(t.tensor_type) not in ("0", "GGMLQuantizationType.F32"):
        continue
    n, lz, r32, r16 = stats(t)
    tot += 1
    if lz > 0.98:
        bad += 1
        worst.append((t.name, r32, r16))
print(f"  F32 tensors: {tot}   with low-16-bits all zero: {bad}  ({100*bad/max(tot,1):.1f}%)")
fams = {}
for nm, r32, r16 in worst:
    key = nm.split(".")[-2] if nm.startswith("blk.") else nm
    fams.setdefault(key, []).append((r32, r16))
for k, v in sorted(fams.items()):
    a = np.mean([x[0] for x in v]); b = np.mean([x[1] for x in v])
    print(f"    {k:>22}  x{len(v):<4} mean rms f32={a:.5f}  as-f16={b:.5f}"
          f"  ratio {b/max(a,1e-12):.1f}x")
