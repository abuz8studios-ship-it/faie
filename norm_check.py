"""||c|| after kv_a_norm is 0.150 at L0. Since rms_norm output has unit RMS, ||c||
IS ||kv_a_norm weights||. A 512-dim RMS-norm weight vector of ~1.0 per element should
have norm ~sqrt(512)=22.6. 0.150 means ~0.0066 per element — 150x small.

Either this model genuinely has tiny norm weights, or they are being read wrong.
Dump the raw values and compare all four norm families.
"""
import sys
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
from gguf import GGUFReader

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
reader = GGUFReader(MODEL, "r")
by_name = {t.name: t for t in reader.tensors}

NORMS = ["attn_norm", "attn_q_a_norm", "attn_kv_a_norm", "ffn_norm"]
print(f"{'tensor':>22} {'L':>4} {'n':>6} {'type':>6} {'mean':>10} {'rms':>10}"
      f" {'norm':>10} {'min':>9} {'max':>9}")
print("-" * 100)
for nm in NORMS:
    for L in (0, 4, 12, 40, 77):
        t = by_name.get(f"blk.{L}.{nm}.weight")
        if t is None:
            continue
        n = int(np.prod([int(s) for s in t.shape]))
        v = np.frombuffer(np.ascontiguousarray(t.data).tobytes(), dtype=np.float32)[:n]
        print(f"{nm:>22} {L:>4} {n:>6} {str(t.tensor_type):>6} {v.mean():>10.5f}"
              f" {np.sqrt((v*v).mean()):>10.5f} {np.linalg.norm(v):>10.4f}"
              f" {v.min():>9.4f} {v.max():>9.4f}")
    print()

t = by_name.get("output_norm.weight")
if t is not None:
    n = int(np.prod([int(s) for s in t.shape]))
    v = np.frombuffer(np.ascontiguousarray(t.data).tobytes(), dtype=np.float32)[:n]
    print(f"{'output_norm':>22} {'-':>4} {n:>6} {str(t.tensor_type):>6} {v.mean():>10.5f}"
          f" {np.sqrt((v*v).mean()):>10.5f} {np.linalg.norm(v):>10.4f}"
          f" {v.min():>9.4f} {v.max():>9.4f}")

print("\nRMS ~1.0 is normal. RMS ~0.007 for kv_a_norm while attn_norm is ~1.0 would")
print("mean kv_a_norm specifically is being misread (or is genuinely tiny).")
print("\nfirst 12 raw values of blk.0.attn_kv_a_norm.weight:")
t = by_name["blk.0.attn_kv_a_norm.weight"]
raw = np.ascontiguousarray(t.data).tobytes()
print("  as f32:", np.frombuffer(raw, dtype=np.float32)[:12])
print("  as f16:", np.frombuffer(raw, dtype=np.float16)[:12])
print(f"  data bytes = {len(raw)}  shape={[int(s) for s in t.shape]}"
      f"  -> {len(raw)/int(np.prod([int(s) for s in t.shape])):.2f} bytes/elem")
