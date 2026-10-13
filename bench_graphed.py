"""A/B: full_moe_layer (baseline) vs full_moe_layer_v2 (preamble CUDA graph).

Correctness gate: top-10 logits must be identical between paths.
"""
import sys, time, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

print("Loading GGUF + kernel...")
reader = GGUFReader(MODEL_PATH, 'r')
lib = load_kernel()
engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)

print("Cold run (registers experts + captures graphs)...")
t0 = time.perf_counter()
engine.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
print(f"  cold: {time.perf_counter()-t0:.1f}s   experts cached: {len(engine._expert_cache)}"
      f"   preamble graphs: {getattr(engine, '_moe_graphs', 0)}")


def bench(label, fn, iters=25, warm=5):
    torch.cuda.synchronize()
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        lib.sync_device(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    avg = sum(ts) / len(ts)
    med = ts[len(ts) // 2]
    print(f"  {label:34s} avg={avg:6.2f} ms  med={med:6.2f} ms  min={ts[0]:6.2f} ms"
          f"  -> {1000/avg:5.1f} tok/s (avg) / {1000/ts[0]:5.1f} (min)")
    return avg, ts[0], out


# ---- PATH A: baseline (individual kernel launches) ----
def warm_v1(self, x, li):
    lib.full_moe_layer(VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
                       VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))


def warm_v2(self, x, li):
    lib.full_moe_layer_v2(VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
                          VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))


print("\n=== A/B (78 layers, single token) ===")
faie_generate.FAIEInference._moe_ffn_warm = warm_v1
a_avg, a_min, a_out = bench("A: full_moe_layer (no graph)", lambda: engine.forward_single_token(1))

faie_generate.FAIEInference._moe_ffn_warm = warm_v2
b_avg, b_min, b_out = bench("B: full_moe_layer_v2 (graphed)", lambda: engine.forward_single_token(1))

top_a = torch.topk(a_out.squeeze().float(), 10)
top_b = torch.topk(b_out.squeeze().float(), 10)
ids_match = top_a.indices.tolist() == top_b.indices.tolist()
max_dev = (a_out.squeeze().float() - b_out.squeeze().float()).abs().max().item()

print(f"\n=== RESULT ===")
print(f"  A (baseline): {a_avg:6.2f} ms avg / {a_min:6.2f} ms min")
print(f"  B (graphed):  {b_avg:6.2f} ms avg / {b_min:6.2f} ms min")
print(f"  Delta:        {b_avg-a_avg:+6.2f} ms avg ({(a_avg/b_avg-1)*100:+.1f}% faster)")
print(f"  CORRECTNESS: top-10 ids match = {ids_match}   max logit dev = {max_dev:.6f}")
print(f"  top-10 A: {top_a.indices.tolist()}")
print(f"  top-10 B: {top_b.indices.tolist()}")
print(f"  TARGET:   33.33 ms = 30.0 tok/s   |  gap = {b_avg-33.33:+.2f} ms")
