"""FAIE HONEST wall-clock proof. Paging cost is INSIDE the timer.

The previous run measured 30-33 tok/s with page_in_misses() outside the timed
region, which is not a real throughput number. Here every millisecond spent making
an expert resident counts against tok/s, because a user waiting for a token pays it.

Reports three regimes, all measured the same way:
  RESIDENT   - fixed token, 100% of top-8 already on GPU (the compute ceiling)
  WARM       - free rollout over seeds the cache has already seen
  COLD       - free rollout over unseen seeds (worst case)
"""
import sys, time, ctypes, json
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

reader = GGUFReader(MODEL_PATH, 'r')
lib = load_kernel()
lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong
lib.drain_miss_log.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.drain_miss_log.restype = ctypes.c_int
lib.get_miss_dropped.restype = ctypes.c_longlong

engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))

engine.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
CAP = engine._expert_cache_max
print(f"  resident={len(engine._expert_cache)}  cap={CAP}")

# Cost of making ONE expert resident (NVMe read + dequant-free upload).
t0 = time.perf_counter()
n_probe, probe = 0, []
for eid in range(128):
    if (40, eid) not in engine._expert_cache:
        t = time.perf_counter()
        engine._ensure_expert_cached(40, eid)
        torch.cuda.synchronize()
        probe.append((time.perf_counter() - t) * 1000)
        n_probe += 1
    if n_probe >= 24:
        break
page_ms = sum(probe) / len(probe) if probe else 0.0
print(f"  page-in cost: {page_ms:.2f} ms/expert (measured over {n_probe} loads)")


def measure(label, seed, n=25, fixed=False):
    """Wall clock INCLUDING paging. This is what a user actually waits."""
    tok = seed
    for _ in range(3):
        engine.forward_single_token(fixed and seed or tok)
        engine.page_in_misses()
    lib.sync_device(); torch.cuda.synchronize()
    lib.reset_expert_counters()
    tok, ms, paged_tot = seed, [], 0
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = engine.forward_single_token(seed if fixed else tok)
        lib.sync_device(); torch.cuda.synchronize()
        paged, _ = engine.page_in_misses()          # INSIDE the timer
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        paged_tot += paged
        tok = int(torch.argmax(logits.squeeze().float()).item())
    req, exe = lib.get_experts_requested(), lib.get_experts_executed()
    resid = 100.0 * exe / req if req else 0.0
    tps = n / (sum(ms) / 1000)
    print(f"  {label:24s} {tps:6.2f} tok/s ({sum(ms)/n:7.2f} ms/tok)"
          f"  residency {resid:6.2f}%  paged {paged_tot:>5} experts")
    return {"label": label, "tps": round(tps, 3), "ms_per_tok": round(sum(ms) / n, 3),
            "residency_pct": round(resid, 3), "experts_paged": paged_tot}


print(f"\n=== A. RESIDENT (fixed token, 100% top-8 on GPU) — the compute ceiling ===")
A = measure("fixed token", 1, n=25, fixed=True)

print(f"\n=== B. WARM free rollout (paging cost INSIDE the timer) ===")
B = [measure(f"warm seed {s}", s, n=25) for s in (2500, 33000)]

print(f"\n=== C. COLD free rollout, unseen seeds (worst case) ===")
C = [measure(f"cold seed {s}", s, n=25) for s in (7777, 101010)]

free_b, total_b = torch.cuda.mem_get_info(0)
per_exp_mb = 13.03
print(f"\n=== THE WALL (physics, not opinion) ===")
print(f"  experts in model      : 9,600 x {per_exp_mb:.2f} MB = {9600*per_exp_mb/1024:.1f} GB")
print(f"  VRAM on one 5090      : {total_b/1024**3:.1f} GB  (now {free_b/1024**3:.1f} GB free)")
print(f"  max resident experts  : {CAP} = {100*CAP/9600:.1f}% of the model")
print(f"  page-in cost          : {page_ms:.2f} ms/expert")
print(f"  -> a 100%-miss token would cost {600*page_ms/1000:.1f} s (600 expert loads)")

print(f"\n=== VERDICT (what is PROVEN vs what is NOT) ===")
print(f"  PROVEN  compute path at 100% residency : {A['tps']:.2f} tok/s "
      f"{'>= 30 PASS' if A['tps'] >= 30 else '< 30 FAIL'}")
worst = min(r['tps'] for r in B + C)
print(f"  MEASURED free rollout, paging included : {worst:.2f} - "
      f"{max(r['tps'] for r in B+C):.2f} tok/s at "
      f"{min(r['residency_pct'] for r in B+C):.0f}-{max(r['residency_pct'] for r in B+C):.0f}% residency")
print(f"  NOT PROVEN: full-model generation with every top-8 expert executed —")
print(f"              {9600*per_exp_mb/1024:.0f} GB of Q2_K experts cannot fit in "
      f"{total_b/1024**3:.0f} GB. Needs the ternary conversion (13.03 -> ~7.5 MB/expert)")
print(f"              plus async streaming overlapped with compute.")

out = {"timestamp_edt": time.strftime("%Y-%m-%d %H:%M:%S"),
       "page_ms_per_expert": round(page_ms, 3), "cache_cap": CAP,
       "resident_pct_of_model": round(100 * CAP / 9600, 2),
       "experts_total_gb": round(9600 * per_exp_mb / 1024, 1),
       "vram_total_gb": round(total_b / 1024**3, 1),
       "A_resident": A, "B_warm": B, "C_cold": C,
       "proven_compute_tps": A["tps"],
       "proven_at_full_residency": bool(A["tps"] >= 30 and A["residency_pct"] >= 99.99)}
p = r"E:\ABU\NEOCORTEX\faie_honest_proof.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  written: {p}")
