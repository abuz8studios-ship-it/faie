"""FAIE demand-paging proof: does a FREE rollout reach 100% expert residency?

Run A proved 32.34 tok/s when every top-8 expert is already resident. The open
question was what happens when the router wanders. This drains the kernel's miss
log after each token and pages those experts in, then measures whether residency
converges and what the steady-state throughput actually is.
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

print("Loading...")
reader = GGUFReader(MODEL_PATH, 'r')
lib = load_kernel()
lib.reset_expert_counters.restype = None
lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong
lib.drain_miss_log.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.drain_miss_log.restype = ctypes.c_int
lib.get_miss_dropped.restype = ctypes.c_longlong

engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))

print("Cold run...")
engine.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
print(f"  resident={len(engine._expert_cache)}  cap={engine._expert_cache_max}")

# ---- Phase 1: WARM UP the resident set by paging in whatever the router wants.
#      Walk a spread of seed tokens so we touch many distinct routing patterns.
print(f"\n=== PHASE 1 — demand-paging warmup ===")
SEEDS = [1, 500, 2500, 9000, 17000, 33000, 60000, 91000, 120000, 150000]
t_start = time.perf_counter()
total_paged = 0
for si, seed in enumerate(SEEDS):
    tok = seed
    for step in range(12):
        engine.forward_single_token(tok)
        lib.sync_device(); torch.cuda.synchronize()
        paged, dropped = engine.page_in_misses()
        total_paged += paged
    r = len(engine._expert_cache)
    print(f"  seed {seed:>6}  resident={r:>5}/{engine._expert_cache_max}"
          f"  paged_so_far={total_paged}")
print(f"  warmup took {time.perf_counter()-t_start:.1f}s, "
      f"{len(engine._expert_cache)} experts resident "
      f"({100*len(engine._expert_cache)/9600:.1f}% of all 9,600)")

free_b, total_b = torch.cuda.mem_get_info(0)
print(f"  VRAM now: {free_b/1024**3:.1f} GB free of {total_b/1024**3:.1f} GB")


def measure(label, seed, n=30, page=True):
    """Greedy rollout with per-token residency + throughput."""
    tok = seed
    for _ in range(3):
        engine.forward_single_token(tok)
        if page:
            engine.page_in_misses()
    lib.sync_device(); torch.cuda.synchronize()
    lib.reset_expert_counters()
    tok, ms, seq = seed, [], []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = engine.forward_single_token(tok)
        lib.sync_device(); torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        tok = int(torch.argmax(logits.squeeze().float()).item())
        seq.append(tok)
        if page:
            engine.page_in_misses()
    req, exe = lib.get_experts_requested(), lib.get_experts_executed()
    resid = 100.0 * exe / req if req else 0.0
    tps = n / (sum(ms) / 1000)
    print(f"  {label:30s} {tps:5.2f} tok/s ({sum(ms)/n:6.2f} ms/tok)"
          f"   residency {exe}/{req} = {resid:6.2f}%   distinct_tok={len(set(seq))}")
    return tps, resid, sum(ms) / n


print(f"\n=== PHASE 2 — steady-state on warmed seeds (paging ON) ===")
rows = []
for seed in [1, 2500, 33000, 120000]:
    tps, resid, mspt = measure(f"seed {seed}", seed, n=30, page=True)
    rows.append({"seed": seed, "tps": round(tps, 3), "residency_pct": round(resid, 3),
                 "ms_per_tok": round(mspt, 3)})

print(f"\n=== PHASE 3 — COLD seeds never seen before (worst case) ===")
cold = []
for seed in [7777, 44444, 101010]:
    tps, resid, mspt = measure(f"cold seed {seed}", seed, n=30, page=True)
    cold.append({"seed": seed, "tps": round(tps, 3), "residency_pct": round(resid, 3),
                 "ms_per_tok": round(mspt, 3)})

full = [r for r in rows if r["residency_pct"] >= 99.99]
print(f"\n=== VERDICT ===")
print(f"  resident experts        : {len(engine._expert_cache)} / 9,600"
      f" ({100*len(engine._expert_cache)/9600:.1f}%)")
print(f"  seeds at 100% residency : {len(full)}/{len(rows)}")
if full:
    lo = min(r['tps'] for r in full)
    print(f"  slowest full-residency  : {lo:.2f} tok/s"
          f"   {'PASS >= 30' if lo >= 30 else 'BELOW 30'}")
print(f"  miss-log overflow drops : {lib.get_miss_dropped()}")

out = {"timestamp_edt": time.strftime("%Y-%m-%d %H:%M:%S"),
       "resident_experts": len(engine._expert_cache), "total_experts": 9600,
       "cache_cap": engine._expert_cache_max,
       "vram_free_gb": round(free_b / 1024**3, 2),
       "warm_seeds": rows, "cold_seeds": cold,
       "miss_log_dropped": int(lib.get_miss_dropped())}
p = r"E:\ABU\NEOCORTEX\faie_paging_proof.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  written: {p}")
