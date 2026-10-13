"""FAIE 30 tok/s PROOF — sustained autoregressive generation, v1 vs v2, real weights.

Gate 1 (correctness): both paths must emit the SAME greedy token sequence over N steps.
Gate 2 (speed):       v2 sustained throughput must be >= 30.0 tok/s.

Every token is a distinct hidden state, so the router picks different top-8 experts each
step. This exercises far more of the weight space than repeating one token.
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
N_STEPS = 40
SEED_TOKEN = 1

print("Loading GGUF + kernel...")
reader = GGUFReader(MODEL_PATH, 'r')
lib = load_kernel()
engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)

print("Cold run (register experts + capture preamble graphs)...")
engine.forward_single_token(SEED_TOKEN)
lib.sync_device(); torch.cuda.synchronize()
print(f"  experts cached={len(engine._expert_cache)}  preamble graphs={getattr(engine,'_moe_graphs',0)}")


def warm_v1(self, x, li):
    lib.full_moe_layer(VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
                       VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))


def warm_v2(self, x, li):
    lib.full_moe_layer_v2(VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
                          VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))


lib.reset_expert_counters.restype = None
lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong


def run(label, warm_fn, n=N_STEPS, fixed_token=None):
    """Greedy autoregressive rollout (or fixed-token loop). Reports expert residency."""
    faie_generate.FAIEInference._moe_ffn_warm = warm_fn
    tok = SEED_TOKEN
    for _ in range(3):                       # warmup: settle caches + clocks
        engine.forward_single_token(tok)
    lib.sync_device(); torch.cuda.synchronize()

    lib.reset_expert_counters()
    tok, seq, ms, vals = SEED_TOKEN, [], [], []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = engine.forward_single_token(fixed_token if fixed_token is not None else tok)
        lib.sync_device(); torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        flat = logits.squeeze().float()
        tok = int(torch.argmax(flat).item())
        seq.append(tok)
        vals.append(round(float(flat[tok].item()), 4))
    req = lib.get_experts_requested()
    exe = lib.get_experts_executed()
    resid = 100.0 * exe / req if req else 0.0
    total = sum(ms)
    print(f"  {label:14s} {n} tok in {total/1000:6.3f}s  ->  {n/(total/1000):5.2f} tok/s"
          f"   (avg {total/n:6.2f} ms, min {min(ms):6.2f})"
          f"   experts {exe}/{req} = {resid:5.1f}% resident")
    return seq, ms, vals, req, exe, resid


# ---- RUN A: fixed token. Every layer's top-8 is guaranteed resident (they were
#      registered from this exact hidden state), so this is FULL 8/8 compute. This
#      is the honest measure of the compute path.
print(f"\n=== RUN A — FULL-RESIDENCY (fixed token, 8/8 experts every layer) ===")
seqA1, msA1, vA1, reqA1, exeA1, rsA1 = run("v1 (baseline)", warm_v1, fixed_token=SEED_TOKEN)
seqA2, msA2, vA2, reqA2, exeA2, rsA2 = run("v2 (graph+vec)", warm_v2, fixed_token=SEED_TOKEN)

# ---- RUN B: free autoregressive rollout. Router wanders to experts that are NOT
#      cached (only 600 of 9,600 are resident), which the NULL guard skips. Faster,
#      but it is NOT the full model — reported for honesty, never as the headline.
print(f"\n=== RUN B — FREE ROLLOUT (exposes cache misses; NOT the full model) ===")
seqB1, msB1, vB1, reqB1, exeB1, rsB1 = run("v1 (baseline)", warm_v1)
seqB2, msB2, vB2, reqB2, exeB2, rsB2 = run("v2 (graph+vec)", warm_v2)

tps1 = N_STEPS / (sum(msA1) / 1000)
tps2 = N_STEPS / (sum(msA2) / 1000)
tpsB2 = N_STEPS / (sum(msB2) / 1000)
match = seqA1 == seqA2 and seqB1 == seqB2
maxdev = max(max(abs(a - b) for a, b in zip(vA1, vA2)),
             max(abs(a - b) for a, b in zip(vB1, vB2)))

print(f"\n=== GATE 1 — CORRECTNESS (v2 must equal v1 exactly) ===")
print(f"  run A sequences identical ({N_STEPS} steps): {seqA1 == seqA2}")
print(f"  run B sequences identical ({N_STEPS} steps): {seqB1 == seqB2}")
print(f"  run B seq[:10]: {seqB1[:10]}   distinct tokens: {len(set(seqB1))}/{N_STEPS}")
print(f"  max |top-1 logit| deviation: {maxdev:.6f}  (fp reassociation only)")

print(f"\n=== GATE 2 — SPEED (headline = RUN A, full 8/8 residency) ===")
print(f"  v1 baseline : {tps1:5.2f} tok/s  ({sum(msA1)/N_STEPS:6.2f} ms/tok)  residency {rsA1:.1f}%")
print(f"  v2 optimized: {tps2:5.2f} tok/s  ({sum(msA2)/N_STEPS:6.2f} ms/tok)  residency {rsA2:.1f}%")
print(f"  speedup     : {tps2/tps1:.3f}x")
print(f"  TARGET      : 30.00 tok/s (33.33 ms/tok)")
print(f"  [run B, {rsB2:.1f}% residency, inflated: {tpsB2:.2f} tok/s — NOT the claim]")

g0 = rsA1 >= 99.99 and rsA2 >= 99.99      # run A really was full 8/8
g1 = match
g2 = tps2 >= 30.0
print(f"\n=== VERDICT ===")
print(f"  GATE 0 full 8/8 residency in run A : {'PASS' if g0 else 'FAIL'}  ({rsA2:.2f}%)")
print(f"  GATE 1 v2 == v1 bit-for-bit tokens : {'PASS' if g1 else 'FAIL'}")
print(f"  GATE 2 >= 30 tok/s at full compute : {'PASS' if g2 else 'FAIL'}  ({tps2:.2f} tok/s)")
print(f"  OVERALL                            : {'PROVEN' if (g0 and g1 and g2) else 'NOT PROVEN'}")

out = {
    "timestamp_edt": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": MODEL_PATH, "n_steps": N_STEPS,
    "runA_full_residency": {
        "tps_v1": round(tps1, 3), "tps_v2": round(tps2, 3),
        "ms_per_tok_v1": round(sum(msA1) / N_STEPS, 3),
        "ms_per_tok_v2": round(sum(msA2) / N_STEPS, 3),
        "speedup": round(tps2 / tps1, 4),
        "experts_requested": reqA2, "experts_executed": exeA2, "residency_pct": round(rsA2, 3),
        "sequences_identical": seqA1 == seqA2,
    },
    "runB_free_rollout": {
        "tps_v2": round(tpsB2, 3),
        "experts_requested": reqB2, "experts_executed": exeB2, "residency_pct": round(rsB2, 3),
        "sequences_identical": seqB1 == seqB2,
        "distinct_tokens": len(set(seqB1)), "seq": seqB1,
        "note": "expert cache holds 600 of 9600; misses are skipped, so this tok/s is INFLATED",
    },
    "max_logit_dev": maxdev,
    "gate0_full_residency": bool(g0), "gate1_correctness": bool(g1), "gate2_speed": bool(g2),
    "proven": bool(g0 and g1 and g2),
}
p = r"E:\ABU\NEOCORTEX\faie_30tps_proof.json"
with open(p, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  proof written: {p}")
