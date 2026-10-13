"""THE ARABIC PRESYNTHESIZER — measured, not projected.

Locality said: an Arabic domain touches 3,754 experts (39% of the model). Too many to
preload on one card. But "touched" != "important". MoE routing is usually heavily
skewed: a minority of experts absorb most of the picks.

This measures the actual skew, then USES it:
  1. Profile a domain corpus -> frequency of every (layer, expert) pick.
  2. Coverage curve: preloading the top-N experts covers what % of picks?
  3. Preload the top-N that fit VRAM, then run HELD-OUT text from the same domain
     and measure real residency + real tok/s.

Held-out matters: profiling and testing on the same text would just be memorisation.
"""
import sys, os, json, time, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
from collections import Counter
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p
RCAP = 32768

# Arabic root-family corpus. PROFILE builds the expert working set; HELDOUT is never
# seen during profiling and is what the speed claim is measured on.
PROFILE = [
    "الرحمن الرحيم الرحمة رحم غفور غفران",
    "رحمة الله ورحمته الرحيم الغفور الودود",
    "المغفرة والرحمة والعفو والغفران والتوبة",
    "العلم والمعرفة والفهم والبصيرة والحكمة",
    "عالم يعلم علماء المعرفة الفقه والفهم",
    "الحكمة والعلم والفهم والإدراك والتدبر",
    "الصبر والشكر والتقوى والإحسان والعدل",
    "الخلق والرزق والحياة والموت والبعث",
]
HELDOUT = [
    "الغفور الرحيم يغفر الذنوب ويرحم العباد",
    "أهل العلم والحكمة يتدبرون ويفقهون",
]


def main():
    print("Loading...")
    reader = GGUFReader(MODEL_PATH, "r")
    lib = load_kernel()
    tok = GGUFTokenizer(reader)
    engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
    faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
        VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
        VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
    engine.forward_single_token(1)
    lib.sync_device(); torch.cuda.synchronize()

    lib.set_route_logging.argtypes = [ctypes.c_int]
    lib.drain_route_log.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.drain_route_log.restype = ctypes.c_int
    lib.get_experts_requested.restype = ctypes.c_longlong
    lib.get_experts_executed.restype = ctypes.c_longlong
    rbuf = (ctypes.c_int * (RCAP * 2))()
    lib.set_route_logging(1)

    CAP = engine._expert_cache_max
    print(f"  VRAM cap: {CAP} experts")

    # ---- 1. PROFILE ------------------------------------------------------
    print(f"\n=== 1. PROFILE the Arabic corpus ===")
    freq = Counter()
    total_picks = 0
    for text in PROFILE:
        ids = tok.encode(text, limit=24)
        lib.drain_route_log(rbuf, RCAP)
        for t in ids:
            engine.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        n = lib.drain_route_log(rbuf, RCAP)
        for i in range(n):
            freq[(rbuf[2 * i], rbuf[2 * i + 1])] += 1
        total_picks += n
        print(f"  {len(ids):>3} tok, {n:>5} picks | {text[:38]}")
    print(f"  distinct experts touched: {len(freq):,}   total picks: {total_picks:,}")

    # ---- 2. COVERAGE CURVE — is routing skewed enough to exploit? --------
    print(f"\n=== 2. SKEW — does a top-N subset absorb most picks? ===")
    ranked = freq.most_common()
    counts = np.array([c for _, c in ranked], dtype=np.float64)
    cum = np.cumsum(counts) / counts.sum()
    print(f"  {'top-N experts':>16} {'% of all picks':>16}")
    for n in (250, 500, 1000, 1500, 2000, 3000, len(ranked)):
        if n <= len(ranked):
            print(f"  {n:>16,} {100*cum[n-1]:>15.2f}%")
    # Gini-ish read: how many experts hold 90% of traffic?
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    print(f"  experts holding 90% of picks: {n90:,}   99%: {n99:,}")

    # ---- 3. PRELOAD the top-N that fit, then test on HELD-OUT ------------
    budget = min(CAP, len(ranked))
    print(f"\n=== 3. PRELOAD top-{budget:,} experts (fits VRAM) ===")
    t0 = time.perf_counter()
    loaded = 0
    for (li, eid), _ in ranked[:budget]:
        if (li, eid) not in engine._expert_cache:
            engine._ensure_expert_cached(li, eid)
            loaded += 1
    torch.cuda.synchronize()
    print(f"  loaded {loaded:,} new experts in {time.perf_counter()-t0:.1f}s"
          f"  (resident now {len(engine._expert_cache):,})")

    def measure(label, texts, gen_per=20):
        lib.reset_expert_counters()
        ms = []
        for text in texts:
            ids = tok.encode(text, limit=24)
            tk = ids[-1] if ids else 1
            for t in ids[:-1]:
                engine.forward_single_token(t)
            lib.sync_device(); torch.cuda.synchronize()
            for _ in range(gen_per):
                torch.cuda.synchronize()
                s = time.perf_counter()
                logits = engine.forward_single_token(tk)
                lib.sync_device(); torch.cuda.synchronize()
                ms.append((time.perf_counter() - s) * 1000)
                tk = int(torch.argmax(logits.squeeze().float()).item())
        req, exe = lib.get_experts_requested(), lib.get_experts_executed()
        resid = 100.0 * exe / req if req else 0.0
        tps = len(ms) / (sum(ms) / 1000)
        print(f"  {label:34s} {tps:6.2f} tok/s ({sum(ms)/len(ms):6.2f} ms)"
              f"   residency {exe:,}/{req:,} = {resid:6.2f}%")
        return tps, resid

    print(f"\n=== 4. HELD-OUT Arabic (never profiled) ===")
    tps_held, res_held = measure("held-out arabic, presynth ON", HELDOUT)

    print(f"\n=== 5. CONTROL — off-domain text, same cache ===")
    tps_off, res_off = measure("english finance, presynth OFF-domain",
                               ["the quarterly earnings call discussed margins"])

    print(f"\n=== VERDICT ===")
    print(f"  skew: {n90:,} experts carry 90% of Arabic picks")
    print(f"  held-out Arabic  : {tps_held:6.2f} tok/s at {res_held:6.2f}% residency"
          f"  {'>= 30 PASS' if tps_held >= 30 else 'below 30'}")
    print(f"  off-domain       : {tps_off:6.2f} tok/s at {res_off:6.2f}% residency")
    print(f"  presynth benefit : {res_held - res_off:+.1f} residency points on-domain")

    out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "profile_distinct": len(freq), "total_picks": total_picks,
           "experts_for_90pct": n90, "experts_for_99pct": n99,
           "coverage": {str(n): float(cum[n-1]) for n in (250,500,1000,1500,2000,3000) if n <= len(ranked)},
           "preloaded": budget, "resident": len(engine._expert_cache),
           "heldout_tps": round(tps_held,3), "heldout_residency": round(res_held,3),
           "offdomain_tps": round(tps_off,3), "offdomain_residency": round(res_off,3)}
    p = r"E:\ABU\NEOCORTEX\presynth_proof.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  written: {p}")


if __name__ == "__main__":
    main()
