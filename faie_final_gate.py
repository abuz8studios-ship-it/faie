"""FINAL GATE — one engine load, three gated measurements.

Replaces the faie_sweep.py + test_predict.py + prove_honest.py chain for this
run because two external llama-imatrix jobs hold ~123 GB RAM and every extra
engine load costs 10+ minutes under that pressure. Measurements and gating are
the same:
  1. sweep:    rank(Paris) vs the oracle for rope styles 0/1 at the GGUF's own
               scales (sscale 1.0, rscale 2.5), 100%-residency gated
  2. predict:  4 trivial prompts, expected token must appear in top-5
  3. tok/s:    greedy 32-token rollout, second pass timed at 100% residency

Every measured pass is gated on lib.get_experts_requested() == executed and 0
post-run misses (handoff trap #3).
"""
import sys, ctypes, io, json, time, atexit
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush()))
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

ORACLE_TOP1 = 12089
ORACLE_TOP5 = {12089, 5990, 9621, 10003, 1429}
PROMPT = "The capital of France is"

t_start = time.perf_counter()
reader = GGUFReader(MODEL, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))

lib.set_rope_style.argtypes = [ctypes.c_int]
lib.set_score_scale.argtypes = [ctypes.c_float]
lib.set_router_scale.argtypes = [ctypes.c_float]
lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong
lib.set_score_scale(ctypes.c_float(1.0))
lib.set_router_scale(ctypes.c_float(2.5))

IDS = tok.encode(PROMPT, limit=32)
assert IDS == [785, 6722, 315, 9621, 374], f"tokenizer mismatch: {IDS}"
print(f"engine loaded in {time.perf_counter()-t_start:.0f}s; prompt ids = {IDS}", flush=True)

eng.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()

RESULTS = {"sweep": [], "predict": [], "throughput": {}}


def gated_prompt_run(ids, warm_iters=40):
    """Warm to 0 misses, then one gated pass. Returns (logits, gate_ok, req, exe)."""
    warmed = False
    for _ in range(warm_iters):
        eng.reset_context()
        for t in ids:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        paged, dropped = eng.page_in_misses()
        if paged == 0 and dropped == 0:
            warmed = True
            break
    lib.reset_expert_counters()
    eng.reset_context()
    for t in ids:
        lg = eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    req = int(lib.get_experts_requested())
    exe = int(lib.get_experts_executed())
    post = eng.page_in_misses()
    ok = warmed and req == exe and post[0] == 0
    return lg.squeeze().float(), ok, req, exe


# ---------------- 1. SWEEP ----------------
print("\n=== 1. SWEEP (gated) ===", flush=True)
for style, name in [(0, "neox"), (1, "gptj")]:
    lib.set_rope_style(style)
    lg, ok, req, exe = gated_prompt_run(IDS)
    rank = int((lg > lg[ORACLE_TOP1]).sum())
    top5_ids = torch.topk(lg, 5).indices.tolist()
    ov = len(ORACLE_TOP5 & set(top5_ids))
    txt = [tok.decode([i]).strip()[:12] for i in top5_ids]
    RESULTS["sweep"].append(dict(rope=name, sscale=1.0, rscale=2.5, rank=rank,
                                 overlap=ov, top5=txt, top5_ids=top5_ids,
                                 gate_ok=ok, requested=req, executed=exe))
    print(f"{name:>5} sscale=1.0 rscale=2.5 | rank(Paris)={rank:,} overlap={ov}/5 "
          f"| residency {req}/{exe} gate_ok={ok} | {txt}", flush=True)

best = sorted(RESULTS["sweep"], key=lambda r: (r["rank"], -r["overlap"]))[0]
print(f"\nBEST: rope={best['rope']} sscale={best['sscale']} rscale={best['rscale']}"
      f"  rank(Paris)={best['rank']:,} of 154,880  overlap={best['overlap']}/5"
      f"  residency={'100%' if best['gate_ok'] else 'FAILED'}", flush=True)

# ---------------- 2. PREDICT ----------------
print("\n=== 2. PREDICT (gated, rope=1/NORM) ===", flush=True)
lib.set_rope_style(1)
PROMPTS = [
    ("The capital of France is", " Paris"),
    ("1, 2, 3, 4,", " 5"),
    ("Once upon a", " time"),
    ("The quick brown fox jumps over the lazy", " dog"),
]
for p, expect in PROMPTS:
    ids = tok.encode(p, limit=64)
    lg, ok, req, exe = gated_prompt_run(ids)
    top5_ids = torch.topk(lg, 5).indices.tolist()
    txt = [tok.decode([i]) for i in top5_ids]
    hit = any(t.strip() == expect.strip() for t in txt)
    RESULTS["predict"].append(dict(prompt=p, expect=expect, top5=[t.strip() for t in txt],
                                   hit=hit, gate_ok=ok, requested=req, executed=exe))
    print(f"  {p!r} -> top5 {[t.strip() for t in txt]}  expect {expect!r} "
          f"{'HIT' if hit else 'MISS'}  residency {req}/{exe} gate_ok={ok}", flush=True)

# ---------------- 3. THROUGHPUT (resident regime) ----------------
print("\n=== 3. THROUGHPUT (greedy 32-token rollout, pass 2 timed) ===", flush=True)
N_GEN = 32

def greedy_rollout(record):
    eng.reset_context()
    toks = list(IDS)
    for t in toks[:-1]:
        eng.forward_single_token(t)
    seq = []
    for _ in range(N_GEN):
        lg = eng.forward_single_token(toks[-1])
        nxt = int(lg.squeeze().float().argmax())
        seq.append(nxt)
        toks.append(nxt)
    lib.sync_device(); torch.cuda.synchronize()
    return seq

# pass 1: page in everything this greedy trajectory needs
for _ in range(6):
    seq1 = greedy_rollout(False)
    paged, dropped = eng.page_in_misses()
    if paged == 0 and dropped == 0:
        break
# pass 2: timed, gated
lib.reset_expert_counters()
t0 = time.perf_counter()
seq2 = greedy_rollout(True)
t1 = time.perf_counter()
req = int(lib.get_experts_requested())
exe = int(lib.get_experts_executed())
post = eng.page_in_misses()
tokps = (len(IDS) - 1 + N_GEN) / (t1 - t0)
same = seq1 == seq2
text = tok.decode(seq2)
RESULTS["throughput"] = dict(n_forwards=len(IDS) - 1 + N_GEN, seconds=t1 - t0,
                             tok_per_s=tokps, requested=req, executed=exe,
                             post_misses=post, deterministic=same, text=text)
print(f"  {len(IDS)-1+N_GEN} forwards in {t1-t0:.2f}s = {tokps:.2f} tok/s "
      f"| residency {req}/{exe} ({'100%' if req==exe else 'PARTIAL'}) "
      f"| post_misses={post} | deterministic={same}", flush=True)
print(f"  greedy continuation: {text!r}", flush=True)

json.dump(RESULTS, open(r"E:\ABU\NEOCORTEX\faie_final_gate.json", "w"), indent=2)
print("\nRESULTS -> faie_final_gate.json", flush=True)
lib.set_rope_style(1)
