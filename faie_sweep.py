"""Sweep FAIE structural variants against the llama.cpp oracle.

ORACLE (llama.cpp b10069, same GGUF, temp 0):
  "The capital of France is" -> " Paris" (id 12089, logprob -0.445)
  runners-up: " usually"(5990) " France"(9621) " mostly"(10003) " most"(1429)

My kernel and my PyTorch reference agree at cos 1.000000, so they can share a wrong
assumption. This tries the assumptions I can toggle and scores each against the
oracle. Scoring: rank of id 12089 in FAIE's logits (lower = better), plus whether any
oracle top-5 token appears in FAIE's top-5.
"""
import sys, ctypes, itertools, json, io
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

ORACLE_TOP1 = 12089                      # " Paris"
ORACLE_TOP5 = {12089, 5990, 9621, 10003, 1429}
PROMPT = "The capital of France is"

reader = GGUFReader(MODEL, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
eng.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()

lib.set_rope_style.argtypes = [ctypes.c_int]
lib.set_score_scale.argtypes = [ctypes.c_float]
lib.set_router_scale.argtypes = [ctypes.c_float]
lib.set_bisect.argtypes = [ctypes.c_int, ctypes.c_int]
lib.set_bisect(0, 0)

IDS = tok.encode(PROMPT, limit=32)
print(f"prompt ids = {IDS}   (llama.cpp also produced 5 tokens, ' France'=9621)")
assert 9621 in IDS, "tokenizer mismatch vs oracle!"

# page in the working set once, at default settings
for _ in range(10):
    eng.reset_context()
    for t in IDS:
        eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    if eng.page_in_misses()[0] == 0:
        break
print(f"resident {len(eng._expert_cache):,}\n")


def run():
    # Re-warm under the CURRENT settings: routing shifts with each variant, and
    # the NULL guard silently skips non-resident experts (handoff trap #3).
    for _ in range(25):
        eng.reset_context()
        for t in IDS:
            eng.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        if eng.page_in_misses()[0] == 0:
            break
    lib.reset_expert_counters()
    eng.reset_context()
    for t in IDS:
        lg = eng.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    req, exe = lib.get_experts_requested(), lib.get_experts_executed()
    if req != exe:
        print(f"  !! residency gate FAILED: requested={req} executed={exe}")
    return lg.squeeze().float()


ROPE = [(0, "neox"), (1, "gptj")]
SSCALE = [1.0, 4.0, 16.0]
RSCALE = [1.0, 2.5]

rows = []
print(f"{'rope':>5} {'sscale':>7} {'rscale':>7} | {'rank(Paris)':>12} {'overlap':>8} | top5")
print("-" * 108)
for (rs, rn), ss, rr in itertools.product(ROPE, SSCALE, RSCALE):
    lib.set_rope_style(rs)
    lib.set_score_scale(ctypes.c_float(ss))
    lib.set_router_scale(ctypes.c_float(rr))
    lg = run()
    rank = int((lg > lg[ORACLE_TOP1]).sum())
    top5 = torch.topk(lg, 5).indices.tolist()
    ov = len(ORACLE_TOP5 & set(top5))
    txt = [tok.decode([i]).strip()[:12] for i in top5]
    rows.append(dict(rope=rn, sscale=ss, rscale=rr, rank=rank, overlap=ov, top5=txt))
    print(f"{rn:>5} {ss:>7.1f} {rr:>7.1f} | {rank:>12,} {ov:>8} | {txt}")

rows.sort(key=lambda r: (r["rank"], -r["overlap"]))
best = rows[0]
print(f"\nBEST: rope={best['rope']} sscale={best['sscale']} rscale={best['rscale']}"
      f"  rank(Paris)={best['rank']:,} of 154,880  overlap={best['overlap']}/5")
print(f"  top5 {best['top5']}")
print(f"\nA correct implementation puts rank(Paris)=0. Rank in the tens of thousands")
print(f"means the defect is structural, not a tunable — none of these knobs is it.")
json.dump(rows, open(r"E:\ABU\NEOCORTEX\faie_sweep.json", "w"), indent=2)
# restore defaults: rope style 1 = llama.cpp LLAMA_ROPE_TYPE_NORM for glm-dsa
lib.set_rope_style(1); lib.set_score_scale(ctypes.c_float(1.0)); lib.set_router_scale(ctypes.c_float(2.5))
