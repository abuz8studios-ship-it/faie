"""Does it TALK? MLA + KV cache coherence test.

Gate: with real attention over history, greedy generation must stop degenerating.
The pre-MLA engine produced 8 distinct tokens in 40 steps (23/10102/22580 looping)
and '999999...'. Anything resembling language is a pass; the specific bar is
distinct-token count and absence of a short repeating cycle.
"""
import sys, time, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
from expert_locality import GGUFTokenizer
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p

print("Loading...")
reader = GGUFReader(MODEL_PATH, "r")
lib = load_kernel()
tok = GGUFTokenizer(reader)
engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))

print(f"  MLA active: {engine.use_mla}")
engine.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
engine.reset_context()

lib.get_experts_requested.restype = ctypes.c_longlong
lib.get_experts_executed.restype = ctypes.c_longlong

PROMPTS = [
    "The capital of France is",
    "بسم الله",
    "def add(a, b):",
]


def _run(ids, n):
    """One full prefill+generate pass. Returns (generated_ids, last_logits)."""
    engine.reset_context()
    for t in ids:
        logits = engine.forward_single_token(t)
    lib.sync_device(); torch.cuda.synchronize()
    out = []
    cur = int(torch.argmax(logits.squeeze().float()).item())
    for _ in range(n):
        out.append(cur)
        logits = engine.forward_single_token(cur)
        lib.sync_device(); torch.cuda.synchronize()
        cur = int(torch.argmax(logits.squeeze().float()).item())
    return out


def generate(prompt, n=32, warm_rounds=10):
    """Warm THIS prompt's working set to 100% residency, then measure.

    Warming on token 1 is useless — every prompt routes somewhere different. The
    only way to judge attention is to remove the residency confound entirely.
    """
    ids = tok.encode(prompt, limit=32)
    for _ in range(warm_rounds):
        _run(ids, n)
        if engine.page_in_misses()[0] == 0:
            break
    lib.reset_expert_counters()
    t0 = time.perf_counter()
    out = _run(ids, n)
    el = time.perf_counter() - t0
    req, exe = lib.get_experts_requested(), lib.get_experts_executed()
    return out, el, (100.0 * exe / req if req else 0.0)


print("\n" + "=" * 74)
for p in PROMPTS:
    ids, el, resid = generate(p)
    text = tok.decode(ids)
    uniq = len(set(ids))
    # shortest repeating cycle length in the tail
    cyc = 0
    for L in range(1, 9):
        if len(ids) >= 4 * L and all(ids[-i] == ids[-i - L] for i in range(1, 2 * L + 1)):
            cyc = L
            break
    print(f"PROMPT   : {p!r}")
    print(f"OUTPUT   : {text!r}")
    print(f"distinct : {uniq}/{len(ids)}   repeat-cycle: {cyc if cyc else 'none'}"
          f"   residency: {resid:.1f}%   {len(ids)/el:.1f} tok/s")
    print("-" * 74)

print("\nBASELINE BEFORE MLA: 8 distinct / 40, cycle length 3, output '999999...'")
