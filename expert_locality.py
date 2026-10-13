"""THE ARABIC THESIS, TESTED ON REAL MoE ROUTING.

The claim to test: MoE experts are semantic specialists, so semantically-related
text activates OVERLAPPING expert sets. If true, a root-semantic presynthesizer can
predict the working set for a prompt and preload it -> residency jumps -> tok/s jumps.
If false, expert activation is content-independent and no amount of Arabic structure
helps residency, and I should say so plainly.

Measurement: for each probe text, run real tokens through the real router and record
the exact (layer, expert) set the model actually selects. Then compare:
  WITHIN-domain  Jaccard overlap  vs  ACROSS-domain Jaccard overlap
A meaningful gap = routing is semantic = the thesis holds for residency.
No gap = routing is diffuse = the thesis does not buy speed here.
"""
import sys, os, json, time, ctypes
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import FAIEInference
import faie_generate

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p


# ---------------------------------------------------------------- tokenizer
def _byte_maps():
    """GPT-2 byte<->unicode table. Byte-level BPE vocabs store each raw byte as a
    printable codepoint via this map, so Arabic (multi-byte UTF-8) appears in the
    vocab as sequences of Latin-ish glyphs, NOT as Arabic characters. Decoding vocab
    entries as UTF-8 is what silently ate the Arabic probes."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    return b2u, {v: k for k, v in b2u.items()}


class GGUFTokenizer:
    """Greedy longest-match over the model's own byte-level BPE vocab.

    Not a full BPE merge walk, but it emits REAL ids from the model's vocabulary and
    \u2014 critically \u2014 never silently drops input: every byte is consumed, falling back
    to the single-byte token when no longer match exists.
    """

    def __init__(self, reader):
        toks = reader.fields["tokenizer.ggml.tokens"]
        self.b2u, self.u2b = _byte_maps()
        self.vocab = []
        for i in range(len(toks.data)):
            self.vocab.append(bytes(toks.parts[toks.data[i]]).decode("utf-8", errors="replace"))
        self.lookup = {}
        for i, t in enumerate(self.vocab):
            self.lookup.setdefault(t, i)
        self.max_len = min(max(len(t) for t in self.vocab), 32)

    def _to_bpe_space(self, text):
        return "".join(self.b2u[b] for b in text.encode("utf-8"))

    def encode(self, text, limit=None):
        s = self._to_bpe_space(text)
        out, i, n = [], 0, len(s)
        dropped = 0
        while i < n:
            hit = None
            for L in range(min(self.max_len, n - i), 0, -1):
                cand = s[i:i + L]
                j = self.lookup.get(cand)
                if j is not None:
                    hit = (j, L)
                    break
            if hit is None:
                dropped += 1          # single char with no token at all \u2014 count it
                i += 1
                continue
            out.append(hit[0])
            i += hit[1]
            if limit and len(out) >= limit:
                break
        self.last_dropped = dropped
        return out

    def decode(self, ids):
        s = "".join(self.vocab[i] if 0 <= i < len(self.vocab) else "" for i in ids)
        return bytes(self.u2b.get(ch, 63) for ch in s).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- probes
# Four domains. Within a domain the texts are semantically tight; across domains
# they are unrelated. Arabic probes included because that is the actual thesis.
PROBES = {
    "arabic_mercy": [
        "الرحمن الرحيم الرحمة رحم غفور",
        "رحمة الله ورحمته الرحيم غفران",
        "المغفرة والرحمة والعفو والغفران",
    ],
    "arabic_knowledge": [
        "العلم والمعرفة والفهم والبصيرة",
        "عالم يعلم علماء المعرفة الفقه",
        "الحكمة والعلم والفهم والإدراك",
    ],
    "code": [
        "def compute_gradient(weights, inputs): return weights.T @ inputs",
        "class TensorBuffer: def __init__(self, shape, dtype=float32):",
        "for i in range(len(matrix)): matrix[i] = matrix[i] * scalar",
    ],
    "finance": [
        "quarterly revenue growth exceeded analyst expectations by margin",
        "the balance sheet shows total assets and shareholder equity",
        "interest rate hikes affect bond yields and equity valuations",
    ],
}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main():
    print("Loading model...")
    reader = GGUFReader(MODEL_PATH, "r")
    lib = load_kernel()
    tok = GGUFTokenizer(reader)
    print(f"  vocab: {len(tok.vocab):,} tokens")

    engine = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
    faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
        VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
        VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
    engine.forward_single_token(1)
    lib.sync_device(); torch.cuda.synchronize()

    # Ground truth comes from the KERNEL's route log — one entry per top-k pick per
    # layer. Reading engine._topk_ids_dev from Python after the forward only ever
    # shows the LAST layer's picks (it is one shared buffer), which is the bug that
    # produced a bogus 600-expert / Jaccard-1.0 result on the first attempt.
    lib.set_route_logging.argtypes = [ctypes.c_int]
    lib.drain_route_log.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.drain_route_log.restype = ctypes.c_int
    lib.get_route_dropped.restype = ctypes.c_longlong
    RCAP = 32768
    rbuf = (ctypes.c_int * (RCAP * 2))()
    lib.set_route_logging(1)

    def experts_for_text(ids):
        """Run every token of one probe, harvest the true (layer, expert) set."""
        lib.drain_route_log(rbuf, RCAP)          # clear
        for t in ids:
            engine.forward_single_token(t)
        lib.sync_device(); torch.cuda.synchronize()
        n = lib.drain_route_log(rbuf, RCAP)
        return {(rbuf[2 * i], rbuf[2 * i + 1]) for i in range(n)}, n

    # Sanity gate: one token MUST produce exactly 75 layers x 8 picks = 600 entries.
    _, n1 = experts_for_text([1])
    print(f"\n  route-log sanity: 1 token -> {n1} picks (expect 600 = 75 layers x top-8)"
          f"  {'OK' if n1 == 600 else 'MISMATCH'}")
    if n1 != 600:
        print("  ABORT: route log is not capturing per-layer decisions correctly.")
        return

    print(f"\n=== HARVESTING REAL ROUTER DECISIONS ===")
    sets = {}
    for domain, texts in PROBES.items():
        sets[domain] = []
        for ti, text in enumerate(texts):
            ids = tok.encode(text, limit=24)
            if not ids:
                print(f"  {domain}[{ti}]: NO TOKENS (encode failed)")
                continue
            acc, npicks = experts_for_text(ids)
            sets[domain].append(acc)
            rt = tok.decode(ids)[:34].replace("\n", " ")
            print(f"  {domain:18s}[{ti}] {len(ids):>3} tok (drop {tok.last_dropped})"
                  f" {npicks:>5} picks -> {len(acc):>4} distinct | {rt!r}")

    print(f"\n=== WITHIN-DOMAIN vs ACROSS-DOMAIN OVERLAP ===")
    within, across = [], []
    for d, lst in sets.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                within.append((d, jaccard(lst[i], lst[j])))
    doms = list(sets.keys())
    for i in range(len(doms)):
        for j in range(i + 1, len(doms)):
            for a in sets[doms[i]]:
                for b in sets[doms[j]]:
                    across.append((f"{doms[i]}|{doms[j]}", jaccard(a, b)))

    w = np.mean([x[1] for x in within]) if within else 0.0
    a = np.mean([x[1] for x in across]) if across else 0.0
    print(f"  within-domain  mean Jaccard : {w:.4f}  (n={len(within)})")
    print(f"  across-domain  mean Jaccard : {a:.4f}  (n={len(across)})")
    print(f"  ratio within/across         : {w/a if a else float('inf'):.3f}x")

    for d in doms:
        vals = [x[1] for x in within if x[0] == d]
        if vals:
            print(f"    {d:18s} within = {np.mean(vals):.4f}")

    # The number that actually decides whether preloading works: how big is the
    # union working set for one domain, vs the whole model?
    print(f"\n=== WORKING-SET SIZE (this is what decides residency) ===")
    per_domain_union = {d: set().union(*lst) if lst else set() for d, lst in sets.items()}
    all_union = set().union(*per_domain_union.values()) if per_domain_union else set()
    for d, u in per_domain_union.items():
        print(f"  {d:18s} union = {len(u):>5} experts  ({100*len(u)/9600:5.2f}% of 9,600)")
    print(f"  {'ALL DOMAINS':18s} union = {len(all_union):>5} experts  ({100*len(all_union)/9600:5.2f}% of 9,600)")
    print(f"  VRAM can hold ~4,400 (2x5090, Q2_K) / ~5,100 (ternary)")

    verdict_semantic = (w / a) if a else float("inf")
    fits = len(all_union) <= 4400
    print(f"\n=== VERDICT ===")
    print(f"  routing is semantic (within > across)  : "
          f"{'YES' if verdict_semantic > 1.15 else 'NO / WEAK'}  ({verdict_semantic:.2f}x)")
    print(f"  a whole-corpus working set fits VRAM   : {'YES' if fits else 'NO'}"
          f"  ({len(all_union)} vs ~4,400 slots)")
    if verdict_semantic > 1.15 and fits:
        print(f"  -> PRESYNTHESIZER WILL WORK: preload the domain union, hit ~100% residency.")
    elif fits:
        print(f"  -> Preloading works by SIZE alone; semantic prediction adds little.")
    else:
        print(f"  -> Working set too large to preload; needs async streaming, not prediction.")

    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "within_mean": float(w), "across_mean": float(a),
        "ratio": float(verdict_semantic) if a else None,
        "domain_union_sizes": {d: len(u) for d, u in per_domain_union.items()},
        "all_union": len(all_union), "total_experts": 9600,
        "semantic_routing": bool(verdict_semantic > 1.15),
        "working_set_fits_vram": bool(fits),
    }
    p = r"E:\ABU\NEOCORTEX\expert_locality.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  written: {p}")


if __name__ == "__main__":
    main()
