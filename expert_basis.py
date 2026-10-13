"""Do GLM-5.2's 128 experts-per-layer actually live in a shared low-dimensional basis?

WHY THIS EXISTS — the reframe:

The imatrix ground truth (76M routing events, Ahmad's real corpus) measured expert
usage concentration at 3.84x vs a uniform-random null. The goal needed 38x. I reported
that as "pruning is dead," which is true and stays true.

But near-uniform usage across 9,600 identically-shaped experts is ALSO the signature of
REDUNDANCY. Specialists produce spiky usage. These do not. Redundancy is not something
you delete -- it is something you FACTOR.

    pruning        asks: which 240 of 9,600 do I keep?          -> measured dead
    factorization  asks: do 9,600 live in a shared basis?       -> never measured

If a shared basis of rank r exists, an expert stops being a private 37.7M-weight matrix
and becomes a small coefficient block against a basis its 127 siblings share:

    per layer, independent : 128 * 2048 * 6144 * 3        = 4.83 B weights
    per layer, shared basis: 3*(128*2048*r + 6144*r)      = 804,864 * r weights

    r=1024 -> 5.86x  -> 75 layers at ternary 1.58bpw = 12.2 GB   (fits 32 GB)
    r= 512 -> 11.7x  -> 75 layers at ternary 1.58bpw =  6.1 GB   (fits easily)
    today            -> 75 layers at ternary 1.58bpw = 73.5 GB   (fits nothing)

TWO INDEPENDENT STRUCTURES, both measured here:

  (A) EXPERT-SPACE rank. Treat each expert as one vector in R^12582912 and ask the rank
      of the 128x128 Gram. Low rank => experts are combinations of a few prototypes.
      Measured on a Johnson-Lindenstrauss sketch (exact is 128 x 12.6M = 6.4 GB), then
      the sketch is VALIDATED against exact inner products on sampled pairs. If the
      validation fails, the sketch numbers are discarded rather than reported.

  (B) FEATURE-SPACE rank. Accumulate G = sum_e W_e^T W_e over the shared 6144 axis,
      exactly, one expert at a time. Its eigenspectrum is the shared subspace all 128
      experts read from (gate/up) or write into (down).

Both are needed: (A) low means few distinct experts; (B) low means each expert only
touches a thin slice of the hidden space. Either one alone buys compression; together
they multiply.

HONESTY CONSTRAINTS BUILT IN:
  * Dequantization goes through gguf.quants.dequantize -- the INDEPENDENT reference.
    Never FAIE's own dequant. Two wrong implementations agreeing is what cost seven
    weeks this morning (see faie-dequant-root-cause-2026-07-26).
  * The random sketch is validated, not trusted. Bad validation => refuse to report.
  * A "shared basis exists" claim here is a claim about WEIGHTS ONLY. It does not prove
    the reconstructed model talks. That needs a reconstruction-error -> logit test,
    which is the next script, not this one.

Usage:
  python expert_basis.py                 # default sample of layers
  python expert_basis.py 3 40 77         # specific layers
"""
import gc
import json
import os
import sys
import time

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
OUT = r"E:\ABU\NEOCORTEX\expert_basis.json"

N_EXPERT = 128
HIDDEN = 6144
FFN = 2048
# JL sketch dims: expert vector -> A^T W B, flattened to SK1*SK2 dims.
# 64x64 = 4096 dims gives inner-product error ~1/sqrt(4096) ~ 1.6%, plenty to read a
# spectrum. Validated below against exact pairs before anything is reported.
SK1, SK2 = 64, 64
N_VALIDATE_PAIRS = 12
SEED = 0

# Energy thresholds to report effective rank at.
THRESH = (0.90, 0.95, 0.99)


def eff_rank(eigvals, thresholds=THRESH):
    """Number of components needed to hold each fraction of total energy."""
    v = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    v = np.clip(v, 0.0, None)
    tot = v.sum()
    if tot <= 0:
        return {f"{t:.2f}": None for t in thresholds}
    cum = np.cumsum(v) / tot
    return {f"{t:.2f}": int(np.searchsorted(cum, t)) + 1 for t in thresholds}


def participation_ratio(eigvals):
    """(sum L)^2 / sum(L^2) -- a threshold-free 'effective number of directions'."""
    v = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    s2 = (v ** 2).sum()
    return float(v.sum() ** 2 / s2) if s2 > 0 else 0.0


def load_expert(t, e):
    """Dequantize expert e of a 3D MoE tensor to float32, via the INDEPENDENT reference.

    gguf-py exposes .data already reshaped so that axis 0 indexes the expert. Slicing
    the raw quantized bytes per expert keeps peak memory at one expert (~50 MB fp32)
    instead of the whole 128-expert stack (~6.4 GB).
    """
    raw = t.data[e]
    w = dequantize(raw, t.tensor_type)
    return np.ascontiguousarray(w, dtype=np.float32)


def analyse_tensor(t, rng, label):
    """Return expert-space and feature-space spectra for one MoE tensor."""
    n_exp = int(t.shape[-1])
    w0 = load_expert(t, 0)
    rows, cols = w0.shape          # gate/up: (2048, 6144); down: (6144, 2048)
    shared_axis = 1 if cols == HIDDEN else 0
    print(f"    {label}: {n_exp} experts of {rows}x{cols} ({t.tensor_type.name}), "
          f"shared 6144-axis = {'cols' if shared_axis == 1 else 'rows'}")

    # Sketch matrices for the JL projection: A^T W B.
    A = rng.standard_normal((rows, SK1), dtype=np.float32) / np.sqrt(rows)
    B = rng.standard_normal((cols, SK2), dtype=np.float32) / np.sqrt(cols)

    sketch = np.empty((n_exp, SK1 * SK2), dtype=np.float64)
    # Exact feature-space Gram over the shared 6144 axis, accumulated in float64.
    G = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    norms = np.empty(n_exp, dtype=np.float64)
    # Keep a few whole experts to validate the sketch against exact inner products.
    val_idx = sorted(rng.choice(n_exp, size=8, replace=False).tolist())
    kept = {}

    t0 = time.time()
    for e in range(n_exp):
        w = load_expert(t, e)
        sketch[e] = (A.T @ w @ B).ravel().astype(np.float64)
        norms[e] = float(np.dot(w.ravel(), w.ravel()))
        if shared_axis == 1:
            G += (w.T.astype(np.float64) @ w.astype(np.float64))
        else:
            G += (w.astype(np.float64) @ w.T.astype(np.float64))
        if e in val_idx:
            kept[e] = w.copy()
        del w
        if e % 32 == 31:
            print(f"      expert {e+1}/{n_exp}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"      accumulated in {time.time()-t0:.0f}s", flush=True)

    # ---- validate the sketch before trusting any expert-space number -------------
    pairs = []
    ks = sorted(kept)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pairs.append((ks[i], ks[j]))
    pairs = pairs[:N_VALIDATE_PAIRS]
    exact_c, sketch_c = [], []
    for i, j in pairs:
        wi, wj = kept[i].ravel(), kept[j].ravel()
        exact_c.append(float(np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj))))
        si, sj = sketch[i], sketch[j]
        sketch_c.append(float(np.dot(si, sj) / (np.linalg.norm(si) * np.linalg.norm(sj))))
    kept.clear()
    gc.collect()
    exact_c = np.array(exact_c)
    sketch_c = np.array(sketch_c)
    max_err = float(np.max(np.abs(exact_c - sketch_c))) if len(exact_c) else float("nan")
    sketch_ok = bool(max_err < 0.05)
    print(f"      sketch validation on {len(exact_c)} exact pairs: "
          f"max |cos_exact - cos_sketch| = {max_err:.4f} -> "
          f"{'OK' if sketch_ok else 'REJECTED'}")

    # ---- (A) expert-space spectrum ----------------------------------------------
    expert_block = None
    if sketch_ok:
        Ge = sketch @ sketch.T                       # 128 x 128
        d = np.sqrt(np.clip(np.diag(Ge), 1e-30, None))
        C = Ge / np.outer(d, d)                      # cosine matrix between experts
        ev = np.linalg.eigvalsh(Ge)[::-1]
        off = C[~np.eye(n_exp, dtype=bool)]
        expert_block = {
            "eff_rank": eff_rank(ev),
            "participation_ratio": participation_ratio(ev),
            "mean_abs_cos_between_experts": float(np.mean(np.abs(off))),
            "max_abs_cos_between_experts": float(np.max(np.abs(off))),
            "top8_eigval_share": float(np.sort(ev)[::-1][:8].sum() / ev.sum()),
            "sketch_max_err": max_err,
            "n_experts": n_exp,
        }

    # ---- (B) feature-space spectrum ---------------------------------------------
    evg = np.linalg.eigvalsh(G)[::-1]
    feature_block = {
        "eff_rank": eff_rank(evg),
        "participation_ratio": participation_ratio(evg),
        "dim": HIDDEN,
        "top1024_share": float(np.clip(evg, 0, None)[:1024].sum()
                               / np.clip(evg, 0, None).sum()),
        "top512_share": float(np.clip(evg, 0, None)[:512].sum()
                              / np.clip(evg, 0, None).sum()),
    }
    del G
    gc.collect()
    return {"expert_space": expert_block, "feature_space": feature_block,
            "rows": rows, "cols": cols, "qtype": t.tensor_type.name}


def main():
    layers = [int(a) for a in sys.argv[1:]] or [3, 40, 77]
    print(f"reading {MODEL}")
    r = GGUFReader(MODEL, "r")
    by_name = {t.name: t for t in r.tensors}
    rng = np.random.default_rng(SEED)

    results = {}
    for L in layers:
        print(f"\n=== layer {L} ===", flush=True)
        layer_res = {}
        for kind in ("gate", "up", "down"):
            name = f"blk.{L}.ffn_{kind}_exps.weight"
            if name not in by_name:
                print(f"    {kind}: MISSING ({name}) -- layer may be dense")
                continue
            layer_res[kind] = analyse_tensor(by_name[name], rng, kind)
        results[str(L)] = layer_res

        # Compression implied by the measured feature-space rank, reported per layer.
        for kind, res in layer_res.items():
            fr = res["feature_space"]["eff_rank"]["0.95"]
            er = (res["expert_space"] or {}).get("eff_rank", {}).get("0.95")
            print(f"    -> {kind}: feature rank@95% = {fr}/{HIDDEN}"
                  + (f", expert rank@95% = {er}/{N_EXPERT}" if er else ""))

    # ---- what it means for the 32 GB budget -------------------------------------
    print("\n=== implied budget (feature-space basis only, 95% energy) ===")
    ranks = [res["feature_space"]["eff_rank"]["0.95"]
             for lr in results.values() for res in lr.values()]
    if ranks:
        rmax = int(max(ranks))
        rmean = float(np.mean(ranks))
        for label, rr in (("mean measured rank", rmean), ("worst measured rank", rmax)):
            per_layer = 3 * (N_EXPERT * FFN * rr + HIDDEN * rr)
            orig = 3 * N_EXPERT * FFN * HIDDEN
            gb158 = per_layer * 75 * 1.58 / 8 / 1024 ** 3
            print(f"  {label:20s} r={rr:7.1f}  compression {orig/per_layer:5.2f}x  "
                  f"75 layers @1.58bpw = {gb158:6.2f} GB")
        print("  (today, independent experts @1.58bpw = 73.5 GB)")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("\nNOTE: this measures WEIGHT redundancy only. It does not prove a "
          "reconstructed model generates correctly -- that is the next test.")


if __name__ == "__main__":
    main()
