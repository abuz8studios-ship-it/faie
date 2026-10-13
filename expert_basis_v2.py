"""FACTOR-1b: the ACTIVATION-METRIC basis measurement — rank in the space the data pays for.

WHY v2 EXISTS — v1's verdict and its blind spot:

expert_basis.py (v1) measured the RAW weight spectrum and found it flat:
feature-rank@95% ~ 5,300-5,700 of 6,144 at layers 3/40 — no shared low-rank basis
in the Frobenius metric. Honest result, correctly reported.

But raw Frobenius is the WRONG metric for behavior. The model never sees isotropic
inputs: E[x xT] is heavily anisotropic, and reconstruction error only matters in
directions the data excites. The llama-imatrix files built from Ahmad's real corpus
(ARB-4, verified from their own headers) hold exactly diag(E[x_i^2]) for the input
of every MoE matmul. Measuring rank in the imatrix-whitened metric answers the
question that decides the 32 GB budget:

    "How many directions does the model actually USE?"
not "How many directions do the weights SPAN?"

Quantization noise (we read Q2_K/Q3_K) is near-isotropic and inflates the raw tail;
whitening by the data metric suppresses exactly the directions where noise lives and
data does not. If the whitened spectrum is ALSO flat, factorization is dead in the
same grave as pruning, and this script says so.

THE MATH (diag-covariance approximation — the same one llama.cpp quantization uses):

  gate/up: y = W x, x in R^6144, D = diag imatrix (6144). x = D^(1/2) u, E[uuT]~I.
           Basis in u-space: G~ = D^(1/2) (Sum_e W_eT W_e) D^(1/2) = D^(1/2) G D^(1/2)
           -> reuses the SAME raw Gram G, whitened post-accumulation.
  down:    y = W h, h in R^2048, D_h = diag imatrix (2048), shared axis = OUTPUT rows.
           Error of output-space projection (I-VVT)W h in the data metric:
           G~ = Sum_e W_e D_h W_eT  -> D goes INSIDE the accumulation.

ALSO MEASURED IN THE SAME PASS (the two fallback structures if shared-basis fails):
  * per-expert rank: exact SVD of N_SVD sampled experts per tensor (SVD-LLM / MoE-SVD
    form: each expert privately low-rank even if the union spans everything).
  * shared-mean split: energy fraction of W-bar = mean_e W_e, plus the spectrum of the
    delta Gram G_delta = G - 128 W-barT W-bar (D2-MoE form: share the mean, low-rank
    the deltas). Computed from already-accumulated G and W-bar — no extra pass.

HONESTY CONSTRAINTS (inherited from v1, non-negotiable):
  * dequantize via gguf.quants.dequantize ONLY — never FAIE's dequant.
  * the imatrix D is validated before use: positive share, finite, size match; a
    tensor whose D fails validation is reported RAW-ONLY, never silently whitened.
  * this measures WEIGHTS UNDER A DATA METRIC. It still does not prove a patched
    model talks — reconstruct_layer.py + llama-perplexity remain the behavior judge.

Usage:
  python expert_basis_v2.py 3 40 77
  python expert_basis_v2.py 40           # single layer
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
IMATRIX = r"E:\ABU\NEOCORTEX\corpus\im_calib_workload.gguf"
OUT = r"E:\ABU\NEOCORTEX\expert_basis_v2.json"
GRAM_DIR = r"E:\ABU\NEOCORTEX\grams"

HIDDEN = 6144
FFN = 2048
N_EXPERT = 128
N_SVD = 4                      # exact per-expert SVDs sampled per tensor
THRESH = (0.90, 0.95, 0.99)
SEED = 0


def eff_rank(eigvals, thresholds=THRESH):
    v = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    v = np.clip(v, 0.0, None)
    tot = v.sum()
    if tot <= 0:
        return {f"{t:.2f}": None for t in thresholds}
    cum = np.cumsum(v) / tot
    return {f"{t:.2f}": int(np.searchsorted(cum, t)) + 1 for t in thresholds}


def shares(eigvals, ranks=(512, 1024, 1536, 2048)):
    v = np.clip(np.sort(np.asarray(eigvals, dtype=np.float64))[::-1], 0, None)
    tot = v.sum()
    return {str(r): float(v[:r].sum() / tot) for r in ranks if r <= len(v)} if tot > 0 else {}


def participation_ratio(eigvals):
    v = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    s2 = (v ** 2).sum()
    return float(v.sum() ** 2 / s2) if s2 > 0 else 0.0


def load_imatrix_D():
    """Return {tensor_name: D vector} of per-input-channel mean squares.

    llama.cpp imatrix GGUF: '<mm-input-name>' holds summed squares, companion
    '<name>.counts' the chunk counts. For MUL_MAT_ID the sums may be stored
    per-expert (128 x ne0) or flat (ne0). Either is accepted; per-expert sums are
    count-weighted-averaged to one D per tensor. D is validated, not trusted.
    """
    r = GGUFReader(IMATRIX, "r")
    tens = {t.name: t for t in r.tensors}
    out = {}
    for name, t in tens.items():
        # llama.cpp imatrix GGUF: sums live in '<mm-name>.in_sum2' (older files use
        # the bare mm name); counts in '<mm-name>.counts'. Accept both layouts.
        if name.endswith(".counts") or "_exps" not in name:
            continue
        stem = name[:-len(".in_sum2")] if name.endswith(".in_sum2") else name
        sums = np.array(t.data, dtype=np.float64)
        cnt_t = tens.get(stem + ".counts") or tens.get(name + ".counts")
        cnt = np.array(cnt_t.data, dtype=np.float64).ravel() if cnt_t is not None else None
        if sums.ndim == 2:                                   # (n_expert, ne0)
            if cnt is not None and cnt.size == sums.shape[0]:
                w = cnt / max(cnt.sum(), 1e-30)
                D = (sums * w[:, None]).sum(axis=0)
                per_chunk = max(cnt.mean(), 1e-30)
                D = D / per_chunk
            else:
                D = sums.mean(axis=0)
        else:
            D = sums.ravel()
            if cnt is not None and cnt.size == 1 and cnt[0] > 0:
                D = D / cnt[0]
        base = stem[:-len(".weight")] if stem.endswith(".weight") else stem
        out[base + ".weight"] = D
    return out


def validate_D(D, want_size, label):
    if D is None:
        return None, f"{label}: no imatrix entry"
    D = np.asarray(D, dtype=np.float64).ravel()
    if D.size != want_size:
        return None, f"{label}: imatrix size {D.size} != {want_size}"
    if not np.all(np.isfinite(D)):
        return None, f"{label}: non-finite imatrix values"
    pos = float((D > 0).mean())
    if pos < 0.99:
        return None, f"{label}: only {100*pos:.1f}% positive channels"
    D = D / D.mean()                       # scale-free whitening
    return D, None


def analyse(t, D, rng, label):
    n_exp = int(t.shape[-1])
    w0 = dequantize(t.data[0], t.tensor_type).astype(np.float32)
    rows, cols = w0.shape
    is_gateup = cols == HIDDEN             # shared axis = input cols (6144)
    d_size = HIDDEN if is_gateup else FFN  # imatrix indexes the INPUT axis
    Dv, derr = validate_D(D, d_size, label)
    if derr:
        print(f"    !! {derr} -> whitened spectrum will be SKIPPED", flush=True)

    G = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    Gw = np.zeros((HIDDEN, HIDDEN), dtype=np.float64) if (not is_gateup and Dv is not None) else None
    Wbar = np.zeros((rows, cols), dtype=np.float64)
    svd_idx = sorted(rng.choice(n_exp, size=N_SVD, replace=False).tolist())
    per_expert_svd = {}
    norms = np.empty(n_exp)

    t0 = time.time()
    for e in range(n_exp):
        w = dequantize(t.data[e], t.tensor_type).astype(np.float32)
        wd = w.astype(np.float64)
        norms[e] = float((wd * wd).sum())
        Wbar += wd
        if is_gateup:
            G += wd.T @ wd
        else:
            G += wd @ wd.T
            if Gw is not None:
                Gw += (wd * Dv[None, :]) @ wd.T
        if e in svd_idx:
            sv = np.linalg.svd(w, compute_uv=False).astype(np.float64)
            per_expert_svd[str(e)] = {
                "rank": eff_rank(sv ** 2),
                "pr": participation_ratio(sv ** 2),
            }
        del w, wd
        if e % 32 == 31:
            print(f"      expert {e+1}/{n_exp}  ({time.time()-t0:.0f}s)", flush=True)
    Wbar /= n_exp
    print(f"      accumulated in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(GRAM_DIR, exist_ok=True)
    np.save(os.path.join(GRAM_DIR, f"{label}.G.npy"), G)

    res = {"rows": rows, "cols": cols, "qtype": t.tensor_type.name}

    ev_raw = np.linalg.eigvalsh(G)[::-1]
    res["raw"] = {"eff_rank": eff_rank(ev_raw), "pr": participation_ratio(ev_raw),
                  "shares": shares(ev_raw)}

    if Dv is not None:
        if is_gateup:
            s = np.sqrt(Dv)
            Gt = (G * s[None, :]) * s[:, None]     # D^1/2 G D^1/2
        else:
            Gt = Gw
        ev_w = np.linalg.eigvalsh(Gt)[::-1]
        res["whitened"] = {"eff_rank": eff_rank(ev_w), "pr": participation_ratio(ev_w),
                           "shares": shares(ev_w)}
        del Gt

    # shared-mean + delta split (D2-MoE form), from G and Wbar alone
    if is_gateup:
        Gm = Wbar.T @ Wbar
    else:
        Gm = Wbar @ Wbar.T
    Gd = G - n_exp * Gm
    mean_energy = float(n_exp * np.trace(Gm) / max(np.trace(G), 1e-30))
    ev_d = np.linalg.eigvalsh(Gd)[::-1]
    res["mean_delta"] = {"mean_energy_frac": mean_energy,
                         "delta_eff_rank": eff_rank(ev_d),
                         "delta_shares": shares(ev_d)}
    res["per_expert_svd"] = per_expert_svd
    del G, Gw, Gd, Gm, Wbar
    gc.collect()

    fr_raw = res["raw"]["eff_rank"]["0.95"]
    fr_w = res.get("whitened", {}).get("eff_rank", {}).get("0.95")
    pe = [v["rank"]["0.95"] for v in per_expert_svd.values()]
    print(f"    -> {label}: raw rank@95={fr_raw}"
          + (f"  WHITENED rank@95={fr_w}" if fr_w else "  (no whitening)")
          + f"  mean-energy={mean_energy:.3f}  per-expert rank@95={pe}", flush=True)
    return res


def main():
    layers = [int(a) for a in sys.argv[1:]] or [3, 40, 77]
    print(f"model   : {MODEL}")
    print(f"imatrix : {IMATRIX}")
    Dmap = load_imatrix_D()
    print(f"imatrix D vectors loaded: {len(Dmap)} MoE tensors")

    r = GGUFReader(MODEL, "r")
    by_name = {t.name: t for t in r.tensors}
    rng = np.random.default_rng(SEED)

    results = {}
    for L in layers:
        print(f"\n=== layer {L} ===", flush=True)
        lr = {}
        for kind in ("gate", "up", "down"):
            name = f"blk.{L}.ffn_{kind}_exps.weight"
            if name not in by_name:
                print(f"    {kind}: MISSING")
                continue
            print(f"    {kind}: analysing ({by_name[name].tensor_type.name})", flush=True)
            lr[kind] = analyse(by_name[name], Dmap.get(name), rng, f"L{L}.{kind}")
        results[str(L)] = lr

    print("\n=== VERDICT TABLE (rank@95%: raw vs data-metric) ===")
    for L, lr in results.items():
        for kind, res in lr.items():
            raw = res["raw"]["eff_rank"]["0.95"]
            wh = res.get("whitened", {}).get("eff_rank", {}).get("0.95", "—")
            s1024 = res.get("whitened", res["raw"]).get("shares", {}).get("1024")
            print(f"  L{L}.{kind:5s} raw {raw:>5}  whitened {wh:>5}  "
                  f"share@1024 {s1024 if s1024 is None else f'{s1024:.3f}'}")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("NOTE: weights-under-a-metric only. Behavior is judged by llama.cpp on a "
          "patched GGUF, never here.")


if __name__ == "__main__":
    main()
