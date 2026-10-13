"""FACTOR-2 harness: rebuild MoE layers from a rank-r shared basis and let llama.cpp judge.

WHAT THIS PROVES (and what it doesn't):
expert_basis.py measures whether the 128 experts of a layer share a low-dim feature
basis. That is a claim about WEIGHTS. This script turns the claim into a claim about
BEHAVIOR: replace chosen layers' expert tensors with their rank-r reconstruction,
write a patched GGUF, and measure perplexity / the fixed oracle through llama.cpp's
own verified forward pass.

FAIE is deliberately not involved anywhere in the measurement path.

Reconstruction (per tensor, shared axis = the 6144 hidden dim):
    gate/up (128 x 2048x6144):  G = sum_e W_e^T W_e,  V = top-r eigvecs (6144 x r)
                                W_e' = (W_e V) V^T
    down    (128 x 6144x2048):  G = sum_e W_e W_e^T,  V = top-r eigvecs (6144 x r)
                                W_e' = V (V^T W_e)
Storage implied if adopted: per-expert coefficients (2048 x r) + one shared V (6144 x r)
per tensor per layer, instead of 128 private 2048x6144 matrices.

Patched tensors are written as F16 — gguf-py cannot encode Q2_K/Q3_K, and F16 makes the
comparison CLEANER, not weaker: the only change between original and patched model is
the rank truncation itself (F16 rounding of dequantized Q2_K values is far below quant
noise). Every other tensor is byte-copied.

Judgement (run after this script, llama.cpp b10069):
  llama-perplexity -m <patched.gguf> -f corpus/calib_holdout.txt -c 4096 -b 4096 -ngl 0
  vs the same command on the original. And the oracle:
  " Paris" must stay the argmax for "The capital of France is" (id 12089, ref logprob
  -0.445 on the unpatched model).

Usage:
  python reconstruct_layer.py --layers 40 --rank 1024 --out F:/GLM52_FACTOR/L40-r1024.gguf
  python reconstruct_layer.py --layers 10,25,40,55,70 --rank 512 --out F:/GLM52_FACTOR/x5-r512.gguf
  python reconstruct_layer.py --selftest          # tiny copy roundtrip on nomic-embed
"""
import argparse
import inspect
import os
import sys
import time

import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType
from gguf.quants import dequantize

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
SELFTEST_MODEL = r"E:\ABU\MODELS\llm\nomic-embed"  # dir; first .gguf inside
HIDDEN = 6144

_HAS_SUBTYPE = "sub_type" in inspect.signature(GGUFWriter.add_key_value).parameters


def copy_fields(reader, writer):
    """Copy every KV field except the virtuals and the arch key (writer adds it)."""
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for field in reader.fields.values():
        if field.name in skip:
            continue
        val = field.contents()
        # types[0] is the main type; arrays carry the element type in types[-1]
        main_type = field.types[0]
        if main_type == GGUFValueType.ARRAY and _HAS_SUBTYPE:
            writer.add_key_value(field.name, val, main_type, sub_type=field.types[-1])
        else:
            writer.add_key_value(field.name, val, main_type)


def ternarize(t, group=128):
    """BitNet b1.58 absmean ternary, group-wise — the SAME operating point Bonsai 27B
    published (Q2_0_g128: {-1,0,+1} + FP16 scale per 128 weights, 95% of FP16 kept on
    a 27B). Values stored as F16 so llama.cpp judges the ternarization itself, with no
    packing format in the way. This is ARB-8: nobody has ever tested whether post-hoc
    ternary preserves GLM-5.2 — every residency projection assumes it.
    """
    n_exp = int(t.shape[-1])
    w0 = dequantize(t.data[0], t.tensor_type)
    out = np.empty((n_exp, *w0.shape), dtype=np.float16)
    errs = np.empty(n_exp)
    for e in range(n_exp):
        w = dequantize(t.data[e], t.tensor_type).astype(np.float32)
        flat = w.reshape(-1, group)
        s = np.mean(np.abs(flat), axis=1, keepdims=True) + 1e-12   # absmean scale
        q = np.clip(np.round(flat / s), -1, 1)
        wt = (q * s).reshape(w.shape)
        errs[e] = float(np.linalg.norm(wt - w) / np.linalg.norm(w))
        out[e] = wt.astype(np.float16)
    return out, errs


def shared_basis_reconstruct(t, rank, D=None):
    """Return (W_recon float16 stack, per-expert relative Frobenius errors).

    D=None      -> raw Frobenius basis (v1 behavior).
    D=(vector)  -> imatrix-whitened DATA-METRIC basis (FACTOR-2b). The projection
                   is chosen to minimize E||(W - W')x||^2 under diag(E[x^2])=D:
        gate/up (shared axis = input cols, |D|=6144):
            Gt = sum_e (W_e S)^T (W_e S), S = diag(sqrt(D));  V = top-r eig(Gt)
            W' = (W S V) V^T S^{-1}          (oblique in raw space, orth. in whitened)
        down    (shared axis = output rows, |D|=2048 on the input cols):
            Gt = sum_e W_e diag(D) W_e^T;    V = top-r eig(Gt);  W' = V V^T W
    """
    n_exp = int(t.shape[-1])
    w0 = dequantize(t.data[0], t.tensor_type)
    gateup = w0.shape[1] == HIDDEN
    s = np.sqrt(np.asarray(D, dtype=np.float64) / np.mean(D)).astype(np.float32) if D is not None else None

    G = np.zeros((HIDDEN, HIDDEN), dtype=np.float64)
    experts = []
    for e in range(n_exp):
        w = dequantize(t.data[e], t.tensor_type).astype(np.float32)
        experts.append(w)
        if gateup:                                     # gate/up: shared axis = cols
            ww = w * s[None, :] if s is not None else w
            G += ww.T.astype(np.float64) @ ww.astype(np.float64)
        else:                                          # down: shared axis = rows
            if s is not None:
                G += (w.astype(np.float64) * (s.astype(np.float64) ** 2)[None, :]) @ w.T.astype(np.float64)
            else:
                G += w.astype(np.float64) @ w.T.astype(np.float64)
    evals, evecs = np.linalg.eigh(G)
    V = np.ascontiguousarray(evecs[:, ::-1][:, :rank].astype(np.float32))  # 6144 x r

    out = np.empty((n_exp, *experts[0].shape), dtype=np.float16)
    errs = np.empty(n_exp)
    for e, w in enumerate(experts):
        if gateup:
            if s is not None:
                w2 = (((w * s[None, :]) @ V) @ V.T) / s[None, :]
            else:
                w2 = (w @ V) @ V.T
        else:
            w2 = V @ (V.T @ w)
        errs[e] = float(np.linalg.norm(w2 - w) / np.linalg.norm(w))
        out[e] = w2.astype(np.float16)
    energy = np.clip(evals[::-1], 0, None)
    kept = float(energy[:rank].sum() / energy.sum())   # in the chosen metric
    return out, errs, kept


def patch(model_path, out_path, layers, rank, mode="rank", metric="raw"):
    Dmap = {}
    if metric == "imatrix":
        from expert_basis_v2 import load_imatrix_D
        Dmap = load_imatrix_D()
        print(f"  imatrix D vectors loaded: {len(Dmap)}")
    reader = GGUFReader(model_path, "r")
    arch = reader.fields["general.architecture"].contents()
    targets = {f"blk.{L}.ffn_{k}_exps.weight" for L in layers for k in ("gate", "up", "down")}
    present = {t.name for t in reader.tensors}
    missing = targets - present
    if missing:
        raise SystemExit(f"missing tensors (dense layer?): {sorted(missing)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = GGUFWriter(out_path, arch)
    copy_fields(reader, writer)

    stats = {}
    t0 = time.time()
    for t in reader.tensors:
        if t.name in targets:
            print(f"  {mode.upper()}: {t.name} ({t.tensor_type.name})", flush=True)
            if mode == "ternary":
                w, errs = ternarize(t)
                stats[t.name] = {"relF_mean": float(errs.mean()),
                                 "relF_max": float(errs.max())}
                print(f"    relF mean {errs.mean():.4f} max {errs.max():.4f}", flush=True)
            else:
                D = Dmap.get(t.name) if metric == "imatrix" else None
                if metric == "imatrix" and D is None:
                    raise SystemExit(f"--metric imatrix but no D for {t.name}")
                w, errs, kept = shared_basis_reconstruct(t, rank, D=D)
                stats[t.name] = {"kept_energy": kept, "metric": metric,
                                 "relF_mean": float(errs.mean()),
                                 "relF_max": float(errs.max())}
                print(f"    kept energy({metric}) {kept:.4f}  relF mean "
                      f"{errs.mean():.4f} max {errs.max():.4f}", flush=True)
            # numpy logical shape -> writer handles F16 natively
            writer.add_tensor(t.name, w)
            del w
        else:
            # byte-exact copy of the original quantized data
            writer.add_tensor(t.name, t.data, raw_shape=t.data.shape,
                              raw_dtype=t.tensor_type)
    print(f"  writing {out_path} ...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"  done in {time.time()-t0:.0f}s")
    return stats


def selftest():
    """Copy a small GGUF with zero replacements; verify fields+tensor bytes match."""
    import glob
    src = glob.glob(os.path.join(SELFTEST_MODEL, "*.gguf"))[0]
    dst = os.path.join(os.environ.get("TEMP", "."), "selftest_copy.gguf")
    r1 = GGUFReader(src, "r")
    arch = r1.fields["general.architecture"].contents()
    w = GGUFWriter(dst, arch)
    copy_fields(r1, w)
    for t in r1.tensors:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False); w.close()

    r2 = GGUFReader(dst, "r")
    assert len(r1.tensors) == len(r2.tensors), "tensor count mismatch"
    for a, b in zip(r1.tensors, r2.tensors):
        assert a.name == b.name and a.tensor_type == b.tensor_type
        assert np.array_equal(np.asarray(a.data), np.asarray(b.data)), f"bytes differ: {a.name}"
    n_kv1 = sum(1 for f in r1.fields if not f.startswith("GGUF."))
    n_kv2 = sum(1 for f in r2.fields if not f.startswith("GGUF."))
    print(f"SELFTEST OK: {len(r2.tensors)} tensors byte-identical, kv {n_kv1}->{n_kv2}")
    del r2  # release the mmap before deleting; Windows locks mapped files
    import gc
    gc.collect()
    try:
        os.remove(dst)
    except PermissionError:
        pass  # harmless leftover in TEMP


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="40")
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--mode", choices=("rank", "ternary"), default="rank")
    ap.add_argument("--metric", choices=("raw", "imatrix"), default="raw")
    ap.add_argument("--out", type=str, default=r"F:\GLM52_FACTOR\patched.gguf")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    layers = [int(x) for x in a.layers.split(",")]
    print(f"patching layers {layers} mode={a.mode} metric={a.metric} rank={a.rank} -> {a.out}")
    stats = patch(a.model, a.out, layers, a.rank, a.mode, a.metric)
    import json
    sp = a.out + ".stats.json"
    json.dump({"layers": layers, "rank": a.rank, "mode": a.mode, "metric": a.metric,
               "stats": stats}, open(sp, "w"), indent=2)
    print(f"wrote {sp}")
