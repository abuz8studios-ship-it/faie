"""Verify the EXPERT / FFN path against PyTorch. This is the one part never checked.

MLA was verified cos 1.000000. The expert path never was. Evidence it is the real
suspect: at L8 the FFN adds only 0.224 to a 0.811 residual, and disabling ALL routed
experts changes the final residual by 3%. If expert output magnitude is far too
small, attention dominates and the representation collapses — which is exactly the
observed failure.

Checks, in order of how damning they'd be:
  1. Q2_K / Q3_K dequant vs a独立 numpy dequant (catches a shared v1/v2 scale bug)
  2. One expert's SwiGLU forward: kernel vs PyTorch
  3. Shared-expert forward: kernel vs PyTorch
  4. Whole MoE layer delta: kernel vs PyTorch
"""
import sys, ctypes, io
sys.path.insert(0, r"E:\ABU\NEOCORTEX")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import torch
from gguf import GGUFReader
from faie_kernel import load_kernel
from faie_generate import (FAIEInference, dequant_any, HIDDEN_DIM, EXPERT_FFN_DIM,
                           N_EXPERTS, TOP_K)
import faie_generate

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
FAIE_DIR = r"E:\ABU\NEOCORTEX\faie_model"
VP = ctypes.c_void_p
DEV = "cuda:0"
L = 8                      # the layer where the collapse starts


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


print("loading...")
reader = GGUFReader(MODEL, "r")
lib = load_kernel()
eng = FAIEInference(reader, lib, FAIE_DIR, n_layers=78)
faie_generate.FAIEInference._moe_ffn_warm = lambda self, x, li: lib.full_moe_layer_v2(
    VP(x.data_ptr()), li, VP(self._norm_buf.data_ptr()),
    VP(self._gate_buf.data_ptr()), VP(self._mega_temp.data_ptr()))
eng.forward_single_token(1)
lib.sync_device(); torch.cuda.synchronize()
lib.set_router_scale.argtypes = [ctypes.c_float]

# ---------- 1. dequant sanity: kernel matvec vs PyTorch dense matvec ----------
print(f"\n=== 1. Q2_K matvec: kernel vs PyTorch dequant (layer {L} shared gate) ===")
t_sg = eng._get_tensor(f"blk.{L}.ffn_gate_shexp.weight")
W = torch.from_numpy(np.ascontiguousarray(dequant_any(t_sg))).to(DEV).float()  # [2048,6144]
print(f"  shared gate dequant shape {list(W.shape)}  rms={float((W*W).mean().sqrt()):.6f}")

torch.manual_seed(0)
x = (torch.randn(HIDDEN_DIM, device=DEV) * 0.05).float()
ref = W @ x

sh = eng._sh_cache.get(L)
y = torch.zeros(W.shape[0], dtype=torch.float32, device=DEV)
eng.cmv.q2k(eng.ffn_gpu[L]["sh_gate"]["gpu"] if "sh_gate" in eng.ffn_gpu.get(L, {}) else None,
            x, y, W.shape[0], HIDDEN_DIM) if False else None
# use the registered shared-expert pointer directly via the generic matvec
lw = eng.ffn_gpu.get(L, {})
key = "sh_gate" if "sh_gate" in lw else ("gate" if "gate" in lw else None)
if key:
    eng._matvec(lw[key], x, y)
    lib.sync_device(); torch.cuda.synchronize()
    print(f"  kernel matvec vs PyTorch: cos={cos(ref, y):+.6f}"
          f"   |ref|={float(ref.norm()):.4f}  |ker|={float(y.norm()):.4f}"
          f"   ratio={float(y.norm()/ref.norm()):.4f}")
else:
    print(f"  ffn_gpu[{L}] keys: {list(lw.keys())}  (no shared gate found)")

# ---------- 2. one routed expert SwiGLU: kernel vs PyTorch ----------
print(f"\n=== 2. routed expert SwiGLU (layer {L}) ===")
gt = eng._get_tensor(f"blk.{L}.ffn_gate_exps.weight")
ut = eng._get_tensor(f"blk.{L}.ffn_up_exps.weight")
dt = eng._get_tensor(f"blk.{L}.ffn_down_exps.weight")
from faie_generate import dequant_q2k_fast, dequant_q3k_fast
from faie_generate import resolve_qtype


def one_expert(t, e, rows, cols):
    """Dequantize a SINGLE expert slice. Dequantizing all 128 is 6 GB and OOMs."""
    qt = resolve_qtype(t.tensor_type)
    raw = np.ascontiguousarray(t.data[e]).tobytes()
    n = rows * cols
    f = dequant_q2k_fast(raw, n) if qt == "Q2_K" else dequant_q3k_fast(raw, n)
    return torch.from_numpy(np.asarray(f, dtype=np.float32).reshape(rows, cols).copy()).to(DEV)


def expert_fwd(e, h):
    g_ = one_expert(gt, e, EXPERT_FFN_DIM, HIDDEN_DIM) @ h
    u_ = one_expert(ut, e, EXPERT_FFN_DIM, HIDDEN_DIM) @ h
    mid = (g_ * torch.sigmoid(g_)) * u_
    return one_expert(dt, e, HIDDEN_DIM, EXPERT_FFN_DIM) @ mid


E = 0
Ge = one_expert(gt, E, EXPERT_FFN_DIM, HIDDEN_DIM)
Ue = one_expert(ut, E, EXPERT_FFN_DIM, HIDDEN_DIM)
De = one_expert(dt, E, HIDDEN_DIM, EXPERT_FFN_DIM)
print(f"  per-expert shapes G{list(Ge.shape)} U{list(Ue.shape)} D{list(De.shape)}")
g = Ge @ x
u = Ue @ x
h = (g * torch.sigmoid(g)) * u
ref_e = De @ h
del Ge, Ue, De
print(f"  PyTorch expert{E}: |gate|={float(g.norm()):.4f} |up|={float(u.norm()):.4f}"
        f" |silu*up|={float(h.norm()):.4f} |out|={float(ref_e.norm()):.4f}")

gw, uw, dw = eng._ensure_expert_cached(L, E)
gb = torch.zeros(EXPERT_FFN_DIM, dtype=torch.float32, device=DEV)
ub = torch.zeros(EXPERT_FFN_DIM, dtype=torch.float32, device=DEV)
ob = torch.zeros(HIDDEN_DIM, dtype=torch.float32, device=DEV)
eng.cmv.expert_swiglu(gw, uw, dw, x, gb, ub, ob)
lib.sync_device(); torch.cuda.synchronize()
print(f"  kernel  expert{E}: |out|={float(ob.norm()):.4f}"
      f"   cos={cos(ref_e, ob):+.6f}   ratio={float(ob.norm()/(ref_e.norm()+1e-12)):.4f}")

# ---------- 3. whole MoE layer delta ----------
print(f"\n=== 3. whole MoE layer {L}: kernel delta vs PyTorch delta ===")
ffn_w = eng.ffn_norms[L].float()
router = eng.routers[L].float()               # [128, 6144]
bias = eng.expert_biases.get(L)
hn = x / torch.sqrt((x * x).mean() + 1e-5) * ffn_w
logits = router @ hn
sc = torch.sigmoid(logits)
sel = sc + (bias.float() if bias is not None else 0)
idx = torch.topk(sel, TOP_K).indices
w = sc[idx]
w = w / w.sum()
routed = torch.zeros(HIDDEN_DIM, device=DEV)
for k in range(TOP_K):
    routed += float(w[k]) * expert_fwd(int(idx[k]), hn)
    torch.cuda.empty_cache()
sg = torch.from_numpy(np.ascontiguousarray(dequant_any(
    eng._get_tensor(f"blk.{L}.ffn_gate_shexp.weight")))).to(DEV).float()
su = torch.from_numpy(np.ascontiguousarray(dequant_any(
    eng._get_tensor(f"blk.{L}.ffn_up_shexp.weight")))).to(DEV).float()
sd = torch.from_numpy(np.ascontiguousarray(dequant_any(
    eng._get_tensor(f"blk.{L}.ffn_down_shexp.weight")))).to(DEV).float()
a = sg @ hn; b = su @ hn
shared = sd @ ((a * torch.sigmoid(a)) * b)
del sg, su, sd
torch.cuda.empty_cache()
print(f"  PyTorch routed(sum w=1) |{float(routed.norm()):.4f}|  shared |{float(shared.norm()):.4f}|")
for scale in (1.0, 2.5):
    tot = routed * scale + shared
    print(f"  PyTorch rscale={scale}: |ffn delta| = {float(tot.norm()):.4f}")
    if scale == 2.5:
        ref_full = tot.clone()
    else:
        ref_1 = tot.clone()

# Make every PyTorch-selected expert resident first, or the kernel silently skips
# them and the comparison is measuring residency, not correctness.
for k in range(TOP_K):
    eng._ensure_expert_cached(L, int(idx[k]))
lib.reset_expert_counters()

# CRITICAL: the preamble CUDA graph is captured against eng._x_buf. Passing any
# other buffer makes the graph replay read STALE data and silently route on the
# wrong vector. Must stage the input into _x_buf itself.
for scale in (1.0, 2.5):
    lib.set_router_scale(ctypes.c_float(scale))
    eng._x_buf.copy_(x)
    xk = eng._x_buf
    eng._moe_ffn_warm(xk, L)
    lib.sync_device(); torch.cuda.synchronize()
    d = xk - x
    r = ref_1 if scale == 1.0 else ref_full
    print(f"  kernel  rscale={scale}: |ffn delta| = {float(d.norm()):.4f}"
          f"   cos vs matching PyTorch = {cos(r, d):+.6f}"
          f"   ratio = {float(d.norm()/(r.norm()+1e-12)):.4f}")

req, exe = lib.get_experts_requested(), lib.get_experts_executed()
print(f"  residency during those 2 calls: {exe}/{req}")

# What did the KERNEL's own router pick? Only this layer ran, so the shared
# topk buffer is unambiguous here.
k_ids = eng._topk_ids_dev.tolist()
k_val = eng._topk_vals_dev.tolist()
print(f"\n  PyTorch top-8 : {idx.tolist()}")
print(f"  kernel  top-8 : {k_ids}")
print(f"  overlap       : {len(set(idx.tolist()) & set(k_ids))}/8")
print(f"  PyTorch w     : {[round(float(v),4) for v in w]}")
print(f"  kernel  w     : {[round(float(v),4) for v in k_val]}")
print(f"\n  If the id sets differ, the ROUTER is the bug. If they match but cos is ~0,")
print(f"  the assembly (ordering / weight pairing / buffer reuse) is the bug.")
lib.set_router_scale(ctypes.c_float(2.5))
print("\n  top-8 experts PyTorch picked:", idx.tolist())
print("  weights:", [round(float(v),4) for v in w])
