# FAIE HANDOFF — fix the architectural bug that stops it producing coherent text

**Date:** 2026-07-26 · **From:** Opus 5 session `7b6ecb39` · **Owner:** Ahmad Abu Zait

---

## 1. THE ONE JOB

`E:\ABU\NEOCORTEX` contains **FAIE**, a custom CUDA inference engine for
`E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf` (glm-dsa, 380.93B params,
129.4 GB). Its CUDA kernels are fast and **numerically exact**, but the assembled
model emits garbage.

**Success = `python faie_sweep.py` reports `rank(Paris) = 0`.**
Today it reports ~**69,775 of 154,880** (random; median is 77,440).

Do **not** re-do performance work. Throughput is already proven (32.03 tok/s at
100% expert residency, bit-identical to the reference kernels). This is purely a
**correctness** bug.

---

## 2. THE ORACLE — llama.cpp runs this EXACT file correctly

This is the ground truth. Use it constantly.

```bash
cd "E:/ABU/MODELS/runtime/llama-b10069-cuda13"
./llama-server.exe -m "E:/ABU/MODELS/llm/glm-5.2-ream/GLM-5.2-REAM-Q2_K.gguf" \
  --host 127.0.0.1 --port 8099 -ngl 99 -sm layer \
  -ot 'blk\.([0-9]|[1-4][0-9]|5[01]|7[6-8])\.ffn_.*_exps\.=CPU' \
  -ot 'blk\.(5[2-9]|6[0-3])\.ffn_.*_exps\.=CUDA0' \
  -ot 'blk\.(6[4-9]|7[0-5])\.ffn_.*_exps\.=CUDA1' \
  -c 16384 -t 16 --no-warmup --reasoning-budget 400
```
Loads in ~36 s. GPU0 ~26.7 GB, GPU1 ~24.9 GB. ~7 tok/s.

**Captured oracle** for prompt `"The capital of France is"` (temp 0):

| rank | token | id | logprob |
|---|---|---|---|
| 1 | ` Paris` | **12089** | −0.4455 |
| 2 | ` usually` | 5990 | −3.5595 |
| 3 | ` France` | 9621 | −3.8038 |
| 4 | ` mostly` | 10003 | −3.9930 |
| 5 | ` most` | 1429 | −4.0700 |

Get more via `POST /v1/completions` with `"n_probs":10,"logprobs":10`.

**Tokenizer is byte-identical to FAIE's**: prompt → `[785, 6722, 315, 9621, 374]`.

---

## 3. ARCHITECTURE — all read from the GGUF, all confirmed

```
arch glm-dsa · block_count 79  -> USE LAYERS 0..77 (78 is the nextn/MTP layer)
leading_dense_block_count 3    -> layers 0-2 dense FFN, layers 3-77 MoE
embedding_length 6144 · vocab 154880
attention.head_count 64 · head_count_kv 1
q_lora_rank 2048 · kv_lora_rank 512
key_length 576 (=512+64) · value_length 512
key_length_mla 256 · value_length_mla 256
  => qk_nope_head_dim 192, qk_rope_head_dim 64, v_head_dim 256
rope.freq_base 8000000.0 · rope.dimension_count 64
attention.layer_norm_rms_epsilon 1e-5      (NOTE: kernel hardcodes 1e-6)
expert_count 128 · expert_used_count 8 · expert_feed_forward_length 2048
expert_shared_count 1
expert_gating_func 2 (SIGMOID) · expert_weights_norm true · expert_weights_scale 2.5
attention.indexer.{head_count 32, key_length 128, top_k 2048}
```

Per-layer attention tensors (GGUF ne, column-major, ne0 = contracted dim):
```
attn_norm        [6144]            F32
attn_q_a         [6144, 2048]      Q2_K
attn_q_a_norm    [2048]            F32
attn_q_b         [2048, 16384]     Q2_K   -> 64 heads x 256 (192 nope | 64 rope)
attn_kv_a_mqa    [6144, 576]       Q2_K   -> 512 latent | 64 shared rope key (MQA)
attn_kv_a_norm   [512]             F32
attn_k_b         [192, 512, 64]    Q4_0   -> per head 192 -> 512  (query absorption)
attn_v_b         [512, 256, 64]    Q2_K   -> per head 512 -> 256
attn_output      [16384, 6144]     Q3_K
indexer.*        only on 21 layers  -> IGNORE (see below)
```

**The DSA indexer is RULED OUT.** Upstream commit `0451c84` is titled
*"model: support GLM MoE DSA arch (NOTE: indexer is not yet supported)"* — llama.cpp
does **dense** attention for glm-dsa and still produces correct text. So the indexer
cannot be the difference.

---

## 4. VERIFIED CORRECT — DO NOT SPEND TIME RE-CHECKING

Every one of these was measured against an *independent* reference this session:

| What | Reference | Result |
|---|---|---|
| Q2_K matvec kernel | PyTorch dense dequant | **cos 1.000000**, ratio 1.0000 |
| Q4_0 dequant (`dequant_q4_0`) | `gguf.quants.dequantize` | **cos 1.00000000**, maxdiff 0 |
| Single routed-expert SwiGLU | PyTorch | **cos 1.000000**, ratio 1.0000 |
| **Whole MoE layer** (8 routed + shared + router) | PyTorch | **cos 1.000000**, ratio 1.0036, **8/8 expert overlap**, 16/16 residency |
| **MLA attention, pos 0..5, real KV history** | PyTorch | **cos 1.000000 at every position**, both RoPE styles |
| Tokenizer | llama.cpp | byte-identical |
| Embedding Q2_K dequant | semantic-structure probe | related +0.324 vs unrelated +0.011 (**sep +0.313**) |
| LM head Q6_K | semantic-structure probe | sep +0.0736 |
| `k_b`/`v_b` orientation | shape algebra | **forced**, not ambiguous (192 must pair q_nope, 512 must pair c) |

Also swept and found **not** to be the cause: RoPE style (neox vs gptj), attention
score scale (1× 2× 4× 8× 16× 32×), router weight scale (0.0 / 1.0 / 2.5), shared
expert (nulling it changes ‖x‖ by <1%), routed experts (nulling ALL of them changes
final ‖x‖ by 3%).

---

## 5. TRAPS — these WILL produce false conclusions. Read before testing.

1. **`full_moe_layer_v2` replays a CUDA graph baked to `eng._x_buf`.** Pass any other
   buffer and the graph reads **stale** data. This made the router look like it picked
   **0/8** different experts; with `_x_buf` it is **8/8 perfect**. Always stage input
   via `eng._x_buf.copy_(x)`.
   *(This is also a genuine latent landmine in the engine — the graph silently ignores
   its `x` argument. Worth fixing regardless.)*
2. **`_score_buf`, `_topk_ids_dev`, `_topk_vals_dev` are shared across all layers.**
   Reading them after a full forward gives **layer 77 only**. Harvest per-layer or use
   the kernel's route log.
3. **Expert residency.** Only ~1,309 of 9,600 experts fit in VRAM. The kernel's NULL
   guard **silently skips** non-resident experts, so a run can look faster/different
   while doing less work. Always gate on
   `lib.get_experts_requested()` vs `lib.get_experts_executed()` and require 100%.
   Warm with `eng.page_in_misses()` until it returns 0.
4. **bf16→f32 zero-extension is NOT corruption.** 430/505 F32 tensors have 100% of
   elements with zero low-16-bits and their high half reads as f16 ≈ 1.0. That is
   lossless bf16→f32. `output_norm` reads a normal 1.21 as f32. The f32 read is
   correct — the norm weights genuinely are small and grow with depth.

---

## 6. THE FAILURE, PRECISELY

**Representation collapse at L8–L12.** Pairwise cosine between cached KV latents
across positions, on 9 unrelated tokens (Paris/zebra/7/quantum/ocean/…):

```
L0  0.119   L4  0.051   L8  0.105   <- healthy, positions differ
L12 0.883   L24 0.968   L40 0.989   L60 0.992   L77 0.994   <- collapsed
final hidden state across all 9 tokens: cos 0.993
```

Mechanism: the softmax is **near-uniform at every layer** (entropy 2.069 vs uniform
ceiling 2.0794 over 8 positions), so `oc = mean(cached latents)` — *the same vector
for every query position*. At **L8** that output (1.053) first exceeds the residual
(0.811). Every position gets the same large vector added, so they converge; once
collapsed, identical positions produce identical scores and no score scale recovers
it.

Attention **is** correctly scale-invariant (16× input → 1.00× output), so the 760×
residual growth (0.574 → 435) is a *consequence*, not the bug.

Contribution split over 78 layers: **attention 2792.7, all FFN 747.7.**

---

## 7. LEADS, RANKED

1. **`mscale` in the attention scale.** llama.cpp `src/models/glm-dsa.cpp` uses
   ```c
   const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));
   ```
   FAIE uses plain `1/sqrtf(qk_nope+qk_rope)` = 1/16, **no mscale**. In deepseek2
   llama.cpp computes
   `mscale = attn_factor * (1 + rope_yarn_log_mul * logf(1/freq_scale))`.
   The GGUF shows **no `rope.scaling.*` keys**, so `freq_scale`=1 → mscale should be
   1 — **but verify whether `rope_yarn_log_mul` has a non-zero arch default for
   glm-dsa.** A constant factor on the scores is exactly the flat-softmax signature.
2. **Read `src/models/glm-dsa.cpp` verbatim** and diff the ggml graph against
   `mla_attention()` in `ternary_kernel.cu`. Pay attention to every
   `ggml_cont` / `ggml_permute` / `ggml_reshape` around `wk_b` and `wv_b`, and to how
   `wv_b` is applied inside `build_attn`.
3. **Head packing of the `attn_output` input.** FAIE packs `headB` head-major
   (`h*256 + d`, 64×256=16384). Confirm llama.cpp's ordering.
4. **rms_norm epsilon**: GGUF says 1e-5, `rms_norm_kernel` hardcodes 1e-6. Small, but
   free to fix.
5. **Layer count**: FAIE runs 0..77. Confirm 78 (nextn/MTP) must be excluded and that
   nothing else special happens at the last layer.

---

## 8. FILES

**Engine**
- `ternary_kernel.cu` — all CUDA. `mla_attention()` is the attention graph;
  `full_moe_layer_v2()` the MoE. Runtime switches already exported:
  `set_rope_style(0=neox,1=gptj)`, `set_score_scale(float)`,
  `set_router_scale(float)`, `set_bisect(no_shared,no_attn)`,
  `reset_expert_counters()/get_experts_requested()/get_experts_executed()`,
  `drain_miss_log()`.
- `faie_generate.py` — model loading, MLA wiring, KV cache, expert paging.
- `faie_kernel.py` — ctypes bindings.

**Build** (must use these exact flags; `--default-stream per-thread` is required for
graph capture):
```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;" + $env:PATH
$env:INCLUDE = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared"
$env:LIB = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
cd E:\ABU\NEOCORTEX
nvcc --default-stream per-thread -shared -O3 --use_fast_math -arch=sm_120 -Xcompiler "/MD" -o tk_new.dll ternary_kernel.cu
if (Test-Path .\tk_new.dll) { Move-Item .\tk_new.dll .\ternary_kernel.dll -Force }
```
Kill any process holding the DLL first (`Get-Process llama-server,python | Stop-Process -Force`).
Always run python with `$env:PYTORCH_ALLOC_CONF="expandable_segments:True"` and
`$env:PYTHONIOENCODING="utf-8"` (console is cp1252 and cannot print Arabic).

**Diagnostics already built — reuse, don't rebuild**
| file | what it proves |
|---|---|
| `faie_sweep.py` | **THE GATE.** rank(Paris) vs the oracle across structural variants |
| `mla_diff.py` | MLA vs PyTorch at pos 0, stage by stage |
| `mla_multipos.py` | MLA vs PyTorch at pos 0..5 with real KV history |
| `expert_diff.py` | matvec / single expert / whole MoE layer vs PyTorch + router picks |
| `faie_bisect.py` | null attention or shared expert, per-layer contribution trace |
| `attn_probe.py` | attention entropy + scale-invariance |
| `score_probe.py` | score spread split into nope vs rope terms |
| `cache_variance.py` | **the collapse metric** — latent cosine across positions |
| `absorb_orient.py` | k_b/v_b orientation (note: variants A and B are provably identical) |
| `norm_check.py`, `f32_lowhalf.py` | norm-weight stats, bf16 pattern |
| `test_predict.py`, `test_talk.py` | top-5 predictions, coherence |

**Proofs**: `faie_30tps_proof.json`, `faie_honest_proof.json`, `faie_paging_proof.json`,
`faie_sweep.json`, `llamacpp_truth.log`, `hermes_link_proof.md`.

---

## 9. RULES OF ENGAGEMENT (Ahmad's, non-negotiable)

- **NO LIES. Triple-verify with tool calls.** Never state a result you have not just
  measured. Verify from disk and live probes, never from memory or a prior claim.
- **Instrument before claiming.** A speedup or a fix is usually *skipped work*. Export
  requested-vs-executed counters and put every user-visible cost inside the timer.
  This session caught two inflated numbers and two false root causes that way.
- **Read before modify; back up before delete; one task → test → next.**
- Timestamps from `Get-Date` only, format `YYYY-MM-DD HH:MM EDT`.
- Address him **"Ahmad"**.
- Do not touch GPU1 processes, spend money, or bind him to anything.

## 10. DEFINITION OF DONE

`python faie_sweep.py` prints `rank(Paris) = 0` with `overlap` ≥ 3/5, at
**100% expert residency**, and `python test_predict.py` returns sensible top-5
predictions on all four probes. Then re-run `python prove_honest.py` to confirm the
32.03 tok/s throughput did not regress.
