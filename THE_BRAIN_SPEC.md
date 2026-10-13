# THE BRAIN — Architecture Spec for the ABUZ8 Frontier Brain under 32 GB
**2026-07-26 · Session e67037de · Fable (plan) → Opus fleet (execution) · Companion to [ROOT_CALCULUS.md](ROOT_CALCULUS.md)**

> Status tags as in the calculus doc: [MEASURED] / [ARITH] / [PRIOR-ART] / [GATED].
> This spec was written the same morning G1/G1b were measured; §1 is the reason
> the architecture looks like this and not like the July-25 plans.

---

## 1. What today's measurements killed, and what that means

Four independent structural escapes were measured on GLM-5.2-REAM (layers 3/40/77,
9 MoE tensors, dequant via `gguf.quants` only):

| escape hatch | instrument | verdict |
|---|---|---|
| Pruning (keep 2.5%) | llama-imatrix, 59M events | **DEAD** — 3.82–3.84× vs needed 37.5×; all 9,600 fire [MEASURED, prior session] |
| Shared basis, raw metric | `expert_basis.py` | **DEAD** — rank@95% = 5,263–5,670/6,144; 1.06–1.08× [MEASURED today 09:20] |
| Shared basis, **activation metric** | `expert_basis_v2.py`, imatrix-whitened | **DEAD** — rank@95% = 4,125–5,636; best case (L3 gate/up) still 4× over the r=1024 budget line [MEASURED today 09:50] |
| Shared mean + low-rank deltas (D²-MoE) | v2 mean/delta split | **DEAD** — mean carries 0.8–1.1% of energy; experts share nothing [MEASURED today] |
| Per-expert low-rank (SVD-LLM form) | v2 exact SVDs | **marginal** — rank@95% ≈ 1,400–1,690 of 2,048 (≤1.3×), nowhere near the 6× needed [MEASURED today] |

Cross-expert cosine sits at the sketch noise floor (|cos| ≈ 0.0127 ≈ √(2/π·4096)):
**REAM's experts are statistically indistinguishable from independent random
matrices in weight space.** The 753B→380B REAM distillation already wrung out the
redundancy that factorization would have harvested.

**The theorem this proves about the thesis itself:** Arabic's root system works
because the jadhr structure is *native* — the generative basis was there from the
first word. It cannot be recovered post-hoc from surface forms alone, and neither
can an expert basis be recovered from a trained-dense model. **Structure must be
present at birth. Therefore: the brain is trained into the factored-ternary form
(tajwīd), not carved out of REAM (post-hoc ishtiqāq).** This is the measured
version of the Unwalked Path's BitNet catch ("must be trained natively ternary")
and Bonsai's QAT-not-post-hoc proof.

Still pending today: G2 (ARB-8) — whether one post-hoc-ternarized layer (relF≈0.50
[MEASURED]) survives behaviorally. It calibrates how much damage tajwīd must
recover, not whether zero-shot ships (it doesn't).

---

## 2. The brain, in one picture

```
                    ┌─ ABUZ8 OS / Hermes / Telegram (the body, exists) ─┐
                    │                                                    │
   ┌────────────────▼───────────────┐      ┌───────────────────────────┐
   │  QALB القلب — the resident core │      │  DHAKIRA الذاكرة — memory  │
   │  ternary QAT student ≤ 12 GB    │◄────►│  root-indexed, hyperbolic  │
   │  + drafter ≤ 2 GB (spec-decode) │      │  coords, SQLite+vec+graph  │
   │  GPU0, always loaded            │      │  (≈3 MB today, unbounded)  │
   └────────────────┬───────────────┘      └───────────────▲───────────┘
                    │ escalation (hard queries)             │ write-back
   ┌────────────────▼───────────────┐      ┌───────────────┴───────────┐
   │  HAKIM الحكيم — the oracle      │      │  TAKAYYUF التكيف — adapt   │
   │  GLM-5.2-REAM via llama.cpp    │      │  LoRA coefficient updates  │
   │  7.3 tok/s 2-GPU / on-demand   │      │  nightly, on session logs  │
   │  also = distillation teacher   │      │  (roots frozen, wazn move) │
   └────────────────────────────────┘      └────────────────────────────┘
```

The 32 GB budget [ARITH, all lines measured or from disk]:

| component | GB | note |
|---|---|---|
| QALB core (ternary 27–35B class) | 7–12 | Bonsai-27B = 7.17 GB on disk today [MEASURED] |
| drafter (DSpark Q4_1) | 1.95 | on disk [MEASURED] |
| KV + activations + CUDA | 3.5 | FAIE-measured allowance |
| TTT/LoRA adapt headroom | 3–5 | Frontier-brief tax: +30–50% of adapted params, not of the model |
| memory + root graph + router glue | <0.1 | 3 MB SQLite today |
| **total** | **≤ 22.5** | **≥ 9.5 GB headroom on one 5090** |

The oracle does NOT live in the 32 GB: it is the *teacher and escalation tier*,
running exactly as it does today (llama.cpp, 2-GPU split or CPU-heavy, 7.3 tok/s
[MEASURED]). The brain under 32 GB is the *student that keeps getting better*.

---

## 3. The three horizons (what runs when)

### Horizon 1 — TODAY: the seed brain, all parts on disk
- **Core:** Ternary-Bonsai-27B-Q2_0 (6.67 GB on disk, Apache 2.0, QAT at 1.71 bpw
  keeping 95% of FP16 [PRIOR-ART]) served by the PrismML fork (mainline b10069
  cannot read Q2_0_g128 — **measured today**, tensor-offset error; fork build in
  progress, task #8).
- **Drafter:** DSpark Q4_1 1.81 GB for speculative decode.
- **Memory:** the 1,651-root graph + unified store (`server.py :8777`, `memory.py`)
  — root-indexed recall, Poincaré coordinates for derivation distance.
- **Escalation:** REAM oracle on demand (existing launch scripts; Ornith :8011
  untouched on GPU1).
- Gate: real completion + tok/s measurement on GPU0 [GATED — today].

### Horizon 2 — THIS WEEK: tajwīd round 1 (make it Ahmad's brain)
- QLoRA distillation of the QALB core on the **already-built strata**:
  `calib_workload` (real 3.7-month tool-heavy distribution: terminal 41k, Read
  13.8k, Bash 12.6k [MEASURED]) + `calib_arabic` as its **own stratum, never
  blended** (0.0227% share, 2.95× tokenizer tax [MEASURED]).
- Teacher signal: REAM logits via llama.cpp (`logprobs` endpoint) at ~7 tok/s →
  ~0.6M teacher tokens/day; plus 1M+ free GLM-5.x traces on HF [vault doctrine].
- Axolotl QLoRA fits one 5090 (~23 GiB) [vault doctrine]; GPU0 has 30.8 GB free
  [MEASURED today]. Ornith keeps GPU1.
- Holdouts already reserved (`calib_holdout`, `calib_arabic_holdout`) — the same
  llama-perplexity judge used on today's gates scores every distill round.
  Target: student holdout PPL within 5% of teacher on workload stratum.

### Horizon 3 — THE REAL BUILD: the born-factored brain (المولود من الجذور)
The student that *is* the calculus, structure present at birth:
- **Expert bank factored by construction:** every expert = coefficient block
  `Cₑ` against a layer-shared frozen-after-pretrain basis `V` (the jadhr layer).
  Storage 804,864·r per layer [ARITH] — r chosen at design time, not discovered.
- **Ternary QAT from step 0** (Bonsai recipe — the proven operating point).
- **Fractal routing** (hive levels: domain → specialist → surgical), so residency
  per token stays tiny; router precision stays high (MXFP8-class) while experts
  go ternary — the Unwalked-Path precision-per-lane rule.
- **Recursion = TTT on coefficients only** (roots frozen, awzān adapt — iʿrāb),
  within the Frontier-brief's honest 1.3B-adaptive-param ceiling on a 5090.
- **Spatial = hyperbolic memory coordinates** (Sarkar: trees embed in H² with
  low distortion; the derivation DAG is near-tree).
- Training data: Ahmad's corpus (167,258 msgs) + GLM-5.x traces + Quran root
  stratum; teacher = REAM + (optionally) Laguna S 2.1 for the coding slice
  [mapped, not pulled].
- Size target: 20–30B-class at 1.58–1.71 bpw ⇒ **5–8 GB weights** — frontier-*for-
  Ahmad's-workload* by construction, not frontier-general. Honest label required.

---

## 4. Decision gates still open today

| gate | question | state |
|---|---|---|
| G2 / ARB-8 | one ternary layer survives behaviorally? | patched GGUF written (147 GB, F:\GLM52_FACTOR\L40-tern.gguf); judge queued behind baseline PPL |
| G3 / FACTOR-2b | rank-1024 in the *data metric*, zero-shot, L40 | reconstruct queued (--metric imatrix, code selftested) — completes the death certificate empirically |
| H1 gate | Bonsai runs + tok/s on GPU0 | fork build in progress |
| H2 gate | first distill round moves holdout PPL toward teacher | needs H1 + GPU hours |

## 5. Falsifiability (carried from the calculus, updated)
- If ARB-8's patched model shows ~no PPL delta, post-hoc ternary re-opens as a
  cheap 1.66× (73.5 GB path stays dead for 32 GB but revives the 2-GPU 13 tok/s
  lane — task #58's old plan).
- If Bonsai-27B quality is too low as a seed, Horizon-1 falls back to the best
  ≤12 GB conventional student (LFM2-24B-A2B class) while Horizon-3 trains.
- Horizon-3's "frontier" claim is scoped to Ahmad's measured workload mix; any
  general-benchmark claim requires general benchmarks. No inflation.
