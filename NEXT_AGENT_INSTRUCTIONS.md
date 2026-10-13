# NEOCORTEX / THE BRAIN (QALB) — exact next steps for Muse / Kimi
Read `THE_BRAIN_SPEC.md` and `ROOT_CALCULUS.md` in this folder first for the
full architecture and the gate definitions (G1–G6). This file translates the
one open gate into exact commands.

**This is NOT the same project as `E:\ABU\02_PROJECTS\AHMAD-1\`.** That one
trains a sub-1B model from scratch. This one factors/distills GLM-5.2-REAM
down to a 20–30B-class ternary model that fits under 32 GB. Do not merge work
between them.

---

## What's already done (verified on disk 2026-09-16 — do not redo this)

1. **The judge tool is already built.** The fork at
   `E:\ABU\NEOCORTEX\prismml-llama.cpp\` has commit `9ca265a` (2026-07-31,
   "cuda: extract Q2_0 elements via __byte_perm") — this is the fix for the
   tensor-offset read bug that blocked gate measurement in the spec doc.
   A compiled build already exists at
   `E:\ABU\NEOCORTEX\prismml-llama.cpp\build\bin\` (built 2026-08-18), including:
   - `llama-perplexity.exe`
   - `llama-server.exe`
   - `llama-cli.exe`

2. **The test artifacts already exist.** Both gate inputs are on disk:
   - `F:\GLM52_FACTOR\L40-tern.gguf` (146.9 GB, 2026-07-26) — for gate G2
   - `F:\GLM52_FACTOR\L40-r1024w.gguf` — for gate G3
   - `F:\GLM52_FACTOR\L40-tern.gguf.stats.json` — check this first, it may
     already contain a partial result.

3. **What's NOT done:** no log file in `E:\ABU\NEOCORTEX\` is dated after
   2026-08-18 (the build date). The newest gate-related logs
   (`v3_ppl_L40tern_20260730_0010.log`, `judge_v3_master.log`) predate the
   fixed build — meaning **the fixed judge has never actually been run against
   these artifacts.** That is the single open action.

---

## Step 1 — check the stats file first, it may already have an answer

```powershell
Get-Content "F:\GLM52_FACTOR\L40-tern.gguf.stats.json" | ConvertFrom-Json | Format-List
```
If this already contains a perplexity/PPL number with a date after 2026-08-18,
the gate may already be closed — stop, log it, don't re-run.

## Step 2 — if not, run the judge (gate G2)

```powershell
cd "E:\ABU\NEOCORTEX\prismml-llama.cpp\build\bin"
.\llama-perplexity.exe -m "F:\GLM52_FACTOR\L40-tern.gguf" -f "E:\ABU\NEOCORTEX\corpus\calib_holdout" --ctx-size 4096
```
Adjust `-f` to whichever holdout corpus file `THE_BRAIN_SPEC.md` §3/§7 names for
this gate — check `E:\ABU\NEOCORTEX\data\` and `corpus\` for the exact holdout
filename before running; multiple `im_probe_*.gguf` calibration files exist in
`corpus\` and the spec is specific about which one is the *holdout* (never
trained/calibrated on) versus a *calibration* set. Using the wrong one
invalidates the gate — read §7's gate table again if unsure.

**Also run the oracle sanity check** (from spec §7): the model must produce
`" Paris"` (token id 12089, reference logprob −0.445) when prompted with
`"The capital of France is"`. If this fails, the gate result is meaningless —
something is broken in the build or the model file, stop and report before
proceeding.

## Step 3 — same for gate G3

```powershell
.\llama-perplexity.exe -m "F:\GLM52_FACTOR\L40-r1024w.gguf" -f <same holdout as step 2> --ctx-size 4096
```

## Step 4 — record the result, per the verdict rules already written

`THE_BRAIN_SPEC.md` §4 and §8 already state exactly what each possible result
means (ΔPPL small = gate open, large = gate dies, etc.) and what to do next in
each case. Read the result against that table — do not interpret it fresh.
Write the outcome to a new dated log file in `E:\ABU\NEOCORTEX\` following the
existing naming pattern (`<gate>_ppl_<config>_<timestamp>.log`), and add one
line to `LAST_SYNC.md`-style summary at the top of this file's own edit history
— there is no LEDGER.md in this folder yet; if gates G2/G3 close either way,
create one following the exact format used in
`E:\ABU\02_PROJECTS\MUTHALLATH-1\LEDGER.md` (dated, proof column, no adjectives).

## Do not do this without asking Ahmad first
- Do not delete `F:\GLM52_FACTOR\*.gguf` regardless of gate outcome — these
  took real GPU-hours to produce and are needed for the falsifiability record
  even if the gate fails.
- Do not start Horizon 2 (QLoRA distillation, §3) until G2/G3 both have a
  written verdict — the spec's own gate order requires this.
