# FAIE — Fused Arabic Inference Engine (research)

Research code for fast local MoE inference: custom kernels, expert paging, ternary experiments,
and the proofs behind each claim. This repo is the **code**; papers, handoffs, and proof bundles
are mirrored on Hugging Face:

- 🤗 Papers + code mirror: https://huggingface.co/abuz8studios/FAIE
- 🤗 Reproducible proofs (nothing published until it reproduces): https://huggingface.co/abuz8studios/FAIE-proofs
- Sister brain built on this line of work: https://huggingface.co/abuz8studios/SUPERBRAIN-1
  (code: https://github.com/abuz8studios-ship-it/superbrain-1)

## Status — research / WIP

FAIE is an active research engine, not a finished product. Speed and paging claims in this repo
travel with their proof scripts, hardware, and dates (see `*_proof.json` files and the HF proofs repo);
anything without a proof file is an experiment, including failed ones kept for the record.

**Weights are not redistributed** pending license review of the base models used in experiments.

## Start here

- `THE_BRAIN_SPEC.md` — what the engine is trying to be
- `ROOT_CALCULUS.md` — the Arabic-root computation notes
- `faie.py`, `faie_kernel.py`, `faie_runner.py` — engine core
- `prove_*.py`, `faie_prove.py`, `verify.py` — reproduce the claims yourself
- `NEXT_AGENT_INSTRUCTIONS.md`, `FAIE_HANDOFF_TO_FABLE.md` — dated research handoffs (history)

Contact: support@abuz8ai.com · Site: https://abuz8ai.com · X: https://x.com/AbuZ8Studios
