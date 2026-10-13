"""Does GLM-5.2's OWN tokenizer already carry Arabic root structure? — the measurement
that decides whether a root-coordinate adapter can replace the embedding path.

THE QUESTION (Ahmad's, 2026-07-26):
A proposed design bypasses the BPE tokenizer and the 154,880-row embedding matrix for
Arabic, computing a 10D Poincare root coordinate + KAN wazn transform, projecting
20D -> 6144D through a small adapter, and injecting that ONE vector at layer 0/1.

Two facts decide whether that can work, and both are measurable with the model's own
tokenizer — no GPU, no training:

  (1) POSITION COUNT. The adapter emits ONE hidden vector per word. The model was
      trained on Arabic where each word occupies N positions. If N >> 1, the adapter
      does not merely change values — it changes SEQUENCE GEOMETRY, and every attention
      pattern the model learned for Arabic assumed the multi-position layout. Measured
      here as tokens-per-word on real Quranic text (and English, for the tax ratio).

  (2) ROOT SIGNAL ALREADY PRESENT. If surface forms sharing a triliteral root already
      share tokens far more than forms from different roots, the tokenizer is ALREADY
      encoding the root — so a root adapter adds no new semantic information and its
      only honest win is sequence-length compression. If they share nothing, root
      identity really is destroyed at tokenization and there is genuine headroom.
      Measured as within-root vs across-root token-set Jaccard, against a null.

Judged with llama.cpp's own tokenizer binary (llama-tokenize) — never a reimplementation.
Ground truth for roots: the Quranic morphology export already on disk, which is where
corpus/arabic.txt itself came from.

Usage:  python root_tokenizer_probe.py
"""
import json
import os
import random
import re
import subprocess
import sys
from collections import defaultdict

MORPH = r"E:\ABU\NEOCORTEX\data\corpus\quran-morphology.txt"
MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
TOKENIZE = r"E:\ABU\MODELS\runtime\llama-b10069-cuda13\llama-tokenize.exe"
OUT = r"E:\ABU\NEOCORTEX\root_tokenizer_probe.json"

N_ROOTS = 40            # roots sampled for the within/across comparison
N_FORMS = 6             # distinct surface forms per root
SEED = 0

ENGLISH = (
    "The capital of France is Paris and the mercy of the merciful is written in the book. "
    "Praise belongs to the lord of the worlds, the compassionate, the ever merciful one. "
    "Knowledge and sight and understanding are given to those who read what was revealed."
)


def tok(text):
    """Token ids for text, via llama.cpp's own tokenizer. Returns list[int]."""
    p = subprocess.run([TOKENIZE, "-m", MODEL, "-p", text, "--ids", "--no-bos"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    # the id list may land on either stream depending on build; take the LAST match seen
    m = None
    for stream in (p.stdout or "", p.stderr or ""):
        for mm in re.finditer(r"\[[\d,\s]+\]", stream):
            m = mm
    if not m:
        sys.exit(f"tokenizer gave no id list.\nstdout={(p.stdout or '')[:400]}\n"
                 f"stderr={(p.stderr or '')[-400:]}")
    return [int(x) for x in re.findall(r"\d+", m.group(0))]


def load_root_forms():
    """root -> set(vocalised surface forms), from the morphology export (col2 form, ROOT: in features)."""
    if not os.path.exists(MORPH):
        sys.exit(f"missing {MORPH}")
    by_root = defaultdict(set)
    with open(MORPH, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            form, feats = parts[1].strip(), parts[3]
            m = re.search(r"ROOT:([^\|\s]+)", feats)
            if not m or not form:
                continue
            by_root[m.group(1)].add(form)
    return by_root


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0


def main():
    rng = random.Random(SEED)

    # ---- (1) position count -----------------------------------------------------
    ar_lines = [l.strip() for l in open(r"E:\ABU\NEOCORTEX\corpus\arabic.txt",
                                       encoding="utf-8").read().splitlines() if l.strip()]
    ar_sample = " ".join(ar_lines[:120])
    ar_ids, en_ids = tok(ar_sample), tok(ENGLISH)
    ar_words = len(ar_sample.split())
    en_words = len(ENGLISH.split())
    tpw_ar = len(ar_ids) / ar_words
    tpw_en = len(en_ids) / en_words
    cpt_ar = len(ar_sample.replace(" ", "")) / len(ar_ids)
    cpt_en = len(ENGLISH.replace(" ", "")) / len(en_ids)
    print("=== (1) POSITION COUNT — what one injected vector would have to replace ===")
    print(f"  Arabic : {ar_words:5d} words -> {len(ar_ids):6d} tokens = {tpw_ar:.2f} tokens/word"
          f"   ({cpt_ar:.2f} chars/token)")
    print(f"  English: {en_words:5d} words -> {len(en_ids):6d} tokens = {tpw_en:.2f} tokens/word"
          f"   ({cpt_en:.2f} chars/token)")
    print(f"  -> Arabic tokenizer tax {cpt_en/cpt_ar:.2f}x; one adapter vector would replace "
          f"{tpw_ar:.2f} trained positions per word.")

    # ---- (2) is root identity already in the token stream? ----------------------
    by_root = load_root_forms()
    usable = {r: sorted(f) for r, f in by_root.items() if len(f) >= N_FORMS}
    roots = rng.sample(sorted(usable), min(N_ROOTS, len(usable)))
    print(f"\n=== (2) ROOT SIGNAL IN THE TOKEN STREAM ===")
    print(f"  roots with >= {N_FORMS} distinct forms: {len(usable):,};  sampling {len(roots)}")

    forms_tok = {}
    for r in roots:
        picked = rng.sample(usable[r], N_FORMS)
        forms_tok[r] = {f: tok(f) for f in picked}

    within, across = [], []
    for r, fm in forms_tok.items():
        fs = sorted(fm)
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                within.append(jaccard(fm[fs[i]], fm[fs[j]]))
    rl = sorted(forms_tok)
    for i in range(len(rl)):
        for j in range(i + 1, len(rl)):
            a = rng.choice(sorted(forms_tok[rl[i]]))
            b = rng.choice(sorted(forms_tok[rl[j]]))
            across.append(jaccard(forms_tok[rl[i]][a], forms_tok[rl[j]][b]))

    mw = sum(within) / len(within)
    ma = sum(across) / len(across)
    tpw_form = sum(len(v) for fm in forms_tok.values() for v in fm.values()) / \
               sum(len(fm) for fm in forms_tok.values())
    print(f"  mean token-set Jaccard, SAME root  : {mw:.4f}  (n={len(within)})")
    print(f"  mean token-set Jaccard, DIFF roots : {ma:.4f}  (n={len(across)})")
    print(f"  ratio (same / diff)                : {mw/ma if ma else float('inf'):.2f}x")
    print(f"  mean tokens per isolated surface form: {tpw_form:.2f}")

    verdict = ("tokenizer ALREADY carries strong root signal -> a root adapter adds little "
               "new semantics; its honest win is sequence length"
               if mw / max(ma, 1e-9) >= 2.0 else
               "root identity is largely LOST at tokenization -> genuine semantic headroom "
               "for a root-aware input path")
    print(f"\n  VERDICT: {verdict}")

    json.dump({
        "tokens_per_word_arabic": tpw_ar, "tokens_per_word_english": tpw_en,
        "chars_per_token_arabic": cpt_ar, "chars_per_token_english": cpt_en,
        "tokenizer_tax": cpt_en / cpt_ar,
        "roots_sampled": len(roots), "forms_per_root": N_FORMS,
        "jaccard_same_root": mw, "jaccard_diff_root": ma,
        "ratio": mw / ma if ma else None,
        "tokens_per_isolated_form": tpw_form,
        "verdict": verdict,
    }, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
