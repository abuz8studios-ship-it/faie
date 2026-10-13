"""Supplement to root_tokenizer_probe.py — the FAIR position count.

The main probe measured 7.65 tokens/word on the Quran corpus, which is fully
DIACRITIZED (harakat on every consonant). Real Arabic input — Ahmad's Telegram
messages, prompts, code comments — is normally undiacritized, and diacritics are
extra codepoints that inflate the token count. Reporting only the diacritized
number would overstate the adapter's problem.

This measures the same text three ways so the position count is honest:
  A. as-is (diacritized)      — the Quran stratum's real cost
  B. diacritics stripped      — how ordinary Arabic prose actually tokenizes
  C. stripped + no tatweel/hamza-normalised — the cheapest realistic form

Also reports the SINGLE-WORD cost, since the adapter operates per word.
"""
import json
import re
import subprocess
import sys

MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
TOKENIZE = r"E:\ABU\MODELS\runtime\llama-b10069-cuda13\llama-tokenize.exe"
OUT = r"E:\ABU\NEOCORTEX\root_tokenizer_probe2.json"

DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")


def tok(text):
    p = subprocess.run([TOKENIZE, "-m", MODEL, "-p", text, "--ids", "--no-bos"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = None
    for stream in (p.stdout or "", p.stderr or ""):
        for mm in re.finditer(r"\[[\d,\s]+\]", stream):
            m = mm
    if not m:
        sys.exit(f"no ids.\n{(p.stderr or '')[-300:]}")
    return [int(x) for x in re.findall(r"\d+", m.group(0))]


def normalise(t):
    t = DIACRITICS.sub("", t)
    return (t.replace("\u0671", "\u0627").replace("\u0622", "\u0627")
             .replace("\u0623", "\u0627").replace("\u0625", "\u0627"))


lines = [l.strip() for l in open(r"E:\ABU\NEOCORTEX\corpus\arabic.txt",
                                encoding="utf-8").read().splitlines() if l.strip()]
sample = " ".join(lines[:120])
variants = {
    "A_diacritized": sample,
    "B_diacritics_stripped": DIACRITICS.sub("", sample),
    "C_stripped_normalised": normalise(sample),
}

res = {}
print("=== POSITION COUNT — three honest variants of the SAME Arabic text ===")
for name, txt in variants.items():
    ids = tok(txt)
    w = len(txt.split())
    res[name] = {"words": w, "tokens": len(ids), "tokens_per_word": len(ids) / w,
                 "chars_per_token": len(txt.replace(" ", "")) / len(ids)}
    print(f"  {name:24s} {w:5d} words -> {len(ids):6d} tokens = "
          f"{len(ids)/w:5.2f} tok/word   ({res[name]['chars_per_token']:.2f} chars/token)")

# per-word cost in isolation (what the adapter actually replaces)
words = [w for w in variants["C_stripped_normalised"].split() if len(w) > 2][:60]
per = [len(tok(w)) for w in words]
res["isolated_words_normalised"] = {"n": len(per), "mean_tokens": sum(per) / len(per),
                                    "min": min(per), "max": max(per)}
print(f"\n  isolated single words (normalised, n={len(per)}): mean "
      f"{sum(per)/len(per):.2f} tokens/word  (min {min(per)}, max {max(per)})")
print("\n  -> This mean is the number of trained sequence positions ONE adapter vector "
      "would have to stand in for.")
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
