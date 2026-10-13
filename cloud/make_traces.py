"""Build the tajwid distillation corpus: teacher answers to Ahmad's REAL prompts.

WHY THIS AND NOT A PUBLIC DATASET:
The student is not meant to be generally frontier — it is meant to be frontier ON
AHMAD'S WORKLOAD. That distribution is already measured and on disk: 167,258 unique
messages over 3.7 months, tool mix dominated by terminal / Read / Bash / PowerShell /
Grep. Public instruction sets do not have that shape. train.jsonl is the shape.

STRATA (never blended — measured 2026-07-26):
Arabic is 0.0227% of real usage and pays a ~2.3-2.8x tokenizer tax. Proportional
sampling gives Arabic almost no signal, so it is generated as its OWN stratum with its
own file, and mixed at training time at a chosen ratio rather than at harvest time.

RESUME SAFETY:
Every record is keyed by a hash of (stratum, prompt). A re-run skips what exists, so a
killed run costs nothing. Learned the hard way: a "stopped" background run once kept
writing the SAME output file as its replacement for 20 minutes.

*** BEFORE POINTING THIS AT A PAID/MANAGED API — READ ***
Generating training data from a hosted model's outputs is governed by that provider's
terms, and some providers restrict using outputs to train other models. Ahmad SELLS
software, so this is commercial exposure, not a theoretical concern. Two clean routes:
  1. SELF-HOST an open-weight teacher whose licence permits distillation (rent the GPU,
     run the weights yourself) — no ambiguity, and you also get logprobs.
  2. Use openly-licensed existing traces.
This script therefore REQUIRES --i-checked-the-licence when TEACHER_BASE_URL is not
localhost. It is a speed bump against an expensive mistake, not legal advice.

Usage:
  python make_traces.py --stratum workload --limit 200
  python make_traces.py --stratum arabic   --limit 100
  python make_traces.py --stratum workload --limit 5000 --i-checked-the-licence
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_client as tc

CORPUS = r"E:\ABU\NEOCORTEX\corpus"
OUTDIR = r"E:\ABU\NEOCORTEX\cloud\traces"
ARABIC = re.compile(r"[\u0600-\u06FF]")

SYSTEM = ("You are a precise engineering assistant for a Windows/PowerShell power user "
          "who runs local AI infrastructure. Answer concretely and completely. Prefer "
          "exact commands, file paths and numbers over generalities.")


def key(stratum, prompt):
    return hashlib.sha256(f"{stratum}\x00{prompt}".encode("utf-8")).hexdigest()[:20]


def load_prompts(stratum, min_len=40, max_len=6000):
    """User-role messages from the real record, filtered and de-duplicated."""
    src = os.path.join(CORPUS, "train.jsonl")
    seen, out = set(), []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("role") != "user":
                continue
            t = (d.get("text") or "").strip()
            if not (min_len <= len(t) <= max_len):
                continue
            is_ar = len(ARABIC.findall(t)) > max(8, 0.15 * len(t))
            if (stratum == "arabic") != is_ar:
                continue
            h = key(stratum, t)
            if h in seen:
                continue
            seen.add(h)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=("workload", "arabic"), default="workload")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=tc.DEFAULT_MAX_TOKENS)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    ap.add_argument("--i-checked-the-licence", action="store_true")
    a = ap.parse_args()

    remote = not re.search(r"(127\.0\.0\.1|localhost)", tc.BASE)
    if remote and not a.i_checked_the_licence:
        sys.exit("REFUSING: TEACHER_BASE_URL is remote and --i-checked-the-licence was not\n"
                 "passed. Confirm the provider's terms permit using outputs as training\n"
                 "data, or self-host an open-weight teacher instead. See the header.")

    os.makedirs(OUTDIR, exist_ok=True)
    out_path = os.path.join(OUTDIR, f"traces_{a.stratum}.jsonl")

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass

    prompts = load_prompts(a.stratum)
    todo = [p for p in prompts if key(a.stratum, p) not in done][: a.limit]
    print(tc.describe())
    print(f"stratum={a.stratum}  available={len(prompts):,}  already done={len(done):,}  "
          f"this run={len(todo):,}  -> {out_path}")
    if not todo:
        print("nothing to do")
        return

    t0, ok, tok = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, p in enumerate(todo, 1):
            r = tc.complete([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": p}],
                            max_tokens=a.max_tokens)
            # An empty completion is a FAILED sample, not a short one. Reasoning models
            # burn the whole budget thinking and return '' — never store that as truth.
            if not r["text"].strip():
                print(f"  [{i}/{len(todo)}] EMPTY completion (finish={r['finish_reason']}) — skipped")
                continue
            rec = {"key": key(a.stratum, p), "stratum": a.stratum,
                   "system": SYSTEM, "prompt": p, "completion": r["text"],
                   "finish_reason": r["finish_reason"], "usage": r["usage"],
                   "teacher": f"{tc.MODEL}@{tc.BASE}"}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            tok += (r["usage"] or {}).get("completion_tokens", 0) or 0
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] kept={ok}  {tok:,} out-tok  "
                      f"{el/60:.1f} min  {tok/max(el,1e-9):.1f} tok/s", flush=True)
            if a.sleep:
                time.sleep(a.sleep)

    el = time.time() - t0
    print(f"\nDONE  kept {ok}/{len(todo)}  {tok:,} completion tokens in {el/60:.1f} min")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
