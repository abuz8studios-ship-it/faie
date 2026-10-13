"""TAJWID H2 — QLoRA-distil the student on Ahmad's own workload traces.

Runs identically on a Lightning Studio A100/H100 and on a local 5090; the only
difference is --model and batch sizes.

*** THE CONSTRAINT THAT DECIDES THE STUDENT ***
You CANNOT train a GGUF. Ternary-Bonsai-27B-Q2_0.gguf on disk is an INFERENCE artifact:
weights are packed ternary with block scales, there is no autograd graph, and llama.cpp
is not a training framework. Training needs HF-format safetensors.

So the honest H2 flow is:
    HF-format student  --QLoRA on traces-->  merge  -->  quantize  -->  GGUF  -->  serve
                       (this script)                    (llama.cpp)        (local)

Pick the student on THREE criteria, in this order:
  1. HF weights exist and the licence permits commercial use (Ahmad sells software).
  2. It survives aggressive quantization afterwards — ideally already ternary-QAT
     (the Bonsai result: 95% of FP16 kept at 1.71 bpw) or at minimum 4-bit-friendly.
  3. It fits the box after quantization: target <= 12 GB so the whole brain lands
     under 32 GB with KV + drafter + adapt headroom (THE_BRAIN_SPEC.md).

Judged the same way every gate in this project is judged: held-out perplexity through
llama.cpp on corpus/calib_holdout.txt (workload) and calib_arabic_holdout.txt (Arabic,
scored SEPARATELY — never blended, per the 0.0227%/tokenizer-tax measurement).

Usage (Lightning Studio, single A100 80GB):
  pip install -r requirements.txt
  python distill_qlora.py --model <hf-id> --traces traces/traces_workload.jsonl \
      --arabic traces/traces_arabic.jsonl --arabic-ratio 0.05 --out out/student-v1
"""
import argparse
import json
import os
import random


def load_jsonl(path, limit=None):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("prompt") and d.get("completion"):
                rows.append(d)
            if limit and len(rows) >= limit:
                break
    return rows


def build_dataset(workload, arabic, arabic_ratio, seed=0):
    """Mix strata at TRAIN time, not harvest time.

    Arabic is 0.0227% of natural traffic. Left proportional it contributes no gradient
    signal at all; blended in at harvest it contaminates the workload distribution.
    Mixing here makes the ratio an explicit, logged, tunable decision.
    """
    rng = random.Random(seed)
    rows = list(workload)
    if arabic and arabic_ratio > 0:
        want = int(len(workload) * arabic_ratio / max(1e-9, 1 - arabic_ratio))
        if want:
            picked = [rng.choice(arabic) for _ in range(want)] if want > len(arabic) \
                else rng.sample(arabic, want)
            rows += picked
    rng.shuffle(rows)
    return rows


def to_chat(rec):
    return {"messages": [
        {"role": "system", "content": rec.get("system", "")},
        {"role": "user", "content": rec["prompt"]},
        {"role": "assistant", "content": rec["completion"]},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local safetensors dir")
    ap.add_argument("--traces", default="traces/traces_workload.jsonl")
    ap.add_argument("--arabic", default="traces/traces_arabic.jsonl")
    ap.add_argument("--arabic-ratio", type=float, default=0.05)
    ap.add_argument("--out", default="out/student-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="build + report the dataset, load nothing, touch no GPU")
    a = ap.parse_args()

    wl = load_jsonl(a.traces, a.limit or None)
    ar = load_jsonl(a.arabic, a.limit or None)
    rows = build_dataset(wl, ar, a.arabic_ratio)
    n_ar = sum(1 for r in rows if r.get("stratum") == "arabic")
    chars = sum(len(r["prompt"]) + len(r["completion"]) for r in rows)
    print(f"workload traces : {len(wl):,}")
    print(f"arabic traces   : {len(ar):,}")
    print(f"mixed dataset   : {len(rows):,} examples "
          f"({n_ar:,} arabic = {100*n_ar/max(len(rows),1):.2f}%)")
    print(f"total chars     : {chars:,}  (~{chars/3.5/1e6:.2f}M tokens rough)")
    if not rows:
        raise SystemExit("no traces — run make_traces.py first")

    if a.dry_run:
        print("\n--dry-run: dataset built, nothing loaded. Sample:")
        s = to_chat(rows[0])
        for m in s["messages"]:
            print(f"  [{m['role']}] {m['content'][:150].replace(chr(10),' ')!r}")
        return

    # Heavy imports live here so --dry-run works on a box without the training stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokzr = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tokzr.pad_token is None:
        tokzr.pad_token = tokzr.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    ds = Dataset.from_list([to_chat(r) for r in rows])
    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        max_length=a.seq_len, bf16=True, logging_steps=5, save_strategy="epoch",
        gradient_checkpointing=True, lr_scheduler_type="cosine", warmup_ratio=0.03,
        report_to=[])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tokzr, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)
    tokzr.save_pretrained(a.out)
    print(f"\nadapter saved -> {a.out}")
    print("NEXT: merge the adapter, convert to GGUF, quantize, then judge with\n"
          "  llama-perplexity -f corpus/calib_holdout.txt        (workload)\n"
          "  llama-perplexity -f corpus/calib_arabic_holdout.txt (arabic, scored apart)")


if __name__ == "__main__":
    main()
