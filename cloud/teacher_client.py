"""Endpoint-agnostic teacher client — one interface for NIM, Lightning-hosted, or local.

WHY THIS SHAPE:
Every candidate teacher speaks the OpenAI chat-completions protocol:
  * NVIDIA NIM      https://integrate.api.nvidia.com/v1   (key: NVIDIA_API_KEY)
  * Lightning-hosted vLLM/llama.cpp on a rented A100/H100 (key: optional)
  * LOCAL llama.cpp  http://127.0.0.1:8011/v1             (no key — Ornith, live today)

So the pipeline is written once and the teacher is a URL. That means the whole
trace-generation lane can be PROVEN today against the local brain, and switched to a
rented GPU by changing one environment variable — no code edit, no second
implementation to keep in sync.

CONFIG (environment only — never hardcode a key, never print one):
  TEACHER_BASE_URL   default http://127.0.0.1:8011/v1
  TEACHER_API_KEY    optional; required by NIM
  TEACHER_MODEL      default "brain" (local); e.g. a NIM model id when remote

NOTE ON WHAT AN API TEACHER CAN AND CANNOT GIVE YOU:
A chat endpoint returns TEXT (and, if the server supports it, top-k logprobs).
Real distillation prefers full logit distributions. Text-only traces still work —
that is sequence-level knowledge distillation, which is what CoderForge/OpenHands
used — but if you want logit-level distillation you must SELF-HOST the teacher
(open weights on a rented GPU), not call a managed API. See cloud/README.md.
"""
import json
import os
import time
import urllib.error
import urllib.request

BASE = os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:8011/v1").rstrip("/")
KEY = os.environ.get("TEACHER_API_KEY", "")
MODEL = os.environ.get("TEACHER_MODEL", "brain")

# Ornith and other reasoning models emit reasoning_content first and return EMPTY
# content if max_tokens is small — a low-budget smoke test reports a healthy brain as
# broken (learned 2026-07-26). Never default this low.
DEFAULT_MAX_TOKENS = 4096


class TeacherError(RuntimeError):
    pass


def describe():
    """Human-readable target, with key presence but never the key itself."""
    return (f"teacher: {MODEL} @ {BASE} "
            f"(api key: {'set, ' + str(len(KEY)) + ' chars' if KEY else 'none — local mode'})")


def complete(messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.0,
             top_logprobs=0, timeout=300, retries=4):
    """One chat completion. Returns dict(text, finish_reason, usage, logprobs|None).

    Retries on transport errors and 429/5xx with exponential backoff. Raises
    TeacherError on a definitive client error (bad key, bad model) so a bulk run
    stops instead of silently writing thousands of failures.
    """
    payload = {"model": MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if top_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs

    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"

    delay = 2.0
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{BASE}/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            ch = d["choices"][0]
            msg = ch.get("message", {})
            return {
                "text": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "finish_reason": ch.get("finish_reason"),
                "usage": d.get("usage", {}),
                "logprobs": ch.get("logprobs"),
            }
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {body}"
            # 4xx other than 429 will not fix themselves — fail fast and loudly.
            if e.code not in (408, 429) and e.code < 500:
                raise TeacherError(f"{last}\n({describe()})")
        except Exception as e:                       # transport, timeout, JSON
            last = f"{type(e).__name__}: {e}"
        time.sleep(delay)
        delay *= 2
    raise TeacherError(f"failed after {retries} attempts — {last}\n({describe()})")


def smoke():
    """Prove the endpoint with a REAL completion, never a /health probe.

    /health returned "ok" on a server that was paged fully out and took 61.6 s for
    8 tokens (2026-07-26). Only a real completion proves a teacher works.
    """
    print(describe())
    t0 = time.time()
    r = complete([{"role": "user", "content": "What is 17*23? Reply with only the number."}])
    dt = time.time() - t0
    usage = r["usage"] or {}
    out_tok = usage.get("completion_tokens")
    print(f"  content        : {r['text'].strip()[:120]!r}")
    print(f"  finish_reason  : {r['finish_reason']}")
    print(f"  reasoning chars: {len(r['reasoning'])}")
    print(f"  latency        : {dt:.1f}s"
          + (f"  ({out_tok} out tok, {out_tok/dt:.1f} tok/s)" if out_tok else ""))
    ok = "391" in r["text"]
    print(f"  ARITHMETIC     : {'PASS' if ok else 'FAIL (check the model, not the pipe)'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if smoke() else 1)
