"""End-to-end proof: make exactly the OpenAI call Hermes makes, resolved from
Hermes's OWN config.yaml. If this passes, Hermes is wired to GLM-5.2.

Writes a UTF-8 markdown report as well as stdout, because the Windows console is
cp1252 and cannot print Arabic.
"""
import json, re, sys, time, urllib.request, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CFG = r"C:\Users\wirec\AppData\Local\hermes\config.yaml"
OUT = r"E:\ABU\NEOCORTEX\hermes_link_proof.md"

txt = open(CFG, encoding="utf-8", errors="replace").read()
prov = re.search(r"^model:\s*\n(?:.*\n)*?\s+provider:\s*(\S+)", txt, re.M).group(1)
blk = re.search(rf"^  {re.escape(prov)}:\s*\n((?:    .*\n)+)", txt, re.M).group(1)
base = re.search(r"base_url:\s*(\S+)", blk).group(1)
model = re.search(r"default_model:\s*(\S+)", blk).group(1)
key = re.search(r"api_key:\s*(\S+)", blk).group(1)

lines = ["# Hermes -> GLM-5.2 link proof", "",
         f"- Hermes provider: `{prov}`", f"- base_url: `{base}`",
         f"- model: `{model}`", f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
print(f"Hermes provider = {prov}\n  base_url = {base}\n  model = {model}\n")


def ask(msg, max_tokens=1200):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": max_tokens, "temperature": 0.3}).encode("utf-8")
    r = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                               headers={"Content-Type": "application/json",
                                        "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(r, timeout=3600).read().decode("utf-8"))
    el = time.perf_counter() - t0
    m = d["choices"][0]["message"]
    return ((m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip(),
            el, d.get("timings", {}))


TESTS = [
    ("english-direct", "Reply with exactly: HERMES LINK OK"),
    ("arabic-root", 'ما هو الجذر الثلاثي لكلمة "مكتوب"؟ أجب بجملة واحدة قصيرة.'),
    ("arabic-meaning", 'اشرح معنى كلمة "رحمة" في جملتين.'),
    ("arithmetic", "What is 17 * 23? Reply with just the number."),
]

ok = 0
for name, q in TESTS:
    try:
        c, rsn, el, tm = ask(q)
    except Exception as e:
        print(f"[{name}] ERROR {e}")
        lines += [f"## {name}", f"**ERROR** {e}", ""]
        continue
    got = bool(c)
    ok += got
    tps = tm.get("predicted_per_second", 0)
    print("=" * 72)
    print(f"[{name}]  {el:.1f}s  {tps:.2f} tok/s  content={'YES' if got else 'EMPTY'}")
    print("Q:", q)
    print("A:", c[:400] if c else "(empty)")
    if rsn:
        print("  thinking:", rsn[:150].replace("\n", " "), "...")
    lines += [f"## {name}", "", f"**Q:** {q}", "", f"**A:** {c or '_(empty)_'}", ""]
    if rsn:
        lines += ["<details><summary>thinking</summary>", "", rsn[:1500], "", "</details>", ""]
    lines += [f"`{el:.1f}s` · `{tps:.2f} tok/s`", ""]

verdict = f"{ok}/{len(TESTS)} returned content"
lines += ["---", f"**VERDICT: {verdict}**", ""]
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("=" * 72)
print(f"VERDICT: {verdict}")
print(f"report: {OUT}")
