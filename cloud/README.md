# cloud/ — the TAJWID lane on rented GPUs
**Built 2026-07-27 · session e67037de · companion to `../ROOT_CALCULUS.md` + `../THE_BRAIN_SPEC.md`**

Everything here is written, selftested, and **proven against the local brain** before any
cloud key exists. When you authenticate, nothing changes but an environment variable.

---

## 1. The correction that reshapes the idea

**NVIDIA NIM cannot train.** It is inference microservices — you call it to *generate*.
So "build it on NVIDIA's GPUs" isn't available through NIM. What NIM *can* do is
generate teacher text fast, and that happens to attack the real bottleneck.

**Lightning AI can train.** Studios give you an A100/H100 you attach and detach. That is
where H2 (QLoRA distillation) and eventually H3 belong.

| lane | where it runs | why there |
|---|---|---|
| GGUF surgery, spectra, llama.cpp judging | **local** | the model is 129 GB on local disk; uploading it is absurd |
| Teacher trace generation | **NIM, or self-hosted teacher on Lightning** | local single-stream is 210 tok/s; batched remote is 20×+ |
| QLoRA distillation (H2 tajwīd) | **Lightning A100/H100** | needs sustained GPU hours — the thing blocked locally |
| Born-factored pretrain (H3) | **Lightning multi-GPU** | far beyond this box |
| Serving the finished student | **local** | it's 5–12 GB. That was the whole point. |

The 129 GB oracle **never leaves the house**. The student is pulled from HF *on the
remote machine*, which is faster than any upload from here.

---

## 2. Measured throughput — why the cloud is worth it, in numbers

From the smoke run against Ornith (2026-07-27, real completions):

- **210 tok/s** single-stream, **~3,957 completion tokens per sample**
- A useful distill set of **5,000 samples ≈ 19.8 M tokens → ~26 hours local**
- Full harvest of all **40,259** available workload prompts ≈ 159 M tokens → **~8.8 days local**
- Batched vLLM on one A100/H100 is conservatively **20×** aggregate → **26 hours becomes ~1.3 hours**

That is the entire argument for renting: not that the local card is weak, but that
single-stream generation cannot be batched and the corpus is the long pole.

---

## 3. What YOU do — two owner actions, both under five minutes

I do not create accounts, accept terms, or authorise spend. Both of these meter money,
so both are yours (hard stop *c*).

### A · Lightning (required for training)
```bash
lightning login
```
Then, from `E:\ABU\NEOCORTEX\cloud`:
```bash
python lightning_launch.py --start --machine A100
python lightning_launch.py --run "pip install -r requirements.txt"
```
`--stop` is what stops the meter. The launcher refuses to do anything until you're
authenticated, and says exactly this if you aren't.

### B · NIM (optional — only for bulk trace speed)
Get a key from NVIDIA's developer portal, then:
```powershell
$env:TEACHER_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:TEACHER_API_KEY  = "<your key>"
$env:TEACHER_MODEL    = "<the model id you chose>"
python teacher_client.py     # must print ARITHMETIC: PASS before anything else
```

**Read this before pointing traces at any managed API.** Generating training data from a
hosted model's outputs is governed by that provider's terms, and some providers restrict
using outputs to train other models. You sell software, so this is commercial exposure,
not a theoretical worry. `make_traces.py` therefore **refuses to run against a non-local
endpoint** unless you pass `--i-checked-the-licence`. That flag is a speed bump against
an expensive mistake — it is not legal advice, and I am not a lawyer.

**The clean way around it entirely:** self-host an open-weight teacher on the Lightning
GPU you're already paying for. No terms ambiguity, and you get logprobs — which
text-only APIs cannot give you and which make for better distillation.

---

## 4. The files

| file | what it does | proven? |
|---|---|---|
| `teacher_client.py` | one OpenAI-compatible client for NIM / Lightning / local. `smoke()` verifies with a **real completion**, never `/health` | ✅ 223 tok/s vs Ornith, `391` correct |
| `make_traces.py` | teacher answers to **your real prompts** (`train.jsonl`), resume-safe, per-stratum, licence gate | ✅ 4/5 kept, caught 1 empty-completion |
| `distill_qlora.py` | the H2 student trainer. `--dry-run` builds the dataset with no GPU and no model load | ✅ dry-run green |
| `lightning_launch.py` | Studio create/start/upload/run/stop. Refuses without auth | ✅ auth gate correct |
| `requirements.txt` | the remote stack. Local box has everything **except `peft`** | — |

---

## 5. Two traps already coded around

**Empty completions are failures, not short answers.** Reasoning models emit
`reasoning_content` first and return `content: ''` if the budget runs out —
`finish_reason: length`. This fired on the very first smoke run (1 of 5). Storing those
as training targets would teach the student to say nothing. `make_traces.py` skips and
logs them; `DEFAULT_MAX_TOKENS` is 4096, never small.

**Strata are mixed at TRAIN time, not harvest time.** Arabic is 0.0227% of natural
traffic and pays a ~2.3–2.8× tokenizer tax. Left proportional it contributes no gradient;
blended at harvest it contaminates the workload distribution. `--arabic-ratio` makes it
an explicit, logged, tunable decision, and the two holdouts are scored **separately**.

---

## 6. The gate that decides whether any of this worked

Same judge as every other gate in this project — llama.cpp on a held-out set, never the
training loss, never a self-written reference:

```bash
llama-perplexity -m <student.gguf> -f corpus/calib_holdout.txt        # workload
llama-perplexity -m <student.gguf> -f corpus/calib_arabic_holdout.txt # arabic, apart
```

Target: student holdout PPL within **5%** of the teacher's on the workload stratum.
Anything else is a number, not a result.
