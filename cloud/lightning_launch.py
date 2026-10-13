"""Drive a Lightning Studio from here: upload the lane, run a job, pull results back.

WHAT THIS DOES AND DOES NOT DO
It does NOT create an account, accept terms, or authorise any spend — those are
owner actions by design (see README). It assumes you have already run `lightning login`
or set LIGHTNING_USER_ID / LIGHTNING_API_KEY, and it fails with a plain message if not.

WHY A STUDIO AND NOT A ONE-SHOT JOB
The lane is iterative: generate traces, train, judge, adjust the mix, train again.
A Studio is a persistent machine you attach a GPU to and detach when idle, so the
corpus and checkpoints survive between rounds instead of being re-uploaded each time.

WHAT GETS UPLOADED (small — deliberately)
  cloud/*.py, requirements.txt, cloud/traces/*.jsonl, corpus/calib_*.txt
NOT the 129 GB GGUF. The oracle stays home; the student is pulled from HF on the
remote machine, which is faster than any upload from here.

Usage:
  python lightning_launch.py --status
  python lightning_launch.py --start --machine A100
  python lightning_launch.py --run "python distill_qlora.py --model <id> --dry-run"
  python lightning_launch.py --stop
"""
import argparse
import os
import sys

NAME = os.environ.get("LIGHTNING_STUDIO", "abuz8-tajwid")
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE")   # optional
UPLOAD = [
    ("cloud/teacher_client.py", "teacher_client.py"),
    ("cloud/make_traces.py", "make_traces.py"),
    ("cloud/distill_qlora.py", "distill_qlora.py"),
    ("cloud/requirements.txt", "requirements.txt"),
]
ROOT = r"E:\ABU\NEOCORTEX"


def need_auth():
    have_env = bool(os.environ.get("LIGHTNING_USER_ID") and os.environ.get("LIGHTNING_API_KEY"))
    have_file = os.path.exists(os.path.expanduser("~/.lightning/credentials.json"))
    return not (have_env or have_file)


def get_studio(create=False):
    from lightning_sdk import Studio
    kw = {"name": NAME}
    if TEAMSPACE:
        kw["teamspace"] = TEAMSPACE
    if create:
        kw["create_ok"] = True
    return Studio(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--run", type=str)
    ap.add_argument("--machine", default="A100",
                    help="Lightning machine type, e.g. CPU / T4 / A10G / A100 / H100")
    a = ap.parse_args()

    if need_auth():
        print("NOT AUTHENTICATED — Lightning needs one owner action first:\n"
              "  lightning login          (opens a browser)\n"
              "or set LIGHTNING_USER_ID and LIGHTNING_API_KEY in the environment.\n"
              "I do not create accounts or accept terms on your behalf.")
        return 2

    try:
        from lightning_sdk import Machine
    except Exception as e:
        print(f"lightning_sdk import failed: {e}")
        return 1

    s = get_studio(create=a.start)

    if a.status:
        print(f"studio  : {NAME}")
        print(f"status  : {getattr(s, 'status', '?')}")
        print(f"machine : {getattr(s, 'machine', '?')}")
        return 0

    if a.start:
        mt = getattr(Machine, a.machine.upper(), None)
        if mt is None:
            print(f"unknown machine '{a.machine}'. Available: "
                  f"{[x for x in dir(Machine) if x.isupper()]}")
            return 1
        print(f"starting {NAME} on {a.machine} …")
        s.start(machine=mt)
        print(f"started. status={getattr(s, 'status', '?')}")

    if a.upload or a.start:
        for local, remote in UPLOAD:
            p = os.path.join(ROOT, local.replace("cloud/", "cloud\\"))
            if os.path.exists(p):
                print(f"  upload {local} -> {remote}")
                s.upload_file(p, remote)
        tdir = os.path.join(ROOT, "cloud", "traces")
        if os.path.isdir(tdir):
            for f in os.listdir(tdir):
                if f.endswith(".jsonl"):
                    print(f"  upload traces/{f}")
                    s.upload_file(os.path.join(tdir, f), f"traces/{f}")
        for f in ("calib_holdout.txt", "calib_arabic_holdout.txt"):
            p = os.path.join(ROOT, "corpus", f)
            if os.path.exists(p):
                print(f"  upload corpus/{f}")
                s.upload_file(p, f"corpus/{f}")

    if a.run:
        print(f"run: {a.run}")
        print(s.run(a.run))

    if a.stop:
        print(f"stopping {NAME} (this is what stops the meter) …")
        s.stop()
        print("stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
