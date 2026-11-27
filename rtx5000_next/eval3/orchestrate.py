"""Unattended run driver for the server: one command for the smoke gate and for slot 1 (5 h, nobody watching).

    python eval3/orchestrate.py --mode smoke --hours 1                 # G3: every slot-1 stage on tiny inputs (results_smoke/)
    python eval3/orchestrate.py --mode slot1 --hours 4.8 --dry-run     # G4: print the schedule, run nothing
    setsid nohup python eval3/orchestrate.py --mode slot1 --hours 4.8 > logs/eval3/slot1.log 2>&1 < /dev/null &
What it guarantees (each item exists because of a failure seen earlier; see BUGS.md and docs/EVAL3_HANDOFF_PLAN.md section 7b):
  - one GPU run per machine: an exclusive flock on rtx5000_next/.eval3_run.lock; a second start exits at once
  - offline: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set for every child, so nothing downloads during the run
  - GPU idle at start (waits up to 10 min for other processes, then aborts); every stage also has the GPU guard of run_eval3.sh
  - slot1 refuses to start unless the git tree has no tracked changes AND a GREEN smoke run exists for the same commit
  - priority order (P0 gates, P1 reviewer-critical, P2, P3); a stage is skipped if its estimate no longer fits before the deadline
  - per-stage hard timeout (2x estimate, at least +10 min), the whole process group is killed on timeout
  - one retry for a stage that failed with an out-of-memory or CUDA error (fresh process); no retry for other errors
  - PASS = run_eval3.sh marked the stage done AND eval3/verify_run.py's checker for that stage passed
  - STATUS.md / STATUS.json rewritten after every stage; the last stage (W399) writes VERIFY.md and packages results_eval3*.tgz
Verdict: GREEN (all passed), YELLOW (all P0/P1 passed), RED (a P0/P1 stage failed or the run aborted).
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (stage, priority, estimated minutes in slot1, estimated minutes in smoke). Slot-1 estimates are scaled from the clean server
# runs of 2026-11-27 (13 configs x 2 sizes x 3 repeats = 15 min; 6 quality variants at 512 px, 30 x 2 = 8 min); smoke estimates
# are guesses (compile time dominates at 256 px).
STAGES = [
    ("W100", 0, 1, 1), ("W104", 1, 6, 2), ("W299", 1, 6, 3), ("W300", 1, 22, 6), ("W302", 1, 6, 3), ("W301", 1, 22, 8),
    ("W201", 1, 5, 3), ("W202", 1, 15, 5), ("W203", 1, 1, 1),
    ("W303", 2, 12, 4), ("W304", 2, 18, 4), ("W305", 2, 5, 2),
    ("W306", 3, 6, 2),
    ("W399", 0, 3, 2),   # verification + packaging: always runs last, reserved time
]
# MODELS: PixArt only by default. SD3.5 Medium and FLUX schnell (slow loads, adapters not yet proven on the server) are added with
# --models for the 6 h slot, and only after a smoke run that included them.
SMOKE_ENV = dict(EVAL3_RESULTS="results_smoke", TAG_SFX="_smoke", RUN_MODE="smoke", STEP_SIZES="256", REPS="1", STEP_ITERS="3",
                 HEAD_REPS="1", ROOF_SIZES="256", ROOF_ITERS="1", STATIC_SIZES="256", CALIB_N="1", SEEDS3="1", SEEDS="1",
                 NPROMPTS="2", NQHI="2", QHI="256", SIZE_Q="256", STEPS="4", XGPU_SIZES="256", ATTN_SIZES="256",
                 PEAKS_ARGS="--quick", KISO_ARGS="--quick", MODELS="pixart_sigma:256")
SLOT1_ENV = dict(RUN_MODE="slot1", MODELS="pixart_sigma:1024")
OFFLINE = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
RETRY_PATTERNS = ("OutOfMemoryError", "CUDA out of memory", "CUDA error", "CUBLAS_STATUS", "cudaErrorLaunchFailure")


def now():
    return dt.datetime.now()


def sh(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def git_state():
    sha = sh(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = sh(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
    return sha, dirty


def gpu_users():
    out = sh(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


class Run:
    def __init__(self, mode, hours, dry, models=None):
        self.mode, self.dry = mode, dry
        self.env = dict(os.environ, **OFFLINE, **(SMOKE_ENV if mode == "smoke" else SLOT1_ENV))
        if models:
            self.env["MODELS"] = models
        self.warnings = []
        self.res = ROOT / self.env.get("EVAL3_RESULTS", "results")
        self.e3 = self.res / "eval3"
        self.start = now()
        self.deadline = self.start + dt.timedelta(hours=hours)
        self.rows = []
        self.sha, self.dirty = git_state()
        self.verdict = None
        self.abort = None

    # ---- status
    def write_status(self, final=False):
        self.e3.mkdir(parents=True, exist_ok=True)
        st = dict(mode=self.mode, sha=self.sha, start=str(self.start), deadline=str(self.deadline), updated=str(now()),
                  minutes_left=round((self.deadline - now()).total_seconds() / 60, 1), stages=self.rows, verdict=self.verdict, abort=self.abort)
        (self.e3 / "STATUS.json").write_text(json.dumps(st, indent=1))
        L = [f"# {self.mode} run status ({'final' if final else 'running'})\n",
             f"commit {self.sha[:10]} | start {self.start:%H:%M} | deadline {self.deadline:%H:%M} | updated {now():%H:%M:%S} | "
             f"{st['minutes_left']} min left\n"]
        if self.abort:
            L.append(f"**ABORTED: {self.abort}**\n")
        for w in self.warnings:
            L.append(f"WARNING: {w}\n")
        L.append(f"models: {self.env.get('MODELS')}\n")
        L += ["| stage | prio | status | minutes (est) | note |", "|---|---|---|---|---|"]
        for r in self.rows:
            L.append(f"| {r['id']} | P{r['prio']} | {r['status']} | {r.get('minutes', '-')} ({r['est']}) | {str(r.get('note', ''))[:160]} |")
        if self.verdict:
            L.append(f"\n{self.mode.upper()} VERDICT: {self.verdict}")
        (self.e3 / "STATUS.md").write_text("\n".join(L) + "\n")

    # ---- gates before any GPU work
    def preconditions(self):
        if self.mode == "slot1":
            if self.dirty:
                return f"tracked files have local changes (git status): {self.dirty.splitlines()[:5]}"
            sm = ROOT / "results_smoke" / "eval3" / "STATUS.json"
            if not sm.exists():
                return "no smoke run found (run --mode smoke first)"
            s = json.loads(sm.read_text())
            if s.get("verdict") != "GREEN":
                return f"smoke verdict is {s.get('verdict')}, not GREEN"
            if s.get("sha") != self.sha:
                return f"smoke ran on commit {str(s.get('sha'))[:10]}, current is {self.sha[:10]}: rerun the smoke after pulling"
        for i in range(60):
            users = gpu_users()
            if not users:
                break
            if i == 0:
                print(f"GPU in use by {users}; waiting up to 10 min", flush=True)
            time.sleep(10)
        else:
            return f"GPU still in use after 10 min: {gpu_users()}"
        # check_env.py checks that pixi.toml DECLARES every import (reproducibility). It is a warning here, not a gate: the runtime
        # gate is the preflight stage W100 (every package actually imports), and the server may run on an env built before the
        # manifest was completed.
        if (ROOT / "eval3" / "check_env.py").exists():
            r = sh([sys.executable, "eval3/check_env.py"], env=self.env)
            if r.returncode:
                self.warnings.append("check_env.py: " + " / ".join((r.stdout + r.stderr).strip().splitlines()[-2:])[:200])
        return None

    def write_meta(self):
        self.e3.mkdir(parents=True, exist_ok=True)
        (self.e3 / "CLEAN_SINCE").write_text(now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        q = ("import json,torch,torchao,diffusers,transformers;print(json.dumps(dict(torch=torch.__version__,torchao=torchao.__version__,"
             "diffusers=diffusers.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,"
             "gpu=torch.cuda.get_device_name(0),cap=list(torch.cuda.get_device_capability(0)))))")
        r = sh([sys.executable, "-c", q], env=self.env)
        meta = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else dict(error=r.stderr[-300:])
        meta.update(sha=self.sha, mode=self.mode, start=str(self.start), driver=sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).stdout.strip(),
                    nvcc=(sh(["bash", "-c", "nvcc --version | tail -1"]).stdout.strip() or None), python=sys.executable)
        (self.e3 / "env_meta.json").write_text(json.dumps(meta, indent=1))

    # ---- one stage
    def run_stage(self, sid, est):
        for m in (".done", ".fail"):
            (self.e3 / m / sid).unlink(missing_ok=True)
        timeout = max(2 * est, est + 10) * 60
        t0 = time.time()
        p = subprocess.Popen(["bash", "run_eval3.sh", "--only", sid], cwd=ROOT, env=self.env, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            timed_out = True
        minutes = round((time.time() - t0) / 60, 1)
        print(out, flush=True)
        done = (self.e3 / ".done" / sid).exists()
        log = ROOT / "logs" / "eval3" / f"{sid}{self.env.get('TAG_SFX', '')}.log"
        tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
        if timed_out:
            return "FAIL", minutes, f"timeout after {timeout // 60} min", tail
        if not done:
            return "FAIL", minutes, "stage failed: " + " / ".join(tail[-3:]), tail
        c = sh([sys.executable, "eval3/verify_run.py", "--stage", sid], env=self.env)
        if c.returncode:
            return "FAIL", minutes, "checker: " + c.stdout.strip()[-200:], tail
        return "PASS", minutes, "", tail

    def main(self):
        est_col = 2 if self.mode == "slot1" else 3
        sched = [(s[0], s[1], s[est_col]) for s in STAGES]
        reserve = sched[-1][2] + 3
        if self.dry:
            t = 0.0
            budget = (self.deadline - self.start).total_seconds() / 60 - reserve
            print(f"{self.mode}: {len(sched)} stages, budget {budget:.0f} min before verification")
            for sid, prio, est in sched:
                fits = t + est <= budget or sid == "W399"
                t += est if fits else 0
                print(f"  {sid} P{prio} est {est:>3} min  cumulative {t:>5.0f}  {'' if fits else 'WOULD BE SKIPPED (time)'}")
            return 0
        lock = open(ROOT / ".eval3_run.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another eval3 run holds the lock; not starting a second one", flush=True)
            return 3
        self.abort = self.preconditions()
        if self.abort:
            self.verdict = "RED"
            self.write_status(final=True)
            print("ABORT:", self.abort, flush=True)
            return 2
        self.write_meta()
        self.write_status()
        for sid, prio, est in sched:
            row = dict(id=sid, prio=prio, est=est)
            self.rows.append(row)
            left = (self.deadline - now()).total_seconds() / 60
            if sid != "W399" and est > left - reserve:
                row.update(status="SKIPPED", note=f"time: {left:.0f} min left, needs {est} + {reserve} reserved")
                self.write_status()
                continue
            row.update(status="RUNNING", note=f"started {now():%H:%M}")
            self.write_status()
            status, minutes, note, tail = self.run_stage(sid, est)
            if status == "FAIL" and sid != "W399" and any(pat in "\n".join(tail) for pat in RETRY_PATTERNS) \
                    and est <= (self.deadline - now()).total_seconds() / 60 - reserve:
                print(f"{sid}: OOM/CUDA error, one retry in a fresh process", flush=True)
                status, minutes2, note, tail = self.run_stage(sid, est)
                minutes += minutes2
                note = "after 1 retry: " + note
            row.update(status=status, minutes=minutes, note=note)
            self.write_status()
        rows = [r for r in self.rows if r["id"] != "W399"]
        p1 = all(r["status"] == "PASS" for r in rows if r["prio"] <= 1)
        allp = all(r["status"] == "PASS" for r in self.rows)
        self.verdict = "GREEN" if allp else ("YELLOW" if p1 else "RED")
        self.write_status(final=True)
        print(f"{self.mode.upper()} VERDICT: {self.verdict}  (see {self.e3 / 'STATUS.md'} and VERIFY.md)", flush=True)
        return 0 if self.verdict != "RED" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "slot1"], required=True)
    ap.add_argument("--hours", type=float, default=None, help="time budget from now (default: smoke 1.0, slot1 4.8)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default=None, help='override MODELS, e.g. "pixart_sigma:1024 sd35_medium:1024 flux_schnell:1024"')
    a = ap.parse_args()
    hours = a.hours if a.hours is not None else (1.0 if a.mode == "smoke" else 4.8)
    sys.exit(Run(a.mode, hours, a.dry_run, a.models).main())


if __name__ == "__main__":
    main()
