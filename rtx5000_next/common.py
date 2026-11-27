import json
import logging
import os
import pathlib
import statistics
import sys
import time

logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)  # torchao import notice, harmless

ROOT = pathlib.Path(__file__).resolve().parent
# EVAL3_RESULTS redirects every result file (the eval3 smoke run writes to results_smoke/ so its tiny runs never mix with real ones)
RESULTS = pathlib.Path(os.environ["EVAL3_RESULTS"]).resolve() if os.environ.get("EVAL3_RESULTS") else ROOT / "results"


def _find_code_dir():
    for c in (os.environ.get("NVFP4_CODE"), ROOT.parent / "rtx_5000_runs", ROOT.parent, ROOT):
        if c and (pathlib.Path(c) / "fused_adaln_nvfp4.py").exists():
            return pathlib.Path(c).resolve()
    raise SystemExit("fused_adaln_nvfp4.py not found: set NVFP4_CODE to the folder that holds it")


CODE = _find_code_dir()
for _p in (str(CODE), str(ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def gpu_name():
    """Name of GPU 0 if torch is already imported and CUDA is up, else None (never imports torch: CPU scripts stay light)."""
    t = sys.modules.get("torch")
    try:
        return t.cuda.get_device_name(0) if t is not None and t.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def append_jsonl(name, rec):
    RESULTS.mkdir(exist_ok=True)
    rec = {**rec, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec.setdefault("gpu", gpu_name())  # eval3 cross-GPU table (reviewer point 5): every result row says which GPU made it
    with open(RESULTS / name, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_json(rel_path, obj):
    p = RESULTS / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def read_jsonl(name):
    p = RESULTS / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return dict(n=n, median=statistics.median(xs), min=xs[0], p10=q(0.1), p90=q(0.9), max=xs[-1],
                std=statistics.pstdev(xs) if n > 1 else 0.0)


def load_prompt_embeds(path):
    """List of {prompt, pe, pm, ne, nm} on CPU. Accepts the multi-prompt file and the old single-prompt embeds.pt."""
    import torch
    d = torch.load(path)
    return d["items"] if "items" in d else [d]


def cuda_ms(fn, iters=100, warmup=20):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out
