"""Eval 3 pre-flight: is this server ready for the 4.5 h run? Read-only, no GPU work, about 10 seconds.

    python eval3/preflight.py                 # check everything
    python eval3/preflight.py --models pixart_sigma:1024,512,256 sd35_medium:1024 flux_schnell:1024
Writes results/eval3/preflight.json and prints one line per check: PASS / WARN / FAIL.
FAIL = a hard blocker for some test (the line says which). WARN = an optional piece is missing and the named tests will be skipped.
Exit code 1 only if torch or CUDA are missing.
"""
import argparse
import importlib
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from common import RESULTS  # noqa: E402

REQUIRED = ["torch", "diffusers", "torchao", "transformers", "lpips", "sentencepiece", "matplotlib", "scipy", "PIL", "numpy"]
OPTIONAL = {  # import name -> what needs it
    "sageattention": "Sage 2.x attention variants (T-attn)",
    "mslk": "stock_tk baseline, torchao's own Triton quantizer (T-step)",
    "torchmetrics": "FID/KID metrics (T-qual)",
    "torch_fidelity": "FID/KID metrics, torchmetrics backend (T-qual)",
    "psutil": "host memory log per variant (bug 27)",
    "huggingface_hub": "downloads and cache check",
}
TOOLS = {"nvcc": "CUDA swizzled quantize kernel", "nsys": "nsys stage timing (T-roof)", "ncu": "ncu DRAM bytes (T-roof)", "ninja": "CUDA extension build"}
DEFAULT_MODELS = ["pixart_sigma:256,512,1024", "sd35_medium:1024", "flux_schnell:1024"]
METRIC_REPOS = {
    "openai/clip-vit-large-patch14": "CLIP score (T-qual)",
    "yuvalkirstain/PickScore_v1": "PickScore preference metric (T-qual)",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "PickScore processor (T-qual)",
}

rows = []


def add(status, name, detail=""):
    rows.append(dict(status=status, name=name, detail=detail))
    print(f"{status:5s} {name}" + (f"  {detail}" if detail else ""))


def check_python():
    for m in REQUIRED:
        try:
            mod = importlib.import_module(m)
            add("PASS", f"import {m}", getattr(mod, "__version__", ""))
        except Exception as e:
            add("FAIL", f"import {m}", f"{type(e).__name__}: {str(e)[:80]}  -> pip install -r requirements_next.txt")
    for m, why in OPTIONAL.items():
        try:
            mod = importlib.import_module(m)
            add("PASS", f"import {m}", getattr(mod, "__version__", ""))
        except Exception as e:
            add("WARN", f"import {m} missing", f"needed for: {why}")
    try:
        from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4  # noqa: F401
        add("PASS", "mslk NVFP4 Triton kernel")
    except Exception:
        add("WARN", "mslk NVFP4 Triton kernel missing (the PyPI 'mslk 0.0.0' is a placeholder)", "stock_tk rows will fail")
    try:
        import sageattention as s
        add("PASS" if hasattr(s, "sageattn_qk_int8_pv_fp8_cuda") else "WARN",
            "sageattention 2.x kernels" if hasattr(s, "sageattn_qk_int8_pv_fp8_cuda") else "sageattention is 1.x (slower)")
    except Exception:
        pass


def check_gpu():
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        add("FAIL", "CUDA not available")
        return False
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    add("PASS", f"GPU {p.name}", f"{p.total_memory / 2**30:.0f} GB, sm_{cap[0]}{cap[1]}, torch {torch.__version__}")
    if cap[0] < 10:
        add("FAIL", "GPU is not Blackwell", "NVFP4 _scaled_mm needs sm_100/sm_120")
    free, tot = torch.cuda.mem_get_info()
    add("PASS" if free > 0.9 * tot else "WARN", "GPU idle", f"{(tot - free) / 2**30:.1f} GB already in use")
    return True


def check_tools():
    for t, why in TOOLS.items():
        add("PASS" if shutil.which(t) else "WARN", f"tool {t}" + ("" if shutil.which(t) else " not found"), "" if shutil.which(t) else f"needed for: {why}")


def check_host():
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 2**30
    add("PASS" if free > 80 else "WARN", f"disk free at HF cache {cache}", f"{free:.0f} GB (SD3.5 + FLUX + encoders need about 60 GB)")
    try:
        import psutil
        vm = psutil.virtual_memory()
        add("PASS" if vm.total > 60 * 2**30 else "WARN", "host RAM", f"{vm.total / 2**30:.0f} GB total, {vm.available / 2**30:.0f} GB free")
    except Exception:
        pass
    add("PASS" if os.environ.get("HF_TOKEN") or (pathlib.Path.home() / ".cache/huggingface/token").exists() else "WARN",
        "HF token", "SD3.5 and FLUX.1 weights are gated; the download needs a token if they are not cached yet")


def cached_files(repo):
    """Set of file names (relative) cached for the repo, from the newest snapshot. Empty set if the repo is not cached."""
    try:
        from huggingface_hub import scan_cache_dir
        for r in scan_cache_dir().repos:
            if r.repo_id == repo and r.repo_type == "model":
                revs = sorted(r.revisions, key=lambda x: x.last_modified, reverse=True)
                return {f.file_path.relative_to(rev.snapshot_path).as_posix() for rev in revs[:1] for f in rev.files}
    except Exception:
        pass
    return set()


def check_models(specs):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hf"))
    from hf_common import CATALOG
    for spec in specs:
        name, _, sizes = spec.partition(":")
        if name not in CATALOG:
            add("FAIL", f"model {name}", "not in hf/catalog.json")
            continue
        for s in (sizes.split(",") if sizes else list(CATALOG[name]["repos"])):
            repo = CATALOG[name]["repos"].get(s)
            if repo is None:
                add("WARN", f"model {name} {s}px", "no repo for this size in the catalog")
                continue
            files = cached_files(repo)
            has_tr = any(f.startswith("transformer/") and f.endswith((".safetensors", ".bin")) for f in files)
            has_vae = any(f.startswith("vae/") for f in files)
            has_te = any(f.startswith("text_encoder") and f.endswith((".safetensors", ".bin")) for f in files)
            add("PASS" if has_tr else "FAIL", f"model {name} {s}px  {repo}",
                f"transformer {'yes' if has_tr else 'NO'}, vae {'yes' if has_vae else 'no'}, text encoder {'yes' if has_te else 'no'}"
                + ("" if has_tr else "  -> python eval3/prefetch.py --models " + name))
    for repo, why in METRIC_REPOS.items():
        ok = bool(cached_files(repo))
        add("PASS" if ok else "WARN", f"metric weights {repo}", "" if ok else f"missing; needed for: {why}  -> python eval3/prefetch.py --metrics")
    try:
        import lpips  # noqa: F401
        p = pathlib.Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
        ok = any(p.glob("alexnet-*.pth")) if p.exists() else False
        add("PASS" if ok else "WARN", "LPIPS AlexNet backbone cached", "" if ok else "will download on first use (prefetch.py --metrics does it now)")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="name:size,size ...")
    a = ap.parse_args()
    check_python()
    gpu_ok = check_gpu()
    check_tools()
    check_host()
    check_models(a.models)
    (RESULTS / "eval3").mkdir(parents=True, exist_ok=True)
    (RESULTS / "eval3" / "preflight.json").write_text(json.dumps(rows, indent=1))
    n = {k: sum(r["status"] == k for r in rows) for k in ("PASS", "WARN", "FAIL")}
    print(f"\npreflight: {n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail -> results/eval3/preflight.json")
    sys.exit(0 if gpu_ok else 1)


if __name__ == "__main__":
    main()
