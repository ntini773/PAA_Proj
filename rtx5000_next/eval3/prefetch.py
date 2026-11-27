"""Download everything the Eval 3 run needs while the GPU is not needed yet. Safe to rerun: cached files are skipped.

    python eval3/prefetch.py --metrics                       # CLIP, PickScore, LPIPS backbone, FID Inception
    python eval3/prefetch.py --models sd35_medium flux_schnell --with-text-encoder
    python eval3/prefetch.py --models pixart_sigma --sizes 512 1024
    python eval3/prefetch.py --dry-run --models flux_schnell --with-text-encoder     # sizes only
Gated repos (SD3.5, FLUX.1): set HF_TOKEN and accept the licence on the model page first.
Every item runs in its own try/except; a failure is printed and the rest continue. Exit code 1 if anything failed.
"""
import argparse
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "hf"))
from hf_common import CATALOG, gb, repo_files, select_files  # noqa: E402

METRIC_REPOS = ["openai/clip-vit-large-patch14", "yuvalkirstain/PickScore_v1"]
METRIC_PROCESSOR_ONLY = {"laion/CLIP-ViT-H-14-laion2B-s32B-b79K": ["*.json", "*.txt", "tokenizer*", "preprocessor*"]}
failed = []


def attempt(name, fn):
    try:
        fn()
        print(f"OK    {name}", flush=True)
    except BaseException as e:  # noqa: BLE001  (SystemExit from library code too)
        failed.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)


def get_model(model, size, with_text, dry):
    repo = CATALOG[model]["repos"][size]
    _, files = repo_files(repo)
    sel = select_files(files, with_text)
    total = sum(sel.values())
    print(f"{repo}: {len(sel)} files, {gb(total):.1f} GB (text encoders {'included' if with_text else 'skipped'})")
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free
    if dry:
        return
    if free < total * 1.1:
        raise RuntimeError(f"free disk {gb(free):.0f} GB < needed {gb(total) * 1.1:.0f} GB")
    from huggingface_hub import snapshot_download
    snapshot_download(repo, allow_patterns=list(sel), token=os.environ.get("HF_TOKEN"))


def get_metrics(dry):
    from huggingface_hub import snapshot_download
    for repo in METRIC_REPOS:
        attempt(f"metric repo {repo}", (lambda r=repo: print("dry-run", r) if dry else snapshot_download(r, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*", "merges*", "vocab*"]))
                )
    for repo, pats in METRIC_PROCESSOR_ONLY.items():
        attempt(f"processor {repo}", (lambda r=repo, p=pats: print("dry-run", r) if dry else snapshot_download(r, allow_patterns=p)))
    if dry:
        return

    def lpips_alex():
        import lpips
        lpips.LPIPS(net="alex", verbose=False)
    attempt("LPIPS AlexNet backbone", lpips_alex)

    def fid_inception():
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            FrechetInceptionDistance(feature=2048)  # triggers the Inception weight download
        except ImportError:
            raise RuntimeError("torchmetrics/torch-fidelity not installed: pip install torchmetrics torch-fidelity")
    attempt("FID Inception weights", fid_inception)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], choices=list(CATALOG))
    ap.add_argument("--sizes", nargs="*", default=None, help="default: 1024 for new models, all sizes for pixart_sigma")
    ap.add_argument("--with-text-encoder", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.metrics:
        get_metrics(a.dry_run)
    for m in a.models:
        sizes = a.sizes or (list(CATALOG[m]["repos"]) if m == "pixart_sigma" else ["1024"])
        done = set()
        for s in sizes:
            repo = CATALOG[m]["repos"].get(s)
            if repo is None or repo in done:  # SD3.5 and FLUX use one repo for every size
                continue
            done.add(repo)
            attempt(f"{m} {s}px", lambda m=m, s=s: get_model(m, s, a.with_text_encoder, a.dry_run))
    print(f"\nprefetch finished; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
