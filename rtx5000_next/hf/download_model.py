"""Download only what the transformer tests need (transformer, vae, configs). Text encoders are skipped.

    python hf/download_model.py dit_xl2 --size 512 --dry-run
    python hf/download_model.py sd35_medium --size 1024
Set HF_TOKEN for gated repos. pip package hf_transfer speeds this up: HF_HUB_ENABLE_HF_TRANSFER=1.
"""
import argparse
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hf_common import CATALOG, gb, repo_files, select_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(CATALOG))
    ap.add_argument("--size", default=None)
    ap.add_argument("--with-text-encoder", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="skip the free-disk check")
    a = ap.parse_args()

    repos = CATALOG[a.model]["repos"]
    repo = repos[a.size] if a.size else next(iter(repos.values()))
    _, files = repo_files(repo)
    sel = select_files(files, a.with_text_encoder)
    total = sum(sel.values())
    print(f"{repo}: {len(sel)} of {len(files)} files, {gb(total):.1f} GB of {gb(sum(files.values())):.1f} GB")
    for n, s in sorted(sel.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {gb(s):7.2f} GB  {n}")
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(cache, exist_ok=True)
    free = shutil.disk_usage(cache).free
    print(f"free disk: {gb(free):.0f} GB")
    if a.dry_run:
        return
    if free < total * 1.2 and not a.force:
        sys.exit("not enough free disk (need 1.2x); use --force to override")

    from huggingface_hub import snapshot_download
    path = snapshot_download(repo, allow_patterns=list(sel), token=os.environ.get("HF_TOKEN"))
    print("downloaded to", path)


if __name__ == "__main__":
    main()
