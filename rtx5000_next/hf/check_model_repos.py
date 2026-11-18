"""Can we get and fit each candidate model? Only metadata is fetched (no weights).

    python hf/check_model_repos.py [--models dit_xl2 sana_600m] [--with-text-encoder]
Gated repos need HF_TOKEN and an accepted licence.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hf_common import CATALOG, gb, repo_files, select_files, transformer_params

ROOT = pathlib.Path(__file__).resolve().parent.parent
VRAM_GB = 48


def status_of(repo):
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    try:
        info, files = repo_files(repo)
        return ("ok" if not info.gated else f"ok (gated={info.gated})"), files
    except GatedRepoError:
        return "GATED: accept licence and set HF_TOKEN", None
    except RepositoryNotFoundError:
        return "NOT FOUND", None
    except Exception as e:
        return f"ERROR {type(e).__name__}: {str(e)[:80]}", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(CATALOG))
    ap.add_argument("--with-text-encoder", action="store_true")
    a = ap.parse_args()

    import diffusers
    import torch
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(cache, exist_ok=True)
    print(f"torch {torch.__version__}  diffusers {diffusers.__version__}  free disk at {cache}: {gb(shutil.disk_usage(cache).free):.0f} GB")
    try:
        from huggingface_hub import whoami
        print("HF user:", whoami(token=os.environ.get("HF_TOKEN"))["name"])
    except Exception:
        print("HF user: not logged in (public repos still work)")

    rows, seen = [], {}
    for key in a.models:
        spec = CATALOG[key]
        cls = next((c for c in spec["classes"] if hasattr(diffusers, c)), None)
        for size, repo in spec["repos"].items():
            if repo not in seen:
                st, files = status_of(repo)
                row = dict(model=key, size=size, repo=repo, status=st, diffusers_class=cls)
                if files:
                    sel = select_files(files, a.with_text_encoder)
                    params, dts = transformer_params(repo, files)
                    row.update(
                        repo_total_gb=round(gb(sum(files.values())), 1),
                        download_gb=round(gb(sum(sel.values())), 1),
                        transformer_params_b=round(params / 1e9, 2),
                        transformer_dtypes=dts,
                        bf16_gb=round(params * 2 / 1e9, 1),
                        nvfp4_gb=round(params * 0.5625 / 1e9, 1),
                        fits_bf16_on_gpu=params * 2 / 1e9 * 1.3 < VRAM_GB,
                    )
                seen[repo] = row
            rows.append({**seen[repo], "model": key, "size": size})

    hdr = ["model", "size", "status", "class", "repo GB", "download GB", "params B", "bf16 GB", "nvfp4 GB", "fits 48GB bf16"]
    print("\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(x) for x in [
            r["model"], r["size"], r["status"], r["diffusers_class"] or "MISSING IN DIFFUSERS",
            r.get("repo_total_gb", "-"), r.get("download_gb", "-"), r.get("transformer_params_b", "-"),
            r.get("bf16_gb", "-"), r.get("nvfp4_gb", "-"), r.get("fits_bf16_on_gpu", "-")]) + " |")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "hf_check.json").write_text(json.dumps(rows, indent=1))
    bad = [r for r in rows if not r["status"].startswith("ok") or not r["diffusers_class"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} usable; details in results/hf_check.json")
    sys.exit(0)


if __name__ == "__main__":
    main()
