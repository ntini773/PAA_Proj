"""Environment declaration check: is every third-party import in the repo declared in pixi.toml? No GPU, no imports of the scanned code, about 1 second.

    python eval3/check_env.py                 # scan every *.py under the repo root, compare with pixi.toml
    python eval3/check_env.py --root .. --manifest ../pixi.toml
Inputs : all *.py files under --root (default: rtx_5000_runs), skipping results, logs, builds, caches and data folders; pixi.toml
         ([dependencies] and [pypi-dependencies]).
Outputs: one line per undeclared package with the files that import it, and the install hint. Source-built optional packages
         (sageattention, sageattn3, flash_attn, xformers) are listed as NOTE with their `pixi run build-*` task, not as failures.
Exit code 1 if any third-party import is not declared and not in the standard library.
"""
import argparse
import ast
import pathlib
import re
import sys
import tomllib
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent.parent  # rtx_5000_runs

SKIP_DIRS = {".git", ".pixi", "__pycache__", "results", "server_results", "logs", "build", "figures", "data",
             "tests_data", "node_modules", ".venv", "venv"}

# import name -> pypi/conda distribution name, only where they differ
IMPORT_TO_DIST = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "skimage": "scikit-image",
    "torch_fidelity": "torch-fidelity",
    "lightning_utilities": "lightning-utilities",
    "huggingface_hub": "huggingface-hub",
    "google": "protobuf",  # google.protobuf
    "attr": "attrs",
    "dateutil": "python-dateutil",
    "triton": "triton",
    "torchvision": "torchvision",
    "pynvml": "nvidia-ml-py",
}

# optional packages that are built from source; pixi task that builds each
SOURCE_BUILT = {
    "sageattention": "build-sage2",
    "sageattn3": "build-sage3",
    "flash_attn": "build-flashattn2",
    "xformers": "build-xformers",
}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_packages(manifest):
    data = tomllib.loads(manifest.read_text())
    names = set()
    for table in ("dependencies", "pypi-dependencies"):
        names |= {norm(k) for k in data.get(table, {})}
    return names


def py_files(root):
    for path in root.rglob("*.py"):
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            yield path


def local_module_names(files, root):
    """Top-level names that resolve to repo code: every .py stem and every directory that holds .py files."""
    names = {p.stem for p in files}
    for p in files:
        names |= {part for part in p.relative_to(root).parts[:-1]}
    return names


def imports_of(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError as err:
        print(f"WARN  cannot parse {path}: {err}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest", default=None, help="default: <root>/pixi.toml")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    manifest = pathlib.Path(args.manifest) if args.manifest else root / "pixi.toml"

    declared = declared_packages(manifest)
    files = sorted(py_files(root))
    local = local_module_names(files, root)
    stdlib = set(sys.stdlib_module_names)

    missing = defaultdict(set)
    optional = defaultdict(set)
    for path in files:
        for name in set(imports_of(path)):
            if name in stdlib or name in local or name == "__future__":
                continue
            rel = str(path.relative_to(root))
            if name in SOURCE_BUILT:
                optional[name].add(rel)
                continue
            if norm(IMPORT_TO_DIST.get(name, name)) not in declared:
                missing[name].add(rel)

    print(f"scanned {len(files)} files under {root}; manifest {manifest} declares {len(declared)} packages")
    for name, task in sorted(SOURCE_BUILT.items()):
        if name in optional:
            print(f"NOTE  optional source build '{name}' ({len(optional[name])} files): pixi run {task}")
    for name in sorted(missing):
        dist = IMPORT_TO_DIST.get(name, name)
        shown = ", ".join(sorted(missing[name])[:3]) + (" ..." if len(missing[name]) > 3 else "")
        print(f"FAIL  import '{name}' (package '{dist}') not declared in pixi.toml; used by {shown}")
    if missing:
        print("Install hint: add each package to [pypi-dependencies] in pixi.toml with an exact version, then `pixi lock` "
              "(or `pixi add --pypi <pkg>==<version>`). If the import name differs from the package name, extend "
              "IMPORT_TO_DIST in eval3/check_env.py.")
        return 1
    print("PASS  every third-party import is declared in pixi.toml or the standard library")
    return 0


if __name__ == "__main__":
    sys.exit(main())
