import json
import os
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "catalog.json").read_text())
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack", ".onnx", ".h5")
TEXT_DIRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3")
KEEP_DIRS = ("transformer", "vae", "scheduler")
VARIANT = re.compile(r"[._-](fp16|bf16|fp32)(?=[._-])")


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def repo_files(repo):
    info = hf_api().model_info(repo, files_metadata=True)
    return info, {s.rfilename: (s.size or 0) for s in info.siblings}


def _pick_weights(names):
    st = [n for n in names if n.endswith(".safetensors")]
    if st:
        plain = [n for n in st if not VARIANT.search(n)]
        return plain or st
    rest = [n for n in names if n.endswith(WEIGHT_EXT)]
    return [n for n in rest if not VARIANT.search(n)] or rest


def select_files(files, with_text=False):
    by_dir = {}
    for name in files:
        by_dir.setdefault(name.split("/")[0] if "/" in name else "", []).append(name)
    keep = []
    for d, names in by_dir.items():
        weights = [n for n in names if n.endswith(WEIGHT_EXT)]
        others = [n for n in names if not n.endswith(WEIGHT_EXT)]
        if d == "":
            keep += others
        elif d in KEEP_DIRS:
            keep += others + _pick_weights(weights)
        elif d in TEXT_DIRS:
            keep += others
            if with_text and d.startswith("text_encoder"):
                keep += _pick_weights(weights)
    return {n: files[n] for n in keep}


def gb(n):
    return n / 1e9


def transformer_params(repo, files, max_files=16):
    api = hf_api()
    total, dtypes = 0, {}
    names = sorted(n for n in files if n.startswith("transformer/") and n.endswith(".safetensors") and not VARIANT.search(n))
    names = names or sorted(n for n in files if n.startswith("transformer/") and n.endswith(".safetensors"))
    for n in names[:max_files]:
        md = api.parse_safetensors_file_metadata(repo_id=repo, filename=n)
        for dt, c in md.parameter_count.items():
            total += c
            dtypes[dt] = dtypes.get(dt, 0) + c
    return total, dtypes
