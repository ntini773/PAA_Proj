"""Encode prompts with the PixArt-Sigma T5-XXL text encoder (bf16, GPU) and save them once.

    python prompts/make_prompt_embeds.py
    python prompts/make_prompt_embeds.py --t5 ~/t5_enc_bf16 --prompts prompts/prompts.json --out prompts/embeds_pixart.pt

Output {items: [{prompt, pe, pm, ne, nm}, ...]}: pe [1,300,4096] and pm [1,300] are the prompt, ne and nm the empty negative prompt.
Same max length (300) as PixArt-Sigma. The file is about 15 MB; copy it back to keep it.
"""
import argparse
import json
import pathlib

import torch
from transformers import T5EncoderModel, T5Tokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5", default=None, help="local T5 folder; default: text_encoder of the PixArt-Sigma repo")
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--out", default=str(HERE / "embeds_pixart.pt"))
    ap.add_argument("--max-len", type=int, default=300)
    a = ap.parse_args()

    src = a.t5 or REPO
    sub = (lambda s: {}) if a.t5 else (lambda s: dict(subfolder=s))
    try:
        import sentencepiece  # noqa: F401  (T5Tokenizer needs it)
    except ImportError:
        raise SystemExit("sentencepiece missing: pip install -r requirements_next.txt")
    tok = T5Tokenizer.from_pretrained(src, **sub("tokenizer"))
    te = T5EncoderModel.from_pretrained(src, torch_dtype=torch.bfloat16, **sub("text_encoder")).cuda().eval()

    def enc(text):
        ids = tok(text, padding="max_length", max_length=a.max_len, truncation=True, return_tensors="pt")
        with torch.no_grad():
            e = te(input_ids=ids.input_ids.cuda(), attention_mask=ids.attention_mask.cuda())[0]
        return e.cpu().to(torch.bfloat16), ids.attention_mask

    ne, nm = enc("")
    items = []
    for p in json.loads(pathlib.Path(a.prompts).read_text()):
        pe, pm = enc(p)
        items.append(dict(prompt=p, pe=pe, pm=pm, ne=ne, nm=nm))
        print(f"{int(pm.sum()):3d} tokens  {p[:70]}")
    torch.save({"items": items}, a.out)
    print("saved", a.out, len(items), "prompts")


if __name__ == "__main__":
    main()
