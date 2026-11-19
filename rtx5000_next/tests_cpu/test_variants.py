"""CPU tests for the variant grammar (variants.py) and for every --variant string used in profiling/configs.sh."""
import re

import _boot
from _boot import ROOT, check

import variants as vr


def test_grammar():
    v = vr.parse("module+be=cuda_swizzled+static+concat+trim+xcudnn+kvds2+dfa0.1+licfg0.2-0.8+compile")
    check("parse: all modifiers", (v.base, v.backend, v.static, v.concat, v.trim, v.attn_cross, v.kv_ds, v.dfa_delta, v.interval, v.compile)
          == ("module", "cuda_swizzled", True, True, True, "cudnn", 2, 0.1, (0.2, 0.8), True))
    check("parse: defaults", vr.parse("stock_tk").base == "stock_tk" and vr.parse("bf16").share is True)
    for bad in ("nope", "bf16+nonsense", "module+kvds"):
        try:
            vr.parse(bad)
            ok = False
        except (ValueError, AssertionError):
            ok = True
        check(f"parse rejects '{bad}'", ok)


def test_every_config_is_valid():
    text = (ROOT / "profiling" / "configs.sh").read_text()
    names = re.findall(r'--variant ([^"]+)"', text)
    check("configs.sh defines eval2 variants", len(names) >= 15, str(len(names)))
    for n in names:
        try:
            vr.parse(n)
            ok = True
        except Exception as e:
            ok = False
            print("   ", n, e)
        check(f"configs.sh variant parses: {n}", ok)


if __name__ == "__main__":
    _boot.run(test_grammar, test_every_config_is_valid)
