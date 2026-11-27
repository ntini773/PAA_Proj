"""Check that every variant name in a run parses (bug 28: 'stock_compile' killed a 15 min stage after two variants).

    python eval3/validate_variants.py stock+compile module+be=cuda_swizzled+concat+trim+xcudnn+compile bf16@seedshift
No GPU needed. Exit code 1 if any name is rejected. The '@seedshift' suffix belongs to eval3/quality_grid.py and is stripped here.
Also checks that files a variant needs exist (static scales, DiTFastAttn plans) and warns if not.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import variants as vr  # noqa: E402


def main(names):  # names may also be a config list: pass variant strings, not config names
    bad = 0
    for n in names:
        base = n.replace("@seedshift", "")
        try:
            v = vr.parse(base)
        except (AssertionError, ValueError) as e:
            print(f"BAD   {n}: {e}")
            bad += 1
            continue
        notes = []
        for size in (256, 512, 1024):
            if v.static and not vr.static_path(size).exists():
                notes.append(f"no static scales for {size}")
            if v.dfa_delta is not None and not vr.plan_path(size, v.dfa_delta).exists():
                notes.append(f"no DiTFastAttn plan for {size}")
        print(f"ok    {n}" + (f"   (note: {', '.join(sorted(set(notes)))})" if notes else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
