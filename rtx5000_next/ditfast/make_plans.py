"""Search and save DiTFastAttn plans for several error thresholds (the calibration stage).

    python ditfast/make_plans.py --size 512 --deltas 0.05 0.1 0.2 --embeds prompts/embeds_calib.pt

Why separate calibration prompts: the plan is tuned on these prompts, so the quality evaluation uses different ones
(prompts/embeds_pixart.pt). If embeds_calib.pt is missing the script says so and does not silently reuse the eval prompts.
Output: results/dfa_plans/pixart_<size>_d<delta>.json and results/dfa_plans/summary.md (technique counts, modeled cost).
Each plan is searched on a bf16 transformer; it is then applied unchanged on top of any quantization.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import time

import attention_patch as ap
import ditfast.plan_search as ps
from common import RESULTS, load_prompt_embeds
from variants import _load_pipe, plan_path


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", type=int, default=512)
    a.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    a.add_argument("--steps", type=int, default=14)
    a.add_argument("--embeds", default="prompts/embeds_calib.pt")
    a.add_argument("--n", type=int, default=6, help="calibration prompts processed together")
    a.add_argument("--window-div", type=int, default=8)
    a.add_argument("--allow-eval-prompts", action="store_true", help="use embeds_pixart.pt if embeds_calib.pt is missing")
    a = a.parse_args()

    emb = pathlib.Path(a.embeds)
    if not emb.exists():
        if not a.allow_eval_prompts:
            sys.exit(f"{emb} missing: run prompts/make_prompt_embeds.py --prompts prompts/prompts_calib.json --out {emb}")
        emb = pathlib.Path("prompts/embeds_pixart.pt")
    items = load_prompt_embeds(emb)[:a.n]
    pipe = _load_pipe(a.size, "bf16")
    ps.enable(pipe, ap.AttnConfig(), use_dfa=True)
    rows = []
    for d in a.deltas:
        t0 = time.time()
        plan = ps.search_plan(pipe, items, a.size, steps=a.steps, delta=d, window_div=a.window_div)
        ps.save_plan(plan, plan_path(a.size, d))
        rows.append((d, plan["counts"], plan["modeled_attention_cost"], time.time() - t0))
        print(f"delta {d}: {plan['counts']}  modeled attention cost {plan['modeled_attention_cost']:.2f}  ({rows[-1][3]:.0f}s)")
    out = RESULTS / "dfa_plans" / f"summary_{a.size}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("| delta | technique counts | modeled attention cost | search s |\n|---|---|---|---|\n" +
                   "\n".join(f"| {d} | {c} | {cost:.2f} | {t:.0f} |" for d, c, cost, t in rows) + "\n")


if __name__ == "__main__":
    main()
