"""Scores the language model's reading of discharge summaries directly against known truth.

Truth comes from the clean summary text (before any noise is added). Run:
  python -m claimsagent.llm_bench llama3.2:3b qwen2.5:3b --seeds 101 --noise abbreviations
Prompt tuning used seed 101 only; report numbers on other seeds.
"""
import argparse
import copy
import json
import random
import re
import statistics
import time

from . import robustness as R
from .llm import get_llm

NOISE = {"clean": lambda i, r: i, "abbreviations": R.abbreviations, "char_noise": R.char_noise, "paraphrase": R.paraphrase,
         "typos": R.ocr_noise}
_COND = {"diabetes": "diabetes", "coronary artery disease": "cardiac"}


def truth_of(text):
    m = re.search(r"Procedure performed: (.+?)\.", text)
    proc = m.group(1).strip() if m else None
    conds = [v for k, v in _COND.items() if re.search(rf"Known case of .*{k}", text, re.I)]
    return proc, conds


def bench(model, seeds, noise, limit=None, verbose=False):
    llm = get_llm(model, cache=True, tasks=("extract",))
    h = llm.health()
    if not h["ok"]:
        raise SystemExit(h["message"])
    items = [it for s in seeds for it in R.held_out(s)]
    rng = random.Random(5)
    rows, times = [], []
    for it in items[:limit]:
        t_proc, t_conds = truth_of(it["claim"]["discharge_summary"])
        noisy = NOISE[noise](copy.deepcopy(it), rng)
        t0 = time.time()
        out = llm.extract(noisy["claim"]["discharge_summary"])
        times.append(time.time() - t0)
        rows.append((t_proc, t_conds, out, noisy["claim"]["discharge_summary"]))
        if verbose and out and ((t_proc and out["procedure_name"] != t_proc) or set(out["conditions"]) != set(t_conds)):
            print("  MISS truth", t_proc, t_conds, "| model", out)
    llm.save_cache()
    with_proc = [r for r in rows if r[0]]
    proc_ok = sum(1 for t, _, o, _ in with_proc if o and o["procedure_name"] and o["procedure_name"].lower() == t.lower())
    proc_wrong = sum(1 for t, _, o, _ in with_proc if o and o["procedure_name"] and o["procedure_name"].lower() != t.lower())
    with_cond = [r for r in rows if r[1]]
    cond_ok = sum(1 for _, c, o, _ in with_cond if o and set(c) <= set(o["conditions"]))
    no_cond = [r for r in rows if not r[1]]
    false_cond = sum(1 for _, _, o, _ in no_cond if o and o["conditions"])
    unusable = sum(1 for r in rows if r[2] is None)
    return dict(model=model, noise=noise, n=len(rows), procedure_found=f"{proc_ok}/{len(with_proc)}",
                procedure_wrong=f"{proc_wrong}/{len(with_proc)}", history_found=f"{cond_ok}/{len(with_cond)}",
                false_history=f"{false_cond}/{len(no_cond)}", unusable_replies=unusable,
                median_s=round(statistics.median(times), 2), mean_s=round(statistics.mean(times), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[101])
    ap.add_argument("--noise", default="abbreviations", choices=list(NOISE))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    for m in a.models:
        print(json.dumps(bench(m, a.seeds, a.noise, a.limit, a.verbose)))
