"""Robustness suite: held-out seeds, realistic noise, and policy boundary pairs.

Truth is fixed by the scenario design or the policy text, never by running the pipeline.
  python -m claimsagent.robustness [out.json]
"""
import copy
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

from . import data_gen, store
from .data_gen import PROC, Factory, _d
from .evaluate import summarize
from .graph import run_claim

HELD_OUT_SEEDS = [101, 102, 103, 104, 105]


# ------------------------------------------------------------- data ----
def held_out(seed):
    f = data_gen.build(seed)
    pre = f"S{seed}-"
    for m in f.members:
        m["member_id"] = pre + m["member_id"]
    store.register_members(f.members)
    out = []
    for item in f.claims:
        c = copy.deepcopy(item["claim"])
        c["member_id"] = pre + c["member_id"]
        c["claim_id"] = pre + c["claim_id"]
        out.append(dict(claim=c, truth=dict(item["truth"])))
    return out


# ------------------------------------------------------- perturbations ----
def _typo(phrase, rng):
    words = phrase.split(" ")
    for i, w in enumerate(words):
        if len(w) >= 6 and rng.random() < 0.6:
            k = rng.randrange(1, len(w) - 2)
            words[i] = w[:k] + w[k + 1] + w[k] + w[k + 2:]
    return " ".join(words)


def ocr_noise(item, rng):
    t = item["claim"]["discharge_summary"]
    t = re.sub(r"(Procedure performed: )(.+?)(\.)", lambda m: m.group(1) + _typo(m.group(2), rng) + m.group(3), t)
    t = re.sub(r"(Known case of )(.+?)( since)", lambda m: m.group(1) + _typo(m.group(2), rng) + m.group(3), t)
    item["claim"]["discharge_summary"] = t
    return item


def paraphrase(item, rng):
    t = item["claim"]["discharge_summary"]
    t = re.sub(r"Procedure performed: (.+?)\.",
               lambda m: rng.choice(["Underwent {}.", "Patient was taken up for {}.", "Operative procedure: {}."]).format(m.group(1)), t)

    def diabetes(m):
        yr = m.group(1)
        return rng.choice([f"Diabetic since {yr} (T2DM).", f"H/o T2DM since {yr}.", f"Long-standing type 2 diabetes, diagnosed {yr}."])

    def cardiac(m):
        yr = m.group(1)
        return rng.choice([f"H/o CAD since {yr}.", f"Coronary artery disease, on treatment since {yr}.", f"Known IHD since {yr}."])

    t = re.sub(r"Known case of Type 2 diabetes since (\d{4})\.", diabetes, t)
    t = re.sub(r"Known case of coronary artery disease since (\d{4})\.", cardiac, t)
    item["claim"]["discharge_summary"] = t
    return item


def _char_noise(phrase, rng, rate=0.12):
    out = []
    for ch in phrase:
        r = rng.random()
        if not ch.isalpha() or r > rate:
            out.append(ch)
        elif r < rate / 3:
            continue  # dropped letter
        elif r < 2 * rate / 3:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))  # substitution
        else:
            out.append(ch + ch)  # doubled letter
    return "".join(out)


def char_noise(item, rng):
    """UNSEEN family (written after the extractor was tuned): random drops, substitutions, doubled letters."""
    t = item["claim"]["discharge_summary"]
    t = re.sub(r"(Procedure performed: )(.+?)(\.)", lambda m: m.group(1) + _char_noise(m.group(2), rng) + m.group(3), t)
    t = re.sub(r"(Known case of )(.+?)( since)", lambda m: m.group(1) + _char_noise(m.group(2), rng) + m.group(3), t)
    item["claim"]["discharge_summary"] = t
    return item


ABBREV = {
    "Total knee replacement": "TKR", "Coronary angioplasty": "PTCA", "Cholecystectomy": "Lap chole",
    "Appendectomy": "Appendicectomy", "Inguinal hernia repair": "Inguinal herniorrhaphy",
    "Cataract surgery": "Phaco with IOL", "Fracture fixation (ORIF)": "ORIF", "Diagnostic coronary angiography": "CAG",
}


def abbreviations(item, rng):
    """UNSEEN family: standard clinical shorthand."""
    t = item["claim"]["discharge_summary"]
    for full, ab in ABBREV.items():
        t = t.replace(f"Procedure performed: {full}.", f"Procedure performed: {ab}.")
    t = t.replace("Known case of Type 2 diabetes", "k/c/o DM-2").replace("Known case of coronary artery disease", "k/c/o CAD")
    item["claim"]["discharge_summary"] = t
    return item


def format_noise(item, rng):
    item["claim"]["discharge_summary"] = item["claim"]["discharge_summary"].lower().replace(". ", ".\n  ")
    return item


def jitter(item, rng):
    c = item["claim"]
    gap = c["claimed_amount"] - sum(c["bill"].values())  # keep any tampering gap intact
    c["bill"] = {k: int(round(v * rng.uniform(0.97, 1.03), -1)) for k, v in c["bill"].items()}
    c["claimed_amount"] = sum(c["bill"].values()) + gap
    return item


def drop_summary(item, rng):
    if item["truth"]["decision"] not in ("APPROVE", "PARTIAL"):
        return None
    item["claim"]["discharge_summary"] = ""
    item["truth"]["decision"] = "QUERY"  # spec: an unreadable mandatory document must be re-requested
    return item


def combined(item, rng):
    return ocr_noise(paraphrase(jitter(item, rng), rng), rng)


CONDITIONS = [
    ("clean (held-out seeds)", lambda i, r: i),
    ("OCR-style typos", ocr_noise),
    ("paraphrased wording", paraphrase),
    ("case and whitespace", format_noise),
    ("+-3% bill jitter", jitter),
    ("summary unreadable", drop_summary),
    ("combined noise", combined),
    ("UNSEEN: random character noise", char_noise),
    ("UNSEEN: clinical abbreviations", abbreviations),
]
UNSEEN = {"UNSEEN: random character noise", "UNSEEN: clinical abbreviations"}


# ------------------------------------------------------------ boundaries ----
def _set_total(claim, total):
    claim["bill"]["professional"] += total - sum(claim["bill"].values())
    claim["claimed_amount"] = total


def _ratio_total(code, rate, r):
    """Total bill whose non-room charges are exactly r x the benchmark's non-room portion."""
    p = PROC[code]
    return int(rate * p["los"] + r * (p["benchmark"] - 3000 * p["los"]))


def boundary_cases():
    f = Factory(seed=500)
    f.n = 5000
    adm = data_gen.AS_OF - dt.timedelta(days=100)
    out = []

    def mk(name, truth, code="P101", plan="GOLD", months=None, days=None, rate=2000, mult=1.0, hospital=None,
           history="No significant past history.", prior=None, total=None, filed_gap=None):
        m = f.member(adm, months=months if months is not None else 60, days=days, plan=plan, prior=prior)
        h = hospital or f.hosp()
        p = PROC[code]
        f.add(name, truth, m, h, p, adm, p["los"], rate, "General ward", mult=mult, history=history, filed_gap=filed_gap)
        item = f.claims[-1]
        if total:
            _set_total(item["claim"], total)
        out.append(item)

    mk("initial waiting 29 days", "REJECT", "P107", days=29)
    mk("initial waiting 30 days", "APPROVE", "P107", days=30, mult=0.95)
    mk("12-month waiting at 11.8 months", "REJECT", "P102", months=11.8)
    mk("12-month waiting at 12.2 months", "APPROVE", "P102", months=12.2)
    mk("24-month waiting at 23.8 months", "REJECT", "P104", months=23.8)
    mk("24-month waiting at 24.2 months", "APPROVE", "P104", months=24.2)
    hist = "Known case of Type 2 diabetes since 2018."
    mk("pre-existing at 23.8 months", "REJECT", "P112", months=23.8, history=hist)
    mk("pre-existing at 24.2 months", "APPROVE", "P112", months=24.2, history=hist, mult=0.9)
    mk("filed 30 days after discharge", "APPROVE", filed_gap=30)
    mk("filed 31 days after discharge", "ESCALATE", filed_gap=31)
    mk("claim Rs 2,99,990", "APPROVE", "P104", total=299990)
    mk("claim Rs 3,00,010", "ESCALATE", "P104", total=300010)
    mk("non-room charges at 1.49x benchmark", "PARTIAL", "P101", rate=3000, total=_ratio_total("P101", 3000, 1.49))
    mk("non-room charges at 2.01x benchmark", "ESCALATE", "P101", rate=3000, total=_ratio_total("P101", 3000, 2.01))
    mk("1.49x benchmark, watchlist hospital", "PARTIAL", "P101", rate=3000, total=_ratio_total("P101", 3000, 1.49), hospital=f.hosp(watch=True))
    mk("1.51x benchmark, watchlist hospital", "ESCALATE", "P101", rate=3000, total=_ratio_total("P101", 3000, 1.51), hospital=f.hosp(watch=True))
    mk("2.2x non-room in a Rs 8,000/day room", "ESCALATE", "P107", rate=8000, total=_ratio_total("P107", 8000, 2.2))
    mk("room rent exactly at cap", "APPROVE", "P101", plan="SILVER", rate=5000)
    mk("room rent Rs 100 over cap", "PARTIAL", "P101", plan="SILVER", rate=5100)
    mk("claim exactly at sub-limit", "APPROVE", "P101", rate=3000, total=75000)
    mk("claim Rs 10 over sub-limit", "PARTIAL", "P101", rate=3000, total=75010)

    def prior(k):
        return [dict(procedure_code="P101", admission_date=_d(adm + dt.timedelta(days=k)), amount=50000, status="paid")]

    mk("duplicate 5 days apart", "REJECT", "P101", prior=prior(5))
    mk("duplicate 6 days apart", "APPROVE", "P101", prior=prior(6))
    store.register_members(f.members)
    return out


# ------------------------------------------------------------------ run ----
def _run(items, llm=None):
    rows = []
    for it in items:
        o = run_claim(it["claim"], llm)
        d = o["decision"]
        rows.append(dict(claim_id=it["claim"]["claim_id"], scenario=it["truth"]["scenario"], truth=it["truth"]["decision"], pred=d["decision"],
                         payable=d["payable"], confidence=d["confidence"], iterations=d["iterations"], ms=o["elapsed_ms"],
                         tools=len(o["done"]), skipped=len(o["skipped"])))
    return rows


def run_all(llm=None):
    base = [it for s in HELD_OUT_SEEDS for it in held_out(s)]
    report = dict(held_out_seeds=HELD_OUT_SEEDS, conditions=[], boundaries=[])
    for name, fn in CONDITIONS:
        rng = random.Random(2026)
        items = []
        for it in base:
            res = fn(copy.deepcopy(it), rng)
            if res is not None:
                items.append(res)
        rows = _run(items, llm)
        s = summarize(rows)
        by_sc = {}
        for r in rows:
            if r["truth"] != r["pred"]:
                by_sc.setdefault(r["scenario"], []).append(f"{r['truth']}->{r['pred']}")
        report["conditions"].append(dict(
            name=name, unseen=name in UNSEEN, claims=s["claims"], accuracy=s["accuracy"], false_payment_rate=s["false_payment_rate"],
            false_payments=s["false_payments"], risky_claims=s["risky_claims"], wrongful_hold_rate=s["wrongful_hold_rate"],
            wrongful_holds=s["wrongful_holds"], payable_claims=s["payable_claims"], critic_replans=s["critic_replans"],
            failures={k: v for k, v in sorted(by_sc.items())}))
    b_rows = _run(boundary_cases(), llm)
    for r in b_rows:
        report["boundaries"].append(dict(case=r["scenario"], expected=r["truth"], got=r["pred"], ok=r["truth"] == r["pred"]))
    report["boundary_pass"] = sum(b["ok"] for b in report["boundaries"])
    report["boundary_total"] = len(report["boundaries"])
    return report


def show(rep):
    print(f"{'condition':26s} {'n':>4s} {'acc':>7s} {'false-pay':>10s} {'wrong-hold':>11s}")
    for c in rep["conditions"]:
        print(f"{c['name']:26s} {c['claims']:4d} {c['accuracy'] * 100:6.1f}% {c['false_payments']:>4d}/{c['risky_claims']:<5d} {c['wrongful_holds']:>4d}/{c['payable_claims']:<5d}")
        for k, v in c["failures"].items():
            print(f"      {k}: {len(v)} x {sorted(set(v))}")
    print(f"\nBoundary pairs: {rep['boundary_pass']}/{rep['boundary_total']}")
    for b in rep["boundaries"]:
        if not b["ok"]:
            print(f"   FAIL {b['case']}: expected {b['expected']} got {b['got']}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else store.DATA / "robustness_report.json"
    rep = run_all()
    out.write_text(json.dumps(rep, indent=2))
    show(rep)
    print("saved", out)
