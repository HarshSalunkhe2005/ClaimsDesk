"""Scores the multi-agent pipeline against the labelled synthetic claims."""
import json
from collections import Counter, defaultdict

from . import store
from .graph import run_claim
from .llm import get_llm

LABELS = ["APPROVE", "PARTIAL", "REJECT", "ESCALATE", "QUERY"]
MANUAL_MIN_PER_CLAIM = 25  # assumption: typical manual first-pass adjudication time, not a measured figure


def evaluate(llm=None, limit=None):
    # Accuracy tests score decisions, which never depend on the model's prose, so they skip the rationale and
    # letter calls and run only the extraction task (the one that can change a decision).
    llm = llm or get_llm(cache=True, tasks=("extract",))
    rows = []
    for item in store.all_claims()[:limit]:
        out = run_claim(item["claim"], llm)
        d = out["decision"]
        rows.append(dict(claim_id=item["claim"]["claim_id"], scenario=item["truth"]["scenario"], truth=item["truth"]["decision"],
                         pred=d["decision"], payable=d["payable"], confidence=d["confidence"], iterations=d["iterations"],
                         ms=out["elapsed_ms"], tools=len(out["done"]), skipped=len(out["skipped"])))
    return rows, summarize(rows)


def summarize(rows):
    n = len(rows)
    correct = sum(r["truth"] == r["pred"] for r in rows)
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    for r in rows:
        cm[r["truth"]][r["pred"]] += 1
    per = {}
    for l in LABELS:
        tp = cm[l][l]; fn = sum(cm[l].values()) - tp; fp = sum(cm[t][l] for t in LABELS) - tp
        per[l] = dict(support=tp + fn, precision=round(tp / (tp + fp), 3) if tp + fp else None, recall=round(tp / (tp + fn), 3) if tp + fn else None)
    risky = [r for r in rows if r["truth"] in ("REJECT", "ESCALATE", "QUERY")]
    false_pay = [r for r in risky if r["pred"] in ("APPROVE", "PARTIAL")]
    safe = [r for r in rows if r["truth"] in ("APPROVE", "PARTIAL")]
    over_esc = [r for r in safe if r["pred"] in ("ESCALATE", "QUERY", "REJECT")]
    avg_ms = sum(r["ms"] for r in rows) / n
    return dict(
        claims=n, accuracy=round(correct / n, 3), correct=correct,
        false_payment_rate=round(len(false_pay) / max(len(risky), 1), 3), false_payments=len(false_pay), risky_claims=len(risky),
        wrongful_hold_rate=round(len(over_esc) / max(len(safe), 1), 3), wrongful_holds=len(over_esc), payable_claims=len(safe),
        straight_through_rate=round(sum(r["pred"] in ("APPROVE", "PARTIAL", "REJECT") for r in rows) / n, 3),
        critic_replans=sum(r["iterations"] > 0 for r in rows), avg_ms=round(avg_ms, 1),
        manual_minutes_baseline=MANUAL_MIN_PER_CLAIM, confusion=cm, per_class=per,
        misses=[r for r in rows if r["truth"] != r["pred"]],
    )


if __name__ == "__main__":
    llm_ = get_llm(cache=True, tasks=("extract",))
    rows, summ = evaluate(llm_)
    summ["model"] = llm_.model
    (store.DATA / "eval_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "misses"}, indent=2))
    print("MISSES:")
    for m in summ["misses"]:
        print(" ", m["claim_id"], m["scenario"], "truth", m["truth"], "pred", m["pred"])
