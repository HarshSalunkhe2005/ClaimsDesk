"""Executor tools. Each tool takes the pipeline state and returns a list of findings.

Finding status:
  pass / info : no consequence
  fail        : hard rule violation  -> REJECT
  query       : evidence missing     -> QUERY (ask the hospital)
  flag        : risk indicator       -> ESCALATE to a human officer
"""
import datetime as dt
import difflib
import re

from . import store

CORE_TOOLS = [
    "verify_documents", "check_policy_status", "check_waiting_periods", "check_exclusions",
    "check_duplicates", "check_filing_window", "check_procedure_match", "bill_integrity", "check_high_value",
]
OPTIONAL_TOOLS = ["tariff_check", "fraud_check"]

TOOL_LABELS = {
    "verify_documents": "Verify mandatory documents",
    "check_policy_status": "Policy in force and initial waiting",
    "check_waiting_periods": "Pre-existing and procedure waiting",
    "check_exclusions": "Permanent exclusions",
    "check_duplicates": "Duplicate claim lookup",
    "check_filing_window": "Filing window",
    "check_procedure_match": "Summary vs claim form match",
    "bill_integrity": "Bill arithmetic integrity",
    "check_high_value": "High-value threshold",
    "tariff_check": "Tariff benchmark comparison",
    "fraud_check": "Fraud signal scoring",
    "compute_payable": "Compute admissible payout",
}


def _date(s):
    return dt.date.fromisoformat(s)


def F(tool, status, message, clause=None, **data):
    return dict(tool=tool, status=status, clause=clause, message=message, data=data)


def nonroom_benchmark(proc):
    return max(proc["benchmark"] - 3000 * proc["los"], 0.45 * proc["benchmark"])


def inflation_ratio(claim):
    """Non-room charges vs the benchmark's non-room portion. Room choice is the patient's and is
    handled by the room-rent clause, so it must not dilute (or inflate) the fraud signal."""
    proc = store.procedures()[claim["procedure_code"]]
    expected = nonroom_benchmark(proc)
    return (claim["claimed_amount"] - claim["bill"]["room"]) / max(expected, 1), expected


def months_covered(member, claim):
    return (_date(claim["admission_date"]) - _date(member["policy_start"])).days / 30.44


# ---------------------------------------------------------------- tools ----
def verify_documents(s):
    c = s["claim"]; rules = store.rules()
    proc = store.procedures()[c["procedure_code"]]
    need = rules["required_docs"] + (rules["surgical_extra_docs"] if proc["kind"] == "surgical" else [])
    missing = [d for d in need if d not in c["documents"]]
    if missing:
        return [F("verify_documents", "query", "Missing documents: " + ", ".join(m.replace("_", " ") for m in missing), "C9", missing=missing)]
    return [F("verify_documents", "pass", f"All {len(need)} mandatory documents present", "C9")]


def check_policy_status(s):
    c, m = s["claim"], s["member"]
    adm = _date(c["admission_date"])
    if _date(m["valid_till"]) < adm:
        return [F("check_policy_status", "fail", f"Policy lapsed on {m['valid_till']}, before admission on {c['admission_date']}", "C1")]
    days = (adm - _date(m["policy_start"])).days
    if days < store.rules()["initial_waiting_days"] and not (c["is_accident"] or s["extracted"]["accident"]):
        return [F("check_policy_status", "fail", f"Admitted {days} days after policy start; illness claims need 30 days", "C2", days=days)]
    return [F("check_policy_status", "pass", f"Policy active; {days} days since inception", "C1", days=days)]


def check_waiting_periods(s):
    c, m, ex = s["claim"], s["member"], s["extracted"]
    proc = store.procedures()[c["procedure_code"]]
    age = months_covered(m, c)
    out = []
    conds = set(m["declared_conditions"]) | set(ex["conditions"])
    tag = proc.get("tag")
    if tag and tag in conds and age < store.rules()["preexisting_months"]:
        undisclosed = tag not in m["declared_conditions"]
        out.append(F("check_waiting_periods", "fail",
                     f"Pre-existing {tag} condition with only {age:.0f} months of cover (24 needed)"
                     + ("; condition not declared at proposal" if undisclosed else ""), "C3", months=round(age, 1), undisclosed=undisclosed))
    if proc["waiting"] and age < proc["waiting"]:
        out.append(F("check_waiting_periods", "fail", f"{proc['name']} has a {proc['waiting']}-month waiting period; member has {age:.0f} months", "C4", months=round(age, 1)))
    return out or [F("check_waiting_periods", "pass", f"No waiting-period conflict ({age:.0f} months of cover)", "C3")]


def check_exclusions(s):
    c, ex = s["claim"], s["extracted"]
    proc = store.procedures()[c["procedure_code"]]
    if proc.get("excluded") and not (proc.get("accident_exempt") and (c["is_accident"] or ex["accident"])):
        return [F("check_exclusions", "fail", f"{proc['name']} is a permanent exclusion", "C5")]
    return [F("check_exclusions", "pass", "Procedure is not excluded", "C5")]


def check_duplicates(s):
    c, m = s["claim"], s["member"]
    adm = _date(c["admission_date"])
    for p in m["prior_claims"]:
        if p["procedure_code"] == c["procedure_code"] and p["status"] == "paid" and abs((_date(p["admission_date"]) - adm).days) <= 5:
            return [F("check_duplicates", "fail", f"Same procedure already paid for admission on {p['admission_date']} (Rs {p['amount']:,})", "C13")]
    return [F("check_duplicates", "pass", "No matching earlier claim", "C13")]


def check_filing_window(s):
    c = s["claim"]
    gap = (_date(c["filed_date"]) - _date(c["discharge_date"])).days
    if gap > store.rules()["filing_window_days"]:
        return [F("check_filing_window", "flag", f"Filed {gap} days after discharge (limit 30); needs manager approval", "C10", gap=gap)]
    return [F("check_filing_window", "pass", f"Filed {gap} days after discharge", "C10")]


def check_procedure_match(s):
    c, ex = s["claim"], s["extracted"]
    proc = store.procedures()
    if len(c["discharge_summary"].strip()) < 40:
        return [F("check_procedure_match", "query", "Discharge summary is missing or unreadable; request a legible copy", "C9")]
    if ex["procedure_code"] is None:
        return [F("check_procedure_match", "info", "Discharge summary names no procedure; falling back to claim form (extraction confidence 0.75)", "C15")]
    if ex["procedure_code"] != c["procedure_code"]:
        return [F("check_procedure_match", "flag",
                  f"Claim form says {proc[c['procedure_code']]['name']} but discharge summary says {proc[ex['procedure_code']]['name']}", "C15")]
    return [F("check_procedure_match", "pass", "Discharge summary matches claim form", "C15")]


def bill_integrity(s):
    c = s["claim"]
    total = sum(c["bill"].values())
    if abs(total - c["claimed_amount"]) > 1:
        return [F("bill_integrity", "flag", f"Itemised bill totals Rs {total:,} but claim form asks Rs {c['claimed_amount']:,}", "C15", total=total)]
    return [F("bill_integrity", "pass", f"Itemised bill reconciles to Rs {total:,}", "C15")]


def check_high_value(s):
    c = s["claim"]
    if c["claimed_amount"] > store.rules()["high_value_threshold"]:
        return [F("check_high_value", "flag", f"Claim of Rs {c['claimed_amount']:,} exceeds Rs 3,00,000; medical officer sign-off required", "C14")]
    return [F("check_high_value", "pass", "Below high-value threshold", "C14")]


def tariff_check(s):
    c = s["claim"]
    ratio, expected = inflation_ratio(c)
    st = "flag" if ratio > 2.0 else "info"
    return [F("tariff_check", st, f"Non-room charges of Rs {c['claimed_amount'] - c['bill']['room']:,} are {ratio:.2f}x the benchmark of Rs {expected:,.0f}", "C15", ratio=round(ratio, 2))]


def fraud_check(s):
    c, m, h = s["claim"], s["member"], s["hospital"]
    proc = store.procedures()[c["procedure_code"]]
    ratio, _ = inflation_ratio(c)
    sig = []
    if ratio > 2.0:
        sig.append((f"non-room charges are {ratio:.1f}x benchmark", 60))
    elif ratio > 1.5:
        sig.append((f"non-room charges are {ratio:.1f}x benchmark", 40))
    if c["los_days"] >= 2 * proc["los"] and c["los_days"] - proc["los"] >= 2:
        sig.append((f"stay of {c['los_days']} days vs typical {proc['los']}", 20))
    if h["watchlist"]:
        sig.append(("hospital on audit watchlist", 25))
    adm = _date(c["admission_date"])
    recent = [p for p in m["prior_claims"] if 0 < (adm - _date(p["admission_date"])).days <= 365]
    if len(recent) >= 3:
        sig.append((f"{len(recent)} claims in the last 12 months", 25))
    if not h["network"] and ratio > 1.3:
        sig.append(("non-network hospital with elevated bill", 10))
    score = min(sum(w for _, w in sig), 100)
    limit = store.rules()["fraud_escalate_score"]
    msg = f"Fraud score {score}/100" + (": " + "; ".join(t for t, _ in sig) if sig else ": no signals")
    return [F("fraud_check", "flag" if score >= limit else "info", msg, "C15", score=score, signals=[t for t, _ in sig])]


def compute_payable(s):
    c, m = s["claim"], s["member"]
    plan = store.plan(m["plan"]); h = s["hospital"]
    proc = store.procedures()[c["procedure_code"]]
    b = c["bill"]
    gross = sum(b.values())
    ded = []
    running = gross
    if b["non_medical"]:
        ded.append(dict(clause="C11", label="Non-medical items", amount=b["non_medical"]))
        running -= b["non_medical"]
    cap = plan["room_cap"]
    if c["room_rate"] > cap:
        room_cut = b["room"] - cap * c["los_days"]
        ratio = cap / c["room_rate"]
        prop_base = b["professional"] + b["medicines"] + b["diagnostics"]
        prop_cut = round(prop_base * (1 - ratio), 0)
        ded.append(dict(clause="C6", label=f"Room rent above Rs {cap:,}/day cap (proportionate deduction)", amount=round(room_cut + prop_cut, 0)))
        running -= room_cut + prop_cut
    if proc["sublimit"] and running > proc["sublimit"]:
        cut = running - proc["sublimit"]
        ded.append(dict(clause="C7", label=f"Above Rs {proc['sublimit']:,} sub-limit for {proc['name']}", amount=round(cut, 0)))
        running -= cut
    if not h["network"]:
        cp = round(running * store.rules()["copay_non_network"], 0)
        ded.append(dict(clause="C8", label="10% non-network co-payment", amount=cp))
        running -= cp
    remaining = plan["sum_insured"] - m["sum_insured_used"]
    if running > remaining:
        ded.append(dict(clause="C12", label="Capped at remaining sum insured", amount=round(running - remaining, 0)))
        running = remaining
    payable = round(running, 0)
    if s.get("inject_fault") and s.get("iteration", 0) == 0:
        payable = round(payable * 1.15, 0)  # simulated executor bug, to show the Critic catching it
    calc = dict(gross=gross, claimed=c["claimed_amount"], deductions=ded, payable=payable, remaining_si=remaining)
    return [F("compute_payable", "info", f"Admissible payout Rs {payable:,.0f} after {len(ded)} deduction(s)", None, calc=calc)]


TOOLS = {f.__name__: f for f in [
    verify_documents, check_policy_status, check_waiting_periods, check_exclusions, check_duplicates,
    check_filing_window, check_procedure_match, bill_integrity, check_high_value, tariff_check, fraud_check, compute_payable,
]}


# ------------------------------------------------------ document parser ----
_KEYS = {
    "diabetes": ["diabetes", "diabetic", "t2dm", "dm2"],
    "cardiac": ["coronary", "cad", "ihd", "cardiac", "heart", "angina"],
    "hypertension": ["hypertension", "hypertensive", "htn"],
}


def _find_procedure(text):
    """Fuzzy-locates a known procedure name anywhere in the summary (tolerates typos and rewording)."""
    body = re.sub(r"(?is)diagnosis:.*?\.(?:\s|$)", " ", text)  # the diagnosis line may echo the claim form
    body = re.sub(r"(?is)history:.*?(?=room category|$)", " ", body)
    tokens = re.findall(r"[a-z0-9()\-]+", body.lower())
    best = (0.0, 0, None, None)
    for code, p in store.procedures().items():
        name = p["name"].lower()
        n = len(name.split())
        for i in range(len(tokens) - n + 1):
            r = difflib.SequenceMatcher(None, " ".join(tokens[i:i + n]), name).ratio()
            if (r, n) > best[:2]:
                best = (r, n, code, p["name"])
    return best if best[0] >= 0.85 else (0.0, 0, None, None)


def _find_conditions(text):
    m = re.search(r"(?is)history:(.*?)(?:room category|$)", text)
    if not m or re.search(r"(?i)no significant|\bnil\b|\bnone\b", m.group(1)):
        return []
    toks = re.findall(r"[a-z0-9]+", m.group(1).lower())
    out = []
    for norm, keys in _KEYS.items():
        if any(difflib.get_close_matches(t, keys, n=1, cutoff=0.9 if len(t) <= 4 else 0.8) for t in toks):
            out.append(norm)
    return out


def parse_summary(text):
    """Fuzzy parser for discharge summaries (the 'document understanding' tool)."""
    score, _, code, name = _find_procedure(text)
    d = re.search(r"(?is)diagnosis:\s*(.+?)\.(?:\s|$)", text)
    return dict(procedure_name=name, procedure_code=code, conditions=_find_conditions(text),
                accident="accident" in text.lower(), diagnosis=d.group(1) if d else None,
                confidence=(1.0 if score >= 0.99 else 0.9) if code else 0.75, source="parser")
