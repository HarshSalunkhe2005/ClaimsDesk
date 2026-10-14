"""Plain-English narration of a finished claim run.

The pipeline's trace is accurate but technical. `explain(state)` turns it into a short story a first-time
viewer can follow: what each agent did, what went wrong (if anything), and what happens next.
"""
from . import store

AGENT_BLURB = {
    "Planner": "Decides how closely to check this claim.",
    "Extractor": "Reads the hospital's discharge summary.",
    "Executor": "Runs the checks and works out what is payable.",
    "Critic": "Double-checks the work and can send it back.",
    "Decision": "Picks the outcome and writes to the member.",
}

OUTCOME = {
    "APPROVE": ("Approved", "Paid in full. No human needed."),
    "PARTIAL": ("Approved in part", "Paid, minus the deductions listed. No human needed."),
    "REJECT": ("Rejected", "Not payable under a policy rule. The member is told which one."),
    "QUERY": ("On hold: documents needed", "The hospital is asked to send what is missing."),
    "ESCALATE": ("Sent to a human officer", "Something looked risky or unusual, so a person decides."),
}

METRIC_HELP = {
    "accuracy": "Share of claims where the system's outcome matched the correct one.",
    "false_payment_rate": "Of the claims that should NOT have been paid, how many were paid anyway. The costly mistake.",
    "wrongful_hold_rate": "Of the claims that were fine to pay, how many were needlessly held for a human.",
    "straight_through_rate": "Share of claims settled with no human involved (approved, part-paid or rejected).",
    "critic_replans": "Claims where the Critic found a problem and sent the work back for another round.",
}

TOOL_PLAIN = {
    "verify_documents": "all required documents are present",
    "check_policy_status": "the policy was active and past its first 30 days",
    "check_waiting_periods": "no waiting period was in the way",
    "check_exclusions": "the treatment is not excluded",
    "check_duplicates": "it has not been paid before",
    "check_filing_window": "it was filed in time",
    "check_procedure_match": "the discharge summary matches the claim form",
    "bill_integrity": "the bill adds up",
    "check_high_value": "the amount is below the senior-review limit",
    "tariff_check": "the price is in line with typical costs",
    "fraud_check": "no fraud warning signs",
}


def _clause(cid):
    c = store.clauses().get(cid)
    return f"{c['title']} ({cid})" if c else cid


def plain_finding(f):
    """One adverse finding as a sentence plus the rule behind it."""
    rule = f" Rule: {_clause(f['clause'])}." if f.get("clause") else ""
    return f["message"].rstrip(".") + "." + rule


def explain(state):
    claim, trace, findings = state["claim"], state["trace"], state["findings"]
    dec, critic, extracted = state["decision"], state["critic"], state["extracted"]
    label, next_step = OUTCOME[dec["decision"]]

    # ---- Planner
    plan_ev = next(e for e in trace if e["agent"] == "Planner" and e["kind"] == "plan")
    reasons = plan_ev["data"].get("reasons", [])
    if reasons:
        planner = "Chose a thorough review because " + " and ".join(reasons) + ". That adds price and fraud checks."
    else:
        planner = "Chose a quick review: network hospital, modest amount, nothing unusual. Price and fraud checks were left out to save effort."

    # ---- Extractor
    proc = extracted.get("procedure_name")
    if proc:
        ex = f"Read the discharge summary and found the treatment: {proc}."
    else:
        ex = "The discharge summary did not name the treatment clearly, so the claim form was used instead."
    if extracted.get("conditions"):
        ex += " It also mentions a long-standing condition: " + ", ".join(extracted["conditions"]) + "."

    # ---- Executor
    adverse = [f for f in findings if f["status"] in ("fail", "flag", "query")]
    ran = [f for f in findings if f["tool"] in TOOL_PLAIN]
    passed = [f for f in ran if f["status"] in ("pass", "info")]
    if adverse:
        exe = (f"Ran {len(ran)} checks and found {'a problem' if len(adverse) == 1 else str(len(adverse)) + ' problems'}: "
               + " ".join(plain_finding(f) for f in adverse))
    else:
        exe = f"Ran {len(ran)} checks and everything passed: " + ", ".join(TOOL_PLAIN[f["tool"]] for f in passed[:4]) + ", and more."
    if state.get("skipped"):
        exe += " It stopped early, so the remaining checks were skipped."

    # ---- Critic
    replans = [e for e in trace if e["agent"] == "Planner" and e["kind"] == "replan"]
    if replans:
        notes = []
        for e in replans:
            msg = e["message"]
            notes.append(msg.split("Critic reported: ")[-1].split(". Re-planning")[0])
        cr = ("Double-checked the work and spotted a problem: " + "; ".join(notes) +
              ". It sent the claim back to the Planner for another round, then re-checked. ")
        cr += "Second look passed." if critic["verdict"] == "pass" else "It still could not fully validate, so a human will review."
    elif critic["verdict"] == "pass":
        cr = "Double-checked the maths, the limits, the rule references and whether the right checks were run. All fine."
    else:
        cr = "Double-checked the work and could not fully validate it, so a human will review."
    cr += f" Confidence {critic['confidence']:.0%}."

    # ---- Decision
    why = dec["reasons"][0] if dec["reasons"] else ""
    if dec["decision"] in ("APPROVE", "PARTIAL"):
        pay = f"Rs {dec['payable']:,.0f} of the Rs {dec['claimed']:,} claimed"
        dn = f"{label}: {pay}. {next_step}"
        if dec["decision"] == "PARTIAL":
            cuts = "; ".join(f"{d['label']}: Rs {d['amount']:,.0f} (rule: {clause_name(d['clause'])})" for d in dec["deductions"])
            dn = f"{label}: {pay}. Deducted: {cuts}. No human needed."
    else:
        dn = f"{label}. {next_step}"

    steps = [
        dict(agent="Planner", blurb=AGENT_BLURB["Planner"], text=planner),
        dict(agent="Extractor", blurb=AGENT_BLURB["Extractor"], text=ex),
        dict(agent="Executor", blurb=AGENT_BLURB["Executor"], text=exe),
        dict(agent="Critic", blurb=AGENT_BLURB["Critic"], text=cr, replanned=bool(replans)),
        dict(agent="Decision", blurb=AGENT_BLURB["Decision"], text=dn),
    ]
    headline = f"{label}" + (f": Rs {dec['payable']:,.0f}" if dec["decision"] in ("APPROVE", "PARTIAL") else "")
    reason_plain = [plain_finding(f) for f in adverse] or ([d["label"] for d in dec["deductions"]] if dec["decision"] == "PARTIAL" else ["Every check passed."])
    return dict(
        headline=headline, next_step=next_step, outcome_label=label, steps=steps, why=reason_plain,
        clauses=[dict(id=c, title=store.clauses()[c]["title"], text=store.clauses()[c]["text"]) for c in dec["clauses"]],
        replanned=bool(replans), agent_blurbs=AGENT_BLURB, metric_help=METRIC_HELP,
    )


def clause_name(cid):
    c = store.clauses().get(cid)
    return c["title"] if c else cid
