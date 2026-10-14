"""Multi-agent orchestration: Planner -> Extractor -> Executor -> Critic -> (re-plan loop) -> Decision.

Built on LangGraph. The Critic can send the claim back to the Planner, so the flow is a
real feedback loop and not a straight pipeline.
"""
import datetime as dt
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from . import store
from .llm import MockLLM
from .tools import (CORE_TOOLS, OPTIONAL_TOOLS, TOOL_LABELS, TOOLS, inflation_ratio, parse_summary)

MAX_ITERATIONS = 2


class State(TypedDict, total=False):
    claim: dict
    member: dict
    hospital: dict
    extracted: dict
    plan: list
    done: list
    skipped: list
    findings: Annotated[list, operator.add]
    calc: dict
    critic: dict
    decision: dict
    iteration: int
    inject_fault: bool
    trace: Annotated[list, operator.add]


def ev(agent, kind, message, iteration=0, **data):
    return dict(t=time.time(), agent=agent, kind=kind, message=message, iteration=iteration, data=data)


# ---------------------------------------------------------------- build ----
def build_graph(llm=None):
    llm = llm or MockLLM()

    def planner(s):
        it = s.get("iteration", 0)
        c = s["claim"]
        if it == 0:
            m, h = store.member(c["member_id"]), store.hospital(c["hospital_id"])
            proc = store.procedures()[c["procedure_code"]]
            reasons = []
            if c["claimed_amount"] >= 100000:
                reasons.append("amount of Rs 1,00,000 or more")
            if not h["network"]:
                reasons.append("non-network hospital")
            if h["watchlist"]:
                reasons.append("hospital on audit watchlist")
            adm = dt.date.fromisoformat(c["admission_date"])
            recent = [p for p in m["prior_claims"] if 0 < (adm - dt.date.fromisoformat(p["admission_date"])).days <= 365]
            if len(recent) >= 2:
                reasons.append("repeat claimant")
            optional = OPTIONAL_TOOLS if reasons else []
            steps = CORE_TOOLS + optional + ["compute_payable"]
            rationale = llm.plan_rationale(dict(kind=proc["kind"], reasons=reasons, optional=optional))
            msg = f"{len(steps)}-step plan ({'deep' if optional else 'lean'} review). {rationale}"
            return dict(member=m, hospital=h, plan=steps, done=[], skipped=[], iteration=0,
                        trace=[ev("Planner", "plan", msg, 0, steps=steps, rationale=rationale, reasons=reasons)])
        actions = s["critic"]["actions"]
        notes = "; ".join(s["critic"]["notes"])
        return dict(plan=actions, trace=[ev("Planner", "replan", f"Critic reported: {notes}. Re-planning (round {it}) with steps: {', '.join(actions)}.", it, steps=actions)])

    def extractor(s):
        c = s["claim"]
        ex = llm.extract(c["discharge_summary"]) if hasattr(llm, "extract") else None
        base = parse_summary(c["discharge_summary"])
        src = "parser"
        if ex and isinstance(ex, dict):
            base["conditions"] = sorted(set(base["conditions"]) | set(ex.get("conditions") or []))
            base["accident"] = base["accident"] or bool(ex.get("is_accident"))
            src = llm.name
        base["source"] = src
        found = base["procedure_name"] or "not stated"
        msg = f"Parsed discharge summary ({src}): procedure '{found}', conditions {base['conditions'] or 'none'}, accident {base['accident']}, confidence {base['confidence']:.2f}."
        return dict(extracted=base, trace=[ev("Extractor", "extract", msg, 0, extracted=base)])

    def executor(s):
        it = s.get("iteration", 0)
        findings, events, done, skipped = [], [], list(s.get("done", [])), []
        state = dict(s)
        state["iteration"] = it
        calc = s.get("calc")
        stop = False
        for name in s["plan"]:
            if stop:
                skipped.append(name)
                events.append(ev("Executor", "skip", f"Skipped {TOOL_LABELS[name]}: claim already ineligible", it, tool=name))
                continue
            out = TOOLS[name](state)
            for f in out:
                f["iteration"] = it
                if f["data"].get("calc"):
                    calc = f["data"]["calc"]; state["calc"] = calc
            findings += out
            done.append(name)
            worst = max(out, key=lambda f: ["pass", "info", "query", "flag", "fail"].index(f["status"]))
            events.append(ev("Executor", "tool", f"{TOOL_LABELS[name]}: {worst['message']}", it, tool=name, status=worst["status"], clause=worst["clause"]))
            if worst["status"] == "fail" and name != "compute_payable":
                stop = True
        return dict(findings=findings, done=done, skipped=s.get("skipped", []) + skipped, calc=calc, trace=events)

    def critic(s):
        it = s.get("iteration", 0)
        c, ex, calc = s["claim"], s["extracted"], s.get("calc")
        clauses = store.clauses()
        checks, actions, notes = [], [], []

        def chk(cid, label, ok, detail, severity="critical"):
            checks.append(dict(id=cid, label=label, status="pass" if ok else ("fail" if severity == "critical" else "warn"), detail=detail))
            return ok

        if calc:
            expect = calc["gross"] - sum(d["amount"] for d in calc["deductions"])
            ok = abs(expect - calc["payable"]) <= 1
            if not chk("arithmetic", "Payout reconciles with deductions", ok,
                       f"gross {calc['gross']:,} - deductions {sum(d['amount'] for d in calc['deductions']):,.0f} = {expect:,.0f}; payable {calc['payable']:,.0f}"):
                actions.append("compute_payable"); notes.append("payout did not reconcile with deductions")
            lim = min(c["claimed_amount"], calc["remaining_si"])
            if not chk("bounds", "Payout within claimed amount and remaining sum insured", 0 <= calc["payable"] <= lim + 1, f"payable {calc['payable']:,.0f} vs limit {lim:,.0f}"):
                if "compute_payable" not in actions:
                    actions.append("compute_payable")
                notes.append("payout exceeded the admissible limit")
        else:
            chk("arithmetic", "Payout reconciles with deductions", True, "Not applicable: claim ended before payout calculation")

        bad = [f for f in s["findings"] if f["status"] in ("fail", "flag", "query") and f["clause"] not in clauses]
        bad += [d for d in (calc["deductions"] if calc else []) if d["clause"] not in clauses]
        chk("citations", "Every adverse finding cites a valid policy clause", not bad, "all clauses resolved" if not bad else f"{len(bad)} uncited")

        low = ex["procedure_code"] is None
        checks.append(dict(id="extraction", label="Extracted facts agree with claim form", status="warn" if low else "pass",
                           detail="Summary lacks a procedure; used claim form value (confidence 0.75)" if low else "procedure and history consistent"))

        ratio, _ = inflation_ratio(c)
        covered = "fraud_check" in s["done"]
        if not chk("risk", "Fraud screening matches the claim's risk", not (ratio > 1.5 and not covered),
                   f"non-room charges are {ratio:.2f}x benchmark; fraud screening {'was run' if covered else 'was skipped by the lean plan'}"):
            actions += ["tariff_check", "fraud_check"]; notes.append(f"non-room charges are {ratio:.1f}x benchmark but fraud screening was skipped")

        fails = [k for k in checks if k["status"] == "fail"]
        warns = [k for k in checks if k["status"] == "warn"]
        conf = max(0.0, round(0.98 - 0.3 * len(fails) - 0.05 * len(warns), 2))
        if not fails:
            verdict = "pass"
        elif it < MAX_ITERATIONS:
            verdict = "replan"
        else:
            verdict = "escalate"
        report = dict(checks=checks, confidence=conf, verdict=verdict, actions=actions, notes=notes)
        events = [ev("Critic", "check", f"{k['label']}: {k['status'].upper()} ({k['detail']})", it, status=k["status"]) for k in checks]
        events.append(ev("Critic", "verdict", f"Verdict: {verdict.upper()} (confidence {conf:.2f})" + (f". Sending back to Planner: {'; '.join(notes)}." if verdict == "replan" else ""), it, verdict=verdict, confidence=conf))
        out = dict(critic=report, trace=events)
        if verdict == "replan":
            out["iteration"] = it + 1
        return out

    def decision(s):
        it = s.get("iteration", 0)
        c, m = s["claim"], s["member"]
        f = s["findings"]
        calc = s.get("calc")
        fails = [x for x in f if x["status"] == "fail"]
        queries = [x for x in f if x["status"] == "query"]
        flags = [x for x in f if x["status"] == "flag"]
        crit = s["critic"]
        payable = 0.0
        if fails:
            d, reasons = "REJECT", [x["message"] for x in fails]
        elif queries:
            d, reasons = "QUERY", [x["message"] for x in queries]
        elif flags or crit["verdict"] == "escalate":
            d = "ESCALATE"
            reasons = [x["message"] for x in flags] or ["Critic could not validate the calculation after re-planning"]
        elif calc and calc["deductions"]:
            d, payable, reasons = "PARTIAL", calc["payable"], [x["label"] for x in calc["deductions"]]
        else:
            d, payable, reasons = "APPROVE", calc["payable"], ["All checks passed"]
        cited = sorted({x["clause"] for x in fails + queries + flags if x["clause"]} | {x["clause"] for x in (calc["deductions"] if calc and d == "PARTIAL" else [])},
                       key=lambda k: int(k[1:]))
        fraud = next((x["data"].get("score") for x in reversed(f) if x["tool"] == "fraud_check"), None)
        ctx = dict(decision=d, member_name=m["name"], claim_id=c["claim_id"], payable=payable, claimed=c["claimed_amount"],
                   deductions=(calc["deductions"] if calc and d == "PARTIAL" else []), reasons=reasons)
        letter = llm.decision_letter(ctx)
        dec = dict(decision=d, payable=payable, claimed=c["claimed_amount"], reasons=reasons, clauses=cited,
                   deductions=(calc["deductions"] if calc and d in ("PARTIAL", "APPROVE") else []),
                   confidence=crit["confidence"], fraud_score=fraud, iterations=it, letter=letter,
                   needs_human=d in ("ESCALATE", "QUERY"))
        msg = f"{d}" + (f" Rs {payable:,.0f} of Rs {c['claimed_amount']:,}" if d in ("APPROVE", "PARTIAL") else "") + f". {reasons[0]}"
        return dict(decision=dec, trace=[ev("Decision", "decision", msg, it, decision=d)])

    def route_after_planner(s):
        return "executor" if s.get("iteration", 0) > 0 else "extractor"

    def route_after_critic(s):
        return "planner" if s["critic"]["verdict"] == "replan" else "decision"

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("extractor", extractor), ("executor", executor), ("critic", critic), ("decision", decision)]:
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner, {"extractor": "extractor", "executor": "executor"})
    g.add_edge("extractor", "executor")
    g.add_edge("executor", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "decision": "decision"})
    g.add_edge("decision", END)
    return g.compile()


def _init(claim, inject_fault):
    return dict(claim=claim, iteration=0, inject_fault=inject_fault, findings=[], trace=[])


def run_claim(claim, llm=None, inject_fault=False):
    t0 = time.perf_counter()
    out = build_graph(llm).invoke(_init(claim, inject_fault))
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def stream_claim(claim, llm=None, inject_fault=False):
    """Yields (node_name, update) as each agent finishes. Use for live UI animation."""
    for chunk in build_graph(llm).stream(_init(claim, inject_fault), stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update


GRAPH_EDGES = [("START", "planner"), ("planner", "extractor"), ("extractor", "executor"), ("executor", "critic"),
               ("critic", "decision"), ("critic", "planner"), ("planner", "executor"), ("decision", "END")]
