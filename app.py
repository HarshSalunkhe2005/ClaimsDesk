"""Claims desk: watch the five agents adjudicate a claim. Run with `streamlit run app.py`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from claimsagent.explain import explain
from claimsagent import graph, store
from claimsagent.llm import LLMUnavailable, get_llm
from ui import render as R

ROOT = Path(__file__).parent
st.set_page_config(page_title="Claims Desk", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(ROOT / 'ui' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

DEMOS = [
    ("CLM-26-0003", False, "Clean approve", "Everything checks out"),
    ("CLM-26-0001", False, "Partial payout", "Room-rent proportionate deduction"),
    ("CLM-26-0004", False, "Reject", "Initial waiting period"),
    ("CLM-26-0007", False, "Reject", "Pre-existing condition"),
    ("CLM-26-0019", False, "Second-look loop", "Quick check misses a 2× bill, Critic sends it back"),
    ("CLM-26-0008", False, "Bill tampering", "Also loops back"),
    ("CLM-26-0006", False, "Procedure mismatch", "Escalate"),
    ("CLM-26-0002", False, "Missing documents", "Query the hospital"),
    ("CLM-26-0003", True, "Calculation error (simulated)", "Critic catches a wrong payout"),
]

ss = st.session_state
ss.setdefault("sel", "CLM-26-0019")
ss.setdefault("fault", False)
ss.setdefault("go", False)
ss.setdefault("res", None)
ss.setdefault("df_last", [])
ss.setdefault("intro", True)


@st.cache_data(show_spinner=False)
def claims_index() -> list[dict]:
    procs = store.procedures()
    rows = []
    for x in store.all_claims():  # truth is deliberately dropped here
        c = x["claim"]
        rows.append({
            "Claim": c["claim_id"], "Member": store.member(c["member_id"]).get("name", ""),
            "Procedure": (procs.get(c["procedure_code"]) or {}).get("name", c["procedure_code"]) if isinstance(procs, dict) else c["procedure_code"],
            "Amount (Rs)": int(c["claimed_amount"]),
        })
    return rows


@st.cache_data(ttl=10, show_spinner=False)
def model_status() -> dict:
    m = get_llm()
    return dict(model=m.model, **m.health())


def read_json(name: str):
    f = ROOT / "data" / name
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except (OSError, ValueError):
        return None


def get_claim(cid: str) -> dict:
    return next(x["claim"] for x in store.all_claims() if x["claim"]["claim_id"] == cid)


def pick(cid: str, fault: bool = False):
    ss.sel, ss.fault, ss.go = cid, fault, True


st.markdown('<div class="brand"><span class="mark">Claims<i>Desk</i></span><span class="sub">HEALTH-INSURANCE ADJUDICATION · FIVE AGENTS · ONE CRITIC THAT CAN SAY NO</span></div>', unsafe_allow_html=True)
tab_desk, tab_eval, tab_rob, tab_arch = st.tabs(["Claim desk", "Evaluation", "Robustness", "Architecture"])

# ------------------------------------------------------------------ claim desk
with tab_desk:
    if ss.intro:
        ia, ib = st.columns([12, 1])
        ia.markdown('<div class="panel intro"><p><b>What is this?</b> A hospital sends an insurer a claim. Five software agents read it, check it against the policy and decide whether to pay.</p>'
                    '<ol><li><span>1</span>Pick a claim</li><li><span>2</span>Press Run</li><li><span>3</span>Watch the agents</li></ol></div>', unsafe_allow_html=True)
        if ib.button("Hide", key="hide_intro", help="Hide this introduction"):
            ss.intro = False
            st.rerun()
    left, right = st.columns([1, 2.35], gap="large")
    with left:
        st.markdown('<div class="eyebrow">Demo cases</div>', unsafe_allow_html=True)
        st.markdown('<div class="demo">', unsafe_allow_html=True)
        for i, (cid, fault, title, sub) in enumerate(DEMOS):
            st.button(f"{title}\n{cid}{' + fault' if fault else ''} · {sub}", key=f"demo{i}", on_click=pick, args=(cid, fault), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="eyebrow" style="margin-top:18px">All claims</div>', unsafe_allow_html=True)
        q = st.text_input("Search claims", placeholder="Search by id, member or procedure", label_visibility="collapsed")
        idx = claims_index()
        shown = [r for r in idx if q.lower() in " ".join(str(v) for v in r.values()).lower()] if q else idx
        ev = st.dataframe(shown, hide_index=True, width="stretch", height=260, on_select="rerun", selection_mode="single-row", key="tbl")
        rows = ev.selection.rows if ev and ev.selection else []
        if rows != ss.df_last:
            ss.df_last = rows
            if rows:
                ss.sel, ss.fault = shown[rows[0]]["Claim"], False

        st.markdown('<div class="eyebrow" style="margin-top:18px">Controls</div>', unsafe_allow_html=True)
        st.toggle("Simulate a calculation error (to show the Critic catching it)", key="fault")
        status = model_status()
        st.markdown(R.model_status_html(status), unsafe_allow_html=True)
        run = st.button("Run claim", type="primary", width="stretch", disabled=not status["ok"])

    with right:
        ph_claim, ph_pipe, ph_story, ph_dec = st.empty(), st.empty(), st.empty(), st.empty()
        with st.expander("Show the technical details", expanded=False):
            ph_tl, ph_crit = st.empty(), st.empty()

    def paint(claim, events, finished, decision, state=None, running=False):
        working = R.next_agent(events) if running else None
        ph_pipe.markdown(R.pipeline_svg(events, finished, decision, working), unsafe_allow_html=True)
        ph_tl.markdown(R.timeline_html(events, store.clauses()), unsafe_allow_html=True)
        ph_crit.markdown(R.critic_html(events), unsafe_allow_html=True)
        if state is not None and decision:
            ph_story.markdown(R.story_html(explain(state)) + R.llm_notes_html(events), unsafe_allow_html=True)
        elif running:
            ph_story.markdown(R.ticker_html(events), unsafe_allow_html=True)
        else:
            ph_story.empty()
        if decision:
            ph_dec.markdown(R.decision_html(decision, store.clauses(), claim["claim_id"]), unsafe_allow_html=True)
        else:
            ph_dec.empty()

    claim = get_claim(ss.sel)
    procs = store.procedures()
    proc = procs.get(claim["procedure_code"]) if isinstance(procs, dict) else next((p for p in procs if p.get("code") == claim["procedure_code"]), None)
    ph_claim.markdown(R.claim_summary(claim, store.member(claim["member_id"]), store.hospital(claim["hospital_id"]), proc), unsafe_allow_html=True)

    if run or ss.go:
        ss.go = False
        events, decision = [], None
        state = {"claim": claim, "findings": [], "trace": []}
        paint(claim, events, False, None, running=True)
        failed = None
        try:
            for node, update in graph.stream_claim(claim, get_llm(), ss.fault):
                for k, v in update.items():
                    state[k] = state.get(k, []) + v if k in ("trace", "findings") else v
                batch = update.get("trace", [])
                for i, t in enumerate(batch):
                    events.append(t)
                    paint(claim, events, False, None, running=(i == len(batch) - 1 and node != "decision"))
                    time.sleep(0.08 if t["agent"] == "Executor" else 0.2)
                if node == "decision":
                    decision = update["decision"]
        except LLMUnavailable as exc:
            failed = str(exc)
        if failed:
            ph_story.markdown(R.model_error_html(failed), unsafe_allow_html=True)
            ph_dec.empty()
            ss.res = None
        else:
            paint(claim, events, True, decision, state)
            ss.res = dict(cid=claim["claim_id"], events=events, decision=decision, fault=ss.fault, state=state)
    elif ss.res and ss.res["cid"] == claim["claim_id"]:
        paint(claim, ss.res["events"], True, ss.res["decision"], ss.res["state"])
    else:
        paint(claim, [], False, None)

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.markdown("### How well does it adjudicate?")
    st.markdown('<p class="howto"><b>How to read this.</b> We know the correct answer for 89 test claims. Each tile compares the system’s decision with that answer. The mistake that costs money is paying a claim that should not be paid, so watch the false-payment rate first. These answers are used only on this tab, never while a claim is being processed.</p>', unsafe_allow_html=True)
    s = read_json("eval_summary.json")
    if s is None:
        st.markdown('<div class="panel"><b>▲ No saved evaluation results.</b> <span class="note">data/eval_summary.json is missing.</span></div>', unsafe_allow_html=True)
        st.stop()
    st.markdown(R.kpis_html(s), unsafe_allow_html=True)
    st.markdown('<p class="note">The 25-minute manual baseline is an assumption, not a measured figure. The average time covers the rule-based checks only; it excludes the language model’s reading and writing, which takes seconds per claim. Results are read from a saved run, not recomputed here.</p>', unsafe_allow_html=True)
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("#### Confusion matrix")
        st.markdown(R.confusion_html(s["confusion"]), unsafe_allow_html=True)
    with b:
        st.markdown("#### Precision and recall by outcome")
        st.markdown(R.perclass_html(s["per_class"]), unsafe_allow_html=True)
    st.markdown("#### Misses")
    if s.get("misses"):
        st.dataframe(s["misses"], hide_index=True, width="stretch")
    else:
        st.markdown('<div class="panel"><b>✓ No misses.</b> <span class="note">Every claim received the expected decision.</span></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ robustness
with tab_rob:
    rb, ra, rl = read_json("robustness_before.json"), read_json("robustness_after.json"), read_json("robustness_llm.json")
    if rb is None or ra is None:
        st.markdown('<div class="panel"><b>▲ No saved robustness results.</b> <span class="note">data/robustness_before.json and robustness_after.json are needed.</span></div>', unsafe_allow_html=True)
        st.stop()
    st.markdown("### Does it hold up on messy input?")
    st.markdown('<p class="howto"><b>How to read this.</b> We made the claims messy (typos, reworded notes, unreadable summaries) and measured the system twice. The first run found weaknesses, we fixed them, and the second run shows the result. Because we fixed what the first run found, the rows marked <b>Not tuned against</b> are the honest test: they use kinds of mess we never trained on.</p>', unsafe_allow_html=True)
    st.caption("Before = first run, which exposed the weaknesses. After = after fixes. Only the two UNSEEN rows were not tuned against."
               + (f" With the language model = the same tests with {rl.get('model', 'the language model')} reading the summaries." if rl else ""))
    st.markdown("#### Accuracy per condition")
    st.markdown(R.robust_accuracy_html(rb, ra, rl), unsafe_allow_html=True)
    st.markdown("#### False payments: risky claims wrongly paid")
    st.caption(R.METRIC_HELP["false_payment_rate"])
    st.markdown(R.robust_fp_html(rb, ra, rl), unsafe_allow_html=True)
    st.markdown(f"#### Boundary cases: {ra['boundary_pass']} of {ra['boundary_total']} pass now (was {rb['boundary_pass']} of {rb['boundary_total']})")
    st.markdown(R.robust_boundaries_html(rb, ra, rl), unsafe_allow_html=True)

# ------------------------------------------------------------------ architecture
with tab_arch:
    svg = ROOT / "docs" / "architecture.svg"
    if svg.exists():
        st.markdown(f'<div class="panel">{svg.read_text(encoding="utf-8")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Architecture")
        st.caption("The finished diagram will appear here once docs/architecture.svg is added. Until then, the pipeline:")
        st.markdown(R.pipeline_svg([], True, None), unsafe_allow_html=True)

