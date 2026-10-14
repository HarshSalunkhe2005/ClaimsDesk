"""Claims desk: watch the five agents adjudicate a claim. Run with `streamlit run app.py`."""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from claimsagent import evaluate, graph, llm as llm_mod, store
from ui import render as R

ROOT = Path(__file__).parent
st.set_page_config(page_title="Claims Desk", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{(ROOT / 'ui' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

DEMOS = [
    ("CLM-26-0003", False, "Clean approve", "Everything checks out"),
    ("CLM-26-0001", False, "Partial payout", "Room-rent proportionate deduction"),
    ("CLM-26-0004", False, "Reject", "Initial waiting period"),
    ("CLM-26-0007", False, "Reject", "Pre-existing condition"),
    ("CLM-26-0019", False, "Re-plan loop", "Lean plan, Critic spots a 2× bill, escalate"),
    ("CLM-26-0008", False, "Bill tampering", "Also loops back"),
    ("CLM-26-0006", False, "Procedure mismatch", "Escalate"),
    ("CLM-26-0002", False, "Missing documents", "Query the hospital"),
    ("CLM-26-0003", True, "Fault injected", "Critic catches a wrong payout"),
]

ss = st.session_state
ss.setdefault("sel", "CLM-26-0019")
ss.setdefault("fault", False)
ss.setdefault("go", False)
ss.setdefault("res", None)
ss.setdefault("df_last", [])


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


@st.cache_data(show_spinner="Scoring all 89 claims…")
def eval_cached(model: str):
    return evaluate.evaluate(llm_mod.get_llm("mock") if model == "Mock" else llm_mod.get_llm("ollama", model=model))


def get_claim(cid: str) -> dict:
    return next(x["claim"] for x in store.all_claims() if x["claim"]["claim_id"] == cid)


def pick(cid: str, fault: bool = False):
    ss.sel, ss.fault, ss.go = cid, fault, True


st.markdown('<div class="brand"><span class="mark">Claims<i>Desk</i></span><span class="sub">HEALTH-INSURANCE ADJUDICATION · FIVE AGENTS · ONE CRITIC THAT CAN SAY NO</span></div>', unsafe_allow_html=True)
tab_desk, tab_eval, tab_rob, tab_arch = st.tabs(["Claim desk", "Evaluation", "Robustness", "Architecture"])

# ------------------------------------------------------------------ claim desk
with tab_desk:
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
        st.toggle("Inject executor fault", key="fault", help="Corrupts the payout so you can watch the Critic catch it.")
        model = st.selectbox("Model", ["Mock", "Ollama"], help="Mock is deterministic and instant. Ollama needs a local server.")
        run = st.button("Run claim", type="primary", width="stretch")

    with right:
        ph_claim, ph_pipe, ph_dec = st.empty(), st.empty(), st.empty()
        c1, c2 = st.columns([1.25, 1], gap="medium")
        ph_tl, ph_crit = c1.empty(), c2.empty()

    def paint(claim, events, finished, decision):
        ph_pipe.markdown(R.pipeline_svg(events, finished, decision), unsafe_allow_html=True)
        ph_tl.markdown(R.timeline_html(events, store.clauses()), unsafe_allow_html=True)
        ph_crit.markdown(R.critic_html(events), unsafe_allow_html=True)
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
        try:
            model_obj = llm_mod.get_llm("mock") if model == "Mock" else llm_mod.get_llm("ollama")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not reach the Ollama model ({exc}). Falling back to Mock.")
            model_obj = llm_mod.get_llm("mock")
        events, decision = [], None
        paint(claim, events, False, None)
        for node, update in graph.stream_claim(claim, model_obj, ss.fault):
            for t in update.get("trace", []):
                events.append(t)
                paint(claim, events, False, None)
                time.sleep(0.12 if t["agent"] == "Executor" else 0.28)
            if node == "decision":
                decision = update["decision"]
        paint(claim, events, True, decision)
        ss.res = dict(cid=claim["claim_id"], events=events, decision=decision, fault=ss.fault)
    elif ss.res and ss.res["cid"] == claim["claim_id"]:
        paint(claim, ss.res["events"], True, ss.res["decision"])
    else:
        paint(claim, [], False, None)

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.markdown("### How well does it adjudicate?")
    st.caption("Scored against the hidden ground truth of all 89 claims. This is the only place the labels are used.")
    rows_, s = eval_cached("Mock")
    st.markdown(R.kpis_html(s), unsafe_allow_html=True)
    st.markdown('<p class="note">The 25-minute manual baseline is an assumption, not a measured figure.</p>', unsafe_allow_html=True)
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("#### Confusion matrix")
        st.markdown(R.confusion_html(s["confusion"]), unsafe_allow_html=True)
    with b:
        st.markdown("#### Precision and recall by outcome")
        st.markdown(R.perclass_html(s["per_class"]), unsafe_allow_html=True)
    st.markdown("#### Misses")
    if s["misses"]:
        st.dataframe(s["misses"], hide_index=True, width="stretch")
    else:
        st.markdown('<div class="panel"><b>✓ No misses.</b> <span class="note">Every claim received the expected decision.</span></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ robustness
with tab_rob:
    import json
    rb = json.loads((ROOT / "data" / "robustness_before.json").read_text(encoding="utf-8"))
    ra = json.loads((ROOT / "data" / "robustness_after.json").read_text(encoding="utf-8"))
    st.markdown("### Does it hold up on messy input?")
    st.caption("Before = first run, which exposed the weaknesses. After = after fixes. Only the two UNSEEN rows were not tuned against.")
    st.markdown("#### Accuracy per condition")
    st.markdown(R.robust_accuracy_html(rb, ra), unsafe_allow_html=True)
    st.markdown("#### False payments: risky claims wrongly paid")
    st.markdown(R.robust_fp_html(rb, ra), unsafe_allow_html=True)
    st.markdown(f"#### Boundary cases: {ra['boundary_pass']} of {ra['boundary_total']} pass now (was {rb['boundary_pass']} of {rb['boundary_total']})")
    st.markdown(R.robust_boundaries_html(rb, ra), unsafe_allow_html=True)

# ------------------------------------------------------------------ architecture
with tab_arch:
    svg = ROOT / "docs" / "architecture.svg"
    if svg.exists():
        st.markdown(f'<div class="panel">{svg.read_text(encoding="utf-8")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Architecture")
        st.caption("The finished diagram will appear here once docs/architecture.svg is added. Until then, the pipeline:")
        st.markdown(R.pipeline_svg([], True, None), unsafe_allow_html=True)

