"""HTML builders for the claims desk. Pure functions of the event list, so a finished run can be redrawn after any rerun.

Status is never carried by colour alone: every chip has an icon and a word, and state changes are also written out.
"""
from __future__ import annotations

from html import escape as e

from claimsagent.explain import AGENT_BLURB, METRIC_HELP, clause_name
from claimsagent.tools import OPTIONAL_TOOLS, TOOL_LABELS

NODES = ["Planner", "Extractor", "Executor", "Critic", "Decision"]
STATUS = {
    "pass": ("✓", "Pass"), "info": ("i", "Info"), "query": ("?", "Query"), "flag": ("⚑", "Flag"), "fail": ("✕", "Fail"), "warn": ("▲", "Warn"),
}
DECISIONS = {
    "APPROVE": ("✓", "Approved"), "PARTIAL": ("◐", "Partially approved"), "REJECT": ("✕", "Rejected"),
    "ESCALATE": ("▲", "Escalated to a person"), "QUERY": ("?", "More information needed"),
}
SIDE = {"APPROVE": "pass", "PARTIAL": "flag", "REJECT": "fail", "ESCALATE": "flag", "QUERY": "query"}


def inr(n) -> str:
    return "Rs " + f"{float(n):,.0f}"


def chip(status: str, text: str | None = None) -> str:
    icon, word = STATUS.get(status, STATUS["info"])
    return f'<span class="chip s-{e(status)}"><b aria-hidden="true">{icon}</b>{e(text or word)}</span>'


def clause_chip(cid, clauses) -> str:
    if not cid:
        return ""
    return f'<span class="clause">{e(clause_name(cid))} ({e(str(cid))})</span>'


# ---------------------------------------------------------------- state from events
def summarise(events: list[dict]) -> dict:
    """Derive what the pipeline view needs from the flat trace."""
    runs: dict[str, set] = {n: set() for n in NODES}
    replans = 0
    verdicts = []
    for ev in events:
        runs[ev["agent"]].add(ev.get("iteration", 0))
        if ev["kind"] == "replan":
            replans += 1
        if ev["kind"] == "verdict":
            verdicts.append(ev["data"].get("verdict"))
    return dict(runs=runs, replans=replans, verdicts=verdicts, last=events[-1]["agent"] if events else None)


# ---------------------------------------------------------------- pipeline
def _wrap(txt: str, x: int, width: int = 21) -> str:
    lines, cur = [], ""
    for w_ in txt.split():
        if len(cur) + len(w_) + 1 > width and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = f"{cur} {w_}".strip()
    lines.append(cur)
    return "".join(f'<tspan x="{x}" dy="{0 if i == 0 else 18}">{e(l)}</tspan>' for i, l in enumerate(lines))


def pipeline_svg(events: list[dict], finished: bool, decision: dict | None) -> str:
    s = summarise(events)
    xs = [112, 332, 552, 772, 992]
    w, h, y, bw, bh = 1104, 290, 170, 192, 150
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Pipeline: Planner, Extractor, Executor, Critic, Decision, with a loop from Critic back to Planner">',
           '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9 z" fill="#3b4a5f"/></marker>'
           '<marker id="arg" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#e9b44c"/></marker></defs>']
    # forward links
    for a, b in zip(xs, xs[1:]):
        out.append(f'<path class="link" d="M{a + bw // 2 + 4} {y + bh // 2} H{b - bw // 2 - 4}" marker-end="url(#ar)"/>')
    # loop-back arrow
    on = s["replans"] > 0
    cx_c, cx_p = xs[3], xs[0]
    out.append(f'<path class="loop{" on" if on else ""}" d="M{cx_c} {y - bh // 2 - 4} C{cx_c} 22, {cx_p} 22, {cx_p} {y - bh // 2 - 4}" marker-end="url(#{"arg" if on else "ar"})"/>')
    label = f"RE-PLAN ×{s['replans']}" if on else ("no re-plan needed" if finished else "Critic can send the claim back")
    out.append(f'<text class="looplbl{" on" if on else ""}" x="{(cx_c + cx_p) // 2}" y="30" text-anchor="middle">{"↺ " if on else ""}{e(label)}</text>')
    for n, x in zip(NODES, xs):
        ran = len(s["runs"][n])
        active = (not finished) and s["last"] == n
        done = ran > 0 and not active
        st = "Working…" if active else ("Done" if done else "Waiting")
        ic = "◉" if active else ("✓" if done else "○")
        extra = ""
        if ran > 1:
            extra = f'<text x="{x + bw // 2 - 10}" y="{y - bh // 2 + 18}" text-anchor="end" class="st" style="fill:#e9b44c;font-weight:700">×{ran}</text>'
        sub = ""
        if n == "Critic" and s["verdicts"]:
            v = (s["verdicts"][-1] or "").upper()
            sub = {"PASS": "verdict: pass ✓", "REPLAN": "verdict: re-plan ↺", "ESCALATE": "verdict: escalate ▲"}.get(v, v)
        if n == "Decision" and decision:
            sub = DECISIONS.get(decision["decision"], ("", decision["decision"]))[1]
        out.append(
            f'<g class="node {"active" if active else ""} {"done" if done else ""}">'
            f'<rect x="{x - bw // 2}" y="{y - bh // 2}" width="{bw}" height="{bh}" rx="16"/>'
            f'<text class="ic" x="{x - bw // 2 + 16}" y="{y - bh // 2 + 30}">{ic}</text>'
            f'<text class="nm" x="{x}" y="{y - 36}" text-anchor="middle">{n}</text>'
            f'<text class="bl" x="{x}" y="{y - 12}" text-anchor="middle">{_wrap(AGENT_BLURB[n], x)}</text>'
            f'<text class="st" x="{x}" y="{y + 56}" text-anchor="middle">{e(sub or st)}</text>{extra}</g>'
        )
    out.append("</svg>")
    return '<div class="pipe panel">' + "".join(out) + "</div>"


# ---------------------------------------------------------------- timeline
def _planner_block(evs, plan_steps, clauses) -> str:
    h = ""
    for ev in evs:
        d = ev.get("data", {})
        steps = d.get("steps", [])
        msg = ev["message"]
        mode = "Second look" if ev["kind"] == "replan" else ("Thorough check" if "deep" in msg.lower() else "Quick check")
        sched = {x for x in steps}
        chips = "".join(f'<span class="step">{e(x)}</span>' for x in steps)
        not_sched = [t for t in OPTIONAL_TOOLS if ev["kind"] == "plan" and t not in sched]
        chips += "".join(f'<span class="step skip" title="Not scheduled by this plan">{e(t)}</span>' for t in not_sched)
        why = d.get("rationale") or msg
        h += (f'<div class="plan"><span class="mode"><span aria-hidden="true">◆</span>{mode}</span>'
              f'<p>{e(why)}</p><div class="steps">{chips}</div>'
              + (f'<p class="note">Struck-through checks were not scheduled by this plan.</p>' if not_sched else "") + "</div>")
        if ev["kind"] == "replan":
            h += f'<div class="ev">{e(msg)}</div>'
    return h


def _executor_block(evs, clauses) -> str:
    h = ""
    for ev in evs:
        d = ev.get("data", {})
        if ev["kind"] == "skip":
            h += f'<div class="tool"><span class="step skip">{e(d.get("tool", ""))}</span>{chip("info", "Skipped")}<span class="note">{e(ev["message"])}</span></div>'
        else:
            h += f'<div class="tool">{chip(d.get("status", "info"))}<span>{e(ev["message"])}</span>{clause_chip(d.get("clause"), clauses)}</div>'
    return h


def _check_status(txt: str) -> str:
    t = txt.upper()
    return "fail" if "FAIL" in t else ("warn" if "WARN" in t else "pass")


def _critic_block(evs, clauses) -> str:
    h = ""
    for ev in evs:
        if ev["kind"] == "check":
            st = ev.get("data", {}).get("status", "pass")
            h += f'<div class="ev">{chip(st if st in STATUS else "pass")} {e(ev["message"])}</div>'
        else:
            h += f'<div class="ev"><b>{e(ev["message"])}</b></div>'
    return h


def timeline_html(events: list[dict], clauses: dict) -> str:
    if not events:
        return '<div class="panel"><div class="eyebrow">Trace</div><p class="note">Pick a claim and press Run to watch the agents work.</p></div>'
    blocks: list[tuple[str, int, list[dict]]] = []
    for ev in events:
        key = (ev["agent"], ev.get("iteration", 0))
        if blocks and (blocks[-1][0], blocks[-1][1]) == key:
            blocks[-1][2].append(ev)
        else:
            blocks.append((ev["agent"], ev.get("iteration", 0), [ev]))
    html = ""
    for agent, it, evs in reversed(blocks):  # newest first so the latest step is always in view
        body = ""
        if agent == "Planner":
            body = _planner_block(evs, [], clauses)
        elif agent == "Executor":
            body = _executor_block(evs, clauses)
        elif agent == "Critic":
            body = _critic_block(evs, clauses)
        else:
            body = "".join(f'<div class="ev">{e(ev["message"])}</div>' for ev in evs)
        rnd = f'<small>round {it + 1}</small>' if it else ""
        html += f'<div class="blk"><h4>{e(agent)}{rnd}</h4>{body}</div>'
    return f'<div class="panel"><div class="eyebrow">Trace, newest first</div><div class="tl" tabindex="0" role="region" aria-label="Agent trace">{html}</div></div>'


# ---------------------------------------------------------------- critic panel
def critic_html(events: list[dict]) -> str:
    rounds: dict[int, dict] = {}
    for ev in events:
        if ev["agent"] != "Critic":
            continue
        r = rounds.setdefault(ev.get("iteration", 0), dict(checks=[], verdict=None, conf=None, msg=""))
        if ev["kind"] == "check":
            lab, _, detail = ev["message"].partition(": ")
            status = ev.get("data", {}).get("status", "pass")
            r["checks"].append((lab, status, detail))
        elif ev["kind"] == "verdict":
            r["verdict"] = ev["data"].get("verdict")
            r["conf"] = ev["data"].get("confidence")
            r["msg"] = ev["message"]
    if not rounds:
        return ""
    out = ""
    for it in sorted(rounds):
        r = rounds[it]
        rows = ""
        for lab, status, detail in r["checks"]:
            st = status if status in STATUS else "pass"
            rows += f'<div class="check">{chip(st)}<div>{e(lab.replace(": PASS", "").replace(": FAIL", ""))}<small>{e(detail.replace("PASS (", "").replace("FAIL (", "").rstrip(")"))}</small></div></div>'
        v = (r["verdict"] or "").lower()
        vcls = {"pass": "s-pass", "replan": "s-flag", "escalate": "s-fail"}.get(v, "s-info")
        vic = {"pass": "✓ Pass", "replan": "↺ Re-plan", "escalate": "▲ Escalate"}.get(v, v)
        conf = r["conf"]
        gauge = f'<div style="flex:1"><div class="gauge" role="img" aria-label="Confidence {conf:.0%}"><i style="width:{(conf or 0) * 100:.0f}%"></i></div></div><b class="mono">{conf:.0%}</b>' if conf is not None else ""
        sent = ""
        if v == "replan":
            why = r["msg"].split("Sending back to Planner:", 1)[-1].strip()
            sent = f'<div class="sentback"><b>↺ Sent back to the Planner.</b> {e(why)}</div>'
        out += (f'<div class="panel" style="margin-bottom:12px"><div class="eyebrow">Critic · round {it + 1}</div>{rows}'
                f'<div style="display:flex;gap:14px;align-items:center;margin-top:12px"><span class="verdict {vcls}">{vic}</span>'
                f'<span class="note">confidence</span>{gauge}</div>{sent}</div>')
    return out


# ---------------------------------------------------------------- decision card
def decision_html(dec: dict, clauses: dict, claim_id: str) -> str:
    d = dec["decision"]
    ic, word = DECISIONS.get(d, ("•", d))
    claimed = float(dec["claimed"])
    rows = f'<div class="wfr"><span>Claimed</span><div class="tr"><div class="bar claimed" style="left:0;width:100%"></div></div><span class="v">{inr(claimed)}</span></div>'
    remaining = claimed
    for x in dec.get("deductions", []):
        amt = float(x["amount"])
        left = max(0.0, (remaining - amt) / claimed * 100) if claimed else 0
        width = amt / claimed * 100 if claimed else 0
        rows += (f'<div class="wfr"><span>− {e(x["label"])} {clause_chip(x.get("clause"), clauses)}</span>'
                 f'<div class="tr"><div class="bar ded" style="left:{left:.1f}%;width:{width:.1f}%"></div></div><span class="v">− {inr(amt)}</span></div>')
        remaining -= amt
    pay = float(dec["payable"])
    rows += (f'<div class="wfr total"><span>Payable</span><div class="tr"><div class="bar pay" style="left:0;width:{(pay / claimed * 100) if claimed else 0:.1f}%"></div></div>'
             f'<span class="v">{inr(pay)}</span></div>')
    chips = "".join(clause_chip(c, clauses) + " " for c in dec.get("clauses", []))
    cl = "".join(
        f'<details class="cl"><summary><span class="clause">{e(clause_name(c))} ({e(c)})</span></summary><p>{e((clauses.get(c) or {}).get("text", ""))}</p></details>'
        for c in dec.get("clauses", [])
    )
    reasons = "".join(f"<li>{e(r)}</li>" for r in dec.get("reasons", []))
    human = '<span class="badge"><span aria-hidden="true">☝</span>Needs human review</span>' if dec.get("needs_human") else ""
    fraud = ""
    if dec.get("fraud_score") not in (None, 0):
        fs = dec["fraud_score"]
        level = "high" if fs >= 60 else ("elevated" if fs >= 30 else "low")
        fraud = (f'<div style="margin-top:14px"><div class="eyebrow">Fraud score</div><div style="display:flex;gap:12px;align-items:center">'
                 f'<div class="gauge" style="flex:1" role="img" aria-label="Fraud score {fs} of 100, {level}"><i style="width:{fs}%"></i></div><b class="mono">{fs}/100 · {level}</b></div></div>')
    return (
        f'<div class="dec"><div class="eyebrow">Decision · {e(claim_id)}</div>'
        f'<div class="big d-{e(d)}"><span class="ico" aria-hidden="true">{ic}</span><span>{e(word)}</span></div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 4px">{human}<span class="badge" style="border-color:#3b4a5f;color:#9aa7b8">'
        f'{dec.get("iterations", 0)} re-plan{"s" if dec.get("iterations", 0) != 1 else ""} · confidence {float(dec.get("confidence", 0)):.0%}</span></div>'
        f'<div class="wf"><div class="eyebrow">How the payout was worked out</div>{rows}</div>'
        f'<div class="eyebrow">Why</div><ul style="margin:0 0 10px 18px;padding:0;font-size:14.5px">{reasons}</ul>'
        f'<div class="eyebrow">Policy clauses</div><div style="margin-bottom:6px">{chips}</div>{cl}{fraud}'
        f'<div class="eyebrow" style="margin-top:16px">Letter to the member</div><div class="letter">{e(dec.get("letter", ""))}</div></div>'
    )


def claim_summary(claim: dict, member: dict, hospital: dict, proc: dict | None) -> str:
    net = "network" if hospital.get("network") else "out of network"
    return (
        f'<div class="panel claim"><div class="eyebrow">Claim</div><div class="id">{e(claim["claim_id"])}</div>'
        f'<div class="amt">{inr(claim["claimed_amount"])}</div><dl>'
        f'<dt>Member</dt><dd>{e(member.get("name", ""))}, {member.get("age", "")} · {e(str(member.get("plan", "")))}</dd>'
        f'<dt>Hospital</dt><dd>{e(hospital.get("name", ""))}, {e(hospital.get("city", ""))} ({net})</dd>'
        f'<dt>Procedure</dt><dd>{e((proc or {}).get("name", claim["procedure_code"]))}</dd>'
        f'<dt>Stay</dt><dd>{claim["los_days"]} day(s), {e(claim["room_type"])} @ {inr(claim["room_rate"])}</dd>'
        f'<dt>Diagnosis</dt><dd>{e(claim["diagnosis"])}</dd></dl></div>'
    )


# ---------------------------------------------------------------- evaluation
def kpis_html(s: dict) -> str:
    manual_s = s["manual_minutes_baseline"] * 60
    sec = s["avg_ms"] / 1000
    tiles = [
        (f'{s["accuracy"]:.1%}', "Decision accuracy", f'{s["correct"]} of {s["claims"]} claims', METRIC_HELP["accuracy"]),
        (f'{s["false_payment_rate"]:.1%}', "False-payment rate", f'{s["false_payments"]} of {s["risky_claims"]} risky claims paid', METRIC_HELP["false_payment_rate"]),
        (f'{s["wrongful_hold_rate"]:.1%}', "Wrongful-hold rate", f'{s["wrongful_holds"]} of {s["payable_claims"]} payable claims held', METRIC_HELP["wrongful_hold_rate"]),
        (f'{s["straight_through_rate"]:.1%}', "Straight-through rate", f'{s["critic_replans"]} claims got a second look from the Critic', METRIC_HELP["straight_through_rate"]),
        (f'{sec:.3f} s', "Average per claim", f'vs {s["manual_minutes_baseline"]} min by hand (assumed) · about {manual_s / max(sec, 1e-6):,.0f}× faster', "Time the system takes per claim. The manual figure is an assumption, not a measurement."),
    ]
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="n">{a}</div><div class="l">{b}</div><div class="t">{c}</div><p class="h">{d}</p></div>' for a, b, c, d in tiles) + "</div>"


def confusion_html(conf: dict) -> str:
    cols = list(next(iter(conf.values())).keys())
    mx = max((v for r in conf.values() for v in r.values()), default=1) or 1
    h = '<table class="cm"><caption class="note" style="caption-side:bottom;padding-top:8px">Rows: true outcome. Columns: predicted. The outlined cells on the diagonal are correct decisions; the number in each cell is the count.</caption><thead><tr><th scope="col">true ↓ / predicted →</th>' + "".join(f'<th scope="col">{c}</th>' for c in cols) + "</tr></thead><tbody>"
    for t, row in conf.items():
        h += f'<tr><th scope="row">{t}</th>'
        for c in cols:
            v = row[c]
            a = 0.08 + 0.55 * v / mx if v else 0
            h += f'<td class="{"diag" if c == t else ""}" style="background:rgba(233,180,76,{a:.2f})">{v}</td>'
        h += "</tr>"
    return h + "</tbody></table>"


def perclass_html(pc: dict) -> str:
    h = '<table class="cls"><thead><tr><th scope="col">Class</th><th scope="col">Support</th><th scope="col">Precision</th><th scope="col">Recall</th></tr></thead><tbody>'
    for k, v in pc.items():
        h += f'<tr><th scope="row">{k}</th><td>{v["support"]}</td><td>{v["precision"]:.0%}</td><td>{v["recall"]:.0%}</td></tr>'
    return h + "</tbody></table>"


# ---------------------------------------------------------------- robustness
def _rob_rows(before: dict, after: dict) -> list[dict]:
    b = {c["name"]: c for c in before["conditions"]}
    rows = []
    for a in after["conditions"]:
        rows.append(dict(name=a["name"], unseen=bool(a.get("unseen")), b=b.get(a["name"]), a=a))
    return rows


def _pct(x) -> str:
    return f"{x * 100:.1f}%"


def _bar(v: float, cls: str) -> str:
    return f'<div class="rb"><i class="{cls}" style="width:{max(v, 0) * 100:.1f}%"></i></div>'


def robust_accuracy_html(before: dict, after: dict) -> str:
    h = ('<table class="rob"><thead><tr><th scope="col">Condition</th><th scope="col">Before</th><th scope="col">After</th></tr></thead><tbody>')
    for r in _rob_rows(before, after):
        name = r["name"].replace("UNSEEN: ", "")
        tag = '<span class="unseen"><b aria-hidden="true">◆</b> Not tuned against</span>' if r["unseen"] else ""
        bcell = (f'<span class="v">{_pct(r["b"]["accuracy"])}</span>{_bar(r["b"]["accuracy"], "bb")}' if r["b"]
                 else '<span class="note">not measured</span>')
        h += (f'<tr class="{"u" if r["unseen"] else ""}"><th scope="row">{e(name)}{tag}</th><td>{bcell}</td>'
              f'<td><span class="v">{_pct(r["a"]["accuracy"])}</span>{_bar(r["a"]["accuracy"], "ba")}</td></tr>')
    return h + "</tbody></table>"


def robust_fp_html(before: dict, after: dict) -> str:
    h = ('<table class="rob"><thead><tr><th scope="col">Condition</th><th scope="col">Before</th><th scope="col">After</th></tr></thead><tbody>')
    for r in _rob_rows(before, after):
        name = r["name"].replace("UNSEEN: ", "")
        tag = '<span class="unseen"><b aria-hidden="true">◆</b> Not tuned against</span>' if r["unseen"] else ""

        def cell(c):
            if not c:
                return '<span class="note">not measured</span>'
            n = c["false_payments"]
            ic = chip("pass", "None") if n == 0 else chip("fail", f"{n} wrongly paid")
            return f'{ic}<span class="note"> of {c["risky_claims"]} risky · {_pct(c["false_payment_rate"])}</span>'
        h += f'<tr class="{"u" if r["unseen"] else ""}"><th scope="row">{e(name)}{tag}</th><td>{cell(r["b"])}</td><td>{cell(r["a"])}</td></tr>'
    return h + "</tbody></table>"


def robust_boundaries_html(before: dict, after: dict) -> str:
    bmap = {x["case"]: x for x in before["boundaries"]}
    h = ('<table class="rob"><thead><tr><th scope="col">Boundary case</th><th scope="col">Expected</th><th scope="col">Before</th><th scope="col">After</th></tr></thead><tbody>')
    for x in after["boundaries"]:
        b = bmap.get(x["case"])

        def res(c):
            if not c:
                return '<span class="note">not measured</span>'
            return chip("pass", f'Pass · {c["got"]}') if c["ok"] else chip("fail", f'Fail · got {c["got"]}')
        h += f'<tr><th scope="row">{e(x["case"])}</th><td class="mono">{e(x["expected"])}</td><td>{res(b)}</td><td>{res(x)}</td></tr>'
    return h + "</tbody></table>"


# ---------------------------------------------------------------- plain-English story
def story_html(ex: dict) -> str:
    steps = ""
    for i, st_ in enumerate(ex["steps"], 1):
        re = st_.get("replanned")
        flag = '<span class="replan"><b aria-hidden="true">↺</b> The Critic sent this back for a second look</span>' if re else ""
        steps += (f'<li class="{"re" if re else ""}"><span class="num" aria-hidden="true">{i}</span><div><b>{e(st_["agent"])}</b>'
                  f'<span class="bl">{e(st_["blurb"])}</span>{flag}<p>{e(st_["text"])}</p></div></li>')
    return (f'<div class="panel story"><div class="eyebrow">What happened</div>'
            f'<h3>{e(ex["headline"])}</h3><p class="next">{e(ex["next_step"])}</p><ol>{steps}</ol></div>')


def ticker_html(events: list[dict]) -> str:
    if not events:
        return ""
    ev = events[-1]
    return (f'<div class="panel story"><div class="eyebrow">Working on it</div>'
            f'<p class="next"><b>{e(ev["agent"])}</b> · {e(AGENT_BLURB[ev["agent"]])}</p><p>{e(ev["message"])}</p></div>')
