"""Synthetic data generator: policies, tariffs, hospitals, members and labelled claims.

Every claim is stored together with a hidden ground-truth label (`truth`) so the
pipeline can be scored. The pipeline itself never sees the `truth` block.
"""
import datetime as dt
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
AS_OF = dt.date(2026, 10, 14)

PLANS = {
    "BASIC": dict(name="Suraksha Basic", sum_insured=300000, room_cap=3000),
    "SILVER": dict(name="Suraksha Silver", sum_insured=500000, room_cap=5000),
    "GOLD": dict(name="Suraksha Gold", sum_insured=1000000, room_cap=10000),
}

RULES = dict(
    initial_waiting_days=30,
    preexisting_months=24,
    copay_non_network=0.10,
    high_value_threshold=300000,
    filing_window_days=30,
    fraud_escalate_score=60,
    required_docs=["claim_form", "discharge_summary", "final_bill", "id_proof"],
    surgical_extra_docs=["operative_notes"],
)

CLAUSES = {
    "C1": ("Policy in force", "Cover applies only while the policy is active on the date of admission."),
    "C2": ("Initial waiting period", "No claim is payable for illness within 30 days of policy inception. Accidents are exempt."),
    "C3": ("Pre-existing disease waiting", "Conditions that pre-date the policy, declared or not, are covered only after 24 months of continuous cover."),
    "C4": ("Specific procedure waiting", "Listed procedures (cataract, hernia, gallbladder: 12 months; joint replacement: 24 months) carry a waiting period."),
    "C5": ("Permanent exclusions", "Cosmetic surgery and dental treatment (other than accidental injury) are excluded."),
    "C6": ("Room rent proportionate deduction", "Room rent is capped per day by plan. If exceeded, associated charges are reduced in the same proportion."),
    "C7": ("Procedure sub-limit", "Payout for a procedure is capped at its listed sub-limit."),
    "C8": ("Non-network co-payment", "Treatment at a non-network hospital carries a 10% co-payment."),
    "C9": ("Mandatory documents", "Claim form, discharge summary, final bill and ID proof (plus operative notes for surgery) are required."),
    "C10": ("Filing window", "Claims must be filed within 30 days of discharge; later claims need manager approval."),
    "C11": ("Non-medical items", "Registration, admission kit, toiletries and similar non-medical charges are not payable."),
    "C12": ("Remaining sum insured", "Payout cannot exceed the sum insured remaining after earlier claims in the policy year."),
    "C13": ("Duplicate claims", "A claim for the same member, procedure and admission period already paid is not payable again."),
    "C14": ("High-value review", "Claims above Rs 3,00,000 require medical officer sign-off."),
    "C15": ("Fraud and misrepresentation", "Claims showing indicators of inflation, tampering or misrepresentation are referred for investigation."),
}

PROCEDURES = [
    dict(code="P101", name="Appendectomy", kind="surgical", benchmark=55000, los=3, sublimit=75000, waiting=0),
    dict(code="P102", name="Cataract surgery", kind="surgical", benchmark=35000, los=1, sublimit=45000, waiting=12),
    dict(code="P103", name="Inguinal hernia repair", kind="surgical", benchmark=65000, los=2, sublimit=85000, waiting=12),
    dict(code="P104", name="Total knee replacement", kind="surgical", benchmark=250000, los=5, sublimit=350000, waiting=24),
    dict(code="P105", name="Coronary angioplasty", kind="surgical", benchmark=220000, los=3, sublimit=290000, waiting=0, tag="cardiac"),
    dict(code="P106", name="Cholecystectomy", kind="surgical", benchmark=70000, los=3, sublimit=90000, waiting=12),
    dict(code="P107", name="Dengue treatment", kind="medical", benchmark=40000, los=4, sublimit=60000, waiting=0),
    dict(code="P108", name="Pneumonia treatment", kind="medical", benchmark=45000, los=5, sublimit=65000, waiting=0),
    dict(code="P109", name="Cosmetic rhinoplasty", kind="surgical", benchmark=120000, los=2, sublimit=0, waiting=0, excluded=True),
    dict(code="P110", name="Dental implant", kind="surgical", benchmark=60000, los=1, sublimit=0, waiting=0, excluded=True, accident_exempt=True),
    dict(code="P111", name="Fracture fixation (ORIF)", kind="surgical", benchmark=130000, los=4, sublimit=165000, waiting=0),
    dict(code="P112", name="Diabetic ketoacidosis management", kind="medical", benchmark=60000, los=4, sublimit=80000, waiting=0, tag="diabetes"),
    dict(code="P113", name="Diagnostic coronary angiography", kind="surgical", benchmark=18000, los=1, sublimit=25000, waiting=0, tag="cardiac"),
]
PROC = {p["code"]: p for p in PROCEDURES}

HOSPITALS = [
    dict(id="H01", name="Apollo Care Hospital", city="Pune", network=True, watchlist=False),
    dict(id="H02", name="Sahyadri Multispeciality", city="Pune", network=True, watchlist=False),
    dict(id="H03", name="Jehangir Medical Centre", city="Pune", network=True, watchlist=False),
    dict(id="H04", name="Ruby Hall Clinic", city="Pune", network=True, watchlist=False),
    dict(id="H05", name="Fortis Mulund", city="Mumbai", network=True, watchlist=False),
    dict(id="H06", name="Narayana Health City", city="Bengaluru", network=True, watchlist=False),
    dict(id="H07", name="Lotus Nursing Home", city="Pune", network=False, watchlist=False),
    dict(id="H08", name="City Care Clinic", city="Nashik", network=False, watchlist=False),
    dict(id="H09", name="SunRise Multispeciality", city="Pune", network=True, watchlist=True),
    dict(id="H10", name="MedCore Hospital", city="Thane", network=True, watchlist=True),
]

FIRST = ["Aarav", "Vihaan", "Ishaan", "Rohan", "Kabir", "Ananya", "Diya", "Meera", "Saanvi", "Isha",
         "Rahul", "Priya", "Neha", "Amit", "Sneha", "Vikram", "Pooja", "Karan", "Riya", "Arjun"]
LAST = ["Sharma", "Patil", "Deshmukh", "Kulkarni", "Joshi", "Mehta", "Iyer", "Nair", "Gupta", "Verma", "Rao", "Shah"]


def _d(x):
    return x.isoformat()


def _months_before(date, months):
    return date - dt.timedelta(days=int(months * 30.44))


class Factory:
    def __init__(self, seed=7):
        self.r = random.Random(seed)
        self.members = []
        self.claims = []
        self.n = 0

    # ---- helpers -------------------------------------------------------
    def hosp(self, network=True, watch=False):
        pool = [h for h in HOSPITALS if h["network"] == network and h["watchlist"] == watch]
        return self.r.choice(pool)

    def member(self, adm, months=None, days=None, plan=None, declared=None, valid_till=None, prior=None):
        self.n += 1
        mid = f"M{1000 + self.n}"
        start = adm - dt.timedelta(days=days) if days is not None else _months_before(adm, months)
        plan = plan or self.r.choice(list(PLANS))
        m = dict(
            member_id=mid,
            name=f"{self.r.choice(FIRST)} {self.r.choice(LAST)}",
            age=self.r.randint(24, 68),
            sex=self.r.choice(["M", "F"]),
            plan=plan,
            policy_start=_d(start),
            valid_till=_d(valid_till or AS_OF + dt.timedelta(days=120)),
            declared_conditions=declared or [],
            sum_insured_used=self.r.choice([0, 0, 0, 15000, 32000]),
            prior_claims=prior or [],
        )
        self.members.append(m)
        return m

    def bill(self, proc, los, rate, mult=1.0, nonmed=0):
        room = rate * los
        base = max(proc["benchmark"] - 3000 * los, 0.45 * proc["benchmark"])  # benchmark excluding a typical room
        rest = base * mult
        rnd = lambda v: int(round(v, -1))
        b = dict(
            room=rnd(room),
            professional=rnd(rest * 0.55),
            medicines=rnd(rest * 0.22),
            diagnostics=rnd(rest * 0.13),
            consumables=rnd(rest * 0.10),
            non_medical=rnd(nonmed),
        )
        return b

    def summary(self, hosp, mem, proc_name, diag, adm, dis, history, room_type, accident, include_proc=True):
        lines = [
            f"DISCHARGE SUMMARY - {hosp['name']}, {hosp['city']}.",
            f"Patient {mem['name']}, {mem['age']}{mem['sex']}. Admitted {adm}, discharged {dis}.",
            f"Diagnosis: {diag}.",
        ]
        if include_proc:
            lines.append(f"Procedure performed: {proc_name}.")
        else:
            lines.append("Underwent treatment as per standard protocol.")
        if accident:
            lines.append("Presented following a road traffic accident.")
        lines.append(f"History: {history}")
        lines.append(f"Room category: {room_type}. Condition at discharge: stable.")
        return " ".join(lines)

    def add(self, scenario, truth, mem, hosp, proc, adm, los, rate, room_type, *, mult=1.0, nonmed=0,
            accident=False, history="No significant past history.", drop_docs=0, filed_gap=None,
            summary_proc=None, include_proc=True, tamper=0, diag=None, fit=False, exceed=False):
        dis = adm + dt.timedelta(days=los)
        gap = filed_gap if filed_gap is not None else self.r.randint(2, 14)
        filed = dis + dt.timedelta(days=gap)
        b = self.bill(proc, los, rate, mult, nonmed)
        while fit and proc['sublimit'] and sum(b.values()) > proc['sublimit'] * 0.97:
            mult *= 0.95
            b = self.bill(proc, los, rate, mult, nonmed)
        while exceed and proc['sublimit'] and sum(b.values()) <= proc['sublimit'] * 1.03 and mult < 1.48:
            mult *= 1.02
            b = self.bill(proc, los, rate, mult, nonmed)
        total = sum(b.values())
        claimed = total + tamper
        docs = list(RULES["required_docs"]) + (RULES["surgical_extra_docs"] if proc["kind"] == "surgical" else [])
        for _ in range(drop_docs):
            docs.remove(self.r.choice([d for d in docs if d != "id_proof" or len(docs) > 4]))
        diag = diag or f"{proc['name']} indicated"
        text = self.summary(hosp, mem, summary_proc or proc["name"], diag, _d(adm), _d(dis), history,
                            room_type, accident, include_proc)
        idx = len(self.claims) + 1
        claim = dict(
            claim_id=f"CLM-26-{idx:04d}", member_id=mem["member_id"], hospital_id=hosp["id"],
            procedure_code=proc["code"], diagnosis=diag, admission_date=_d(adm), discharge_date=_d(dis),
            filed_date=_d(filed), los_days=los, room_type=room_type, room_rate=rate, is_accident=accident,
            bill=b, claimed_amount=claimed, documents=docs, discharge_summary=text,
        )
        self.claims.append(dict(claim=claim, truth=dict(decision=truth, scenario=scenario)))

    def adm(self, lo=70, hi=160):
        return AS_OF - dt.timedelta(days=self.r.randint(lo, hi))


def build(seed=7):
    f = Factory(seed)
    r = f.r
    P = PROC

    def rate_ok(plan):
        return r.choice(range(1500, int(PLANS[plan]["room_cap"] * 0.95), 100))

    def room_name(rate):
        return "General ward" if rate < 2600 else "Semi-private" if rate < 6000 else "Private"

    def clean_proc(plan):
        pool = [p for p in PROCEDURES if not p.get("excluded") and not p.get("tag") and p["benchmark"] * 1.2 < 0.8 * PLANS[plan]["sum_insured"]
                and p["benchmark"] * 1.2 < RULES["high_value_threshold"]]
        return r.choice(pool)

    def veteran(adm, plan=None, **kw):
        return f.member(adm, months=r.randint(36, 72), plan=plan, **kw)

    def n(k, fn):
        for _ in range(k):
            fn()

    # --- APPROVE ---------------------------------------------------------
    def approve_clean():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("approve_clean", "APPROVE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.85, 1.1), fit=True)
    n(14, approve_clean)

    def approve_borderline():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("approve_borderline", "APPROVE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(1.25, 1.35), fit=True)
    n(3, approve_borderline)

    def approve_frequent():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan)
        prior = [dict(procedure_code=r.choice(["P107", "P108", "P101"]), admission_date=_d(adm - dt.timedelta(days=r.randint(40, 330))), amount=r.randint(20000, 50000), status="paid") for _ in range(3)]
        m = veteran(adm, plan, prior=prior)
        rt = rate_ok(plan); f.add("approve_frequent_flag", "APPROVE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.1), fit=True)
    n(2, approve_frequent)

    def approve_ambiguous():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("approve_ambiguous_summary", "APPROVE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.1), include_proc=False, fit=True)
    n(3, approve_ambiguous)

    # --- PARTIAL ---------------------------------------------------------
    def partial_room():
        adm = f.adm(); plan = r.choice(["BASIC", "SILVER"]); p = clean_proc(plan); m = veteran(adm, plan)
        rt = int(PLANS[plan]["room_cap"] * r.uniform(1.4, 2.0)); f.add("partial_room_rent", "PARTIAL", m, f.hosp(), p, adm, p["los"], rt, "Private", mult=r.uniform(0.9, 1.05))
    n(6, partial_room)

    def partial_nonnet():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("partial_non_network", "PARTIAL", m, f.hosp(network=False), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.85, 1.0))
    n(5, partial_nonnet)

    def partial_nonmed():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("partial_non_medical", "PARTIAL", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.85, 1.0), nonmed=r.randint(3000, 9000))
    n(5, partial_nonmed)

    def partial_sublimit():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = P[r.choice(["P111", "P101", "P106"])]; m = veteran(adm, plan)
        rt = int(PLANS[plan]["room_cap"] * 0.9 // 100 * 100); f.add("partial_sublimit", "PARTIAL", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(1.4, 1.46), exceed=True)
    n(3, partial_sublimit)

    # --- REJECT ----------------------------------------------------------
    def reject_initial():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = P[r.choice(["P107", "P108", "P101"])]
        m = f.member(adm, days=r.randint(8, 24), plan=plan)
        rt = rate_ok(plan); f.add("reject_initial_waiting", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05))
    n(4, reject_initial)

    def reject_specific():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"])
        if r.random() < 0.5:
            p = P[r.choice(["P102", "P103", "P106"])]; m = f.member(adm, months=r.randint(3, 9), plan=plan)
        else:
            p = P["P104"]; m = f.member(adm, months=r.randint(13, 20), plan="GOLD")
        rt = rate_ok(m["plan"]); f.add("reject_specific_waiting", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05))
    n(5, reject_specific)

    def reject_preexisting():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"])
        if r.random() < 0.5:
            p = P["P112"]; hist = f"Known case of Type 2 diabetes since {r.randint(2015, 2021)}."; cond = "diabetes"
        else:
            p = P["P105"]; hist = f"Known case of coronary artery disease since {r.randint(2016, 2022)}."; cond = "cardiac"
        declared = [cond] if r.random() < 0.5 else []
        m = f.member(adm, months=r.randint(5, 18), plan=plan, declared=declared)
        rt = rate_ok(plan); f.add("reject_pre_existing", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05), history=hist)
    n(5, reject_preexisting)

    def reject_excl():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = P[r.choice(["P109", "P110"])]; m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("reject_exclusion", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05))
    n(4, reject_excl)

    def reject_inactive():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = clean_proc(plan)
        m = f.member(adm, months=r.randint(36, 60), plan=plan, valid_till=adm - dt.timedelta(days=r.randint(20, 90)))
        rt = rate_ok(plan); f.add("reject_lapsed_policy", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05))
    n(3, reject_inactive)

    def reject_dup():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan)
        prior = [dict(procedure_code=p["code"], admission_date=_d(adm + dt.timedelta(days=r.randint(-2, 2))), amount=int(p["benchmark"] * 0.9), status="paid")]
        m = veteran(adm, plan, prior=prior)
        rt = rate_ok(plan); f.add("reject_duplicate", "REJECT", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05))
    n(4, reject_dup)

    # --- ESCALATE --------------------------------------------------------
    def esc_inflated():
        adm = f.adm(); plan = "GOLD"; p = P[r.choice(["P101", "P102", "P107", "P103", "P106", "P111"])]; m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("escalate_inflated_bill", "ESCALATE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(2.2, 2.7))
    n(5, esc_inflated)

    def esc_watch():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = P[r.choice(["P101", "P103", "P106", "P111"])]; m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("escalate_watchlist_hospital", "ESCALATE", m, f.hosp(watch=True), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(1.6, 1.85))
    n(3, esc_watch)

    def esc_mismatch():
        adm = f.adm(); m = veteran(adm, "GOLD"); rt = rate_ok("GOLD")
        f.add("escalate_procedure_mismatch", "ESCALATE", m, f.hosp(), P["P105"], adm, 3, rt, room_name(rt), mult=r.uniform(0.9, 1.0),
              summary_proc=P["P113"]["name"], diag="Chest pain, evaluation")
    n(3, esc_mismatch)

    def esc_tamper():
        adm = f.adm(); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("escalate_bill_tampering", "ESCALATE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05), tamper=r.choice([18000, 24500, 31000]))
    n(3, esc_tamper)

    def esc_high():
        adm = f.adm(); m = veteran(adm, "GOLD"); rt = rate_ok("GOLD")
        f.add("escalate_high_value", "ESCALATE", m, f.hosp(), P["P104"], adm, 5, rt, room_name(rt), mult=r.uniform(1.27, 1.33))
    n(3, esc_high)

    def esc_late():
        adm = f.adm(80, 160); plan = r.choice(["SILVER", "GOLD"]); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("escalate_late_filing", "ESCALATE", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05), filed_gap=r.randint(38, 60))
    n(2, esc_late)

    # --- QUERY -----------------------------------------------------------
    def query_docs():
        adm = f.adm(); plan = r.choice(list(PLANS)); p = clean_proc(plan); m = veteran(adm, plan)
        rt = rate_ok(plan); f.add("query_missing_documents", "QUERY", m, f.hosp(), p, adm, p["los"], rt, room_name(rt), mult=r.uniform(0.9, 1.05), drop_docs=r.choice([1, 2]))
    n(4, query_docs)

    r.shuffle(f.claims)
    for i, c in enumerate(f.claims, 1):
        old = c["claim"]["claim_id"]
        c["claim"]["claim_id"] = f"CLM-26-{i:04d}"
    return f


def main():
    DATA.mkdir(exist_ok=True)
    f = build()
    (DATA / "plans.json").write_text(json.dumps(dict(plans=PLANS, rules=RULES, clauses={k: dict(title=v[0], text=v[1]) for k, v in CLAUSES.items()}), indent=2))
    (DATA / "procedures.json").write_text(json.dumps(PROCEDURES, indent=2))
    (DATA / "hospitals.json").write_text(json.dumps(HOSPITALS, indent=2))
    (DATA / "members.json").write_text(json.dumps(f.members, indent=2))
    (DATA / "claims.json").write_text(json.dumps(f.claims, indent=2))
    print(f"wrote {len(f.claims)} claims, {len(f.members)} members")


if __name__ == "__main__":
    main()
