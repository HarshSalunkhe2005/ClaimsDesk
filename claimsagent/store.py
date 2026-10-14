"""Read-only access to the synthetic data stores (stands in for the insurer's core systems)."""
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=None)
def _load(name):
    return json.loads((DATA / name).read_text())


def policy_book():
    return _load("plans.json")


def rules():
    return policy_book()["rules"]


def clauses():
    return policy_book()["clauses"]


def plan(plan_id):
    return policy_book()["plans"][plan_id]


def procedures():
    return {p["code"]: p for p in _load("procedures.json")}


def hospital(hid):
    return next(h for h in _load("hospitals.json") if h["id"] == hid)


_EXTRA_MEMBERS = {}


def register_members(members):
    """Lets the robustness suite add held-out members without touching the data files."""
    _EXTRA_MEMBERS.update({m["member_id"]: m for m in members})


def member(mid):
    if mid in _EXTRA_MEMBERS:
        return _EXTRA_MEMBERS[mid]
    return next(m for m in _load("members.json") if m["member_id"] == mid)


def all_claims():
    """Returns [{'claim': ..., 'truth': ...}]. Truth must never be passed into the pipeline."""
    return _load("claims.json")


def claim_only(claim_id):
    return next(c["claim"] for c in all_claims() if c["claim"]["claim_id"] == claim_id)
