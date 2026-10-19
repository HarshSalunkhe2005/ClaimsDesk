"""Local language-model layer (Ollama).

There is no fallback model. If Ollama is not running, or the model has not been pulled, calls raise
LLMUnavailable and the app says so. The model does three jobs:

  extract         read a discharge summary (grounded: it must quote the summary, and quotes are verified)
  plan_rationale  explain in plain words why the Planner chose its plan
  decision_letter write the letter to the member (numbers are checked against the real figures)

Every decision, payout and deduction is computed by the rules in tools.py, never by the model.
"""
import hashlib
import json
import os
import re

import requests

from . import store

DEFAULT_MODEL = os.environ.get("CLAIMS_LLM_MODEL", "llama3.2:3b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CONDITIONS = ["diabetes", "cardiac", "hypertension"]
CACHE_FILE = store.DATA / "llm_cache.json"
_NO_HISTORY = re.compile(r"no significant|no past|\bnil\b|\bnone\b|not significant|unremarkable|no known", re.I)


class LLMUnavailable(RuntimeError):
    """Ollama is not reachable, the model is missing, or the model timed out."""


_CATEGORY_WORDS = {
    "diabetes": ("diabet", "dm", "t2dm", "iddm", "niddm", "sugar"),
    "cardiac": ("cardi", "heart", "coronary", "cad", "ihd", "angina", "myocard", "mi"),
    "hypertension": ("hypertens", "htn", "blood pressure", "bp"),
}


def _to_category(label):
    """Maps whatever label the model wrote ("CAD", "DM-2", "heart disease") onto our three categories."""
    if not isinstance(label, str):
        return None
    words = set(re.findall(r"[a-z0-9]+", label.lower()))
    low = label.lower()
    for cat, keys in _CATEGORY_WORDS.items():
        if any((k in words) if len(k) <= 3 else (k in low) for k in keys):
            return cat
    return None


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


class OllamaLLM:
    def __init__(self, model=DEFAULT_MODEL, host=HOST, timeout=180, cache=False, tasks=("extract", "rationale", "letter")):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"
        self.tasks = set(tasks)  # accuracy tests only score decisions, so they can skip the prose tasks
        self._cache = None
        if cache:
            try:
                self._cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
            except Exception:
                self._cache = {}

    # ------------------------------------------------------------ plumbing
    def health(self):
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception:
            return dict(ok=False, message="Ollama is not running. Start it, then reload.")
        have = [m["name"] for m in r.json().get("models", [])]
        if not any(n == self.model or n.split(":")[0] == self.model for n in have):
            return dict(ok=False, message=f"Model not installed. Run: ollama pull {self.model}")
        return dict(ok=True, message=f"{self.model} is ready")

    def _generate(self, prompt, *, json_mode=False, max_tokens=160):
        key = hashlib.sha256(f"{self.model}|{json_mode}|{max_tokens}|{prompt}".encode()).hexdigest()
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        body = dict(model=self.model, prompt=prompt, stream=False, keep_alive="30m",
                    options=dict(temperature=0, seed=7, num_predict=max_tokens))
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.host}/api/generate", json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise LLMUnavailable("Ollama is not running. Start it, then try again.") from e
        except requests.exceptions.Timeout as e:
            raise LLMUnavailable(f"{self.model} timed out after {self.timeout}s.") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Model not installed. Run: ollama pull {self.model}")
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama returned HTTP {r.status_code}.")
        out = r.json()["response"].strip()
        if self._cache is not None:
            self._cache[key] = out
            CACHE_FILE.write_text(json.dumps(self._cache))
        return out

    def save_cache(self):
        if self._cache is not None:
            CACHE_FILE.write_text(json.dumps(self._cache))

    # ------------------------------------------------------------- extract
    def extract(self, text):
        """Grounded reading of a discharge summary.

        The model must quote the words it relies on. A treatment is accepted only if its quote really appears in
        the summary, and conditions only if a real quote from the history section supports them and that quote is
        not a "no history" statement. This stops a small model inventing facts. Returns
        {procedure_name|None, conditions[], accident} or None if the reply was unusable.
        """
        if "extract" not in self.tasks or len(text.strip()) < 40:  # nothing to read: the Executor queries the hospital
            return None
        names = [p["name"] for p in store.procedures().values()]
        prompt = (
            "Read this hospital discharge summary and report two things. Quote the summary exactly where asked.\n\n"
            "1. TREATMENT: the main procedure or treatment performed on this admission.\n"
            '   - "procedure_quote": copy the exact words from the summary that name it, or "" if none is stated.\n'
            '   - "procedure": the matching name from this list, or null if the quote matches none of them:\n'
            + "\n".join(f"     - {n}" for n in names)
            + "\n\n2. PAST HISTORY: long-standing conditions the patient already had BEFORE this admission "
            "(not the problem being treated now).\n"
            '   - "history_quote": copy the exact history sentence from the summary, or "" if there is none.\n'
            '   - "conditions": which of these the history quote says the patient has: '
            + ", ".join(f'"{c}"' for c in CONDITIONS)
            + " (cardiac means any heart disease). Use [] if the quote mentions none of them.\n\n"
            '3. "accident": true only if the admission followed an accident or injury.\n\n'
            "Clinical abbreviations are common here, so use your medical knowledge. "
            'Reply with JSON only: {"procedure_quote": "", "procedure": null, "history_quote": "", "conditions": [], "accident": false}\n\n'
            f"Discharge summary:\n<<<\n{text}\n>>>"
        )
        raw = self._generate(prompt, json_mode=True, max_tokens=220)
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                d = json.loads(m.group(0)) if m else None
            except Exception:
                d = None
        if not isinstance(d, dict):
            return None
        hay = _norm(text)
        # treatment: needs a quote that really appears, and a name from our own list
        proc = d.get("procedure")
        pq = _norm(d.get("procedure_quote") if isinstance(d.get("procedure_quote"), str) else "")
        lookup = {n.lower(): n for n in names}
        proc = lookup.get(_norm(proc)) if isinstance(proc, str) else None
        if proc and not (pq and pq in hay):
            proc = None
        # history: needs a real, non-negative quote
        hq_raw = d.get("history_quote") if isinstance(d.get("history_quote"), str) else ""
        hq = _norm(hq_raw)
        conds = d.get("conditions") if isinstance(d.get("conditions"), list) else []
        conds = sorted({k for k in (_to_category(c) for c in conds) if k})
        if not hq or hq not in hay or _NO_HISTORY.search(hq_raw):
            conds = []
        return dict(procedure_name=proc, conditions=conds, accident=bool(d.get("accident")))

    # ---------------------------------------------------------- rationale
    def plan_rationale(self, ctx):
        if "rationale" not in self.tasks:
            return None
        facts = (f"claim type: {ctx['kind']}; review chosen: {'thorough' if ctx['optional'] else 'quick'}; "
                 f"risk reasons: {', '.join(ctx['reasons']) or 'none'}; "
                 f"extra checks added: {', '.join(ctx['optional']) or 'none'}")
        prompt = ("You explain an insurance claims system's decision to a non-expert. In ONE plain sentence of at most 20 words, "
                  "say why this review level was chosen. Use only these facts, add nothing, and do not call the claim low-priority.\nFacts: " + facts)
        return self._generate(prompt, max_tokens=60).split("\n")[0].strip() or None

    # ------------------------------------------------------------- letter
    def decision_letter(self, ctx):
        if "letter" not in self.tasks:
            return None
        d = ctx["decision"]
        say = {
            "APPROVE": "Your claim has been approved and paid in full.",
            "PARTIAL": "Your claim has been approved in part.",
            "REJECT": "Your claim could not be paid.",
            "QUERY": "We need more documents before we can decide.",
            "ESCALATE": "A senior claims officer will review your claim and contact you.",
        }[d]
        facts = [f"Say this: {say}", f"Claim number: {ctx['claim_id']}"]
        must = [ctx["claim_id"], ctx["member_name"]]
        if d in ("APPROVE", "PARTIAL"):
            facts += [f"Amount claimed: Rs {ctx['claimed']:,.0f}", f"Amount paid: Rs {ctx['payable']:,.0f}"]
            must += [f"{ctx['payable']:,.0f}"]
        if d == "PARTIAL":
            for x in ctx["deductions"]:
                facts.append(f"Deduction: {x['label']}, Rs {x['amount']:,.0f}")
                must.append(f"{x['amount']:,.0f}")
        if d in ("REJECT", "QUERY") and ctx["reasons"]:
            facts.append("Explain: " + "; ".join(str(r) for r in ctx["reasons"][:2]))
        # an escalated claim gets no reasons: the member must not be told about internal risk checks
        prompt = (f"Write a short, polite letter from an insurer's claims team to a policyholder, 70 words at most.\n"
                  f"Start with exactly: Dear {ctx['member_name']},\n"
                  "Rules: use ONLY the facts below and copy every number exactly. Do not use titles such as Mr or Ms. "
                  "Do not use square brackets or placeholders. Never mention fraud, suspicion, scores or internal checks. "
                  "End with: Claims Team\n" + "\n".join(facts))
        banned = re.compile(r"\[|\]|\b(?:mr|ms|mrs|miss)\b\.?|fraud|suspic|score", re.I)
        status_words = re.compile(r"\b(?:APPROVE|PARTIAL|REJECT|QUERY|ESCALATE)\b")  # internal labels, case-sensitive
        for attempt in range(2):
            letter = self._generate(prompt + ("\nFollow every rule exactly." if attempt else ""), max_tokens=160)
            if all(m in letter for m in must) and not banned.search(letter) and not status_words.search(letter):
                return letter
        return None  # failed the checks twice: show no letter rather than a wrong one


def get_llm(model=None, **kw):
    return OllamaLLM(model or DEFAULT_MODEL, **kw)
