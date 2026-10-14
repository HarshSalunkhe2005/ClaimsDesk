"""Pluggable language-model layer.

MockLLM is a deterministic, template-driven stand-in so the PoC runs anywhere with no
downloads. OllamaLLM talks to a local Ollama server and falls back to the mock on any
failure, so a flaky model can never break the pipeline.
"""
import json
import re

import requests


class MockLLM:
    name = "mock-rule-llm"

    def extract(self, text):
        return None  # defer to the regex document parser

    def plan_rationale(self, ctx):
        bits = []
        if ctx["optional"]:
            bits.append("risk indicators present (" + ", ".join(ctx["reasons"]) + ") so deeper checks were added: " + ", ".join(ctx["optional"]))
        else:
            bits.append("low-risk profile (network hospital, modest amount, no watchlist hit) so the lean plan is sufficient")
        if ctx.get("replan"):
            bits.append("re-planning after Critic feedback: " + "; ".join(ctx["replan"]))
        return "Plan built for a " + ctx["kind"] + " claim: " + "; ".join(bits) + "."

    def decision_letter(self, ctx):
        d = ctx["decision"]
        who = ctx["member_name"]
        if d == "APPROVE":
            return f"Dear {who}, your claim {ctx['claim_id']} has been approved for Rs {ctx['payable']:,.0f}. No deductions were applicable."
        if d == "PARTIAL":
            ded = "; ".join(f"{x['label']} (Rs {x['amount']:,.0f}, clause {x['clause']})" for x in ctx["deductions"])
            return (f"Dear {who}, your claim {ctx['claim_id']} is approved in part: Rs {ctx['payable']:,.0f} of Rs {ctx['claimed']:,.0f} claimed. "
                    f"Deductions: {ded}.")
        if d == "REJECT":
            return f"Dear {who}, we are unable to pay claim {ctx['claim_id']}. Reason: {ctx['reasons'][0]}"
        if d == "QUERY":
            return f"Dear {who}, claim {ctx['claim_id']} is on hold. Please submit: {ctx['reasons'][0]}"
        return (f"Claim {ctx['claim_id']} is referred to a senior claims officer. Reasons: " + " | ".join(ctx["reasons"]))


class OllamaLLM(MockLLM):
    def __init__(self, model="llama3.2:3b", host="http://localhost:11434", timeout=60):
        self.model, self.host, self.timeout = model, host, timeout
        self.name = f"ollama:{model}"

    def _gen(self, prompt):
        r = requests.post(f"{self.host}/api/generate", json=dict(model=self.model, prompt=prompt, stream=False,
                                                                  options=dict(temperature=0)), timeout=self.timeout)
        r.raise_for_status()
        return r.json()["response"].strip()

    def extract(self, text):
        try:
            out = self._gen(
                "Extract from this discharge summary and reply with JSON only, keys: "
                "procedure (string or null), conditions (list of chronic conditions with 'known case' history), "
                "is_accident (bool).\n\n" + text)
            m = re.search(r"\{.*\}", out, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception:
            return None

    def plan_rationale(self, ctx):
        try:
            return self._gen("In two sentences, explain to an insurance claims manager why this processing plan "
                             "was chosen. Context: " + json.dumps(ctx))
        except Exception:
            return super().plan_rationale(ctx)

    def decision_letter(self, ctx):
        try:
            return self._gen("Write a short, polite, factual letter (max 80 words) to the policyholder informing "
                             "them of this claim decision. Do not invent facts. Context: " + json.dumps(ctx))
        except Exception:
            return super().decision_letter(ctx)


def get_llm(kind="mock", **kw):
    return OllamaLLM(**kw) if kind == "ollama" else MockLLM()
