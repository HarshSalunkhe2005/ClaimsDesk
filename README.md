# Claims Adjudication Agent

A multi-agent proof of concept that makes the first decision on health-insurance claims. Built for CA-3 (Agentic AI, domain-specific PoC) by Harsh Salunkhe (24070126066) and Dharna Sharma (24070126053).

## In plain words
When you are treated in hospital, the hospital sends your insurer a claim: a bill plus a discharge summary. Someone at the insurer has to decide whether to pay, how much, or whether something looks wrong. This project does that first decision with five cooperating software agents, and hands only the doubtful claims to a person.

| Agent | What it does, in one line |
|---|---|
| Planner | Decides how closely to check this claim |
| Extractor | Reads the hospital's discharge summary |
| Executor | Runs the checks and works out what is payable |
| Critic | Double-checks the work, and can send it back |
| Decision | Picks the outcome and writes to the member |

**Example (claim CLM-26-0019, a cataract operation):** the Planner sees a small claim at a normal hospital and picks a quick check. The Executor finds the price is about 2.4 times the normal cost for that operation. The Critic notices that a bill that high should have had a fraud check, which the quick plan skipped, so it sends the claim back. The fraud check runs, scores 60 out of 100, and the claim goes to a human officer instead of being paid automatically.

The five possible outcomes are: **approved**, **approved in part** (with deductions explained), **rejected** (a policy rule applies), **on hold** (documents missing), and **sent to a human**.

## What the language model does (and does not do)
A real local model, `llama3.2:3b` served by [Ollama](https://ollama.com), does three jobs: it **reads the discharge summary** (including shorthand like "k/c/o CAD" or "TKR"), explains the Planner's choice in one sentence, and writes the **letter to the member**. It never makes a decision. Every outcome, payout and deduction comes from rules in `claimsagent/tools.py`.

Guardrails, because a small model can invent things:
- When reading a summary it must answer from our own fixed lists of procedures and conditions, and **quote the summary**. A treatment is accepted only if the quote really appears in the text, and a condition only if a real, non-negative quote from the history supports it.
- Letters are checked: they must contain the exact figures, must not guess "Mr" or "Ms", must not contain placeholders, and must never mention fraud checks. A letter that fails twice is not shown.
- There is no fallback model. If Ollama is not running, the app says so and Run is disabled.

## Problem
Insurers and TPAs adjudicate claims by hand: an officer reads the bill and discharge summary, checks the policy clauses, looks for inflation or fraud, and decides. It is slow, inconsistent, and fraud slips through. This PoC automates the first pass and sends only uncertain or risky claims to a human.

## Agents (LangGraph)
| Agent | Responsibility |
|---|---|
| Planner | Assesses claim risk and builds a quick or thorough plan; re-plans when the Critic objects |
| Extractor | Reads the discharge summary (fuzzy text parser plus the language model, with verified quotes) |
| Executor | Runs the plan with Python tools; stops early on a hard rule failure; computes the payout |
| Critic | Re-verifies payout arithmetic and limits, clause citations, extraction, and whether fraud screening matched the risk; can send the claim back (at most two rounds) |
| Decision | Applies precedence (reject, query, escalate, partial, approve), cites clauses, writes the letter |

Architecture diagram: `docs/architecture.svg` (also `docs/architecture.png`).

## Run
```bash
# 1. Install Ollama (https://ollama.com), then pull the model once (about 2 GB):
ollama pull llama3.2:3b

# 2. Python side
pip install -r requirements.txt
streamlit run app.py                # demo UI (Ollama must be running)

# Optional
python -m claimsagent.data_gen      # regenerate the synthetic data (seeded)
python -m claimsagent.evaluate      # score the pipeline on the 89 labelled claims (a few minutes)
python -m claimsagent.robustness    # robustness suite (long; see below)
python -m claimsagent.llm_bench llama3.2:3b qwen2.5:3b --seeds 101 --noise abbreviations   # compare models
```
A laptop with a 4 GB GPU runs the model comfortably. Reading one summary takes about 3.5 seconds; a full live claim with the plan note and letter takes roughly 10 to 15 seconds, and the first request is slower while the model loads. A hosted website cannot run the model, so the demo is local.

## Data
All data is synthetic: 89 labelled claims across 21 scenarios, 89 members, 10 hospitals, 13 procedures, 15 policy clauses. Ground-truth labels live beside each claim and are never passed into the pipeline.

## Results
**Main set:** with the model, 89 of 89 claims get the correct outcome, with 0 risky claims wrongly paid and 0 payable claims wrongly held. The labels come from the same design as the rules, so this shows that the rules cover the scenarios, not accuracy on real claims. That is why there is a harder test.

**Robustness test 1, hardening the rules** (`data/robustness_before.json`, `data/robustness_after.json`): 445 further claims from five new seeds with damaged summaries and bills, plus 23 boundary cases (for example 29 versus 30 days of cover). The first run found real weaknesses: typos and rewording hid treatment mismatches and pre-existing conditions (up to 33 risky claims wrongly paid out of 240), an unreadable summary was not put on hold, and an expensive private room diluted the price check. After fixes (fuzzy matching, a hold for unreadable summaries, a price check on non-room charges) all noise types the fixes were written after scored 100% and all 23 boundary cases passed.

**Robustness test 2, the language model on unseen noise** (`data/robustness_rules_only.json`, `data/robustness_llm.json`): three noise families written after the rules were tuned, on one held-out seed. The model's prompt was developed on separate claims; "telegraphic notes" was written after the prompt was fixed.

| Noise family | Rules only: accuracy | Rules only: risky claims wrongly paid | With llama3.2:3b: accuracy | With llama3.2:3b: wrongly paid | Payable claims wrongly held |
|---|---|---|---|---|---|
| Clean wording | 100% | 0 of 48 | 100% | 0 of 48 | 0 of 41 |
| Random character noise | 96.6% | 3 of 48 | 100% | 0 of 48 | 0 of 41 |
| Clinical abbreviations (TKR, k/c/o DM-2) | 94.4% | 5 of 48 | 100% | 0 of 48 | 0 of 41 |
| Telegraphic notes (PMH: IHD, Op: CAG) | 91% | 8 of 48 | 100% | 0 of 48 | 0 of 41 |

Both columns read the same 89 claims. All 23 boundary cases also pass with the model. With only the text parser, shorthand slips through; with the model, none of the 48 risky claims was wrongly paid and none of the 41 payable claims was wrongly held. That is 89 claims from one seed, so treat it as strong evidence on this data, not proof of real-world accuracy.

**Choosing the model** (development claims, abbreviations): `llama3.2:3b` read the treatment correctly in 258 of 258 summaries and found 15 of 15 real histories with 0 of 252 invented ones, at 3.4 s median. `qwen2.5:3b` read 226 of 258 treatments, 15 of 15 histories, 0 of 252 invented, at 3.3 s. Llama was chosen.

## Limitations
- All data is synthetic and the summaries are generated from templates.
- The main-set labels share their design with the rules, so that score measures coverage, not real-world accuracy.
- The model results use one held-out seed, so the uncertainty is wide, and the prompt was tuned on separate development claims.
- A 3-billion-parameter model is slow on a laptop and can still miss shorthand it has not seen. The quote check makes it fail safe, but not perfect.
- Fraud weights, price benchmarks and thresholds are hand-set, not learned.
- The manual-processing baseline (25 minutes per claim) is an assumption.
- The system needs a local Ollama server; there is no cloud demo.
