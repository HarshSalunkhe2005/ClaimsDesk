# Claims Adjudication Agent

A multi-agent proof of concept that performs first-pass adjudication of health-insurance claims. Built for CA-3 (Agentic AI, domain-specific PoC).

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

## Problem
Insurers and TPAs adjudicate claims by hand: an officer reads the bill and discharge summary, checks the policy clauses, looks for inflation or fraud, and decides. It is slow, inconsistent, and fraud slips through. This PoC automates the first pass and sends only uncertain or risky claims to a human.

## Agents (LangGraph)
| Agent | Responsibility |
|---|---|
| Planner | Assesses claim risk and builds a lean or deep plan; re-plans when the Critic objects |
| Extractor | Parses the discharge summary (procedure, chronic history, accident) |
| Executor | Runs the plan with Python tools; stops early on a hard rule failure; computes the payout |
| Critic | Re-verifies payout arithmetic and limits, clause citations, extraction, and whether fraud screening matched the risk; can send the claim back |
| Decision | Applies precedence (reject, query, escalate, partial, approve), cites clauses, writes the letter |

Architecture diagram: `docs/architecture.svg`.

## Run
```bash
pip install -r requirements.txt
python -m claimsagent.data_gen      # regenerate synthetic data (seeded)
python -m claimsagent.evaluate      # score the pipeline against labelled claims
streamlit run app.py                # demo UI
```

## Models
The default is a deterministic mock model, so the PoC runs offline. A local Ollama model can be selected in the UI; any model failure falls back to the mock.

## Data
All data is synthetic: 89 labelled claims across 21 scenarios, 89 members, 10 hospitals, 13 procedures, 15 policy clauses. Ground-truth labels live beside each claim and are never passed into the pipeline.

## Robustness test (`python -m claimsagent.robustness`)
The main set scores 89/89, but its labels come from the same scenario design as the rules, so that number only shows rule coverage. To test more honestly, the suite uses 445 held-out claims (five fresh seeds), applies noise to the discharge summaries and bills, and adds 23 policy boundary cases (for example 29 vs 30 days, Rs 2,99,990 vs Rs 3,00,010). Ground truth is fixed by the scenario or the policy text, never by running the pipeline.

The first run found real weaknesses (`data/robustness_before.json`): typos and rewording hid procedure mismatches and pre-existing conditions (up to 33 false payments out of 240 risky claims), an unreadable summary was not held for resubmission, and an expensive private room diluted the inflation check. Fixes: fuzzy procedure and history matching, a query when the summary is unreadable, and an inflation ratio based on non-room charges. Two of my own test bugs were also corrected during this pass. Results after the fixes (`data/robustness_after.json`):

| Condition | Accuracy | False payments |
|---|---|---|
| Clean held-out seeds | 100% | 0 / 240 |
| OCR-style typos, paraphrasing, case and whitespace, bill jitter, combined | 100% | 0 / 240 |
| Summary unreadable | 100% | 0 / 205 |
| **Unseen** random character noise | 99.8% | 1 / 240 |
| **Unseen** clinical abbreviations (TKR, PTCA, k/c/o DM-2) | 94.8% | 23 / 240 |
| Boundary cases | 23 / 23 | n/a |

The first five rows are noise types the fixes were written after seeing, so 100% there is partly tuned. The two unseen rows were added afterwards and were not tuned against; they are the honest figure.

## Limitations
- Labels come from the scenario generator, so accuracy on the main set measures rule coverage, not real-world performance.
- The extractor fails on clinical abbreviations: a pre-existing condition written as "k/c/o DM-2" or a procedure written as "TKR" can slip through (23 false payments out of 240 risky claims in the abbreviation test). A language-model extractor is the intended fix. A strict mode that holds any claim whose history cannot be verified would trade these misses for more manual reviews.
- Discharge summaries are templated; real documents need OCR and a stronger extractor.
- Tariff benchmarks and fraud weights are hand-set, not learned.
- The manual-processing baseline (25 minutes per claim) is an assumption.
