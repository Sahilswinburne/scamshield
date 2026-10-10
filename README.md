# ScamShield — Investment & Loan Scam Detector

An AI for Finance Build Project: paste an investment listing, loan or credit offer, ad, or pitch and
get an evidence-backed scam-risk score, not just a label.

## What it does

| | |
|---|---|
| **Input** | Free text of a crypto/stock listing, loan/credit offer, ad, or social post (single or batch CSV) |
| **Scope** | Investments and trading, crypto, loans and credit, personal finance. Other text is reported as **Out of scope** (no risk score) |
| **Transformation** | Rule-based red-flag engine (10 scam categories, including loan/credit red flags, suspicious links and threats) + HF zero-shot LLM classifier (`facebook/bart-large-mnli`): a topic check, then eight specific scam-sign checks |
| **Output** | 0–100 risk score, Low/Medium/High verdict, and a full evidence breakdown (every category matched or clear, AI probability split, score arithmetic) |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first run with the AI toggle on
will download `facebook/bart-large-mnli` (~1.6 GB) from the HuggingFace Hub —
this requires internet access. If the model can't be downloaded, the app
automatically falls back to the rule-based score alone (the AI toggle stays
available but is reported as unavailable in the results).

## Deploy it (for your Build guide submission)

Two free options that give you a public URL:

**Streamlit Community Cloud**
1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set the main file to `app.py`.
3. Deploy — you'll get a public `*.streamlit.app` URL to put in your Build guide.

**HuggingFace Spaces**
1. Create a new Space, SDK = Streamlit.
2. Upload `app.py`, `scam_detector.py`, `requirements.txt`.
3. The Space builds automatically and gives you a public URL.

Either way, include the deployed URL **and** a link to the locked GitHub repo/commit
in your Build guide, as the brief requires.

## How to test the core function

1. Open the **Analyze a listing** tab.
2. Pick an example from the dropdown (crypto, pyramid and advance-fee loan scams; a legitimate seed round, personal loan and market commentary), or paste your own.
3. Click **Analyze listing** — you'll see the risk gauge, verdict badge, the rule-based
   vs AI score breakdown, and the evidence panel: every category with its matched phrases
   and points (or what was checked, if clear), the AI model's scam/legitimate split, and
   the calculation behind the final score.
4. For batch testing, use the **Batch scan (CSV)** tab and upload `sample_listings.csv`
   (included in this folder, 17 rows: mix of obvious scams, subtle scams, loan scams, a fake debt-collection email, legitimate text including a genuine overdue notice, and one off-topic delivery text).

## Files

- `app.py` — Streamlit UI (theming, gauge chart, single + batch modes)
- `scam_detector.py` — core detection logic (rule engine + AI layer + score fusion); this
  is the module your evaluation code should import and call directly for reproducible testing
- `requirements.txt` — dependencies
- `sample_listings.csv` — demo data for batch mode
- `evaluation/` — labelled test set, evaluation script, original-build baseline, results

## Evaluation

See [`evaluation/README.md`](evaluation/README.md) for the success criteria, test set,
results (current vs original build, rules-only vs rules+AI), improvement log and
failure analysis. To repeat it: `python evaluation/evaluate.py`.

Held-out result (current build, rules + AI): **precision 0.83, recall 0.71, F1 0.77**
(the original build: F1 0.38).

## Notes for the Evaluation report

`scam_detector.analyze(text, use_ai=True/False)` returns a structured `Assessment`
object (`final_score`, `verdict`, `rule_result.matches`, `semantic_score`) — call it
directly in a separate evaluation script against a labelled test set (see the project
brief's Evaluation report requirements) rather than driving the Streamlit UI for testing.
Setting `use_ai=False` isolates the rule engine's own performance, which is useful for
ablation testing (rule-only vs rule+AI accuracy).

## Limitations

- Rule engine only catches patterns it has been given — novel phrasing slips through.
- The AI model can be swayed by genuinely enthusiastic legitimate marketing copy (false positives).
- English-language only.
- Scope is investment pitches and consumer loan/credit offers; other financial scams
  (fake invoices, romance/impersonation) aren't specifically targeted.
- The AI scam-sign wordings were chosen on a small set (~30 finance texts); a larger
  labelled test set is needed before trusting the exact numbers.
- The topic check has grey areas: a toll-bill text passes as finance, and the council
  grant announcement (row 10) is ruled out of scope.

## How the AI layer works

1. **Topic check** — zero-shot over 4 finance topics and 9 non-finance topics
   (delivery, food, sport, personal chat, subscriptions, health, government,
   bills/tolls, social plans). If < 30% of the confidence lands on finance topics
   and no rule matched, the result is **Out of scope** with no score.
2. **Scam signs** — eight plain-English hypotheses, each scored independently
   (multi-label); the strongest is the AI score. Two target debt-collection and
   account scams (threats of legal action; threats to freeze or close an account).
   A sign for "tells the reader to pay through a link" was tested and dropped: it
   scored 99% on genuine bank reminders, so links are judged by the rule engine.
3. **Combination** — `final = 100 × [1 − (1 − rule/100) × (1 − AI/100)]`, so either
   layer alone can flag a scam and agreement pushes the score higher.

An earlier version asked the model to choose between "scam financial offer" and
"legitimate financial product" and averaged it 45/55 with the rules. That gave obvious
scams 5–20% and left the "Obvious crypto scam" example at Medium (48). On the same
texts the scam-sign approach scores every scam ≥ 85 and every legitimate text ≤ 21.

## Rule engine: links and threats

- **Suspicious links / payment channels** (`check_links` in `scam_detector.py`) extracts
  every web address and email domain and flags: a brand name (CommBank, Westpac, ANZ,
  NAB, Macquarie, ING, Latitude, Afterpay, PayPal, Binance, Coinbase, ATO, myGov…) on a
  domain that isn't that brand's official one; throwaway endings (`.top`, `.xyz`,
  `.info`…); link shorteners; raw IP links; bait-word addresses (`loan-settlement`,
  `secure-verify`). Each finding says why it was flagged.
- **Threats / intimidation** — final demands, legal/arrest/repossession threats, "pay
  instantly", "account suspended". Weighted lightly (14) because genuine overdue
  notices also mention consequences.
- These two categories are generic, so a match does **not** pull off-topic text into
  scope (the AusPost text stays Out of scope, but its fake link is still shown).

## Evaluation notes

- **Spot check (53 texts: sample CSV + 12 unseen + 9 off-topic + 11 collection/account
  notices):** 51 correct. Misses: council grant notice ruled out of scope; Linkt
  toll-bill scam treated as finance and scored Low (24).
- **Collection / account notices:** all 5 fakes flagged (4 High, Afterpay `bit.ly` text
  Medium from rules alone); all 6 genuine notices Low — including NAB/Latitude overdue
  notices that mention credit reporting and defaults, and messages with real bank links.
- **Fake Macquarie debt-collection email** — before: rules 0, AI 92 via a loosely
  fitting "pressure" sign. After: rules 38 (lookalike `macquarie-finance-group.net`,
  `.top` payment link, "secure external portal", 7 threat phrases) + AI 99.8
  ("threatens legal action to force payment") → 99.9 High.
- **AusPost delivery text (row 15)** — scored Low (18) before the topic check was added; now
  correctly reported as Out of scope (4% finance-related, top topic "a parcel delivery").

- **Legitimate personal-loan ad (row 11)** — "Borrow up to $50,000 … pay no establishment or
  ongoing fees without providing collateral …" scores Low with 0 rule matches: a correct true
  negative on a legitimate product, despite wording ("no fees", "without collateral") that
  naive keyword rules could misread.
- Decision-support tool, not a verdict — does not replace due diligence.
