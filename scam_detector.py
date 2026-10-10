"""
scam_detector.py
-----------------
Core AI/ML transformation for the ScamShield app.

Input:  raw text of an investment/crypto listing, loan or credit offer,
        pitch, ad, or social post
Output: a structured risk assessment — a numeric risk score, a verdict,
        and the specific evidence (matched red-flag phrases + an AI
        semantic judgement) that produced it.

Two layers combine into the final score:

1. Rule-based red-flag engine (always available, instant, fully
   explainable). Each category is a set of regex patterns tuned to
   language patterns documented in real scam-warning sources
   (ASIC Moneysmart, FTC consumer alerts, r/CryptoCurrency scam
   reports) — see README.md for citations.

2. HF zero-shot semantic classifier (facebook/bart-large-mnli).
   This is the actual "AI/ML transformation", run in two steps:
     a. Topic check — is the text about investing, crypto, loans/credit
        or personal finance at all? If not, it's reported as out of
        scope rather than given a misleading "Low risk".
     b. Scam-sign check — the model tests the text against specific,
        plain-English scam signs (guaranteed returns, urgency, send
        crypto to get more back, recovery-phrase requests, no-credit-check
        loans, recruitment rewards, threats of legal action or account
        closure). The strongest sign is the AI score,
        catching scams that don't use any of the literal red-flag phrases.
   Loaded lazily and cached, and the app degrades gracefully
   (rule-based score only) if the model can't be downloaded.

The two scores are combined (probabilistic OR, see below) into a single
0-100 risk score and a three-tier verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 1. Rule-based red-flag engine
# ---------------------------------------------------------------------------

@dataclass
class RedFlagCategory:
    name: str
    description: str
    patterns: list[str]
    weight: float          # contribution to rule_score if matched at all
    looks_for: str = ""    # plain-English summary of what the patterns catch
    icon: str = "🚩"
    # Optional extra check for things regex alone can't judge (e.g. whether a
    # web address really belongs to the bank it names). Returns evidence strings.
    checker: Optional[Callable[[str], list[str]]] = None
    # Finance-specific categories also mark the text as in scope; generic ones
    # (links, threats) appear in all kinds of scams, so they don't.
    finance_specific: bool = True


# --- Link / sender checks ---------------------------------------------------
# Every web address and email domain in the text is pulled out and judged on
# its own: who it claims to be, what ending it uses, and what words it's made of.

# Brands commonly impersonated in Australian finance scams -> their real domains.
# A host naming the brand that isn't one of these (or a subdomain of one) is a lookalike.
OFFICIAL_DOMAINS: dict[str, list[str]] = {
    "commbank": ["commbank.com.au", "commsec.com.au", "cba.com.au"],
    "commonwealth": ["commbank.com.au", "cba.com.au"],
    "westpac": ["westpac.com.au", "westpac.co.nz"],
    "anz": ["anz.com.au", "anz.com", "anz.co.nz"],
    "nab": ["nab.com.au"],
    "macquarie": ["macquarie.com.au", "macquarie.com"],
    "ing": ["ing.com.au", "ing.com"],
    "stgeorge": ["stgeorge.com.au"],
    "bankwest": ["bankwest.com.au"],
    "suncorp": ["suncorp.com.au", "suncorpbank.com.au"],
    "latitude": ["latitudefinancial.com.au", "latitudefinancial.com"],
    "afterpay": ["afterpay.com", "afterpay.com.au"],
    "paypal": ["paypal.com", "paypal.com.au"],
    "binance": ["binance.com"],
    "coinbase": ["coinbase.com"],
    "coinspot": ["coinspot.com.au"],
    "kraken": ["kraken.com"],
    "swyftx": ["swyftx.com", "swyftx.com.au"],
    "ato": ["ato.gov.au"],
    "mygov": ["my.gov.au"],
}

# Cheap / throwaway domain endings that real banks, lenders and exchanges don't use.
RISKY_TLDS = {
    "top", "xyz", "info", "click", "live", "online", "site", "icu", "buzz", "rest",
    "cc", "tk", "ml", "ga", "cf", "gq", "work", "support", "cfd", "sbs", "monster",
    "quest", "cyou", "vip", "bond", "win", "loan", "cash", "bid", "lol", "zip",
}
# Endings we recognise as real domains (so "e.g." or "p.a." aren't read as links).
KNOWN_TLDS = RISKY_TLDS | {
    "com", "net", "org", "gov", "edu", "io", "co", "app", "dev", "ai", "biz",
    "finance", "money", "bank", "exchange", "capital", "fund", "me", "us", "ly", "at", "gd", "gy",
}
URL_SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "is.gd", "rb.gy", "shorturl.at",
                  "tiny.cc", "ow.ly", "t.ly", "rebrand.ly"}
# Words scammers glue onto addresses to make them look official.
BAIT_WORDS = {
    "settlement", "secure", "verify", "verification", "recovery", "redelivery", "refund",
    "claim", "update", "login", "signin", "portal", "payment", "payments", "pay", "unlock",
    "wallet", "support", "billing", "alert", "rewards", "bonus", "collections", "debt",
    "arrears", "helpdesk", "auth", "validate",
}
HOST_RE = re.compile(r"(?<![@\w.-])(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})\b"
                     r"|@((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})\b")
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")


def _registrable(host: str) -> str:
    """'secure.loan-settlement.top' -> 'loan-settlement.top'; handles .com.au style endings."""
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in {"com", "net", "org", "gov", "edu", "co", "asn", "id"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _brand_in(host: str) -> Optional[str]:
    for token in re.split(r"[.\-\d]+", host):
        for brand in OFFICIAL_DOMAINS:
            if token == brand or (len(brand) >= 5 and token.startswith(brand)):
                return brand
    return None


def check_links(text: str) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()
    for m in HOST_RE.finditer(text.lower()):
        host = (m.group(1) or m.group(2)).removeprefix("www.")
        tld = host.rsplit(".", 1)[-1]
        if host in seen or not (tld in KNOWN_TLDS or len(tld) == 2):
            continue
        seen.add(host)
        reg = _registrable(host)
        reasons = []

        brand = _brand_in(host)
        if brand and not any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS[brand]):
            reasons.append(f"uses the name '{brand}' but isn't its official site "
                           f"({' / '.join(OFFICIAL_DOMAINS[brand][:2])})")
        if tld in RISKY_TLDS:
            reasons.append(f"'.{tld}' ending, which real banks and lenders don't use")
        if reg in URL_SHORTENERS:
            reasons.append("link shortener that hides the real destination")
        first = reg.split(".")[0]
        bait = [w for w in first.split("-") if w in BAIT_WORDS]
        if "-" in first and bait and not brand:
            reasons.append(f"made-up address built from bait words ('{'-'.join(bait)}')")

        if reasons:
            findings.append(f"{host} — " + "; ".join(reasons))
    for m in IP_URL_RE.finditer(text):
        findings.append(f"{m.group(0)} — raw IP address instead of a named website")
    return findings


RED_FLAG_CATEGORIES: list[RedFlagCategory] = [
    RedFlagCategory(
        name="Guaranteed / risk-free returns",
        description="Legitimate investments never guarantee profit. This is the single most common scam tell — including implied guarantees like 'zero losses' or 'never lost a trade'.",
        patterns=[
            r"\bguaranteed?\s+(profit|returns?|income|earnings?)\b",
            r"\brisk[-\s]?free\b",
            r"\bno\s+risk\b",
            r"\b(zero|0%|no)\s+(risk|losses?)\b",
            r"\bnever\s+lost\b",
            r"\b100%\s+(safe|secure|guaranteed|success)\b",
            r"\bcan.?t\s+lose\b",
        ],
        weight=22,
        looks_for="'guaranteed profit', 'risk-free', 'no risk', 'zero losses', 'never lost', '100% safe', 'can't lose'",
        icon="💰",
    ),
    RedFlagCategory(
        name="Unrealistic returns",
        description="Return figures far outside what any real market offers — as multiples, percentages, fixed rates, or a specific dollar-amount jump over a short window.",
        patterns=[
            r"\b\d{2,}\s*x\s+(your\s+)?(money|investment|profit)\b",
            r"\b(double|triple|10x|100x)\s+your\s+(money|crypto|investment)\b",
            r"\b\d{3,}%\s*(return|profit|roi|apy|gain)\b",
            r"\b\d{1,3}%\s*(daily|per\s+day|every\s+day|a\s+day|weekly|a\s+week)\b",
            r"\bturn\s+\$?[\d,]+\s+into\s+\$?[\d,]+\b",
            r"\$?[\d,]+\s+(into|to)\s+\$?[\d,]+\s+in\s+(just\s+)?\d+\s*(days?|hours?|weeks?)\b",
        ],
        weight=22,
        looks_for="'10x your money', 'double your crypto', '500% return', '5% daily', 'turn $500 into $5,000', '$1k to $10k in 7 days'",
        icon="📈",
    ),
    RedFlagCategory(
        name="Urgency / scarcity pressure",
        description="Artificial deadlines and scarcity push victims to skip due diligence.",
        patterns=[
            r"\bact\s+now\b",
            r"\blimited\s+(time|spots|slots|offer)\b",
            r"\bonly\s+\d+\s+(spots|slots|spaces)\s+(left|remaining)\b",
            r"\bdon.?t\s+miss\s+(this|out)\b",
            r"\boffer\s+ends\s+(today|soon|tonight)\b",
            r"\bhurry\b",
            r"\bonce\s+in\s+a\s+lifetime\b",
        ],
        weight=14,
        looks_for="'act now', 'limited time/spots', 'only 5 spots left', 'don't miss out', 'offer ends today', 'hurry', 'once in a lifetime'",
        icon="⏳",
    ),
    RedFlagCategory(
        name="Referral / pyramid structure",
        description="Returns depend on recruiting new investors rather than any real asset or business.",
        patterns=[
            r"\brefer\s+(\d+\s+|a\s+)?friends?\b(.{0,25}\bearn\b)?",
            r"\b(multi[-\s]?level|mlm)\b",
            r"\bearn\s+(from|on)\s+your\s+(downline|referrals?)\b",
            r"\bbring\s+\d+\s+(people|members|investors)\b",
        ],
        weight=18,
        looks_for="'refer friends and earn', 'multi-level / MLM', 'earn from your downline', 'bring 3 people'",
        icon="🔺",
    ),
    RedFlagCategory(
        name="Upfront payment / private key requests",
        description="Legitimate platforms and lenders never ask for your seed phrase, or for a fee before they will 'release' your funds or loan (the classic advance-fee scam).",
        patterns=[
            r"\b(seed|recovery|secret)\s+phrase\b",
            r"\b(12|24|twelve|twenty[-\s]four)[-\s]word\s+(seed\s+|recovery\s+)?phrase\b",
            r"\bprivate\s+key\b",
            r"\bsend\s+[\d.]*\s*(btc|eth|crypto|usdt)\b",
            r"\bsend\b.{0,40}\b(get|receive)\s+(double|twice|2x)\b",
            r"\b(processing|activation|release|withdrawal|insurance)\s+fee\s+required\b",
            r"\bpay\s+.{0,15}\bfee\b.{0,15}\bwithdraw\b",
            r"\b(pay|send|transfer|deposit)\b.{0,40}\bfee\b.{0,40}\b(release|unlock|receive|disburse|activate)\b",
            r"\b(to|before\s+we\s+can)\s+(release|unlock|receive|disburse)\b.{0,40}\b(pay|send|transfer|deposit)\b.{0,40}\bfee\b",
            r"\b(gift\s+cards?|itunes\s+cards?|western\s+union|moneygram)\b",
        ],
        weight=24,
        looks_for="'seed phrase', 'private key', 'send 0.1 BTC', 'release/withdrawal fee required', paying a fee to unlock, release or withdraw funds, payment by gift card or wire service",
        icon="🔑",
    ),
    RedFlagCategory(
        name="Loan / credit offer red flags",
        description="Real lenders must assess your ability to repay, so they never guarantee approval or skip credit checks. These phrases are typical of predatory or fake loan offers.",
        patterns=[
            r"\bguaranteed\s+(loan\s+)?(approval|approved|acceptance)\b",
            r"\b(approval|approved)\s+guaranteed\b",
            r"\b(no|without\s+(a|any))\s+credit\s+checks?\b",
            r"\b(approved?|approval)\s+regardless\s+of\s+(your\s+)?(credit|income|history)\b",
            r"\b(bad|poor|no)\s+credit\s+(ok|okay|accepted|welcome|no\s+problem)\b",
            r"\beveryone\s+(is\s+)?approved\b",
            r"\bpre[-\s]?approved\b.{0,60}\bfee\b",
        ],
        weight=20,
        looks_for="'guaranteed approval', 'no credit check', 'approved regardless of credit', 'bad credit OK', 'everyone approved', 'pre-approved' plus a fee",
        icon="🏦",
    ),
    RedFlagCategory(
        name="Celebrity / brand impersonation",
        description="Fake endorsements from public figures or exchanges are a very common lure.",
        patterns=[
            r"\b(elon\s+musk|jeff\s+bezos|warren\s+buffett)\b",
            r"\bendorsed\s+by\b",
            r"\bofficial\s+partner\s+of\b",
            r"\bas\s+seen\s+on\s+(cnbc|forbes|bbc)\b",
        ],
        weight=16,
        looks_for="Elon Musk / Jeff Bezos / Warren Buffett, 'endorsed by', 'official partner of', 'as seen on CNBC/Forbes/BBC'",
        icon="🎭",
    ),
    RedFlagCategory(
        name="Anonymous / unverifiable operators",
        description="No verifiable team, company registration, or track record.",
        patterns=[
            r"\banonymous\s+team\b",
            r"\bteam\s+wishes\s+to\s+remain\s+anonymous\b",
            r"\bno\s+kyc\s+required\b",
        ],
        weight=12,
        looks_for="'anonymous team', 'team wishes to remain anonymous', 'no KYC required'",
        icon="👤",
    ),
    RedFlagCategory(
        name="Suspicious links / payment channels",
        description="Banks, lenders and exchanges only take payments and logins through their own "
                    "official website or app. Lookalike addresses, throwaway domain endings, link "
                    "shorteners and 'external portals' are how scammers collect money and passwords.",
        patterns=[
            r"\b(secure\s+)?external\s+(payment\s+)?(portal|link|site|website|page)\b",
            r"\b(click|tap|visit|use|follow|open)\b.{0,30}\b(link|portal)\b.{0,40}\b(pay|settle|verify|confirm|update|unlock|claim|restore)\w*\b",
            r"\b(pay|settle|verify|confirm|update|restore)\w*\b.{0,40}\b(via|through|using|at)\s+(the|this|our)\s+(secure\s+)?(link|portal)\b",
            r"\b(log\s*in|sign\s*in)\s+(through|via|using)\s+the\s+(secure\s+)?link\b",
        ],
        weight=24,
        looks_for="lookalike addresses (e.g. a bank's name on a site that isn't the bank's), endings like "
                  ".top/.xyz/.info, link shorteners (bit.ly), raw IP links, bait-word addresses "
                  "('loan-settlement', 'secure-verify'), 'pay through our secure external portal'",
        icon="🔗",
        checker=check_links,
        finance_specific=False,
    ),
    RedFlagCategory(
        name="Threats / intimidation to force payment",
        description="Scammers threaten arrest, repossession, legal action or account closure so you pay "
                    "before thinking. Real lenders do warn about missed payments, so this is weighted "
                    "lightly on its own — it matters most alongside a suspicious link or payment request.",
        patterns=[
            r"\bfinal\s+(demand|warning)\b",
            r"\bdo\s+not\s+ignore\b",
            r"\b(immediate|urgent)\s+legal\s+action\b",
            r"\b(warrant\s+for\s+your\s+arrest|arrest\s+warrant|you\s+will\s+be\s+arrested)\b",
            r"\b(police|sheriff|court\s+officers?|bailiffs?)\b.{0,40}\b(attend|visit|come\s+to|arrest)\w*\b",
            r"\b(asset|vehicle|car|property|home)\s+repossession\b",
            r"\brepossess(ed|ion)?\b.{0,30}\b(immediately|today|within)\b",
            r"\bseverely\s+damag\w*\s+your\s+credit\b",
            r"\b(pay|settle|transfer)\w*\b.{0,40}\b(immediately|instantly|right\s+now)\b",
            r"\b(pay|act)\s+(now|today|immediately)\b.{0,30}\bto\s+avoid\b",
            r"\bwithin\s+\d+\s+(hours?|minutes?)\b.{0,40}\b(or|to\s+avoid|otherwise)\b",
            r"\b(account|card|wallet|funds)\b.{0,20}\b(will\s+be|has\s+been|have\s+been|is|are)\s+(suspended|frozen|locked|closed|terminated|blocked)\b",
        ],
        weight=14,
        looks_for="'final demand', 'do not ignore', 'immediate legal action', arrest or police threats, "
                  "'asset repossession', 'severely damaging your credit', 'pay immediately/instantly', "
                  "'within 24 hours or…', 'your account has been suspended/frozen'",
        icon="⚠️",
        finance_specific=False,
    ),
]


@dataclass
class RuleMatch:
    category: RedFlagCategory
    matched_phrases: list[str]


@dataclass
class RuleResult:
    score: float                     # 0-100
    matches: list[RuleMatch] = field(default_factory=list)
    raw_score: float = 0.0           # sum of matched weights before the 100 cap


def run_rule_engine(text: str) -> RuleResult:
    """Scan text against every red-flag category and produce a weighted score."""
    text_l = text.lower()
    matches: list[RuleMatch] = []
    raw_score = 0.0

    for cat in RED_FLAG_CATEGORIES:
        found = []
        for pat in cat.patterns:
            for m in re.finditer(pat, text_l, flags=re.IGNORECASE):
                snippet = text[m.start():m.end()]
                if snippet not in found:
                    found.append(snippet)
        if cat.checker:
            found.extend(f for f in cat.checker(text) if f not in found)
        if found:
            matches.append(RuleMatch(category=cat, matched_phrases=found))
            raw_score += cat.weight

    score = min(100.0, raw_score)
    return RuleResult(score=score, matches=matches, raw_score=raw_score)


# ---------------------------------------------------------------------------
# 2. HF zero-shot semantic layer (the AI/ML transformation)
# ---------------------------------------------------------------------------

# Step a: topic check. Explicit non-finance topics give the model somewhere
# else to put its confidence; with a bare "something else" option it rated
# sports news and recipes as finance.
FINANCE_TOPICS = [
    "investing, shares or trading",
    "cryptocurrency",
    "a loan, credit card or borrowing money",
    "a bank account or personal finance",
]
OTHER_TOPICS = [
    "a parcel delivery",
    "food or cooking",
    "sport",
    "a personal chat with family or friends",
    "a subscription, streaming or online account",
    "health or medical appointments",
    "government services or council news",
    "a bill, fine, toll or utility payment",
    "social plans or entertainment",
]
TOPIC_TEMPLATE = "This text is about {}."
TOPIC_THRESHOLD = 30   # % of topic confidence on finance topics to count as in scope

# Step b: scam signs, each scored independently (multi-label). Wordings were
# chosen on sample_listings.csv plus extra examples so that each sign stays
# near zero on legitimate finance text; broader phrasings (e.g. "pressures
# the reader to act urgently", "no credit check") fired on news and on real
# unsecured-loan ads.
SCAM_SIGNALS = {
    "Guaranteed / unrealistic returns": "promises guaranteed, risk-free or unrealistically high returns",
    # "...buy or join immediately before it is too late" read a bank's "contact us
    # early" hardship message as pressure (40%), so the wording names the sale.
    "Pressure to buy or join now": "urges the reader to buy, invest or join immediately before they miss out",
    "Send crypto to get more back": "asks the reader to send cryptocurrency to receive more back",
    "Asks for recovery phrase / keys": "asks the reader to share their recovery phrase, password or private key",
    "Loan approved without credit check": "says a loan is approved without checking the borrower's credit",
    "Rewards for recruiting": "rewards people for recruiting other people into the scheme",
    # Debt-collection / account scams. A broader "tells the reader to pay through a
    # link" sign fired at 99% on genuine bank reminders, so links are left to the
    # rule engine and these two target the threat itself.
    "Threatens legal action to force payment": "threatens the reader with legal action, arrest or repossession to force an urgent payment",
    "Threatens to freeze or close account": "threatens to freeze, suspend or close the reader's account unless they act within hours",
}
SIGNAL_TEMPLATE = "This text {}."


@dataclass
class SemanticResult:
    finance_score: float               # 0-100: confidence the text is finance-related
    top_topic: str                     # the single most likely topic
    in_scope: bool
    signals: dict[str, float] = field(default_factory=dict)   # sign name -> 0-100
    scam_score: Optional[float] = None  # strongest sign; None when out of scope


@lru_cache(maxsize=1)
def _load_classifier():
    """
    Lazily load the HF zero-shot-classification pipeline.
    Cached so the (large) model is only loaded once per process.
    Returns None if it can't be loaded (no internet / not installed) —
    the app then falls back to the rule-based score alone.
    """
    try:
        from transformers import pipeline
        clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return clf
    except Exception:
        return None


def run_semantic_classifier(text: str) -> Optional[SemanticResult]:
    """
    Topic check, then (if in scope) scam-sign check.
    Returns None if the model is unavailable in this environment.
    """
    clf = _load_classifier()
    if clf is None:
        return None
    try:
        topic = clf(text, candidate_labels=FINANCE_TOPICS + OTHER_TOPICS,
                    hypothesis_template=TOPIC_TEMPLATE)
        topic_probs = dict(zip(topic["labels"], topic["scores"]))
        finance_score = round(sum(topic_probs[t] for t in FINANCE_TOPICS) * 100, 1)
        result = SemanticResult(
            finance_score=finance_score,
            top_topic=topic["labels"][0],
            in_scope=finance_score >= TOPIC_THRESHOLD,
        )
        if not result.in_scope:
            return result

        signs = clf(text, candidate_labels=list(SCAM_SIGNALS.values()),
                    hypothesis_template=SIGNAL_TEMPLATE, multi_label=True)
        sign_probs = dict(zip(signs["labels"], signs["scores"]))
        result.signals = {name: round(sign_probs[hyp] * 100, 1) for name, hyp in SCAM_SIGNALS.items()}
        result.scam_score = max(result.signals.values())
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Combine into a final assessment
# ---------------------------------------------------------------------------

# The two layers are combined as a probabilistic OR:
#     final = 1 - (1 - rule/100) * (1 - ai/100)
# Either layer can raise the alarm on its own (a strong AI scam sign isn't
# diluted by a rule score of 0, and vice versa), and when both flag the
# text the score rises further. A weighted average capped AI-only
# detections at 55 and let a weak AI score drag down obvious scams.

HIGH_THRESHOLD = 60
MEDIUM_THRESHOLD = 30

OUT_OF_SCOPE = "Out of scope"


@dataclass
class Assessment:
    text: str
    rule_result: RuleResult
    semantic: Optional[SemanticResult]  # None if AI layer off/unavailable
    semantic_score: Optional[float]     # strongest AI scam sign; None if not scored
    final_score: Optional[float]        # 0-100; None when out of scope
    verdict: str                        # "Low", "Medium", "High", or OUT_OF_SCOPE
    used_ai: bool
    in_scope: bool                      # always True when the topic wasn't checked


def analyze(text: str, use_ai: bool = True) -> Assessment:
    text = (text or "").strip()
    rule_result = run_rule_engine(text)

    semantic = run_semantic_classifier(text) if (use_ai and text) else None

    # A finance-specific rule match also counts as in scope.
    finance_hit = any(m.category.finance_specific for m in rule_result.matches)
    in_scope = semantic is None or semantic.in_scope or finance_hit
    if not in_scope:
        return Assessment(
            text=text,
            rule_result=rule_result,
            semantic=semantic,
            semantic_score=None,
            final_score=None,
            verdict=OUT_OF_SCOPE,
            used_ai=True,
            in_scope=False,
        )

    semantic_score = semantic.scam_score if semantic is not None else None
    if semantic_score is not None:
        final_score = 100 * (1 - (1 - rule_result.score / 100) * (1 - semantic_score / 100))
        used_ai = True
    else:
        final_score = rule_result.score
        used_ai = False

    final_score = round(min(100.0, final_score), 1)

    if final_score >= HIGH_THRESHOLD:
        verdict = "High"
    elif final_score >= MEDIUM_THRESHOLD:
        verdict = "Medium"
    else:
        verdict = "Low"

    return Assessment(
        text=text,
        rule_result=rule_result,
        semantic=semantic,
        semantic_score=semantic_score,
        final_score=final_score,
        verdict=verdict,
        used_ai=used_ai,
        in_scope=True,
    )