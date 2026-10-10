"""
ScamShield — Investment & Loan Scam Detector
=====================================================
AI for Finance — Build Project

Input:  a crypto/stock investment listing, loan or credit offer, ad, or pitch (pasted text,
        or a CSV of many listings for batch mode)
Transformation: a hybrid rule-based red-flag engine + an HF zero-shot
        LLM classifier (facebook/bart-large-mnli) judge how likely the
        text is to be a scam
Output: a risk score (0-100), a Low/Medium/High verdict, and the exact
        evidence (matched phrases + category explanations) behind it —
        so a user can verify the call rather than just trust a label
"""

import html
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scam_detector import (
    FINANCE_TOPICS,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    OUT_OF_SCOPE,
    RED_FLAG_CATEGORIES,
    TOPIC_THRESHOLD,
    analyze,
)

# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ScamShield — Scam Listing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0B1120",
    "bg_card": "#141B2E",
    "bg_card_alt": "#1A2338",
    "border": "#26314A",
    "text": "#E8ECF4",
    "text_dim": "#8B94AC",
    "accent": "#7C5CFF",
    "accent_soft": "#7C5CFF22",
    "low": "#2DD4A7",
    "medium": "#F5B942",
    "high": "#FF5C7A",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 10% 0%, #151E36 0%, {PALETTE['bg']} 55%);
    color: {PALETTE['text']};
}}

h1, h2, h3, .hero-title {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

.hero-title {{
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7C5CFF 0%, #2DD4A7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}}

.hero-sub {{
    color: {PALETTE['text_dim']};
    font-size: 1.02rem;
    margin-bottom: 1.6rem;
}}

.card {{
    background: {PALETTE['bg_card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}}

.flag-card {{
    background: {PALETTE['bg_card_alt']};
    border: 1px solid {PALETTE['border']};
    border-left: 4px solid {PALETTE['high']};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
}}

.flag-title {{
    font-weight: 600;
    font-size: 0.98rem;
    color: {PALETTE['text']};
}}

.clear-card {{
    background: {PALETTE['bg_card_alt']};
    border: 1px solid {PALETTE['border']};
    border-left: 4px solid {PALETTE['low']};
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.5rem;
}}

.flag-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.8rem;
}}

.flag-points {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    white-space: nowrap;
}}

.section-label {{
    color: {PALETTE['text_dim']};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.1rem 0 0.5rem 0;
}}

.prob-row {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-top: 0.55rem;
    font-size: 0.85rem;
}}

.prob-name {{ flex: 0 0 46%; color: {PALETTE['text_dim']}; }}
.prob-track {{ flex: 1; height: 8px; background: {PALETTE['border']}; border-radius: 999px; overflow: hidden; }}
.prob-fill {{ height: 100%; border-radius: 999px; }}
.prob-val {{ flex: 0 0 3.2rem; text-align: right; font-family: 'Space Grotesk', sans-serif; font-weight: 600; }}

.calc-line {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    margin: 0.3rem 0;
}}

.flag-desc {{
    color: {PALETTE['text_dim']};
    font-size: 0.85rem;
    margin-top: 0.15rem;
}}

.phrase-pill {{
    display: inline-block;
    background: {PALETTE['high']}22;
    color: {PALETTE['high']};
    border: 1px solid {PALETTE['high']}55;
    border-radius: 10px;
    padding: 0.15rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.3rem 0 0;
    font-family: 'Space Grotesk', sans-serif;
}}

.badge {{
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
}}

.badge-low {{ background: {PALETTE['low']}22; color: {PALETTE['low']}; border: 1px solid {PALETTE['low']}66; }}
.badge-medium {{ background: {PALETTE['medium']}22; color: {PALETTE['medium']}; border: 1px solid {PALETTE['medium']}66; }}
.badge-scope {{ background: {PALETTE['text_dim']}22; color: {PALETTE['text_dim']}; border: 1px solid {PALETTE['text_dim']}66; }}
.badge-high {{ background: {PALETTE['high']}22; color: {PALETTE['high']}; border: 1px solid {PALETTE['high']}66; }}

.metric-label {{
    color: {PALETTE['text_dim']};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.metric-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}}

section[data-testid="stSidebar"] {{
    background: {PALETTE['bg_card']};
    border-right: 1px solid {PALETTE['border']};
}}

.stTextArea textarea {{
    background: {PALETTE['bg_card_alt']} !important;
    color: {PALETTE['text']} !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}}

.stButton>button {{
    background: linear-gradient(90deg, #7C5CFF 0%, #6247D9 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    padding: 0.55rem 1.6rem;
}}

.stButton>button:hover {{
    background: linear-gradient(90deg, #8E72FF 0%, #7458EE 100%);
    color: white;
}}

.stDataFrame {{ border-radius: 10px; overflow: hidden; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


VERDICT_COLORS = {"Low": PALETTE["low"], "Medium": PALETTE["medium"], "High": PALETTE["high"],
                  OUT_OF_SCOPE: PALETTE["text_dim"]}
VERDICT_CLASS = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high",
                 OUT_OF_SCOPE: "badge-scope"}

EXAMPLES = {
    "— choose an example —": "",
    "🚩 Obvious crypto scam": (
        "GUARANTEED 500% RETURN in 7 days! Our AI trading bot is endorsed by Elon Musk. "
        "Act now, only 12 spots left! Refer 3 friends and earn extra bonus tokens. "
        "To activate your account, send 0.05 BTC to the address below. No risk, 100% guaranteed profit."
    ),
    "🚩 Subtle pyramid pitch": (
        "Join our exclusive wealth circle today. Bring 2 people into the program and your "
        "downline earns you passive income for life. Limited spots open this week only — "
        "don't miss this once in a lifetime opportunity to change your financial future."
    ),
    "✅ Legitimate seed round": (
        "We are raising a seed round for our fintech startup. Our team has 8 years of "
        "experience in payments infrastructure. Past performance is not indicative of "
        "future results, and investing carries risk of loss. Please review our audited "
        "financials before investing."
    ),
    "🚩 Advance-fee loan scam": (
        "Guaranteed approval on loans up to $20,000 — no credit check, bad credit OK! "
        "Just pay a small insurance fee upfront and we release your funds within the hour."
    ),
    "✅ Legitimate personal loan": (
        "Borrow up to $50,000 for any worthwhile purchase and pay no establishment or ongoing "
        "fees without providing collateral. You can also benefit from a more competitive "
        "interest rate if you have a strong credit score."
    ),
    "🚩 Fake debt-collection email": (
        "Subject: URGENT: Notice of Default and Immediate Legal Action - Account Ref: LN-994012\n"
        "From: Legal Recovery Department <collections@macquarie-finance-group.net>\n\n"
        "Dear Borrower,\nOur records indicate that your monthly personal loan repayment of $612.40 "
        "is now 45 days past due, constituting a material breach of your loan contract.\n\n"
        "Failure to settle the outstanding balance immediately will result in an automated adverse "
        "report to Equifax and Illion credit bureaus, severely damaging your credit rating. To avoid "
        "immediate asset repossession and legal debt collection costs, you must pay the arrears "
        "instantly using our secure external portal:\n\n--> https://loan-settlement.top\n\n"
        "Do not ignore this notice. This is a final demand."
    ),
    "✅ Genuine overdue notice": (
        "Final notice: your Latitude credit card account is 60 days overdue. Unless payment is "
        "received, we may refer the account to a collection agency and list a default with Equifax. "
        "Log in at latitudefinancial.com.au or call 13 28 26 to arrange payment or discuss hardship options."
    ),
    "🚫 Off-topic (delivery text)": (
        "AusPost: Your package could not be delivered to your address due to an unpaid customs "
        "handling fee of $3.45. To update your address details and settle the fee to schedule "
        "redelivery, please visit: https://australiapost-redelivery.info"
    ),
    "✅ Regular market commentary": (
        "The ASX 200 closed slightly higher today as mining stocks rallied on stronger "
        "iron ore prices. Analysts expect volatility to continue ahead of next week's "
        "RBA rate decision."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gauge_figure(score: float, verdict: str) -> go.Figure:
    color = VERDICT_COLORS[verdict]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 34, "color": PALETTE["text"], "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": PALETTE["text_dim"], "tickfont": {"color": PALETTE["text_dim"]}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": PALETTE["bg_card_alt"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": PALETTE["low"] + "22"},
                    {"range": [30, 60], "color": PALETTE["medium"] + "22"},
                    {"range": [60, 100], "color": PALETTE["high"] + "22"},
                ],
            },
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": PALETTE["text"]},
    )
    return fig


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _prob_row(label: str, value: float, color: str) -> str:
    return (
        f"<div class='prob-row'><div class='prob-name'>{_esc(label)}</div>"
        f"<div class='prob-track'><div class='prob-fill' style='width:{value:.0f}%; background:{color};'></div></div>"
        f"<div class='prob-val' style='color:{color};'>{value:.0f}%</div></div>"
    )


def render_summary(assessment):
    color = VERDICT_COLORS[assessment.verdict]
    if not assessment.in_scope:
        sem = assessment.semantic
        general = [m.category.name for m in assessment.rule_result.matches]
        warn = (f"<br><br><b style='color:{PALETTE['high']};'>⚠️ Still, general warning signs were found:</b> "
                f"{_esc(', '.join(general))} — see the details below and treat this message with caution."
                if general else "")
        st.markdown(
            f"<div class='card'><b style='color:{color};'>Out of scope.</b> This text doesn't look like an "
            f"investment, crypto, loan or other finance offer (the AI topic check puts it at "
            f"<b>{sem.finance_score:.0f}%</b> finance-related; most likely topic: <i>{_esc(sem.top_topic)}</i>). "
            f"ScamShield is only built to judge finance, loan and crypto text, so it gives no risk score "
            f"here — that doesn't mean the message is safe.{warn}</div>",
            unsafe_allow_html=True,
        )
        return

    n_total = len(RED_FLAG_CATEGORIES)
    n_hit = len(assessment.rule_result.matches)
    n_phrases = sum(len(m.matched_phrases) for m in assessment.rule_result.matches)
    if n_hit:
        rule_part = (f"<b>{n_hit} of {n_total}</b> red-flag categories matched "
                     f"({n_phrases} phrase{'s' if n_phrases != 1 else ''})")
    else:
        rule_part = f"<b>0 of {n_total}</b> red-flag categories matched"
    if assessment.semantic_score is not None:
        ai_part = f"the AI model's strongest scam sign scores <b>{assessment.semantic_score:.0f}%</b>"
    else:
        ai_part = "the AI model was not used"
    st.markdown(
        f"<div class='card'>{rule_part}, and {ai_part} — giving a final score of "
        f"<b style='color:{color};'>{assessment.final_score:.0f}/100 ({assessment.verdict} risk)</b>.</div>",
        unsafe_allow_html=True,
    )


def render_rule_checks(assessment):
    """Every category, matched or not, so the user sees exactly what was checked."""
    hits = {m.category.name: m for m in assessment.rule_result.matches}
    st.markdown(
        f"<div class='section-label'>Rule engine — {len(RED_FLAG_CATEGORIES)} categories checked</div>",
        unsafe_allow_html=True,
    )

    # Matched categories first, heaviest first, then the ones that came back clear.
    ordered = sorted(RED_FLAG_CATEGORIES, key=lambda c: (c.name not in hits, -c.weight))
    for cat in ordered:
        m = hits.get(cat.name)
        if m:
            pills = "".join(f"<span class='phrase-pill'>{_esc(p)}</span>" for p in m.matched_phrases)
            st.markdown(
                f"""<div class="flag-card">
                        <div class="flag-head">
                            <div class="flag-title">{cat.icon} {_esc(cat.name)}</div>
                            <div class="flag-points" style="color:{PALETTE['high']};">+{cat.weight:.0f} pts</div>
                        </div>
                        <div class="flag-desc">{_esc(cat.description)}</div>
                        <div class="flag-desc" style="margin-top:0.45rem;">Matched in your text:</div>
                        <div>{pills}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="clear-card">
                        <div class="flag-head">
                            <div class="flag-title">{cat.icon} {_esc(cat.name)}</div>
                            <div class="flag-points" style="color:{PALETTE['low']};">✓ clear · 0 pts</div>
                        </div>
                        <div class="flag-desc">Checked for: {_esc(cat.looks_for)}</div>
                    </div>""",
                unsafe_allow_html=True,
            )


def render_ai_judgement(assessment, ai_requested: bool):
    st.markdown("<div class='section-label'>AI semantic model — facebook/bart-large-mnli</div>",
                unsafe_allow_html=True)

    sem = assessment.semantic
    if sem is None:
        reason = ("Turned off in the sidebar, so only the rule engine was used and the topic wasn't checked."
                  if not ai_requested else
                  "The model couldn't be loaded in this environment (no internet or "
                  "<code>transformers</code> not installed), so only the rule engine was used.")
        st.markdown(f"<div class='card flag-desc'>{reason}</div>", unsafe_allow_html=True)
        return

    scope_color = PALETTE["low"] if sem.in_scope else PALETTE["text_dim"]
    topic_html = (
        f"<div class='flag-desc'><b>Step 1 — topic check.</b> Is this about "
        f"{', '.join(_esc(t) for t in FINANCE_TOPICS)}? Needs ≥ {TOPIC_THRESHOLD}% to be scored.</div>"
        f"{_prob_row('Finance-related', sem.finance_score, scope_color)}"
        f"<div class='flag-desc' style='margin-top:0.4rem;'>Most likely topic: <i>{_esc(sem.top_topic)}</i></div>"
    )

    if not sem.in_scope:
        if assessment.in_scope:
            step2 = ("<b>Step 2 — scam signs:</b> skipped. The topic check didn't see finance, but the rule "
                     "engine matched finance-specific red flags, so the text is scored on those rules alone.")
        else:
            step2 = "<b>Step 2 — scam signs:</b> skipped, because the text is out of scope."
        st.markdown(
            f"<div class='card'>{topic_html}<div class='flag-desc' style='margin-top:0.8rem;'>{step2}</div></div>",
            unsafe_allow_html=True,
        )
        return

    rows = "".join(
        _prob_row(name, val, PALETTE["high"] if val >= HIGH_THRESHOLD
                  else PALETTE["medium"] if val >= MEDIUM_THRESHOLD else PALETTE["low"])
        for name, val in sorted(sem.signals.items(), key=lambda kv: -kv[1])
    )
    scam = sem.scam_score
    top_sign = max(sem.signals, key=sem.signals.get)
    if scam >= HIGH_THRESHOLD:
        reading = (f"The model sees a strong scam sign — <b>{_esc(top_sign)}</b> — in the overall meaning "
                   f"of the text, even where the exact wording doesn't match a rule.")
    elif scam >= MEDIUM_THRESHOLD:
        reading = (f"The model sees a weak-to-moderate hint of <b>{_esc(top_sign)}</b>, "
                   f"but nothing it's confident about.")
    else:
        reading = "None of the scam signs fit this text — it reads like a normal finance product, investment or news item."

    st.markdown(
        f"""<div class="card">
                {topic_html}
                <div class="flag-desc" style="margin-top:1rem;"><b>Step 2 — scam signs.</b> The model tests
                each sign separately; the AI score is the strongest one.</div>
                {rows}
                <div class="flag-desc" style="margin-top:0.8rem;">{reading}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_calculation(assessment):
    if not assessment.in_scope:
        return
    st.markdown("<div class='section-label'>How the final score was calculated</div>",
                unsafe_allow_html=True)
    rr = assessment.rule_result
    if rr.matches:
        if len(rr.matches) > 1:
            parts = " + ".join(f"{m.category.weight:.0f}" for m in rr.matches)
            rule_line = f"Rule score = {parts} = {rr.raw_score:.0f}"
        else:
            rule_line = f"Rule score = {rr.raw_score:.0f}"
        if rr.raw_score > 100:
            rule_line += " → capped at 100"
    else:
        rule_line = "Rule score = 0 (no categories matched)"

    if assessment.used_ai:
        ai = assessment.semantic_score
        final_line = (f"Final = 100 × [1 − (1 − {rr.score:.0f}/100) × (1 − {ai:.0f}/100)] = "
                      f"<b>{assessment.final_score:.1f}</b>")
        explain = ("Either layer can raise the alarm on its own, and when both flag the text "
                   "the score goes higher than either alone.")
    else:
        final_line = f"Final = rule score = <b>{assessment.final_score:.1f}</b>"
        explain = "The AI score wasn't available, so the rule score is used alone."

    color = VERDICT_COLORS[assessment.verdict]
    st.markdown(
        f"""<div class="card">
                <div class="calc-line">{rule_line}</div>
                <div class="calc-line">AI score = {f"{assessment.semantic_score:.0f} (strongest scam sign)" if assessment.used_ai else "—"}</div>
                <div class="calc-line">{final_line}</div>
                <div class="flag-desc" style="margin-top:0.4rem;">{explain}</div>
                <div class="flag-desc" style="margin-top:0.6rem;">
                    Verdict bands: Low &lt; {MEDIUM_THRESHOLD} · Medium {MEDIUM_THRESHOLD}–{HIGH_THRESHOLD - 1} ·
                    High ≥ {HIGH_THRESHOLD} → <b style="color:{color};">{assessment.verdict} risk</b>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_evidence(assessment, ai_requested: bool):
    render_summary(assessment)
    render_rule_checks(assessment)
    render_ai_judgement(assessment, ai_requested)
    render_calculation(assessment)


def render_single_result(assessment, ai_requested: bool):
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        badge_cls = VERDICT_CLASS[assessment.verdict]
        if assessment.in_scope:
            st.plotly_chart(gauge_figure(assessment.final_score, assessment.verdict), width="stretch")
            badge_text = f"{assessment.verdict.upper()} RISK"
        else:
            st.markdown(
                f"<div class='card' style='text-align:center; padding:2.2rem 1rem;'>"
                f"<div style='font-size:2.4rem;'>🚫</div>"
                f"<div class='metric-value' style='color:{PALETTE['text_dim']};'>Not scored</div>"
                f"<div class='flag-desc'>Not finance, loan or crypto related</div></div>",
                unsafe_allow_html=True,
            )
            badge_text = OUT_OF_SCOPE.upper()
        st.markdown(
            f"<div style='text-align:center; margin-top:-1rem;'>"
            f"<span class='badge {badge_cls}'>{badge_text}</span></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"<div class='card'><div class='metric-label'>Rule-based score</div>"
                f"<div class='metric-value'>{assessment.rule_result.score:.0f}</div></div>",
                unsafe_allow_html=True,
            )
        with c2:
            ai_val = f"{assessment.semantic_score:.0f}" if assessment.semantic_score is not None else "—"
            if assessment.semantic is None:
                note = "AI model off or unavailable — rule score used alone"
            elif not assessment.semantic.in_scope:
                note = "Skipped — topic not finance-related"
            else:
                note = ""
            note_html = f"<br><span style='font-size:0.72rem;color:#8B94AC;'>{note}</span>" if note else ""
            st.markdown(
                f"<div class='card'><div class='metric-label'>AI semantic score</div>"
                f"<div class='metric-value'>{ai_val}</div>{note_html}</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("##### Evidence detected")
        render_evidence(assessment, ai_requested)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛡️ ScamShield")
    st.markdown(
        "<span style='color:#8B94AC; font-size:0.88rem;'>"
        "Paste any crypto or stock investment listing, or a loan or credit offer, and get an evidence-backed "
        "scam risk assessment — not just a label.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    use_ai = st.toggle("Use AI semantic model", value=True, help="Runs a HuggingFace zero-shot classifier (facebook/bart-large-mnli) alongside the rule engine: first a topic check (finance, loans, crypto only), then a scam-sign check. Requires the model to be downloadable in this environment.")
    st.markdown("---")
    st.markdown("##### Red-flag categories tracked")
    for cat in RED_FLAG_CATEGORIES:
        st.markdown(f"{cat.icon} **{cat.name}**")
    st.markdown("---")
    st.caption("Scope: investments, crypto, loans and credit. Other text is reported as out of scope.")
    st.caption("Built for the AI for Finance Build Project. Not financial advice.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("<div class='hero-title'>ScamShield</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-sub'>An AI-assisted second opinion before you invest in a listing or sign up for a loan offer.</div>",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_about = st.tabs(["🔍 Analyze a listing", "📂 Batch scan (CSV)", "ℹ️ About this build"])

# ---------------------------------------------------------------------------
# Tab 1: single listing
# ---------------------------------------------------------------------------

with tab_single:
    example_choice = st.selectbox("Try an example", list(EXAMPLES.keys()))
    default_text = EXAMPLES[example_choice]

    listing_text = st.text_area(
        "Paste the listing, pitch, or ad text",
        value=default_text,
        height=160,
        placeholder="Paste a Telegram post, tweet, ad copy, or whitepaper excerpt here...",
    )

    analyze_clicked = st.button("Analyze listing", type="primary")

    if analyze_clicked:
        if not listing_text.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Scanning for red flags and running the AI model..."):
                result = analyze(listing_text, use_ai=use_ai)
            st.markdown("<br>", unsafe_allow_html=True)
            render_single_result(result, ai_requested=use_ai)

# ---------------------------------------------------------------------------
# Tab 2: batch CSV
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown(
        "Upload a CSV with a `text` column — one listing per row — to score many at once. "
        "Useful for scanning a scraped batch of ads or social posts."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            df = None

        if df is not None:
            if "text" not in df.columns:
                st.error("CSV must have a column named 'text'.")
            else:
                run_batch = st.button("Run batch scan", type="primary")
                if run_batch:
                    progress = st.progress(0, text="Scoring listings...")
                    scores, verdicts, flag_counts, ai_scores, finance_scores = [], [], [], [], []
                    n = len(df)
                    for i, row in enumerate(df["text"].astype(str)):
                        r = analyze(row, use_ai=use_ai)
                        scores.append(r.final_score)
                        verdicts.append(r.verdict)
                        flag_counts.append(len(r.rule_result.matches))
                        ai_scores.append(r.semantic_score)
                        finance_scores.append(r.semantic.finance_score if r.semantic else None)
                        progress.progress((i + 1) / n, text=f"Scoring listings... {i+1}/{n}")
                    progress.empty()

                    out = df.copy()
                    out["risk_score"] = scores
                    out["verdict"] = verdicts
                    out["red_flag_categories"] = flag_counts
                    out["ai_semantic_score"] = ai_scores
                    out["finance_topic_pct"] = finance_scores

                    def _color_verdict(v):
                        c = VERDICT_COLORS.get(v, PALETTE["text"])
                        return f"background-color: {c}22; color: {c}; font-weight: 600;"

                    st.markdown(f"**{n} listings scored** — {sum(1 for v in verdicts if v=='High')} high risk, "
                                f"{sum(1 for v in verdicts if v=='Medium')} medium, "
                                f"{sum(1 for v in verdicts if v=='Low')} low, "
                                f"{sum(1 for v in verdicts if v==OUT_OF_SCOPE)} out of scope (not finance-related).")

                    st.dataframe(
                        out.style.map(_color_verdict, subset=["verdict"]),
                        width="stretch",
                        height=420,
                    )

                    csv_buf = io.StringIO()
                    out.to_csv(csv_buf, index=False)
                    st.download_button(
                        "Download results as CSV",
                        data=csv_buf.getvalue(),
                        file_name="scamshield_batch_results.csv",
                        mime="text/csv",
                    )
    else:
        st.caption("Example: a CSV with columns `id,text` where each row is one scraped listing.")

# ---------------------------------------------------------------------------
# Tab 3: about / methodology
# ---------------------------------------------------------------------------

with tab_about:
    st.markdown(
        """
##### How it works

**Input** → the raw text of a listing, ad, or pitch.

**Scope** → investments and trading, crypto, loans and credit, and personal finance.
Anything else (delivery texts, streaming accounts, personal messages…) is reported as
**Out of scope** with no risk score, rather than a misleading "Low risk".

**Transformation** → two layers combine into one risk score:
1. A rule-based red-flag engine checks the text against 10 documented scam categories
   (guaranteed returns, urgency pressure, referral/pyramid structure, upfront-fee or
   private-key requests, celebrity impersonation, loan/credit red flags such as
   "guaranteed approval" or "no credit check", and more). Two of them cover
   debt-collection and account scams:
   - **Suspicious links** — every web address and email domain is pulled out and
     checked: does it use a bank's or exchange's name without being its official site
     (e.g. `macquarie-finance-group.net`)? Does it end in `.top`, `.xyz`, `.info`…? Is it a
     link shortener, a raw IP address, or built from bait words (`loan-settlement`)?
   - **Threats** — "final demand", "immediate legal action", repossession or arrest
     threats, "pay instantly", "account suspended". Real lenders also warn about missed
     payments, so this is weighted lightly on its own.
2. An HF zero-shot LLM classifier (`facebook/bart-large-mnli`) runs two steps:
   a **topic check** (is this finance, loan or crypto text at all?), then a
   **scam-sign check** that tests the meaning of the text against eight specific signs —
   guaranteed or unrealistic returns, pressure to buy or join now, send crypto to get
   more back, requests for a recovery phrase or keys, loans approved without a credit
   check, rewards for recruiting, threats of legal action to force payment, and threats
   to freeze or close an account. The strongest sign is the AI score, so it catches
   scams that avoid the literal red-flag phrasing.

The final score is `100 × [1 − (1 − rule/100) × (1 − AI/100)]`: either layer can raise
the alarm on its own, and both agreeing pushes the score higher.

**Output** → a 0–100 risk score, a Low / Medium / High verdict, and a full evidence
breakdown on every run: each category marked as matched (with the exact phrases and
points) or clear (with what was checked), the AI topic check and every scam sign, and the
arithmetic behind the final score — so the user can check the reasoning rather than
just trust a label.

##### Why this design
Purely keyword-based detectors are easy to fool by rewording; a black-box LLM label
with no evidence is hard to trust. Combining both, and always showing the matched
phrases, is the deliberate design choice — it mirrors the brief's own example of
showing supporting evidence alongside an AI answer.

##### Limitations (see the Evaluation report for full test results)
- The rule engine only catches patterns it has been given — novel scam language slips through.
- The AI scam signs were worded and checked on a small set (~30 finance texts); a larger
  labelled test set is needed before trusting the exact numbers.
- The topic check can let finance-adjacent text through (e.g. a toll-bill text) or rule
  out borderline text (e.g. a council grant announcement).
- This is an English-language detector; non-English listings are out of scope.
- Non-finance scams (fake parcel-delivery texts, streaming-account phishing, "hi mum"
  messages) are deliberately out of scope and are not scored.
- Sample check: a legitimate personal-loan ad ("Borrow up to $50,000 … no establishment
  or ongoing fees …") scored Low with 0 rule matches — a correct true negative for a
  legitimate product, including wording like "no fees" and "without collateral" that
  naive keyword rules could misread.
- This is a decision-support tool, not a verdict — it does not replace due diligence.
"""
    )
