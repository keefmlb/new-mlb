"""Streamlit web app for the MLB predictor.

Run:
    streamlit run app.py

Pages:
  - Slate (default): date picker, top value bets, per-game cards, drill-down
"""
from __future__ import annotations
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predict_core, projections as proj, bet_tracker
from src import dispersion as disp_mod

_DISP_PATH = ROOT / "data" / "models" / "dispersion.json"
_TEAM_RUN_DISP = 1.4   # typical MLB over-dispersion for team scoring (var/mean)

_STAT_REL_PATH = ROOT / "data" / "models" / "stat_reliability.json"
try:
    _stat_rel_weights: dict = json.loads(_STAT_REL_PATH.read_text(encoding="utf-8"))
except Exception:
    _stat_rel_weights = {}


# ---------- Page config ----------
st.set_page_config(
    page_title="MLB Predictor",
    page_icon=":baseball:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    /* Tighten the main container padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

    /* Cleaner tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid rgba(120, 120, 120, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 99, 71, 0.10);
        color: #ff6347 !important;
    }

    /* Metric card refinement */
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.75; }

    /* Pick card — prominent top panel */
    .pick-card {
        background: linear-gradient(135deg, rgba(255,99,71,0.08), rgba(255,99,71,0.02));
        border: 1px solid rgba(255,99,71,0.20);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        min-height: 130px;          /* keeps cards uniform height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    /* Bet line: allow wrapping and break-anywhere on long descriptions
       (game totals like 'San Francisco Giants @ Philadelphia Phillies
       Under 8.0' won't fit on one line). Font size scales down via the
       inline style attribute set per-card. */
    .pick-bet {
        font-weight: 600;
        line-height: 1.25;
        word-wrap: break-word;
        overflow-wrap: anywhere;
        hyphens: auto;
    }
    .pick-meta { opacity: 0.75; font-size: 0.85rem; }
    .pick-edge { color: #2ea043; font-weight: 600; font-size: 1.1rem; }

    /* Reduce vertical padding inside expanders */
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }

    /* Subtle dividers */
    hr { margin: 1rem 0; opacity: 0.4; }

    /* Hide the top-right "fork on github" Streamlit footer noise */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- CI helpers ----------
def _ci(mu: float, phi: float, lo: float = 0.10, hi: float = 0.90):
    """80% CI (lo_count, hi_count) for NegBin(mean=mu, var/mean=phi).

    Falls back to normal approximation if scipy is unavailable.
    """
    if mu <= 0:
        return 0, 0
    phi = max(phi, 1.01)
    try:
        from scipy.stats import nbinom
        r = mu / (phi - 1.0)
        p_succ = r / (r + mu)
        return int(nbinom.ppf(lo, r, p_succ)), int(nbinom.ppf(hi, r, p_succ))
    except Exception:
        import math
        std = (phi * mu) ** 0.5
        return max(0, math.floor(mu - 1.28 * std)), math.ceil(mu + 1.28 * std)


def _ci_str(lo: int, hi: int) -> str:
    return f"{lo}–{hi}"


def _p_at_least_one(mu: float, phi: float) -> float:
    """P(X >= 1) = 1 - P(X = 0) under NegBin(mean=mu, var/mean=phi)."""
    if mu <= 0:
        return 0.0
    phi = max(phi, 1.01)
    try:
        from scipy.stats import nbinom
        r = mu / (phi - 1.0)
        p_succ = r / (r + mu)
        return float(1.0 - nbinom.pmf(0, r, p_succ))
    except Exception:
        import math
        return float(1.0 - math.exp(-mu))


@st.cache_data(ttl=3600, show_spinner=False)
def _load_disp_fits():
    return disp_mod.load_fits(_DISP_PATH)


def _batter_ci_rows(batters: list[dict], fits: dict) -> pd.DataFrame:
    """Build a DataFrame of batter projections with 80% CIs for each stat."""
    STAT_MAP = [
        ("H",  "proj_h",    "hits"),
        ("HR", "proj_hr",   "hr"),
        ("TB", "proj_tb",   "tb"),
        ("RBI","proj_rbi",  "rbi"),
        ("R",  "proj_runs", "runs"),
        ("K",  "proj_k",    "k"),
        ("BB", "proj_bb",   "bb"),
    ]
    rows = []
    for b in batters:
        row: dict = {"Player": b.get("name", "?"), "PA": round(b.get("expected_pa", 0), 1)}
        for label, proj_key, disp_key in STAT_MAP:
            mu = b.get(proj_key, 0) or 0
            phi = disp_mod.disp_for(disp_key, mu, fits)
            lo, hi = _ci(mu, phi)
            row[label]          = round(mu, 2)
            row[f"{label} CI"]  = _ci_str(lo, hi)
        rows.append(row)
    return pd.DataFrame(rows)


def _starter_ci_rows(starter: dict | None, fits: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame for a starter with 80% CIs."""
    if not starter:
        return pd.DataFrame()
    STAT_MAP = [
        ("K",   "proj_k",          "pitcher_k"),
        ("IP",  "expected_ip",     "pitcher_outs"),   # dispersion fit on outs; IP = outs/3
        ("ER",  "proj_er",         "pitcher_er"),
        ("H",   "proj_h",          "pitcher_h"),
        ("BB",  "proj_bb",         "pitcher_bb"),
        ("HR",  "proj_hr_allowed", "pitcher_hr"),
    ]
    row: dict = {"Pitcher": starter.get("name", "?")}
    for label, proj_key, disp_key in STAT_MAP:
        mu = starter.get(proj_key, 0) or 0
        if disp_key == "pitcher_outs":
            # dispersion fit on expected_outs; convert to IP for display
            mu_outs = mu * 3
            phi = disp_mod.disp_for(disp_key, mu_outs, fits)
            lo_o, hi_o = _ci(mu_outs, phi)
            row[label]         = round(mu, 2)
            row[f"{label} CI"] = _ci_str(round(lo_o / 3, 1), round(hi_o / 3, 1))
        else:
            phi = disp_mod.disp_for(disp_key, mu, fits)
            lo, hi = _ci(mu, phi)
            row[label]         = round(mu, 2)
            row[f"{label} CI"] = _ci_str(lo, hi)
    return pd.DataFrame([row])


# ---------- Formatting helpers ----------
def _amer(v):
    if v is None:
        return "?"
    return f"+{v}" if v > 0 else str(v)


def _stake_dollars(kelly: float, bankroll: float, kelly_frac: float) -> float:
    """Recommended stake = bankroll × full-Kelly fraction × the chosen Kelly
    fraction. The bet's `kelly` field is already the (capped) full-Kelly
    fraction of bankroll."""
    return max(0.0, float(bankroll) * float(kelly) * float(kelly_frac))


def _render_value_df(rows: list[dict], bankroll: float = 0.0,
                     kelly_frac: float = 0.0) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    # Recommended stake / to-win (before we string-format odds)
    if bankroll and kelly_frac:
        _stake = df["kelly"].apply(lambda k: _stake_dollars(k, bankroll, kelly_frac))
        _dec = df["odds"].apply(lambda o: (1.0 + o / 100.0) if o > 0 else (1.0 + 100.0 / (-o)))
        df["Stake"] = _stake.apply(lambda s: f"${s:,.0f}")
        df["To win"] = (_stake * (_dec - 1.0)).apply(lambda w: f"${w:,.0f}")
    df["odds"] = df["odds"].apply(_amer)
    df["model"] = (df["model_prob"] * 100).round(1).astype(str) + "%"
    df["no-vig"] = (df["novig_prob"] * 100).round(1).astype(str) + "%"
    df["edge"] = "+" + df["edge_pct"].round(1).astype(str) + "%"
    df["EV/$"] = df["ev_per_dollar"].apply(lambda x: f"{x:+.3f}")
    df["Kelly"] = (df["kelly"] * 100).round(2).astype(str) + "%"
    if "score" in df.columns:
        df["Score"] = df["score"].round(2)
    cols = ["description", "odds", "model", "no-vig", "edge"]
    if "Score" in df.columns:
        cols.append("Score")
    cols += ["EV/$", "Kelly"]
    if "Stake" in df.columns:
        cols += ["Stake", "To win"]
    return df[cols].rename(columns={"description": "Bet"})


def _render_batter_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    keep = ["name", "expected_pa", "proj_h", "proj_hr", "proj_tb",
            "proj_rbi", "proj_runs", "proj_k", "proj_bb"]
    df = df[keep].rename(columns={
        "name": "Player", "expected_pa": "PA",
        "proj_h": "H", "proj_hr": "HR", "proj_tb": "TB",
        "proj_rbi": "RBI", "proj_runs": "R",
        "proj_k": "K", "proj_bb": "BB",
    })
    for c in ["PA", "H", "HR", "TB", "RBI", "R", "K", "BB"]:
        df[c] = df[c].round(2)
    return df


def _render_pitcher_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.copy()
    keep = ["name", "expected_ip", "proj_k", "proj_bb", "proj_h", "proj_er", "proj_hr_allowed"]
    df = df[keep].rename(columns={
        "name": "Pitcher", "expected_ip": "IP",
        "proj_k": "K", "proj_bb": "BB", "proj_h": "H",
        "proj_er": "ER", "proj_hr_allowed": "HR",
    })
    for c in ["IP", "K", "BB", "H", "ER", "HR"]:
        df[c] = df[c].round(2)
    return df


# ---------- Cached prediction call ----------
@st.cache_data(ttl=600, show_spinner=False)
def run_prediction(target_iso: str, edge: float, fetch_odds: bool):
    res = predict_core.predict_slate(
        target_date=target_iso,
        edge_threshold=edge,
        fetch_odds=fetch_odds,
        top_n=60,
    )
    return res


# ---------- Sidebar controls ----------
with st.sidebar:
    st.title(":baseball: MLB Predictor")
    st.caption("Fanatics lines + props · Polymarket sharp reference")

    st.divider()

    # ----- Slate selection -----
    st.markdown("**:date: Slate**")
    today = datetime.now(timezone.utc).date()
    selected_date = st.date_input(
        "Game date",
        value=today,
        min_value=today - timedelta(days=14),
        max_value=today + timedelta(days=14),
        label_visibility="collapsed",
    )
    fetch_odds = st.checkbox("Pull live odds (Fanatics)", value=True)

    st.divider()

    # ----- Filters -----
    st.markdown("**:control_knobs: Filters**")
    edge_threshold = st.slider(
        "Minimum edge",
        min_value=0.01, max_value=0.20, value=0.04, step=0.01,
        format="%.2f",
        help="Only show bets where model probability beats no-vig market by at least this much.",
    )
    ranking_mode = st.radio(
        "Rank by",
        ["Score", "EV / $"],
        horizontal=True,
        help=(
            "**Score** = (edge / √(p·(1−p))) × stat-reliability — Sharpe-like; favours high-confidence edges. "
            "**EV / $** = expected profit per dollar wagered (raw model value)."
        ),
    )

    st.divider()

    # ----- Bankroll / staking (Kelly) -----
    st.markdown("**:moneybag: Staking**")
    bankroll = st.number_input(
        "Bankroll ($)", min_value=0.0, value=1000.0, step=50.0,
        help="Recommended stakes are computed as a fraction of this bankroll.",
    )
    kelly_frac = st.select_slider(
        "Kelly fraction",
        options=[0.10, 0.125, 0.25, 0.33, 0.50, 1.0],
        value=0.25,
        format_func=lambda x: {0.10: "1/10", 0.125: "1/8", 0.25: "1/4",
                               0.33: "1/3", 0.50: "1/2", 1.0: "Full"}.get(x, str(x)),
        help=("Stake = bankroll × Kelly × this fraction. Fractional Kelly (¼ is "
              "the common default) curbs variance and the cost of any model "
              "over-confidence. Full Kelly maximises growth but is high-variance "
              "and unforgiving of mis-estimated edges."),
    )

    st.divider()

    # ----- Actions -----
    if st.button(":arrows_counterclockwise: Refresh data", use_container_width=True):
        run_prediction.clear()
        proj.reload_prop_models()
        from src import value as _value
        _value.reload_winprob_cal()
        _value.reload_prop_cal()
        st.rerun()

    st.caption(
        ":information_source: MLB Stats API · Open-Meteo · Baseball Savant · "
        "Fanatics (odds-api.io) · Polymarket sharp reference."
    )


# ---------- Header ----------
try:
    _date_label = selected_date.strftime("%A, %B %d, %Y").replace(" 0", " ")
except Exception:
    _date_label = selected_date.isoformat()

_h_left, _h_right = st.columns([3, 1])
with _h_left:
    st.markdown(f"## :baseball: MLB Slate — {_date_label}")
with _h_right:
    st.caption("&nbsp;")  # spacer

with st.spinner("Pulling slate, weather, lines, projections..."):
    try:
        slate = run_prediction(selected_date.isoformat(), edge_threshold, fetch_odds)
    except FileNotFoundError as e:
        st.error(
            f"Missing data file: `{e.filename}`. Run the build pipeline first:\n\n"
            "```\npython -m scripts.build_dataset\npython -m scripts.train\npython -m scripts.train_props\npython -m scripts.fit_dispersion\n```"
        )
        st.stop()

if slate.n_games == 0:
    st.info("No games scheduled for this date.")
    st.stop()


# ---------- Top metrics ----------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Games", slate.n_games)
m2.metric("Books loaded", slate.n_books)
m3.metric("Props", slate.n_props_loaded)
m4.metric("Value bets", len(slate.top_value))
_avg_edge = (
    sum(b.get("edge_pct", 0) for b in slate.top_value) / max(1, len(slate.top_value))
    if slate.top_value else 0.0
)
m5.metric("Avg edge", f"+{_avg_edge:.1f}%" if _avg_edge else "—")

_status_bits = []
if slate.odds_source != "none":
    _status_bits.append(f"Odds: **{slate.odds_source}**")
_status_bits.append(f"Edge ≥ **{edge_threshold:.0%}**")
_status_bits.append(f"Ranking: **{ranking_mode}**")
st.caption(" · ".join(_status_bits))

if slate.concentration_warning:
    st.warning(":warning: " + slate.concentration_warning)


# ---------- Today's Top 5 panel ----------
sort_key = "score" if ranking_mode == "Score" else "ev_per_dollar"
_sort_col = "Score" if sort_key == "score" else "EV/$"

# Pitcher-confirmation: exclude bets from games where one or both starters
# are TBD. Their pitcher projections fall back to defaults so any "edge" is
# unreliable. Flagged games are listed in a separate warning below.
# getattr defends against stale pickles from older cached SlateResult
# instances that don't have the field.
_unconfirmed_games = sorted({
    f"{gp.away_team} @ {gp.home_team}"
    for gp in slate.games
    if not getattr(gp, "starters_confirmed", True)
})
if _unconfirmed_games:
    st.warning(
        ":warning: **Starter not yet announced for "
        + ", ".join(_unconfirmed_games)
        + "** — bets from these games are flagged in the leaderboard and "
        "excluded from Top 5 (pitcher projections aren't reliable until "
        "the starter is posted)."
    )

_top5_pool = [b for b in slate.top_value if b.get("starters_confirmed", True)]
_top5 = sorted(_top5_pool, key=lambda x: -x.get(sort_key, 0))[:5]
if _top5:
    st.markdown("### :star: Top 5 picks")
    _cols = st.columns(min(5, len(_top5)))
    for col, vb in zip(_cols, _top5):
        with col:
            _bet = vb["description"]
            # Trim noisy juice tags from card display — full description
            # is preserved otherwise; CSS handles wrapping for long text.
            _bet_full = _bet.replace(" [1-sided, ~6% juice est.]", "")
            # Auto-scale font size based on length so game-total descriptions
            # like "San Francisco Giants @ Philadelphia Phillies Under 8.0"
            # still fit cleanly on the card without truncation.
            _n = len(_bet_full)
            if   _n <= 30:  _font = "1.00rem"
            elif _n <= 40:  _font = "0.92rem"
            elif _n <= 50:  _font = "0.82rem"
            elif _n <= 65:  _font = "0.74rem"
            else:           _font = "0.68rem"
            _odds = _amer(vb["odds"])
            _edge = vb.get("edge_pct", 0)
            _kelly = (vb.get("kelly", 0) or 0) * 100
            _model_p = (vb.get("model_prob", 0) or 0) * 100
            st.markdown(
                f"""<div class="pick-card">
                <div class="pick-bet" style="font-size: {_font};" title="{_bet_full}">{_bet_full}</div>
                <div class="pick-meta">{_odds} · model {_model_p:.0f}%</div>
                <div class="pick-edge">+{_edge:.1f}% edge</div>
                <div class="pick-meta">Kelly {_kelly:.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.write("")  # spacer


# ---------- Main tabs ----------
_all_bets: list[dict] = []
for _gp in slate.games:
    _all_bets.extend(_gp.game_value)
    _all_bets.extend(_gp.prop_value)

# Tab labels with badge counts
_n_value  = len(slate.top_value)
_n_games  = slate.n_games
_n_ci     = len(getattr(slate, "model_ci_bets", []) or [])
_n_parlay = len(getattr(slate, "parlays", []) or [])

(main_tab_value, main_tab_ci, main_tab_parlay, main_tab_sim, main_tab_intervals,
 main_tab_p1, main_tab_games, main_tab_track) = st.tabs([
    f":moneybag: Value Bets ({_n_value})",
    f":dart: Model CI ({_n_ci})",
    f":game_die: Parlays ({_n_parlay})",
    f":crystal_ball: Simulation ({_n_games})",
    ":bar_chart: Intervals",
    ":dart: P(≥1) Confidence",
    f":baseball: Games ({_n_games})",
    ":trophy: Track Record",
])


# ===== Model CI — bets inside the model's confidence region =====
with main_tab_ci:
    st.caption(
        ":dart: **Model CI** — alt-line bets where the line sits at/beyond the edge "
        "of the model's own predictive distribution (raw model side probability "
        "≥ 60%, before any market blend). These are the spots the model is most "
        "confident in a direction; one best-EV line per pick."
    )
    _ci_bets = getattr(slate, "model_ci_bets", []) or []
    if not _ci_bets:
        st.info("No bets currently inside the model's 60% confidence region.")
    else:
        import pandas as _pd
        _amerf = lambda o: (f"+{int(o)}" if o > 0 else f"{int(o)}")
        _ci_df = _pd.DataFrame([{
            "Bet": (b.get("description", "") if b.get("starters_confirmed", True)
                    else "⚠️ " + b.get("description", "")),
            "Market": b.get("market", "").replace("prop_", ""),
            "Odds": _amerf(b.get("odds", -110)),
            "Model%": round(b.get("model_prob_raw", 0) * 100, 1),
            "Blended%": round(b.get("model_prob", 0) * 100, 1),
            "No-vig%": round(b.get("novig_prob", 0) * 100, 1),
            "Edge%": round(b.get("edge_pct", 0), 1),
            "EV/$": round(b.get("ev_per_dollar", 0), 3),
        } for b in _ci_bets]).sort_values("Model%", ascending=False)
        st.dataframe(_ci_df, use_container_width=True, hide_index=True)
        st.caption(
            "Model% = raw model conviction (the CI criterion). Blended% = "
            "calibrated probability after the market blend (what pricing/EV use). "
            "A big Model%−Blended% gap means the model strongly disagrees with the "
            "market — historically that disagreement has NOT been reliably right, "
            "so treat high-Model%/low-Edge% rows with caution."
        )


# ===== Parlays — daily 3- & 4-leg, all-markets by default =====
with main_tab_parlay:
    from src import parlays as _parlays_mod
    st.caption(
        ":game_die: **Parlays** — daily 3- and 4-leg parlays built from any "
        "prop market by default (TB, HR, hits, RBI, runs, K, walks, plus "
        "pitcher props). One leg per game so legs are from independent games. "
        "Shown with EV under our calibrated estimate AND under the market's "
        "implied probability."
    )
    _MARKET_LABELS = {
        "prop_hits": "Hits", "prop_hr": "HR", "prop_tb": "TB",
        "prop_rbi": "RBI", "prop_runs": "Runs", "prop_k": "Batter K",
        "prop_bb": "Walks",
        "prop_pitcher_k": "Pitcher K", "prop_pitcher_bb": "Pitcher BB",
        "prop_pitcher_h": "Pitcher H", "prop_pitcher_er": "Pitcher ER",
        "prop_pitcher_hr": "Pitcher HR", "prop_pitcher_outs": "Pitcher Outs",
    }
    _c1, _c2 = st.columns([3, 1])
    _picked_labels = _c1.multiselect(
        "Markets eligible for parlay legs",
        options=list(_MARKET_LABELS.values()),
        default=list(_MARKET_LABELS.values()),
        help="Uncheck a market to exclude its legs. Skill-backed evidence: "
             "Runs & RBI are the only markets with a CI that excludes 0 on the "
             "encompassing test; TB/HR/hits/K show no measured edge yet.",
        key="parlay_markets",
    )
    _skill_only = _c2.toggle(
        "Skill-backed only",
        value=False,
        help="Restrict legs to markets with measured forecasting skill "
             "(runs/rbi). When ON, overrides the multi-select.",
        key="parlay_skill_only",
    )
    _label_to_market = {v: k for k, v in _MARKET_LABELS.items()}
    if _skill_only:
        _allow = _parlays_mod.SKILL_MARKETS
    else:
        _allow = tuple(_label_to_market[l] for l in _picked_labels) \
                  or _parlays_mod.ALL_PROP_MARKETS

    # Rebuild parlays under the current selection (cheap — same all_bets pool).
    _all_pool = getattr(slate, "all_bets", []) or []
    _parlays = _parlays_mod.build_parlays(_all_pool, skill_markets=_allow)

    if not _parlays:
        st.info(
            "No parlays match the current selection (need ≥3 legs at "
            "raw model prob ≥ 0.55 across different games). Try widening the "
            "market selection or wait for more odds to post."
        )
    else:
        st.warning(
            "Reality check: parlays multiply the book's hold. Most markets "
            "show no measured forecasting edge yet (only runs/rbi clear the "
            "CI bar). **EV/$ (model)** is usually below 0 and **EV/$ (market)** "
            "≈ the negative hold. This tab exists to TRACK whether high-"
            "conviction parlays beat their EV, not to assert they win.",
            icon="⚠️",
        )
        for p in _parlays:
            _ev = p.get("ev_per_dollar", 0)
            _badge = "🟢" if _ev > 0 else "🔴"
            with st.expander(
                f"{_badge} {p['label']}  ·  {p['american_odds']:+d}  ·  "
                f"hit {p.get('model_prob', 0):.1%}  ·  EV/$ {_ev:+.3f}",
                expanded=False,
            ):
                import pandas as _pd
                _lg = _pd.DataFrame([{
                    "Leg": (l.get("description", "") if l.get("starters_confirmed", True)
                            else "⚠️ " + l.get("description", "")),
                    "Odds": (f"+{l['odds']}" if l["odds"] > 0 else f"{l['odds']}"),
                    "Model%": round(l.get("model_prob", 0) * 100, 1),
                    "Conviction%": round(l.get("model_prob_raw", 0) * 100, 1),
                } for l in p.get("legs", [])])
                st.dataframe(_lg, use_container_width=True, hide_index=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Parlay odds", f"{p['american_odds']:+d}")
                c2.metric("Model hit %", f"{p.get('model_prob', 0):.1%}")
                c3.metric("EV/$ (model)", f"{_ev:+.3f}",
                          delta=f"mkt {p.get('ev_market', 0):+.3f}")
                c4.metric("$1 pays", f"${p.get('payout_per_dollar', 0):.2f}")

    # ----- HR-specific 2- and 3-leg parlays -----
    st.divider()
    st.subheader(":boom: Home Run Parlays")
    st.caption(
        "Dedicated 2- and 3-leg HR parlays — one player per game, ranked by "
        "the model's HR conviction (no min-conviction floor since HR overs "
        "are inherently ~10-15% per game). Payouts are big, EV is honest."
    )
    _hr_parlays = getattr(slate, "hr_parlays", []) or []
    if not _hr_parlays:
        st.info(
            "No HR parlays today (need ≥2 HR-prop offers across different games "
            "from the live feed)."
        )
    else:
        st.warning(
            "Reality check: HR is one-sided pricing with an 8% juice estimate "
            "and the model has no demonstrated edge on HR markets. The combined "
            "hit probability is small (a 2-leg ~1-3%, a 3-leg <1%); the "
            "displayed EV/$ will usually be negative. These are high-payout "
            "lotto tickets, not value plays.",
            icon="⚠️",
        )
        for p in _hr_parlays:
            _ev = p.get("ev_per_dollar", 0)
            _badge = "🟢" if _ev > 0 else "🔴"
            with st.expander(
                f"{_badge} {p['label']}  ·  {p['american_odds']:+d}  ·  "
                f"hit {p.get('model_prob', 0):.2%}  ·  EV/$ {_ev:+.3f}",
                expanded=False,
            ):
                import pandas as _pd
                _lg = _pd.DataFrame([{
                    "Leg": (l.get("description", "") if l.get("starters_confirmed", True)
                            else "⚠️ " + l.get("description", "")),
                    "Odds": (f"+{l['odds']}" if l["odds"] > 0 else f"{l['odds']}"),
                    "Model% (blended)": round(l.get("model_prob", 0) * 100, 1),
                    "Conviction% (raw)": round(l.get("model_prob_raw", 0) * 100, 1),
                } for l in p.get("legs", [])])
                st.dataframe(_lg, use_container_width=True, hide_index=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Parlay odds", f"{p['american_odds']:+d}")
                c2.metric("Model hit %", f"{p.get('model_prob', 0):.2%}")
                c3.metric("EV/$ (model)", f"{_ev:+.3f}",
                          delta=f"mkt {p.get('ev_market', 0):+.3f}")
                c4.metric("$1 pays", f"${p.get('payout_per_dollar', 0):.2f}")


# ===== Simulation — play-by-play Monte Carlo box scores =====
def _gp_sim_dict(gp) -> dict:
    return {
        "game_pk": gp.game_pk, "away_team": gp.away_team, "home_team": gp.home_team,
        "pred_away_runs": gp.pred_away_runs, "pred_home_runs": gp.pred_home_runs,
        "away_batters": gp.away_batters, "home_batters": gp.home_batters,
        "away_starter": gp.away_starter, "home_starter": gp.home_starter,
        "away_sp_id": gp.away_sp_id, "home_sp_id": gp.home_sp_id,
    }


@st.cache_data(show_spinner=False)
def _sim_leaderboard_cached(date: str, n: int, _games, _all_bets):
    """Per-market top-20 offered props/lines by simulated hit rate."""
    from src import game_sim
    def _sim_for_game(gp):
        return game_sim.simulate_game(_gp_sim_dict(gp), n=n, seed=gp.game_pk)
    return game_sim.build_sim_leaderboard_by_market(
        _games, _sim_for_game, _all_bets, top=20)


@st.cache_data(show_spinner="Simulating…")
def _sim_cached(game_pk: int, n: int, date: str, _gp: dict):
    from src import game_sim
    try:
        return game_sim.simulate_game(_gp, n=n, seed=game_pk)
    except Exception as e:
        return str(e)


with main_tab_sim:
    st.caption(
        ":crystal_ball: **Game Simulation** — a play-by-play Monte Carlo that bats "
        "each lineup through 9 innings (base-runners, outs, starter→bullpen hook) "
        "so runs and RBI EMERGE from the same game instead of independent marginals. "
        "Run it many times to see what the model actually favors. Team runs are "
        "**anchored** to the model's projection (the bottom-up total is shown too)."
    )
    _sim_games = [g for g in slate.games
                  if len(g.away_batters) >= 9 and len(g.home_batters) >= 9]
    if not _sim_games:
        st.info("No games have full 9-batter lineups projected yet (lineups post "
                "closer to first pitch).")
    else:
        # ---- Simulation leaderboard (10k sims/game) ----
        st.markdown("##### :trophy: Simulation Leaderboard")
        st.caption(
            "One tab per stat. Each shows the top 20 offered bets of that stat "
            "across the slate, ranked by how many of **10,000 sims per game** "
            "they HIT. Heavy compute, so it runs on demand and caches."
        )
        # market -> display label / tab order (props first, then game lines).
        _SIM_LB_LABELS = {
            "prop_hits": "Hits", "prop_hr": "HR", "prop_tb": "TB",
            "prop_rbi": "RBI", "prop_runs": "Runs", "prop_k": "Batter K",
            "prop_bb": "Walks",
            "prop_pitcher_k": "Pitcher K", "prop_pitcher_bb": "Pitcher BB",
            "prop_pitcher_h": "Pitcher H", "prop_pitcher_er": "Pitcher ER",
            "prop_pitcher_hr": "Pitcher HR", "prop_pitcher_outs": "Pitcher Outs",
            "moneyline": "Moneyline", "total": "Total", "run_line": "Run Line",
        }
        if st.button("Build 10,000-sim leaderboard", key="sim_lb_btn"):
            st.session_state["sim_lb_run"] = True
        if st.session_state.get("sim_lb_run"):
            with st.spinner("Simulating 10,000 games per matchup…"):
                _lb_by_mkt = _sim_leaderboard_cached(
                    slate.target_date, 10000, _sim_games,
                    getattr(slate, "sim_bets", None) or slate.all_bets)
            if not _lb_by_mkt:
                st.info("No offered props/lines could be matched to the simulation "
                        "yet (need live odds + projected lineups).")
            else:
                # Record the picks (live date only) so we can grade the sim's
                # most-confident calls against actuals later. Idempotent.
                if slate.target_date == today.isoformat():
                    try:
                        from src import sim_tracker
                        _added = sim_tracker.log_sim_picks(slate.target_date, _lb_by_mkt)
                        if _added:
                            st.caption(f":floppy_disk: Logged {_added} new sim "
                                       f"picks to the record.")
                    except Exception as _e:
                        st.caption(f"(sim-pick logging skipped: {_e})")
                import pandas as _pd
                _amerf = lambda o: (f"+{int(o)}" if o and o > 0 else f"{int(o)}" if o else "—")
                # Tab order: labelled markets in declared order, then any extras.
                _ordered = [m for m in _SIM_LB_LABELS if m in _lb_by_mkt]
                _ordered += [m for m in _lb_by_mkt if m not in _SIM_LB_LABELS]
                _tab_labels = [f"{_SIM_LB_LABELS.get(m, m)} ({len(_lb_by_mkt[m])})"
                               for m in _ordered]
                for _tab, _mkt in zip(st.tabs(_tab_labels), _ordered):
                    with _tab:
                        _rows = _lb_by_mkt[_mkt]
                        _lbdf = _pd.DataFrame([{
                            "#": i + 1,
                            "Matchup": r["matchup"],
                            "Bet": r["description"],
                            "Sim hit": f"{r['sim_hit']:.1%}",
                            "Hits / 10k": f"{r['sim_hits_n']:,}",
                            "Odds": _amerf(r["odds"]),
                            "Book no-vig": (f"{r['novig_prob']:.1%}"
                                            if r.get("novig_prob") not in (None, "") else "—"),
                            "Sim − book": (f"{(r['sim_hit'] - float(r['novig_prob'])) * 100:+.1f}pp"
                                           if r.get("novig_prob") not in (None, "") else "—"),
                        } for i, r in enumerate(_rows)])
                        st.dataframe(_lbdf, use_container_width=True, hide_index=True)
                st.caption(
                    "Each tab ranks that stat's offers purely by **Sim hit** "
                    "(simulated win frequency). **Sim − book** = simulation hit "
                    "rate minus the book's no-vig implied probability — positive "
                    "means the sim is more confident than the market (a model-vs-"
                    "market edge signal, not a guarantee). One row per "
                    "game/player/side/line."
                )

        # ---- Sim pick record (graded against actuals) ----
        with st.expander(":bar_chart: Sim pick record — how the sim's picks have done"):
            try:
                from src import sim_tracker
                _rec = sim_tracker.get_sim_record(days=60)
            except Exception as _e:
                _rec = None
                st.caption(f"(sim record unavailable: {_e})")
            if _rec and _rec["total"]:
                _wr = _rec["win_rate"]
                st.caption(
                    f"Last 60 days · **{_rec['wins']}W-{_rec['losses']}L"
                    f"-{_rec['pushes']}P** "
                    + (f"({_wr:.0%})" if _wr is not None else "")
                    + f" · {_rec['pending']} pending. Picks are logged the day "
                    "you build the leaderboard and graded once games go final."
                )
                _cal = _rec.get("calibration", {})
                _crows = []
                for _m, _bm in sorted(_rec["by_market"].items()):
                    _c = _cal.get(_m, {})
                    _msh, _mwr = _c.get("mean_sim_hit"), _c.get("win_rate")
                    _crows.append({
                        "Stat": _SIM_LB_LABELS.get(_m, _m),
                        "W": _bm["wins"], "L": _bm["losses"], "P": _bm["pushes"],
                        "Pending": _bm["pending"],
                        "Mean sim hit": f"{_msh:.0%}" if _msh is not None else "—",
                        "Actual win%": f"{_mwr:.0%}" if _mwr is not None else "—",
                    })
                if _crows:
                    st.dataframe(pd.DataFrame(_crows), use_container_width=True,
                                 hide_index=True)
                    st.caption(
                        "**Mean sim hit** vs **Actual win%** is the calibration "
                        "check: when the sim says a pick hits X% of the time, does "
                        "it win about X% live? Close agreement = trustworthy sim."
                    )
            elif _rec is not None:
                st.caption("No sim picks recorded yet. Build the leaderboard on a "
                           "live slate to start the record.")
        st.divider()

        # ---- Slate overview (low N, cached) ----
        st.markdown("##### Slate overview")
        st.caption("Quick scan — 400 sims/game, anchored. Drill into a game below "
                   "for a full box score at higher resolution.")
        _ov_rows = []
        for g in _sim_games:
            r = _sim_cached(g.game_pk, 400, slate.target_date, _gp_sim_dict(g))
            if isinstance(r, str) or r is False:
                continue
            _ov_rows.append({
                "Game": f"{g.away_team} @ {g.home_team}",
                "Sim score (H–A)": f"{r.anchored_home:.1f}–{r.anchored_away:.1f}",
                "P(home win)": round(r.p_home_win, 3),
                "Mean total": r.mean_total,
                "Bottom-up H/A": f"{r.free_home:.1f}/{r.free_away:.1f}",
                "Model H/A": f"{r.glm_home:.1f}/{r.glm_away:.1f}",
            })
        if _ov_rows:
            st.dataframe(pd.DataFrame(_ov_rows), use_container_width=True, hide_index=True)

        st.divider()
        # ---- Per-game drill-down (high N) ----
        st.markdown("##### Drill-down")
        _labels = [f"{g.away_team} @ {g.home_team}" for g in _sim_games]
        c1, c2 = st.columns([3, 1])
        _pick = c1.selectbox("Game", range(len(_sim_games)),
                             format_func=lambda i: _labels[i], key="sim_game")
        _nsim = c2.select_slider("Sims", options=[1000, 2000, 5000, 10000],
                                 value=5000, key="sim_n")
        g = _sim_games[_pick]
        res = _sim_cached(g.game_pk, _nsim, slate.target_date, _gp_sim_dict(g))
        if isinstance(res, str):
            st.error(f"Simulation failed: {res}")
        elif res is False:
            st.warning("Could not simulate this game.")
        else:
            # Team totals — anchored vs bottom-up vs model
            st.markdown(f"**{res.away_team} @ {res.home_team}** · {res.n:,} sims")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric(f"Sim score ({res.home_team})", f"{res.anchored_home:.2f}",
                      delta=f"bottom-up {res.free_home:.2f}")
            t2.metric(f"Sim score ({res.away_team})", f"{res.anchored_away:.2f}",
                      delta=f"bottom-up {res.free_away:.2f}")
            t3.metric("P(home win)", f"{res.p_home_win:.1%}")
            t4.metric("Mean total", f"{res.mean_total:.2f}")
            st.caption(
                f"Anchored to model ({res.glm_home:.1f}–{res.glm_away:.1f}); "
                f"anchor factors H×{res.anchor_f_home} / A×{res.anchor_f_away}. "
                f"Bottom-up = the lineups' own rates, free of the anchor — where it "
                f"diverges from the model, the two disagree. Most common finals: "
                + ", ".join(f"{k}×{v}" for k, v in list(res.score_dist.items())[:4])
            )

            def _box_df(box):
                return pd.DataFrame([{
                    "Batter": b["name"], "PA": b["pa"], "H": b["h"], "HR": b["hr"],
                    "TB": b["tb"], "RBI": b["rbi"], "R": b["r"], "K": b["k"],
                    "BB": b["bb"], "P(hit)": b["p_hit"], "P(HR)": b["p_hr"],
                } for b in box])

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(f"**{res.away_team} batters** (mean per sim)")
                st.dataframe(_box_df(res.box_away), use_container_width=True, hide_index=True)
                st.markdown(f"**Starter — {res.pit_away['name']}**")
                st.dataframe(pd.DataFrame([res.pit_away]).rename(
                    columns={"ip": "IP", "k": "K", "bb": "BB", "h": "H",
                             "hr": "HR", "er": "ER"}).drop(columns=["name"]),
                    use_container_width=True, hide_index=True)
            with cc2:
                st.markdown(f"**{res.home_team} batters** (mean per sim)")
                st.dataframe(_box_df(res.box_home), use_container_width=True, hide_index=True)
                st.markdown(f"**Starter — {res.pit_home['name']}**")
                st.dataframe(pd.DataFrame([res.pit_home]).rename(
                    columns={"ip": "IP", "k": "K", "bb": "BB", "h": "H",
                             "hr": "HR", "er": "ER"}).drop(columns=["name"]),
                    use_container_width=True, hide_index=True)
            st.caption(
                "P(hit)/P(HR) = share of sims the player recorded ≥1 — the "
                "'how often did it hold true' read. Box columns are means across "
                "all sims. Coherence: each side's batter runs and RBI sum to the "
                "team total because they come from one simulated game, not separate "
                "marginals. Caveats: coarse base-running (no steals/DP/errors), ER "
                "charges all runs to the pitcher on the mound."
            )


# ===== TAB 1 — Value Bets leaderboard =====
with main_tab_value:
    if ranking_mode == "Score":
        st.caption(
            "Ranked by **Score** = (edge / √(p·(1−p))) × stat-reliability — Sharpe-like; "
            "favours high-confidence edges over longshot lottery tickets."
        )
    else:
        st.caption(
            "Ranked by **EV / $** = expected profit per dollar wagered — raw model value "
            "with no penalty for longshots or low-reliability markets."
        )

    # ===== Sharp value — the lead strategy =====
    # Model-free +EV (Fanatics price beats the Polymarket no-vig line) is the
    # only strategy here with a defensible a-priori edge; model-priced bets
    # are the research program. When nothing fires, say WHY — an empty
    # section that silently swallows game lines looks like a bug.
    _sharp_pool = [b for b in (getattr(slate, "all_bets", []) or [])
                   if str(b.get("market", "")).startswith("sharp_")]
    _sharp_sum = getattr(slate, "sharp_summary", None) or {}
    if _sharp_pool:
        st.subheader(":zap: Sharp value — lead strategy")
        st.caption(
            "Fanatics' offered price beats the **Polymarket near-vigless** probability — "
            "market-measured +EV, no model required. Bet these first; everything below "
            "is model-priced."
        )
        _sdf = pd.DataFrame(_sharp_pool)
        _scols: dict = {
            "Bet":     _sdf["description"],
            "Odds":    _sdf["odds"].apply(_amer),
            "Sharp%":  (_sdf["model_prob"] * 100).round(1),
            "No-vig%": (_sdf["novig_prob"] * 100).round(1),
            "Edge%":   _sdf["edge_pct"].round(1),
            "EV/$":    _sdf["ev_per_dollar"].round(4),
            "Kelly%":  (_sdf["kelly"] * 100).round(3),
        }
        if bankroll and kelly_frac:
            _stk = _sdf["kelly"].apply(lambda k: _stake_dollars(k, bankroll, kelly_frac))
            _scols["Stake"] = _stk.round(0)
        st.dataframe(pd.DataFrame(_scols).sort_values("EV/$", ascending=False),
                     use_container_width=True, hide_index=True)
        st.divider()
    elif _sharp_sum.get("games_checked"):
        _bev = _sharp_sum.get("best_ev")
        st.caption(
            f":zap: **Sharp check:** {_sharp_sum['games_checked']} games compared against the "
            f"Polymarket sharp line — nothing +EV today"
            + (f" (best side {_bev:+.1%} EV/$)" if _bev is not None else "")
            + ". Fanatics is pricing these games efficiently; model game-line edges on them "
              "are suppressed by the sharp veto rather than bet into a sharper line."
        )

    MARKET_TABS = [
        ("Overall",        None),
        ("Confidence",     "__confidence__"),
        ("Pure Conf.",     "__pure_confidence__"),
        ("Model Conf.",    "__model_confidence__"),
        ("HR",          ["prop_hr"]),
        # Split Hits into two markets-by-line buckets so 1-hit (line 0.5)
        # and 2+ hit (line >= 1.5) bets get their own top-20 list.
        ("1 Hit",       "__hits_one__"),
        ("2+ Hits",     "__hits_multi__"),
        ("TB",          ["prop_tb"]),
        ("RBI",         ["prop_rbi"]),
        ("Runs",        ["prop_runs"]),
        ("Batter K",    ["prop_k"]),
        ("Walks",       ["prop_bb"]),
        ("Pitcher K",   ["prop_pitcher_k"]),
        ("Pitcher",     ["prop_pitcher_outs", "prop_pitcher_er", "prop_pitcher_h", "prop_pitcher_bb", "prop_pitcher_hr"]),
        ("Game Lines",  ["moneyline", "total", "run_line"]),
    ]


    def _hits_subset(pool: list[dict], multi: bool) -> list[dict]:
        """Filter prop_hits bets by whether the line is 1+-hit (0.5) or
        multi-hit (1.5, 2.5)."""
        out = []
        for b in pool:
            if b.get("market") != "prop_hits":
                continue
            line = b.get("line")
            try:
                line = float(line)
            except (TypeError, ValueError):
                continue
            if multi and line >= 1.0:
                out.append(b)
            elif (not multi) and line < 1.0:
                out.append(b)
        return out

    # Compute counts per market tab for badge labels
    _market_pool_for_counts = getattr(slate, "all_bets", []) or _all_bets
    _filtered_pool = [b for b in _market_pool_for_counts if b.get("edge_pct", 0) >= edge_threshold * 100]
    def _tab_count(markets) -> int:
        if markets is None:
            return min(40, len(slate.top_value))
        if markets == "__confidence__":
            return min(20, len(_all_bets))
        if markets == "__pure_confidence__":
            return min(20, len(_market_pool_for_counts))
        if markets == "__model_confidence__":
            return min(80, len(_market_pool_for_counts))
        if markets == "__hits_one__":
            return min(20, len(_hits_subset(_market_pool_for_counts, multi=False)))
        if markets == "__hits_multi__":
            return min(20, len(_hits_subset(_market_pool_for_counts, multi=True)))
        return min(20, len([b for b in _filtered_pool if b.get("market") in markets]))

    _tab_labels = [f"{lab} ({_tab_count(mk)})" for lab, mk in MARKET_TABS]

    with st.expander(":information_source: About these edges (read once)", expanded=False):
        st.markdown(
            "- **Game lines** are priced against the **Polymarket sharp** (near-vigless) "
            "reference: we only surface a game bet when Fanatics' price beats the true line "
            "(model-free +EV). When the market is efficient there will be none — that's honest.\n"
            "- **Props** are model projections blended toward the Fanatics market price. "
            "One-sided props show `[1-sided, ~8% juice est.]` — the no-vig estimate is rough at "
            "extreme odds; treat +500 longshots with caution.\n"
            "- **Stake** uses fractional Kelly on your sidebar bankroll. ¼ Kelly is the "
            "conservative default; it curbs variance and the cost of model over-confidence.\n"
            "- **Score** = (edge / √(p·(1−p))) × stat-reliability — Sharpe-like ranking. "
            "**EV / $** = expected profit per dollar."
        )

    if not slate.top_value and not _all_bets:
        st.info(":mag: No value bets exceed the edge threshold. Lower the **Minimum edge** slider in the sidebar to see more.")
    else:
        tabs = st.tabs(_tab_labels)
        for tab, (label, markets) in zip(tabs, MARKET_TABS):
            with tab:
                if markets is None:
                    pool = sorted(slate.top_value, key=lambda x: -x.get(sort_key, 0))[:40]
                elif markets == "__confidence__":
                    pool = sorted(
                        _all_bets,
                        key=lambda x: -(x.get("model_prob", 0) * _stat_rel_weights.get(x.get("market", ""), 0.10)),
                    )[:20]
                elif markets == "__pure_confidence__":
                    pool = sorted(
                        getattr(slate, "all_bets", []) or [],
                        key=lambda x: -(x.get("model_prob", 0) * _stat_rel_weights.get(x.get("market", ""), 0.10)),
                    )[:20]
                elif markets == "__model_confidence__":
                    pool = sorted(
                        getattr(slate, "all_bets", []) or _all_bets,
                        key=lambda x: -x.get("model_prob", 0),
                    )[:80]
                elif markets == "__hits_one__":
                    _market_pool = getattr(slate, "all_bets", []) or _all_bets
                    pool = sorted(
                        _hits_subset(_market_pool, multi=False),
                        key=lambda x: -x.get(sort_key, 0),
                    )[:20]
                elif markets == "__hits_multi__":
                    _market_pool = getattr(slate, "all_bets", []) or _all_bets
                    pool = sorted(
                        _hits_subset(_market_pool, multi=True),
                        key=lambda x: -x.get(sort_key, 0),
                    )[:20]
                else:
                    _market_pool = getattr(slate, "all_bets", []) or _all_bets
                    pool = sorted(
                        [b for b in _market_pool if b.get("market") in markets],
                        key=lambda x: -x.get(sort_key, 0),
                    )[:20]
                if not pool:
                    st.info(f"No value bets for {label}.")
                    continue

                is_confidence_tab = (markets in ("__confidence__", "__pure_confidence__"))
                is_pure_confidence_tab = (markets == "__pure_confidence__")
                is_model_confidence_tab = (markets == "__model_confidence__")
                if markets == "__confidence__":
                    st.caption(
                        "**Confidence** = stat-reliability × 4·p·(1−p), filtered to bets that beat "
                        "the no-vig line by your edge threshold. Top picks here are logged daily for outcome tracking."
                    )
                elif is_pure_confidence_tab:
                    st.caption(
                        "**Pure Confidence** = stat-reliability × 4·p·(1−p) on **every** evaluated bet, "
                        "ignoring the edge filter. Surfaces the bets the model is most certain about "
                        "regardless of whether the book line agrees — useful for spotting model conviction "
                        "even when the price isn't favourable."
                    )
                elif is_model_confidence_tab:
                    st.caption(
                        "Top 80 bets sorted by **raw model probability** — no edge, no reliability weighting. "
                        "Every evaluated bet is included regardless of edge threshold."
                    )

                _raw = pd.DataFrame(pool)
                # Prefix bet descriptions with a warning emoji when the
                # game's starter isn't confirmed yet.
                if "starters_confirmed" in _raw.columns:
                    _bet_disp = [
                        (d if c else f":warning: {d}")
                        for d, c in zip(_raw["description"], _raw["starters_confirmed"])
                    ]
                else:
                    _bet_disp = _raw["description"]
                _df_cols: dict = {
                    "Bet":        _bet_disp,
                    "Odds":       _raw["odds"].apply(_amer),
                    "Model%":     (_raw["model_prob"] * 100).round(1),
                    "No-vig%":    (_raw["novig_prob"] * 100).round(1),
                    "Edge%":      _raw["edge_pct"].round(1),
                    "Confidence": _raw["confidence"].round(3) if "confidence" in _raw.columns else 0.0,
                    "Pure Conf%": (
                        _raw["model_prob"] * _raw["market"].map(
                            lambda m: _stat_rel_weights.get(str(m), 0.10)
                        )
                    ).round(3) if "market" in _raw.columns else (_raw["model_prob"] * 0.10).round(3),
                    "Score":      _raw["score"].round(3),
                    "EV/$":       _raw["ev_per_dollar"].round(4),
                    "Kelly%":     (_raw["kelly"] * 100).round(3),
                }
                # Recommended stake (fractional Kelly) + profit if it wins.
                if bankroll and kelly_frac:
                    _stk = _raw["kelly"].apply(lambda k: _stake_dollars(k, bankroll, kelly_frac))
                    _decf = _raw["odds"].apply(lambda o: (1.0 + o / 100.0) if o > 0 else (1.0 + 100.0 / (-o)))
                    _df_cols["Stake"] = _stk.round(0)
                    _df_cols["To win"] = (_stk * (_decf - 1.0)).round(0)
                _sort_for_tab = (
                    "Pure Conf%" if is_confidence_tab
                    else "Model%" if is_model_confidence_tab
                    else _sort_col
                )
                df_lb = pd.DataFrame(_df_cols).sort_values(_sort_for_tab, ascending=False)

                # Color-code Edge% column: deeper green = bigger edge
                def _edge_style(v):
                    if pd.isna(v):
                        return ""
                    if v >= 15:
                        return "background-color: rgba(46,160,67,0.45); color: white; font-weight: 600;"
                    if v >= 10:
                        return "background-color: rgba(46,160,67,0.30); font-weight: 600;"
                    if v >= 5:
                        return "background-color: rgba(46,160,67,0.15);"
                    if v < 0:
                        return "color: #ff6347;"
                    return ""

                # Styler.applymap was renamed to Styler.map in pandas 2.1; use
                # whichever exists to stay portable across local & Streamlit Cloud.
                _styler = df_lb.style
                _style_fn = getattr(_styler, "map", None) or _styler.applymap
                _fmt = {
                    "Model%":     "{:.1f}%",
                    "No-vig%":    "{:.1f}%",
                    "Edge%":      "+{:.1f}%",
                    "Confidence": "{:.3f}",
                    "Pure Conf%": "{:.3f}",
                    "Score":      "{:.2f}",
                    "EV/$":       "{:+.3f}",
                    "Kelly%":     "{:.2f}%",
                }
                if "Stake" in df_lb.columns:
                    _fmt["Stake"] = "${:,.0f}"
                    _fmt["To win"] = "${:,.0f}"
                _styled = _style_fn(_edge_style, subset=["Edge%"]).format(_fmt)

                st.dataframe(
                    _styled,
                    use_container_width=True, hide_index=True,
                    height=min(500, 38 * len(df_lb) + 38),
                )


# ===== TAB 2 — Confidence Intervals =====
with main_tab_intervals:
    st.caption(
        "**80% confidence intervals** (10th–90th percentile) for every player stat and game total. "
        "Computed from the model's projected mean using empirically-fitted NegBin dispersion per stat. "
        "A narrow CI means the outcome is fairly predictable; a wide CI means high variance."
    )

    fits = _load_disp_fits()

    # --- Section 1: Game totals ---
    st.subheader("Game totals")

    game_rows = []
    for gp in slate.games:
        lo_a, hi_a = _ci(gp.pred_away_runs, _TEAM_RUN_DISP)
        lo_h, hi_h = _ci(gp.pred_home_runs, _TEAM_RUN_DISP)
        lo_t, hi_t = _ci(gp.pred_total, _TEAM_RUN_DISP * 0.9)  # totals slightly less dispersed
        game_rows.append({
            "Matchup":      f"{gp.away_team} @ {gp.home_team}",
            "Away pred":    round(gp.pred_away_runs, 2),
            "Away 80%CI":   _ci_str(lo_a, hi_a),
            "Home pred":    round(gp.pred_home_runs, 2),
            "Home 80%CI":   _ci_str(lo_h, hi_h),
            "Total pred":   round(gp.pred_total, 2),
            "Total 80%CI":  _ci_str(lo_t, hi_t),
            "P(home win)":  round(gp.p_home_win * 100, 1),
            "P(over 8.5)":  round(gp.p_over_8_5 * 100, 1),
        })

    game_df = pd.DataFrame(game_rows)
    st.dataframe(
        game_df,
        column_config={
            "Away pred":   st.column_config.NumberColumn("Away pred",  format="%.2f"),
            "Home pred":   st.column_config.NumberColumn("Home pred",  format="%.2f"),
            "Total pred":  st.column_config.NumberColumn("Total pred", format="%.2f"),
            "P(home win)": st.column_config.NumberColumn("P(home win)", format="%.1f%%"),
            "P(over 8.5)": st.column_config.NumberColumn("P(over 8.5)", format="%.1f%%"),
        },
        use_container_width=True, hide_index=True,
    )

    # --- Section 2: Player intervals — all games ---
    st.subheader("Player stat intervals")

    def _show_batter_intervals(batters: list[dict], team: str):
        if not batters:
            st.caption(f"No batter data for {team}.")
            return
        df = _batter_ci_rows(batters, fits)
        if df.empty:
            return
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in ["PA", "H", "HR", "TB", "RBI", "R", "K", "BB"]}
        st.dataframe(df, column_config=num_cfg,
                     use_container_width=True, hide_index=True,
                     height=min(600, 38 * len(df) + 38))

    def _show_starter_intervals(starter: dict | None, team: str):
        if not starter:
            st.caption(f"No starter data for {team}.")
            return
        df = _starter_ci_rows(starter, fits)
        if df.empty:
            return
        num_cfg = {c: st.column_config.NumberColumn(c, format="%.2f")
                   for c in ["K", "IP", "ER", "H", "BB", "HR"]}
        st.dataframe(df, column_config=num_cfg,
                     use_container_width=True, hide_index=True)

    for gp in slate.games:
        matchup_label = f"{gp.away_team} @ {gp.home_team}  —  {gp.pred_away_runs:.1f}–{gp.pred_home_runs:.1f} pred"
        with st.expander(matchup_label, expanded=True):
            bat_c1, bat_c2 = st.columns(2)
            with bat_c1:
                st.markdown(f"**{gp.away_team} batters**")
                _show_batter_intervals(gp.away_batters, gp.away_team)
                st.markdown(f"**{gp.away_team} starter:** {gp.away_sp_name or '?'}")
                _show_starter_intervals(gp.away_starter, gp.away_team)
            with bat_c2:
                st.markdown(f"**{gp.home_team} batters**")
                _show_batter_intervals(gp.home_batters, gp.home_team)
                st.markdown(f"**{gp.home_team} starter:** {gp.home_sp_name or '?'}")
                _show_starter_intervals(gp.home_starter, gp.home_team)

    # --- Section 3: CI width summary (which bets are most certain?) ---
    st.subheader("Narrowest intervals — highest certainty bets")
    st.caption(
        "Bets where the model's 80% CI is tightest relative to the prop line. "
        "A narrow spread (hi - lo) vs a book line at, say, 0.5 means the outcome is predictable."
    )

    width_rows = []
    for gp in slate.games:
        for b in gp.away_batters + gp.home_batters:
            for label, proj_key, disp_key in [
                ("H",  "proj_h",    "hits"),
                ("HR", "proj_hr",   "hr"),
                ("K",  "proj_k",    "k"),
                ("TB", "proj_tb",   "tb"),
            ]:
                mu = b.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits)
                lo, hi = _ci(mu, phi)
                # Skip if the entire 80% CI is zero — stat is effectively never occurring
                if hi == 0:
                    continue
                width_rows.append({
                    "Player":   b.get("name", "?"),
                    "Team":     gp.away_team if b in gp.away_batters else gp.home_team,
                    "Stat":     label,
                    "Proj":     round(mu, 2),
                    "CI low":   lo,
                    "CI high":  hi,
                    "CI width": hi - lo,
                })

    if width_rows:
        width_df = (
            pd.DataFrame(width_rows)
            .sort_values("CI width")
            .reset_index(drop=True)
        )
        st.dataframe(
            width_df,
            column_config={
                "Proj":     st.column_config.NumberColumn("Proj",     format="%.2f"),
                "CI low":   st.column_config.NumberColumn("CI low",   format="%d"),
                "CI high":  st.column_config.NumberColumn("CI high",  format="%d"),
                "CI width": st.column_config.NumberColumn("CI width", format="%d"),
            },
            use_container_width=True, hide_index=True,
        )


# ===== TAB 3 — P(≥1) Confidence =====
with main_tab_p1:
    st.caption(
        "Props ranked by how confident the model is that the player achieves **at least 1** of that stat. "
        "**Fair Odds** = American odds implied by the model probability (what the line *should* be). "
        "**Confidence** = stat reliability × outcome uncertainty — higher means the model is both "
        "accurate on this market and the result is genuinely uncertain."
    )

    fits_p1 = _load_disp_fits()

    # Load stat reliability weights for confidence calculation
    _STAT_REL_PATH = ROOT / "data" / "models" / "stat_reliability.json"
    try:
        import json as _json
        _stat_rel: dict = _json.loads(_STAT_REL_PATH.read_text(encoding="utf-8"))
    except Exception:
        _stat_rel = {}

    def _fair_odds(p: float) -> str:
        """Convert model probability to American odds string."""
        p = max(0.001, min(0.999, p))
        if p >= 0.5:
            return f"-{round(p / (1 - p) * 100)}"
        return f"+{round((1 - p) / p * 100)}"

    def _confidence(p: float, rel_key: str) -> float:
        rel = _stat_rel.get(rel_key, 0.10)
        return round(rel * 4 * p * (1 - p), 4)

    # (display label, proj key, dispersion key, stat_reliability key)
    BATTER_STATS_P1 = [
        ("H",   "proj_h",    "hits",  "prop_hits"),
        ("HR",  "proj_hr",   "hr",    "prop_hr"),
        ("TB",  "proj_tb",   "tb",    "prop_tb"),
        ("RBI", "proj_rbi",  "rbi",   "prop_rbi"),
        ("R",   "proj_runs", "runs",  "prop_runs"),
        ("K",   "proj_k",    "k",     "prop_k"),
        ("BB",  "proj_bb",   "bb",    "prop_bb"),
    ]
    PITCHER_STATS_P1 = [
        ("K",  "proj_k",   "pitcher_k",  "prop_pitcher_k"),
        ("H",  "proj_h",   "pitcher_h",  "prop_pitcher_h"),
        ("BB", "proj_bb",  "pitcher_bb", "prop_pitcher_bb"),
        ("ER", "proj_er",  "pitcher_er", "prop_pitcher_er"),
    ]

    p1_rows = []
    for gp in slate.games:
        matchup = f"{gp.away_team} @ {gp.home_team}"

        for b in gp.away_batters + gp.home_batters:
            team = gp.away_team if b in gp.away_batters else gp.home_team
            for label, proj_key, disp_key, rel_key in BATTER_STATS_P1:
                mu = b.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits_p1)
                p1 = _p_at_least_one(mu, phi)
                _, hi = _ci(mu, phi)
                if hi == 0:
                    continue  # 90th pct is still 0 — skip
                p1_rows.append({
                    "Player":     b.get("name", "?"),
                    "Team":       team,
                    "Matchup":    matchup,
                    "Stat":       label,
                    "Proj":       round(mu, 2),
                    "P(≥1)":      round(p1 * 100, 1),
                    "Fair Odds":  _fair_odds(p1),
                    "Confidence": _confidence(p1, rel_key),
                })

        for starter, team in [(gp.away_starter, gp.away_team), (gp.home_starter, gp.home_team)]:
            if not starter:
                continue
            for label, proj_key, disp_key, rel_key in PITCHER_STATS_P1:
                mu = starter.get(proj_key, 0) or 0
                if mu <= 0:
                    continue
                phi = disp_mod.disp_for(disp_key, mu, fits_p1)
                p1 = _p_at_least_one(mu, phi)
                _, hi = _ci(mu, phi)
                if hi == 0:
                    continue
                p1_rows.append({
                    "Player":     starter.get("name", "?") + " (SP)",
                    "Team":       team,
                    "Matchup":    matchup,
                    "Stat":       label,
                    "Proj":       round(mu, 2),
                    "P(≥1)":      round(p1 * 100, 1),
                    "Fair Odds":  _fair_odds(p1),
                    "Confidence": _confidence(p1, rel_key),
                })

    if not p1_rows:
        st.info("No projection data available.")
    else:
        p1_df = pd.DataFrame(p1_rows).sort_values("P(≥1)", ascending=False).reset_index(drop=True)

        # Controls row
        ctrl_c1, ctrl_c2 = st.columns([2, 1])
        with ctrl_c1:
            all_stats = sorted(p1_df["Stat"].unique())
            sel_stats = st.multiselect(
                "Filter by stat", all_stats, default=all_stats, key="p1_stat_filter"
            )
        with ctrl_c2:
            min_p1 = st.slider(
                "Min P(≥1)%", min_value=1, max_value=99, value=50, step=1,
                key="p1_thresh",
                help="Show props where the model gives at least this probability of ≥1 occurring.",
            )

        if sel_stats:
            p1_df = p1_df[p1_df["Stat"].isin(sel_stats)]
        p1_df = p1_df[p1_df["P(≥1)"] >= min_p1].reset_index(drop=True)

        if p1_df.empty:
            st.info("No props meet the threshold. Lower the slider.")
        else:
            st.dataframe(
                p1_df,
                column_config={
                    "Proj":       st.column_config.NumberColumn("Proj",       format="%.2f"),
                    "P(≥1)":      st.column_config.NumberColumn("P(≥1)",      format="%.1f%%"),
                    "Fair Odds":  st.column_config.TextColumn("Fair Odds"),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
                },
                use_container_width=True,
                hide_index=True,
                height=min(700, 38 * len(p1_df) + 38),
            )


# ===== TAB 4 — Per-game cards =====
with main_tab_games:
    for gp in slate.games:
        fav = gp.home_team if gp.p_home_win >= 0.5 else gp.away_team
        fav_pct = gp.p_home_win if fav == gp.home_team else (1 - gp.p_home_win)
        summary = (
            f"**{gp.away_team}** @ **{gp.home_team}**  •  "
            f"{gp.pred_away_runs:.1f} - {gp.pred_home_runs:.1f}  •  "
            f"{fav} {fav_pct:.0%}"
        )
        n_value = len(gp.game_value) + len(gp.prop_value)
        if n_value:
            summary += f"  •  :money_with_wings: {n_value} value bet(s)"

        with st.expander(summary, expanded=False):
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                roof = f" ({gp.park_roof})" if gp.park_roof != "open" else ""
                local_pitch = ""
                try:
                    d = datetime.fromisoformat(gp.first_pitch_utc)
                    local_pitch = d.strftime("%H:%M UTC")
                except Exception:
                    pass
                st.markdown(f"**Venue:** {gp.venue}{roof}")
                st.markdown(f"**First pitch:** {local_pitch}")
                st.markdown(f"**Park factors:** runs {gp.park_pf_runs:.2f}, HR {gp.park_pf_hr:.2f}")
            with c2:
                st.markdown(f"**Weather (game time):**")
                st.markdown(f"&nbsp;&nbsp;{gp.temp_f:.0f}°F • wind to CF {gp.wind_to_cf_mph:+.1f} mph")
                st.markdown(f"&nbsp;&nbsp;runs ×{gp.runs_mult:.2f} • HR ×{gp.hr_mult:.2f}")
            with c3:
                st.markdown(f"**Starters (FIP / xFIP):**")
                st.markdown(f"&nbsp;&nbsp;{gp.away_sp_name or '?'} ({gp.away_sp_fip:.2f} / {gp.away_sp_xfip:.2f})")
                st.markdown(f"&nbsp;&nbsp;{gp.home_sp_name or '?'} ({gp.home_sp_fip:.2f} / {gp.home_sp_xfip:.2f})")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Predicted total", f"{gp.pred_total:.2f}")
            p2.metric(f"{gp.away_team[:18]}", f"{gp.pred_away_runs:.2f}")
            p3.metric(f"{gp.home_team[:18]}", f"{gp.pred_home_runs:.2f}")
            p4.metric("P(over 8.5)", f"{gp.p_over_8_5:.0%}")

            if gp.book:
                ml = (gp.book.get("moneyline") or {})
                tot = (gp.book.get("total") or {})
                rl = (gp.book.get("run_line") or {})
                st.markdown(
                    f"**Lines ({gp.book_source}):** "
                    f"ML {gp.away_team[:8]} {_amer(ml.get('away'))} / "
                    f"{gp.home_team[:8]} {_amer(ml.get('home'))}  •  "
                    f"Total {tot.get('line', '?')} (O {_amer(tot.get('over'))} / U {_amer(tot.get('under'))})  •  "
                    f"RL ±{abs(rl.get('line', 1.5)):.1f}"
                )

            if gp.game_value:
                st.markdown("**Game-line value:**")
                st.dataframe(_render_value_df(gp.game_value, bankroll, kelly_frac), use_container_width=True, hide_index=True)
            if gp.prop_value:
                st.markdown("**Player prop value:**")
                st.dataframe(_render_value_df(gp.prop_value, bankroll, kelly_frac), use_container_width=True, hide_index=True)

            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(f"**{gp.away_team} batters**")
                st.dataframe(_render_batter_df(gp.away_batters),
                             use_container_width=True, hide_index=True)
                if gp.away_starter:
                    st.markdown(f"**Starter:** {gp.away_starter['name']}")
                    st.dataframe(_render_pitcher_df([gp.away_starter]),
                                 use_container_width=True, hide_index=True)
            with b2:
                st.markdown(f"**{gp.home_team} batters**")
                st.dataframe(_render_batter_df(gp.home_batters),
                             use_container_width=True, hide_index=True)
                if gp.home_starter:
                    st.markdown(f"**Starter:** {gp.home_starter['name']}")
                    st.dataframe(_render_pitcher_df([gp.home_starter]),
                                 use_container_width=True, hide_index=True)


# ===== TAB 5 — Track Record =====
with main_tab_track:
    st.caption("Top-10 confidence picks logged automatically each day. Outcomes evaluated against final boxscores.")

    try:
        _n_updated = bet_tracker.evaluate_outcomes()
        _record = bet_tracker.get_track_record(days=30)

        if _record["total"] == 0:
            st.info("No picks logged yet. Run predictions for today's slate to start tracking.")
        else:
            _decided = _record["wins"] + _record["losses"]
            _wr = _record["win_rate"]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Picks logged", _record["total"])
            r2.metric("Decided", _decided)
            r3.metric("Wins", _record["wins"])
            r4.metric(
                "Win rate",
                f"{_wr:.1%}" if _wr is not None else "—",
                delta=f"{(_wr - 0.5)*100:+.1f}pp vs 50%" if _wr is not None else None,
            )

            # Closing Line Value — the fastest, most reliable edge signal.
            _clv = _record.get("clv") or {}
            _clv_props = _clv.get("props") or {}
            if _clv.get("n") or _clv_props.get("n"):
                st.caption(
                    ":dart: **Closing Line Value** — did we bet a better price than the line closed at? "
                    "Beating the close is the gold-standard proof of edge and converges far faster than ROI."
                )
                if _clv.get("n"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Game-line bets w/ CLV", _clv["n"])
                    _pbc = _clv.get("pct_beat_close")
                    c2.metric("Beat the close",
                              f"{_pbc:.0%}" if _pbc is not None else "—",
                              delta=f"{(_pbc-0.5)*100:+.0f}pp vs 50%" if _pbc is not None else None)
                    _ac = _clv.get("avg_clv_pct")
                    c3.metric("Avg CLV", f"{_ac:+.2f} pp" if _ac is not None else "—")
                if _clv_props.get("n"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Prop bets w/ CLV", _clv_props["n"])
                    _pbc = _clv_props.get("pct_beat_close")
                    c2.metric("Beat the close (props)",
                              f"{_pbc:.0%}" if _pbc is not None else "—",
                              delta=f"{(_pbc-0.5)*100:+.0f}pp vs 50%" if _pbc is not None else None)
                    _ac = _clv_props.get("avg_clv_pct")
                    c3.metric("Avg CLV (props)", f"{_ac:+.2f} pp" if _ac is not None else "—")
            else:
                st.caption(
                    ":dart: **Closing Line Value** populates as logged bets and their captured "
                    "closing lines accumulate (capture began Jun 4). It will be the primary edge "
                    "gauge going forward."
                )

            if _record["by_market"]:
                _mkt_rows = []
                for mkt, bm in sorted(_record["by_market"].items()):
                    dec = bm["wins"] + bm["losses"]
                    _mkt_rows.append({
                        "Market":  mkt,
                        "Total":   bm["total"],
                        "W":       bm["wins"],
                        "L":       bm["losses"],
                        "Push":    bm.get("pushes", 0),
                        "Pending": bm["pending"],
                        "Win%":    f"{bm['wins']/dec:.0%}" if dec else "—",
                    })
                st.dataframe(pd.DataFrame(_mkt_rows), use_container_width=True, hide_index=True)

            with st.expander("Recent pick log", expanded=False):
                _log_rows = []
                for e in _record["entries"][:50]:
                    outcome = e.get("outcome") or "pending"
                    _log_rows.append({
                        "Date":       e["date"],
                        "Bet":        e["description"],
                        "Market":     e["market"],
                        "Line":       e.get("line", ""),
                        "Odds":       _amer(e.get("odds")) if e.get("odds") else "?",
                        "Model%":     f"{float(e.get('model_prob', 0))*100:.1f}%",
                        "Confidence": f"{float(e.get('confidence', 0)):.3f}",
                        "Outcome":    outcome,
                        "Actual":     e.get("actual", ""),
                    })
                if _log_rows:
                    _log_df = pd.DataFrame(_log_rows)
                    st.dataframe(_log_df, use_container_width=True, hide_index=True,
                                 height=min(600, 38 * len(_log_df) + 38))
    except Exception as _e:
        st.info(f"Track record unavailable: {_e}")
