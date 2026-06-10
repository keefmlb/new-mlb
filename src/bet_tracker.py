"""Log top-confidence picks and evaluate their outcomes against actuals.

Usage:
  # In predict_core.py, after building the slate:
  from . import bet_tracker
  bet_tracker.log_picks(target_date, top_value_bets, top_n=10)

  # In app.py Track Record tab:
  from src import bet_tracker
  record = bet_tracker.get_track_record(days=30)
"""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "bets" / "bet_log.json"
GAMES_CSV = ROOT / "data" / "games" / "games_2026.csv"
BOX_CSV   = ROOT / "data" / "games" / "box_2026.csv"
CLOSING_CSV = ROOT / "data" / "odds" / "closing_lines.csv"
CLOSING_PROPS_CSV = ROOT / "data" / "odds" / "closing_props.csv"


# ---------- Closing Line Value (CLV) ----------
# CLV measures whether we bet at a better price than the line closed at. It is
# the professional gold standard for evaluating a betting model: consistently
# beating the close is the single most reliable indicator of genuine edge, and
# it converges FAR faster than win/loss ROI (which needs hundreds of bets to
# separate skill from variance). Closing lines are captured per-game by
# predict_core into data/odds/closing_lines.csv as the slate is run near first
# pitch; CLV becomes computable for any logged game-line bet once its closing
# line has been captured. Props are captured the same way (closing_props.csv,
# one row per game/player/market, latest capture wins) so prop bets — most of
# the leaderboard — get CLV too.

def _load_closing_map() -> dict[int, dict]:
    """{game_pk: closing-line row} from data/odds/closing_lines.csv."""
    import csv as _csv
    out: dict[int, dict] = {}
    if not CLOSING_CSV.exists():
        return out
    try:
        with CLOSING_CSV.open("r", encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                try:
                    out[int(r["game_pk"])] = r
                except (TypeError, ValueError, KeyError):
                    continue
    except Exception:
        pass
    return out


def _load_closing_props_map() -> dict[tuple, dict]:
    """{(game_pk, player_id, raw_market): closing-prop row} from
    data/odds/closing_props.csv. Rows without a player_id are also keyed by
    (game_pk, lowercased player name, raw_market) as a fallback."""
    import csv as _csv
    out: dict[tuple, dict] = {}
    if not CLOSING_PROPS_CSV.exists():
        return out
    try:
        with CLOSING_PROPS_CSV.open("r", encoding="utf-8", newline="") as fh:
            for r in _csv.DictReader(fh):
                try:
                    gpk = int(r["game_pk"])
                except (TypeError, ValueError, KeyError):
                    continue
                mkt = (r.get("market") or "").strip()
                try:
                    pid = int(r.get("player_id") or 0)
                except (TypeError, ValueError):
                    pid = 0
                if pid:
                    out[(gpk, pid, mkt)] = r
                name = (r.get("player") or "").strip().lower()
                if name:
                    out.setdefault((gpk, name, mkt), r)
    except Exception:
        pass
    return out


def _amer_to_prob(odds) -> Optional[float]:
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o == 0:
        return None
    return (100.0 / (o + 100.0)) if o > 0 else (abs(o) / (abs(o) + 100.0))


def _devig(p_a: Optional[float], p_b: Optional[float]) -> Optional[float]:
    """No-vig probability of side A from a two-way market."""
    if p_a is None or p_b is None or (p_a + p_b) <= 0:
        return None
    return p_a / (p_a + p_b)


def clv_for_entry(entry: dict, closing: dict) -> Optional[dict]:
    """Compute CLV for a single logged game-line bet against its closing line.

    Returns {clv_prob_pct, beat_close} where clv_prob_pct is
    (closing no-vig prob of OUR side) - (bet-time no-vig prob of our side), in
    percentage points; positive = the line moved toward us = we beat the close.
    Totals fall back to a runs-based line-movement signal. Returns None when the
    bet can't be matched (e.g. props, missing closing odds).
    """
    mkt = entry.get("market"); desc = (entry.get("description") or "")
    bettime_nv = entry.get("novig_prob")
    home = (closing.get("home_team") or ""); away = (closing.get("away_team") or "")

    def _our_side_is_home() -> Optional[bool]:
        if home and home.lower() in desc.lower():
            return True
        if away and away.lower() in desc.lower():
            return False
        return None

    if mkt == "moneyline":
        is_home = _our_side_is_home()
        if is_home is None or bettime_nv is None:
            return None
        p_h = _amer_to_prob(closing.get("ml_home")); p_a = _amer_to_prob(closing.get("ml_away"))
        close_nv = _devig(p_h, p_a) if is_home else _devig(p_a, p_h)
        if close_nv is None:
            return None
        return {"clv_prob_pct": (close_nv - bettime_nv) * 100.0,
                "beat_close": close_nv > bettime_nv}

    if mkt == "run_line":
        is_home = _our_side_is_home()
        if is_home is None or bettime_nv is None:
            return None
        p_h = _amer_to_prob(closing.get("rl_home")); p_a = _amer_to_prob(closing.get("rl_away"))
        close_nv = _devig(p_h, p_a) if is_home else _devig(p_a, p_h)
        if close_nv is None:
            return None
        return {"clv_prob_pct": (close_nv - bettime_nv) * 100.0,
                "beat_close": close_nv > bettime_nv}

    if mkt == "total":
        try:
            close_total = float(closing.get("market_total"))
            bet_line = float(entry.get("line"))
        except (TypeError, ValueError):
            return None
        is_over = " over " in desc.lower()
        # Line moving UP helps an Over (we locked a lower number) and hurts an
        # Under. Express CLV as the favourable line movement in runs.
        move = (close_total - bet_line) if is_over else (bet_line - close_total)
        return {"clv_runs": move, "beat_close": move > 0}

    return None


# Mirror of value.evaluate_prop's one-sided juice estimate (0.08 overround,
# half attributed to each side). Used to no-vig a one-sided closing price the
# same way the bet-time novig_prob was estimated, so the two are comparable.
_ONE_SIDED_JUICE = 0.08


def clv_for_prop_entry(entry: dict, closing: dict) -> Optional[dict]:
    """Compute CLV for a logged player-prop bet against its captured closing
    prop offer.

    Same line at close → price CLV: (closing no-vig prob of OUR side) -
    (bet-time no-vig prob), in percentage points. Line moved → favourable
    line movement in stat units (clv_units), like totals' clv_runs: an OVER
    bettor beats the close when the line closes HIGHER than they locked,
    an UNDER bettor when it closes lower. Returns None when the bet can't
    be priced (missing closing odds, unknown side).
    """
    desc = entry.get("description") or ""
    if " OVER " in desc:
        is_over = True
    elif " UNDER " in desc:
        is_over = False
    else:
        return None
    try:
        bet_line = float(entry.get("line"))
        close_line = float(closing.get("line"))
    except (TypeError, ValueError):
        return None

    if abs(close_line - bet_line) > 1e-9:
        move = (close_line - bet_line) if is_over else (bet_line - close_line)
        return {"clv_units": move, "beat_close": move > 0}

    bettime_nv = entry.get("novig_prob")
    if bettime_nv is None:
        return None
    p_o = _amer_to_prob(closing.get("over"))
    p_u = _amer_to_prob(closing.get("under"))
    if p_o is not None and p_u is not None:
        close_nv = _devig(p_o, p_u) if is_over else _devig(p_u, p_o)
    elif p_o is not None and is_over:
        # One-sided "Yes" price at close: strip the same juice estimate the
        # bet-time pricing used so the comparison is apples-to-apples.
        close_nv = max(0.01, min(0.99, p_o - _ONE_SIDED_JUICE / 2.0))
    else:
        return None
    if close_nv is None:
        return None
    return {"clv_prob_pct": (close_nv - bettime_nv) * 100.0,
            "beat_close": close_nv > bettime_nv}


def _summarize_clv_rows(rows: list[dict]) -> dict:
    n = len(rows)
    beat = sum(1 for r in rows if r.get("beat_close"))
    prob_vals = [r["clv_prob_pct"] for r in rows if "clv_prob_pct" in r]
    return {
        "n": n,
        "beat_close": beat,
        "pct_beat_close": (beat / n) if n else None,
        "avg_clv_pct": (sum(prob_vals) / len(prob_vals)) if prob_vals else None,
        "n_priced": len(prob_vals),
    }


def get_clv_summary(days: int = 30) -> dict:
    """Aggregate CLV over recently logged bets that have a captured close.

    Top-level keys cover game lines (ML / RL / total) — unchanged contract.
    A "props" sub-dict carries the same aggregate for player props, which
    are matched to closing_props.csv by (game_pk, player_id, market).
    """
    entries = _load_log()
    closing = _load_closing_map()
    closing_props = _load_closing_props_map()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    game_rows: list[dict] = []
    prop_rows: list[dict] = []
    for e in entries:
        if e.get("date", "") < cutoff:
            continue
        if not e.get("game_pk"):
            continue
        gpk = int(e["game_pk"])
        mkt = e.get("market", "")
        if mkt.startswith("prop_"):
            raw_mkt = mkt[len("prop_"):]
            try:
                pid = int(e.get("player_id") or 0)
            except (TypeError, ValueError):
                pid = 0
            cl = closing_props.get((gpk, pid, raw_mkt)) if pid else None
            if not cl:
                # name fallback: description starts with the player name but
                # isn't cleanly separable; only the pid key is reliable.
                continue
            r = clv_for_prop_entry(e, cl)
            if r:
                prop_rows.append(r)
        else:
            cl = closing.get(gpk)
            if not cl:
                continue
            r = clv_for_entry(e, cl)
            if r:
                game_rows.append(r)
    out = _summarize_clv_rows(game_rows)
    out["props"] = _summarize_clv_rows(prop_rows)
    return out


# ---------- Statistics helpers ----------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a win rate. Returns (lo, hi) or None."""
    if n <= 0:
        return None
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------- Logging ----------

def _find_duplicate(entry: dict, existing: list[dict], date_str: str) -> Optional[dict]:
    """Return the already-logged equivalent of `entry` for this date, if any.

    Rules:
    - Same description on same date is always a duplicate.
    - For game-line markets (total, moneyline, run_line): block any bet on the
      same game_pk + market, regardless of direction (Over vs Under).
      This prevents the app being run twice in a day from logging conflicting
      sides of the same game total.
    - For player props: block same game_pk + market + player_id combination.
    """
    game_pk = entry.get("game_pk")
    market  = entry.get("market", "")
    pid     = entry.get("player_id")

    for e in existing:
        if e.get("date") != date_str:
            continue
        # Always block exact description match
        if e.get("description") == entry.get("description"):
            return e
        # Game-line markets: one side per game per day
        if market in ("moneyline", "total", "run_line") and game_pk is not None:
            if e.get("game_pk") == game_pk and e.get("market") == market:
                return e
        # Player props: one bet per player per market per game per day
        elif pid is not None and game_pk is not None:
            if (e.get("game_pk") == game_pk
                    and e.get("market") == market
                    and e.get("player_id") == pid):
                return e
    return None


def _is_duplicate(entry: dict, existing: list[dict], date_str: str) -> bool:
    return _find_duplicate(entry, existing, date_str) is not None


def log_picks(target_date: str | date, bets: list[dict], top_n: int = 10,
              policy_version: str | None = None, shadow: bool = False) -> None:
    """Save the top_n bets to the log file, IN THE ORDER GIVEN.

    `bets` is predict_core's score-ranked leaderboard — the same ordering the
    user sees in the app. (Previously this re-sorted by `confidence`, which
    peaks at p=0.5, so the log recorded a systematically different sample
    from the displayed picks — and that biased every constant later tuned
    against it.) `policy_version` tags each entry with the filter/calibration
    policy in force, so the log can be segmented when rules change.

    `shadow=True` marks entries as shadow picks: bets the policy surfaced but
    that aren't part of the headline top-10 betting record. They exist purely
    to tighten confidence intervals — outcomes and CLV are evaluated exactly
    like primary picks, but get_track_record and the headline records exclude
    them. If a bet was first logged as shadow and later qualifies as primary
    (e.g. a later slate run ranks it top-10), the existing entry is PROMOTED
    rather than duplicated.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = _load_log()

    date_str = target_date.isoformat() if isinstance(target_date, date) else target_date
    # Skip bets from games where starters haven't been confirmed — the
    # pitcher projections are placeholder values and edges are unreliable.
    eligible = [b for b in bets if b.get("starters_confirmed", True)]
    picks = eligible[:top_n]

    for p in picks:
        entry = {
            "date":        date_str,
            "logged_at":   datetime.now(timezone.utc).isoformat(),
            "policy_version": policy_version,
            "shadow":      bool(shadow),
            "description": p.get("description", ""),
            "market":      p.get("market", ""),
            "line":        p.get("line", 0.0),
            "odds":        p.get("odds", 0),
            "model_prob":  p.get("model_prob", 0.0),
            "novig_prob":  p.get("novig_prob", 0.0),
            "edge_pct":    p.get("edge_pct", 0.0),
            "confidence":  p.get("confidence", 0.0),
            "score":       p.get("score", 0.0),
            "game_pk":     p.get("game_pk"),
            "player_id":   p.get("player_id"),
            "outcome":     None,   # filled by evaluate_outcomes()
            "actual":      None,   # actual stat value or score
        }
        dup = _find_duplicate(entry, existing, date_str)
        if dup is None:
            existing.append(entry)
        elif not shadow and dup.get("shadow"):
            dup["shadow"] = False   # promote shadow -> primary

    LOG_PATH.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def _load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


# ---------- Outcome evaluation ----------

def regrade_outcomes() -> int:
    """Wipe ALL settled outcomes and re-grade the entire log from boxscores.

    Run this once after pulling a grading fix (e.g. the Jun 2026 away
    run-line sign bug, push handling) so historical entries reflect the
    corrected rules:  python -m src.bet_tracker regrade
    """
    entries = _load_log()
    if not entries:
        return 0
    for e in entries:
        e["outcome"] = None
        e["actual"] = None
    LOG_PATH.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")
    return evaluate_outcomes()


def evaluate_outcomes() -> int:
    """Fill in outcome fields for any logged picks whose games are now final.

    Returns the number of picks updated.
    Reads games_2026.csv and box_2026.csv. Only fills in picks where
    outcome is still None and the game date has passed.
    """
    entries = _load_log()
    if not entries:
        return 0

    try:
        import pandas as pd
        games = pd.read_csv(GAMES_CSV) if GAMES_CSV.exists() else None
        box   = pd.read_csv(BOX_CSV)   if BOX_CSV.exists()   else None
    except Exception:
        return 0

    today = datetime.now(timezone.utc).date()
    updated = 0

    for e in entries:
        if e.get("outcome") is not None:
            continue
        if e["date"] >= today.isoformat():
            continue  # game hasn't been played yet

        result = _resolve_outcome(e, games, box)
        if result is not None:
            e["outcome"] = result["outcome"]
            e["actual"]  = result["actual"]
            updated += 1

    if updated:
        LOG_PATH.write_text(json.dumps(entries, indent=2, default=str), encoding="utf-8")

    return updated


def _resolve_outcome(entry: dict, games, box) -> dict | None:
    """Determine win/loss for one logged bet. Returns {outcome, actual} or None."""
    market   = entry.get("market", "")
    desc     = entry.get("description", "").lower()
    line     = float(entry.get("line") or 0.0)
    game_pk  = entry.get("game_pk")
    player_id = entry.get("player_id")

    # --- Game line markets ---
    if market in ("moneyline", "total", "run_line"):
        if games is None or game_pk is None:
            return None
        row = games[games["game_pk"] == int(game_pk)]
        if row.empty or not row.iloc[0].get("is_final", False):
            return None
        r = row.iloc[0]
        home_s = float(r.get("home_score", 0) or 0)
        away_s = float(r.get("away_score", 0) or 0)
        total  = home_s + away_s

        if market == "total":
            side = "over" if "over" in desc else "under"
            actual = total
            if total == line:
                return {"outcome": "P", "actual": actual}   # push on whole-number lines
            if side == "over":
                won = total > line
            else:
                won = total < line
        elif market == "moneyline":
            home_team = str(r.get("home_team", "")).lower()
            away_team = str(r.get("away_team", "")).lower()
            if home_team and home_team in desc:
                won = home_s > away_s
            elif away_team and away_team in desc:
                won = away_s > home_s
            else:
                return None
            actual = f"{away_s:.0f}-{home_s:.0f}"
        elif market == "run_line":
            # `line` is the spread of the side we BET (stored per-side by
            # evaluate_game_lines): home bets carry the home line (e.g. -1.5),
            # away bets carry the away line (e.g. +1.5). A side covers when
            # its own margin plus its own spread is positive.
            home_team = str(r.get("home_team", "")).lower()
            away_team = str(r.get("away_team", "")).lower()
            margin = home_s - away_s
            if home_team and home_team in desc:
                cover = margin + line
                actual = margin
            elif away_team and away_team in desc:
                # BUG FIX (Jun 2026): this used to be (-margin + (-line)),
                # which negated the away spread a second time and graded every
                # away run-line bet as if the sign were flipped. Re-grade the
                # log after pulling this fix: python -m src.bet_tracker regrade
                cover = -margin + line
                actual = -margin
            else:
                return None
            if cover == 0:
                return {"outcome": "P", "actual": actual}
            won = cover > 0
        else:
            return None
        return {"outcome": "W" if won else "L", "actual": actual}

    # --- Player prop markets ---
    if box is None or player_id is None:
        return None

    prows = box[box["player_id"] == int(player_id)]
    # Filter to the specific game if we have game_pk
    if game_pk is not None:
        prows = prows[prows["game_pk"] == int(game_pk)]
    if prows.empty:
        return None

    r = prows.iloc[0]
    stat_map = {
        "prop_hr":           ("hr",    "h"),
        "prop_hits":         ("h",     "h"),
        "prop_tb":           ("tb",    "h"),
        "prop_rbi":          ("rbi",   "h"),
        "prop_runs":         ("runs_b","h"),
        "prop_k":            ("k_b",   "h"),
        "prop_bb":           ("bb_b",  "h"),
        "prop_pitcher_k":    ("k_p",   "h"),
        "prop_pitcher_outs": ("outs",  "h"),
        "prop_pitcher_er":   ("er",    "h"),
        "prop_pitcher_h":    ("h_p",   "h"),
        "prop_pitcher_bb":   ("bb_p",  "h"),
        "prop_pitcher_hr":   ("hr_p",  "h"),
    }
    if market not in stat_map:
        return None

    col, _ = stat_map[market]
    if col not in r:
        return None
    actual = float(r[col] or 0)
    if actual == line:
        return {"outcome": "P", "actual": actual}   # push on whole-number lines
    side = "over" if "over" in desc else "under"
    if side == "over":
        won = actual > line
    else:
        won = actual < line
    return {"outcome": "W" if won else "L", "actual": actual}


# ---------- Track record summary ----------

def get_track_record(days: int = 30) -> dict:
    """Return a summary dict for the last `days` days of logged picks.

    Keys:
      total, wins, losses, pending, win_rate, by_market
    """
    entries = _load_log()
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    # Shadow picks exist to tighten CIs on measurement (CLV, market records in
    # analyze_bets); they are NOT part of the headline betting record.
    recent = [e for e in entries if e["date"] >= cutoff and not e.get("shadow")]

    total   = len(recent)
    wins    = sum(1 for e in recent if e.get("outcome") == "W")
    losses  = sum(1 for e in recent if e.get("outcome") == "L")
    pushes  = sum(1 for e in recent if e.get("outcome") == "P")
    pending = sum(1 for e in recent if e.get("outcome") is None)
    decided = wins + losses          # pushes excluded from win rate

    by_market: dict[str, dict] = {}
    for e in recent:
        m = e.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"total": 0, "wins": 0, "losses": 0, "pushes": 0, "pending": 0}
        bm = by_market[m]
        bm["total"] += 1
        if e.get("outcome") == "W":
            bm["wins"] += 1
        elif e.get("outcome") == "L":
            bm["losses"] += 1
        elif e.get("outcome") == "P":
            bm["pushes"] += 1
        else:
            bm["pending"] += 1

    return {
        "total":    total,
        "wins":     wins,
        "losses":   losses,
        "pushes":   pushes,
        "pending":  pending,
        "win_rate": wins / decided if decided else None,
        "by_market": by_market,
        "clv":      get_clv_summary(days),
        "entries":  sorted(recent, key=lambda x: x["date"], reverse=True),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "regrade":
        n = regrade_outcomes()
        print(f"Re-graded log: {n} picks settled.")
    else:
        n = evaluate_outcomes()
        print(f"Evaluated outcomes: {n} picks updated.")
