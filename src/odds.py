"""Betting line client.

We support two sources, falling back gracefully:
  1. The Odds API (https://the-odds-api.com)  — set ODDS_API_KEY in env.
     Free tier: 500 calls/month, includes MLB h2h, totals, spreads from many books.
  2. A local JSON file at data/odds/manual.json  — for player props or when
     the user paste-loads lines from a sportsbook.

Player props are usually paywalled; we treat them as user-supplied via the
manual file. Format (data/odds/manual.json):

  {
    "as_of": "2026-04-27T18:00",
    "games": [
      {"home_team": "Yankees", "away_team": "Red Sox",
       "moneyline": {"home": -150, "away": +130},
       "total":     {"line": 8.5,  "over": -110, "under": -110},
       "run_line":  {"line": -1.5, "home": +160, "away": -185}}
    ],
    "player_props": [
      {"player": "Aaron Judge", "market": "hr",  "line": 0.5,  "over": +320, "under": -420},
      {"player": "Aaron Judge", "market": "hits", "line": 1.5, "over": +110, "under": -135}
    ]
  }
"""
from __future__ import annotations
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _secret(name: str) -> Optional[str]:
    """Read a key from the git-ignored data/secrets.json (env var wins)."""
    try:
        sec = json.loads(
            (Path(__file__).resolve().parent.parent / "data" / "secrets.json")
            .read_text(encoding="utf-8"))
        v = sec.get(name)
        return v.strip() if isinstance(v, str) and v.strip() else None
    except Exception:
        return None


def fetch_mlb_lines(force: bool = False, ttl_seconds: int = 600) -> list[dict]:
    """Fetch current MLB game lines (h2h + totals + spreads) from The Odds API.

    Returns a list of game records with consensus lines averaged across books.
    Returns empty list (no error) if no API key is set — let callers fall back
    to manual or no-lines mode.
    """
    api_key = os.environ.get("ODDS_API_KEY") or _secret("ODDS_API_KEY")
    if not api_key:
        return []
    cf = _cache("mlb_lines")
    if not force and cf.exists():
        import time as _t
        if _t.time() - cf.stat().st_mtime < ttl_seconds:
            try:
                return json.loads(cf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,totals,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(f"{ODDS_API_BASE}/sports/baseball_mlb/odds", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cf.write_text(json.dumps(data), encoding="utf-8")
    return data


def consensus_lines(odds_api_data: list[dict]) -> list[dict]:
    """Reduce per-book quotes to a single consensus line per game (mean of books)."""
    out = []
    for g in odds_api_data:
        home = g.get("home_team"); away = g.get("away_team")
        markets = {"h2h": [], "totals": [], "spreads": []}
        for bk in g.get("bookmakers", []):
            for m in bk.get("markets", []):
                key = m.get("key")
                if key not in markets:
                    continue
                markets[key].append(m.get("outcomes", []))

        ml = _avg_h2h(markets["h2h"], home, away)
        tot = _avg_total(markets["totals"])
        rl = _avg_spread(markets["spreads"], home, away)

        out.append({
            "home_team": home, "away_team": away,
            "commence_time": g.get("commence_time"),
            "moneyline": ml, "total": tot, "run_line": rl,
            "n_books": len(g.get("bookmakers", [])),
        })
    return out


def _avg_h2h(books: list[list[dict]], home: str, away: str) -> Optional[dict]:
    if not books:
        return None
    h, a, n = 0.0, 0.0, 0
    for outcomes in books:
        h_p = next((o["price"] for o in outcomes if o["name"] == home), None)
        a_p = next((o["price"] for o in outcomes if o["name"] == away), None)
        if h_p is None or a_p is None:
            continue
        h += h_p; a += a_p; n += 1
    if not n:
        return None
    return {"home": int(round(h / n)), "away": int(round(a / n))}


def _avg_total(books: list[list[dict]]) -> Optional[dict]:
    if not books:
        return None
    line = 0.0; over = 0.0; under = 0.0; n = 0
    for outcomes in books:
        ov = next((o for o in outcomes if o["name"] == "Over"), None)
        un = next((o for o in outcomes if o["name"] == "Under"), None)
        if not ov or not un:
            continue
        line += ov.get("point", 0); over += ov.get("price", -110)
        under += un.get("price", -110); n += 1
    if not n:
        return None
    return {"line": round(line / n, 1),
            "over": int(round(over / n)),
            "under": int(round(under / n))}


def _avg_spread(books: list[list[dict]], home: str, away: str) -> Optional[dict]:
    if not books:
        return None
    line = 0.0; h_p = 0.0; a_p = 0.0; n = 0
    for outcomes in books:
        h = next((o for o in outcomes if o["name"] == home), None)
        a = next((o for o in outcomes if o["name"] == away), None)
        if not h or not a:
            continue
        # MLB run line is virtually always +/- 1.5
        line += -h.get("point", -1.5)            # home line stated as e.g. -1.5; we store positive 1.5
        h_p += h.get("price", -110); a_p += a.get("price", -110); n += 1
    if not n:
        return None
    return {"line": round(line / n, 1),
            "home": int(round(h_p / n)),
            "away": int(round(a_p / n))}


def load_manual() -> dict:
    p = CACHE_DIR / "manual.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_lines_with_fallback() -> tuple[list[dict], list[dict], str]:
    """Load best-available lines. Returns (game_books, player_props, source_label).

    Order of preference:
      1. odds-api.io — Fanatics (the user's book + props) with a Polymarket
         near-vigless SHARP reference attached per game. Primary source.
      2. The Odds API (if ODDS_API_KEY set; consensus US-book game lines) —
         fallback for game lines only when Fanatics is unavailable.
      3. Manual JSON file.

    Bovada was removed Jun 2026: the user bets exclusively at Fanatics, so
    Bovada lines/props (which they can't act on) only added noise — no point
    line-shopping a book you won't bet.
    """
    # 1. odds-api.io (Fanatics + Polymarket sharp reference)
    try:
        from . import odds_api_io
        g, p, s = odds_api_io.fetch_mlb()
        if g:
            return g, p, s
    except Exception:
        pass

    # 2. The Odds API (game-line fallback only)
    try:
        raw = fetch_mlb_lines()
        if raw:
            return consensus_lines(raw), [], "the-odds-api"
    except Exception:
        pass

    # 3. Manual
    m = load_manual()
    if m.get("games") or m.get("player_props"):
        return m.get("games", []), m.get("player_props", []), "manual"

    return [], [], "none"


def snapshot_odds(books: list[dict], props: list[dict]) -> None:
    """Append a timestamped snapshot of current lines to the odds history log.

    Stored at data/odds/odds_history.json. Keeps 14 days of snapshots.
    Used to compute line-movement features once enough history accumulates.
    """
    hist_path = CACHE_DIR / "odds_history.json"
    history: list[dict] = []
    if hist_path.exists():
        try:
            history = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    now_ts = datetime.now(timezone.utc).isoformat()
    history.append({"ts": now_ts, "books": books, "props": props})

    # Trim to last 14 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    history = [h for h in history if h.get("ts", "") >= cutoff]

    try:
        hist_path.write_text(json.dumps(history, default=str), encoding="utf-8")
    except Exception:
        pass


def get_line_movement(home_team: str, away_team: str,
                      hours_back: float = 6.0) -> dict | None:
    """Return opening vs current line for a game if history exists.

    Returns {"home_ml_open": int, "home_ml_now": int, "move_pct": float}
    or None if insufficient history.
    """
    hist_path = CACHE_DIR / "odds_history.json"
    if not hist_path.exists():
        return None
    try:
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    window = [h for h in history if h.get("ts", "") >= cutoff]
    if len(window) < 2:
        return None

    def _find_game(snapshot: dict) -> dict | None:
        for b in snapshot.get("books", []):
            hn = (b.get("home_team") or "").lower()
            an = (b.get("away_team") or "").lower()
            if home_team.lower() in hn and away_team.lower() in an:
                return b
        return None

    opening = _find_game(window[0])
    current = _find_game(window[-1])
    if not opening or not current:
        return None

    open_ml = (opening.get("moneyline") or {}).get("home")
    curr_ml = (current.get("moneyline") or {}).get("home")
    if open_ml is None or curr_ml is None:
        return None

    return {
        "home_ml_open": open_ml,
        "home_ml_now":  curr_ml,
        "move_pts":     curr_ml - open_ml,
    }
