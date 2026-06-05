"""odds-api.io client — Fanatics (bettable book + props) + Polymarket (sharp ref).

Why this exists: our team-runs model beats a constant baseline by only ~2.5%
on totals — it cannot out-predict the market. The professional edge for such a
model is to (a) anchor to the SHARPEST available probability and (b) bet where
a softer, bettable book diverges from it. odds-api.io gives us both in one feed:

  - Polymarket — a real-money prediction market with NEAR-ZERO vig (MLB
    moneylines de-vig to ~1% overround). Its implied probabilities are about
    the sharpest free estimate of true win probability available.
  - Fanatics  — a full US sportsbook (ML / totals / run-line + player props)
    that the user can actually bet, and a second sharp reference.

Auth: API key (query param `apiKey`). Loaded from env ODDS_API_IO_KEY or
data/secrets.json. Base https://api.odds-api.io/v3. Rate limit 100/hour — a
full slate costs ~1 (events) + N (odds) calls; we cache per-process.

Returns lines in the SAME internal format as src/odds.py so the existing value
pipeline consumes them unchanged:
  game dict: {home_team, away_team, commence_time,
              moneyline:{home,away}, total:{line,over,under},
              run_line:{line,home,away}, sharp:{...}}   (American odds, ints)
  prop dict: {player, market, line, over, under, game}
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
_BASE = "https://api.odds-api.io/v3"
_MLB_LEAGUE = "usa-mlb"
_BOOKS = ["Fanatics", "Polymarket"]

# Map Fanatics player-prop label suffixes to our internal market names.
_PROP_MARKET_MAP = {
    "Total Bases":      "tb",
    "Runs Batted In":   "rbi",
    "Runs Scored":      "runs",
    "Home Runs":        "hr",
    "Total Strikeouts": "pitcher_k",
    "Pitching Hits":    "pitcher_h",
    "Hits Allowed":     "pitcher_h",
    "Hits":             "hits",
    # "Hits+Runs+RBIs" intentionally unmapped — we don't price that combo.
}


# ---------- key + http ----------
def _api_key() -> Optional[str]:
    import os
    k = os.environ.get("ODDS_API_IO_KEY")
    if k:
        return k.strip()
    try:
        sec = json.loads((_ROOT / "data" / "secrets.json").read_text(encoding="utf-8"))
        return (sec.get("ODDS_API_IO_KEY") or "").strip() or None
    except Exception:
        return None


def _get(path: str, key: str, timeout: int = 30) -> object:
    sep = "&" if "?" in path else "?"
    url = f"{_BASE}/{path}{sep}apiKey={urllib.parse.quote(key)}"
    req = urllib.request.Request(url, headers={"User-Agent": "mlb-predictor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------- odds math ----------
def _dec_to_american(dec) -> Optional[int]:
    try:
        d = float(dec)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    return int(round((d - 1) * 100)) if d >= 2.0 else int(round(-100.0 / (d - 1)))


def _dec_to_prob(dec) -> Optional[float]:
    try:
        d = float(dec)
    except (TypeError, ValueError):
        return None
    return 1.0 / d if d > 0 else None


# ---------- market parsing ----------
def _parse_book_markets(markets: list) -> dict:
    """Convert one bookmaker's market list into {moneyline, total, run_line}
    in our internal American-odds format. `markets` is the array under
    bookmakers[<book>]."""
    out: dict = {}
    for m in markets or []:
        name = m.get("name")
        odds = m.get("odds") or []
        if name == "ML" and odds:
            o = odds[0]
            ah = _dec_to_american(o.get("home")); aa = _dec_to_american(o.get("away"))
            if ah is not None and aa is not None:
                out["moneyline"] = {"home": ah, "away": aa,
                                    "home_dec": float(o["home"]), "away_dec": float(o["away"])}
        elif name == "Totals" and odds:
            # Pick the most balanced line (the hung main number): minimise the
            # gap between over and under decimal prices.
            best = None
            for o in odds:
                ov, un = o.get("over"), o.get("under")
                if ov in (None, "N/A") or un in (None, "N/A"):
                    continue
                try:
                    gap = abs(float(ov) - float(un))
                except (TypeError, ValueError):
                    continue
                if best is None or gap < best[0]:
                    best = (gap, o)
            if best:
                o = best[1]
                out["total"] = {"line": float(o["hdp"]),
                                "over": _dec_to_american(o["over"]),
                                "under": _dec_to_american(o["under"]),
                                "over_dec": float(o["over"]), "under_dec": float(o["under"])}
        elif name == "Spread" and odds:
            # MLB run line is ±1.5. Prefer the -1.5 (home favourite laying)
            # entry; fall back to +1.5.
            pick = None
            for target in (-1.5, 1.5):
                for o in odds:
                    try:
                        if abs(float(o.get("hdp")) - target) < 1e-6:
                            pick = o; break
                    except (TypeError, ValueError):
                        continue
                if pick:
                    break
            if pick:
                ah = _dec_to_american(pick.get("home")); aa = _dec_to_american(pick.get("away"))
                if ah is not None and aa is not None:
                    out["run_line"] = {"line": float(pick["hdp"]), "home": ah, "away": aa}
    return out


def _parse_props(markets: list, game_str: str) -> list[dict]:
    """Parse Fanatics 'Player Props' into our internal prop dicts."""
    import re
    out: list[dict] = []
    for m in markets or []:
        if m.get("name") != "Player Props":
            continue
        for o in m.get("odds") or []:
            lab = o.get("label", "")
            mm = re.match(r"^(.*) \((.*)\)$", lab)
            if not mm:
                continue
            player, raw_mkt = mm.group(1).strip(), mm.group(2).strip()
            market = _PROP_MARKET_MAP.get(raw_mkt)
            if not market:
                continue
            over = o.get("over"); under = o.get("under")
            over_am = _dec_to_american(over) if over not in (None, "N/A") else None
            under_am = _dec_to_american(under) if under not in (None, "N/A") else None
            if over_am is None and under_am is None:
                continue
            out.append({
                "player": player, "market": market,
                "line": float(o["hdp"]) if o.get("hdp") is not None else None,
                "over": over_am, "under": under_am,
                "game": game_str, "source": "fanatics",
            })
    return out


def _sharp_reference(books_raw: dict) -> dict:
    """Build the sharp no-vig probabilities for a game from Polymarket (and
    Fanatics as backup). Returns {ml_home, ml_away, total_line, p_over, p_under}
    where probabilities are de-vigged. Polymarket is preferred for ML (near-zero
    vig); totals fall back to Fanatics if Polymarket lacks them."""
    sharp: dict = {}
    poly = _parse_book_markets(books_raw.get("Polymarket"))
    fan = _parse_book_markets(books_raw.get("Fanatics"))

    # Moneyline: prefer Polymarket (sharpest), else Fanatics.
    for src in (poly, fan):
        ml = src.get("moneyline")
        if ml:
            ph = _dec_to_prob(ml["home_dec"]); pa = _dec_to_prob(ml["away_dec"])
            if ph and pa and (ph + pa) > 0:
                sharp["ml_home"] = ph / (ph + pa)
                sharp["ml_away"] = pa / (ph + pa)
                sharp["ml_source"] = "Polymarket" if src is poly else "Fanatics"
                break
    # Totals: prefer whichever has it (Polymarket if present, else Fanatics).
    for src in (poly, fan):
        tot = src.get("total")
        if tot:
            po = _dec_to_prob(tot["over_dec"]); pu = _dec_to_prob(tot["under_dec"])
            if po and pu and (po + pu) > 0:
                sharp["total_line"] = tot["line"]
                sharp["p_over"] = po / (po + pu)
                sharp["p_under"] = pu / (po + pu)
                sharp["total_source"] = "Polymarket" if src is poly else "Fanatics"
                break
    return sharp


# ---------- public API ----------
_CACHE: dict = {"ts": 0.0, "data": None}
_CACHE_TTL = 300  # seconds


def fetch_mlb(bettable: str = "Fanatics", force: bool = False) -> tuple[list[dict], list[dict], str]:
    """Fetch today's MLB lines. Returns (book_games, props, source).

    book_games: list of game dicts in internal format from the `bettable` book
                (default Fanatics), each carrying a `sharp` field with the
                Polymarket-derived no-vig reference probabilities.
    props:      Fanatics player props in internal format.
    source:     'odds-api.io' on success, 'none' on failure.
    """
    key = _api_key()
    if not key:
        return [], [], "none"
    now = time.time()
    if not force and _CACHE["data"] and (now - _CACHE["ts"]) < _CACHE_TTL:
        return _CACHE["data"]

    try:
        events = _get(f"events?sport=baseball&league={_MLB_LEAGUE}", key)
    except Exception as exc:
        print(f"[odds-api.io] events fetch failed: {exc}")
        return [], [], "none"

    pending = [e for e in (events or []) if e.get("status") == "pending"
               and e.get("id") and e.get("home") and e.get("away")]
    # The events feed spans many days; a team in a series appears 2-3 times.
    # Keep only the EARLIEST pending occurrence of each matchup (= the next /
    # today's game) so downstream team-name matching can't grab a future date.
    pending.sort(key=lambda e: e.get("date", ""))
    seen: set = set()
    deduped = []
    for e in pending:
        mk = (e["away"], e["home"])
        if mk in seen:
            continue
        seen.add(mk)
        deduped.append(e)
    pending = deduped
    book_games: list[dict] = []
    props: list[dict] = []

    # Batch via /odds/multi (up to 10 event ids per call) — this keeps a full
    # slate to ~1 (events) + ceil(N/10) (odds) calls, well under the 100/hour
    # cap even with frequent refreshes.
    bk_param = ",".join(_BOOKS)
    for i in range(0, len(pending), 10):
        chunk = pending[i:i + 10]
        ids = ",".join(str(e["id"]) for e in chunk)
        try:
            batch = _get(f"odds/multi?eventIds={ids}&bookmakers={bk_param}", key)
        except Exception:
            continue
        for od in (batch or []):
            home = od.get("home"); away = od.get("away")
            if not (home and away):
                continue
            books_raw = od.get("bookmakers") or {}
            bettable_markets = _parse_book_markets(books_raw.get(bettable))
            if not bettable_markets:
                continue
            game_str = f"{away} @ {home}"
            g = {
                "home_team": home, "away_team": away,
                "commence_time": od.get("date"),
                "sharp": _sharp_reference(books_raw),
                "book_name": bettable,
            }
            g.update(bettable_markets)
            book_games.append(g)
            props.extend(_parse_props(books_raw.get(bettable), game_str))

    result = (book_games, props, "odds-api.io" if book_games else "none")
    _CACHE.update(ts=now, data=result)
    return result
