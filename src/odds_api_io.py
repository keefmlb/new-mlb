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
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
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
    # The feed has shipped this under BOTH labels. Aug 11 2026 it was sending
    # "Pitcher Strikeouts" while the map only knew "Total Strikeouts", so all
    # 215 daily strikeout props were dropped — the single market the sim
    # leaderboard leans on hardest. Keep both, and see the fallback below.
    "Total Strikeouts": "pitcher_k",
    "Pitcher Strikeouts": "pitcher_k",
    "Strikeouts Thrown": "pitcher_k",
    "Pitching Hits":    "pitcher_h",
    "Hits Allowed":     "pitcher_h",
    "Hits":             "hits",
    "Total Hits":       "hits",
    "Batter Hits":      "hits",
    # Fanatics' largest player-prop market by volume. Priced from the summed
    # H+R+RBI projection with its own empirically-fitted dispersion, and scored
    # exactly (not as independent marginals) by the play-by-play simulator.
    "Hits+Runs+RBIs":   "hrr",
}


def _map_prop_market(raw_mkt: str) -> Optional[str]:
    """Map a Fanatics prop label to an internal market name.

    Exact lookup first, then a guarded fallback for batter hits: the feed has
    shipped this market under several labels ("Hits", "Total Hits", …) and an
    exact-match-only map silently dropped ALL hits props when the wording
    changed. The fallback accepts any label mentioning hits while excluding
    the pitcher variants (Hits Allowed / Pitching Hits, already mapped above)
    and the Hits+Runs+RBIs combo, which we don't price.
    """
    m = _PROP_MARKET_MAP.get(raw_mkt)
    if m:
        return m
    low = raw_mkt.lower()
    if ("hit" in low and "+" not in raw_mkt
            and "allow" not in low and "pitch" not in low
            and "run" not in low and "rbi" not in low):
        return "hits"
    # Strikeouts are a PITCHER market here. Batter-strikeout props would also
    # say "strikeout", so require a pitcher-ish qualifier rather than matching
    # the bare word; batter K is not a market we price.
    if "strikeout" in low and ("pitcher" in low or "total" in low
                               or "thrown" in low):
        return "pitcher_k"
    _note_unmapped(raw_mkt)
    return None


# Labels the feed sent that we do not understand. Silent drops are how this
# module has lost whole markets twice (batter hits, then pitcher strikeouts):
# the prop count stays plausibly large because other markets fill it in, so
# nothing looks wrong. Surfacing them makes the next rename obvious.
_UNMAPPED_SEEN: set = set()


def _note_unmapped(raw_mkt: str) -> None:
    if raw_mkt in _UNMAPPED_SEEN:
        return
    _UNMAPPED_SEEN.add(raw_mkt)
    print(f"[odds-api.io] WARNING unmapped prop market {raw_mkt!r} — dropping "
          f"these props. Add it to _PROP_MARKET_MAP if we price it.")


def unmapped_markets() -> list[str]:
    """Prop labels seen this run that no rule matched (for UI surfacing)."""
    return sorted(_UNMAPPED_SEEN)


# ---------- key + http ----------
def _api_keys() -> list[str]:
    """All configured odds-api.io keys, in priority order. Supports rotation
    on rate-limit (429). Sources, de-duplicated, first-seen order:
      - env ODDS_API_IO_KEY, then ODDS_API_IO_KEY2..ODDS_API_IO_KEY5
      - secrets.json ODDS_API_IO_KEY (str) and ODDS_API_IO_KEYS (list[str])
    Add a second key by setting ODDS_API_IO_KEY2 or the ODDS_API_IO_KEYS list.
    """
    import os
    keys: list[str] = []

    def _add(v):
        if isinstance(v, str) and v.strip() and v.strip() not in keys:
            keys.append(v.strip())

    _add(os.environ.get("ODDS_API_IO_KEY"))
    for i in range(2, 6):
        _add(os.environ.get(f"ODDS_API_IO_KEY{i}"))
    try:
        sec = json.loads((_ROOT / "data" / "secrets.json").read_text(encoding="utf-8"))
        _add(sec.get("ODDS_API_IO_KEY"))
        for v in (sec.get("ODDS_API_IO_KEYS") or []):
            _add(v)
        for i in range(2, 6):
            _add(sec.get(f"ODDS_API_IO_KEY{i}"))
    except Exception:
        pass
    return keys


def _api_key() -> Optional[str]:
    ks = _api_keys()
    return ks[0] if ks else None


def _get(path: str, key: str | None = None, timeout: int = 30,
         retries: int = 2, backoff: float = 1.5) -> object:
    """GET with rate-limit resilience. On HTTP 429, rotate to the next
    configured key; if all keys are limited, back off and retry. `key` is
    accepted for backward compatibility but the full key list is always used
    so a single failing call can still succeed on a backup key."""
    import time
    keys = _api_keys()
    if key and key not in keys:
        keys = [key] + keys
    if not keys:
        raise RuntimeError("no odds-api.io key configured")
    sep = "&" if "?" in path else "?"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        for k in keys:
            url = f"{_BASE}/{path}{sep}apiKey={urllib.parse.quote(k)}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "mlb-predictor/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code == 429:
                    continue  # this key is limited — try the next key
                raise
            except Exception as e:
                last_exc = e
                continue
        # every key limited/failed this pass — wait and retry the whole set
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("odds-api.io request failed")


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
            market = _map_prop_market(raw_mkt)
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
    """Build the sharp no-vig probabilities for a game from Polymarket ONLY.
    Returns {ml_home, ml_away, total_line, p_over, p_under, ...} de-vigged.

    Polymarket-only on purpose (Jun 2026): the old Fanatics fallback made the
    "sharp" reference the de-vig of the bettable book itself. That could never
    produce a sharp-value bet (a book can't beat its own no-vig line), but it
    DID flip predict_core's has_sharp switch, silently suppressing the
    model's game lines while generating nothing in their place. No Polymarket
    market -> no sharp entry -> predict_core falls back to model pricing."""
    sharp: dict = {}
    poly = _parse_book_markets(books_raw.get("Polymarket"))

    ml = poly.get("moneyline")
    if ml:
        ph = _dec_to_prob(ml["home_dec"]); pa = _dec_to_prob(ml["away_dec"])
        if ph and pa and (ph + pa) > 0:
            sharp["ml_home"] = ph / (ph + pa)
            sharp["ml_away"] = pa / (ph + pa)
            sharp["ml_source"] = "Polymarket"
    tot = poly.get("total")
    if tot:
        po = _dec_to_prob(tot["over_dec"]); pu = _dec_to_prob(tot["under_dec"])
        if po and pu and (po + pu) > 0:
            sharp["total_line"] = tot["line"]
            sharp["p_over"] = po / (po + pu)
            sharp["p_under"] = pu / (po + pu)
            sharp["total_source"] = "Polymarket"
    # Run line (±1.5): de-vig the home/away decimal at the captured line.
    rl = poly.get("run_line")
    if rl:
        ph = _dec_to_prob(american_to_decimal_local(rl["home"]))
        pa = _dec_to_prob(american_to_decimal_local(rl["away"]))
        if ph and pa and (ph + pa) > 0:
            sharp["rl_line"] = rl["line"]
            sharp["p_home_cover"] = ph / (ph + pa)
            sharp["p_away_cover"] = pa / (ph + pa)
            sharp["rl_source"] = "Polymarket"
    return sharp


def american_to_decimal_local(odds) -> Optional[float]:
    try:
        o = int(odds)
    except (TypeError, ValueError):
        return None
    return (1.0 + o / 100.0) if o > 0 else (1.0 + 100.0 / (-o))


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
    # Keep ONLY the current slate (today + tomorrow UTC).
    #
    # BUG FIX (Jul 28 2026): this used to keep each matchup's EARLIEST calendar
    # day, which was fine when the feed only published a few days ahead. The
    # feed now returns the WHOLE REMAINING SEASON (~850 events): for any team
    # not playing today the "earliest day" is a future date, so ~300 events
    # survived -> ceil(300/10) = ~31 odds calls per refresh. At a 100/hour cap
    # two or three slate runs exhausted the budget and every later call 429'd,
    # silently killing player props for days at a time.
    #
    # A UTC day window covers the US slate (night games roll into tomorrow UTC)
    # and keeps BOTH games of a doubleheader; predict_core._find_book still
    # disambiguates twin games by commence_time. ~15-30 events -> 2-3 calls.
    _today = datetime.now(timezone.utc).date()
    _window = {_today.isoformat(), (_today + timedelta(days=1)).isoformat()}
    pending = [e for e in pending if str(e.get("date", ""))[:10] in _window]
    pending.sort(key=lambda e: e.get("date", ""))
    book_games: list[dict] = []
    props: list[dict] = []

    # Batch via /odds/multi (up to 10 event ids per call) — this keeps a full
    # slate to ~1 (events) + ceil(N/10) (odds) calls, well under the 100/hour
    # cap even with frequent refreshes.
    # Polymarket (the sharp reference) is a PAID-PLAN book on odds-api.io. On a
    # free plan, asking for it returns HTTP 403 for the WHOLE request — which
    # silently zeroed out games AND props for days (Jul 25-28 2026 outage).
    # Ask for it, but on a 403 drop it and retry with the bettable book only;
    # no sharp entry just means predict_core falls back to model pricing.
    global _BOOKS
    bk_param = ",".join(_BOOKS)
    for i in range(0, len(pending), 10):
        chunk = pending[i:i + 10]
        ids = ",".join(str(e["id"]) for e in chunk)
        try:
            batch = _get(f"odds/multi?eventIds={ids}&bookmakers={bk_param}", key)
        except urllib.error.HTTPError as e:
            if e.code == 403 and len(_BOOKS) > 1:
                print("[odds-api.io] sharp book unavailable on this plan — "
                      "continuing with the bettable book only.")
                _BOOKS = [bettable]
                bk_param = bettable
                try:
                    batch = _get(f"odds/multi?eventIds={ids}&bookmakers={bk_param}",
                                 key)
                except Exception:
                    continue
            else:
                continue
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
