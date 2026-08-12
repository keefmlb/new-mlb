"""Does batter-vs-pitch-type matchup predict our hits and misses?

Hypothesis (user's): pitchers throw a fixed arsenal, batters handle some pitch
types better than others, so a batter facing a starter whose mix suits him
should outperform his generic projection. Nothing in the model captures this —
pitcher arsenal is currently used ONLY to predict the pitcher's own stats, and
batter-vs-pitch-type data was never fetched at all.

Matchup score:
    expected_woba = sum_p ( pitcher_usage(p) * batter_woba_vs(p) ) / sum_p usage
    matchup_delta = expected_woba - batter_overall_woba
Positive delta = this starter throws pitches this batter hits well.

The delta is what matters, not the level: the level is already baked into the
batter's projection, so only the pitcher-specific deviation is new information.

LEAKAGE CAVEAT (important): Savant splits are SEASON-TO-DATE, which includes
the games being predicted. That biases the test IN FAVOUR of the hypothesis.
So a positive result here is an optimistic upper bound and would still need a
point-in-time rebuild before trusting it — but a NULL result is decisive,
because the feature couldn't even work with hindsight helping it.

Run:  python -m scripts.pitch_matchup_research [market] [min_conf]
"""
from __future__ import annotations
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bet_tracker import roi_ci  # noqa: E402
from src import statcast  # noqa: E402

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
GAMES = ROOT / "data" / "games" / "games_2026.csv"
BOX = ROOT / "data" / "games" / "box_2026.csv"
CACHE = ROOT / "data" / "cache"
_ARSENAL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"

# pitcher-arsenal column prefixes -> Savant batter pitch_type codes
_CODES = {"ff": "FF", "si": "SI", "fc": "FC", "sl": "SL",
          "cu": "CU", "ch": "CH", "fs": "FS", "st": "ST"}


def _batter_by_pitch(year: int, min_pa: int = 10) -> dict:
    """{batter_id: {PITCH_TYPE: {'woba': x, 'pa': n}}} plus '_ALL' overall."""
    cache = CACHE / f"batter_pitch_type_{year}.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
    else:
        p = urllib.parse.urlencode({"type": "batter", "pitchType": "",
                                    "year": year, "team": "", "min": min_pa,
                                    "csv": "true"})
        req = urllib.request.Request(
            f"{_ARSENAL}?{p}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; mlb-predictor/1.0)",
                     "Accept": "text/csv,*/*"})
        with urllib.request.urlopen(req, timeout=60) as r:
            df = pd.read_csv(io.StringIO(r.read().decode("utf-8")))
        raw = df.to_dict(orient="records")
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(raw), encoding="utf-8")

    out: dict = {}
    for row in raw:
        try:
            pid = int(row["player_id"])
            pt = str(row["pitch_type"])
            woba = float(row["woba"])
            pa = float(row["pa"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(pid, {})[pt] = {"woba": woba, "pa": pa}
    # overall wOBA per batter = PA-weighted mean across his pitch types
    for pid, d in out.items():
        tot = sum(v["pa"] for v in d.values())
        if tot > 0:
            d["_ALL"] = {"woba": sum(v["woba"] * v["pa"] for v in d.values()) / tot,
                         "pa": tot}
    return out


def _pitcher_mix(year: int) -> dict:
    """{pitcher_id: {PITCH_TYPE: usage_pct}} from the same leaderboard the
    model already pulls for arsenal shape."""
    sel = ",".join(f"n_{c}_formatted" for c in _CODES)
    try:
        df = statcast._fetch_leaderboard(year, "pitcher", sel, 30,
                                         cache_key="mix_for_matchup")
    except Exception as exc:
        print(f"[statcast] pitcher mix fetch failed: {exc}")
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        try:
            pid = int(row["player_id"])
        except (KeyError, TypeError, ValueError):
            continue
        mix = {}
        for c, code in _CODES.items():
            v = row.get(f"n_{c}_formatted")
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v and v > 0:
                mix[code] = v
        if mix:
            out[pid] = mix
    return out


def matchup_delta(bat: dict, mix: dict, min_pa: int = 15) -> float | None:
    """Usage-weighted wOBA this batter posts against this arsenal, minus his
    own overall wOBA. Pitch types where he has too few PA fall back to his
    overall (contributing zero delta) rather than injecting noise."""
    if not bat or not mix:
        return None
    overall = (bat.get("_ALL") or {}).get("woba")
    if overall is None:
        return None
    num = den = 0.0
    for pt, usage in mix.items():
        rec = bat.get(pt)
        w = rec["woba"] if rec and rec["pa"] >= min_pa else overall
        num += usage * w
        den += usage
    if den <= 0:
        return None
    return (num / den) - overall


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "prop_tb"
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    bats = _batter_by_pitch(2026)
    mixes = _pitcher_mix(2026)
    print(f"batters with pitch-type splits: {len(bats)}   "
          f"pitchers with mix: {len(mixes)}")

    games = pd.read_csv(GAMES).drop_duplicates("game_pk").set_index("game_pk")
    box = (pd.read_csv(BOX, usecols=["game_pk", "player_id", "side"])
           .drop_duplicates(["game_pk", "player_id"])
           .set_index(["game_pk", "player_id"]))

    def _prob(a):
        a = float(a)
        return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)

    def _dec(a):
        a = float(a)
        return 1.0 + (a / 100.0 if a > 0 else 100.0 / (-a))

    rows = []
    for r in json.loads(PICKS.read_text(encoding="utf-8")):
        o = r.get("odds")
        if (r.get("outcome") not in ("W", "L") or r.get("market") != market
                or not isinstance(o, (int, float)) or not o or -100 < o < 100):
            continue
        raw = r.get("sim_hit_raw", r.get("sim_hit"))
        if raw is None or float(raw) < min_conf:
            continue
        gpk, pid = int(r["game_pk"]), int(r.get("player_id") or 0)
        try:
            g = games.loc[gpk]
            side = str(box.loc[(gpk, pid)]["side"])
        except Exception:
            continue
        is_home = side.lower().startswith("h")
        spid = g.get("away_sp_id") if is_home else g.get("home_sp_id")
        try:
            spid = int(spid)
        except (TypeError, ValueError):
            continue
        delta = matchup_delta(bats.get(pid), mixes.get(spid))
        if delta is None:
            continue
        rows.append({"won": 1 if r["outcome"] == "W" else 0,
                     "prof": (_dec(o) - 1.0) if r["outcome"] == "W" else -1.0,
                     "delta": delta, "raw": float(raw)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows matched."); return
    hit, miss = df[df.won == 1], df[df.won == 0]
    se = math.sqrt(hit.delta.var(ddof=1) / len(hit)
                   + miss.delta.var(ddof=1) / len(miss))
    z = (hit.delta.mean() - miss.delta.mean()) / se if se else 0.0
    print(f"\n{market}, sim>= {min_conf:.0%}   n={len(df)}  hit {df.won.mean():.1%}")
    print(f"  matchup_delta  HIT {hit.delta.mean():+.4f}   "
          f"MISS {miss.delta.mean():+.4f}   z={z:+.2f}"
          + ("   <-- signal" if abs(z) >= 2 else "   (no signal)"))

    print("\nBY MATCHUP DELTA (higher = arsenal suits the batter)")
    try:
        df["q"] = pd.qcut(df.delta, 4, labels=["Q1 worst", "Q2", "Q3",
                                               "Q4 best"], duplicates="drop")
    except Exception:
        return
    for q, g in df.groupby("q", observed=True):
        ci = roi_ci(list(g.prof))
        cis = f"[{ci[0]:+.0%},{ci[1]:+.0%}]" if ci else ""
        print(f"  {str(q):9s} n={len(g):3d}  hit {g.won.mean():5.1%}  "
              f"ROI {g.prof.mean():+7.1%} {cis}")
    print("\nNOTE: season-to-date splits include the graded games, so this test "
          "is biased\nTOWARD the hypothesis. A null here is therefore decisive; "
          "a positive would\nstill need a point-in-time rebuild to trust.")


if __name__ == "__main__":
    main()
