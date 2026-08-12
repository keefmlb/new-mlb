"""META-MODEL: when the sim said a pick would hit, what separated the ones
that DID from the ones that DIDN'T?

The sim's ranking is the best rule we have, but it is only a probability. This
asks the next question: conditional on the sim being confident, which PRE-GAME
conditions actually predicted the outcome? Anything found here is information
the sim is not already using — i.e. a genuine lever on top-N quality.

Method:
  1. Take graded picks where the sim was confident (raw >= CONF).
  2. Join pre-game context: park factors, weather, umpire K tendency, the
     OPPOSING starter's quality, bullpen quality, day/night, plus features of
     the bet itself (line height, price, rank, the sim's own edge vs price).
  3. Split HIT vs MISS and compare each feature's distribution.
  4. Report the gap with a z-score, and a hit-rate table by quartile so the
     direction is legible rather than just "significant".

Only PRE-GAME fields are used — nothing derived from the result.

CAVEAT: many features are screened at once, so some will look significant by
chance. Treat this as hypothesis generation; a lever is only real if it holds
up on later data.

Run:  python -m scripts.sim_meta_model [market] [min_conf]
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PICKS = ROOT / "data" / "bets" / "sim_picks.json"
GAMES = ROOT / "data" / "games" / "games_2026.csv"
BOX = ROOT / "data" / "games" / "box_2026.csv"


def _prob(a):
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / ((-a) + 100.0)


def _usable(r):
    o = r.get("odds")
    return (r.get("outcome") in ("W", "L")
            and isinstance(o, (int, float)) and o and not (-100 < o < 100))


def build(market: str | None, min_conf: float) -> pd.DataFrame:
    picks = json.loads(PICKS.read_text(encoding="utf-8"))
    games = pd.read_csv(GAMES).drop_duplicates("game_pk").set_index("game_pk")
    box = (pd.read_csv(BOX)[["game_pk", "player_id", "side"]]
           .drop_duplicates(["game_pk", "player_id"])
           .set_index(["game_pk", "player_id"]))
    rows = []
    for r in picks:
        if not _usable(r):
            continue
        if market and r.get("market") != market:
            continue
        raw = r.get("sim_hit_raw", r.get("sim_hit"))
        if raw is None or float(raw) < min_conf:
            continue
        gpk = r.get("game_pk")
        pid = r.get("player_id")
        try:
            g = games.loc[int(gpk)]
        except Exception:
            continue
        # which side is the player on? -> who is the OPPOSING starter
        try:
            side = str(box.loc[(int(gpk), int(pid))]["side"])
        except Exception:
            side = ""
        is_home = side.lower().startswith("h")
        opp_sp_fip = g.get("away_sp_fip") if is_home else g.get("home_sp_fip")
        opp_bp_fip = g.get("away_bp_fip") if is_home else g.get("home_bp_fip")
        imp = _prob(r["odds"])
        rows.append({
            "won": 1 if r["outcome"] == "W" else 0,
            "market": r.get("market"),
            "sim_raw": float(raw),
            "rank": float(r.get("rank") or 10),
            "line": float(r.get("line") or 0),
            "implied": imp,
            "edge": float(raw) - imp,
            "is_home": 1.0 if is_home else 0.0,
            "opp_sp_fip": pd.to_numeric(opp_sp_fip, errors="coerce"),
            "opp_bp_fip": pd.to_numeric(opp_bp_fip, errors="coerce"),
            "park_pf_runs": pd.to_numeric(g.get("park_pf_runs"), errors="coerce"),
            "park_pf_hr": pd.to_numeric(g.get("park_pf_hr"), errors="coerce"),
            "temp_f": pd.to_numeric(g.get("temp_f"), errors="coerce"),
            "wind_to_cf": pd.to_numeric(g.get("wind_to_cf_mph"), errors="coerce"),
            "runs_mult": pd.to_numeric(g.get("runs_mult"), errors="coerce"),
            "ump_k_mult": pd.to_numeric(g.get("ump_k_mult"), errors="coerce"),
            "is_day": pd.to_numeric(g.get("is_day_game"), errors="coerce"),
            "elev": pd.to_numeric(g.get("park_elev_ft"), errors="coerce"),
        })
    return pd.DataFrame(rows)


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else None
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
    df = build(market, min_conf)
    if df.empty:
        print("No rows."); return
    hit = df[df.won == 1]
    miss = df[df.won == 0]
    print(f"Market: {market or 'ALL'}   sim raw >= {min_conf:.0%}")
    print(f"n={len(df)}   HIT={len(hit)} ({len(hit)/len(df):.1%})   "
          f"MISS={len(miss)}")

    feats = [c for c in df.columns if c not in ("won", "market")]
    print(f"\n{'feature':16s} {'HIT mean':>10} {'MISS mean':>10} {'gap':>9} "
          f"{'z':>7}")
    print("-" * 58)
    scored = []
    for f in feats:
        a, b = hit[f].dropna(), miss[f].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = math.sqrt(va / len(a) + vb / len(b))
        if not se:
            continue
        z = (a.mean() - b.mean()) / se
        scored.append((abs(z), f, a.mean(), b.mean(), z))
    scored.sort(reverse=True)
    for _, f, am, bm, z in scored:
        star = "  <-- " + ("higher when it HITS" if z > 0 else "higher when it MISSES") if abs(z) >= 2 else ""
        print(f"{f:16s} {am:10.3f} {bm:10.3f} {am-bm:+9.3f} {z:+7.2f}{star}")

    # direction check for the strongest features: hit rate by quartile
    print("\nHIT RATE BY QUARTILE (strongest features)")
    for _, f, *_ in scored[:4]:
        s = df[[f, "won"]].dropna()
        if s[f].nunique() < 4:
            continue
        try:
            s["q"] = pd.qcut(s[f], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                             duplicates="drop")
        except Exception:
            continue
        g = s.groupby("q", observed=True)["won"].agg(["mean", "size"])
        cells = "   ".join(f"{ix}: {row['mean']:.0%} (n={int(row['size'])})"
                           for ix, row in g.iterrows())
        print(f"  {f:16s} {cells}")

    print("\nCAVEAT: ~13 features screened, so |z|>=2 will appear by chance "
          "roughly once\nper run. Only act on a lever that repeats on new data.")


if __name__ == "__main__":
    main()
