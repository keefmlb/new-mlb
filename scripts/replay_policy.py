"""Replay the season's logged bets under the CURRENT policy.

Answers "what would our numbers have been if today's pipeline had been
running all season?" — honestly, within the limits of what we have prices
for. Real book odds exist only for bets that were actually logged, so the
replay universe is the settled bet log. Each prop bet is REPRICED from
scratch with the current pipeline:

    projection mean (props_{bat,pit}_2026.csv, leak-free weekly snapshots)
      -> NegBin tail prob (current mu-conditional dispersion fits)
      -> fitted per-market calibration (prop_calibration.json)
      -> 50/50 blend toward the bet-time book no-vig prob
      -> current per-market edge floors + market/direction guards

A bet is KEPT if the repriced edge clears today's floors and guards, and
DROPPED otherwise. Kept/dropped records use the already-settled outcomes.

Honest caveats (read before quoting the numbers):
  - SELECTION: the universe is bets the OLD policies surfaced. The current
    policy would also have surfaced bets the old one never logged; we have
    no odds for those, so this replay measures the FILTER, not the full
    strategy. Directionally useful, not a P&L simulation.
  - Projections are the ANALYTICAL means (the CSVs don't store the live
    analytical+ML blend). Same approximation the calibration fit makes.
  - Player-level Statcast guards (elite-arm K, TB barrel) and the
    disagreement/short-start guards that need slate context can only be
    partially applied: line-based parts yes, Statcast parts no.
  - Game lines (ML / total / run line) are NOT repriced: under the current
    setup every Fanatics game carries a Polymarket sharp reference, the
    sharp veto suppresses model game-line bets, and the sharp detector has
    never fired (Jun 4-10 audit: best side -0.2% EV). So the current-policy
    game-line volume is approximately ZERO; their historical record is
    shown separately as what the veto walks away from.

Run:  python -m scripts.replay_policy
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import bet_tracker, value
from src.predict_core import _effective_edge_threshold_pct

SLIDER_PCT = 3.0          # app default edge slider
PROP_BLEND = value.PROP_MARKET_BLEND_WEIGHT

# log market -> (csv kind, projection column)
PROJ_COL = {
    "prop_hits":        ("bat", "proj_h"),
    "prop_hr":          ("bat", "proj_hr"),
    "prop_tb":          ("bat", "proj_tb"),
    "prop_rbi":         ("bat", "proj_rbi"),
    "prop_runs":        ("bat", "proj_runs"),
    "prop_k":           ("bat", "proj_k"),
    "prop_bb":          ("bat", "proj_bb"),
    "prop_pitcher_k":    ("pit", "proj_k"),
    "prop_pitcher_bb":   ("pit", "proj_bb"),
    "prop_pitcher_h":    ("pit", "proj_h"),
    "prop_pitcher_er":   ("pit", "proj_er"),
    "prop_pitcher_hr":   ("pit", "proj_hr"),
    "prop_pitcher_outs": ("pit", "expected_outs"),
}

GAME_MARKETS = ("moneyline", "total", "run_line")


def _roi(bets: list[dict]) -> tuple[int, int, int, float | None]:
    w = sum(1 for b in bets if b.get("outcome") == "W")
    l = sum(1 for b in bets if b.get("outcome") == "L")
    p = sum(1 for b in bets if b.get("outcome") == "P")
    staked = profit = 0.0
    for b in bets:
        if b.get("outcome") not in ("W", "L"):
            continue
        d = value.american_to_decimal(int(b.get("odds") or -110))
        staked += 1.0
        profit += (d - 1.0) if b["outcome"] == "W" else -1.0
    return w, l, p, (profit / staked if staked else None)


def _fmt(label: str, bets: list[dict]) -> str:
    w, l, p, roi = _roi(bets)
    dec = w + l
    wr = f"{w/dec:5.0%}" if dec else "    —"
    roi_s = f"{roi:+7.1%}" if roi is not None else "      —"
    push = f" {p}P" if p else ""
    return f"  {label:42s} {w}W {l}L{push}  {wr}  ROI {roi_s}  (n={dec})"


def replay_prop(entry: dict, proj_row: pd.Series | None) -> tuple[str, str, float | None]:
    """Reprice one logged prop under current policy.

    Returns (verdict, reason, new_edge_pct) where verdict is "kept",
    "dropped", or "unpriceable"."""
    market = entry["market"]
    raw_mkt = market[len("prop_"):]
    desc = entry.get("description") or ""
    is_over = " OVER " in desc
    is_under = " UNDER " in desc
    if not (is_over or is_under):
        return "unpriceable", "no side in description", None
    if proj_row is None:
        return "unpriceable", "no projection row", None

    kind, col = PROJ_COL[market]
    mean = proj_row.get(col)
    try:
        mean = float(mean)
    except (TypeError, ValueError):
        return "unpriceable", "projection NaN", None
    if not mean or mean <= 0 or pd.isna(mean):
        return "unpriceable", "projection NaN", None

    line = float(entry.get("line") or 0)
    nv = entry.get("novig_prob")
    if nv is None:
        return "unpriceable", "no novig prob", None
    odds = int(entry.get("odds") or 0)
    one_sided = "1-sided" in desc

    # --- current pricing ---
    disp = value.get_dispersion(raw_mkt, mean)
    p_over = value.calibrate_prop_prob(value.prob_over_count(mean, line, disp), raw_mkt)
    p_side = p_over if is_over else 1.0 - p_over
    p_blend = (1 - PROP_BLEND) * p_side + PROP_BLEND * float(nv)
    edge_pct = (p_blend - float(nv)) * 100.0

    # --- current guards (the parts computable retroactively) ---
    if one_sided and odds > 400:
        return "dropped", "one-sided +400 longshot cap", edge_pct
    if market == "prop_pitcher_k":
        if is_over:
            return "dropped", "K OVER blanket block", edge_pct
        if line >= 6.0:
            return "dropped", "K UNDER high-line guard", edge_pct
        if edge_pct > 15.0:
            return "dropped", "K edge cap 15%", edge_pct
    if market == "prop_pitcher_outs" and is_under:
        exp_outs = proj_row.get("expected_outs")
        if line >= 17.5 or (pd.notna(exp_outs) and float(exp_outs) >= 16.5):
            return "dropped", "workhorse outs UNDER guard", edge_pct
    if kind == "pit" and is_over:
        exp_outs = proj_row.get("expected_outs")
        if pd.notna(exp_outs) and float(exp_outs) < 14.5:
            return "dropped", "short-start OVER guard", edge_pct

    # --- current edge floor ---
    floor = _effective_edge_threshold_pct(market, SLIDER_PCT)
    if edge_pct < floor:
        bucket = ("repriced edge now NEGATIVE" if edge_pct < 0
                  else "repriced edge below current floor")
        return "dropped", bucket, edge_pct
    return "kept", "", edge_pct


def main():
    log = bet_tracker._load_log()
    settled = [b for b in log if b.get("outcome") in ("W", "L", "P")]
    if not settled:
        print("No settled bets to replay.")
        return

    bat = pd.read_csv(ROOT / "data" / "games" / "props_bat_2026.csv")
    pit = pd.read_csv(ROOT / "data" / "games" / "props_pit_2026.csv")
    bat_ix = {(int(r.game_pk), int(r.player_id)): r for r in bat.itertuples()}
    pit_ix = {(int(r.game_pk), int(r.player_id)): r for r in pit.itertuples()}

    props = [b for b in settled if str(b.get("market", "")).startswith("prop_")]
    games = [b for b in settled if b.get("market") in GAME_MARKETS]
    other = [b for b in settled if b not in props and b not in games]

    kept, dropped, unpriceable = [], [], []
    drop_reasons: dict[str, list[dict]] = defaultdict(list)
    for e in props:
        if e["market"] not in PROJ_COL:
            unpriceable.append(e)
            continue
        kind, _ = PROJ_COL[e["market"]]
        ix = bat_ix if kind == "bat" else pit_ix
        try:
            key = (int(e.get("game_pk") or 0), int(e.get("player_id") or 0))
        except (TypeError, ValueError):
            key = (0, 0)
        row = ix.get(key)
        prow = pd.Series(row._asdict()) if row is not None else None
        verdict, reason, edge = replay_prop(e, prow)
        e = dict(e, _new_edge=edge, _reason=reason)
        if verdict == "kept":
            kept.append(e)
        elif verdict == "dropped":
            dropped.append(e)
            drop_reasons[reason].append(e)
        else:
            unpriceable.append(e)

    print("=" * 76)
    print("SEASON REPLAY UNDER CURRENT POLICY (2026-06-10-prop-calibration)")
    print("=" * 76)
    print(f"Settled logged bets: {len(settled)}  "
          f"(props {len(props)}, game lines {len(games)}, other {len(other)})")

    print("\n--- PROPS, repriced with current pipeline " + "-" * 33)
    print(_fmt("ORIGINAL (all logged props)", props))
    print(_fmt("KEPT by current policy", kept))
    print(_fmt("DROPPED by current policy", dropped))
    if unpriceable:
        print(_fmt("unpriceable (no projection row)", unpriceable))

    print("\n  Kept, by market:")
    by_mkt: dict[str, list] = defaultdict(list)
    for e in kept:
        side = "OVER" if " OVER " in (e.get("description") or "") else "UNDER"
        by_mkt[f"{e['market']}_{side}"].append(e)
    for k in sorted(by_mkt, key=lambda k: -len(by_mkt[k])):
        print(_fmt(f"  {k}", by_mkt[k]))

    print("\n  Dropped, by reason (what the current policy walks away from):")
    for r in sorted(drop_reasons, key=lambda r: -len(drop_reasons[r])):
        print(_fmt(f"  {r}", drop_reasons[r]))

    print("\n--- GAME LINES, not repriced " + "-" * 46)
    print("  Under the current setup (Fanatics + Polymarket sharp on every")
    print("  game), the sharp veto suppresses ALL model game-line bets and the")
    print("  sharp detector has never fired -> current-policy volume here ~ 0.")
    print("  Historical record the veto walks away from:")
    print(_fmt("ALL game lines", games))
    for m in GAME_MARKETS:
        sub = [b for b in games if b.get("market") == m]
        if sub:
            print(_fmt(f"  {m}", sub))

    print("\n--- BOTTOM LINE " + "-" * 59)
    w, l, p, roi = _roi(settled)
    print(_fmt("Old policies, everything logged", settled))
    print(_fmt("Current policy (kept props only)", kept))
    print("\n  Caveat: replay measures the FILTER on old-policy bets; bets the")
    print("  current policy would have newly surfaced are invisible (no odds).")


if __name__ == "__main__":
    main()
