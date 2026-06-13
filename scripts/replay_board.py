"""Replay the FULL prop board under the current policy — every offer, real odds.

replay_policy.py answers "which of our old top-10 picks survive today's
filter?" — useful but capped at ~120 settled props, so its confidence
intervals are ±20 points. This script attacks the n problem directly: the
14-day odds history (data/odds/odds_history.json) stores EVERY Fanatics prop
offer (~4,000/day with real over/under prices). We reprice all of them with
the current pipeline, keep what clears today's floors and guards, and grade
against actual boxscore outcomes. Hundreds of bets per day instead of ten.

Method, per snapshot day:
  1. Take the EARLIEST substantial snapshot before 16:00 ET (pre-game lines;
     evening snapshots are mid-slate and would mix in live prices).
  2. Match each offer to its game (team names -> games_2026.csv) and player
     (name_match -> props_{bat,pit}_2026.csv rows, which carry the leak-free
     analytical projection AND the actual stat).
  3. Price with value.evaluate_prop — the production path: NegBin tail ->
     fitted per-market calibration -> 50/50 market blend -> edge.
  4. Apply the current guards (K OVER block, K UNDER high-line, K edge cap,
     short-start, workhorse outs, the production one-sided +400 cap) and the
     per-market edge floors.
  5. Grade survivors: over wins if actual > line, push if equal.

Reported with Wilson 95% CIs. Also reports the "daily top-10 by score"
subset — the actual betting policy — inside the same sample.

Honest caveats:
  - Only days present in the rolling 14-day odds history can be replayed.
    The sample grows every day the slate runs; re-run this any time.
  - Projections are analytical means (no live ML blend) — same
    approximation as fit_prop_calibration, and the prop calibration was
    FITTED on a pool that includes these same weeks (mild in-sample
    flattery for a 2-parameter fit; the guards/floors are unaffected).
  - Statcast-based guards (elite-arm K UNDER, TB barrel) can't be applied
    retroactively without leaking today's stats; line-based parts are.
  - DNP players simply have no boxscore row -> treated as void (skipped),
    matching how books void DNP props.

Run:  python -m scripts.replay_board
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import name_match, value
from src.predict_core import _effective_edge_threshold_pct

HIST = ROOT / "data" / "odds" / "odds_history.json"
# Permanent archive of graded full-board replay bets. odds_history.json is a
# ROLLING 14-day window — without this, every replayed day ages out of the
# sample two weeks later and the CIs stop tightening. Days are re-replayed
# from the live window when present (policy may have changed) and the
# archive supplies the days that have rolled off.
ARCHIVE = ROOT / "data" / "odds" / "replay_archive.csv"
ARCHIVE_FIELDS = ["date", "market", "description", "line", "odds",
                  "edge_pct", "score", "outcome", "policy_version"]
SLIDER_PCT = 3.0
ET_OFFSET = timedelta(hours=-4)          # EDT (June)
MIN_PROPS_SNAPSHOT = 1000
LATEST_ET_HOUR = 16                      # pre-game cutoff

# raw market -> (csv kind, projection column, actual column)
MKT = {
    "hits":         ("bat", "proj_h",        "h"),
    "hr":           ("bat", "proj_hr",       "hr"),
    "tb":           ("bat", "proj_tb",       "tb"),
    "rbi":          ("bat", "proj_rbi",      "rbi"),
    "runs":         ("bat", "proj_runs",     "runs_b"),
    "k":            ("bat", "proj_k",        "k_b"),
    "bb":           ("bat", "proj_bb",       "bb_b"),
    "pitcher_k":    ("pit", "proj_k",        "k_p"),
    "pitcher_bb":   ("pit", "proj_bb",       "bb_p"),
    "pitcher_h":    ("pit", "proj_h",        "h_p"),
    "pitcher_er":   ("pit", "proj_er",       "er"),
    "pitcher_hr":   ("pit", "proj_hr",       "hr_p"),
    "pitcher_outs": ("pit", "expected_outs", "outs"),
}


def _select_snapshots() -> dict[str, dict]:
    """{slate_date_iso: snapshot} — earliest substantial pre-game capture/day."""
    hist = json.loads(HIST.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for snap in hist:
        if len(snap.get("props", [])) < MIN_PROPS_SNAPSHOT:
            continue
        ts = datetime.fromisoformat(snap["ts"].replace("Z", "+00:00"))
        et = ts + ET_OFFSET
        if et.hour >= LATEST_ET_HOUR:
            continue
        day = et.date().isoformat()
        if day not in out or snap["ts"] < out[day]["ts"]:
            out[day] = snap
    return out


def _team_match(needle: str, haystack: str) -> bool:
    n, h = needle.lower().strip(), haystack.lower().strip()
    return bool(n) and (n in h or h in n)


def _build_indexes():
    games = pd.read_csv(ROOT / "data" / "games" / "games_2026.csv",
                        usecols=["game_pk", "date", "home_team", "away_team", "is_final"])
    games_by_date: dict[str, list] = defaultdict(list)
    for r in games.itertuples():
        games_by_date[str(r.date)].append(r)

    bat = pd.read_csv(ROOT / "data" / "games" / "props_bat_2026.csv")
    pit = pd.read_csv(ROOT / "data" / "games" / "props_pit_2026.csv")
    by_game: dict[str, dict[int, dict[str, pd.Series]]] = {"bat": defaultdict(dict),
                                                           "pit": defaultdict(dict)}
    for kind, df in (("bat", bat), ("pit", pit)):
        for r in df.itertuples():
            by_game[kind][int(r.game_pk)][str(r.name)] = r
    return games_by_date, by_game


def _find_game(games_by_date, day: str, game_str: str):
    """Resolve 'Away @ Home' city names to a games_2026 row on day or day±1."""
    if " @ " not in (game_str or ""):
        return None
    away, home = game_str.split(" @ ", 1)
    d0 = datetime.fromisoformat(day).date()
    for d in (d0, d0 - timedelta(days=1), d0 + timedelta(days=1)):
        for g in games_by_date.get(d.isoformat(), []):
            if _team_match(away, g.away_team) and _team_match(home, g.home_team):
                return g
    return None


def _grade(desc: str, line: float, actual: float) -> str:
    if actual == line:
        return "P"
    over = actual > line
    return "W" if (over == (" OVER " in desc)) else "L"


def _record(bets: list[dict]) -> tuple[int, int, int, float | None]:
    w = sum(1 for b in bets if b["outcome"] == "W")
    l = sum(1 for b in bets if b["outcome"] == "L")
    p = sum(1 for b in bets if b["outcome"] == "P")
    staked = profit = 0.0
    for b in bets:
        if b["outcome"] not in ("W", "L"):
            continue
        d = value.american_to_decimal(int(b["odds"]))
        staked += 1.0
        profit += (d - 1.0) if b["outcome"] == "W" else -1.0
    return w, l, p, (profit / staked if staked else None)


def _fmt(label: str, bets: list[dict]) -> str:
    from src.bet_tracker import wilson_ci, roi_ci, bet_profit
    w, l, p, roi = _record(bets)
    dec = w + l
    wr = f"{w/dec:5.0%}" if dec else "    —"
    ci = wilson_ci(w, dec)
    ci_s = f" CI[{ci[0]:.0%}-{ci[1]:.0%}]" if ci else ""
    roi_s = f"{roi:+7.1%}" if roi is not None else "      —"
    profs = [bp for b in bets if (bp := bet_profit(b.get("odds"), b["outcome"])) is not None]
    rci = roi_ci(profs)
    rci_s = f" roiCI[{rci[0]:+.0%},{rci[1]:+.0%}]" if rci else ""
    push = f" {p}P" if p else ""
    return f"  {label:34s} {w}W {l}L{push}  {wr}{ci_s}  ROI {roi_s}{rci_s}  (n={dec})"


def main():
    snaps = _select_snapshots()
    if not snaps:
        print("No usable snapshots in odds_history.json.")
        return
    games_by_date, by_game = _build_indexes()

    all_bets: list[dict] = []
    pending_days: list[str] = []
    for day in sorted(snaps):
        props = snaps[day]["props"]
        day_bets: list[dict] = []
        n_nogame = n_noplayer = n_skipmkt = 0
        for pp in props:
            market = pp.get("market", "")
            if market not in MKT:
                n_skipmkt += 1
                continue
            g = _find_game(games_by_date, day, pp.get("game", ""))
            if g is None:
                n_nogame += 1
                continue
            kind, proj_col, act_col = MKT[market]
            rows = by_game[kind].get(int(g.game_pk), {})
            resolved = name_match.find_match(pp.get("player", ""), rows.keys())
            row = rows.get(resolved) if resolved else None
            if row is None:
                n_noplayer += 1    # includes DNP -> void, correctly skipped
                continue
            mean = getattr(row, proj_col, None)
            if mean is None or pd.isna(mean) or float(mean) <= 0:
                continue
            mean = float(mean)
            line = float(pp.get("line") or 0)
            vbs = value.evaluate_prop(pp.get("player", ""), market, mean, line,
                                      pp.get("over"), pp.get("under"),
                                      edge_threshold=-1.0)
            exp_outs = getattr(row, "expected_outs", None) if kind == "pit" else None
            for vb in vbs:
                desc = vb.description
                # --- current guards (line-computable parts) ---
                if market == "pitcher_k":
                    if " OVER " in desc:
                        continue                       # K OVER blanket block
                    if line >= 6.0 and " UNDER " in desc:
                        continue                       # K UNDER high-line guard
                    if vb.edge_pct > 15.0:
                        continue                       # K edge cap
                if (market == "pitcher_outs" and " UNDER " in desc
                        and (line >= 17.5
                             or (exp_outs is not None and not pd.isna(exp_outs)
                                 and float(exp_outs) >= 16.5))):
                    continue                           # workhorse outs guard
                if (kind == "pit" and " OVER " in desc
                        and exp_outs is not None and not pd.isna(exp_outs)
                        and float(exp_outs) < 14.5):
                    continue                           # short-start guard
                # --- current edge floor ---
                if vb.edge_pct < _effective_edge_threshold_pct(vb.market, SLIDER_PCT):
                    continue
                actual = getattr(row, act_col, None)
                if actual is None or pd.isna(actual):
                    continue
                day_bets.append({
                    "date": day, "market": vb.market, "description": desc,
                    "line": line, "odds": vb.odds, "edge_pct": vb.edge_pct,
                    "score": vb.score, "outcome": _grade(desc, line, float(actual)),
                })
        if day_bets:
            all_bets.extend(day_bets)
        else:
            pending_days.append(day)   # e.g. today's games not yet final
        print(f"  {day}: offers={len(props)}  bets={len(day_bets)}  "
              f"(no-game={n_nogame}, no-player/DNP={n_noplayer}, other-mkt={n_skipmkt})")

    # Merge with the permanent archive: freshly replayed days REPLACE their
    # archived versions (policy may have changed); archived days that have
    # rolled out of the 14-day odds history are added back.
    import csv as _csv
    from src.predict_core import POLICY_VERSION
    archived: list[dict] = []
    if ARCHIVE.exists():
        with ARCHIVE.open("r", encoding="utf-8", newline="") as fh:
            archived = list(_csv.DictReader(fh))
    fresh_days = {b["date"] for b in all_bets}
    carried = [dict(b, line=float(b["line"]), odds=int(b["odds"]),
                    edge_pct=float(b["edge_pct"]),
                    score=float(b["score"]) if b.get("score") else 0.0)
               for b in archived if b.get("date") not in fresh_days]
    if carried:
        print(f"\n  + {len(carried)} archived bets from "
              f"{len({b['date'] for b in carried})} rolled-off day(s)")
    all_bets = sorted(all_bets, key=lambda b: b["date"])
    merged = sorted(carried + all_bets, key=lambda b: b["date"])

    try:
        with ARCHIVE.open("w", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=ARCHIVE_FIELDS)
            w.writeheader()
            for b in merged:
                row = {k: b.get(k, "") for k in ARCHIVE_FIELDS}
                if not row.get("policy_version"):
                    row["policy_version"] = POLICY_VERSION
                w.writerow(row)
        print(f"  archive -> {ARCHIVE.name} ({len(merged)} bets)")
    except Exception as e:
        print(f"  archive write failed: {e}")
    all_bets = merged

    if not all_bets:
        print("No gradable bets.")
        return

    print("\n" + "=" * 78)
    print("FULL-BOARD REPLAY UNDER CURRENT POLICY — every floor-clearing offer")
    print("=" * 78)
    print(_fmt("ALL floor-clearing bets", all_bets))

    print("\n  By market / direction:")
    by_mkt: dict[str, list] = defaultdict(list)
    for b in all_bets:
        side = "OVER" if " OVER " in b["description"] else "UNDER"
        by_mkt[f"{b['market']}_{side}"].append(b)
    for k in sorted(by_mkt, key=lambda k: -len(by_mkt[k])):
        print(_fmt(f"  {k}", by_mkt[k]))

    print("\n  By day:")
    by_day: dict[str, list] = defaultdict(list)
    for b in all_bets:
        by_day[b["date"]].append(b)
    for d in sorted(by_day):
        print(_fmt(f"  {d}", by_day[d]))

    top10: list[dict] = []
    for d in sorted(by_day):
        top10.extend(sorted(by_day[d], key=lambda b: -(b["score"] or 0))[:10])
    print("\n  Actual betting policy (daily top-10 by score) inside this sample:")
    print(_fmt("  top-10/day subset", top10))

    print("\n  Caveats: analytical projections (no live ML blend); prop")
    print("  calibration was fitted on a pool including these weeks; Statcast")
    print("  guards not applied. Re-run as more snapshot days accumulate.")


if __name__ == "__main__":
    main()
