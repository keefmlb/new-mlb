"""Honest bet-log report: record, ROI, and CLV by market and by date.

This replaces the old guard-simulation script, which estimated the "impact"
of proposed filters by re-scoring the same logged bets that motivated them —
including one guard that literally hardcoded the names of players whose bets
had lost. That is selection on the dependent variable: removing known losers
always "improves" the record, and it says nothing about future bets.

This version only REPORTS. Policy decisions should come from:
  - CLV (closing line value) — converges much faster than win/loss,
  - settled record under a FROZEN policy (see policy_version on log entries),
  - mechanisms you can articulate before looking at outcomes.

NOTE: if the log predates the Jun 9 2026 grading fixes (away run-line sign
bug, push handling), re-grade first:  python -m src.bet_tracker regrade

Run:  python -m scripts.analyze_bets
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import bet_tracker
from src.value import american_to_decimal


def _direction(b: dict) -> str:
    desc = (b.get("description") or "").upper()
    if "UNDER" in desc:
        return "UNDER"
    if "OVER" in desc:
        return "OVER"
    return ""


def _roi(bets: list[dict]) -> float | None:
    """Realised ROI per $1 flat-staked across settled bets (pushes = $0)."""
    staked = profit = 0.0
    for b in bets:
        o = b.get("outcome")
        if o not in ("W", "L"):
            continue
        try:
            d = american_to_decimal(int(b.get("odds") or -110))
        except (TypeError, ValueError):
            continue
        staked += 1.0
        profit += (d - 1.0) if o == "W" else -1.0
    return (profit / staked) if staked else None


def _line(label: str, bets: list[dict]) -> str:
    w = sum(1 for b in bets if b.get("outcome") == "W")
    l = sum(1 for b in bets if b.get("outcome") == "L")
    p = sum(1 for b in bets if b.get("outcome") == "P")
    dec = w + l
    wr = f"{w / dec:5.0%}" if dec else "    —"
    roi = _roi(bets)
    roi_s = f"{roi:+7.1%}" if roi is not None else "      —"
    push = f" {p}P" if p else ""
    ci = bet_tracker.wilson_ci(w, dec)
    ci_s = f"  CI[{ci[0]:.0%}-{ci[1]:.0%}]" if ci else ""
    return f"  {label:38s} {w}W {l}L{push}  {wr}{ci_s}  ROI {roi_s}  (n={dec})"


def main():
    full_log = bet_tracker._load_log()
    if not full_log:
        print("Bet log is empty (data/bets/bet_log.json).")
        return
    # Shadow picks (every floor-clearing bet, logged since Jun 10 2026 to
    # tighten CIs) are excluded from the headline record but reported in
    # their own section below with primary+shadow pooled per market.
    log = [b for b in full_log if not b.get("shadow")]
    shadow = [b for b in full_log if b.get("shadow")]
    settled = [b for b in log if b.get("outcome") in ("W", "L", "P")]
    pending = sum(1 for b in log if b.get("outcome") is None)
    print(f"Logged: {len(log)}  settled: {len(settled)}  pending: {pending}"
          + (f"  (+{len(shadow)} shadow picks)" if shadow else ""))

    # Flag un-regraded logs: entries graded before the Jun 9 2026 fixes.
    if any(b.get("outcome") in ("W", "L") and b.get("market") == "run_line"
           for b in settled):
        print("\nREMINDER: away run-line grading was buggy before Jun 9 2026.")
        print("If this log predates that fix, run: python -m src.bet_tracker regrade\n")

    print("=" * 72)
    print("BY MARKET / DIRECTION")
    print("=" * 72)
    by_key: dict[str, list] = defaultdict(list)
    for b in settled:
        d = _direction(b)
        key = f"{b.get('market', 'unknown')}{'_' + d if d else ''}"
        by_key[key].append(b)
    for k, bets in sorted(by_key.items(), key=lambda x: -len(x[1])):
        print(_line(k, bets))
    print(_line("TOTAL", settled))

    print("\n" + "=" * 72)
    print("BY POLICY VERSION  (only compare records WITHIN a version)")
    print("=" * 72)
    by_pol: dict[str, list] = defaultdict(list)
    for b in settled:
        by_pol[str(b.get("policy_version") or "pre-tagging")].append(b)
    for k in sorted(by_pol):
        print(_line(k, by_pol[k]))

    print("\n" + "=" * 72)
    print("BY DATE")
    print("=" * 72)
    by_date: dict[str, list] = defaultdict(list)
    for b in settled:
        by_date[b.get("date", "?")].append(b)
    for dt in sorted(by_date):
        print(_line(dt, by_date[dt]))

    shadow_settled = [b for b in shadow if b.get("outcome") in ("W", "L", "P")]
    if shadow_settled:
        print("\n" + "=" * 72)
        print("BY MARKET — PRIMARY + SHADOW POOLED  (CI-tightening sample;")
        print("shadow picks are every floor-clearing bet, not the betting record)")
        print("=" * 72)
        pooled = settled + shadow_settled
        by_key2: dict[str, list] = defaultdict(list)
        for b in pooled:
            d = _direction(b)
            key = f"{b.get('market', 'unknown')}{'_' + d if d else ''}"
            by_key2[key].append(b)
        for k, bets in sorted(by_key2.items(), key=lambda x: -len(x[1])):
            print(_line(k, bets))
        print(_line("TOTAL (pooled)", pooled))

    print("\n" + "=" * 72)
    print("CLOSING LINE VALUE (bets with a captured close)")
    print("=" * 72)
    clv = bet_tracker.get_clv_summary(days=365)

    def _print_clv(label: str, c: dict, empty_note: str) -> None:
        if c.get("n"):
            pbc = c.get("pct_beat_close")
            ac = c.get("avg_clv_pct")
            line = f"  {label:<12} n={c['n']}"
            if pbc is not None:
                line += f"  beat close: {pbc:.0%}"
            if ac is not None:
                line += f"  avg CLV: {ac:+.2f} prob-pts (n_priced={c.get('n_priced')})"
            print(line)
        else:
            print(f"  {label:<12} {empty_note}")

    _print_clv("game lines", clv,
               "No CLV data yet (closing-line capture began Jun 4 2026).")
    _print_clv("props", clv.get("props", {}),
               "No CLV data yet (closing-prop capture began Jun 10 2026).")


if __name__ == "__main__":
    main()
