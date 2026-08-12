"""Per-team BULLPEN rate stats, computed from our own boxscores.

Why this exists: the team pitching stats the feature rows carry as
`bp_era` / `bp_fip` are actually TEAM-WIDE pitching (starters included), so
they cannot say how a bullpen differs from the starter it replaces. That
difference is real and one-directional — measured on 11,448 relief appearances,
bullpens walk noticeably MORE than starters:

    league bullpen BB/9 3.71   vs   starter BB/9 3.16   (+17%)
    team bullpen BB/9 spread: p10 3.13 -> p90 4.40 (0.84x - 1.18x of league)

The simulator swaps in a bullpen offense multiplier at the hook, but that
multiplier scales every on-base event uniformly, so a batter's WALK rate stays
pinned to the starter's control for the whole game. These rates let the sim
move walks to the relief corps' own control instead.

Relief rows are `started == False` with recorded outs.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOX = ROOT / "data" / "games" / "box_2026.csv"

# League relief baselines (recomputed on load; these are fallbacks).
LG_BP_BB9 = 3.71
LG_BP_K9 = 8.64
_MIN_IP = 50.0          # below this a team's rate is too noisy; use league

_CACHE: dict | None = None


def team_bullpen_rates(path: Path | None = None) -> dict:
    """{team_id: {'bb9':x, 'k9':y, 'ip':n}} plus '_league' with the same keys."""
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    src = path or BOX
    out: dict = {"_league": {"bb9": LG_BP_BB9, "k9": LG_BP_K9, "ip": 0.0}}
    try:
        import pandas as pd
        df = pd.read_csv(src, usecols=["team_id", "started", "outs", "bb_p", "k_p"])
    except Exception:
        if path is None:
            _CACHE = out
        return out
    rel = df[(df["started"] == False) & (df["outs"].fillna(0) > 0)]  # noqa: E712
    if rel.empty:
        if path is None:
            _CACHE = out
        return out
    tot_ip = rel["outs"].sum() / 3.0
    if tot_ip > 0:
        out["_league"] = {"bb9": rel["bb_p"].sum() / tot_ip * 9.0,
                          "k9": rel["k_p"].sum() / tot_ip * 9.0, "ip": tot_ip}
    for tid, d in rel.groupby("team_id"):
        ip = d["outs"].sum() / 3.0
        if ip < _MIN_IP:
            continue
        try:
            out[int(tid)] = {"bb9": d["bb_p"].sum() / ip * 9.0,
                             "k9": d["k_p"].sum() / ip * 9.0, "ip": ip}
        except (TypeError, ValueError):
            continue
    if path is None:
        _CACHE = out
    return out


def bp_bb9(team_id, rates: dict | None = None) -> float:
    """A team's bullpen BB/9, falling back to the league relief rate."""
    r = rates or team_bullpen_rates()
    lg = r.get("_league", {}).get("bb9", LG_BP_BB9)
    try:
        return float(r.get(int(team_id), {}).get("bb9", lg))
    except (TypeError, ValueError):
        return lg


def reload() -> None:
    global _CACHE
    _CACHE = None
