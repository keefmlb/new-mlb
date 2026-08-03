"""Empirically fit dispersion (variance / mean) per stat from boxscores.

For each prop type we ask: given a projected mean μ, what's the variance of
the actual outcome? Counting stats are usually moderately over-dispersed:
  - var/mean ≈ 1.0 means Poisson (rare for MLB box stats)
  - var/mean ≈ 1.3-1.6 means moderate over-dispersion (typical)
  - var/mean > 2.0 means heavy tails (HR, big games)

We bin by μ to get a μ-conditional dispersion curve. For prop probability we
use Negative Binomial(μ, dispersion(μ)), which gives tighter probabilities
than the hardcoded fallbacks.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class DispersionFit:
    """Mapping from projected mean -> over-dispersion ratio."""
    bin_edges: list[float]      # right-edges of mu bins
    dispersions: list[float]    # var/mean inside each bin
    overall: float              # global var/mean (fallback)

    def at(self, mu: float) -> float:
        """Return the over-dispersion ratio for a given projected mean."""
        for i, edge in enumerate(self.bin_edges):
            if mu <= edge:
                return float(self.dispersions[i])
        return float(self.dispersions[-1])

    def to_dict(self) -> dict:
        return {"bin_edges": self.bin_edges, "dispersions": self.dispersions,
                "overall": self.overall}

    @classmethod
    def from_dict(cls, d: dict) -> "DispersionFit":
        return cls(bin_edges=list(d["bin_edges"]),
                   dispersions=list(d["dispersions"]),
                   overall=float(d["overall"]))


def fit_dispersion(proj: np.ndarray, actual: np.ndarray, n_bins: int = 4,
                   floor: float = 1.0) -> DispersionFit:
    """Fit a piecewise-constant dispersion curve.

    Bins by projection quantile; in each bin computes var(actual)/mean(proj).
    Floors at `floor` so we never go BELOW Poisson (which would imply
    under-dispersion — implausible for non-truncated count stats).
    """
    df = pd.DataFrame({"p": proj, "a": actual.astype(float)}).dropna()
    df = df[df["p"] >= 0]
    if len(df) < 50 or df["p"].nunique() < 5:
        # Not enough data — fall back to a single global ratio.
        var = df["a"].var() if len(df) else 1.0
        mean = df["a"].mean() if len(df) else 1.0
        ratio = max(floor, var / max(mean, 0.01))
        return DispersionFit([float("inf")], [ratio], ratio)

    # Quantile bins on the projection
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop", labels=False)
    bins = df.groupby("bin").agg(
        upper=("p", "max"),
        mean_p=("p", "mean"),
        var_a=("a", "var"),
        mean_a=("a", "mean"),
        n=("a", "size"),
    ).reset_index()
    # var/mean ratio with smoothing
    bins["disp"] = (bins["var_a"] / bins["mean_p"].clip(lower=0.05)).clip(lower=floor)
    edges = bins["upper"].tolist()
    edges[-1] = float("inf")           # last bin extends to infinity
    overall = max(floor, df["a"].var() / max(df["a"].mean(), 0.05))
    return DispersionFit(edges, bins["disp"].tolist(), float(overall))


def fit_all(bat_csv: Path, pit_csv: Path) -> dict:
    """Fit dispersion for every prop stat. Saves a JSON-serializable dict.

    Pitcher stats use floor=1.5 because the analytical projection itself is
    biased (early-season K/9 estimates are noisy). With a Poisson floor of 1.0
    the resulting fit produced phantom UNDER edges — see the May 2026 bet log:
    pitcher_k UNDERs went 9-12 with 21/34 picks, mostly because the model
    projection sat below the line and the tight dispersion put 60%+ probability
    on UNDER. Floor of 1.5 brings these closer to the empirical pitcher_er
    ratio (~1.65) and deflates the spurious edges.
    """
    bat = pd.read_csv(bat_csv)
    pit = pd.read_csv(pit_csv)

    # Hits+Runs+RBIs is a book market (the largest one Fanatics offers). It is
    # a SUM of three positively-correlated counts, so its dispersion is much
    # wider than any single component — fit it empirically rather than assuming
    # the parts are independent.
    _hrr_actual = [c for c in ("h", "runs_b", "rbi") if c in bat.columns]
    _hrr_proj = [c for c in ("proj_h", "proj_runs", "proj_rbi") if c in bat.columns]
    if len(_hrr_actual) == 3 and len(_hrr_proj) == 3:
        bat["hrr"] = bat[_hrr_actual].sum(axis=1)
        bat["proj_hrr"] = bat[_hrr_proj].sum(axis=1)

    out: dict[str, dict] = {"batter": {}, "pitcher": {}}

    for stat, actual_col, proj_col in [
        ("hr",   "hr",     "proj_hr"),
        ("hits", "h",      "proj_h"),
        ("tb",   "tb",     "proj_tb"),
        ("rbi",  "rbi",    "proj_rbi"),
        ("runs", "runs_b", "proj_runs"),
        ("k",    "k_b",    "proj_k"),
        ("bb",   "bb_b",   "proj_bb"),
        ("hrr",  "hrr",    "proj_hrr"),
    ]:
        if proj_col not in bat.columns:
            continue
        fit = fit_dispersion(bat[proj_col].values, bat[actual_col].values, floor=1.0)
        out["batter"][stat] = fit.to_dict()

    for stat, actual_col, proj_col in [
        ("pitcher_k",    "k_p",  "proj_k"),
        ("pitcher_outs", "outs", "expected_outs"),
        ("pitcher_er",   "er",   "proj_er"),
        ("pitcher_h",    "h_p",  "proj_h"),
        ("pitcher_bb",   "bb_p", "proj_bb"),
        ("pitcher_hr",   "hr_p", "proj_hr"),
    ]:
        if proj_col not in pit.columns:
            continue
        fit = fit_dispersion(pit[proj_col].values, pit[actual_col].values, floor=1.5)
        out["pitcher"][stat] = fit.to_dict()

    return out


def load_fits(path: Path) -> dict[str, DispersionFit]:
    """Load fitted dispersions, returning a flat {market_key: DispersionFit} dict."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat: dict[str, DispersionFit] = {}
    for batter_key, d in raw.get("batter", {}).items():
        flat[batter_key] = DispersionFit.from_dict(d)
    for pitcher_key, d in raw.get("pitcher", {}).items():
        flat[pitcher_key] = DispersionFit.from_dict(d)
    return flat


def disp_for(market: str, mean_proj: float, fits: dict[str, DispersionFit],
             default: float = 1.3) -> float:
    """Lookup dispersion at the given projection mean for a market key."""
    f = fits.get(market)
    if f is None:
        return default
    return f.at(mean_proj)
