"""Per-stat calibration for player projections.

Two-layer correction:

1. **Rate factor** — a simple `actual_mean / proj_mean` multiplier computed
   over the calibration window. Captures global run-environment shifts (cool
   pitching weeks, league-wide BABIP regression) that isotonic regression
   cannot capture robustly at the tails. Always applied if present.

2. **Isotonic curve** — monotonic `actual = f(projected)` per stat. Captures
   per-decile compression/expansion (the original purpose: bottom-bin
   over-projection, top-bin under-projection). Applied with a soft blend
   weight; iso outputs are clamped to ±25% of the rate-corrected projection
   so degenerate tail fits can't corrupt the leaderboard.

May 11 2026 redesign: the iso-only approach worked when the bias was tail
compression. But running it at blend 0.85 caused catastrophic over-correction
when the recent window flipped to systematic over-projection (May 7-10
backtest showed batter_h iso predicting 2.33 from a raw 1.50 — a 56% upshift
on a stat the data actually wanted scaled DOWN by ~15%). The rate-factor
layer captures that systematic shift directly while iso handles only the
residual shape.

Calibrators live at data/models/projection_calibration.joblib as a dict.
Each entry may be:
  - an sklearn IsotonicRegression object (legacy), or
  - a dict with keys {"iso": IsotonicRegression, "rate_factor": float}.

If the calibration file is missing or a stat key isn't present, we return
the raw projection unchanged — safe-fallback behaviour.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

# Lazy-loaded cache
_CALIBRATORS: Optional[dict] = None
_PATH = Path(__file__).resolve().parent.parent / "data" / "models" / "projection_calibration.joblib"


def load() -> dict:
    """Load the per-stat calibrators from disk; return empty if absent."""
    global _CALIBRATORS
    if _CALIBRATORS is not None:
        return _CALIBRATORS
    try:
        import joblib
        if _PATH.exists():
            _CALIBRATORS = joblib.load(_PATH)
        else:
            _CALIBRATORS = {}
    except Exception:
        _CALIBRATORS = {}
    return _CALIBRATORS


def reload() -> None:
    """Reset the cache so the next call to load() reloads from disk."""
    global _CALIBRATORS
    _CALIBRATORS = None


# Iso blend weight. Lowered from 0.85 -> 0.30 after May 11 audit showed iso
# curves were producing extreme tail upshifts (batter_h 1.50 -> 2.33) and zero
# outputs at low ends (pitcher_er 1.50 -> 0.0). The rate-factor layer now
# absorbs the systematic level shifts that iso was forced to chase, so iso
# only contributes shape correction at lower weight.
_ISO_BLEND_WEIGHT = 0.30

# Cap iso output to within +/-25% of the rate-corrected projection. Prevents
# pathological extrapolations from corrupting the leaderboard when iso fits
# get unstable at the tails (sparse training data -> wild conditional means).
_ISO_DEVIATION_CAP = 0.25


def apply(stat: str, raw_proj: float) -> float:
    """Apply rate-factor + isotonic calibration for `stat` if available.

    Order of operations:
      1. raw_proj      — analytical/ML output
      2. * rate_factor — global mean correction (always trusted)
      3. blend with iso(rate_corrected) — soft shape correction, capped
    """
    if raw_proj is None:
        return raw_proj
    cals = load()
    entry = cals.get(stat)
    if entry is None:
        return float(raw_proj)

    # Back-compat: legacy entries are bare IsotonicRegression objects.
    if hasattr(entry, "predict"):
        iso = entry
        rate_factor = 1.0
    elif isinstance(entry, dict):
        iso = entry.get("iso")
        rate_factor = float(entry.get("rate_factor", 1.0))
    else:
        return float(raw_proj)

    # Step 1: apply rate factor (always)
    after_rate = float(raw_proj) * rate_factor

    # Step 2: optional iso blend on the rate-corrected value
    if iso is None:
        return max(0.0, after_rate)
    try:
        iso_pred = float(iso.predict([after_rate])[0])
    except Exception:
        return max(0.0, after_rate)

    # Cap iso deviation. If iso wants to push more than +/-25% from rate-corrected,
    # clamp -- protects against degenerate tail extrapolations like the
    # batter_h 1.50 -> 2.33 mapping we saw before the redesign.
    lo = after_rate * (1.0 - _ISO_DEVIATION_CAP)
    hi = after_rate * (1.0 + _ISO_DEVIATION_CAP)
    iso_capped = min(hi, max(lo, iso_pred))

    blended = _ISO_BLEND_WEIGHT * iso_capped + (1.0 - _ISO_BLEND_WEIGHT) * after_rate
    return max(0.0, blended)
