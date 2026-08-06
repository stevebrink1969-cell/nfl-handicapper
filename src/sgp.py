"""Same-game parlay builder: 2-3 leg combos from +EV props, priced with a
Gaussian copula over empirically measured same-game correlations
(output/sgp_correlations.json, from scripts/sgp_correlations.py).

Since DK's SGP quotes aren't available via API, each suggestion shows our
joint win probability, fair odds, and the minimum quote worth taking (+5% EV
threshold) — check DK's quoted price against that number.
"""

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

CORR_FILE = Path(__file__).resolve().parent.parent / "output" / "sgp_correlations.json"
MARKET_STAT = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_yds",  # tds ~ pass proxy
    "player_rush_yds": "rush_yds", "player_rush_attempts": "carries",
    "player_receptions": "receptions", "player_reception_yds": "rec_yds",
    "player_anytime_td": "td",
}
# Priors for TD legs (binary; not in the residual study)
TD_PRIORS = {"same_player": 0.30, "qb_receiver": 0.20, "qb_rusher": 0.10,
             "teammates": 0.02, "opp": 0.05}
MAX_LEGS_CONSIDERED = 6   # top legs per game fed to the combiner
MAX_PAIR_R = 0.60         # skip near-duplicate legs (e.g. rec + rec yds, same player)
EV_THRESHOLD = 0.05       # min-quote line is set at +5% EV
TOP_PER_GAME = 3


def _load_corr() -> dict:
    if CORR_FILE.exists():
        return json.loads(CORR_FILE.read_text())
    return {}


def _pair_r(l1: dict, l2: dict, corr: dict) -> float:
    st1, st2 = MARKET_STAT.get(l1["market_key"]), MARKET_STAT.get(l2["market_key"])
    if st1 is None or st2 is None:
        return 0.0
    same_player = l1["nkey"] == l2["nkey"]
    same_team = l1["team"] == l2["team"]
    if "td" in (st1, st2):
        if same_player:
            base = TD_PRIORS["same_player"]
        elif same_team and "pass_yds" in (st1, st2):
            base = TD_PRIORS["qb_receiver"]
        elif same_team:
            base = TD_PRIORS["teammates"]
        else:
            base = TD_PRIORS["opp"]
    else:
        s1, s2 = sorted([st1, st2])
        pair = f"{s1}|{s2}"
        passing = {"pass_yds"}
        recv = {"receptions", "rec_yds"}
        rush = {"rush_yds", "carries"}
        if same_player:
            cat = "same_player"
        elif same_team:
            if (s1 in passing and s2 in recv) or (s2 in passing and s1 in recv):
                cat = "qb_receiver"
            elif s1 in recv and s2 in recv:
                cat = "teammates_recv"
            elif (s1 in passing and s2 in rush) or (s1 in rush and s2 in passing):
                cat = "qb_rusher"
            else:
                cat = "teammates_other"
        else:
            cat = "opp_pass_pass" if (s1 in passing and s2 in passing) else "opp_other"
        base = corr.get(f"{cat}|{pair}", {}).get("r", 0.0)
    # Unders flip the sign of the correlation for that leg
    sign = 1.0
    if l1["side"] == "Under":
        sign *= -1.0
    if l2["side"] == "Under":
        sign *= -1.0
    return base * sign


def _joint_prob(legs: list[dict], corr: dict, n_samples: int = 40000) -> tuple[float, float]:
    """Monte Carlo Gaussian copula. Returns (p_joint, avg pairwise |r|)."""
    k = len(legs)
    R = np.eye(k)
    rs = []
    for i, j in combinations(range(k), 2):
        r = _pair_r(legs[i], legs[j], corr)
        R[i, j] = R[j, i] = r
        rs.append(r)
    # ensure positive semi-definite
    w, v = np.linalg.eigh(R)
    if w.min() < 1e-6:
        w = np.clip(w, 1e-6, None)
        R = v @ np.diag(w) @ v.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    L = np.linalg.cholesky(R)
    rng = np.random.default_rng(20260906)
    X = rng.standard_normal((n_samples, k)) @ L.T
    thresholds = [_ppf(l["p_win"]) for l in legs]
    hits = np.ones(n_samples, dtype=bool)
    for i in range(k):
        hits &= X[:, i] <= thresholds[i]
    return float(hits.mean()), float(np.mean(rs)) if rs else 0.0


def _ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _dec(a: int) -> float:
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def _american(dec: float) -> int:
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def build(plays: pl.DataFrame) -> list[dict]:
    """plays: full +EV legs table from propscan (needs market_key, nkey)."""
    if plays.height == 0:
        return []
    corr = _load_corr()
    out = []
    for game, grp in plays.group_by(["game"], maintain_order=True):
        legs_all = (
            grp.filter(pl.col("ev_pct") > 0)
            .sort("ev_pct", descending=True)
            .head(MAX_LEGS_CONSIDERED)
            .to_dicts()
        )
        cands = []
        for k in (2, 3):
            for combo in combinations(legs_all, k):
                if any(
                    abs(_pair_r(a, b, corr)) > MAX_PAIR_R
                    for a, b in combinations(combo, 2)
                ):
                    continue
                p_joint, avg_r = _joint_prob(list(combo), corr)
                if p_joint <= 0.02:
                    continue
                dec_naive = math.prod(_dec(l["price"]) for l in combo)
                ev_naive = p_joint * dec_naive - 1
                min_dec = (1 + EV_THRESHOLD) / p_joint
                cands.append({
                    "game": game[0] if isinstance(game, tuple) else game,
                    "slot": combo[0]["slot"],
                    "legs": [
                        {"player": l["player"], "market": l["market"],
                         "side": l["side"], "line": l["line"], "price": l["price"]}
                        for l in combo
                    ],
                    "p_joint": round(p_joint, 3),
                    "fair_odds": _american(1 / p_joint),
                    "min_quote": _american(min_dec),
                    "ev_at_naive": round(ev_naive * 100, 1),
                    "avg_r": round(avg_r, 2),
                })
        cands.sort(key=lambda c: -c["ev_at_naive"])
        out.extend(cands[:TOP_PER_GAME])
    out.sort(key=lambda c: -c["ev_at_naive"])
    return out
