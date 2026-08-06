"""Rank player props by expected value: model probability vs DK price."""

import math
import re
from datetime import datetime, timedelta, timezone

import polars as pl

from . import config as C
from . import propodds, props

MARKET_LABEL = {
    "player_pass_yds": "Pass yds", "player_pass_tds": "Pass TDs",
    "player_rush_yds": "Rush yds", "player_rush_attempts": "Carries",
    "player_receptions": "Receptions", "player_reception_yds": "Rec yds",
    "player_anytime_td": "Anytime TD",
}
MARKET_PROJ = {
    "player_pass_yds": "proj_pass_yds", "player_pass_tds": "proj_pass_tds",
    "player_rush_yds": "proj_rush_yds", "player_rush_attempts": "proj_carries",
    "player_receptions": "proj_receptions", "player_reception_yds": "proj_rec_yds",
    "player_anytime_td": "td_lambda",
}


def _norm_name(s: str) -> str:
    s = re.sub(r"[.'’]", "", (s or "").lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return s


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _pois_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


def _american_to_dec(a: int) -> float:
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def _p_over(market: str, proj: float, line: float) -> float | None:
    if market == "player_anytime_td":
        return 1.0 - math.exp(-max(proj, 0.001))
    if market == "player_pass_tds":
        return 1.0 - _pois_cdf(int(math.floor(line)), max(proj, 0.01))
    sd_ab = C.PROP_SD.get(market)
    if sd_ab is None:
        return None
    sd = max(sd_ab[0] + sd_ab[1] * proj, 1.0)
    return 1.0 - _phi((line - proj) / sd)


def _slot(commence_iso: str) -> tuple[str, int]:
    """Time-slot label + sort order from kickoff (ET ~ UTC-4 in season)."""
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    et = dt.astimezone(timezone(timedelta(hours=-4)))
    wd, hr = et.weekday(), et.hour  # Mon=0
    if wd == 3:
        return "Thursday", 0
    if wd == 4:
        return "Friday", 1
    if wd == 5:
        return "Saturday", 2
    if wd == 6:
        if hr < 16:
            return "Sunday early", 3
        if hr < 19:
            return "Sunday late", 4
        return "Sunday night", 5
    if wd == 0:
        return "Monday", 6
    return et.strftime("%A"), 7


def scan(target_season: int) -> pl.DataFrame:
    """Fetch current props, price every side with model probabilities, apply
    Steve's odds filter, return top plays per time slot."""
    empty = pl.DataFrame(schema={
        "slot": pl.String, "slot_order": pl.Int64, "player": pl.String,
        "team": pl.String, "game": pl.String, "market": pl.String,
        "market_key": pl.String, "nkey": pl.String,
        "side": pl.String, "line": pl.Float64, "price": pl.Int64,
        "proj": pl.Float64, "p_win": pl.Float64, "ev_pct": pl.Float64,
    })
    odds = propodds.fetch_props()
    if odds.height == 0:
        return empty
    proj = props.current_projections(target_season).filter(
        (pl.col("gp") + pl.col("p_gp")) >= C.PROP_MIN_HISTORY_GAMES
    ).with_columns(
        pl.col("player_display_name")
        .map_elements(_norm_name, return_dtype=pl.String).alias("nkey")
    )
    odds = odds.with_columns(
        pl.col("player").map_elements(_norm_name, return_dtype=pl.String).alias("nkey")
    )
    joined = odds.join(proj, on="nkey", how="inner")

    rows = []
    for r in joined.iter_rows(named=True):
        market = r["market"]
        proj_col = MARKET_PROJ.get(market)
        if proj_col is None or r[proj_col] is None:
            continue
        price = r["price"]
        if price == 0 or price < C.PROP_MIN_ODDS:
            continue
        p_over = _p_over(market, float(r[proj_col]), float(r["line"]))
        if p_over is None:
            continue
        if r["side"] == "Under":
            p = 1.0 - p_over
        else:  # Over / Yes
            p = p_over
        ev = p * _american_to_dec(price) - 1.0
        slot, order = _slot(r["commence_time"])
        rows.append({
            "slot": slot, "slot_order": order, "player": r["player_display_name"],
            "team": r["team"], "game": f'{r["away"]} @ {r["home"]}',
            "market": MARKET_LABEL.get(market, market), "market_key": market,
            "nkey": r["nkey"], "side": r["side"],
            "line": r["line"], "price": price, "proj": round(float(r[proj_col]), 1),
            "p_win": round(p, 3), "ev_pct": round(ev * 100, 1),
        })
    if not rows:
        return empty
    out = pl.DataFrame(rows)
    # best side per player-market (an Over and Under can't both be listed)
    return (
        out.sort("ev_pct", descending=True)
        .unique(subset=["player", "market", "game"], keep="first")
        .sort(["slot_order", "ev_pct"], descending=[False, True])
    )


def top_per_slot(plays: pl.DataFrame) -> pl.DataFrame:
    if plays.height == 0:
        return plays
    return plays.group_by("slot", maintain_order=True).head(C.PROP_TOP_N)
