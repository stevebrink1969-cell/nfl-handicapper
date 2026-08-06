"""OOS test of research-driven upgrades vs current production model (B):

  D: turnover-luck discounting only
  E: opponent adjustment only
  F: both

Same protocol as oos_test2: weekly walk-forward components, each test season
2023-2025 graded with coefficients fit only on prior seasons.
Baseline B (current prod): MAE 3.03 | ATS>=1 53.6% | >=2 54.9% | >=3 57.5%.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src import data, teams, valuation

TEST_SEASONS = [2023, 2024, 2025]
CUR_SEASON_WEIGHT = C.CUR_SEASON_WEIGHT

VARIANTS = [
    ("D_turnover", {"OPP_ADJUST": False, "TO_INT_WEIGHT": 0.55, "TO_FUM_WEIGHT": 0.45}),
    ("E_oppadj",   {"OPP_ADJUST": True,  "TO_INT_WEIGHT": 1.0,  "TO_FUM_WEIGHT": 1.0}),
    ("F_both",     {"OPP_ADJUST": True,  "TO_INT_WEIGHT": 0.55, "TO_FUM_WEIGHT": 0.45}),
]


def build_weekly_comps(cache: Path, pbp_all, snaps_all, rosters_all, sched) -> pl.DataFrame:
    if cache.exists():
        return pl.read_parquet(cache)
    rows = []
    for S in C.BACKTEST_SEASONS:
        weeks = sorted(sched.filter(pl.col("season") == S)["week"].unique().to_list())
        roster_S = data.rosters(S)
        wts = C.season_weights(S)
        wts[S] = CUR_SEASON_WEIGHT
        seasons = list(wts)
        for W in weeks:
            pbp = pbp_all.filter(
                pl.col("season").is_in(seasons)
                & ((pl.col("season") != S) | (pl.col("week") < W))
            )
            snaps = snaps_all.filter(
                pl.col("season").is_in(seasons)
                & ((pl.col("season") != S) | (pl.col("week") < W))
            )
            players = valuation.player_values(pbp, snaps, roster_S, rosters_all, wts)
            rows.append(
                teams.team_components(players).with_columns(
                    pl.lit(S).cast(pl.Int32).alias("season"),
                    pl.lit(W).cast(pl.Int32).alias("week"),
                )
            )
    comps = pl.concat(rows)
    comps.write_parquet(cache)
    return comps


def evaluate(g: pl.DataFrame, label: str) -> None:
    feats = ["qb_d", "perf_d", "home"]
    tot = {t: [0, 0] for t in (1.0, 2.0, 3.0)}
    maes = []
    for S in TEST_SEASONS:
        train = g.filter(pl.col("season") < S)
        test = g.filter(pl.col("season") == S)
        Xtr = np.column_stack([train[f].to_numpy() for f in feats])
        Xte = np.column_stack([test[f].to_numpy() for f in feats])
        beta, *_ = np.linalg.lstsq(Xtr, train["spread_line"].to_numpy(), rcond=None)
        pred = Xte @ beta
        spread = test["spread_line"].to_numpy()
        margin = test["margin"].to_numpy().astype(float)
        maes.append(np.abs(pred - spread).mean())
        edge = pred - spread
        cover = np.sign(margin - spread)
        for t in (1.0, 2.0, 3.0):
            mask = (np.abs(edge) >= t) & (cover != 0)
            tot[t][0] += int((np.sign(edge[mask]) == cover[mask]).sum())
            tot[t][1] += int(mask.sum())
    line = f"{label}: MAE {np.mean(maes):.2f}"
    for t in (1.0, 2.0, 3.0):
        w, n = tot[t]
        line += f" | ATS>={t:.0f}: {w}-{n - w} ({w / n * 100:.1f}%)"
    print(line, flush=True)


def main() -> None:
    lookback = list(range(C.LOOKBACK_START, max(C.BACKTEST_SEASONS) + 1))
    pbp_all = data.pbp(lookback).with_columns(pl.col("season").cast(pl.Int32))
    snaps_all = data.snap_counts(lookback)
    rosters_all = pl.concat([data.rosters(s) for s in lookback], how="vertical_relaxed")
    sched = (
        data.schedules(C.BACKTEST_SEASONS)
        .filter(pl.col("spread_line").is_not_null() & pl.col("home_score").is_not_null())
        .select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "location",
        )
    )

    print("Baseline B (prod): MAE 3.03 | ATS>=1: 53.6% | >=2: 54.9% | >=3: 57.5%")
    for name, cfg in VARIANTS:
        for k, v in cfg.items():
            setattr(C, k, v)
        cache = data.DATA_DIR / f"weekly_comps_{name}.parquet"
        print(f"building {name}...", flush=True)
        comps = build_weekly_comps(cache, pbp_all, snaps_all, rosters_all, sched)
        h = comps.rename({"team": "home_team", "qb_c": "qb_h", "perf_c": "perf_h", "base_c": "base_h"})
        a = comps.rename({"team": "away_team", "qb_c": "qb_a", "perf_c": "perf_a", "base_c": "base_a"})
        g = (
            sched.join(h, on=["season", "week", "home_team"], how="inner")
            .join(a, on=["season", "week", "away_team"], how="inner")
            .with_columns(
                (pl.col("qb_h") - pl.col("qb_a")).alias("qb_d"),
                (pl.col("perf_h") - pl.col("perf_a")).alias("perf_d"),
                (pl.col("location") == "Home").cast(pl.Float64).alias("home"),
                (pl.col("home_score") - pl.col("away_score")).alias("margin"),
            )
        )
        evaluate(g, name)


if __name__ == "__main__":
    main()
