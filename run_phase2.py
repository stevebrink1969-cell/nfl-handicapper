"""Phase 2: walk-forward backtest and calibration.

For each season 2021-2025, team ratings are built ONLY from prior seasons'
data (no leakage), then component point-scales and home-field advantage are
fitted against real closing spreads. Performance is reported against both the
closing line and actual game margins.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C
from src import data, teams, valuation

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def season_components(S: int, pbp_all, snaps_all, rosters_all) -> pl.DataFrame:
    wts = C.season_weights(S)
    seasons = list(wts)
    pbp = pbp_all.filter(pl.col("season").is_in(seasons))
    snaps = snaps_all.filter(pl.col("season").is_in(seasons))
    players = valuation.player_values(pbp, snaps, data.rosters(S), rosters_all, wts)
    return teams.team_components(players).with_columns(
        pl.lit(S).cast(pl.Int32).alias("season")
    )


def main() -> None:
    lookback = list(range(C.LOOKBACK_START, max(C.BACKTEST_SEASONS) + 1))
    pbp_all = data.pbp(lookback).with_columns(pl.col("season").cast(pl.Int32))
    snaps_all = data.snap_counts(lookback)
    rosters_all = pl.concat(
        [data.rosters(s) for s in lookback], how="vertical_relaxed"
    )

    comps = []
    for S in C.BACKTEST_SEASONS:
        print(f"Building {S} team ratings from {S - len(C.LAG_WEIGHTS)}-{S - 1} data...")
        comps.append(season_components(S, pbp_all, snaps_all, rosters_all))
    comps = pl.concat(comps)

    sched = (
        data.schedules(C.BACKTEST_SEASONS)
        .filter(
            pl.col("spread_line").is_not_null()
            & pl.col("home_score").is_not_null()
        )
        .select(
            pl.col("season").cast(pl.Int32), "game_id", "week", "game_type",
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "location",
        )
    )
    h = comps.rename({"team": "home_team", "qb_c": "qb_h", "perf_c": "perf_h", "base_c": "base_h"})
    a = comps.rename({"team": "away_team", "qb_c": "qb_a", "perf_c": "perf_a", "base_c": "base_a"})
    g = (
        sched.join(h, on=["season", "home_team"], how="inner")
        .join(a, on=["season", "away_team"], how="inner")
        .with_columns(
            (pl.col("qb_h") - pl.col("qb_a")).alias("qb_d"),
            (pl.col("perf_h") - pl.col("perf_a")).alias("perf_d"),
            (pl.col("base_h") - pl.col("base_a")).alias("base_d"),
            (pl.col("location") == "Home").cast(pl.Float64).alias("home"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        )
    )
    print(f"\nBacktest games: {g.height}")

    # Base component is excluded from the fit: team base sums are near-identical
    # (everyone fields ~22 starters), so its coefficient would just fit noise.
    X = np.column_stack(
        [g["qb_d"].to_numpy(), g["perf_d"].to_numpy(), g["home"].to_numpy()]
    )
    y_spread = g["spread_line"].to_numpy()
    y_margin = g["margin"].to_numpy().astype(float)

    beta_s, *_ = np.linalg.lstsq(X, y_spread, rcond=None)
    beta_m, *_ = np.linalg.lstsq(X, y_margin, rcond=None)
    pred = X @ beta_s

    mae_close = np.abs(pred - y_spread).mean()
    corr = np.corrcoef(pred, y_spread)[0, 1]
    mae_margin_model = np.abs(pred - y_margin).mean()
    mae_margin_close = np.abs(y_spread - y_margin).mean()

    print(f"\nFit to closing spread: k_qb={beta_s[0]:.3f} k_perf={beta_s[1]:.3f} "
          f"HFA={beta_s[2]:.2f}")
    print(f"Fit to actual margin:  k_qb={beta_m[0]:.3f} k_perf={beta_m[1]:.3f} "
          f"HFA={beta_m[2]:.2f}")
    print(f"\nModel line vs closing line:  MAE {mae_close:.2f} pts, corr {corr:.3f}")
    print(f"Model line vs actual margin: MAE {mae_margin_model:.2f} pts")
    print(f"Close line vs actual margin: MAE {mae_margin_close:.2f} pts (market benchmark)")

    print("\nATS record when model disagrees with close by >= threshold:")
    edge = pred - y_spread
    cover = np.sign(y_margin - y_spread)  # +1 home covers, -1 away, 0 push
    for t in [1.0, 2.0, 3.0, 4.0]:
        mask = (np.abs(edge) >= t) & (cover != 0)
        if mask.sum() == 0:
            continue
        wins = (np.sign(edge[mask]) == cover[mask]).sum()
        n = int(mask.sum())
        print(f"  edge >= {t:.0f}: {wins}-{n - wins} ({wins / n * 100:.1f}%) over {n} games")

    per_season = (
        g.with_columns(
            pl.Series("pred", pred), pl.Series("abs_err", np.abs(pred - y_spread))
        )
        .group_by("season")
        .agg(pl.col("abs_err").mean().round(2).alias("mae_vs_close"), pl.len().alias("games"))
        .sort("season")
    )
    print("\nPer-season MAE vs closing line:")
    print(per_season)

    cal = {
        "k_qb": float(beta_s[0]), "k_perf": float(beta_s[1]),
        "k_base": C.K_BASE_DISPLAY, "hfa": float(beta_s[2]),
        "fit_target": "closing_spread",
        "games": int(g.height),
        "mae_vs_close": float(mae_close), "corr_vs_close": float(corr),
        "mae_vs_margin": float(mae_margin_model),
        "market_mae_vs_margin": float(mae_margin_close),
        "margin_fit": {"k_qb": float(beta_m[0]), "k_perf": float(beta_m[1]),
                        "hfa": float(beta_m[2])},
    }
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=2))
    g.with_columns(pl.Series("model_spread", np.round(pred, 2))).select(
        "season", "week", "game_type", "home_team", "away_team",
        "spread_line", "model_spread", "margin",
    ).write_csv(OUT / "backtest_games.csv")
    print(f"\nSaved calibration.json and backtest_games.csv ({g.height} games)")


if __name__ == "__main__":
    main()
