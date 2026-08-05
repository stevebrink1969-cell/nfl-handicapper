"""Out-of-sample test: for each season 2023-2025, calibrate ONLY on prior
seasons, then grade that season's projected lines vs close and vs results."""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_phase2 import season_components
from src import config as C
from src import data

TEST_SEASONS = [2023, 2024, 2025]


def main() -> None:
    lookback = list(range(C.LOOKBACK_START, max(C.BACKTEST_SEASONS) + 1))
    pbp_all = data.pbp(lookback).with_columns(pl.col("season").cast(pl.Int32))
    snaps_all = data.snap_counts(lookback)
    rosters_all = pl.concat([data.rosters(s) for s in lookback], how="vertical_relaxed")

    comps = pl.concat(
        [season_components(S, pbp_all, snaps_all, rosters_all) for S in C.BACKTEST_SEASONS]
    )
    sched = (
        data.schedules(C.BACKTEST_SEASONS)
        .filter(pl.col("spread_line").is_not_null() & pl.col("home_score").is_not_null())
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
            (pl.col("location") == "Home").cast(pl.Float64).alias("home"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        )
    )

    def design(df):
        return np.column_stack(
            [df["qb_d"].to_numpy(), df["perf_d"].to_numpy(), df["home"].to_numpy()]
        )

    tot = {t: [0, 0] for t in (1.0, 2.0, 3.0)}
    tot_su = [0, 0]
    print("Season | games | fit-on | MAE vs close | model MAE vs margin | market MAE")
    all_rows = []
    for S in TEST_SEASONS:
        train = g.filter(pl.col("season") < S)
        test = g.filter(pl.col("season") == S)
        beta, *_ = np.linalg.lstsq(design(train), train["spread_line"].to_numpy(), rcond=None)
        pred = design(test) @ beta
        spread = test["spread_line"].to_numpy()
        margin = test["margin"].to_numpy().astype(float)

        mae_c = np.abs(pred - spread).mean()
        mae_m = np.abs(pred - margin).mean()
        mae_mkt = np.abs(spread - margin).mean()
        print(f"{S}   | {test.height}   | {train['season'].min()}-{train['season'].max()}"
              f" | {mae_c:.2f}         | {mae_m:.2f}               | {mae_mkt:.2f}")

        su_pick = np.sign(pred)
        su_ok = (su_pick == np.sign(margin)) & (margin != 0) & (su_pick != 0)
        su_n = int(((margin != 0) & (su_pick != 0)).sum())
        tot_su[0] += int(su_ok.sum()); tot_su[1] += su_n

        edge = pred - spread
        cover = np.sign(margin - spread)
        print(f"       straight-up winners: {su_ok.sum()}/{su_n} "
              f"({su_ok.sum() / su_n * 100:.1f}%)")
        for t in (1.0, 2.0, 3.0):
            mask = (np.abs(edge) >= t) & (cover != 0)
            n = int(mask.sum())
            w = int((np.sign(edge[mask]) == cover[mask]).sum())
            tot[t][0] += w; tot[t][1] += n
            print(f"       ATS edge>={t:.0f}: {w}-{n - w} ({w / n * 100:.1f}%)" if n else "")
        all_rows.append(
            test.with_columns(pl.Series("model_spread", np.round(pred, 2)))
        )

    print("\n=== 3-season out-of-sample totals ===")
    print(f"Straight-up winners: {tot_su[0]}/{tot_su[1]} ({tot_su[0] / tot_su[1] * 100:.1f}%)")
    for t in (1.0, 2.0, 3.0):
        w, n = tot[t]
        roi = (w * (100 / 110) - (n - w)) / n * 100 if n else 0
        print(f"ATS edge>={t:.0f}: {w}-{n - w} ({w / n * 100:.1f}%) | "
              f"ROI at -110: {roi:+.1f}% | breakeven 52.4%")

    out = Path(__file__).resolve().parent.parent / "output" / "oos_test_games.csv"
    pl.concat(all_rows).select(
        "season", "week", "game_type", "home_team", "away_team",
        "spread_line", "model_spread", "margin",
    ).write_csv(out)
    print(f"\nPer-game detail: {out}")


if __name__ == "__main__":
    main()
