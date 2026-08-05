"""Out-of-sample test of model improvements, 2023-2025:

  A. baseline: season-start values (current model)
  B. + in-season updating: values recomputed each week from games so far
  C. B + rest differential + prior-season team anchor

Weekly team components are cached to data/weekly_comps.parquet (delete to
rebuild). Every variant is graded train-on-past / test-on-future only.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src import data, teams, valuation

CUR_SEASON_WEIGHT = 1.25
TEST_SEASONS = [2023, 2024, 2025]
CACHE = data.DATA_DIR / "weekly_comps.parquet"


def build_weekly_comps(pbp_all, snaps_all, rosters_all, sched) -> pl.DataFrame:
    if CACHE.exists():
        return pl.read_parquet(CACHE)
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
            print(f"  built {S} week {W}", flush=True)
    comps = pl.concat(rows)
    comps.write_parquet(CACHE)
    return comps


def team_anchor(pbp_all) -> pl.DataFrame:
    """Prior-season leverage-weighted net EPA per game, per team."""
    seasons = sorted(pbp_all["season"].unique().to_list())
    p = valuation.add_weights(pbp_all, {s: 1.0 for s in seasons})
    p = p.filter(pl.col("epa").is_not_null())
    off = p.group_by("posteam", "season").agg(
        ((pl.col("lw") * pl.col("epa")).sum() / pl.col("game_id").n_unique()).alias("off")
    )
    de = p.group_by("defteam", "season").agg(
        ((pl.col("lw") * pl.col("epa")).sum() / pl.col("game_id").n_unique()).alias("de")
    )
    net = off.join(
        de, left_on=["posteam", "season"], right_on=["defteam", "season"]
    ).with_columns(
        (pl.col("off") - pl.col("de")).alias("net"),
        (pl.col("season") + 1).alias("target_season"),  # anchor applies NEXT season
    )
    return net.select(
        pl.col("posteam").alias("team"),
        pl.col("target_season").cast(pl.Int32).alias("season"),
        pl.col("net").alias("anchor"),
    )


def games_frame(sched, comps, anchor) -> pl.DataFrame:
    h = comps.rename({"team": "home_team", "qb_c": "qb_h", "perf_c": "perf_h", "base_c": "base_h"})
    a = comps.rename({"team": "away_team", "qb_c": "qb_a", "perf_c": "perf_a", "base_c": "base_a"})
    an_h = anchor.rename({"team": "home_team", "anchor": "an_h"})
    an_a = anchor.rename({"team": "away_team", "anchor": "an_a"})
    return (
        sched.join(h, on=["season", "week", "home_team"], how="inner")
        .join(a, on=["season", "week", "away_team"], how="inner")
        .join(an_h, on=["season", "home_team"], how="left")
        .join(an_a, on=["season", "away_team"], how="left")
        .with_columns(
            (pl.col("qb_h") - pl.col("qb_a")).alias("qb_d"),
            (pl.col("perf_h") - pl.col("perf_a")).alias("perf_d"),
            (pl.col("location") == "Home").cast(pl.Float64).alias("home"),
            (pl.col("an_h").fill_null(0.0) - pl.col("an_a").fill_null(0.0)).alias("anchor_d"),
            (
                pl.col("home_rest").fill_null(7).cast(pl.Float64).clip(4, 14)
                - pl.col("away_rest").fill_null(7).cast(pl.Float64).clip(4, 14)
            ).alias("rest_d"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        )
    )


def evaluate(g: pl.DataFrame, feats: list[str], label: str) -> None:
    print(f"\n--- {label} | features: {feats} ---")
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
        mae = np.abs(pred - spread).mean()
        maes.append(mae)
        line = f"{S}: MAE vs close {mae:.2f} | vs margin {np.abs(pred - margin).mean():.2f}"
        edge = pred - spread
        cover = np.sign(margin - spread)
        for t in (1.0, 2.0, 3.0):
            mask = (np.abs(edge) >= t) & (cover != 0)
            n = int(mask.sum())
            w = int((np.sign(edge[mask]) == cover[mask]).sum())
            tot[t][0] += w
            tot[t][1] += n
            line += f" | ATS>={t:.0f}: {w}-{n - w}"
        print(line)
    print(f"TOTAL: MAE {np.mean(maes):.2f}", end="")
    for t in (1.0, 2.0, 3.0):
        w, n = tot[t]
        print(f" | ATS>={t:.0f}: {w}-{n - w} ({w / n * 100:.1f}%)", end="")
    print()


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
            "game_type", "home_team", "away_team", "home_score", "away_score",
            "spread_line", "location", "home_rest", "away_rest",
        )
    )

    print("Building weekly walk-forward components (cached after first run)...")
    comps = build_weekly_comps(pbp_all, snaps_all, rosters_all, sched)
    anchor = team_anchor(pbp_all)
    g = games_frame(sched, comps, anchor)
    print(f"\nGames with features: {g.height}")

    evaluate(g, ["qb_d", "perf_d", "home"], "B: in-season weekly updating")
    evaluate(g, ["qb_d", "perf_d", "home", "rest_d"], "B + rest")
    evaluate(g, ["qb_d", "perf_d", "home", "anchor_d"], "B + anchor")
    evaluate(g, ["qb_d", "perf_d", "home", "rest_d", "anchor_d"], "C: full")
    print("\nBaseline A (season-start values, prior run): MAE 3.33 | "
          "ATS>=1: 52.8% | >=2: 52.7% | >=3: 52.8%")


if __name__ == "__main__":
    main()
