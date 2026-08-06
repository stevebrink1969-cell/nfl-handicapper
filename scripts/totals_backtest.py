"""Totals (over/under) model backtest, walk-forward OOS like the spread model.

Features per game (all lagged — only info available before kickoff):
  offsum   combined offensive player value of both teams (rotation-weighted)
  defsum   combined defensive playmaking value of both teams
  pace     combined plays/game of both teams (season-to-date blended w/ prior yr)
  env      league scoring environment (lagged league-average game total)
  wind/cold/dome  weather at kickoff (historical actuals; live uses forecast)

Fit vs closing total on train seasons; grade OOS 2023-2025: MAE vs close and
over/under record when the model disagrees with the closing total by >= t.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src import data

TEST_SEASONS = [2023, 2024, 2025]
PACE_PRIOR_GAMES = 6


def pace_table(pbp_all: pl.DataFrame) -> pl.DataFrame:
    """Offensive plays per (season, team, week) — for lagged pace features."""
    plays = pbp_all.filter(
        pl.col("play_type").is_in(["pass", "run"]) & pl.col("posteam").is_not_null()
    )
    return plays.group_by("season", "posteam", "week").agg(pl.len().alias("plays"))


def lagged_pace(pace: pl.DataFrame) -> pl.DataFrame:
    """For each (season, team, week): blended plays/game from weeks < W this
    season and all of last season."""
    season_avg = pace.group_by("season", "posteam").agg(
        pl.col("plays").mean().alias("prior_pace"),
    ).with_columns((pl.col("season") + 1).alias("season_next"))
    rows = []
    for (s, t), grp in pace.group_by(["season", "posteam"], maintain_order=False):
        g = grp.sort("week")
        wks = g["week"].to_list()
        pls = g["plays"].to_list()
        for i, w in enumerate(wks):
            cur = pls[:i]
            rows.append({"season": s, "posteam": t, "week": w,
                         "cur_pace": sum(cur) / len(cur) if cur else None,
                         "n_cur": len(cur)})
    lag = pl.DataFrame(rows)
    lag = lag.join(
        season_avg.select(pl.col("season_next").alias("season"), "posteam", "prior_pace"),
        on=["season", "posteam"], how="left",
    )
    lg_mean = pace["plays"].mean()
    return lag.with_columns(
        (
            (pl.col("cur_pace").fill_null(0.0) * pl.col("n_cur")
             + pl.col("prior_pace").fill_null(lg_mean) * PACE_PRIOR_GAMES)
            / (pl.col("n_cur") + PACE_PRIOR_GAMES)
        ).alias("pace")
    ).select("season", "posteam", "week", "pace")


def league_env(sched: pl.DataFrame) -> pl.DataFrame:
    """Lagged league-average game total per (season, week)."""
    g = sched.select("season", "week", (pl.col("home_score") + pl.col("away_score")).alias("tot"))
    season_avg = g.group_by("season").agg(pl.col("tot").mean().alias("prior_env")) \
        .with_columns((pl.col("season") + 1).alias("season_next"))
    rows = []
    for s, grp in g.group_by(["season"], maintain_order=False):
        gg = grp.sort("week")
        for w in gg["week"].unique().sort().to_list():
            past = gg.filter(pl.col("week") < w)["tot"]
            rows.append({"season": s[0] if isinstance(s, tuple) else s, "week": w,
                         "cur_env": past.mean() if past.len() else None,
                         "n": past.len()})
    lag = pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))
    lag = lag.join(
        season_avg.select(pl.col("season_next").cast(pl.Int32).alias("season"), "prior_env"),
        on="season", how="left",
    )
    overall = g["tot"].mean()
    return lag.with_columns(
        (
            (pl.col("cur_env").fill_null(0.0) * pl.col("n")
             + pl.col("prior_env").fill_null(overall) * 30)
            / (pl.col("n") + 30)
        ).alias("env")
    ).select("season", "week", "env")


def lagged_team_rates(pbp_all: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Per (season, team, week): lagged points for/against per game and
    offensive/defensive EPA per play, season-to-date blended with prior year."""
    home = sched.select("season", "week", pl.col("home_team").alias("team"),
                        pl.col("home_score").alias("pf"), pl.col("away_score").alias("pa"))
    away = sched.select("season", "week", pl.col("away_team").alias("team"),
                        pl.col("away_score").alias("pf"), pl.col("home_score").alias("pa"))
    scores = pl.concat([home, away])
    plays = pbp_all.filter(
        pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()
    )
    oepa = plays.group_by("season", pl.col("posteam").alias("team"), "week").agg(
        pl.col("epa").mean().alias("oepa"))
    depa = plays.group_by("season", pl.col("defteam").alias("team"), "week").agg(
        pl.col("epa").mean().alias("depa"))
    wk = (scores.join(oepa, on=["season", "team", "week"], how="left")
          .join(depa, on=["season", "team", "week"], how="left"))
    prior = wk.group_by("season", "team").agg(
        pl.col("pf").mean().alias("p_pf"), pl.col("pa").mean().alias("p_pa"),
        pl.col("oepa").mean().alias("p_oepa"), pl.col("depa").mean().alias("p_depa"),
    ).with_columns((pl.col("season") + 1).alias("season"))
    K = 6
    rows = []
    for (s, t), grp in wk.group_by(["season", "team"], maintain_order=False):
        g = grp.sort("week")
        for i, w in enumerate(g["week"].to_list()):
            past = g.head(i)
            rows.append({
                "season": s, "team": t, "week": w, "n": i,
                "c_pf": past["pf"].mean(), "c_pa": past["pa"].mean(),
                "c_oepa": past["oepa"].mean(), "c_depa": past["depa"].mean(),
            })
    lag = pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32))
    lag = lag.join(prior.with_columns(pl.col("season").cast(pl.Int32)),
                   on=["season", "team"], how="left")
    def blend(cur, pri, default):
        return ((pl.col(cur).fill_null(0.0) * pl.col("n")
                 + pl.col(pri).fill_null(default) * K) / (pl.col("n") + K))
    return lag.with_columns(
        blend("c_pf", "p_pf", 22.0).alias("t_pf"),
        blend("c_pa", "p_pa", 22.0).alias("t_pa"),
        blend("c_oepa", "p_oepa", 0.0).alias("t_oepa"),
        blend("c_depa", "p_depa", 0.0).alias("t_depa"),
    ).select("season", "team", "week", "t_pf", "t_pa", "t_oepa", "t_depa")


def build_games() -> pl.DataFrame:
    comps = pl.read_parquet(data.DATA_DIR / "weekly_comps.parquet")
    lookback = list(range(C.LOOKBACK_START, max(C.BACKTEST_SEASONS) + 1))
    pbp_all = data.pbp(lookback).with_columns(pl.col("season").cast(pl.Int32))
    sched = (
        data.schedules(C.BACKTEST_SEASONS)
        .filter(pl.col("total_line").is_not_null() & pl.col("home_score").is_not_null())
        .select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            "home_team", "away_team", "home_score", "away_score",
            "total_line", "roof", "temp", "wind",
        )
    )
    pace = lagged_pace(pace_table(pbp_all)).with_columns(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)
    )
    env = league_env(sched)
    h = comps.select("season", "week", pl.col("team").alias("home_team"),
                     pl.col("off_c").alias("off_h"), pl.col("defp_c").alias("def_h"))
    a = comps.select("season", "week", pl.col("team").alias("away_team"),
                     pl.col("off_c").alias("off_a"), pl.col("defp_c").alias("def_a"))
    ph = pace.rename({"posteam": "home_team", "pace": "pace_h"})
    pa = pace.rename({"posteam": "away_team", "pace": "pace_a"})
    rates = lagged_team_rates(pbp_all, sched)
    rh = rates.rename({"team": "home_team", "t_pf": "pf_h", "t_pa": "pa_h",
                       "t_oepa": "oepa_h", "t_depa": "depa_h"})
    ra = rates.rename({"team": "away_team", "t_pf": "pf_a", "t_pa": "pa_a",
                       "t_oepa": "oepa_a", "t_depa": "depa_a"})
    g = (
        sched.join(h, on=["season", "week", "home_team"], how="inner")
        .join(a, on=["season", "week", "away_team"], how="inner")
        .join(ph, on=["season", "week", "home_team"], how="left")
        .join(pa, on=["season", "week", "away_team"], how="left")
        .join(rh, on=["season", "week", "home_team"], how="left")
        .join(ra, on=["season", "week", "away_team"], how="left")
        .join(env, on=["season", "week"], how="left")
        .with_columns(
            pl.col("pace_h").fill_null(63.0), pl.col("pace_a").fill_null(63.0),
            pl.col("env").fill_null(44.0),
        )
        .with_columns(
            (pl.col("off_h") + pl.col("off_a")).alias("offsum"),
            (pl.col("def_h") + pl.col("def_a")).alias("defsum"),
            (pl.col("pace_h") + pl.col("pace_a") - 126.0).alias("pacex"),
            pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64).alias("dome"),
            ((pl.col("wind").fill_null(0).cast(pl.Float64) - 8).clip(lower_bound=0)).alias("windx"),
            (pl.col("temp").fill_null(60).cast(pl.Float64) <= 35).cast(pl.Float64).alias("cold"),
            (pl.col("home_score") + pl.col("away_score")).alias("actual"),
            pl.lit(1.0).alias("const"),
        )
        .with_columns(
            (pl.col("pf_h").fill_null(22.0) + pl.col("pf_a").fill_null(22.0)
             + pl.col("pa_h").fill_null(22.0) + pl.col("pa_a").fill_null(22.0)).alias("scoring"),
            (pl.col("oepa_h").fill_null(0.0) + pl.col("oepa_a").fill_null(0.0)).alias("oepasum"),
            (pl.col("depa_h").fill_null(0.0) + pl.col("depa_a").fill_null(0.0)).alias("depasum"),
        )
    )
    return g


def evaluate(g: pl.DataFrame, feats: list[str], label: str) -> None:
    tot = {t: [0, 0] for t in (1.0, 2.0, 3.0)}
    maes, mkt_maes, model_maes = [], [], []
    for S in TEST_SEASONS:
        train = g.filter(pl.col("season") < S)
        test = g.filter(pl.col("season") == S)
        Xtr = np.column_stack([train[f].to_numpy() for f in feats])
        Xte = np.column_stack([test[f].to_numpy() for f in feats])
        beta, *_ = np.linalg.lstsq(Xtr, train["total_line"].to_numpy(), rcond=None)
        pred = Xte @ beta
        close = test["total_line"].to_numpy()
        actual = test["actual"].to_numpy().astype(float)
        maes.append(np.abs(pred - close).mean())
        model_maes.append(np.abs(pred - actual).mean())
        mkt_maes.append(np.abs(close - actual).mean())
        edge = pred - close          # + => model says Over
        result = np.sign(actual - close)  # + over, - under, 0 push
        for t in (1.0, 2.0, 3.0):
            mask = (np.abs(edge) >= t) & (result != 0)
            tot[t][0] += int((np.sign(edge[mask]) == result[mask]).sum())
            tot[t][1] += int(mask.sum())
    line = (f"{label}: MAE vs close {np.mean(maes):.2f} | model vs actual "
            f"{np.mean(model_maes):.2f} | market vs actual {np.mean(mkt_maes):.2f}")
    for t in (1.0, 2.0, 3.0):
        w, n = tot[t]
        pct = f"{w / n * 100:.1f}%" if n else "n/a"
        line += f" | O/U>={t:.0f}: {w}-{n - w} ({pct})"
    print(line, flush=True)


def main() -> None:
    g = build_games()
    print(f"games: {g.height}\n")
    evaluate(g, ["const", "offsum", "defsum"], "ratings only        ")
    evaluate(g, ["const", "offsum", "defsum", "pacex"], "+ pace              ")
    evaluate(g, ["const", "offsum", "defsum", "pacex", "env"], "+ scoring env       ")
    evaluate(g, ["const", "offsum", "defsum", "pacex", "env", "windx", "cold", "dome"],
             "+ weather (full)    ")
    evaluate(g, ["const", "scoring", "pacex", "env"], "team scoring        ")
    evaluate(g, ["const", "scoring", "oepasum", "depasum", "pacex", "env"],
             "team scoring + EPA  ")
    evaluate(g, ["const", "scoring", "oepasum", "depasum", "pacex", "env",
                 "windx", "cold", "dome"], "team full + weather ")
    evaluate(g, ["const", "offsum", "defsum", "scoring", "oepasum", "depasum",
                 "pacex", "env", "windx", "cold", "dome"], "everything          ")


if __name__ == "__main__":
    main()
