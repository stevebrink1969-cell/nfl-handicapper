"""Build the upcoming week's slate: projected lines, market lines, edges."""

import polars as pl

from . import data


def upcoming_week(season: int) -> tuple[pl.DataFrame, int]:
    sched = data.schedules([season])
    unplayed = sched.filter(pl.col("home_score").is_null())
    week = int(unplayed["week"].min()) if unplayed.height else int(sched["week"].max())
    return sched, week


def build(tr: pl.DataFrame, inj_adj: pl.DataFrame, hfa: float, season: int,
          week: int | None = None) -> tuple[pl.DataFrame, int]:
    sched, next_week = upcoming_week(season)
    week = week or next_week
    games = sched.filter(pl.col("week") == week).select(
        "game_id", "gameday", "weekday", "gametime",
        "away_team", "home_team", "spread_line", "location",
    )
    rat = tr.join(inj_adj, on="team", how="left").with_columns(
        pl.col("inj_adj").fill_null(0.0),
        (pl.col("rating") + pl.col("inj_adj").fill_null(0.0)).alias("adj_rating"),
    )
    h = rat.rename(
        {"team": "home_team", "rating": "rat_h", "inj_adj": "adj_h", "adj_rating": "eff_h"}
    ).select("home_team", "rat_h", "adj_h", "eff_h")
    a = rat.rename(
        {"team": "away_team", "rating": "rat_a", "inj_adj": "adj_a", "adj_rating": "eff_a"}
    ).select("away_team", "rat_a", "adj_a", "eff_a")
    out = (
        games.join(h, on="home_team", how="left")
        .join(a, on="away_team", how="left")
        .with_columns(
            (
                pl.col("eff_h") - pl.col("eff_a")
                + pl.when(pl.col("location") == "Home").then(hfa).otherwise(0.0)
            )
            .round(1)
            .alias("model_line")
        )
        .with_columns(
            (pl.col("model_line") - pl.col("spread_line")).round(1).alias("edge")
        )
        .sort(pl.col("edge").abs(), descending=True, nulls_last=True)
    )
    return out, week
