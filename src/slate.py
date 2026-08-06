"""Build the upcoming week's slate and the full-season board."""

import polars as pl

from . import data


def build_board(tr: pl.DataFrame, hfa: float, season: int,
                odds: pl.DataFrame | None = None) -> pl.DataFrame:
    """Model line vs posted line for EVERY unplayed game this season.

    Future weeks use base ratings (no injury adjustments — future injury
    reports don't exist). Look-ahead lines get less market attention than
    game-week lines, which is where this board hunts.
    """
    sched = data.schedules([season])
    games = sched.filter(pl.col("home_score").is_null()).select(
        "game_id", pl.col("week").cast(pl.Int32), "gameday", "weekday",
        "gametime", "away_team", "home_team", "spread_line", "location",
        "total_line", "roof",
    )
    if odds is not None and odds.height > 0:
        games = games.join(
            odds.rename({"home": "home_team", "away": "away_team"})
            .unique(subset=["home_team", "away_team"], keep="last"),
            on=["home_team", "away_team"], how="left",
        )
    else:
        games = games.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("open_line"),
            pl.lit(None, dtype=pl.Float64).alias("book_line"),
        )
    games = games.with_columns(
        pl.coalesce("book_line", "spread_line").alias("market_line"),
        pl.when(pl.col("book_line").is_not_null())
        .then(pl.lit("book")).otherwise(pl.lit("consensus")).alias("mkt_src"),
    )
    h = tr.rename({"team": "home_team", "rating": "rat_h"}).select("home_team", "rat_h")
    a = tr.rename({"team": "away_team", "rating": "rat_a"}).select("away_team", "rat_a")
    return (
        games.join(h, on="home_team", how="left")
        .join(a, on="away_team", how="left")
        .with_columns(
            (
                pl.col("rat_h") - pl.col("rat_a")
                + pl.when(pl.col("location") == "Home").then(hfa).otherwise(0.0)
            ).round(1).alias("model_line")
        )
        .with_columns(
            (pl.col("model_line") - pl.col("market_line")).round(1).alias("edge")
        )
        .sort(pl.col("edge").abs(), descending=True, nulls_last=True)
    )


def upcoming_week(season: int) -> tuple[pl.DataFrame, int]:
    sched = data.schedules([season])
    unplayed = sched.filter(pl.col("home_score").is_null())
    week = int(unplayed["week"].min()) if unplayed.height else int(sched["week"].max())
    return sched, week


def build(tr: pl.DataFrame, inj_adj: pl.DataFrame, hfa: float, season: int,
          week: int | None = None,
          odds: pl.DataFrame | None = None) -> tuple[pl.DataFrame, int]:
    sched, next_week = upcoming_week(season)
    week = week or next_week
    games = sched.filter(pl.col("week") == week).select(
        "game_id", "gameday", "weekday", "gametime",
        "away_team", "home_team", "spread_line", "location",
        "total_line", "roof",
    )
    if odds is not None and odds.height > 0:
        games = games.join(
            odds.rename({"home": "home_team", "away": "away_team"}),
            on=["home_team", "away_team"], how="left",
        )
    else:
        games = games.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("open_line"),
            pl.lit(None, dtype=pl.Float64).alias("book_line"),
        )
    # Market line: sportsbook when we have it, else nflverse consensus
    games = games.with_columns(
        pl.coalesce("book_line", "spread_line").alias("market_line"),
        pl.when(pl.col("book_line").is_not_null())
        .then(pl.lit("book")).otherwise(pl.lit("consensus")).alias("mkt_src"),
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
            (pl.col("model_line") - pl.col("market_line")).round(1).alias("edge")
        )
        .sort(pl.col("edge").abs(), descending=True, nulls_last=True)
    )
    return out, week
