"""Team ratings: sum of expected-lineup player values, scaled to spread points."""

import polars as pl

from . import config as C


def team_ratings(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    """Returns (team table, players with scaled points + lineup weight, scale)."""
    ranked = players.sort("value_raw", descending=True).with_columns(
        pl.int_range(pl.len()).over("team", "grp").alias("rk")
    )
    rot = pl.DataFrame(
        [
            {"grp": g, "rk": i, "rot_w": w}
            for g, ws in C.ROTATION.items()
            for i, w in enumerate(ws)
        ]
    )
    ranked = ranked.join(rot, on=["grp", "rk"], how="left").with_columns(
        pl.col("rot_w").fill_null(0.0)
    )
    tr = ranked.group_by("team").agg(
        (pl.col("value_raw") * pl.col("rot_w")).sum().alias("raw")
    )
    scale = C.TEAM_RATING_STD / tr["raw"].std()
    mean = tr["raw"].mean()
    tr = tr.with_columns(((pl.col("raw") - mean) * scale).round(2).alias("rating")).sort(
        "rating", descending=True
    )
    ranked = ranked.with_columns(
        (pl.col("value_raw") * scale).round(2).alias("points"),
        (pl.col("qb_pts") * scale).round(2).alias("qb"),
        (pl.col("rush_pts") * scale).round(2).alias("rush"),
        (pl.col("recv_pts") * scale).round(2).alias("recv"),
        (pl.col("def_pts") * scale).round(2).alias("def"),
        (pl.col("base_pts") * scale).round(2).alias("base"),
    )
    return tr, ranked, scale
