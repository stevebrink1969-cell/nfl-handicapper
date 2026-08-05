"""Player point-value engine.

Every player's value is built from leverage-weighted, recency-weighted EPA:
plays in garbage time (win prob outside the band, or 4th-qtr blowouts) are
discounted, recent seasons count more, and small samples are shrunk toward
baseline so one fluky game can't inflate a player.

Components (all in raw points, scaled to spread-points at the team stage):
  qb_pts    dropback EPA vs replacement-level QB
  rush_pts  rushing EPA (includes designed QB runs)
  recv_pts  receiving EPA on targets
  def_pts   credited defensive plays (sacks, INTs, PDs, TFLs, FFs), capped per play
  base_pts  snap-share x positional base value (OL and non-stat defenders live here)
"""

import polars as pl

from . import config as C


def _season_weights(wts: dict[int, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"season": list(wts), "sw": list(wts.values())},
        schema={"season": pl.Int32, "sw": pl.Float64},
    )


def add_weights(pbp: pl.DataFrame, wts: dict[int, float]) -> pl.DataFrame:
    """Attach leverage weight (lw) and season recency weight (sw) to each play."""
    pbp = pbp.with_columns(pl.col("wp").fill_null(0.5), pl.col("season").cast(pl.Int32))
    in_band = (pl.col("wp") >= C.WP_LO) & (pl.col("wp") <= C.WP_HI)
    blowout = (pl.col("qtr") >= 4) & (
        pl.col("score_differential").abs() >= C.BLOWOUT_MARGIN
    )
    lw = (
        pl.when(blowout | ~in_band)
        .then(pl.lit(C.GARBAGE_WEIGHT))
        .otherwise(pl.lit(1.0))
        .alias("lw")
    )
    return (
        pbp.with_columns(lw)
        .join(_season_weights(wts), on="season", how="left")
        .with_columns(pl.col("sw").fill_null(0.0))
    )


def qb_values(pbp: pl.DataFrame) -> pl.DataFrame:
    db = (
        pbp.filter(pl.col("qb_dropback") == 1)
        .with_columns(
            pl.coalesce("passer_player_id", "rusher_player_id").alias("pid"),
        )
        .filter(pl.col("pid").is_not_null() & pl.col("qb_epa").is_not_null())
        .with_columns((pl.col("sw") * pl.col("lw")).alias("w"))
    )
    g = db.group_by("pid").agg(
        pl.col("w").sum().alias("wdb"),
        (pl.col("w") * pl.col("qb_epa")).sum().alias("wepa"),
    )
    # Replacement level from qualified QBs' raw rates; low-sample QBs shrink
    # TOWARD replacement (not league mean), so a hot 100-dropback cameo can't
    # outrank established starters.
    qualified = g.filter(pl.col("wdb") >= C.QB_MIN_QUALIFYING_W).with_columns(
        (pl.col("wepa") / pl.col("wdb")).alias("raw_rate")
    )
    repl = qualified["raw_rate"].quantile(C.QB_REPLACEMENT_PCTILE)
    g = g.with_columns(
        (
            (pl.col("wepa") + C.QB_PRIOR_DROPBACKS * repl)
            / (pl.col("wdb") + C.QB_PRIOR_DROPBACKS)
        ).alias("epa_db")
    )
    return g.with_columns(
        ((pl.col("epa_db") - repl) * C.QB_DROPBACKS_PER_GAME * C.QB_CREDIT).alias(
            "qb_pts"
        )
    ).select("pid", "qb_pts")


def _rate_per_game(df: pl.DataFrame, pid_col: str, val_expr, out: str) -> pl.DataFrame:
    """Weighted value per game with shrinkage: sum(val) / (games + prior)."""
    per_season = (
        df.filter(pl.col(pid_col).is_not_null() & pl.col("epa").is_not_null())
        .group_by(pid_col, "season")
        .agg(
            val_expr.sum().alias("v_s"),
            pl.col("game_id").n_unique().alias("games_s"),
            pl.col("sw").first().alias("sw"),
        )
    )
    return (
        per_season.group_by(pid_col)
        .agg(
            (pl.col("sw") * pl.col("v_s")).sum().alias("v"),
            (pl.col("sw") * pl.col("games_s")).sum().alias("wgames"),
        )
        .with_columns(
            (pl.col("v") / (pl.col("wgames") + C.SKILL_PRIOR_GAMES)).alias(out)
        )
        .select(pl.col(pid_col).alias("pid"), out)
    )


def rush_values(pbp: pl.DataFrame) -> pl.DataFrame:
    runs = pbp.filter(
        (pl.col("play_type") == "run") & (pl.col("qb_scramble").fill_null(0) == 0)
    )
    return _rate_per_game(
        runs,
        "rusher_player_id",
        pl.col("lw") * pl.col("epa") * C.RUSH_CREDIT,
        "rush_pts",
    )


def recv_values(pbp: pl.DataFrame) -> pl.DataFrame:
    targets = pbp.filter(pl.col("play_type") == "pass")
    return _rate_per_game(
        targets,
        "receiver_player_id",
        pl.col("lw") * pl.col("epa") * C.RECV_CREDIT,
        "recv_pts",
    )


def def_values(pbp: pl.DataFrame) -> pl.DataFrame:
    """Credit defenders on plays they're individually tagged on. EPA credit is
    capped per play so a fluky pick-six doesn't mint a star."""
    dcred = (-pl.col("epa")).clip(-1.0, C.DEF_PLAY_EPA_CAP)
    types = [
        ("sack_player_id", C.DEF_SACK_CREDIT),
        ("half_sack_1_player_id", 0.5 * C.DEF_SACK_CREDIT),
        ("half_sack_2_player_id", 0.5 * C.DEF_SACK_CREDIT),
        ("interception_player_id", C.DEF_INT_CREDIT),
        ("pass_defense_1_player_id", C.DEF_PD_CREDIT),
        ("pass_defense_2_player_id", C.DEF_PD_CREDIT),
        ("tackle_for_loss_1_player_id", C.DEF_TFL_CREDIT),
        ("forced_fumble_player_1_player_id", C.DEF_FF_CREDIT),
    ]
    parts = []
    for col, factor in types:
        if col not in pbp.columns:
            continue
        parts.append(
            pbp.filter(pl.col(col).is_not_null() & pl.col("epa").is_not_null()).select(
                pl.col(col).alias("pid"),
                "season",
                "game_id",
                "sw",
                "epa",
                (pl.col("lw") * dcred * factor).alias("val"),
            )
        )
    stacked = pl.concat(parts)
    return _rate_per_game(stacked, "pid", pl.col("val"), "def_pts")


def snap_base(snaps: pl.DataFrame, rosters_all: pl.DataFrame,
              wts: dict[int, float]) -> pl.DataFrame:
    """Recency-weighted snap share per player (gsis id), for the base component."""
    xwalk = (
        rosters_all.filter(
            pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null()
        )
        .select("pfr_id", "gsis_id")
        .unique(subset=["pfr_id"], keep="last")
    )
    s = snaps.with_columns(
        pl.max_horizontal(
            pl.col("offense_pct").fill_null(0.0), pl.col("defense_pct").fill_null(0.0)
        ).alias("pct"),
        pl.col("season").cast(pl.Int32),
    )
    if s["pct"].max() is not None and s["pct"].max() > 1.5:
        s = s.with_columns(pl.col("pct") / 100.0)
    per_season = s.group_by("pfr_player_id", "season").agg(
        pl.col("pct").mean().alias("pct_s")
    )
    g = (
        per_season.join(_season_weights(wts), on="season", how="inner")
        .group_by("pfr_player_id")
        .agg(((pl.col("sw") * pl.col("pct_s")).sum() / pl.col("sw").sum()).alias("snap_pct"))
    )
    return g.join(xwalk, left_on="pfr_player_id", right_on="pfr_id", how="inner").select(
        pl.col("gsis_id").alias("pid"), "snap_pct"
    )


def player_values(pbp: pl.DataFrame, snaps: pl.DataFrame, roster: pl.DataFrame,
                  rosters_all: pl.DataFrame, wts: dict[int, float]) -> pl.DataFrame:
    """Join all components onto the target roster and total them (raw units)."""
    pbp = add_weights(pbp, wts)
    ro = (
        roster.filter(
            pl.col("gsis_id").is_not_null()
            & pl.col("status").is_in(["ACT", "INA", "DEV"])
        )
        .sort("week")
        .unique(subset=["gsis_id"], keep="last")
        .with_columns(
            pl.col("position")
            .replace_strict(C.POS_GROUP, default=None)
            .alias("grp")
        )
        .filter(pl.col("grp").is_not_null())  # drops K/P/LS for now
        .select("gsis_id", "full_name", "team", "position", "grp", "years_exp")
    )
    out = (
        ro.join(qb_values(pbp), left_on="gsis_id", right_on="pid", how="left")
        .join(rush_values(pbp), left_on="gsis_id", right_on="pid", how="left")
        .join(recv_values(pbp), left_on="gsis_id", right_on="pid", how="left")
        .join(def_values(pbp), left_on="gsis_id", right_on="pid", how="left")
        .join(snap_base(snaps, rosters_all, wts), left_on="gsis_id", right_on="pid", how="left")
    )
    base_map = pl.DataFrame(
        {"grp": list(C.POS_BASE), "pos_base": list(C.POS_BASE.values())}
    )
    return (
        out.join(base_map, on="grp", how="left")
        .with_columns(
            pl.col("qb_pts", "rush_pts", "recv_pts", "def_pts").fill_null(0.0),
            pl.col("snap_pct").fill_null(C.ROOKIE_SNAP_DEFAULT),
        )
        .with_columns((pl.col("pos_base") * pl.col("snap_pct")).alias("base_pts"))
        .with_columns(
            (
                pl.col("qb_pts")
                + pl.col("rush_pts")
                + pl.col("recv_pts")
                + pl.col("def_pts")
                + pl.col("base_pts")
            ).alias("value_raw")
        )
    )
