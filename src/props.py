"""Player stat projection engine for props.

For each player-game, projects the main prop stats as role x volume x
efficiency, all walk-forward (only data from before that game):

  role       player's share of team targets/carries/attempts, current season
             blended with prior season (K_SHARE pseudo-games)
  volume     team pass attempts / carries per game, blended (K_TEAM)
  efficiency catch rate, yards per target/carry/attempt, TD rates — shrunk
             toward position/league norms so small samples can't lie

Projected stats: pass_yds, pass_tds, completions, rush_yds, carries,
receptions, rec_yds, anytime_td_lambda (rush+rec TDs).
"""

import polars as pl

import nflreadpy as nfl

from . import data

K_SHARE = 4     # pseudo-games of prior-season share
K_TEAM = 6      # pseudo-games of prior-season team volume
PR_CR = 15      # shrinkage targets for catch rate
PR_YPT = 30     # ... yards per target
PR_YPC = 60     # ... yards per carry
PR_YPA = 100    # ... yards per attempt (QB)
PR_TD = 10      # pseudo-games for TD rate

# League norms (stable constants, computed from 2021-2022)
LG = {"cr": 0.65, "ypt": 7.6, "ypc": 4.3, "ypa": 7.0,
      "comp_rate": 0.64, "td_per_att": 0.044}


def _cached_stats(seasons: list[int]) -> pl.DataFrame:
    tag = f"pstats_{min(seasons)}_{max(seasons)}"
    return data._cached(tag, lambda: nfl.load_player_stats(seasons))


def load_stats(seasons: list[int]) -> pl.DataFrame:
    ps = _cached_stats(seasons)
    pos = "position" if "position" in ps.columns else "position_group"
    return ps.select(
        "player_id", "player_display_name", pl.col(pos).alias("pos"),
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        "team", "completions", "attempts", "passing_yards", "passing_tds",
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ).filter(pl.col("team").is_not_null())


def build_projections(seasons: list[int]) -> pl.DataFrame:
    """Walk-forward projections for every player-game in `seasons` (needs the
    season before min(seasons) loaded too for priors)."""
    all_seasons = [min(seasons) - 1] + list(seasons)
    ps = load_stats(all_seasons)

    team_wk = ps.group_by("team", "season", "week").agg(
        pl.col("attempts").sum().alias("tm_att"),
        pl.col("carries").sum().alias("tm_car"),
        pl.col("targets").sum().alias("tm_tgt"),
    ).sort("team", "season", "week")
    team_wk = team_wk.with_columns(
        pl.int_range(pl.len()).over("team", "season").alias("tm_gp"),
        pl.col("tm_att").cum_sum().shift(1, fill_value=0).over("team", "season").alias("ctm_att"),
        pl.col("tm_car").cum_sum().shift(1, fill_value=0).over("team", "season").alias("ctm_car"),
        pl.col("tm_tgt").cum_sum().shift(1, fill_value=0).over("team", "season").alias("ctm_tgt"),
    )
    tm_prior = team_wk.group_by("team", "season").agg(
        pl.col("tm_att").mean().alias("p_tm_att"),
        pl.col("tm_car").mean().alias("p_tm_car"),
        pl.col("tm_tgt").mean().alias("p_tm_tgt"),
    ).with_columns((pl.col("season") + 1).alias("season"))

    p = ps.sort("player_id", "season", "week")
    cums = ["attempts", "completions", "passing_yards", "passing_tds",
            "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds"]
    p = p.with_columns(
        pl.int_range(pl.len()).over("player_id", "season").alias("gp"),
        *[pl.col(c).cum_sum().shift(1, fill_value=0).over("player_id", "season")
          .alias(f"c_{c}") for c in cums],
    )
    prior = ps.group_by("player_id", "season").agg(
        pl.len().alias("p_gp"),
        *[pl.col(c).sum().alias(f"p_{c}") for c in cums],
        pl.col("team").last().alias("p_team"),
    ).with_columns((pl.col("season") + 1).alias("season"))
    prior_team_vol = tm_prior.rename({"team": "p_team"})
    prior = prior.join(prior_team_vol, on=["p_team", "season"], how="left")

    g = (
        p.join(team_wk.select("team", "season", "week", "tm_gp", "ctm_att",
                              "ctm_car", "ctm_tgt"),
               on=["team", "season", "week"], how="left")
        .join(prior.drop("p_team"), on=["player_id", "season"], how="left")
        .join(tm_prior, on=["team", "season"], how="left")
        .with_columns(pl.col("p_gp").fill_null(0))
    )

    def team_vol(cur_cum: str, prior_pg: str, default: float, out: str):
        return (
            ((pl.col(cur_cum) / pl.max_horizontal(pl.col("tm_gp"), 1)) * pl.col("tm_gp")
             + pl.col(prior_pg).fill_null(default) * K_TEAM)
            / (pl.col("tm_gp") + K_TEAM)
        ).alias(out)

    def share(cur_num: str, cur_den: str, p_num: str, p_den: str, out: str):
        cur_share = pl.when(pl.col(cur_den) > 0).then(pl.col(cur_num) / pl.col(cur_den)).otherwise(None)
        pri_share = pl.when(pl.col(p_den).fill_null(0) * pl.col("p_gp") > 0).then(
            pl.col(p_num).fill_null(0.0) / (pl.col(p_den) * pl.col("p_gp"))).otherwise(None)
        return (
            (cur_share.fill_null(0.0) * pl.col("gp") + pri_share.fill_null(0.0) * K_SHARE)
            / (pl.col("gp") + K_SHARE)
        ).alias(out)

    def eff(cur_num: str, cur_den: str, p_num: str, p_den: str, prior_n: float,
            lg: float, out: str):
        num = pl.col(cur_num) + pl.col(p_num).fill_null(0.0)
        den = pl.col(cur_den) + pl.col(p_den).fill_null(0.0)
        return ((num + prior_n * lg) / (den + prior_n)).alias(out)

    g = g.with_columns(
        team_vol("ctm_att", "p_tm_att", 33.0, "tv_att"),
        team_vol("ctm_car", "p_tm_car", 26.0, "tv_car"),
        team_vol("ctm_tgt", "p_tm_tgt", 33.0, "tv_tgt"),
        share("c_targets", "ctm_tgt", "p_targets", "p_tm_tgt", "sh_tgt"),
        share("c_carries", "ctm_car", "p_carries", "p_tm_car", "sh_car"),
        share("c_attempts", "ctm_att", "p_attempts", "p_tm_att", "sh_att"),
        eff("c_receptions", "c_targets", "p_receptions", "p_targets", PR_CR, LG["cr"], "e_cr"),
        eff("c_receiving_yards", "c_targets", "p_receiving_yards", "p_targets", PR_YPT, LG["ypt"], "e_ypt"),
        eff("c_rushing_yards", "c_carries", "p_rushing_yards", "p_carries", PR_YPC, LG["ypc"], "e_ypc"),
        eff("c_passing_yards", "c_attempts", "p_passing_yards", "p_attempts", PR_YPA, LG["ypa"], "e_ypa"),
        eff("c_completions", "c_attempts", "p_completions", "p_attempts", PR_YPA, LG["comp_rate"], "e_comp"),
        eff("c_passing_tds", "c_attempts", "p_passing_tds", "p_attempts", PR_YPA, LG["td_per_att"], "e_ptd"),
    )
    td_pg = (
        (pl.col("c_rushing_tds") + pl.col("c_receiving_tds")
         + pl.col("p_rushing_tds").fill_null(0.0) + pl.col("p_receiving_tds").fill_null(0.0))
        / (pl.col("gp") + pl.col("p_gp") + PR_TD)
    )
    g = g.with_columns(
        (pl.col("sh_tgt") * pl.col("tv_tgt")).alias("proj_targets"),
        (pl.col("sh_car") * pl.col("tv_car")).alias("proj_carries"),
        (pl.col("sh_att") * pl.col("tv_att")).alias("proj_attempts"),
        td_pg.alias("td_lambda"),
    ).with_columns(
        (pl.col("proj_targets") * pl.col("e_cr")).round(2).alias("proj_receptions"),
        (pl.col("proj_targets") * pl.col("e_ypt")).round(1).alias("proj_rec_yds"),
        (pl.col("proj_carries") * pl.col("e_ypc")).round(1).alias("proj_rush_yds"),
        (pl.col("proj_attempts") * pl.col("e_ypa")).round(1).alias("proj_pass_yds"),
        (pl.col("proj_attempts") * pl.col("e_comp")).round(2).alias("proj_completions"),
        (pl.col("proj_attempts") * pl.col("e_ptd")).round(3).alias("proj_pass_tds"),
    )
    return g.filter(pl.col("season").is_in(seasons)).select(
        "player_id", "player_display_name", "pos", "season", "week", "team",
        "gp", "p_gp",
        "proj_targets", "proj_carries", "proj_attempts",
        "proj_receptions", "proj_rec_yds", "proj_rush_yds",
        "proj_pass_yds", "proj_completions", "proj_pass_tds", "td_lambda",
        "receptions", "receiving_yards", "rushing_yards", "carries",
        "passing_yards", "completions", "passing_tds",
        (pl.col("rushing_tds") + pl.col("receiving_tds")).alias("any_tds"),
    )
