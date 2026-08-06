"""Production totals model: predict each upcoming game's fair total.

Uses the coefficients fitted by scripts/fit_totals.py on 2021-2025. Team
rates (points for/against, EPA, pace) blend the current season's games with
last season's, so predictions sharpen as the season progresses — same
formulas as the backtest.

Weather: windx/cold enter as 0 until a forecast source is wired in (games
are outside forecast range before the season anyway); dome comes from the
schedule's roof field. TODO: Open-Meteo forecasts by stadium in-season.
"""

import json
from pathlib import Path

import polars as pl

from . import config as C
from . import data

CAL_FILE = Path(__file__).resolve().parent.parent / "output" / "totals_calibration.json"
K_TEAM = 6      # prior-season weight (games) for team rates
K_ENV = 30      # prior-season weight (games) for scoring environment
LG_PACE = 63.0  # league-average offensive plays/game
LG_PTS = 22.0


def load_calibration() -> dict | None:
    if CAL_FILE.exists():
        return json.loads(CAL_FILE.read_text())
    return None


def _team_rates(target_season: int) -> pl.DataFrame:
    """Blended per-team pf/pa/oepa/depa/pace as of now."""
    prior_season = target_season - 1
    sched_p = data.schedules([prior_season]).filter(pl.col("home_score").is_not_null())
    try:
        sched_c = data.schedules([target_season]).filter(pl.col("home_score").is_not_null())
    except Exception:
        sched_c = sched_p.clear()

    def scores_long(s: pl.DataFrame) -> pl.DataFrame:
        h = s.select(pl.col("home_team").alias("team"), pl.col("home_score").alias("pf"),
                     pl.col("away_score").alias("pa"))
        a = s.select(pl.col("away_team").alias("team"), pl.col("away_score").alias("pf"),
                     pl.col("home_score").alias("pa"))
        return pl.concat([h, a])

    pbp_seasons = [prior_season]
    try:
        pbp = data.pbp(C.SEASONS + [target_season])
        pbp_seasons.append(target_season)
    except Exception:
        pbp = data.pbp(C.SEASONS)
    pbp = pbp.with_columns(pl.col("season").cast(pl.Int32))
    plays = pbp.filter(
        pl.col("play_type").is_in(["pass", "run"]) & pl.col("epa").is_not_null()
    )

    def epa_pace(season: int) -> pl.DataFrame:
        p = plays.filter(pl.col("season") == season)
        o = p.group_by(pl.col("posteam").alias("team")).agg(
            pl.col("epa").mean().alias("oepa"),
            (pl.len() / pl.col("game_id").n_unique()).alias("pace"),
        )
        d = p.group_by(pl.col("defteam").alias("team")).agg(
            pl.col("epa").mean().alias("depa"))
        return o.join(d, on="team", how="full", coalesce=True)

    prior = (
        scores_long(sched_p).group_by("team")
        .agg(pl.col("pf").mean().alias("p_pf"), pl.col("pa").mean().alias("p_pa"))
        .join(epa_pace(prior_season).rename(
            {"oepa": "p_oepa", "depa": "p_depa", "pace": "p_pace"}), on="team", how="left")
    )
    if sched_c.height > 0:
        cur = (
            scores_long(sched_c).group_by("team")
            .agg(pl.col("pf").mean().alias("c_pf"), pl.col("pa").mean().alias("c_pa"),
                 pl.len().alias("n"))
            .join(epa_pace(target_season).rename(
                {"oepa": "c_oepa", "depa": "c_depa", "pace": "c_pace"}), on="team", how="left")
        )
        t = prior.join(cur, on="team", how="left").with_columns(pl.col("n").fill_null(0))
    else:
        t = prior.with_columns(
            pl.lit(0).alias("n"), pl.lit(None, pl.Float64).alias("c_pf"),
            pl.lit(None, pl.Float64).alias("c_pa"), pl.lit(None, pl.Float64).alias("c_oepa"),
            pl.lit(None, pl.Float64).alias("c_depa"), pl.lit(None, pl.Float64).alias("c_pace"),
        )

    def blend(cur_c: str, pri_c: str, default: float, out: str):
        return (
            (pl.col(cur_c).fill_null(0.0) * pl.col("n")
             + pl.col(pri_c).fill_null(default) * K_TEAM)
            / (pl.col("n") + K_TEAM)
        ).alias(out)

    return t.with_columns(
        blend("c_pf", "p_pf", LG_PTS, "pf"), blend("c_pa", "p_pa", LG_PTS, "pa"),
        blend("c_oepa", "p_oepa", 0.0, "oepa"), blend("c_depa", "p_depa", 0.0, "depa"),
        blend("c_pace", "p_pace", LG_PACE, "pace"),
    ).select("team", "pf", "pa", "oepa", "depa", "pace")


def _env(target_season: int) -> float:
    prior = data.schedules([target_season - 1]).filter(pl.col("home_score").is_not_null())
    p_env = (prior["home_score"] + prior["away_score"]).mean()
    try:
        cur = data.schedules([target_season]).filter(pl.col("home_score").is_not_null())
        n = cur.height
        c_env = (cur["home_score"] + cur["away_score"]).mean() if n else 0.0
    except Exception:
        n, c_env = 0, 0.0
    return (c_env * n + p_env * K_ENV) / (n + K_ENV)


def predict(games: pl.DataFrame, comps: pl.DataFrame, target_season: int) -> pl.DataFrame:
    """Add model_total to the slate games frame (needs roof column)."""
    cal = load_calibration()
    if cal is None:
        return games.with_columns(pl.lit(None, pl.Float64).alias("model_total"))
    rates = _team_rates(target_season)
    env = _env(target_season)
    co = comps.select("team", "off_c", "defp_c")
    g = (
        games.join(co.rename({"team": "home_team", "off_c": "off_h", "defp_c": "def_h"}),
                   on="home_team", how="left")
        .join(co.rename({"team": "away_team", "off_c": "off_a", "defp_c": "def_a"}),
              on="away_team", how="left")
        .join(rates.rename({"team": "home_team", "pf": "pf_h", "pa": "pa_h",
                            "oepa": "oepa_h", "depa": "depa_h", "pace": "pace_h"}),
              on="home_team", how="left")
        .join(rates.rename({"team": "away_team", "pf": "pf_a", "pa": "pa_a",
                            "oepa": "oepa_a", "depa": "depa_a", "pace": "pace_a"}),
              on="away_team", how="left")
        .with_columns(
            pl.lit(1.0).alias("const"),
            (pl.col("off_h").fill_null(0) + pl.col("off_a").fill_null(0)).alias("offsum"),
            (pl.col("def_h").fill_null(0) + pl.col("def_a").fill_null(0)).alias("defsum"),
            (pl.col("pf_h").fill_null(LG_PTS) + pl.col("pf_a").fill_null(LG_PTS)
             + pl.col("pa_h").fill_null(LG_PTS) + pl.col("pa_a").fill_null(LG_PTS)).alias("scoring"),
            (pl.col("oepa_h").fill_null(0) + pl.col("oepa_a").fill_null(0)).alias("oepasum"),
            (pl.col("depa_h").fill_null(0) + pl.col("depa_a").fill_null(0)).alias("depasum"),
            (pl.col("pace_h").fill_null(LG_PACE) + pl.col("pace_a").fill_null(LG_PACE)
             - 2 * LG_PACE).alias("pacex"),
            pl.lit(env).alias("env"),
            pl.lit(0.0).alias("windx"),   # TODO: Open-Meteo forecast in-season
            pl.lit(0.0).alias("cold"),
            pl.col("roof").is_in(["dome", "closed"]).cast(pl.Float64).fill_null(0.0).alias("dome"),
        )
    )
    expr = pl.lit(0.0)
    for f, b in zip(cal["features"], cal["coefs"]):
        expr = expr + pl.col(f) * b
    g = g.with_columns(expr.round(1).alias("model_total"))
    return games.join(
        g.select("game_id", "model_total"), on="game_id", how="left"
    )
