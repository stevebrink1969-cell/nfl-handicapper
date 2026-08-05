"""Injury adjustment engine.

Each listed player gets a play probability from his report designation and
practice participation. The team's line impact is the value gap between the
player and his replacement (next man up in the same position group), scaled
by his expected snap weight and the probability he sits.
"""

import polars as pl

from . import data

P_PLAY_STATUS = {"Out": 0.0, "Doubtful": 0.05}
P_PLAY_QUESTIONABLE = {
    "Did Not Participate In Practice": 0.50,
    "Limited Participation in Practice": 0.75,
    "Full Participation in Practice": 0.90,
}
P_PLAY_QUESTIONABLE_DEFAULT = 0.75


def week_report(season: int, week: int) -> pl.DataFrame:
    """Official injury report rows for one game week (empty in offseason)."""
    empty = pl.DataFrame(
        schema={"gsis_id": pl.String, "report_status": pl.String,
                "practice_status": pl.String}
    )
    try:
        inj = data.injuries(season)
    except Exception:
        return empty
    inj = inj.filter(pl.col("week") == week)
    if inj.height == 0:
        return empty
    return (
        inj.select("gsis_id", "report_status", "practice_status")
        .filter(pl.col("gsis_id").is_not_null())
        .unique(subset=["gsis_id"], keep="last")
    )


def _p_play(report_status: str | None, practice_status: str | None) -> float:
    if report_status in P_PLAY_STATUS:
        return P_PLAY_STATUS[report_status]
    if report_status == "Questionable":
        return P_PLAY_QUESTIONABLE.get(practice_status, P_PLAY_QUESTIONABLE_DEFAULT)
    return 1.0


def apply(ranked: pl.DataFrame, report: pl.DataFrame) -> pl.DataFrame:
    """Attach p_play and per-player expected line impact to the ranked table.

    impact = (player points - replacement points) x rotation weight x P(sits).
    Replacement is the best same-team/group player outside the rotation.
    """
    out = ranked.join(report, on="gsis_id", how="left")
    p = pl.struct("report_status", "practice_status").map_elements(
        lambda s: _p_play(s["report_status"], s["practice_status"]),
        return_dtype=pl.Float64,
    )
    out = out.with_columns(p.alias("p_play"))

    repl = (
        out.filter(pl.col("rot_w") == 0.0)
        .group_by("team", "grp")
        .agg(pl.col("points").max().alias("repl_pts"))
    )
    out = out.join(repl, on=["team", "grp"], how="left").with_columns(
        pl.col("repl_pts").fill_null(0.0)
    )
    return out.with_columns(
        (
            (pl.col("points") - pl.col("repl_pts")).clip(lower_bound=0.0)
            * pl.col("rot_w")
            * (1.0 - pl.col("p_play"))
        )
        .round(2)
        .alias("impact")
    )


def team_adjustments(adjusted: pl.DataFrame) -> pl.DataFrame:
    """Points each team loses to injuries this week (negative adjustment)."""
    return adjusted.group_by("team").agg((-pl.col("impact").sum()).round(2).alias("inj_adj"))
