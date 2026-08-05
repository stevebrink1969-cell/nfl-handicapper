"""Team ratings: sum of expected-lineup player values, scaled to spread points.

The model splits each team's raw value into three components — QB play,
non-QB on-field production (rush/recv/defensive plays), and snap-share base —
because Phase 2 calibration fits a separate points-scale to each against real
closing lines. Production ratings apply those fitted coefficients."""

import json
from pathlib import Path

import polars as pl

from . import config as C

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "output" / "calibration.json"


def load_calibration() -> dict | None:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text())
    return None


def ranked_with_rotation(players: pl.DataFrame) -> pl.DataFrame:
    """Rank players within team/position group and attach expected snap weight."""
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
    return (
        ranked.join(rot, on=["grp", "rk"], how="left")
        .with_columns(pl.col("rot_w").fill_null(0.0))
        .with_columns(
            (pl.col("rush_pts") + pl.col("recv_pts") + pl.col("def_pts")).alias(
                "perf_raw"
            )
        )
    )


def team_components(players: pl.DataFrame) -> pl.DataFrame:
    """Rotation-weighted raw component sums per team (calibration inputs)."""
    r = ranked_with_rotation(players)
    return r.group_by("team").agg(
        (pl.col("qb_pts") * pl.col("rot_w")).sum().alias("qb_c"),
        (pl.col("perf_raw") * pl.col("rot_w")).sum().alias("perf_c"),
        (pl.col("base_pts") * pl.col("rot_w")).sum().alias("base_c"),
    )


def team_ratings(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Returns (team table, players with scaled points + lineup weight, cal info).

    With calibration: player points = k_qb*qb + k_perf*perf + k_base*base,
    already in real spread-points units. Without: fall back to normalizing
    team std to TEAM_RATING_STD.
    """
    ranked = ranked_with_rotation(players)
    cal = load_calibration()
    if cal:
        k_qb, k_perf, k_base = cal["k_qb"], cal["k_perf"], cal["k_base"]
        ranked = ranked.with_columns(
            (
                k_qb * pl.col("qb_pts")
                + k_perf * pl.col("perf_raw")
                + k_base * pl.col("base_pts")
            ).alias("pts_scaled"),
            (k_qb * pl.col("qb_pts")).round(2).alias("qb"),
            (k_perf * pl.col("rush_pts")).round(2).alias("rush"),
            (k_perf * pl.col("recv_pts")).round(2).alias("recv"),
            (k_perf * pl.col("def_pts")).round(2).alias("def"),
            (k_base * pl.col("base_pts")).round(2).alias("base"),
        )
        info = {"mode": "calibrated", **cal}
    else:
        scale = C.TEAM_RATING_STD / (
            team_components(players)
            .with_columns((pl.col("qb_c") + pl.col("perf_c") + pl.col("base_c")).alias("t"))["t"]
            .std()
        )
        ranked = ranked.with_columns(
            (scale * pl.col("value_raw")).alias("pts_scaled"),
            (scale * pl.col("qb_pts")).round(2).alias("qb"),
            (scale * pl.col("rush_pts")).round(2).alias("rush"),
            (scale * pl.col("recv_pts")).round(2).alias("recv"),
            (scale * pl.col("def_pts")).round(2).alias("def"),
            (scale * pl.col("base_pts")).round(2).alias("base"),
        )
        info = {"mode": "uncalibrated", "scale": scale}
    ranked = ranked.with_columns(pl.col("pts_scaled").round(2).alias("points"))
    tr = ranked.group_by("team").agg(
        (pl.col("pts_scaled") * pl.col("rot_w")).sum().alias("raw")
    )
    mean = tr["raw"].mean()
    tr = tr.with_columns(((pl.col("raw") - mean) * 1.0).round(2).alias("rating")).sort(
        "rating", descending=True
    )
    return tr, ranked, info
