"""Measure same-game stat correlations from projection residuals, 2023-2025.

For every player-game, z = (actual - projection) / sd. Pairs of z-scores
within the same game are classified by relationship and stat pair; Pearson r
per category feeds the SGP joint-probability model.

Writes output/sgp_correlations.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src import props

SEASONS = [2023, 2024, 2025]

STATS = [
    # (stat key, actual col, proj col, sd market key, usage col, min usage)
    ("pass_yds", "passing_yards", "proj_pass_yds", "player_pass_yds", "proj_attempts", 15.0),
    ("rush_yds", "rushing_yards", "proj_rush_yds", "player_rush_yds", "proj_carries", 6.0),
    ("carries", "carries", "proj_carries", "player_rush_attempts", "proj_carries", 6.0),
    ("receptions", "receptions", "proj_receptions", "player_receptions", "proj_targets", 3.0),
    ("rec_yds", "receiving_yards", "proj_rec_yds", "player_reception_yds", "proj_targets", 3.0),
]


def classify(row: dict) -> str:
    same_player = row["player_id"] == row["player_id_2"]
    same_team = row["team"] == row["team_2"]
    s1, s2 = sorted([row["stat"], row["stat_2"]])
    pair = f"{s1}|{s2}"
    passing = {"pass_yds"}
    receiving = {"receptions", "rec_yds"}
    rushing = {"rush_yds", "carries"}
    if same_player:
        return f"same_player|{pair}"
    if same_team:
        if (s1 in passing and s2 in receiving) or (s2 in passing and s1 in receiving):
            return f"qb_receiver|{pair}"
        if s1 in receiving and s2 in receiving:
            return f"teammates_recv|{pair}"
        if (s1 in passing and s2 in rushing) or (s1 in rushing and s2 in passing):
            return f"qb_rusher|{pair}"
        return f"teammates_other|{pair}"
    # opponents
    if s1 in passing and s2 in passing:
        return f"opp_pass_pass|{pair}"
    return f"opp_other|{pair}"


def main() -> None:
    g = props.build_projections(SEASONS)
    parts = []
    for stat, actual, proj, sdkey, use_col, min_use in STATS:
        a, b = C.PROP_SD[sdkey]
        d = g.filter(
            (pl.col(use_col) >= min_use) & pl.col(actual).is_not_null()
            & (pl.col("gp") + pl.col("p_gp") >= 4) & pl.col("opponent").is_not_null()
        ).with_columns(
            pl.min_horizontal(pl.col("team"), pl.col("opponent")).alias("t1"),
            pl.max_horizontal(pl.col("team"), pl.col("opponent")).alias("t2"),
        ).with_columns(
            (pl.col("season").cast(pl.String) + "-" + pl.col("week").cast(pl.String)
             + "-" + pl.col("t1") + "-" + pl.col("t2")).alias("game_key"),
            ((pl.col(actual) - pl.col(proj))
             / pl.max_horizontal(a + b * pl.col(proj), 1.0)).alias("z"),
            pl.lit(stat).alias("stat"),
        ).select("game_key", "team", "player_id", "stat", "z")
        parts.append(d)
    z = pl.concat(parts)
    pairs = z.join(z, on="game_key", suffix="_2").filter(
        (pl.col("player_id") + pl.col("stat")) < (pl.col("player_id_2") + pl.col("stat_2"))
    )
    print(f"z-scores: {z.height} | within-game pairs: {pairs.height}")

    s1 = pl.min_horizontal("stat", "stat_2")
    s2 = pl.max_horizontal("stat", "stat_2")
    pair = s1 + pl.lit("|") + s2
    passing = s1.is_in(["pass_yds"]), s2.is_in(["pass_yds"])
    recv1, recv2 = s1.is_in(["receptions", "rec_yds"]), s2.is_in(["receptions", "rec_yds"])
    rush1, rush2 = s1.is_in(["rush_yds", "carries"]), s2.is_in(["rush_yds", "carries"])
    same_player = pl.col("player_id") == pl.col("player_id_2")
    same_team = pl.col("team") == pl.col("team_2")
    cat = (
        pl.when(same_player).then(pl.lit("same_player|") + pair)
        .when(same_team & ((passing[0] & recv2) | (passing[1] & recv1)))
        .then(pl.lit("qb_receiver|") + pair)
        .when(same_team & recv1 & recv2).then(pl.lit("teammates_recv|") + pair)
        .when(same_team & ((passing[0] & rush2) | (passing[1] & rush1)))
        .then(pl.lit("qb_rusher|") + pair)
        .when(same_team).then(pl.lit("teammates_other|") + pair)
        .when(passing[0] & passing[1]).then(pl.lit("opp_pass_pass|") + pair)
        .otherwise(pl.lit("opp_other|") + pair)
    )
    stats = (
        pairs.with_columns(cat.alias("cat"))
        .group_by("cat")
        .agg(pl.corr("z", "z_2").alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= 300)
        .sort("cat")
    )
    out = {}
    for row in stats.iter_rows(named=True):
        out[row["cat"]] = {"r": round(row["r"], 3), "n": row["n"]}
        print(f'{row["cat"]:<45} r={row["r"]:+.3f}  n={row["n"]}')
    path = Path(__file__).resolve().parent.parent / "output" / "sgp_correlations.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
