"""Phase 1 pipeline: compute player point values and 2026 team ratings."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C
from src import data, teams, valuation

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main() -> None:
    pbp = data.pbp(C.SEASONS)
    snaps = data.snap_counts(C.SEASONS)
    rosters_all = pl.concat(
        [data.rosters(s) for s in C.SEASONS], how="vertical_relaxed"
    )
    try:
        target = data.rosters(C.TARGET_SEASON)
        target_season = C.TARGET_SEASON
    except Exception:
        target = data.rosters(max(C.SEASONS))
        target_season = max(C.SEASONS)
    print(f"Valuing {target_season} rosters from {min(C.SEASONS)}-{max(C.SEASONS)} data")

    players = valuation.player_values(pbp, snaps, target, rosters_all)
    tr, ranked, scale = teams.team_ratings(players)

    cols = ["full_name", "team", "position", "grp", "points", "qb", "rush", "recv",
            "def", "base", "snap_pct", "rot_w", "years_exp"]
    ranked.sort("points", descending=True).select(cols).write_csv(
        OUT / f"player_values_{target_season}.csv"
    )
    tr.select("team", "rating").write_csv(OUT / f"team_ratings_{target_season}.csv")

    print(f"\nScale: {scale:.4f} raw->points | {ranked.height} players valued\n")
    print("=== Top 15 players (all positions) ===")
    print(ranked.sort("points", descending=True).select(
        "full_name", "team", "position", "points", "qb", "rush", "recv", "def", "base"
    ).head(15))
    print("\n=== Top 12 QBs ===")
    print(ranked.filter(pl.col("grp") == "QB").sort("points", descending=True).select(
        "full_name", "team", "points", "qb", "rush", "base"
    ).head(12))
    print("\n=== Team ratings (points vs average team) ===")
    with pl.Config(tbl_rows=32):
        print(tr.select("team", "rating"))
    best, worst = tr.row(0), tr.row(-1)
    print(f"\nSanity: {best[0]} vs {worst[0]} on neutral field -> "
          f"{best[0]} -{best[2] - worst[2]:.1f}")


if __name__ == "__main__":
    main()
