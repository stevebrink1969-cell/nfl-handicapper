"""Production refresh: pull fresh data, rebuild values, injuries, slate, site.

This is the entry point the scheduled job (Phase 4) runs. Use --keep-cache to
skip re-downloading current-season data (useful for local dev)."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C
from src import data, injuries, odds, pipeline, propscan, sgp, site, slate, totals

OUT = Path(__file__).resolve().parent / "output"


def clear_current_caches() -> None:
    """Delete caches that go stale during the season; history stays cached."""
    patterns = [
        f"rosters_{C.TARGET_SEASON}", f"injuries_{C.TARGET_SEASON}",
        f"depth_{C.TARGET_SEASON}", f"sched_{C.TARGET_SEASON}_{C.TARGET_SEASON}",
        f"pbp_{min(C.SEASONS)}_{C.TARGET_SEASON}",
        f"snaps_{min(C.SEASONS)}_{C.TARGET_SEASON}",
    ]
    for p in patterns:
        f = data.DATA_DIR / f"{p}.parquet"
        if f.exists():
            f.unlink()


def main() -> None:
    if "--keep-cache" not in sys.argv:
        clear_current_caches()

    odds.fetch_snapshot()

    built = pipeline.build()
    sched_season = C.TARGET_SEASON  # slate uses the target season's schedule
    _, week = slate.upcoming_week(sched_season)

    report = injuries.week_report(built.season, week)
    adjusted = injuries.apply(built.ranked, report)
    inj_adj = injuries.team_adjustments(adjusted)

    games, week = slate.build(built.tr, inj_adj, built.info.get("hfa", 0.0),
                              sched_season, week, odds.lines_table())
    games = totals.predict(games, built.comps, sched_season)
    tot_odds = odds.totals_lines_table()
    if tot_odds.height > 0:
        games = games.join(
            tot_odds.rename({"home": "home_team", "away": "away_team"}),
            on=["home_team", "away_team"], how="left",
        )
    else:
        games = games.with_columns(
            pl.lit(None, pl.Float64).alias("open_total"),
            pl.lit(None, pl.Float64).alias("book_total"),
        )
    games = games.with_columns(
        pl.coalesce("book_total", "total_line").alias("market_total"),
        pl.when(pl.col("book_total").is_not_null())
        .then(pl.lit("book")).otherwise(pl.lit("consensus")).alias("tot_src"),
    ).with_columns(
        (pl.col("model_total") - pl.col("market_total")).round(1).alias("total_edge")
    )

    try:
        all_plays = propscan.scan(sched_season)
        prop_plays = propscan.top_per_slot(all_plays)
        sgps = sgp.build(all_plays)
    except Exception as e:
        print(f"props scan failed: {e}")
        prop_plays, sgps = None, []

    payload = site.assemble(games, week, sched_season, built.tr, adjusted,
                            inj_adj, built.info, prop_plays, sgps)
    out = site.write(payload)

    listed = report.height
    print(f"Season {sched_season} week {week}: {games.height} games, "
          f"{listed} injury listings, in_season={built.in_season}")
    print(games.select("away_team", "home_team", "market_line", "mkt_src",
                       "model_line", "edge").head(16))
    print(f"Site written: {out}")


if __name__ == "__main__":
    main()
