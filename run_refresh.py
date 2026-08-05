"""Production refresh: pull fresh data, rebuild values, injuries, slate, site.

This is the entry point the scheduled job (Phase 4) runs. Use --keep-cache to
skip re-downloading current-season data (useful for local dev)."""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C
from src import data, injuries, odds, pipeline, site, slate

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

    payload = site.assemble(games, week, sched_season, built.tr, adjusted,
                            inj_adj, built.info)
    out = site.write(payload)

    listed = report.height
    print(f"Season {sched_season} week {week}: {games.height} games, "
          f"{listed} injury listings, in_season={built.in_season}")
    print(games.select("away_team", "home_team", "market_line", "mkt_src",
                       "model_line", "edge").head(16))
    print(f"Site written: {out}")


if __name__ == "__main__":
    main()
