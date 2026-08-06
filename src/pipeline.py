"""One-call production build shared by run_phase1 and the site refresh."""

from types import SimpleNamespace

import polars as pl

from . import config as C
from . import data, teams, valuation


def build() -> SimpleNamespace:
    # In-season: include the target season's own games once they exist
    # (weighted CUR_SEASON_WEIGHT). Before Week 1 this falls back cleanly.
    try:
        pbp = data.pbp(C.SEASONS + [C.TARGET_SEASON])
        snaps = data.snap_counts(C.SEASONS + [C.TARGET_SEASON])
        in_season = True
    except Exception:
        pbp = data.pbp(C.SEASONS)
        snaps = data.snap_counts(C.SEASONS)
        in_season = False
    rosters_all = pl.concat(
        [data.rosters(s) for s in C.SEASONS], how="vertical_relaxed"
    )
    try:
        target = data.rosters(C.TARGET_SEASON)
        season = C.TARGET_SEASON
    except Exception:
        season = max(C.SEASONS)
        target = data.rosters(season)

    wts = C.season_weights(season)
    if in_season:
        wts[season] = C.CUR_SEASON_WEIGHT
    players = valuation.player_values(pbp, snaps, target, rosters_all, wts)
    tr, ranked, info = teams.team_ratings(players)
    comps = teams.team_components(players)
    return SimpleNamespace(
        tr=tr, ranked=ranked, info=info, comps=comps, season=season,
        in_season=in_season
    )
