"""Cached nflverse data loaders. First call downloads and snapshots to data/;
later calls read the local parquet (delete data/ to force a refresh)."""

from pathlib import Path

import nflreadpy as nfl
import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

PBP_COLS = [
    "season", "week", "game_id", "posteam", "defteam", "play_type", "qtr",
    "epa", "qb_epa", "wp", "score_differential", "qb_dropback", "qb_scramble",
    "sack", "passer_player_id", "passer_player_name", "rusher_player_id",
    "rusher_player_name", "receiver_player_id", "receiver_player_name",
    "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id",
    "interception_player_id", "pass_defense_1_player_id",
    "pass_defense_2_player_id", "tackle_for_loss_1_player_id",
    "forced_fumble_player_1_player_id",
]


def _cached(name: str, fetch) -> pl.DataFrame:
    f = DATA_DIR / f"{name}.parquet"
    if f.exists():
        return pl.read_parquet(f)
    df = fetch()
    df.write_parquet(f)
    return df


def pbp(seasons: list[int]) -> pl.DataFrame:
    tag = f"pbp_{min(seasons)}_{max(seasons)}"
    df = _cached(tag, lambda: nfl.load_pbp(seasons))
    keep = [c for c in PBP_COLS if c in df.columns]
    return df.select(keep)


def rosters(season: int) -> pl.DataFrame:
    return _cached(f"rosters_{season}", lambda: nfl.load_rosters(season))


def snap_counts(seasons: list[int]) -> pl.DataFrame:
    tag = f"snaps_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_snap_counts(seasons))


def schedules(seasons: list[int]) -> pl.DataFrame:
    tag = f"sched_{min(seasons)}_{max(seasons)}"
    return _cached(tag, lambda: nfl.load_schedules(seasons))


def injuries(season: int) -> pl.DataFrame:
    return _cached(f"injuries_{season}", lambda: nfl.load_injuries(season))


def depth_charts(season: int) -> pl.DataFrame:
    return _cached(f"depth_{season}", lambda: nfl.load_depth_charts(season))
