"""DraftKings player prop odds via The Odds API event endpoints.

Every scan appends snapshots to output/props_history.csv (line movement +
future grading). The /events list is free; each event odds call costs
credits per market, so fetching is limited to games within
PROPS_MAX_DAYS_AHEAD.
"""

import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import config as C
from .odds import BOOK, NAME2ABBR, _api_key

PROP_HIST = Path(__file__).resolve().parent.parent / "output" / "props_history.csv"
PROP_FIELDS = ["fetched_at", "commence_time", "home", "away", "market",
               "player", "line", "side", "price"]
MARKETS = [
    "player_pass_yds", "player_pass_tds", "player_rush_yds",
    "player_rush_attempts", "player_receptions", "player_reception_yds",
    "player_anytime_td",
]


def fetch_props() -> pl.DataFrame:
    """Fetch current DK props for games inside the window. Returns long frame
    (one row per player-market-side)."""
    empty = pl.DataFrame(schema={f: pl.String for f in PROP_FIELDS[:7]}
                         ).with_columns(pl.lit(0.0).alias("line"), pl.lit(0).alias("price"))
    key = _api_key()
    if not key:
        print("props: no API key")
        return empty
    try:
        url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={key}"
        with urllib.request.urlopen(url, timeout=30) as r:
            events = json.load(r)
    except Exception as e:
        print(f"props: events fetch failed ({e})")
        return empty
    now = datetime.now(timezone.utc)
    upcoming = []
    for ev in events:
        try:
            ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        if 0 <= (ct - now).days <= C.PROPS_MAX_DAYS_AHEAD:
            upcoming.append(ev)
    if not upcoming:
        print("props: no games inside fetch window; skipping (saves quota)")
        return empty

    rows = []
    stamp = now.isoformat(timespec="seconds")
    for ev in upcoming:
        home = NAME2ABBR.get(ev.get("home_team", ""))
        away = NAME2ABBR.get(ev.get("away_team", ""))
        if not home or not away:
            continue
        url = (
            f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/"
            f"{ev['id']}/odds?apiKey={key}&regions=us&markets={','.join(MARKETS)}"
            f"&bookmakers={BOOK}&oddsFormat=american"
        )
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                eo = json.load(r)
        except Exception as e:
            print(f"props: event fetch failed for {away}@{home} ({e})")
            continue
        for bk in eo.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    player = o.get("description") or o.get("name")
                    side = o.get("name") if o.get("description") else "Yes"
                    if side not in ("Over", "Under", "Yes"):
                        continue
                    rows.append({
                        "fetched_at": stamp, "commence_time": ev["commence_time"],
                        "home": home, "away": away, "market": mk.get("key"),
                        "player": player, "line": float(o.get("point") or 0.0),
                        "side": side, "price": int(o.get("price") or 0),
                    })
    if rows:
        new_file = not PROP_HIST.exists()
        PROP_HIST.parent.mkdir(exist_ok=True)
        with PROP_HIST.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PROP_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(rows)
    print(f"props: {len(rows)} prop odds rows from {len(upcoming)} games")
    if not rows:
        return empty
    return pl.DataFrame(rows)
