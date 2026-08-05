"""Book line fetching (DraftKings) + snapshot history (The Odds API).

Every refresh appends a snapshot of the current book spread for each
upcoming game to output/odds_history.csv. The first snapshot ever seen for a
game is its opening line; the latest is the current line. That history powers
open->current movement display and, after games close, closing-line-value
analysis.

The API key is read from the ODDS_API_KEY env var or a local .env file —
never committed. Without a key, callers fall back to consensus lines.
"""

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

HIST = Path(__file__).resolve().parent.parent / "output" / "odds_history.csv"
BOOK = "draftkings"  # Caesars unavailable via The Odds API; DK tracks it closely
HIST_FIELDS = ["fetched_at", "commence_time", "home", "away", "book", "home_line"]

NAME2ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _api_key() -> str | None:
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def fetch_snapshot() -> int:
    """Fetch current book spreads and append to history. Returns row count
    appended (0 if no key or no lines posted)."""
    key = _api_key()
    if not key:
        print("odds: no ODDS_API_KEY set, using consensus lines")
        return 0
    url = (
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        f"?apiKey={key}&regions=us&markets=spreads&bookmakers={BOOK}"
        "&oddsFormat=american"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            events = json.load(r)
    except Exception as e:
        print(f"odds: fetch failed ({e}), using consensus lines")
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for ev in events:
        home = NAME2ABBR.get(ev.get("home_team", ""))
        away = NAME2ABBR.get(ev.get("away_team", ""))
        if not home or not away:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "spreads":
                    continue
                pt = next(
                    (o.get("point") for o in mk.get("outcomes", [])
                     if o.get("name") == ev["home_team"]),
                    None,
                )
                if pt is not None:
                    rows.append({
                        "fetched_at": now, "commence_time": ev["commence_time"],
                        "home": home, "away": away, "book": bk["key"],
                        # API points are the handicap (favorite negative);
                        # store as home margin to match nflverse convention
                        "home_line": -float(pt),
                    })
    if rows:
        new_file = not HIST.exists()
        HIST.parent.mkdir(exist_ok=True)
        with HIST.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HIST_FIELDS)
            if new_file:
                w.writeheader()
            w.writerows(rows)
    print(f"odds: appended {len(rows)} book line snapshots")
    return len(rows)


def lines_table() -> pl.DataFrame:
    """Opening (first-seen) and current (latest) book line per game."""
    empty = pl.DataFrame(schema={
        "home": pl.String, "away": pl.String,
        "open_line": pl.Float64, "book_line": pl.Float64,
    })
    if not HIST.exists():
        return empty
    h = pl.read_csv(HIST).sort("fetched_at")
    if h.height == 0:
        return empty
    return h.group_by("home", "away", "commence_time").agg(
        pl.col("home_line").first().alias("open_line"),
        pl.col("home_line").last().alias("book_line"),
    ).select("home", "away", "open_line", "book_line")
