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
TOT_HIST = Path(__file__).resolve().parent.parent / "output" / "totals_history.csv"
BOOK = "draftkings"  # Caesars unavailable via The Odds API; DK tracks it closely
HIST_FIELDS = ["fetched_at", "commence_time", "home", "away", "book", "home_line"]
TOT_FIELDS = ["fetched_at", "commence_time", "home", "away", "book", "total"]

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
        f"?apiKey={key}&regions=us&markets=spreads,totals&bookmakers={BOOK}"
        "&oddsFormat=american"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            events = json.load(r)
    except Exception as e:
        print(f"odds: fetch failed ({e}), using consensus lines")
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, tot_rows = [], []
    for ev in events:
        home = NAME2ABBR.get(ev.get("home_team", ""))
        away = NAME2ABBR.get(ev.get("away_team", ""))
        if not home or not away:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "spreads":
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
                elif mk.get("key") == "totals":
                    pt = next(
                        (o.get("point") for o in mk.get("outcomes", [])
                         if o.get("name") == "Over"),
                        None,
                    )
                    if pt is not None:
                        tot_rows.append({
                            "fetched_at": now, "commence_time": ev["commence_time"],
                            "home": home, "away": away, "book": bk["key"],
                            "total": float(pt),
                        })

    def _append(path: Path, fields: list[str], data_rows: list[dict]) -> None:
        if not data_rows:
            return
        new_file = not path.exists()
        path.parent.mkdir(exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            w.writerows(data_rows)

    _append(HIST, HIST_FIELDS, rows)
    _append(TOT_HIST, TOT_FIELDS, tot_rows)
    print(f"odds: appended {len(rows)} spread + {len(tot_rows)} total snapshots")
    return len(rows) + len(tot_rows)


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


def movement_series() -> dict:
    """Per-game daily line timelines (change points only) for display.

    Returns {"AWAY @ HOME": {"s": [["MM-DD", line], ...], "t": [...]}} using
    the last snapshot of each day; consecutive unchanged days are dropped."""
    out: dict = {}
    for path, key, col in [(HIST, "s", "home_line"), (TOT_HIST, "t", "total")]:
        if not path.exists():
            continue
        h = pl.read_csv(path).sort("fetched_at").with_columns(
            pl.col("fetched_at").str.slice(0, 10).alias("day")
        )
        daily = h.group_by("home", "away", "commence_time", "day",
                           maintain_order=True).agg(pl.col(col).last().alias("v"))
        for (home, away, _ct), grp in daily.group_by(
                ["home", "away", "commence_time"], maintain_order=True):
            g = grp.sort("day")
            series, prev = [], None
            for d, v in zip(g["day"].to_list(), g["v"].to_list()):
                if prev is None or v != prev:
                    series.append([d[5:], round(v, 1)])
                    prev = v
            game = f"{away} @ {home}"
            slot = out.setdefault(game, {})
            # if a matchup repeats, keep the series with the freshest data
            if key not in slot or series[-1][0] >= slot[key][-1][0]:
                slot[key] = series
    return out


def totals_lines_table() -> pl.DataFrame:
    """Opening (first-seen) and current (latest) book total per game."""
    empty = pl.DataFrame(schema={
        "home": pl.String, "away": pl.String,
        "open_total": pl.Float64, "book_total": pl.Float64,
    })
    if not TOT_HIST.exists():
        return empty
    h = pl.read_csv(TOT_HIST).sort("fetched_at")
    if h.height == 0:
        return empty
    return h.group_by("home", "away", "commence_time").agg(
        pl.col("total").first().alias("open_total"),
        pl.col("total").last().alias("book_total"),
    ).select("home", "away", "open_total", "book_total")
