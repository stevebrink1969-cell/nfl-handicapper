"""Paper-trading ledger: every flagged model edge becomes a flat $100 bet.

Recorded the FIRST time an edge is flagged (line and price locked then),
deduped by play, graded automatically once results exist, CLV stamped from
the consensus closing line. Persisted in output/ledger.csv (committed by the
refresh job, so it accumulates in the cloud).

Thresholds mirror what the site badges: spread edges >= 2 pts, total edges
>= 3 pts, props = the top-5-per-slot lists (actual DK prices). SGPs are
excluded — their true price only exists inside DK's app.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import config as C
from . import data
from .propscan import _norm_name

LEDGER = Path(__file__).resolve().parent.parent / "output" / "ledger.csv"
FIELDS = ["key", "first_seen", "season", "week", "type", "game", "desc",
          "team", "side", "player", "nkey", "market_key", "line", "price",
          "model_val", "market_val", "edge", "status", "payout", "clv"]
STAKE = 100.0
SPREAD_FLAG = 2.0
TOTAL_FLAG = 3.0
STAT_COL = {"player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds",
            "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries",
            "player_receptions": "receptions",
            "player_reception_yds": "receiving_yards"}


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    with LEDGER.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("line", "price", "model_val", "market_val", "edge", "payout", "clv"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
        r["season"], r["week"] = int(r["season"]), int(r["week"])
    return rows


def _save(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _dec(a: float) -> float:
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def _settle(row: dict, status: str, clv: float | None) -> None:
    payout = STAKE * (_dec(row["price"]) - 1) if status == "won" \
        else -STAKE if status == "lost" else 0.0
    row.update(status=status, payout=round(payout, 2), clv=clv)


def _grade(rows: list[dict], season: int) -> int:
    sched = data.schedules([season]).filter(pl.col("home_score").is_not_null())
    if sched.height == 0:
        return 0
    games = {
        f'{r["away_team"]} @ {r["home_team"]}': r
        for r in sched.select("away_team", "home_team", "week", "home_score",
                              "away_score", "spread_line", "total_line").to_dicts()
    }
    pstats = {}
    try:
        from . import props as props_mod
        ps = props_mod.load_stats([season])
        for r in ps.to_dicts():
            pstats[(_norm_name(r["player_display_name"]), r["week"])] = r
    except Exception:
        pass

    graded = 0
    for row in rows:
        if row["status"] != "open":
            continue
        gm = games.get(row["game"])
        if gm is None or gm["week"] != row["week"]:
            continue
        margin = gm["home_score"] - gm["away_score"]
        total = gm["home_score"] + gm["away_score"]
        if row["type"] == "spread":
            tm = margin if row["team"] == gm["home_team"] else -margin
            d = tm + row["line"]
            close = -gm["spread_line"] if row["team"] == gm["home_team"] else gm["spread_line"]
            clv = round(row["line"] - close, 1) if gm["spread_line"] is not None else None
            _settle(row, "won" if d > 0 else "lost" if d < 0 else "push", clv)
        elif row["type"] == "total":
            d = total - row["line"] if row["side"] == "Over" else row["line"] - total
            clv = None
            if gm["total_line"] is not None:
                clv = round(gm["total_line"] - row["line"], 1) if row["side"] == "Over" \
                    else round(row["line"] - gm["total_line"], 1)
            _settle(row, "won" if d > 0 else "lost" if d < 0 else "push", clv)
        elif row["type"] == "prop":
            st = pstats.get((row["nkey"], row["week"]))
            if st is None:
                _settle(row, "void", None)
                graded += 1
                continue
            if row["market_key"] == "player_anytime_td":
                v = (st["rushing_tds"] or 0) + (st["receiving_tds"] or 0)
                won = (v > 0) == (row["side"] != "Under")
                _settle(row, "won" if won else "lost", None)
            else:
                v = st.get(STAT_COL[row["market_key"]])
                if v is None:
                    _settle(row, "void", None)
                    graded += 1
                    continue
                d = row["line"] - v if row["side"] == "Under" else v - row["line"]
                _settle(row, "won" if d > 0 else "lost" if d < 0 else "push", None)
        graded += 1
    return graded


def update(games: pl.DataFrame, prop_plays: pl.DataFrame | None,
           season: int, week: int) -> dict:
    """Grade open entries, record newly flagged edges, return site summary."""
    rows = _load()
    graded = _grade(rows, season)
    keys = {r["key"] for r in rows}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def add(entry: dict) -> None:
        if entry["key"] in keys:
            return
        keys.add(entry["key"])
        rows.append({**{f: None for f in FIELDS}, "first_seen": now,
                     "season": season, "week": week, "status": "open",
                     "payout": None, "clv": None, **entry})

    for g in games.to_dicts():
        gname = f'{g["away_team"]} @ {g["home_team"]}'
        if g.get("edge") is not None and abs(g["edge"]) >= SPREAD_FLAG \
                and g.get("market_line") is not None:
            team = g["home_team"] if g["edge"] > 0 else g["away_team"]
            line = -g["market_line"] if g["edge"] > 0 else g["market_line"]
            add({"key": f"{season}-{week}-spread-{gname}",
                 "type": "spread", "game": gname, "team": team, "line": line,
                 "price": -110.0,
                 "desc": f"{team} {'+' if line >= 0 else ''}{line}",
                 "model_val": g.get("model_line"), "market_val": g["market_line"],
                 "edge": abs(g["edge"])})
        if g.get("total_edge") is not None and abs(g["total_edge"]) >= TOTAL_FLAG \
                and g.get("market_total") is not None:
            side = "Over" if g["total_edge"] > 0 else "Under"
            add({"key": f"{season}-{week}-total-{gname}",
                 "type": "total", "game": gname, "side": side,
                 "line": g["market_total"], "price": -110.0,
                 "desc": f"{side} {g['market_total']} {gname}",
                 "model_val": g.get("model_total"), "market_val": g["market_total"],
                 "edge": abs(g["total_edge"])})
    if prop_plays is not None and prop_plays.height:
        for p in prop_plays.to_dicts():
            add({"key": f"{season}-{week}-prop-{p['nkey']}-{p['market_key']}",
                 "type": "prop", "game": p["game"], "side": p["side"],
                 "player": p["player"], "nkey": p["nkey"],
                 "market_key": p["market_key"], "line": p["line"],
                 "price": float(p["price"]),
                 "desc": f"{p['player']} {p['side']} {p['line']} {p['market']}",
                 "model_val": p["proj"], "market_val": p["line"],
                 "edge": p["ev_pct"]})
    _save(rows)
    if graded:
        print(f"ledger: graded {graded} entries")
    print(f"ledger: {len(rows)} total entries "
          f"({sum(1 for r in rows if r['status'] == 'open')} open)")
    return summary(rows)


def summary(rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else _load()
    by_type = {}
    for t in ("spread", "total", "prop"):
        sub = [r for r in rows if r["type"] == t and r["status"] in ("won", "lost", "push")]
        w = sum(1 for r in sub if r["status"] == "won")
        lo = sum(1 for r in sub if r["status"] == "lost")
        pu = sum(1 for r in sub if r["status"] == "push")
        net = sum(r["payout"] or 0 for r in sub)
        staked = STAKE * (w + lo)
        clvs = [r["clv"] for r in sub if r["clv"] is not None]
        by_type[t] = {
            "w": w, "l": lo, "p": pu,
            "net": round(net, 2),
            "roi": round(net / staked * 100, 1) if staked else None,
            "clv": round(sum(clvs) / len(clvs), 2) if clvs else None,
            "open": sum(1 for r in rows if r["type"] == t and r["status"] == "open"),
        }
    recent = sorted(
        [r for r in rows if r["status"] in ("won", "lost", "push")],
        key=lambda r: (r["week"],), reverse=True,
    )[:12]
    return {
        "types": by_type,
        "recent": [
            {"desc": r["desc"], "type": r["type"], "week": r["week"],
             "status": r["status"], "payout": r["payout"], "clv": r["clv"]}
            for r in recent
        ],
    }
