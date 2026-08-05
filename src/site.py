"""Static site generator: one self-contained mobile-first index.html."""

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

SITE_DIR = Path(__file__).resolve().parent.parent / "site"


def assemble(slate: pl.DataFrame, week: int, season: int, tr: pl.DataFrame,
             adjusted: pl.DataFrame, inj_adj: pl.DataFrame, info: dict) -> dict:
    teams_tbl = tr.join(inj_adj, on="team", how="left").with_columns(
        pl.col("inj_adj").fill_null(0.0)
    )
    roster_cols = ["team", "full_name", "position", "grp", "points", "rot_w",
                   "report_status", "p_play", "impact"]
    rosters = {}
    for t in teams_tbl["team"].to_list():
        rows = (
            adjusted.filter(pl.col("team") == t)
            .sort("points", descending=True)
            .head(45)
            .select(roster_cols)
            .to_dicts()
        )
        rosters[t] = [
            {
                "n": r["full_name"], "p": r["position"], "pts": r["points"],
                "st": r["rot_w"] > 0,
                "inj": r["report_status"], "pp": r["p_play"], "imp": r["impact"],
            }
            for r in rows
        ]
    return {
        "meta": {
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "season": season, "week": week,
            "hfa": round(info.get("hfa", 0), 2),
            "mae": round(info.get("mae_vs_close", 0), 2),
            "corr": round(info.get("corr_vs_close", 0), 3),
            "oos": info.get("oos_note", ""),
        },
        "slate": slate.select(
            "gameday", "weekday", "gametime", "away_team", "home_team",
            "spread_line", "model_line", "edge", "adj_h", "adj_a",
        ).to_dicts(),
        "teams": teams_tbl.select("team", "rating", "inj_adj").to_dicts(),
        "rosters": rosters,
    }


def render(payload: dict) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA__", data_json)


def write(payload: dict) -> Path:
    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "index.html"
    out.write_text(render(payload), encoding="utf-8")
    return out


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NFL Line Model</title>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#16202b; --ink2:#5b6673; --ink3:#8a94a1;
  --line:#e3e7ec; --good:#0f7b3d; --good-bg:#e7f5ec; --lean:#8a6100;
  --lean-bg:#fdf3d9; --flat:#5b6673; --flat-bg:#eef0f3; --neg:#a33333;
  --accent:#1f5eaa;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#10151b; --card:#1a212a; --ink:#e8edf2; --ink2:#a7b1bd; --ink3:#78828e;
    --line:#2a333e; --good:#4ade80; --good-bg:#12301e; --lean:#e8c35a;
    --lean-bg:#332a10; --flat:#a7b1bd; --flat-bg:#232b35; --neg:#f08c8c;
    --accent:#7ab3f0;
  }
}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);
  font:16px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased;padding-bottom:56px}
header{padding:18px 16px 10px;max-width:720px;margin:0 auto}
h1{font-size:20px;letter-spacing:.2px}
.sub{color:var(--ink2);font-size:13px;margin-top:2px}
nav{position:fixed;bottom:0;left:0;right:0;display:flex;background:var(--card);
  border-top:1px solid var(--line);z-index:5}
nav button{flex:1;padding:12px 0 max(12px, env(safe-area-inset-bottom));background:none;border:none;
  color:var(--ink3);font:600 13px inherit;cursor:pointer}
nav button.on{color:var(--accent)}
main{max-width:720px;margin:0 auto;padding:8px 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:14px;margin-bottom:10px}
.gm-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.matchup{font-weight:700;font-size:17px}
.when{color:var(--ink3);font-size:12px;white-space:nowrap}
.lines{display:flex;gap:18px;margin-top:10px}
.lines div{flex:0 0 auto}
.lbl{font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.5px}
.val{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
.badge{margin-left:auto;align-self:center;border-radius:8px;padding:6px 10px;
  font-size:13px;font-weight:700;white-space:nowrap}
.b-good{background:var(--good-bg);color:var(--good)}
.b-lean{background:var(--lean-bg);color:var(--lean)}
.b-flat{background:var(--flat-bg);color:var(--flat)}
.injnote{margin-top:8px;font-size:12px;color:var(--ink2)}
.trow{display:flex;align-items:center;gap:10px;padding:10px 4px;
  border-bottom:1px solid var(--line);cursor:pointer}
.trow:last-child{border-bottom:none}
.tabbr{font-weight:800;width:44px}
.tbar-wrap{flex:1;height:8px;border-radius:4px;background:var(--flat-bg);
  position:relative;overflow:hidden}
.tbar{position:absolute;top:0;bottom:0;border-radius:4px}
.trat{width:52px;text-align:right;font-weight:700;font-variant-numeric:tabular-nums}
.tinj{width:56px;text-align:right;font-size:12px;color:var(--neg)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.5px;
  text-align:left;padding:6px 4px;border-bottom:1px solid var(--line)}
td{padding:7px 4px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:6px;height:6px;border-radius:3px;
  background:var(--accent);margin-right:6px;vertical-align:2px}
.istat{font-size:11px;font-weight:700;border-radius:6px;padding:2px 6px}
.i-out{background:var(--lean-bg);color:var(--neg)}
.i-q{background:var(--lean-bg);color:var(--lean)}
.back{background:none;border:none;color:var(--accent);font:600 14px inherit;
  padding:6px 0;cursor:pointer}
.about p{margin:10px 0;color:var(--ink2);font-size:14px}
.about b{color:var(--ink)}
.empty{color:var(--ink3);text-align:center;padding:32px 0;font-size:14px}
h2{font-size:15px;margin:14px 4px 8px;color:var(--ink2)}
</style>
</head>
<body>
<header>
  <h1>NFL Line Model</h1>
  <div class="sub" id="sub"></div>
</header>
<main id="view"></main>
<nav>
  <button id="nav-slate" onclick="show('slate')">This Week</button>
  <button id="nav-teams" onclick="show('teams')">Teams</button>
  <button id="nav-about" onclick="show('about')">Model</button>
</nav>
<script>
const D = __DATA__;
const $ = (h) => { const d = document.createElement('div'); d.innerHTML = h; return d.firstElementChild; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmtLine = (away, home, m) => m == null ? '—'
  : Math.abs(m) < 0.05 ? 'PK'
  : m > 0 ? `${home} -${(+m).toFixed(1)}` : `${away} -${(-m).toFixed(1)}`;

document.getElementById('sub').textContent =
  `${D.meta.season} · Week ${D.meta.week} · updated ${D.meta.updated}`;

function slateView(){
  const el = $('<div></div>');
  if (!D.slate.length){ el.append($('<div class="empty">No games found.</div>')); return el; }
  for (const g of D.slate){
    const model = fmtLine(g.away_team, g.home_team, g.model_line);
    const market = fmtLine(g.away_team, g.home_team, g.spread_line);
    let badge = '<span class="badge b-flat">no line</span>';
    if (g.edge != null){
      const a = Math.abs(g.edge), side = g.edge > 0 ? g.home_team : g.away_team;
      const cls = a >= 2 ? 'b-good' : a >= 1 ? 'b-lean' : 'b-flat';
      const txt = a < 1 ? 'fair' : `${side} +${a.toFixed(1)}`;
      badge = `<span class="badge ${cls}">${txt}</span>`;
    }
    const inj = [];
    if (g.adj_a) inj.push(`${g.away_team} ${g.adj_a.toFixed(1)} inj`);
    if (g.adj_h) inj.push(`${g.home_team} ${g.adj_h.toFixed(1)} inj`);
    el.append($(`<div class="card">
      <div class="gm-top">
        <span class="matchup">${g.away_team} @ ${g.home_team}</span>
        <span class="when">${esc(g.weekday)} ${esc(g.gameday)} ${esc(g.gametime ?? '')}</span>
      </div>
      <div class="lines">
        <div><div class="lbl">Model</div><div class="val">${model}</div></div>
        <div><div class="lbl">Market</div><div class="val">${market}</div></div>
        ${badge}
      </div>
      ${inj.length ? `<div class="injnote">Injury adj: ${inj.join(' · ')}</div>` : ''}
    </div>`));
  }
  return el;
}

function teamsView(){
  const el = $('<div class="card"></div>');
  const max = Math.max(...D.teams.map(t => Math.abs(t.rating)), 1);
  for (const t of D.teams){
    const w = Math.abs(t.rating) / max * 50;
    const bar = t.rating >= 0
      ? `<div class="tbar" style="left:50%;width:${w}%;background:var(--good)"></div>`
      : `<div class="tbar" style="right:50%;width:${w}%;background:var(--neg)"></div>`;
    const row = $(`<div class="trow">
      <span class="tabbr">${t.team}</span>
      <div class="tbar-wrap">${bar}</div>
      <span class="trat">${t.rating > 0 ? '+' : ''}${t.rating.toFixed(1)}</span>
      <span class="tinj">${t.inj_adj ? t.inj_adj.toFixed(1) + ' inj' : ''}</span>
    </div>`);
    row.onclick = () => show('roster', t);
    el.append(row);
  }
  return el;
}

function rosterView(t){
  const el = $('<div></div>');
  const b = $('<button class="back">← All teams</button>');
  b.onclick = () => show('teams');
  el.append(b);
  el.append($(`<h2>${t.team} · rating ${t.rating > 0 ? '+' : ''}${t.rating.toFixed(1)}${t.inj_adj ? ` · injuries ${t.inj_adj.toFixed(1)}` : ''}</h2>`));
  const rows = (D.rosters[t.team] || []).map(p => {
    let inj = '';
    if (p.inj === 'Out' || p.inj === 'Doubtful') inj = `<span class="istat i-out">${p.inj.toUpperCase()}</span>`;
    else if (p.inj === 'Questionable') inj = `<span class="istat i-q">Q ${Math.round(p.pp * 100)}%</span>`;
    const imp = p.imp ? `<td class="num" style="color:var(--neg)">-${p.imp.toFixed(1)}</td>` : '<td class="num"></td>';
    return `<tr><td>${p.st ? '<span class="dot"></span>' : ''}${esc(p.n)}</td>
      <td>${p.p}</td><td class="num">${p.pts.toFixed(1)}</td>${imp}<td>${inj}</td></tr>`;
  }).join('');
  el.append($(`<div class="card"><table>
    <thead><tr><th>Player</th><th>Pos</th><th class="num">Pts</th><th class="num">Inj impact</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table></div>`));
  return el;
}

function aboutView(){
  return $(`<div class="card about">
    <p><b>How the line is made.</b> Every player carries a point value learned from
    five seasons of play-by-play data: garbage-time production is discounted,
    recent seasons count more, and values update weekly during the season.
    A team's rating sums its expected lineup; the projected spread is the rating
    gap plus ${D.meta.hfa} pts of home field, minus injury adjustments.</p>
    <p><b>Injuries.</b> Out/Doubtful players are replaced by the next man up.
    Questionable players count at their probability of playing (50–90%
    depending on practice participation).</p>
    <p><b>Track record.</b> Calibrated on 1,424 games (2021–2025):
    average miss vs the closing line ${D.meta.mae} pts, correlation ${D.meta.corr}.
    ${esc(D.meta.oos)}.</p>
    <p><b>Reading the badge.</b> "BUF +2.5" means the model's number is 2.5 pts
    better for BUF than the market line — the bigger the gap, the stronger
    the disagreement. Under 1 pt is priced fairly. This is a research tool, not
    betting advice; lines move and the market is sharp.</p>
  </div>`);
}

function show(which, arg){
  const v = document.getElementById('view');
  v.replaceChildren();
  for (const n of ['slate','teams','about'])
    document.getElementById('nav-' + n).classList.toggle('on', n === which || (which === 'roster' && n === 'teams'));
  if (which === 'slate') v.append(slateView());
  else if (which === 'teams') v.append(teamsView());
  else if (which === 'roster') v.append(rosterView(arg));
  else v.append(aboutView());
  window.scrollTo(0, 0);
}
show('slate');
</script>
</body>
</html>
"""
