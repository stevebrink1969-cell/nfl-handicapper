"""Static site generator: one self-contained mobile-first index.html."""

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import data

# docs/ because GitHub Pages serves from the repo's /docs folder
SITE_DIR = Path(__file__).resolve().parent.parent / "docs"


def build_results(season: int) -> dict:
    """Completed-game scores and player stat lines for client-side bet
    grading (empty preseason; grows through the season)."""
    out = {"games": [], "pstats": []}
    try:
        sched = data.schedules([season])
    except Exception:
        return out
    done = sched.filter(pl.col("home_score").is_not_null())
    if done.height:
        out["games"] = done.select(
            (pl.col("away_team") + " @ " + pl.col("home_team")).alias("game"),
            pl.col("week").cast(pl.Int64), "home_team", "away_team",
            "home_score", "away_score", "spread_line", "total_line",
        ).to_dicts()
    try:
        from . import props as props_mod
        from .propscan import _norm_name
        ps = props_mod.load_stats([season])
        if ps.height:
            compact = ps.select(
                pl.col("player_display_name")
                .map_elements(_norm_name, return_dtype=pl.String).alias("nkey"),
                pl.col("week").cast(pl.Int64),
                pl.col("passing_yards").alias("py"),
                pl.col("passing_tds").alias("ptd"),
                pl.col("rushing_yards").alias("ry"),
                pl.col("carries").alias("ca"),
                pl.col("receptions").alias("rec"),
                pl.col("receiving_yards").alias("recy"),
                (pl.col("rushing_tds") + pl.col("receiving_tds")).alias("atd"),
            )
            out["pstats"] = compact.to_dicts()
    except Exception:
        pass
    return out


def assemble(slate: pl.DataFrame, week: int, season: int, tr: pl.DataFrame,
             adjusted: pl.DataFrame, inj_adj: pl.DataFrame, info: dict,
             prop_plays: pl.DataFrame | None = None,
             sgps: list | None = None,
             led: dict | None = None) -> dict:
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
            "market_line", "mkt_src", "open_line", "model_line", "edge",
            "model_total", "market_total", "open_total", "total_edge", "tot_src",
            "adj_h", "adj_a",
        ).to_dicts(),
        "teams": teams_tbl.select("team", "rating", "inj_adj").to_dicts(),
        "rosters": rosters,
        "props": prop_plays.to_dicts() if prop_plays is not None and prop_plays.height else [],
        "sgps": sgps or [],
        "results": build_results(season),
        "ledger": led,
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
#fresh{display:none;position:fixed;top:10px;left:50%;transform:translateX(-50%);
  background:var(--accent);color:#fff;border:none;border-radius:20px;
  padding:8px 16px;font:600 13px inherit;z-index:10;cursor:pointer;
  box-shadow:0 2px 10px rgba(0,0,0,.25)}
.logbtn{background:none;border:1px solid var(--line);border-radius:6px;
  color:var(--accent);font:600 11px inherit;padding:3px 8px;cursor:pointer;margin-left:6px}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:20;
  align-items:flex-end;justify-content:center}
#modal.on{display:flex}
#mbox{background:var(--card);border-radius:16px 16px 0 0;padding:18px 16px
  max(18px, env(safe-area-inset-bottom));width:100%;max-width:520px}
#mbox h3{font-size:16px;margin-bottom:10px}
#mbox label{display:block;font-size:12px;color:var(--ink3);margin:10px 0 4px}
#mbox input,#mbox select{width:100%;padding:10px;border:1px solid var(--line);
  border-radius:8px;background:var(--bg);color:var(--ink);font:15px inherit}
.mrow{display:flex;gap:10px}
.mrow>div{flex:1}
.mbtns{display:flex;gap:10px;margin-top:16px}
.mbtns button{flex:1;padding:12px;border-radius:10px;border:none;
  font:600 14px inherit;cursor:pointer}
.msave{background:var(--accent);color:#fff}
.mcancel{background:var(--flat-bg);color:var(--ink2)}
.bet{border-bottom:1px solid var(--line);padding:10px 2px}
.bet:last-child{border-bottom:none}
.bet .b1{display:flex;justify-content:space-between;gap:8px}
.bet .bdesc{font-weight:600;font-size:14px}
.bet .bmeta{color:var(--ink3);font-size:12px;margin-top:2px}
.st-won{color:var(--good);font-weight:700}
.st-lost{color:var(--neg);font-weight:700}
.st-open{color:var(--ink3);font-weight:600}
.st-push,.st-void,.st-check{color:var(--lean);font-weight:600}
.bsum{display:flex;gap:14px;margin-bottom:4px}
.bsum div{flex:1;text-align:center}
.bsum .val{font-size:17px}
.tiny{background:none;border:none;color:var(--ink3);font-size:11px;cursor:pointer;
  text-decoration:underline;padding:2px 4px}
</style>
</head>
<body>
<header>
  <h1>NFL Line Model</h1>
  <div class="sub" id="sub"></div>
</header>
<button id="fresh">New version — tap to refresh</button>
<main id="view"></main>
<nav>
  <button id="nav-slate" onclick="show('slate')">This Week</button>
  <button id="nav-props" onclick="show('props')">Props</button>
  <button id="nav-bets" onclick="show('bets')">Bets</button>
  <button id="nav-teams" onclick="show('teams')">Teams</button>
  <button id="nav-about" onclick="show('about')">Model</button>
</nav>
<div id="modal"><div id="mbox"></div></div>
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
    const market = fmtLine(g.away_team, g.home_team, g.market_line);
    const mktLbl = g.mkt_src === 'book' ? 'DraftKings' : 'Market';
    let badge = '<span class="badge b-flat">no line</span>';
    if (g.edge != null){
      const a = Math.abs(g.edge), side = g.edge > 0 ? g.home_team : g.away_team;
      const cls = a >= 2 ? 'b-good' : a >= 1 ? 'b-lean' : 'b-flat';
      const txt = a < 1 ? 'fair' : `${side} +${a.toFixed(1)}`;
      badge = `<span class="badge ${cls}">${txt}</span>`;
    }
    let totRow = '';
    if (g.model_total != null){
      let tb = '';
      if (g.total_edge != null && Math.abs(g.total_edge) >= 3){
        tb = `<span class="badge b-lean">${g.total_edge > 0 ? 'Over' : 'Under'} +${Math.abs(g.total_edge).toFixed(1)}</span>`;
      }
      const mktT = g.market_total != null ? g.market_total.toFixed(1) : '—';
      totRow = `<div class="lines">
        <div><div class="lbl">Total (model)</div><div class="val">${g.model_total.toFixed(1)}</div></div>
        <div><div class="lbl">${g.tot_src === 'book' ? 'DK total' : 'Mkt total'}</div><div class="val">${mktT}</div></div>
        ${tb}</div>`;
    }
    const notes = [];
    if (g.open_line != null){
      const moved = g.market_line != null && g.open_line !== g.market_line;
      const delta = moved ? ` · moved ${Math.abs(g.market_line - g.open_line).toFixed(1)}` : '';
      notes.push(`Opened ${fmtLine(g.away_team, g.home_team, g.open_line)}${delta}`);
    }
    if (g.open_total != null && g.market_total != null && g.open_total !== g.market_total)
      notes.push(`Total opened ${g.open_total.toFixed(1)}`);
    if (g.adj_a) notes.push(`${g.away_team} ${g.adj_a.toFixed(1)} inj`);
    if (g.adj_h) notes.push(`${g.home_team} ${g.adj_h.toFixed(1)} inj`);
    el.append($(`<div class="card">
      <div class="gm-top">
        <span class="matchup">${g.away_team} @ ${g.home_team}</span>
        <span class="when">${esc(g.weekday)} ${esc(g.gameday)} ${esc(g.gametime ?? '')}</span>
      </div>
      <div class="lines">
        <div><div class="lbl">Model</div><div class="val">${model}</div></div>
        <div><div class="lbl">${mktLbl}</div><div class="val">${market}</div></div>
        ${badge}
      </div>
      ${totRow}
      ${notes.length ? `<div class="injnote">${notes.join(' · ')}</div>` : ''}
      <div style="margin-top:8px;text-align:right">
        <button class="logbtn" onclick="openLog(LOGPRE[${LOGPRE.push({type:'spread',game:g.away_team+' @ '+g.home_team,home:g.home_team,away:g.away_team,mline:g.market_line}) - 1}])">+ bet spread</button>
        <button class="logbtn" onclick="openLog(LOGPRE[${LOGPRE.push({type:'total',game:g.away_team+' @ '+g.home_team,mtotal:g.market_total}) - 1}])">+ bet total</button>
      </div>
    </div>`));
  }
  return el;
}

function propsView(){
  const el = $('<div></div>');
  if (!D.props.length){
    el.append($(`<div class="empty">No prop plays right now.<br><br>
      DraftKings posts player props a few days before kickoff — the scanner
      runs on every refresh and this tab fills in automatically. Props priced
      worse than -140 are filtered out by design.</div>`));
    return el;
  }
  let slot = null, card = null;
  for (const p of D.props){
    if (p.slot !== slot){
      slot = p.slot;
      el.append($(`<h2>${slot}</h2>`));
      card = $('<div class="card"><table><thead><tr><th>Player</th><th>Prop</th><th class="num">Proj</th><th class="num">EV</th></tr></thead><tbody></tbody></table></div>');
      el.append(card);
    }
    const sideTxt = p.side === 'Yes' ? '' : p.side + ' ';
    const priceTxt = p.price > 0 ? '+' + p.price : p.price;
    card.querySelector('tbody').append($(`<tr>
      <td>${esc(p.player)}<br><span style="color:var(--ink3);font-size:12px">${p.team} · ${esc(p.game)}</span></td>
      <td>${sideTxt}${p.line || ''} ${esc(p.market)}<br><span style="color:var(--ink3);font-size:12px">${priceTxt}</span></td>
      <td class="num">${p.proj}</td>
      <td class="num" style="color:var(--good);font-weight:700">+${p.ev_pct}%<br>
        <button class="logbtn" style="margin:4px 0 0" onclick="openLog(LOGPRE[${LOGPRE.push({type:'prop',game:p.game,player:p.player,nkey:p.nkey,market:p.market,market_key:p.market_key,side:p.side,line:p.line,price:p.price}) - 1}])">+ bet</button></td>
    </tr>`));
  }
  el.append($(`<div class="empty" style="padding:16px 8px;font-size:12px">Top ${5} per slot by expected value · win probabilities from the projection model · rookies and small samples excluded</div>`));
  if (D.sgps.length){
    el.append($('<h2>Same-game parlays</h2>'));
    for (const s of D.sgps){
      const legs = s.legs.map(l => {
        const sideTxt = l.side === 'Yes' ? '' : l.side + ' ';
        const priceTxt = l.price > 0 ? '+' + l.price : l.price;
        return `<div style="padding:3px 0">${esc(l.player)} — ${sideTxt}${l.line || ''} ${esc(l.market)} <span style="color:var(--ink3)">(${priceTxt})</span></div>`;
      }).join('');
      const fair = s.fair_odds > 0 ? '+' + s.fair_odds : s.fair_odds;
      const minq = s.min_quote > 0 ? '+' + s.min_quote : s.min_quote;
      el.append($(`<div class="card">
        <div class="gm-top"><span class="matchup" style="font-size:15px">${esc(s.game)}</span>
        <span class="when">${esc(s.slot)}</span></div>
        <div style="margin-top:8px">${legs}</div>
        <div class="injnote">Win ${(s.p_joint * 100).toFixed(1)}% · fair ${fair} ·
        <b>worth it if DK quotes ${minq} or better</b> · corr ${s.avg_r >= 0 ? '+' : ''}${s.avg_r}</div>
        <div style="margin-top:6px;text-align:right">
          <button class="logbtn" onclick="openLog(LOGPRE[${LOGPRE.push({type:'sgp',game:s.game,legs:s.legs,min_quote:s.min_quote}) - 1}])">+ bet this parlay</button>
        </div>
      </div>`));
    }
    el.append($('<div class="empty" style="padding:12px 8px;font-size:12px">Correlations measured from 3 seasons of same-game results. Check DK\'s quoted SGP price against the "worth it" number — below it, pass.</div>'));
  }
  return el;
}

// ===== Bet tracker (device-local storage, dollars) =====
const LOGPRE = [];
const BKEY = 'nfl_bets_v1';
const loadBets = () => { try { return JSON.parse(localStorage.getItem(BKEY) || '[]'); } catch { return []; } };
const saveBets = (b) => localStorage.setItem(BKEY, JSON.stringify(b));
const decOdds = (a) => a > 0 ? 1 + a / 100 : 1 + 100 / (-a);
const FMAP = {player_pass_yds:'py', player_pass_tds:'ptd', player_rush_yds:'ry',
  player_rush_attempts:'ca', player_receptions:'rec', player_reception_yds:'recy',
  player_anytime_td:'atd'};

function openLog(pre){
  const box = document.getElementById('mbox');
  let fields = '';
  if (pre.type === 'spread'){
    const hl = pre.mline != null ? -pre.mline : 0, al = pre.mline != null ? pre.mline : 0;
    fields = `<label>Side</label>
      <select id="f-side" onchange="document.getElementById('f-line').value = this.value === 'home' ? ${hl} : ${al}">
      <option value="away">${pre.away} ${al >= 0 ? '+' + al : al}</option>
      <option value="home">${pre.home} ${hl >= 0 ? '+' + hl : hl}</option></select>
      <label>Line taken (points for your team)</label><input id="f-line" type="number" step="0.5" value="${al}">
      `;
  } else if (pre.type === 'total'){
    fields = `<label>Side</label><select id="f-side"><option>Over</option><option>Under</option></select>
      <label>Total line</label><input id="f-line" type="number" step="0.5" value="${pre.mtotal ?? ''}">`;
  } else if (pre.type === 'prop'){
    fields = `<div class="bmeta" style="margin-top:6px">${esc(pre.player)} — ${pre.side === 'Yes' ? '' : pre.side + ' '}${pre.line || ''} ${esc(pre.market)}</div>`;
  } else if (pre.type === 'sgp'){
    fields = `<div class="bmeta" style="margin-top:6px">${pre.legs.map(l => esc(l.player) + ' ' + (l.side === 'Yes' ? '' : l.side + ' ') + (l.line || '') + ' ' + esc(l.market)).join('<br>')}</div>
      <div class="bmeta">Enter DK's quoted parlay price (worth it at ${pre.min_quote > 0 ? '+' + pre.min_quote : pre.min_quote}+)</div>`;
  } else {
    fields = `<label>Description</label><input id="f-desc" type="text" placeholder="e.g. BUF ML week 3">`;
  }
  box.innerHTML = `<h3>Log bet — ${esc(pre.game || 'manual')}</h3>${fields}
    <div class="mrow"><div><label>Odds (American)</label>
      <input id="f-price" type="number" step="5" value="${pre.price ?? -110}"></div>
    <div><label>Stake ($)</label><input id="f-stake" type="number" step="1" inputmode="decimal"></div></div>
    <div class="mbtns"><button class="mcancel" onclick="closeLog()">Cancel</button>
    <button class="msave" onclick="saveLog()">Save bet</button></div>`;
  document.getElementById('modal').classList.add('on');
  window._pre = pre;
}
function closeLog(){ document.getElementById('modal').classList.remove('on'); }
function saveLog(){
  const pre = window._pre;
  const stake = parseFloat(document.getElementById('f-stake').value);
  const price = parseInt(document.getElementById('f-price').value);
  if (!stake || !price){ alert('Need stake and odds'); return; }
  const b = {id: Date.now(), placed: D.meta.updated, type: pre.type,
             game: pre.game || '', price, stake, status: 'open'};
  if (pre.type === 'spread'){
    const sideEl = document.getElementById('f-side');
    b.team = sideEl.value === 'home' ? pre.home : pre.away;
    b.line = parseFloat(document.getElementById('f-line').value);
    b.desc = `${b.team} ${b.line >= 0 ? '+' + b.line : b.line}`;
  } else if (pre.type === 'total'){
    b.side = document.getElementById('f-side').value;
    b.line = parseFloat(document.getElementById('f-line').value);
    b.desc = `${b.side} ${b.line} — ${pre.game}`;
  } else if (pre.type === 'prop'){
    Object.assign(b, {player: pre.player, nkey: pre.nkey, market_key: pre.market_key,
                      side: pre.side, line: pre.line});
    b.desc = `${pre.player} ${pre.side === 'Yes' ? '' : pre.side + ' '}${pre.line || ''} ${pre.market}`;
  } else if (pre.type === 'sgp'){
    b.legs = pre.legs.map(l => ({nkey: _nk(l.player), market_key: _mk(l.market),
                                 side: l.side, line: l.line, player: l.player}));
    b.desc = `SGP: ${pre.legs.map(l => l.player.split(' ').slice(-1)[0]).join(' + ')}`;
  } else {
    b.desc = document.getElementById('f-desc').value || 'manual bet';
  }
  const bets = loadBets(); bets.push(b); saveBets(bets); closeLog();
  show('bets');
}
const _nk = (s) => (s || '').toLowerCase().replace(/[.'’]/g, '').replace(/\s+(jr|sr|ii|iii|iv|v)$/, '');
const MK_REV = {'Pass yds':'player_pass_yds','Pass TDs':'player_pass_tds','Rush yds':'player_rush_yds',
  'Carries':'player_rush_attempts','Receptions':'player_receptions','Rec yds':'player_reception_yds',
  'Anytime TD':'player_anytime_td'};
const _mk = (label) => MK_REV[label] || label;

function _gradeLeg(l, gm){
  const rows = (D.results.pstats || []).filter(x => x.nkey === l.nkey && x.week === gm.week);
  if (!rows.length) return null;
  const v = rows[0][FMAP[l.market_key]];
  if (v == null) return null;
  if (l.market_key === 'player_anytime_td')
    return v > 0 === (l.side !== 'Under') ? 'won' : 'lost';
  const d = l.side === 'Under' ? l.line - v : v - l.line;
  return d > 0 ? 'won' : d < 0 ? 'lost' : 'push';
}
function gradeBet(b){
  const gm = (D.results.games || []).find(x => x.game === b.game);
  if (!gm) return null;
  const margin = gm.home_score - gm.away_score, total = gm.home_score + gm.away_score;
  if (b.type === 'spread'){
    const tm = b.team === gm.home_team ? margin : -margin;
    const d = tm + b.line;
    const closeLine = b.team === gm.home_team ? -gm.spread_line : gm.spread_line;
    const clv = gm.spread_line != null ? +(b.line - closeLine).toFixed(1) : null;
    return _settle(b, d > 0 ? 'won' : d < 0 ? 'lost' : 'push', clv);
  }
  if (b.type === 'total'){
    const d = b.side === 'Over' ? total - b.line : b.line - total;
    const clv = gm.total_line != null
      ? +((b.side === 'Over' ? gm.total_line - b.line : b.line - gm.total_line)).toFixed(1) : null;
    return _settle(b, d > 0 ? 'won' : d < 0 ? 'lost' : 'push', clv);
  }
  if (b.type === 'prop'){
    const st = _gradeLeg(b, gm);
    return st ? _settle(b, st, null) : null;
  }
  if (b.type === 'sgp'){
    const sts = b.legs.map(l => _gradeLeg(l, gm));
    if (sts.includes(null)) return null;
    if (sts.includes('lost')) return _settle(b, 'lost', null);
    if (sts.includes('push')) return _settle(b, 'check', null);
    return _settle(b, 'won', null);
  }
  return null;
}
function _settle(b, status, clv){
  const payout = status === 'won' ? b.stake * (decOdds(b.price) - 1)
    : status === 'lost' ? -b.stake : 0;
  return {status, payout: +payout.toFixed(2), clv};
}
function gradeAll(){
  const bets = loadBets();
  let changed = false;
  for (const b of bets){
    if (b.status !== 'open') continue;
    const res = gradeBet(b);
    if (res){ Object.assign(b, res); changed = true; }
  }
  if (changed) saveBets(bets);
  return bets;
}
function manualGrade(id, status){
  const bets = loadBets();
  const b = bets.find(x => x.id === id);
  if (b){ Object.assign(b, _settle(b, status, null)); saveBets(bets); show('bets'); }
}
function delBet(id){
  if (!confirm('Delete this bet?')) return;
  saveBets(loadBets().filter(x => x.id !== id));
  show('bets');
}
function exportBets(){
  const blob = new Blob([JSON.stringify(loadBets(), null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nfl-bets-backup.json';
  a.click();
}
function importBets(ev){
  const f = ev.target.files[0];
  if (!f) return;
  f.text().then(t => {
    const incoming = JSON.parse(t);
    const cur = loadBets();
    const ids = new Set(cur.map(b => b.id));
    for (const b of incoming) if (!ids.has(b.id)) cur.push(b);
    saveBets(cur);
    show('bets');
  }).catch(() => alert('Could not read that file'));
}

function betsView(){
  const bets = gradeAll().slice().sort((a, b) => (a.status === 'open' ? 0 : 1) - (b.status === 'open' ? 0 : 1) || b.id - a.id);
  const el = $('<div></div>');
  const settled = bets.filter(b => ['won', 'lost', 'push'].includes(b.status));
  const staked = settled.reduce((s, b) => s + b.stake, 0);
  const net = settled.reduce((s, b) => s + (b.payout || 0), 0);
  const w = settled.filter(b => b.status === 'won').length;
  const l = settled.filter(b => b.status === 'lost').length;
  const p = settled.filter(b => b.status === 'push').length;
  el.append($(`<div class="card"><div class="bsum">
    <div><div class="lbl">Record</div><div class="val">${w}-${l}${p ? '-' + p : ''}</div></div>
    <div><div class="lbl">Net</div><div class="val" style="color:${net >= 0 ? 'var(--good)' : 'var(--neg)'}">${net >= 0 ? '+' : ''}$${net.toFixed(2)}</div></div>
    <div><div class="lbl">ROI</div><div class="val">${staked ? (net / staked * 100).toFixed(1) + '%' : '—'}</div></div>
  </div>
  <div style="text-align:center;margin-top:8px">
    <button class="logbtn" onclick="openLog({type:'other'})">+ log a bet manually</button>
    <button class="tiny" onclick="exportBets()">export backup</button>
    <label class="tiny" style="cursor:pointer">import<input type="file" accept=".json" style="display:none" onchange="importBets(event)"></label>
  </div></div>`));
  if (!bets.length){
    el.append($('<div class="empty">No bets logged yet on this device.<br><br>Use the "+ bet" buttons on games, props, and parlays — or log one manually above. Bets grade themselves as results come in. Data stays on this device; export a backup now and then.</div>'));
    return el;
  }
  const card = $('<div class="card"></div>');
  for (const b of bets){
    const stCls = 'st-' + b.status;
    const pay = b.status === 'won' ? `+$${b.payout.toFixed(2)}` : b.status === 'lost' ? `-$${b.stake.toFixed(2)}` : '';
    const clv = b.clv != null ? ` · CLV ${b.clv >= 0 ? '+' : ''}${b.clv}` : '';
    const manual = (b.status === 'open' && b.type === 'other') || b.status === 'check'
      ? `<button class="tiny" onclick="manualGrade(${b.id},'won')">won</button>
         <button class="tiny" onclick="manualGrade(${b.id},'lost')">lost</button>
         <button class="tiny" onclick="manualGrade(${b.id},'push')">push</button>` : '';
    card.append($(`<div class="bet">
      <div class="b1"><span class="bdesc">${esc(b.desc)}</span>
      <span class="${stCls}">${b.status.toUpperCase()}${pay ? ' ' + pay : ''}</span></div>
      <div class="bmeta">${esc(b.game)} · $${b.stake} @ ${b.price > 0 ? '+' + b.price : b.price}${clv}
      ${manual} <button class="tiny" onclick="delBet(${b.id})">delete</button></div>
    </div>`));
  }
  el.append(card);
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
  const wrap = $('<div></div>');
  if (D.ledger){
    const L = D.ledger.types;
    const row = (label, t) => {
      const anyGraded = t.w + t.l + t.p > 0;
      return `<tr><td>${label}</td>
        <td class="num">${anyGraded ? `${t.w}-${t.l}${t.p ? '-' + t.p : ''}` : '—'}</td>
        <td class="num" style="color:${(t.net ?? 0) >= 0 ? 'var(--good)' : 'var(--neg)'}">${anyGraded ? (t.net >= 0 ? '+' : '') + '$' + t.net.toFixed(0) : '—'}</td>
        <td class="num">${t.roi != null ? t.roi + '%' : '—'}</td>
        <td class="num">${t.clv != null ? (t.clv >= 0 ? '+' : '') + t.clv : '—'}</td>
        <td class="num" style="color:var(--ink3)">${t.open}</td></tr>`;
    };
    wrap.append($(`<h2>Live record — every flagged edge, flat $100 paper bets</h2>`));
    wrap.append($(`<div class="card"><table>
      <thead><tr><th>Type</th><th class="num">W-L</th><th class="num">Net</th><th class="num">ROI</th><th class="num">CLV</th><th class="num">Open</th></tr></thead>
      <tbody>${row('Spreads (3+ pts)', L.spread)}${row('Totals (3+ pts)', L.total)}${row('Props (top 5s)', L.prop)}</tbody>
      </table>
      <div class="injnote" style="margin-top:10px">Recorded automatically the first time the model flags each edge — whether or not anyone bets it. This is the model's public track record; judge it here before trusting it with real money.</div>
      </div>`));
    if (D.ledger.recent.length){
      const rows = D.ledger.recent.map(r => `<tr>
        <td>${esc(r.desc)}<br><span style="color:var(--ink3);font-size:11px">wk ${r.week} · ${r.type}</span></td>
        <td class="num st-${r.status}">${r.status.toUpperCase()}</td>
        <td class="num">${r.payout > 0 ? '+' : ''}${r.payout ? '$' + r.payout.toFixed(0) : ''}</td>
        <td class="num">${r.clv != null ? (r.clv >= 0 ? '+' : '') + r.clv : ''}</td></tr>`).join('');
      wrap.append($(`<div class="card"><table>
        <thead><tr><th>Recent plays</th><th class="num">Result</th><th class="num">P/L</th><th class="num">CLV</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`));
    }
  }
  wrap.append($(`<div class="card about">
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
    <p><b>Totals.</b> Each card also shows the model's fair total vs the book's
    over/under. Honest caveat: in backtesting, the totals model predicts the
    closing total very accurately (within ~2.2 pts) but its disagreements only
    broke even against results — the totals market is sharper than the spread
    market. Over/Under badges appear only on 3+ pt gaps and deserve extra
    skepticism until live tracking proves otherwise. Weather forecasts will be
    added in-season (wind matters for totals).</p>
  </div>`));
  return wrap;
}

function show(which, arg){
  const v = document.getElementById('view');
  v.replaceChildren();
  for (const n of ['slate','props','bets','teams','about'])
    document.getElementById('nav-' + n).classList.toggle('on', n === which || (which === 'roster' && n === 'teams'));
  if (which === 'slate') v.append(slateView());
  else if (which === 'props') v.append(propsView());
  else if (which === 'bets') v.append(betsView());
  else if (which === 'teams') v.append(teamsView());
  else if (which === 'roster') v.append(rosterView(arg));
  else v.append(aboutView());
  window.scrollTo(0, 0);
}
show('slate');

// Stale-cache buster: fetch the live page bypassing cache; if its build stamp
// differs from ours, offer a one-tap refresh that forces the new version.
fetch(location.pathname + '?cb=' + Date.now(), {cache: 'no-store'})
  .then(r => r.text())
  .then(t => {
    const m = t.match(/"updated":"([^"]+)"/);
    if (m && m[1] !== D.meta.updated){
      const b = document.getElementById('fresh');
      b.style.display = 'block';
      b.onclick = () => location.href = location.pathname + '?v=' + Date.now();
    }
  }).catch(() => {});
</script>
</body>
</html>
"""
