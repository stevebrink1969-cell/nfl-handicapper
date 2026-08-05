# NFL Point-Based Handicapping Tool — Project Spec

*Spec agreed 2026-08-05 via interview with Steve. Season opener ~Sept 10, 2026.*

## Goal
Assign an automatic point value to every player on every NFL roster, adjust team
totals using the official injury report before each game, project the correct
spread for every game, and surface the edge versus the Caesars line.

## Decisions (from interview)

| Area | Decision |
|---|---|
| Valuation | Fully automatic, with full visibility into every player's assigned points |
| Garbage time | Discount low-leverage production (blowout/fluke scoring) using win probability on each play |
| Data | Free public data — nflverse via `nflreadpy` (rosters, depth charts, official injury reports, snap counts, EPA/WP play-by-play) |
| Output | Projected spread per game + edge vs. market |
| Market line | Caesars specifically |
| Line movement | Store every odds snapshot from the scheduled polls: opening line (first seen), full timestamped movement history, and closing line. Show open → current movement per game; measure closing line value (does the market move toward our number?) as a model-validation metric. Free-tier Odds API has no historical archive, so live history accumulates from launch; backtest uses free historical open/close datasets where available. |
| Questionable players | Probability-weighted (~75–80% base play rate, adjusted by position and practice participation) — line adjusts smoothly |
| Validation | Full backtest: build values from prior seasons, generate lines for past games, compare vs. actual closing lines/results to calibrate the point scale |
| Access | Free auto-updating website (GitHub Pages or Vercel), bookmarkable on PC and iPhone; scheduled job re-pulls data several times daily |
| Alerts | None — dashboard sorts slate by edge size; Steve checks the site |
| User role | Fully hands-off technically; Claude builds and maintains everything |
| v1 priority | Both model rigor AND polished dashboard before Week 1 (accepts longer build) |

## Model design

1. **Player value**: EPA-based contribution from play-by-play, weighted by
   leverage (win probability between ~5% and ~95% counts fully; garbage-time
   plays discounted toward zero). Combined with snap share and positional
   value curves (QB >> other positions). Multi-season weighted (recent seasons
   count more). Rookies/new players seeded from draft capital + depth chart slot.
2. **Team rating** = sum of active player values (starters weighted by snap
   share expectations from depth charts).
3. **Projected spread** = rating gap + home-field advantage (calibrated, ~1.5–2 pts)
   + rest/travel adjustments if backtest shows they matter.
4. **Injury adjustment** = (player value − replacement's value) × P(out).
   Out/IR/Doubtful ≈ certain; Questionable probability-weighted by position and
   Wed/Thu/Fri practice participation (DNP/Limited/Full).
5. **Calibration**: backtest 2021–2025, tune the point scale so projected
   spreads minimize error vs. closing lines and actual margins.

## Architecture

- Python 3.12 pipeline (already on Steve's PC; note: use `polars-lts-cpu`, plain
  polars crashes on this CPU).
- Data refresh via GitHub Actions on a schedule (more frequent Fri–Sun).
- Static site (mobile-friendly) published to GitHub Pages, regenerated on each refresh.
- Odds: The Odds API free tier (has Caesars) — needs a free API key.

## Items Steve must do personally (account creation — one-time)
1. Create/confirm a GitHub account (hosting + scheduled refresh).
2. Sign up for a free The Odds API key (the-odds-api.com) for Caesars lines.

## Verified working (2026-08-05)
- `nflreadpy` on Steve's machine: 2025 rosters (3,137 players), injuries (6,068
  rows incl. report_status + practice participation), pbp (48,771 plays with
  `wp`, `epa`), snap counts, depth charts.

## Build phases
1. **Data pipeline + valuation engine** (local): pull multi-season data, compute
   leverage-weighted player values, team ratings.
2. **Backtest + calibration**: 2021–2025 projected lines vs. closing lines;
   tune scale, HFA, injury weights.
3. **Dashboard**: mobile-friendly site — week slate sorted by edge, per-team
   roster with point values, injury adjustments shown per player.
4. **Hosting + automation**: GitHub repo, Pages deploy, scheduled Actions
   refresh, Odds API integration.
