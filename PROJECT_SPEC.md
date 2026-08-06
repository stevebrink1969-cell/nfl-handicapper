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
| Market line | DraftKings (Caesars chosen originally, but The Odds API no longer carries Caesars NFL spreads; DK has full coverage and tracks Caesars within ~0.5 pt — decided 2026-08-05) |
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

## Model upgrade (2026-08-05): in-season weekly updating adopted
Player values now recompute weekly during the season, including the current
season's games at weight 1.25. Out-of-sample 2023–2025 (each season graded on
prior-year fits only): MAE vs close 3.33 → 3.03; ATS on 3+ pt edges 52.8% →
57.5% (203–150). Tested and REJECTED: rest differential (market prices it,
no gain) and prior-season team anchor (improves line-matching to MAE 2.91 but
erodes betting edge — it copies what the market already knows). Production
calibration refit on weekly walk-forward components: k_qb=0.853, k_perf=1.574,
HFA=1.66.

## Calibration results (Phase 2, run 2026-08-05)
Walk-forward backtest, 1,424 games 2021–2025, each season rated only from
prior-season data. Fitted vs. Caesars-era consensus closing spreads:
- Coefficients: k_qb=0.724, k_perf=1.583, HFA=1.69 (stored in output/calibration.json)
- Model vs closing line: MAE 3.45 pts, correlation 0.686
- Model vs actual margin: MAE 10.12 (market closing line benchmark: 9.76)
- In-sample ATS at edge ≥2 pts: 54.3% (889 games) — optimistic (in-sample, no
  injury info in backtest); treat as upper bound, validate live via CLV.
- Base (snap-share) component carries no market signal (teams field identical
  bodies); excluded from spread fit, kept at fixed 0.5 scale for player
  display + injury-depth math. OL injury impact is understated — future work.

## Model upgrade (2026-08-06): turnover-luck discount + opponent adjustment
Research survey (nfelo WEPA, DVOA, turnover-persistence studies) produced two
adopted upgrades: EPA on interception plays keeps 55% / lost fumbles 45%
(turnover margin persists only ~11% YoY), and all EPA credits are adjusted
for opponent strength. OOS 2023–2025: MAE vs close 3.03 → 2.92, ATS ≥1pt
53.6% → 54.5% (≥2/≥3 within noise of prior best). Recalibrated: k_qb=0.821,
k_perf=1.901, HFA=1.66. Considered and rejected again: market reversion
(erodes independent edge). Future candidates: kicker/ST component, faster
QB rolling values, rookie draft-capital seeding, OL injury calibration.

## Deployed (2026-08-05) — all four phases complete
- Live site: https://stevebrink1969-cell.github.io/nfl-handicapper/
- Repo: https://github.com/stevebrink1969-cell/nfl-handicapper (public)
- Auto-refresh: GitHub Actions, 3x daily Mon–Thu, ~3-hourly Fri–Sun (UTC),
  first cloud run verified green. ODDS_API_KEY stored as repo secret; local
  key in .env (gitignored).
- Odds usage: 1 API request per refresh ≈ 140/month vs 500 free quota.
- Opening lines for all 272 games of 2026 captured 2026-08-05.

## Phase 5-8: Totals, Props, SGPs, Bet Tracker (interviewed 2026-08-06)

| Area | Decision |
|---|---|
| Rollout | Ship each piece when ready; totals first |
| Totals model | Exploit O/U mispricing: offense/defense split ratings + pace + weather (wind matters for totals, unlike spreads); backtest vs 5 seasons of closing totals, same OOS protocol |
| Props scanner | Scan ALL player prop markets per time slot, rank by EV, show top 5 per slot |
| Odds filter | Exclude any prop priced -140 or worse (Steve's discipline rule, 2026-08-06). Config: PROP_MIN_ODDS = -140. Applies to scanner display and SGP legs. |
| Prop odds | The Odds API paid tier (~$30/mo, 20K credits) — Steve upgrades his existing key |
| Parlays | Same-game parlays, 2-3 legs, correlation-aware pricing |
| Bet tracker | On-site log: device-local storage (no sync), dollars, auto-graded from final scores/stats, running P&L + CLV |
| Weather | Open-Meteo (free, no key) forecasts by stadium for totals; historical temp/wind from nflverse schedules for backtest |

## Phases 9-13 (agreed 2026-08-06): all recommended additions adopted
9. Model report card + paper-trading ledger (auto-grade every flagged edge as
   flat $100; record by bet type; CLV on every play) — the trust engine.
10. Full-season edge board (model vs all 272 posted DK lines; look-ahead
    lines are soft).
11. Line movement detail per game (timeline; flag market moving through us).
12. League-wide injury impact panel.
13. Weather (Open-Meteo) into totals + display, once games are in range (Sept).
Explicitly rejected: playoff simulators, social/consensus features, new bet
types (moneylines/teasers) before the ledger validates existing ones.

## Build phases
1. **Data pipeline + valuation engine** (local): pull multi-season data, compute
   leverage-weighted player values, team ratings.
2. **Backtest + calibration**: 2021–2025 projected lines vs. closing lines;
   tune scale, HFA, injury weights.
3. **Dashboard**: mobile-friendly site — week slate sorted by edge, per-team
   roster with point values, injury adjustments shown per player.
4. **Hosting + automation**: GitHub repo, Pages deploy, scheduled Actions
   refresh, Odds API integration.
