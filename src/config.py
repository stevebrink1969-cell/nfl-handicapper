"""All tunable model constants live here so Phase 2 calibration touches one file."""

SEASONS = [2021, 2022, 2023, 2024, 2025]
TARGET_SEASON = 2026  # season we're projecting; falls back to latest with rosters

# Recency: how much each past season counts toward a player's current value
SEASON_WEIGHTS = {2025: 1.0, 2024: 0.60, 2023: 0.35, 2022: 0.20, 2021: 0.10}

# Leverage weighting — discounts garbage-time / low-leverage production
WP_LO, WP_HI = 0.05, 0.95          # win-prob band where plays count fully
GARBAGE_WEIGHT = 0.15              # weight for plays outside the band
BLOWOUT_MARGIN = 17                # 4th-qtr lead where plays are also discounted

# Attribution credit shares (portion of play EPA credited to the individual)
QB_CREDIT = 0.50
RUSH_CREDIT = 0.50
RECV_CREDIT = 0.35
DEF_SACK_CREDIT = 0.55
DEF_INT_CREDIT = 0.45
DEF_PD_CREDIT = 0.30
DEF_TFL_CREDIT = 0.40
DEF_FF_CREDIT = 0.40
DEF_PLAY_EPA_CAP = 4.0             # cap credited EPA per defensive play (pick-6 flukes)

# Shrinkage priors (larger = more regression toward baseline for small samples)
QB_PRIOR_DROPBACKS = 250
SKILL_PRIOR_GAMES = 6
QB_DROPBACKS_PER_GAME = 36
QB_REPLACEMENT_PCTILE = 0.15       # replacement level among qualified QBs
QB_MIN_QUALIFYING_W = 100          # weighted dropbacks to qualify for the pctile calc

# Snap-share base value: what simply manning a position at X% of snaps is worth.
# Captures OL and non-stat defenders; performance components add on top.
POS_BASE = {
    "QB": 2.0, "RB": 0.5, "WR": 0.7, "TE": 0.6, "OL": 0.9,
    "DL": 1.1, "LB": 0.8, "DB": 1.0,
}
ROOKIE_SNAP_DEFAULT = 0.35         # assumed snap share for players with no history

# Map raw roster positions -> position groups (nflverse rosters use coarse
# groups already; secondary modeled as one DB group — matches nickel rotation)
POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "OT": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "EDGE": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "DB": "DB", "FS": "DB", "SS": "DB", "S": "DB", "SAF": "DB",
}

# Expected-lineup rotation weights: per position group, snap-share weights
# applied to a team's players sorted by value (starter first).
ROTATION = {
    "QB": [1.0],
    "RB": [0.65, 0.30, 0.10],
    "WR": [0.85, 0.80, 0.60, 0.30, 0.15],
    "TE": [0.80, 0.40, 0.15],
    "OL": [0.95, 0.95, 0.95, 0.95, 0.95, 0.15, 0.10],
    "DL": [0.80, 0.75, 0.65, 0.55, 0.35, 0.25],
    "LB": [0.90, 0.75, 0.45, 0.20],
    "DB": [0.92, 0.88, 0.82, 0.62, 0.30, 0.15],
}

# Final scaling: team ratings normalized so their std dev matches typical NFL
# team-strength spread in points (calibrated properly in Phase 2).
TEAM_RATING_STD = 6.0
