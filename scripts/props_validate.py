"""Validate the props projection engine on 2023-2025 (walk-forward).

For each stat: MAE vs actuals, compared against a naive baseline (player's
trailing per-game average). Then distribution calibration: fit sd ~ a + b*mean
on 2023-2024 residuals, check 50% / 80% interval coverage on 2025.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import props

SEASONS = [2023, 2024, 2025]

STATS = [
    # (projection col, actual col, usage filter col, min usage)
    ("proj_receptions", "receptions", "proj_targets", 3.0),
    ("proj_rec_yds", "receiving_yards", "proj_targets", 3.0),
    ("proj_rush_yds", "rushing_yards", "proj_carries", 6.0),
    ("proj_carries", "carries", "proj_carries", 6.0),
    ("proj_pass_yds", "passing_yards", "proj_attempts", 15.0),
    ("proj_completions", "completions", "proj_attempts", 15.0),
    ("proj_pass_tds", "passing_tds", "proj_attempts", 15.0),
]


def main() -> None:
    g = props.build_projections(SEASONS)
    print(f"player-games projected: {g.height}\n")
    print(f"{'stat':<18}{'n':>7}{'model MAE':>11}{'naive MAE':>11}{'edge':>8}")
    for proj, actual, use_col, min_use in STATS:
        d = g.filter(
            (pl.col(use_col) >= min_use) & pl.col(actual).is_not_null()
            & (pl.col("gp") + pl.col("p_gp") >= 4)
        )
        p = d[proj].to_numpy()
        a = d[actual].to_numpy().astype(float)
        mae = np.abs(p - a).mean()
        # naive: trailing per-game average of the actual stat itself
        naive = d.with_columns(
            ((pl.col(actual).cum_sum().shift(1).over("player_id", "season"))
             / pl.max_horizontal(pl.col("gp"), 1)).alias("nv")
        )
        nv = naive["nv"].to_numpy()
        ok = ~np.isnan(nv)
        naive_mae = np.abs(nv[ok] - a[ok]).mean()
        print(f"{actual:<18}{d.height:>7}{mae:>11.2f}{naive_mae:>11.2f}"
              f"{(naive_mae - mae) / naive_mae * 100:>7.1f}%")

    # Distribution calibration: sd ~ a + b*mean fit on 2023-24, coverage on 2025
    print("\nCalibration (fit sd on 2023-24, test coverage on 2025):")
    for proj, actual, use_col, min_use in STATS[:5]:
        d = g.filter(
            (pl.col(use_col) >= min_use) & pl.col(actual).is_not_null()
            & (pl.col("gp") + pl.col("p_gp") >= 4)
        )
        tr = d.filter(pl.col("season") < 2025)
        te = d.filter(pl.col("season") == 2025)
        resid = (tr[actual] - tr[proj]).to_numpy().astype(float)
        mean_tr = tr[proj].to_numpy()
        # bucket by projection size, fit linear sd
        A = np.column_stack([np.ones(len(mean_tr)), mean_tr])
        coef, *_ = np.linalg.lstsq(A, np.abs(resid) * 1.2533, rcond=None)  # E|x|->sd
        sd_te = coef[0] + coef[1] * te[proj].to_numpy()
        sd_te = np.maximum(sd_te, 0.5)
        z = (te[actual].to_numpy().astype(float) - te[proj].to_numpy()) / sd_te
        in50 = float(np.mean(np.abs(z) <= 0.674))
        in80 = float(np.mean(np.abs(z) <= 1.282))
        print(f"  {actual:<18} 50% band: {in50 * 100:.1f}% (want 50) | "
              f"80% band: {in80 * 100:.1f}% (want 80) | sd = {coef[0]:.1f} + {coef[1]:.2f}*mean")


if __name__ == "__main__":
    main()
