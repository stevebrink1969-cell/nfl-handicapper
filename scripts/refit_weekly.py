"""Final calibration: fit spread coefficients on all 2021-2025 games using
weekly walk-forward components (in-season updating), matching production."""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C
from src import data

CACHE = data.DATA_DIR / "weekly_comps.parquet"
OUT = Path(__file__).resolve().parent.parent / "output"


def main() -> None:
    comps = pl.read_parquet(CACHE)
    sched = (
        data.schedules(C.BACKTEST_SEASONS)
        .filter(pl.col("spread_line").is_not_null() & pl.col("home_score").is_not_null())
        .select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            "home_team", "away_team", "home_score", "away_score",
            "spread_line", "location",
        )
    )
    h = comps.rename({"team": "home_team", "qb_c": "qb_h", "perf_c": "perf_h", "base_c": "base_h"})
    a = comps.rename({"team": "away_team", "qb_c": "qb_a", "perf_c": "perf_a", "base_c": "base_a"})
    g = (
        sched.join(h, on=["season", "week", "home_team"], how="inner")
        .join(a, on=["season", "week", "away_team"], how="inner")
        .with_columns(
            (pl.col("qb_h") - pl.col("qb_a")).alias("qb_d"),
            (pl.col("perf_h") - pl.col("perf_a")).alias("perf_d"),
            (pl.col("location") == "Home").cast(pl.Float64).alias("home"),
            (pl.col("home_score") - pl.col("away_score")).alias("margin"),
        )
    )
    X = np.column_stack([g["qb_d"].to_numpy(), g["perf_d"].to_numpy(), g["home"].to_numpy()])
    y = g["spread_line"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    margin = g["margin"].to_numpy().astype(float)
    cal = {
        "k_qb": float(beta[0]), "k_perf": float(beta[1]),
        "k_base": C.K_BASE_DISPLAY, "hfa": float(beta[2]),
        "fit_target": "closing_spread", "fit_mode": "weekly_in_season",
        "games": int(g.height),
        "mae_vs_close": float(np.abs(pred - y).mean()),
        "corr_vs_close": float(np.corrcoef(pred, y)[0, 1]),
        "mae_vs_margin": float(np.abs(pred - margin).mean()),
        "market_mae_vs_margin": float(np.abs(y - margin).mean()),
        "oos_note": "OOS 2023-25: MAE 3.03, ATS edge>=2 54.9%, edge>=3 57.5%",
    }
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"k_qb={beta[0]:.3f} k_perf={beta[1]:.3f} HFA={beta[2]:.2f} | "
          f"in-sample MAE {cal['mae_vs_close']:.2f}, corr {cal['corr_vs_close']:.3f} "
          f"({g.height} games)")


if __name__ == "__main__":
    main()
