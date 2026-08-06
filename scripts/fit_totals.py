"""Fit production totals coefficients on all 2021-2025 games (walk-forward
features), write output/totals_calibration.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from totals_backtest import build_games  # noqa: E402  (same scripts dir)

FEATS = ["const", "offsum", "defsum", "scoring", "oepasum", "depasum",
         "pacex", "env", "windx", "cold", "dome"]


def main() -> None:
    g = build_games()
    X = np.column_stack([g[f].to_numpy() for f in FEATS])
    y = g["total_line"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    actual = g["actual"].to_numpy().astype(float)
    cal = {
        "features": FEATS,
        "coefs": [float(b) for b in beta],
        "games": int(g.height),
        "mae_vs_close": float(np.abs(pred - y).mean()),
        "mae_vs_actual": float(np.abs(pred - actual).mean()),
        "market_mae_vs_actual": float(np.abs(y - actual).mean()),
        "oos_note": ("OOS 2023-25: MAE vs close 2.18; O/U on 3+pt edges 53.3% "
                     "(210 games) - marginal; treat totals edges cautiously, "
                     "validate via live CLV"),
    }
    out = Path(__file__).resolve().parent.parent / "output" / "totals_calibration.json"
    out.write_text(json.dumps(cal, indent=2))
    print(f"fitted {len(FEATS)} coefs on {g.height} games | "
          f"MAE vs close {cal['mae_vs_close']:.2f}")
    print(dict(zip(FEATS, [round(b, 3) for b in beta])))


if __name__ == "__main__":
    main()
