import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import polars as pl

from src import data

r26 = None
try:
    r26 = data.rosters(2026)
    print("rosters 2026:", r26.shape)
except Exception as e:
    print("rosters 2026 unavailable:", e)

r25 = data.rosters(2025)
print("rosters 2025 cols:", r25.columns)
print("positions:", sorted(r25["position"].drop_nulls().unique().to_list()))
print("status values:", sorted(r25["status"].drop_nulls().unique().to_list())[:15])

snaps = data.snap_counts([2024, 2025])
print("snaps cols:", snaps.columns)

sched = data.schedules([2024])
line_cols = [c for c in sched.columns if "line" in c or "spread" in c or "total" in c or "moneyline" in c]
print("schedule betting cols:", line_cols)

dc = data.depth_charts(2025)
print("depth cols:", dc.columns)
