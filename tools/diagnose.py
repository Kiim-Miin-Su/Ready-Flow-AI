"""Evidence-gathering diagnosis of the flood dataset/model (read-only, no side effects).
Quantifies the root causes documented in DIAGNOSIS.md:
  §1 per-구 vs per-동 rain   §2 feature variance / terrain NaNs
  §3 univariate signal vs y  §4 calibration / probability ceiling  §5 extreme-rain coverage
Run:  python tools/diagnose.py        (needs requirements-dev: pandas, numpy, pyarrow)
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(ROOT, "build", "samples.parquet"))
df["year"] = df["date"].dt.year
print(f"rows={len(df)} dongs={df.adm_cd.nunique()} gu={df.gu_code.nunique()} "
      f"pos={int(df.y.sum())} ({df.y.mean()*100:.2f}%)")
print("positives by year:", df[df.y == 1].year.value_counts().sort_index().to_dict())

CONT = ["rain_day","rain_1d","rain_3d","rain_7d","rain_14d","rain_30d","rain_ante7",
        "rain_max10","rain_max60","hist_flood_cnt_prior","hist_flood_rate_prior",
        "days_since_last_flood","prev_year_same_month_cnt","prev_year_same_week_cnt",
        "neighbor_gu_flood_prior"]

print("\n=== §1 Is rain per-구 (identical across dongs in same 구/date)? ===")
g = df.groupby(["gu_code","date"])["rain_day"].nunique()
print(f"   (gu,date) groups with >1 distinct rain_day: {int((g>1).sum())} / {len(g)}  (0 = per-구 rain)")
dpg = df.groupby("gu_code")["adm_cd"].nunique()
print(f"   dongs per 구: min={dpg.min()} max={dpg.max()} mean={dpg.mean():.1f}")

print("\n=== §2 Feature variance / inert check ===")
for c in CONT:
    s = df[c]
    print(f"   {c:26s} std={s.std():9.4f} nz%={100*(s!=0).mean():5.1f} uniq={s.nunique()}")
st = pd.read_parquet(os.path.join(ROOT, "build", "dong_static.parquet"))
print("   terrain scaffold populated?:",
      {c: int(st[c].notna().sum()) for c in ["elev_mean_m","dist_to_river_m","drainage_density"] if c in st},
      f"/ {len(st)}")

print("\n=== §3 Univariate signal vs target (corr & top-decile lift) ===")
for c in CONT:
    x = df[c].fillna(df[c].median()).values; y = df.y.values
    cr = np.corrcoef(x, y)[0,1] if np.std(x) > 0 else float("nan")
    thr = np.quantile(x, 0.9); top = y[x >= thr]
    lift = (top.mean()/y.mean()) if len(top) and y.mean() else float("nan")
    print(f"   {c:26s} corr={cr:+.3f}  top10%_lift={lift:5.2f}x")

print("\n=== §4 Probability ceiling / calibration semantics ===")
sys.path.insert(0, os.path.join(ROOT, "api"))
import flood_model as fm
m = fm.load(os.path.join(ROOT, "build", "model_np.json"))
print(f"   isotonic y range: [{m['iso_y'][0]:.4f}, {m['iso_y'][-1]:.4f}]  #distinct={len(set(np.round(m['iso_y'],4)))}")
for thr in (80, 150):
    hr = df[df.rain_day >= thr]
    print(f"   empirical flood rate on rain_day>={thr}mm: {hr.y.mean()*100:.2f}% (n={len(hr)}, pos={int(hr.y.sum())})")

print("\n=== §5 Extreme-rain coverage ===")
print(f"   max rain_day={df.rain_day.max():.0f}mm | days>200mm={(df.rain_day>200).sum()} | days>300mm={(df.rain_day>300).sum()}")
