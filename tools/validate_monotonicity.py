"""Replicate PREDICTION.md probes A-E + a monotonicity unit test against the
deployed model, using the EXACT serving feature logic (api/index.py rain_windows +
build/serve_tables.json + api/flood_model.py). Pure local, no network.

Run:  python tools/validate_monotonicity.py
Exit code 0 if all checks pass (monotonicity violations == 0), else 1 — CI-friendly.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "api"))
import flood_model as fm

MODEL = fm.load(os.path.join(ROOT, "build", "model_np.json"))
TABLES = json.load(open(os.path.join(ROOT, "build", "serve_tables.json"), encoding="utf-8"))
HIST_KEYS = ("hist_flood_cnt_prior","hist_flood_rate_prior","days_since_last_flood",
             "prev_year_same_month_cnt","prev_year_same_week_cnt",
             "neighbor_gu_flood_prior","gu_code")
SILLIM = 1162010200  # 관악구 신림동


def rain_windows(series):
    s = np.asarray(series, dtype=float)
    if s.size == 0: s = np.zeros(1)
    today = float(s[-1]); csum = lambda w: float(s[-w:].sum())
    return {"rain_day":today,"rain_1d":today,"rain_3d":csum(3),"rain_7d":csum(7),
            "rain_14d":csum(14),"rain_30d":csum(30),
            "rain_ante7": float(s[-8:-1].sum()) if s.size>1 else 0.0,
            "rain_max10":today/6.0,"rain_max60":today/2.0}


def prob(adm_cd, rain):
    info = TABLES[str(adm_cd)]
    feat = rain_windows(rain); feat.update({k: info[k] for k in HIST_KEYS})
    return fm.predict_proba(MODEL, feat)


def p(adm, rain): return round(prob(adm, rain), 4)

print("A. 오늘 강우만 변화 (신림동):",
      {mm: p(SILLIM, [0,0,0,mm]) for mm in [0,30,80,200,400,600]})
seqsB = [[5,10,20,30],[10,20,40,60],[50,80,120,180],[100,150,200,300],[200,300,400,500]]
valsB = [p(SILLIM, s) for s in seqsB]
print("B. 스케일업 단조:", list(zip([s[-1] for s in seqsB], valsB)))
print("치명검증 2022-08-08 신림동 폭우 vs 평온:",
      p(SILLIM,[10,30,80,180,381]), "vs", p(SILLIM,[0,0,0,5]))

# monotonicity unit test: increasing any rainfall day must not decrease probability
rng = np.random.default_rng(0); fails = tested = 0
adms = [a for a in TABLES][:8]
for adm in adms:
    for _ in range(40):
        base = np.sort(rng.uniform(0, 200, 4))
        bump = base.copy(); bump[rng.integers(0, 4)] += rng.uniform(5, 150)
        pb = prob(int(adm), base.tolist()); pp = prob(int(adm), np.sort(bump).tolist())
        tested += 1
        if pp < pb - 1e-9: fails += 1
print(f"\nMONOTONICITY UNIT TEST: {tested} cases, violations={fails}")
ok = (fails == 0 and valsB == sorted(valsB))
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
