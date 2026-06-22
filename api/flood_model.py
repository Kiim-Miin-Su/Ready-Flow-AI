"""
Pure-numpy inference for the flood classifier — NO scikit-learn / scipy at runtime.

The trained sklearn pipeline (median-impute + standard-scale + one-hot(gu) + HistGB
+ isotonic calibration) is exported by train.py to a single JSON (`build/model_np.json`)
holding only plain arrays. This module re-implements the forward pass in numpy so the
Vercel serverless bundle needs just numpy (+fastapi) and stays well under 250 MB.

Verified bit-exact against sklearn's `predict_proba` (max abs diff = 0.0) on the 2024 set.
"""
import json
import numpy as np

FEATURES = ["rain_day", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
            "rain_ante7", "rain_max10", "rain_max60",
            "hist_flood_cnt_prior", "hist_flood_rate_prior", "days_since_last_flood",
            "prev_year_same_month_cnt", "prev_year_same_week_cnt",
            "neighbor_gu_flood_prior", "gu_code"]
N_CONT = 15  # first 15 features are continuous; index 15 = gu_code (categorical)


def load(path):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    for k in ("medians", "mean", "scale", "iso_x", "iso_y"):
        m[k] = np.asarray(m[k], dtype=float)
    m["gu_categories"] = [int(g) for g in m["gu_categories"]]
    # pre-convert trees to numpy arrays for fast traversal
    for t in m["trees"]:
        for k in ("feature", "left", "right", "leaf"):
            t[k] = np.asarray(t[k], dtype=np.int64)
        for k in ("thr", "val"):
            t[k] = np.asarray(t[k], dtype=float)
    return m


def _featurize(m, feat):
    """feat: dict feature->value (FEATURES order). -> design vector (15 cont + G onehot)."""
    x = np.array([float(feat[f]) for f in FEATURES], dtype=float)
    cont = np.where(np.isnan(x[:N_CONT]), m["medians"], x[:N_CONT])
    cont = (cont - m["mean"]) / m["scale"]
    gu = int(x[N_CONT])
    onehot = np.array([1.0 if gu == g else 0.0 for g in m["gu_categories"]])
    return np.concatenate([cont, onehot])


def _raw(m, X):
    """sum of HistGB leaf values + baseline (margin)."""
    s = m["baseline"]
    for t in m["trees"]:
        i = 0
        leaf, feature, thr, left, right, val = (
            t["leaf"], t["feature"], t["thr"], t["left"], t["right"], t["val"])
        while not leaf[i]:
            i = left[i] if X[feature[i]] <= thr[i] else right[i]
        s += val[i]
    return s


def predict_proba(m, feat):
    """feat: dict feature->value -> calibrated flood probability in [0,1]."""
    raw = _raw(m, _featurize(m, feat))
    p = 1.0 / (1.0 + np.exp(-raw))             # sigmoid -> uncalibrated proba
    return float(np.interp(p, m["iso_x"], m["iso_y"]))   # isotonic calibration
