"""
Train flood-probability classifier (scikit-learn) — see MODEL.md.

Pipeline:
  split        : train = 2023, held-out TEST = 2024  (time split, no leakage)
  validation   : StratifiedKFold(5) on 2023 for Optuna  (floods are seasonal &
                 sparse, so a within-year date cut leaves the val fold ~0 positives;
                 stratified CV keeps positives in every fold)
  preprocess   : ColumnTransformer
                   - continuous -> SimpleImputer(median) + StandardScaler  (scale diff.)
                   - 구 (gu_code) -> OneHotEncoder                          (categorical)
  imbalance    : balanced sample_weight (positives ~1.1%)
  model        : HistGradientBoostingClassifier
  tuning       : Optuna, maximize mean CV PR-AUC
  calibration  : CalibratedClassifierCV(isotonic, cv=3) -> usable probabilities
  eval         : held-out 2024 (PR-AUC / ROC-AUC / recall@k)

Run:  python train.py [--trials N]
Out:  build/model.pkl, build/metrics.json, build/serve_tables.json
"""
import os, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "build")

CONT = ["rain_day", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d",
        "rain_ante7", "rain_max10", "rain_max60",
        "hist_flood_cnt_prior", "hist_flood_rate_prior", "days_since_last_flood",
        "prev_year_same_month_cnt", "prev_year_same_week_cnt", "neighbor_gu_flood_prior"]
CAT = ["gu_code"]
FEATURES = CONT + CAT
PR_AUC = make_scorer(average_precision_score, response_method="predict_proba")


NUM_IDX = list(range(len(CONT)))      # positional indices -> serve with plain numpy (no pandas)
CAT_IDX = [len(CONT) + i for i in range(len(CAT))]


def make_pipe(**params):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_IDX),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_IDX),
    ])
    return Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(
                         random_state=0, early_stopping=False, **params))])


def bal_weight(y):
    return np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)


def recall_at_k(y, p, k):
    idx = np.argsort(p)[::-1][:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return dict(n=int(len(y)), pos=int(y.sum()),
               pr_auc=round(average_precision_score(y, p), 4),
               roc_auc=round(roc_auc_score(y, p), 4),
               r_at_20=round(recall_at_k(y, p, 20), 3),
               r_at_50=round(recall_at_k(y, p, 50), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(OUT, "samples.parquet"))
    df["year"] = df["date"].dt.year
    tr = df[df.year == 2023]; te = df[df.year == 2024]
    Xtr, ytr = tr[FEATURES].to_numpy(float), tr["y"].values   # numpy -> positional CT, pandas-free serve
    Xte, yte = te[FEATURES].to_numpy(float), te["y"].values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print(f"train(2023)={len(tr)} pos={ytr.sum()} | test(2024)={len(te)} pos={yte.sum()}")

    # ---- Optuna: maximize mean CV PR-AUC on 2023 ----------------------------
    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 500),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 60),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 31),
        )
        scores = cross_val_score(make_pipe(**params), Xtr, ytr, cv=cv, scoring=PR_AUC,
                                 params={"clf__sample_weight": bal_weight(ytr)})
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    print(f"optuna best CV PR-AUC={study.best_value:.3f}")
    print(f"best params={json.dumps(study.best_params)}")

    # ---- refit on 2023 + isotonic calibration -------------------------------
    base = make_pipe(**study.best_params)
    base.fit(Xtr, ytr, clf__sample_weight=bal_weight(ytr))
    cal = CalibratedClassifierCV(make_pipe(**study.best_params), method="isotonic", cv=3)
    cal.fit(Xtr, ytr, sample_weight=bal_weight(ytr))

    metrics = {"val_cv_pr_auc": round(study.best_value, 4),
               "best_params": study.best_params,
               "train": evaluate(base, Xtr, ytr),
               "test_uncalibrated": evaluate(base, Xte, yte),
               "test_calibrated": evaluate(cal, Xte, yte),
               "baseline_rain7d_roc_auc_test":
                   round(roc_auc_score(yte, te["rain_7d"].fillna(0).values), 4)}
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- export serving artifacts -------------------------------------------
    with open(os.path.join(OUT, "model.pkl"), "wb") as f:
        pickle.dump({"model": cal, "uncalibrated": base, "features": FEATURES,
                     "cont": CONT, "cat": CAT}, f)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # pandas-free serving table: latest history snapshot per dong (-> small JSON)
    hist_cols = ["hist_flood_cnt_prior", "hist_flood_rate_prior", "days_since_last_flood",
                 "prev_year_same_month_cnt", "prev_year_same_week_cnt",
                 "neighbor_gu_flood_prior", "gu_code", "gu", "dong_label"]
    snap = (df.sort_values("date").groupby("adm_cd").last()[hist_cols].reset_index())
    tables = {str(int(r.adm_cd)): {
                 "gu": r.gu, "dong_label": (None if pd.isna(r.dong_label) else r.dong_label),
                 "gu_code": int(r.gu_code),
                 "hist_flood_cnt_prior": float(r.hist_flood_cnt_prior),
                 "hist_flood_rate_prior": float(r.hist_flood_rate_prior),
                 "days_since_last_flood": float(r.days_since_last_flood),
                 "prev_year_same_month_cnt": float(r.prev_year_same_month_cnt),
                 "prev_year_same_week_cnt": float(r.prev_year_same_week_cnt),
                 "neighbor_gu_flood_prior": float(r.neighbor_gu_flood_prior),
              } for r in snap.itertuples()}
    with open(os.path.join(OUT, "serve_tables.json"), "w") as f:
        json.dump(tables, f, ensure_ascii=False)
    print(f"wrote build/model.pkl, metrics.json, serve_tables.json ({len(tables)} dongs)")


if __name__ == "__main__":
    main()
