"""
Flood-classification preprocessing pipeline.

Builds an ML-ready (법정동 × 날짜) panel from:
  - data/seoul_rain/*.csv        : 48 rain gauges, 10-min rainfall, 2023-2024 (CP949)
  - data/20{23,24}년 침수흔적도.. : flood-trace records (DBF, CP949)

Target task: binary classification  y in {0,1}  =  "did dong d flood on day t?"
Serving target: input = user address (-> dong) + weather forecast (-> rain windows)
                output = P(flood).  Only serve-time-available features go in the model matrix.

Outputs (./build):
  samples.parquet / samples.csv  : the (dong x date) training panel + features + y
  dong_static.parquet            : per-dong static attributes (identity, building, terrain scaffold)
  edges.csv                      : dong adjacency edge list (proxy) for GNN
  rain_daily.parquet             : intermediate per-구 daily rain
See DATA.md for full rationale.
"""

import os, glob, struct, unicodedata
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAIN_DIR = os.path.join(ROOT, "data", "seoul_rain")
OUT = os.path.join(ROOT, "build")
os.makedirs(OUT, exist_ok=True)
nfc = lambda s: unicodedata.normalize("NFC", s)

# ---- window config -----------------------------------------------------------
RAIN_WINDOWS = [1, 3, 7, 14, 30]  # cumulative-precip windows (days), ending at t
RAIN_MIN_MM = 1.0  # a "candidate day" = 구 daily rain >= this (mm)
SEASON_MONTHS = None  # set e.g. {5..10} to restrict; None = all days with rain

# ---- station name -> 구 ------------------------------------------------------
# Every 구 has a 구청 gauge (primary). Non-구청 gauges geolocated to their 구.
STATION_GU = {
    "가산2펌프장": "금천구",
    "갈현1동": "은평구",
    "개봉2동": "구로구",
    "개포2동": "강남구",
    "고덕2동": "강동구",
    "공항펌프장": "강서구",
    "도림2펌프장": "영등포구",
    "뚝섬펌프장": "광진구",
    "마천2동": "송파구",
    "면목펌프장": "중랑구",
    "목동펌프장": "양천구",
    "반포펌프장": "서초구",
    "봉원펌프장": "서대문구",
    "부암동": "종로구",
    "상계1동": "노원구",
    "상월곡동": "성북구",
    "서소문": "중구",
    "세곡동": "강남구",
    "신림펌프장": "관악구",
    "증산펌프장": "은평구",
    "한남펌프장": "용산구",
    "휘경펌프장": "동대문구",
    "흑석펌프장": "동작구",
}


def station_to_gu(name):
    name = nfc(name)
    if name.endswith("구청"):
        return name[:-2] + "구" if not name[:-2].endswith("구") else name[:-2]
    return STATION_GU.get(name)


# =============================================================================
# 1. RAIN  ->  per-구 daily rainfall + intensity
# =============================================================================
def load_rain():
    rows = []
    for f in glob.glob(os.path.join(RAIN_DIR, "*.csv")):
        base = nfc(os.path.basename(f))
        if "구청" not in base:  # primary series = 구청 gauge only (1 per 구, complete)
            continue
        df = pd.read_csv(f, encoding="cp949")
        df.columns = ["station", "time", "rain10"]
        df["station"] = df["station"].map(nfc)
        df["gu"] = df["station"].map(station_to_gu)
        df["dt"] = pd.to_datetime(df["time"], format="mixed")
        df["rain10"] = pd.to_numeric(df["rain10"], errors="coerce").fillna(0.0)
        rows.append(df[["gu", "dt", "rain10"]])
    rain = pd.concat(rows, ignore_index=True)
    rain["date"] = rain["dt"].dt.normalize()

    # daily total + peak intensity (max 10-min, max rolling 60-min) per 구
    rain = rain.sort_values(["gu", "dt"])
    rain["roll60"] = rain.groupby("gu")["rain10"].transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    )
    daily = (
        rain.groupby(["gu", "date"])
        .agg(
            rain_day=("rain10", "sum"),
            rain_max10=("rain10", "max"),
            rain_max60=("roll60", "max"),
        )
        .reset_index()
    )

    # continuous daily index per 구 (fill dry days with 0) so calendar windows are correct
    full = []
    for gu, g in daily.groupby("gu"):
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx).fillna(0.0)
        g.index.name = "date"
        g["gu"] = gu
        # cumulative-precip windows
        for w in RAIN_WINDOWS:
            g[f"rain_{w}d"] = g["rain_day"].rolling(w, min_periods=1).sum()
        # antecedent precip (excludes today): rain over t-7..t-1
        g["rain_ante7"] = (
            g["rain_day"].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
        )
        full.append(g.reset_index())
    daily = pd.concat(full, ignore_index=True)
    return daily


# =============================================================================
# 2. FLOOD  ->  (dong, date) events
# =============================================================================
def read_dbf(path):
    with open(path, "rb") as f:
        hdr = f.read(32)
        nrec = struct.unpack("<I", hdr[4:8])[0]
        hlen = struct.unpack("<H", hdr[8:10])[0]
        rlen = struct.unpack("<H", hdr[10:12])[0]
        nf = (hlen - 32 - 1) // 32
        fields = []
        for _ in range(nf):
            fd = f.read(32)
            fields.append((fd[:11].split(b"\x00")[0].decode("cp949"), fd[16]))
        f.seek(hlen)
        recs = []
        for _ in range(nrec):
            rec = f.read(rlen)
            off = 1
            d = {}
            for name, flen in fields:
                d[name] = rec[off : off + flen].decode("cp949", "replace").strip()
                off += flen
            recs.append(d)
    return pd.DataFrame(recs)


BTYPE_MAP = [  # raw TYPE substring -> coarse bucket
    ("주택", "residential"),
    ("아파트", "residential"),
    ("빌라", "residential"),
    ("상가", "commercial"),
    ("건물", "commercial"),
    ("시설", "commercial"),
    ("공장", "industrial"),
    ("도로", "road"),
    ("지하", "underground"),
]


def coarse_btype(t):
    t = nfc(str(t))
    for k, v in BTYPE_MAP:
        if k in t:
            return v
    return "unknown" if not t else "other"


def load_flood():
    paths = {
        2023: os.path.join(
            ROOT,
            "data",
            nfc("2023년 침수흔적도_260105 수정"),
            nfc("2023 침수흔적도 최신수정.dbf"),
        ),
        2024: os.path.join(
            ROOT, "data", nfc("2024년 침수흔적도_260105 수정"), nfc("2024 최종.dbf")
        ),
    }
    out = []
    for yr, p in paths.items():
        df = read_dbf(p)
        df = df.apply(lambda c: c.map(nfc) if c.dtype == object else c)
        df["adm_cd"] = pd.to_numeric(df["ADM_CD"], errors="coerce").astype(
            "Int64"
        )  # 10-digit 법정동
        df["gu"] = df["GU_NAM"]
        df["date"] = pd.to_datetime(df["F_SAT_YMD"], format="%Y%m%d", errors="coerce")
        df["depth_m"] = pd.to_numeric(df["F_SHIM"], errors="coerce")
        df["elev_pt"] = pd.to_numeric(
            df["F_AVR_HGT"], errors="coerce"
        )  # leaky -> NOT a feature
        df["btype"] = df["TYPE"].map(coarse_btype)
        df["src_year"] = yr
        out.append(
            df[
                [
                    "adm_cd",
                    "gu",
                    "date",
                    "depth_m",
                    "elev_pt",
                    "btype",
                    "F_ZONE_NM",
                    "src_year",
                ]
            ]
        )
    flood = pd.concat(out, ignore_index=True)
    flood = flood.dropna(subset=["adm_cd", "date"])
    flood = flood[(flood["date"].dt.year >= 2023) & (flood["date"].dt.year <= 2024)]

    # --- fix malformed 구 prefix in ADM_CD (data-driven) ----------------------
    # Some records carry a typo'd 구 code (e.g. 노원구 1035010500 should be 1135010500).
    # Canonical 구 code = most common well-formed (11xxx) prefix per GU_NAM; rebuild
    # adm_cd = canonical_gu_code*1e5 + 동코드(last 5 digits, trusted).
    pref = (flood["adm_cd"] // 100000).astype("int64")
    wf = flood[pref // 1000 == 11]                       # Seoul prefixes start with 11
    canon = (wf.assign(p=wf["adm_cd"] // 100000)
               .groupby("gu")["p"].agg(lambda s: s.mode().iloc[0]))
    fixed = flood["gu"].map(canon).astype("Int64") * 100000 + (flood["adm_cd"] % 100000)
    n_fix = int((fixed != flood["adm_cd"]).sum())
    if n_fix:
        print(f"     [fix] normalized {n_fix} malformed ADM_CD 구-prefix")
    flood["adm_cd"] = fixed.fillna(flood["adm_cd"]).astype("Int64")
    # human-readable dong label (best-effort from address text)
    flood["dong_label"] = flood["F_ZONE_NM"].str.extract(r"([가-힣]+\d?동)")
    return flood


# =============================================================================
# 3. PANEL  +  leak-free features
# =============================================================================
def build_panel(daily, flood):
    # dong universe = dongs observed in flood data (flood-prone set); 구 from code prefix
    dongs = (
        flood.groupby("adm_cd")
        .agg(gu=("gu", "first"), dong_label=("dong_label", "first"))
        .reset_index()
    )
    dongs["gu_code"] = (dongs["adm_cd"] // 100000).astype("int64")  # 5-digit 구 code

    # candidate days per 구 (rain >= threshold) ∪ all actual flood days
    cand = daily[daily["rain_day"] >= RAIN_MIN_MM][["gu", "date"]].copy()
    fl_days = flood[["gu"]].assign(date=flood["date"]).drop_duplicates()
    cand = pd.concat([cand, fl_days], ignore_index=True).drop_duplicates()
    if SEASON_MONTHS:
        cand = cand[cand["date"].dt.month.isin(SEASON_MONTHS)]

    # cross dongs × their 구's candidate days
    panel = dongs.merge(cand, on="gu", how="inner")

    # attach rain features
    panel = panel.merge(daily, on=["gu", "date"], how="left")

    # target
    fset = set(zip(flood["adm_cd"].astype("int64"), flood["date"]))
    panel["y"] = [
        int((a, d) in fset)
        for a, d in zip(panel["adm_cd"].astype("int64"), panel["date"])
    ]

    # ---- leak-free history features (strictly prior to t) --------------------
    fl = flood.sort_values("date")
    fl_by_dong = {a: g["date"].sort_values().values for a, g in fl.groupby("adm_cd")}
    fl_by_gu = {g_: gg.sort_values().values for g_, gg in fl.groupby("gu")["date"]}

    panel = panel.sort_values(["adm_cd", "date"]).reset_index(drop=True)

    def prior_count(arr, t):  # floods strictly before t
        return (
            int(np.searchsorted(arr, np.datetime64(t), side="left"))
            if arr is not None
            else 0
        )

    hist_cnt, days_since, prev_y_m, prev_y_w = [], [], [], []
    for a, t in zip(panel["adm_cd"].astype("int64"), panel["date"]):
        arr = fl_by_dong.get(a)
        c = prior_count(arr, t)
        hist_cnt.append(c)
        if arr is not None and c > 0:
            last = arr[c - 1]
            days_since.append(int((np.datetime64(t) - last) / np.timedelta64(1, "D")))
        else:
            days_since.append(-1)  # never flooded before -> sentinel
        # prev-year same month / same ISO week (uses full prior year => leak-free)
        ty = pd.Timestamp(t)
        if arr is not None:
            ad = pd.DatetimeIndex(arr)
            prev_y_m.append(
                int(((ad.year == ty.year - 1) & (ad.month == ty.month)).sum())
            )
            iso_t = ty.isocalendar()
            prev_y_w.append(
                int(
                    (
                        (ad.isocalendar().year == iso_t.year - 1)
                        & (ad.isocalendar().week == iso_t.week)
                    ).sum()
                )
            )
        else:
            prev_y_m.append(0)
            prev_y_w.append(0)
    panel["hist_flood_cnt_prior"] = hist_cnt
    panel["days_since_last_flood"] = days_since
    panel["prev_year_same_month_cnt"] = prev_y_m
    panel["prev_year_same_week_cnt"] = prev_y_w

    # expanding flood RATE per dong (prior floods / prior candidate days)
    panel["row_in_dong"] = panel.groupby(
        "adm_cd"
    ).cumcount()  # # of prior candidate days
    panel["hist_flood_rate_prior"] = panel["hist_flood_cnt_prior"] / panel[
        "row_in_dong"
    ].replace(0, np.nan)
    panel["hist_flood_rate_prior"] = panel["hist_flood_rate_prior"].fillna(0.0)

    # ---- neighbor proxy: same-구 prior flood rate (excl. self), leak-free -----
    # per (gu,date): floods in 구 strictly before date, normalized by #dongs*priordays approx.
    nb = []
    gu_dong_cnt = dongs.groupby("gu")["adm_cd"].nunique().to_dict()
    for g_, t, a, c in zip(
        panel["gu"],
        panel["date"],
        panel["adm_cd"].astype("int64"),
        panel["hist_flood_cnt_prior"],
    ):
        arr = fl_by_gu.get(g_)
        gu_prior = prior_count(arr, t) if arr is not None else 0
        nb.append((gu_prior - c) / max(gu_dong_cnt.get(g_, 1), 1))  # exclude self count
    panel["neighbor_gu_flood_prior"] = nb

    panel = panel.drop(columns=["row_in_dong"])
    return dongs, panel


# =============================================================================
# 4. dong static (building composition + terrain scaffold) + GNN edges
# =============================================================================
def build_static(dongs, flood):
    # building-type composition from flood history (WEAK PRIOR; label-derived -> caveat)
    ct = pd.crosstab(flood["adm_cd"], flood["btype"])
    comp = ct.div(ct.sum(axis=1), axis=0)
    comp.columns = [f"btype_frac_{c}" for c in comp.columns]
    dominant = flood.groupby("adm_cd")["btype"].agg(lambda s: s.value_counts().idxmax())
    static = (
        dongs.set_index("adm_cd").join(comp).join(dominant.rename("dominant_btype"))
    )
    # terrain scaffold (EXTERNAL data required -> NaN placeholders, see DATA.md)
    for col in ["elev_mean_m", "dist_to_river_m", "drainage_density"]:
        static[col] = np.nan
    return static.reset_index()


def build_edges(dongs):
    # proxy adjacency: dongs in the same 구 are connected (clique). See DATA.md for upgrade.
    edges = []
    for gu, g in dongs.groupby("gu"):
        ids = sorted(g["adm_cd"].astype("int64").tolist())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.append((ids[i], ids[j], "same_gu"))
    return pd.DataFrame(edges, columns=["src_adm_cd", "dst_adm_cd", "edge_type"])


# =============================================================================
def main():
    print("[1/4] rain ...")
    daily = load_rain()
    daily.to_parquet(os.path.join(OUT, "rain_daily.parquet"))
    print(
        "     구:",
        daily["gu"].nunique(),
        "| days:",
        daily["date"].nunique(),
        "| span:",
        daily["date"].min().date(),
        "->",
        daily["date"].max().date(),
    )

    print("[2/4] flood ...")
    flood = load_flood()
    print(
        "     events:",
        len(flood),
        "| dongs:",
        flood["adm_cd"].nunique(),
        "| 구:",
        flood["gu"].nunique(),
    )

    print("[3/4] panel ...")
    dongs, panel = build_panel(daily, flood)
    print(
        "     rows:",
        len(panel),
        "| dongs:",
        panel["adm_cd"].nunique(),
        "| positives:",
        int(panel["y"].sum()),
        f"({panel['y'].mean()*100:.2f}%)",
    )

    print("[4/4] static + edges ...")
    static = build_static(dongs, flood)
    edges = build_edges(dongs)

    panel.to_parquet(os.path.join(OUT, "samples.parquet"))
    panel.to_csv(os.path.join(OUT, "samples.csv"), index=False, encoding="utf-8-sig")
    static.to_parquet(os.path.join(OUT, "dong_static.parquet"))
    static.to_csv(
        os.path.join(OUT, "dong_static.csv"), index=False, encoding="utf-8-sig"
    )
    edges.to_csv(os.path.join(OUT, "edges.csv"), index=False, encoding="utf-8-sig")
    print("     wrote build/: samples, dong_static, edges, rain_daily")
    print("     edges:", len(edges))

    # quick feature summary for DATA.md
    print("\n--- positives by year ---")
    print(panel[panel.y == 1]["date"].dt.year.value_counts().sort_index().to_string())
    print("--- feature columns ---")
    print(list(panel.columns))


if __name__ == "__main__":
    main()
