"""
FastAPI serving app (Vercel entrypoint) — pandas-free for a small deploy bundle.

Input : 유저 주소 + 실시간 예보 일강우 시퀀스
Output: 침수 확률 P(flood)

Swagger UI : /docs    ·    ReDoc : /redoc    ·    OpenAPI JSON : /openapi.json
Deps (requirements.txt): fastapi, numpy  ONLY  (NO scikit-learn / scipy / pandas / torch)
-> the sklearn model is exported to build/model_np.json and run by flood_model.py.
"""
import os
import sys
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Accepted building-type values (optional input). Coarse buckets derived from the
# flood-trace TYPE field. NOTE: the current model does NOT consume this — it is a
# reserved/forward-compatible field so the client can collect it now and a future
# model version can use it without a breaking payload change. See DATA.md §6.
BuildingType = Literal["residential", "commercial", "industrial", "underground", "road", "etc"]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "build")
sys.path.insert(0, HERE)          # bundle flood_model.py alongside this file
import flood_model as fm

MODEL = fm.load(os.path.join(ART, "model_np.json"))
FEATURES = MODEL["features"]
with open(os.path.join(ART, "serve_tables.json"), encoding="utf-8") as f:
    TABLES = json.load(f)            # adm_cd(str) -> dong static/history snapshot
LABEL2ADM = {v["dong_label"]: k for k, v in TABLES.items() if v.get("dong_label")}

HIST_KEYS = ("hist_flood_cnt_prior", "hist_flood_rate_prior", "days_since_last_flood",
             "prev_year_same_month_cnt", "prev_year_same_week_cnt",
             "neighbor_gu_flood_prior", "gu_code")

app = FastAPI(
    title="Seoul Flood Risk API",
    version="1.0.0",
    description=(
        "서울 법정동 단위 침수 확률 예측 API.\n\n"
        "- 입력: 유저 주소(→법정동) + 실시간 예보 일강우 시퀀스\n"
        "- 출력: 침수 확률 `flood_probability` ∈ [0,1]\n"
        "- 커버리지: 침수 이력이 있는 93개 법정동 (그 외 404)\n\n"
        "Flutter 연동 가이드는 FRONTEND.md 참고."),
    contact={"name": "doubled_seven"},
)
# Flutter 등 외부 클라이언트 호출 허용 (운영 시 도메인 화이트리스트로 좁히기)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# ----------------------------- schemas ---------------------------------------
class PredictRequest(BaseModel):
    address: str = Field(
        "", description="유저 주소. 지오코딩→법정동. adm_cd를 직접 주면 비워도 됨.",
        examples=["서울 노원구 중계동 23-28"])
    forecast_daily_rain: list[float] = Field(
        ..., min_length=1, max_length=30,
        description="일강우(mm) 시퀀스. 과거→오늘 순서(오늘이 마지막). 최대 30개.",
        examples=[[5, 40, 60, 100]])
    adm_cd: int | None = Field(
        None, description="10자리 법정동코드. 주면 지오코딩을 건너뜀.",
        examples=[1135010600])
    building_type: Optional[BuildingType] = Field(
        None, description=(
            "(선택) 건물 유형. 허용값: residential(주거)·commercial(상가/시설)·"
            "industrial(공장)·underground(지하/반지하)·road(도로)·etc(기타). "
            "현재 모델은 사용하지 않으며 안 보내도 됨 — 향후 확장용 예약 필드."),
        examples=["residential"])

    model_config = {"json_schema_extra": {"examples": [
        {"address": "서울 노원구 중계동 23-28", "forecast_daily_rain": [5, 40, 60, 100]},
        {"address": "", "adm_cd": 1135010600, "forecast_daily_rain": [10, 30, 80]},
    ]}}


RiskLevel = Literal["info", "warning", "danger"]


class PredictResponse(BaseModel):
    adm_cd: int = Field(..., examples=[1135010600], description="법정동코드")
    gu: str = Field(..., examples=["노원구"])
    dong: str | None = Field(None, examples=["중계동"], description="동 라벨(없을 수 있음)")
    flood_probability: float = Field(..., ge=0, le=1, examples=[0.7473],
                                     description="침수 확률 [0,1] (isotonic 보정값)")
    risk_level: RiskLevel | None = Field(
        None, examples=["warning"],
        description=("상대 위험등급: info(관심)·warning(주의)·danger(경고). "
                     "보정확률은 폭우일에도 낮으므로(현실 반영) 절대%가 아닌 "
                     "학습분포 백분위 기준 등급을 함께 제공. cut: warning=상위15%, danger=상위1%."))
    risk_percentile: int | None = Field(
        None, ge=0, le=100, examples=[88],
        description="이 예측이 학습 예측분포에서 차지하는 백분위(0-100). 높을수록 상대적 고위험.")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    dongs: int = Field(..., examples=[93], description="커버리지 동 수")
    features: int = Field(..., examples=[16])


class ErrorResponse(BaseModel):
    detail: str


class FloodForecastDay(BaseModel):
    date: str = Field(..., examples=["2026-06-24"])
    rain_mm: float = Field(..., ge=0, examples=[42.5], description="해당일 예보 강수량 합계(mm)")
    flood_probability: float = Field(..., ge=0, le=1, examples=[0.0765])
    risk_level: RiskLevel | None = Field(None, examples=["warning"])
    risk_percentile: int | None = Field(None, ge=0, le=100, examples=[88])


class FloodWeekForecastResponse(BaseModel):
    adm_cd: int
    gu: str
    dong: str | None
    days: list[FloodForecastDay]
    peak: FloodForecastDay | None
    source: str = Field("kma_vilage_fcst")
    detail: str | None = None


class AsosMonthlyResponse(BaseModel):
    year: int
    month: int
    stn: str = Field(..., examples=["108"], description="관측소(108=서울)")
    rain_by_day: dict[str, float] = Field(
        ..., description="일(1~31) → 일강수량(mm). 관측 없으면 0.0", examples=[{"1": 0.0, "2": 12.5}])
    source: str = Field("kma_asos")


class WarningBulletinResponse(BaseModel):
    bulletin: str | None = Field(None, description="특보 통보문 전문(t6). 없으면 null")
    tm_fc: str | None = Field(None, description="발표시각(yyyyMMddHHmm)")
    has_warning: bool = Field(False, description="발효 통보문 존재 여부")
    source: str = Field("kma_wrn")


# ----------------------------- logic -----------------------------------------
def geocode_to_admcd(address: str) -> str:
    """STUB geocoder. 운영 시 VWorld/Kakao 주소→법정동코드로 교체."""
    for label, adm in LABEL2ADM.items():
        if label and label in address:
            return adm
    raise HTTPException(404, "dong not resolved from address (plug in a real geocoder)")


def rain_windows(series: list[float]) -> dict:
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        s = np.zeros(1)
    today = float(s[-1])
    csum = lambda w: float(s[-w:].sum())
    return {
        "rain_day": today, "rain_1d": today,
        "rain_3d": csum(3), "rain_7d": csum(7), "rain_14d": csum(14), "rain_30d": csum(30),
        "rain_ante7": float(s[-8:-1].sum()) if s.size > 1 else 0.0,
        # 일강우만 있을 때 분단위 강도 근사 (시간당 예보 있으면 교체)
        "rain_max10": today / 6.0, "rain_max60": today / 2.0,
    }


def _predict_for_adm(adm: str, rain_series: list[float]) -> dict:
    if adm not in TABLES:
        raise HTTPException(404, f"adm_cd {adm} not in coverage (93 flood-prone dongs)")
    info = TABLES[adm]
    feat = rain_windows(rain_series)
    feat.update({k: info[k] for k in HIST_KEYS})
    prob = fm.predict_proba(MODEL, feat)
    return {
        "adm_cd": int(adm),
        "gu": info["gu"],
        "dong": info.get("dong_label"),
        "flood_probability": round(prob, 4),
        "risk_level": fm.risk_level(MODEL, prob),
        "risk_percentile": fm.risk_percentile(MODEL, prob),
    }


# KMA 동네예보 격자 좌표(nx, ny) — 서울 25개 자치구청 기준(약 5km 격자).
# adm_cd 앞 5자리 = gu_code 로 매핑한다. 동 단위 정밀 좌표가 없어 자치구 대표
# 격자를 쓰며, 국지성 호우 시에도 구 단위로는 충분히 구분된다(격자 ≈ 5km).
GU_GRID: dict[int, tuple[int, int]] = {
    11110: (60, 127),  # 종로구
    11140: (60, 127),  # 중구
    11170: (60, 126),  # 용산구
    11200: (61, 127),  # 성동구
    11215: (62, 126),  # 광진구
    11230: (61, 127),  # 동대문구
    11260: (62, 128),  # 중랑구
    11290: (61, 127),  # 성북구
    11305: (61, 128),  # 강북구
    11320: (61, 129),  # 도봉구
    11350: (61, 129),  # 노원구
    11380: (59, 127),  # 은평구
    11410: (59, 127),  # 서대문구
    11440: (59, 127),  # 마포구
    11470: (58, 126),  # 양천구
    11500: (58, 126),  # 강서구
    11530: (58, 125),  # 구로구
    11545: (59, 124),  # 금천구
    11560: (58, 126),  # 영등포구
    11590: (59, 125),  # 동작구
    11620: (59, 125),  # 관악구
    11650: (61, 125),  # 서초구
    11680: (61, 126),  # 강남구
    11710: (62, 126),  # 송파구
    11740: (62, 126),  # 강동구
}
_SEOUL_CENTER_GRID = (60, 127)  # fallback (종로/중구)


def _grid_for_adm(adm: str) -> tuple[int, int]:
    """법정동코드 → 소속 자치구의 KMA 격자(nx, ny). 미상이면 서울 중심."""
    info = TABLES.get(adm, {})
    gu_code = int(info.get("gu_code") or 0)
    return GU_GRID.get(gu_code, _SEOUL_CENTER_GRID)


def _kma_key() -> str:
    return os.environ.get("KMA_SERVICE_KEY", "").strip()


def _forecast_base(now: datetime) -> tuple[str, str]:
    # KMA village forecast base times. Use a conservative previous slot because
    # newly announced data can lag the nominal base time.
    kst = now.astimezone(timezone(timedelta(hours=9))) - timedelta(hours=3)
    base_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    hour = max((h for h in base_hours if h <= kst.hour), default=23)
    if hour == 23 and kst.hour < 2:
        kst = kst - timedelta(days=1)
    return kst.strftime("%Y%m%d"), f"{hour:02d}00"


def _pcp_mm(value: str) -> float:
    text = (value or "").strip()
    if not text or "강수없음" in text:
        return 0.0
    if "1mm 미만" in text:
        return 0.5
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return 0.0
    return max(nums)


def _fetch_kma_daily_rain(days: int = 7, nx: int = 60, ny: int = 127) -> tuple[list[dict], str]:
    key = _kma_key()
    if not key:
        raise HTTPException(503, "KMA_SERVICE_KEY is not configured on the backend")

    base_date, base_time = _forecast_base(datetime.now(timezone.utc))
    params = {
        "serviceKey": key if "%" in key else urllib.parse.quote(key, safe=""),
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": str(nx),
        "ny": str(ny),
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = (
        "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/"
        f"getVilageFcst?{query}"
    )

    try:
        with urllib.request.urlopen(url, timeout=8) as res:
            body = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise HTTPException(
                503,
                "이 KMA 키는 '기상청_단기예보 조회서비스(VilageFcstInfoService_2.0)'에 "
                "활용신청되어 있지 않습니다. data.go.kr 에서 같은 키로 해당 서비스를 "
                "추가 신청하면 동작합니다. (기상특보·ASOS와는 별개 신청)",
            ) from e
        raise HTTPException(502, f"KMA forecast request failed: HTTP {e.code}") from e
    except Exception as e:
        raise HTTPException(502, f"KMA forecast request failed: {e}") from e

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(502, "KMA forecast returned a non-JSON response") from e

    response = payload.get("response", {})
    header = response.get("header", {})
    code = str(header.get("resultCode", ""))
    if code != "00":
        raise HTTPException(502, f"KMA forecast error {code}: {header.get('resultMsg')}")

    items = response.get("body", {}).get("items", {}).get("item", [])
    daily: dict[str, float] = {}
    for item in items:
        if item.get("category") != "PCP":
            continue
        date = str(item.get("fcstDate", ""))
        if len(date) != 8:
            continue
        iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        daily[iso] = daily.get(iso, 0.0) + _pcp_mm(str(item.get("fcstValue", "")))

    # 실제 예보가 있는 날짜만 포함한다(단기예보는 보통 오늘~+3일).
    # 예보 지평 밖의 날을 0mm로 채우면 '침수확률 0%'로 오해되므로 넣지 않는다.
    today = datetime.now(timezone(timedelta(hours=9))).date()
    result = []
    for iso in sorted(daily):
        try:
            d = datetime.strptime(iso, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < today:
            continue
        result.append({"date": iso, "rain_mm": round(daily[iso], 1)})
        if len(result) >= days:
            break
    return result, f"KMA getVilageFcst base {base_date} {base_time} nx={nx} ny={ny}"


def _kma_get(url: str) -> str:
    """공통 KMA GET — HTTP 오류를 의미 있는 HTTPException 으로 변환."""
    try:
        with urllib.request.urlopen(url, timeout=8) as res:
            return res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise HTTPException(
                503,
                "이 KMA 키가 해당 서비스에 활용신청되어 있지 않습니다(403). "
                "data.go.kr 에서 같은 키로 해당 서비스를 추가 신청하세요.",
            ) from e
        raise HTTPException(502, f"KMA request failed: HTTP {e.code}") from e
    except Exception as e:
        raise HTTPException(502, f"KMA request failed: {e}") from e


def _fetch_asos_monthly(year: int, month: int, stn: str = "108") -> dict[str, float]:
    """기상청 ASOS 일자료 — 해당 월의 일강수량(mm) 맵. 키는 백엔드 환경변수."""
    key = _kma_key()
    if not key:
        raise HTTPException(503, "KMA_SERVICE_KEY is not configured on the backend")

    last_day = (datetime(year + (month == 12), (month % 12) + 1, 1)
                - timedelta(days=1)).day
    # ASOS 일자료는 '전날'까지만 제공한다. 이번 달이면 endDt 를 어제로 제한.
    yesterday = (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=1)).date()
    end_day = last_day
    if (year, month) == (yesterday.year, yesterday.month):
        end_day = yesterday.day
    if year > yesterday.year or (year == yesterday.year and month > yesterday.month):
        return {}  # 미래 달 — 관측 자료 없음
    start = f"{year:04d}{month:02d}01"
    end = f"{year:04d}{month:02d}{end_day:02d}"
    params = {
        "serviceKey": key if "%" in key else urllib.parse.quote(key, safe=""),
        "pageNo": "1", "numOfRows": "40", "dataType": "JSON",
        "dataCd": "ASOS", "dateCd": "DAY",
        "startDt": start, "endDt": end, "stnIds": stn,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = ("https://apis.data.go.kr/1360000/AsosDalyInfoService/"
           f"getWthrDataList?{query}")
    body = _kma_get(url)

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(502, "KMA ASOS returned a non-JSON response") from e

    header = payload.get("response", {}).get("header", {})
    code = str(header.get("resultCode", ""))
    if code == "03":  # NODATA
        return {}
    if code and code != "00":
        raise HTTPException(502, f"KMA ASOS error {code}: {header.get('resultMsg')}")

    items = payload.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    rain: dict[str, float] = {}
    for item in items:
        tm = str(item.get("tm", ""))           # YYYY-MM-DD
        parts = tm.split("-")
        if len(parts) != 3:
            continue
        try:
            day = int(parts[2])
        except ValueError:
            continue
        raw = str(item.get("sumRn", "")).strip()
        try:
            rain[str(day)] = float(raw) if raw else 0.0
        except ValueError:
            rain[str(day)] = 0.0
    return rain


def _fetch_warning_bulletin(stn: str = "109") -> tuple[str | None, str | None]:
    """현재 발효 중인 기상특보 현황(t6)을 가져온다. (bulletin, tmFc).

    getPwnStatus(특보 현황)는 전국 현재 발효 특보를 t6 한 필드에
    "o <재해><주의보|경보> : <지역들>" 형식으로 돌려준다. 지역 매칭은
    클라이언트(parseBulletin)가 수행하므로 stnId 없이 전국 현황을 받는다.
    (getWthrWrnMsg 는 tmFc 단건 조회 시 대부분 NO_DATA 라 사용하지 않는다.)
    """
    key = _kma_key()
    if not key:
        raise HTTPException(503, "KMA_SERVICE_KEY is not configured on the backend")
    enc = key if "%" in key else urllib.parse.quote(key, safe="")
    base = "https://apis.data.go.kr/1360000/WthrWrnInfoService"

    now = datetime.now(timezone(timedelta(hours=9)))
    frm = (now - timedelta(days=3)).strftime("%Y%m%d")
    to = now.strftime("%Y%m%d")

    url = (f"{base}/getPwnStatus?serviceKey={enc}&dataType=JSON"
           f"&numOfRows=10&pageNo=1&fromTmFc={frm}&toTmFc={to}")
    body = _kma_get(url)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        raise HTTPException(502, "KMA warning status non-JSON") from e

    header = payload.get("response", {}).get("header", {})
    code = str(header.get("resultCode", ""))
    if code == "03":  # NO_DATA — 발효 특보 없음
        return None, None
    if code and code != "00":
        raise HTTPException(502, f"KMA warning status error {code}: {header.get('resultMsg')}")

    items = payload.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    # 최신 발표(tmFc 큰 것)의 t6 를 사용.
    best_t6, best_tm = None, None
    for it in items:
        tm = str(it.get("tmFc", ""))
        t6 = str(it.get("t6", "")).strip()
        if not t6:
            continue
        if best_tm is None or tm > best_tm:
            best_t6, best_tm = t6, tm
    return best_t6, best_tm


# ----------------------------- routes ----------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["meta"],
         summary="헬스체크")
def health():
    return {"status": "ok", "dongs": len(TABLES), "features": len(FEATURES)}


@app.post("/api/predict", response_model=PredictResponse, tags=["predict"],
          summary="침수 확률 예측",
          responses={404: {"model": ErrorResponse, "description": "동 미해결/커버리지 밖"},
                     422: {"description": "페이로드 검증 실패"}})
def predict(req: PredictRequest):
    """주소(또는 adm_cd) + 예보 일강우 → 해당 법정동의 침수 확률."""
    adm = str(req.adm_cd) if req.adm_cd is not None else geocode_to_admcd(req.address)
    return _predict_for_adm(adm, req.forecast_daily_rain)


@app.get("/api/forecast/flood-week", response_model=FloodWeekForecastResponse,
         tags=["forecast"], summary="향후 7일 침수 확률 예보",
         responses={404: {"model": ErrorResponse, "description": "커버리지 밖"},
                    502: {"model": ErrorResponse, "description": "기상청 호출 실패"},
                    503: {"model": ErrorResponse, "description": "기상청 키 미설정"}})
def flood_week(
    adm_cd: int = Query(..., description="10자리 법정동코드"),
    building_type: Optional[BuildingType] = Query(None, description="예약 필드. 현재 모델 미사용."),
):
    """기상청 단기예보를 서버에서 호출해 일별 강수량으로 합산하고 침수 확률을 계산한다."""
    adm = str(adm_cd)
    base = _predict_for_adm(adm, [0.0])  # validates coverage (404 if 밖)
    nx, ny = _grid_for_adm(adm)
    kma_days, detail = _fetch_kma_daily_rain(days=7, nx=nx, ny=ny)

    days = []
    history: list[float] = []
    for item in kma_days:
        history.append(float(item["rain_mm"]))
        pred = _predict_for_adm(adm, history)
        days.append({
            "date": item["date"],
            "rain_mm": item["rain_mm"],
            "flood_probability": pred["flood_probability"],
            "risk_level": pred["risk_level"],
            "risk_percentile": pred["risk_percentile"],
        })

    peak = max(days, key=lambda x: (x["flood_probability"], x["rain_mm"])) if days else None
    return {
        "adm_cd": int(adm),
        "gu": base["gu"],
        "dong": base["dong"],
        "days": days,
        "peak": peak,
        "source": "kma_vilage_fcst",
        "detail": detail,
    }


@app.get("/api/weather/asos-monthly", response_model=AsosMonthlyResponse,
         tags=["weather"], summary="월간 ASOS 일강수량(캘린더 과거 실측)",
         responses={502: {"model": ErrorResponse, "description": "기상청 호출 실패"},
                    503: {"model": ErrorResponse, "description": "기상청 키 미설정"}})
def asos_monthly(
    year: int = Query(..., ge=2000, le=2100, description="연도"),
    month: int = Query(..., ge=1, le=12, description="월"),
    stn: str = Query("108", description="관측소 번호(108=서울)"),
):
    """기상청 ASOS 일자료를 서버에서 호출해 일별 강수량(mm)을 돌려준다(웹 CORS 회피)."""
    rain = _fetch_asos_monthly(year, month, stn)
    return {"year": year, "month": month, "stn": stn,
            "rain_by_day": rain, "source": "kma_asos"}


@app.get("/api/weather/warning-bulletin", response_model=WarningBulletinResponse,
         tags=["weather"], summary="기상특보 통보문(알림 탭)",
         responses={502: {"model": ErrorResponse, "description": "기상청 호출 실패"},
                    503: {"model": ErrorResponse, "description": "기상청 키 미설정"}})
def warning_bulletin(
    stn: str = Query("109", description="특보구역(109=서울지방기상청)"),
):
    """기상특보 최신 통보문 전문을 서버에서 호출해 돌려준다(지역 매칭은 클라이언트가 수행)."""
    bulletin, tm_fc = _fetch_warning_bulletin(stn)
    return {"bulletin": bulletin, "tm_fc": tm_fc,
            "has_warning": bool(bulletin), "source": "kma_wrn"}


@app.get("/api/dongs", tags=["meta"], summary="커버리지 동 목록",
         description="예측 가능한 93개 법정동 (adm_cd, 구, 동) 목록 — 클라이언트 자동완성/선택 UI용")
def dongs():
    return [{"adm_cd": int(k), "gu": v["gu"], "dong": v.get("dong_label")}
            for k, v in sorted(TABLES.items())]
