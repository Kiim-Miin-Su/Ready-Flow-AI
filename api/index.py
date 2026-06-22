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
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
        "- 커버리지: 침수 이력이 있는 95개 법정동 (그 외 404)\n\n"
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

    model_config = {"json_schema_extra": {"examples": [
        {"address": "서울 노원구 중계동 23-28", "forecast_daily_rain": [5, 40, 60, 100]},
        {"address": "", "adm_cd": 1135010600, "forecast_daily_rain": [10, 30, 80]},
    ]}}


class PredictResponse(BaseModel):
    adm_cd: int = Field(..., examples=[1135010600], description="법정동코드")
    gu: str = Field(..., examples=["노원구"])
    dong: str | None = Field(None, examples=["중계동"], description="동 라벨(없을 수 있음)")
    flood_probability: float = Field(..., ge=0, le=1, examples=[0.7473],
                                     description="침수 확률 [0,1]")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    dongs: int = Field(..., examples=[95], description="커버리지 동 수")
    features: int = Field(..., examples=[16])


class ErrorResponse(BaseModel):
    detail: str


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
    if adm not in TABLES:
        raise HTTPException(404, f"adm_cd {adm} not in coverage (95 flood-prone dongs)")
    info = TABLES[adm]
    feat = rain_windows(req.forecast_daily_rain)
    feat.update({k: info[k] for k in HIST_KEYS})
    prob = fm.predict_proba(MODEL, feat)
    return {"adm_cd": int(adm), "gu": info["gu"], "dong": info.get("dong_label"),
            "flood_probability": round(prob, 4)}


@app.get("/api/dongs", tags=["meta"], summary="커버리지 동 목록",
         description="예측 가능한 95개 법정동 (adm_cd, 구, 동) 목록 — 클라이언트 자동완성/선택 UI용")
def dongs():
    return [{"adm_cd": int(k), "gu": v["gu"], "dong": v.get("dong_label")}
            for k, v in sorted(TABLES.items())]
