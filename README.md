# 서울 침수 위험 예측 API

유저 **주소** + **실시간 예보 강우**를 입력받아 해당 법정동의 **침수 확률**을 반환하는 FastAPI 서비스.

- 전처리/데이터 설계 → [`DATA.md`](DATA.md)
- 모델/학습/평가 → [`MODEL.md`](MODEL.md)
- 배포(Vercel) → [`INFRA.md`](INFRA.md)

---

## 빠른 시작 (로컬)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # 학습/전처리 포함

python build_dataset.py        # data/ -> build/ 피처 패널 생성
python train.py --trials 40    # 모델 학습 -> build/model.pkl, serve_tables.json

uvicorn api.index:app --reload # http://127.0.0.1:8000
```

---

## 엔드포인트

### `GET /api/health`
헬스체크. 로드된 동 수·피처 수 반환.
```json
{ "status": "ok", "dongs": 93, "features": 16 }
```

### `POST /api/predict`
침수 확률 예측.

**Request body** (`application/json`)

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `address` | string | ✅* | 유저 주소. 지오코딩→법정동. (*`adm_cd` 직접 제공 시 생략 가능) |
| `forecast_daily_rain` | number[] | ✅ | 일강우(mm) 시퀀스, **과거→오늘**(오늘이 마지막). 최대 30개. 누적/선행 강우 윈도우 계산용 |
| `adm_cd` | int | ❌ | 10자리 법정동코드 직접 지정(지오코딩 건너뜀) |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"address":"서울 노원구 중계동 23-28","forecast_daily_rain":[5,40,60,100]}'
```

**Response 200**
```json
{ "adm_cd": 1135010600, "gu": "노원구", "dong": "중계동", "flood_probability": 0.34 }
```

**오류**
| 코드 | 의미 |
|---|---|
| `404 dong not resolved` | 주소에서 법정동을 못 찾음(지오코더 미연동/범위 밖) |
| `404 adm_cd not in coverage` | 학습 커버리지(침수 이력 93개 동) 밖 |
| `422` | 페이로드 검증 실패(pydantic) |

---

## 입력 → 피처 변환 (서버 내부)

| 단계 | 처리 | 출력 |
|---|---|---|
| 1. 주소 | 지오코딩 | `adm_cd`(법정동) |
| 2. 동 조회 | `serve_tables.json` | 이력/이웃/구코드 정적 피처 |
| 3. 예보 | 누적합 | `rain_1d/3d/7d/14d/30d`, `rain_ante7`, 강도 프록시 |
| 4. 조립 | 학습 피처 순서로 정렬 | `model.predict_proba` → 확률 |

> `serve_tables.json` = 동별 **최신 이력 스냅샷**(학습 시 내보냄). 이력 피처는 추론 시 재계산하지 않고 조회.

---

## ⚠️ 운영 전 반드시 연동/확인할 것

1. **지오코더**: `api/index.py`의 `geocode_to_admcd()`는 **스텁**(동 라벨 부분일치)임. 운영 시 **VWorld/Kakao 주소→법정동코드 API**로 교체.
2. **강우 강도**: 예보가 일강우만 줄 경우 `rain_max10/60`은 근사치(=일강우/6, /2). 기상청 단기예보·초단기실황의 시간당 강우가 있으면 교체하면 정확도↑.
3. **커버리지**: 침수 이력이 있는 **93개 법정동**만 예측. 그 외 동은 404 → 향후 전체 법정동 마스터·지형 피처로 확장(`DATA.md §7`).
4. **모델 버전**: `model.pkl`은 `scikit-learn==1.9.0`으로 피클됨. 서버 `requirements.txt`와 버전 일치 필수.
5. **확률 보정**: isotonic 보정 적용했으나 양성 희소(2년치)로 절대확률은 보수적. 의사결정 임계값은 운영 데이터로 재튜닝 권장.

---

## 파일 구조
```
build_dataset.py     전처리: data/ -> build/ 피처 패널
train.py             학습: Optuna + 보정 -> build/model.pkl, serve_tables.json
api/index.py         FastAPI 서빙 (Vercel entrypoint, pandas-free)
vercel.json          Vercel 함수/라우팅 설정
requirements.txt     서빙 의존성(경량)   /  requirements-dev.txt 학습 의존성
build/               산출물(model.pkl, serve_tables.json 등)
DATA.md MODEL.md INFRA.md
```
