# 서울 침수 위험 예측 (doubled_seven)

유저 **주소 → 법정동** + **실시간 예보 강우** 를 입력받아 해당 동의 **침수 확률**을 반환하는 ML 서비스.
서울 침수흔적도(2023–2024) + 강우 관측소 데이터를 (법정동×날짜) 분류 문제로 전처리·학습하고, FastAPI로 서빙해 Vercel에 배포합니다.

```
data/ ──(build_dataset.py)──▶ build/ 피처 패널 ──(train.py)──▶ build/model_np.json
                                                                      │
                              Flutter 앱 ──HTTP──▶ FastAPI(api/) ◀─────┘  (numpy 추론)
```

---

## 📚 문서 (주제별)

| 문서 | 주제 | 독자 |
|---|---|---|
| **README.md** (이 문서) | 프로젝트 개요·빠른 시작·구조 | 전체 |
| [`DATA.md`](DATA.md) | 데이터 원천·전처리·피처 엔지니어링·근거·한계 | 데이터/ML |
| [`MODEL.md`](MODEL.md) | 모델 선택·학습·보정·내보내기·성능·예측계약 | ML |
| [`INFRA.md`](INFRA.md) | Vercel 배포·GitHub 연동·CI·번들 용량 해법 | 인프라/배포 |
| [`FRONTEND.md`](FRONTEND.md) | **API 계약·payload·SDK 연동·Flutter 예제** | 프론트(Flutter) |
| [`openapi.json`](openapi.json) | 기계 판독 OpenAPI(Swagger) 스펙 | 프론트/툴링 |

---

## 빠른 시작 (로컬)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # 학습/전처리 (서빙은 requirements.txt만)

python build_dataset.py        # data/ -> build/ 피처 패널 생성
python train.py --trials 40    # 학습 -> build/model_np.json, serve_tables.json

uvicorn api.index:app --reload # http://127.0.0.1:8000  (Swagger: /docs)
```

빠른 호출 확인:
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"adm_cd":1135010600,"forecast_daily_rain":[5,40,60,100]}'
# -> { "adm_cd":1135010600, "gu":"노원구", "dong":"중계동", "flood_probability":0.0765 }
```
> 엔드포인트·payload·SDK 연동 전체 스펙은 **[`FRONTEND.md`](FRONTEND.md)**.

---

## 핵심 설계 요약

- **분석 단위 = (법정동 × 날짜)**. 침수흔적도는 양성-only라 (동,날짜) 격자로 음성을 생성해 분류 성립. (`DATA.md`)
- **강우 조인 = 구 단위**(25개 구 모두 구청 관측소 보유), **동 식별 = `ADM_CD`**. 동별 자체 우량계는 불가능(48관측소<424동). (`DATA.md`)
- **모델 = HistGradientBoosting**(불균형 1.1%, 16피처). GNN은 소규모·프록시그래프라 비권장. (`MODEL.md`)
- **서빙 = 순수 numpy**: 학습모델을 평문 JSON으로 내보내 sklearn/scipy 없이 평가 → **Vercel 250MB 한도 회피**. (`INFRA.md`)
- 성능(held-out 2024): **ROC-AUC 0.88**, 강우-only 베이스라인(0.72) 대비 큰 향상. (`MODEL.md`)

---

## 파일 구조
```
build_dataset.py     전처리: data/ -> build/ 피처 패널 (+ DATA.md)
train.py             학습: Optuna + 보정 + numpy 내보내기 (+ MODEL.md)
api/index.py         FastAPI 서빙 (Vercel entrypoint)
api/flood_model.py   순수 numpy 추론기 (sklearn 불필요)
vercel.json          Vercel 함수/라우팅 설정 (+ INFRA.md)
requirements.txt     서빙 = fastapi+numpy   /  requirements-dev.txt = 학습
build/               산출물: model_np.json, serve_tables.json, metrics.json …
openapi.json         OpenAPI(Swagger) 스펙
```

---

## ⚠️ 운영 전 체크 (요약 — 상세는 각 문서)
1. **지오코더**: `api/index.py`의 `geocode_to_admcd()`는 스텁 → 운영 시 VWorld/Kakao 연동(또는 클라이언트가 `adm_cd` 직접 전송). (`FRONTEND.md §2`)
2. **예보 강우**: 클라이언트가 SDK의 시간별 강우를 일강우 배열로 정규화해 전송. (`FRONTEND.md §2`)
3. **커버리지**: 침수 이력 93개 법정동만 예측(그 외 404). (`DATA.md §7`)
4. **확률 해석**: base-rate 보정된 보수적 값 → 임계값/상대위험으로 사용. (`FRONTEND.md §6`)
