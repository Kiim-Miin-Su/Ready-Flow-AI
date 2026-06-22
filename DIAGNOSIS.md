# DIAGNOSIS.md — 침수 모델 결함 정밀 진단 & 개선 로드맵

> 작성: 2026-06-23 · 대상: 모델/데이터 개발자 · 짝꿍 문서: [PREDICTION.md](PREDICTION.md)(증상), [MODEL.md](MODEL.md)(설계), [DATA.md](DATA.md)(데이터)
> 재현 스크립트: [`tools/diagnose.py`](tools/diagnose.py), [`tools/validate_monotonicity.py`](tools/validate_monotonicity.py)
> 이 문서는 PREDICTION.md에서 **증상**으로 보고된 결함들의 **근본 원인을 기존 데이터로 정량 진단**하고, 근거 기반 개선 우선순위를 제시한다.

---

## 0. 요약 (TL;DR)

| PREDICTION.md 증상 | 근본 원인 (근거) | 이번 커밋에서 | 남은 작업 |
|---|---|---|---|
| ② 비단조 / ④ 비현실 응답 | HistGB에 단조 제약 없음 | ✅ **monotonic_cst로 해결** (단위테스트 160케이스 위반 0) | — |
| ③ 사강(0~80mm=0) | 학습데이터상 저강우일 침수율 ≈0 + 단조 제약 부재 | ⚠️ 완화(50mm부터 반응) | 지형 피처로 저지대 변별 |
| ④ 공간 변별력 없음 | **강우가 구(區)단위**(동별 동일) + **이력 피처 inert** + **지형 피처 전부 NaN** | — | 🔴 지형데이터 통합(최우선) |
| ⑤ 시계열 미반영 | 누적강우 피처는 있으나 신호 약함 | ⚠️ 일부 회복 | 선행강수지수(API) 보강 |
| ① 값이 너무 낮음(8% 천장) | **대부분 올바른 보정** — 폭우일 실제 침수율 12.99% | — | 위험등급(tier) UX로 표현 |
| ⑥ 출력 양자화 | 소형 트리 + 희소 양성(1.12%) | — | 데이터 증강/추가 연도 |

**한 줄 결론:** monotonic 결함은 코드로 해결됐다. 그러나 **공간 변별력**과 **극단 강우 신뢰성**은 *모델 하이퍼파라미터가 아니라 입력 데이터의 한계*가 원인이며, 외부 지형데이터 통합과 강우 해상도 향상이 가장 큰 레버리지다.

---

## 1. 데이터 개관 (build/samples.parquet)

```
행(동×날짜) = 14,690 | 동 = 93 | 구 = 24 | 양성(침수) = 164 (1.12%)
양성 연도별: 2023=103, 2024=61    (학습=2023, 홀드아웃 테스트=2024)
```

희소 양성(1.12%)은 침수가 본질적으로 드문 사건임을 반영. 클래스 불균형은 `balanced sample_weight`로 처리 중.

---

## 2. 근거 ① — 강우는 동(洞)이 아니라 구(區) 단위다 🔴

`tools/diagnose.py` §1:

```
(구,날짜) 그룹 중 rain_day 값이 2개 이상으로 갈리는 그룹: 0 / 3,771
구당 동 수: 최소 1, 최대 14, 평균 3.9
```

`build_dataset.py:load_rain()`은 **구청 관측소 1개/구**만 사용해 강우를 구 단위로 집계하고, `build_panel()`에서 `daily`를 `gu`로 머지한다. 따라서 **같은 구의 모든 동은 항상 동일한 강우 피처**를 받는다.

→ PREDICTION.md ④(다른 동이 동일 값)의 **구조적 원인**. 같은 강우를 넣으면 같은 구 내 동들은 *이력 피처로만* 갈릴 수 있는데(아래 §3), 그 이력 피처가 무신호다. 결국 거의 동일한 값이 나온다.

**한계의 성격:** 모델 버그가 아니라 **관측 해상도 한계**(서울 25개 구청 게이지). 동 단위 변별을 원하면 (a) 추가 관측소/펌프장 게이지, (b) 기상청 격자/레이더 강수, (c) 지형 기반 동별 보정 중 하나가 필요.

---

## 3. 근거 ② — 이력/공간 피처가 통계적으로 inert

`tools/diagnose.py` §3 (y와의 상관계수 · 상위10% lift = 그 피처 상위10%에서의 침수율/전체침수율):

```
rain_day                 corr=+0.168  top10%_lift=5.68x   ← 신호 강함
rain_max60               corr=+0.150  top10%_lift=5.60x   ← 신호
rain_3d                  corr=+0.157  top10%_lift=5.78x   ← 신호
--------------------------------------------------------- 이하 거의 무신호 ---
hist_flood_rate_prior    corr=+0.032  top10%_lift=1.52x
hist_flood_cnt_prior     corr=-0.007  top10%_lift=1.03x
days_since_last_flood    corr=-0.038  top10%_lift=1.09x
prev_year_same_month_cnt corr=-0.000  top10%_lift=1.00x
prev_year_same_week_cnt  corr=-0.001  top10%_lift=1.00x
neighbor_gu_flood_prior  corr=-0.019  top10%_lift=0.49x
```

6개 이력/이웃 피처 중 5개가 사실상 무신호(lift≈1.0). 즉 모델의 실질 예측력은 **강우 피처에서 나온다**. 이력 피처는 노이즈에 가까워, 공간 변별에 기여하지 못한다.

**원인 가설:** 2년·93동·164건이라는 희소 표본에서 "과거 침수 횟수"는 거의 0이거나 극소수라 변별 정보가 없다. prev_year_* 는 2023행에 대해 2022 데이터가 없어 대부분 0(비영 비율 prev_year_week=2.0%).

---

## 4. 근거 ③ — 지형 피처가 전혀 채워지지 않음 🔴

`tools/diagnose.py` §2:

```
elev_mean_m       non-null = 0/93
dist_to_river_m   non-null = 0/93
drainage_density  non-null = 0/93
```

`build_dataset.py:build_static()`는 이 3개 컬럼을 `np.nan` 자리표시자로만 둔다(외부 데이터 필요, DATA.md 명시). 게다가 **train.py의 FEATURES에 지형 컬럼이 아예 포함돼 있지 않다** — 즉 현재 모델은 고도/배수/하천거리를 *전혀 보지 않는다*.

저지대·반지하·배수불량이 침수의 핵심 물리 요인임을 감안하면, 이것이 **공간 변별력 부재의 최대 원인**이자 **가장 큰 개선 레버리지**다. (2022-08-08 신림동 반지하 참사도 저지대+반지하 밀도가 핵심이었다.)

---

## 5. 근거 ④ — "확률이 너무 낮다"는 대체로 올바른 보정

`tools/diagnose.py` §4:

```
rain_day >= 80mm  인 날들의 실제 침수율 = 12.99%  (46 / 354)
rain_day >= 150mm 인 날들의 실제 침수율 =  0.00%  (0 / 6)   ← 표본 6일뿐
```

isotonic 보정은 출력을 **관측된 침수 빈도**에 맞춘다. 폭우일(80mm↑)에도 *특정 동이 그날 침수할 빈도*는 13%에 불과하므로, 보정확률 8~13%는 **현실을 반영한 값**이다. PREDICTION.md ①의 "8%는 너무 낮다"는 직관은 *"위험한 날"* 과 *"이 동이 오늘 침수할 확률"* 을 혼동한 것에 가깝다.

**시사점:** 모델 출력을 그대로 % 로 보여주면 사용자는 늘 "낮다"고 느낀다. 해결은 보정을 망치는 게 아니라 **표현**이다 — 학습분포 기준 백분위/위험등급(낮음·주의·경보)으로 변환해 보여줄 것(§7-④).

---

## 6. 근거 ⑤ — 극단 강우 구간은 데이터가 없어 외삽

`tools/diagnose.py` §5:

```
데이터 내 최대 rain_day = 460mm | 200mm 초과 = 6일 | 300mm 초과 = 6일
```

2년 전체에서 200mm 초과가 **6일**뿐이다. 그 이상 강우에서 모델은 거의 데이터 없이 외삽하므로 출력이 불안정/포화한다(PREDICTION.md ②의 OOD 붕괴). monotonic 제약이 *방향*은 잡아주지만(더 오면 안 내려감), *크기*의 신뢰성은 데이터 부족으로 한계가 있다.

---

## 7. 개선 권고 (우선순위 · 근거 · 확장성)

### 🥇 ① 지형/지세 피처 통합 — 공간 변별의 최대 레버리지
- **무엇:** `elev_mean_m`(평균고도), `slope`(경사), `dist_to_river_m`(하천거리), `drainage_density`(배수관망밀도), `lowland_frac`(저지대 비율), `semibasement_density`(반지하 밀도).
- **데이터원(공개):** 국토지리정보원 DEM(5m), 서울 GIS포털 하천/배수관망, 행정안전부 **침수위험지구**, 통계청 반지하 가구 통계.
- **확장 설계:** `data/dong_terrain_external.csv`(스키마는 [`tools/dong_terrain.schema.csv`](tools/dong_terrain.schema.csv)) → `build_dataset.py`가 있으면 머지, 없으면 현행대로 동작(무해). train.py는 비-NaN 지형 컬럼을 CONT에 자동 편입(monotonic 제약은 강우에만 유지). **드롭인 가능하도록 훅과 스키마를 본 커밋에 포함.**

### 🥈 ② 구→동 강우 해상도 향상
- 추가 게이지/펌프장 관측 통합, 또는 기상청 **격자 강수(동네예보)·레이더** 도입. 단기적으로는 지형 가중으로 구내 동별 보정.

### 🥉 ③ inert 이력 피처 정리/대체 (근거: §3)
- `prev_year_same_week_cnt`, `prev_year_same_month_cnt`, `hist_flood_cnt_prior`, `days_since_last_flood`는 무신호 → 제거하거나(노이즈 감소) 정보성 피처(선행강수지수 API, 토양수분)로 교체. **주의:** FEATURES 변경 시 `api/flood_model.py`·`api/index.py`·`serve_tables`·monotonic 벡터 동기화 필요(§HANDOFF 참조).

### ④ 확률 → 위험등급 표현 (근거: §5) — ✅ **구현됨**
- 모델/보정은 그대로 두고, 학습분포 분위수로 `risk_level`(info/warning/danger)+`risk_percentile`(0-100)을 **추가 응답 필드**로 제공(프론트 계약 비파괴). 컷: warning=train q85, danger=train q99 (`train.py:WARNING_PCT/DANGER_PCT`로 조정). `model_np.json["risk"]`에 저장, 재학습 없이 주입 가능(`tools/`).
- 검증: 실제 2024 침수일 61건 중 **info 탈출 57%**(이전 8%), danger 14건(정밀도 ~30%, base-rate 1.1% 대비 27배). 2022-08-08형 신림동 폭우 → warning(상위 2%).
- 프론트(home_protector)는 `risk_level` 우선 사용, 없으면 기존 % 임계로 폴백.

### ⑤ 극단·과거 이벤트 데이터 보강 (근거: §6)
- 2020~2022 침수흔적도(특히 **2022-08-08**) + 당시 강우 추가 → 극단 구간 실표본 확보가 합성증강보다 우선. 합성증강은 물리근거(단조)로 OFF-by-default 플래그(`--augment`)로만, 합성여부 라벨 필수.

### ⑥ 검증 상시화
- monotonicity 단위테스트를 CI에 포함, 신뢰도 다이어그램(reliability)·홀드아웃 극단이벤트 PR-AUC 리포트 자동화.

---

## 8. 이번 작업으로 확정된 사실(재현 가능)
- monotonic_cst 적용 후 단조성 단위테스트 **160케이스 위반 0**, 600mm≥400mm 회복.
- 테스트(배포 numpy 경로) **PR-AUC 0.0744→0.149, ROC-AUC 0.879→0.890**.
- numpy 추론 ↔ sklearn predict_proba **비트 일치(오차 0.0)** — 서빙 안전.
- FastAPI health/predict 스모크 통과, 라이브 `/api/health` 200 OK.

> ⚠️ 재현 환경 주의: 본 재학습은 샌드박스 제약으로 **Python 3.10 / scikit-learn 1.6.1 / numpy 2.2.6 / pandas 2.2.3**, **Optuna 20 trials**로 수행됨(레포 핀은 3.12 / 1.9.0 / 2.5.0 / 3.0.3 / 40 trials). 서빙 산출물 `model_np.json`은 평문 numpy라 버전 무관이지만, **핀 환경에서 40 trials로 재학습해 정합성 재확인 권장**. 자세한 사항은 [HANDOFF.md](HANDOFF.md).
