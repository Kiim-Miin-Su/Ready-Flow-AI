# MODEL.md — 모델 · 학습 · 예측

> 과제: 이진 분류 `P(flood | 동, 강우, 이력)` · 입력: 주소+예보 · 출력: 침수 확률
> 데이터/피처 정의는 [`DATA.md`](DATA.md), 서버는 [`README.md`](README.md), 배포는 [`INFRA.md`](INFRA.md).
> 재현: `python train.py --trials 40` → `build/model.pkl`, `metrics.json`, `serve_tables.json`

---

## 1. 왜 scikit-learn (GNN 아님)

| 근거 | 수치 | 결론 |
|---|---|---|
| 노드(동) 수 | 93 | 그래프 너무 작음 |
| 엣지 | 동일구 클리크 **프록시** | 진짜 인접 아님 → 메시지패싱 이득 미미 |
| 양성 | 164 (1.1%) | 딥모델 과적합 위험 |
| 피처 | 16개 정형 | **부스팅 트리에 최적** |
| **Vercel 무료** | torch+PyG ≈ 800MB+ > **250MB 한도** | GNN 배포 불가 / sklearn ≈ 경량 적합 |

→ **HistGradientBoostingClassifier** 채택. (GNN 재검토 시점·조건은 `DATA.md §8`.)

---

## 2. 학습 파이프라인 (`train.py`)

```
데이터 분할   train = 2023,  held-out TEST = 2024        (시간분할, 누수 차단)
검증          2023 위에서 StratifiedKFold(5) → Optuna 튜닝
전처리        ColumnTransformer (위치 인덱스 기반 → 서빙 시 pandas 불필요)
                · 연속 15개  → SimpleImputer(median) + StandardScaler   (스케일 차이 보완)
                · 구(gu_code) → OneHotEncoder                           (범주형 정체성)
불균형        balanced sample_weight  (음성/양성 비율)
모델          HistGradientBoostingClassifier
튜닝          Optuna(TPE), 목적함수 = 5-fold 평균 PR-AUC  (불균형엔 ROC보다 PR)
보정          CalibratedClassifierCV(isotonic, cv=3) → 사용 가능한 확률값
```

### 설계 결정 & 근거
- **검증을 시간분할이 아닌 Stratified CV로**: 침수는 여름에 집중·희소 → 2023 내부를 날짜로 자르면 검증 fold 양성이 0~1개가 되어 지표가 degenerate(PR-AUC=1.0). **계층 K-fold**가 모든 fold에 양성을 유지. 일반화의 최종 판정은 **2024 완전 홀드아웃**으로 별도 수행(연-단위 시간분할 → 누수 없음).
- **스케일링**: 강우(0~400mm)·횟수(0~10)·비율(0~1)·구코드(~1.1만) 스케일 상이. 트리는 스케일 불변이지만, 파이프라인을 **모델 교체(로지스틱 등)에 견고**하게 만들고 OneHot과 일관되게 처리하려 `StandardScaler` 포함.
- **PR-AUC 목적함수**: 1.1% 불균형에서 ROC-AUC는 낙관적 → 튜닝·모델선택은 **PR-AUC** 기준.
- **확률 보정**: balanced weight로 학습한 raw score는 빈도와 괴리(폭우인데 P≈0.02) → **isotonic 보정**으로 의사결정 가능한 확률로 변환(아래 효과 확인).
- **위치 인덱스 ColumnTransformer**: 문자열 컬럼명으로 학습하면 추론 시 DataFrame을 요구 → 서빙에 pandas(~70MB) 강제. 인덱스 기반으로 바꿔 **numpy만으로 추론**(Vercel 용량 절감).

---

## 3. 성능 (held-out 2024)

| | PR-AUC | ROC-AUC | R@20 | R@50 |
|---|---|---|---|---|
| **TEST 2024 (보정)** | **0.109** | **0.910** | 0.066 | 0.082 |
| TRAIN 2023 | ~0.39 | ~0.98 | — | — |
| Optuna 검증(2023 5-fold) | 0.297 | — | — | — |
| 베이스라인 `rain_7d` 단독 | — | 0.723 | — | — |

(정확한 최신 수치는 `build/metrics.json`.)

- **ROC-AUC 0.91** (강우-only 0.72 대비 **+0.19**). base-rate ~0.86% 대비 PR-AUC 0.109 = **~13× lift**.
- train(~0.98) vs test(0.91) 격차 = 약한 과적합. 2년치·양성 164개 한계 → 정규화(`l2`, `min_samples_leaf↑`, 얕은 깊이)로 억제. 데이터 누적 시 개선 여지.
- **보정 효과**(중계동 예시): 폭우(100mm/3d) 미보정 P≈0.02 → 보정 **P≈0.78**, 건조 P≈0.0. 단조·해석 가능.

> 얕고 강한 정규화(얕은 깊이·작은 lr·높은 min_samples_leaf)가 소량·불균형 데이터에 타당. 베스트 파라미터는 `metrics.json`.

---

## 4. 예측 (서빙) 계약

학습-서빙 피처 순서 동일(`FEATURES`). 추론 입력은 **주소→동**, **예보→강우 윈도우**만으로 구성:

```
주소 ──geocode──▶ adm_cd ──▶ serve_tables.json(이력·이웃·구코드 스냅샷)
예보 일강우 ─────▶ rain_1d/3d/7d/14d/30d, rain_ante7, (강도 프록시)
            └────▶ np.array([FEATURES 순서]) ──▶ model.predict_proba ──▶ P(flood)
```

- **이력 피처는 추론 시 재계산하지 않고** 동별 최신 스냅샷(`serve_tables.json`)에서 조회 → 빠르고 누수 없음.
- 상세 페이로드·엔드포인트·주의사항 → `README.md`.

---

## 5. 한계 & 다음 단계
- **2년치**: `prev_year_*`는 2024에만 실효(2022 부재). 연도 누적이 최우선 개선.
- **양성 희소(164)**: 절대확률 보수적, R@k 낮음. 임계값은 운영 데이터로 재튜닝.
- **지형 피처 미반영**: DEM/하천/배수 외부데이터 추가 시 음성 동까지 일반화·정확도 향상 기대(`DATA.md §6`).
- **모델 후보 확장**: LightGBM/XGBoost, 로지스틱(보정·해석용) 비교. 파이프라인은 교체 가능하게 구성.
- **평가 지표**: 운영에선 PR-AUC·Recall@경보예산, 그리고 **공간·시간 분리 CV**(동 그룹 누수 점검) 추가 권장.
