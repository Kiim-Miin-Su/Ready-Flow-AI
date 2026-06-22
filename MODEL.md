# MODEL.md — 모델 · 학습 · 예측

> 과제: 이진 분류 `P(flood | 동, 강우, 이력)` · 입력: 주소+예보 · 출력: 침수 확률
> 데이터/피처 정의는 [`DATA.md`](DATA.md), 서버는 [`README.md`](README.md), 배포는 [`INFRA.md`](INFRA.md).
> 재현: `python train.py --trials 40` → `build/model_np.json`(서빙), `metrics.json`, `serve_tables.json`

---

## 1. 왜 scikit-learn (GNN 아님)

| 근거 | 수치 | 결론 |
|---|---|---|
| 노드(동) 수 | 93 | 그래프 너무 작음 |
| 엣지 | 동일구 클리크 **프록시** | 진짜 인접 아님 → 메시지패싱 이득 미미 |
| 양성 | 164 (1.1%) | 딥모델 과적합 위험 |
| 피처 | 16개 정형 | **부스팅 트리에 최적** |
| **Vercel 무료** | torch+PyG ≈ 800MB+ > **250MB 한도** | GNN 배포 불가 |

→ **HistGradientBoostingClassifier** 채택. (GNN 재검토 시점·조건은 `DATA.md §8`.)
> 단, **scikit-learn 런타임도 250MB 초과**(scipy ~130MB) → 학습은 sklearn, **서빙은 모델을 평문 JSON으로 내보내 numpy로만 평가**(§2 배포 내보내기, `INFRA.md §2`).

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
보정          OOF(cross_val_predict) 점수에 IsotonicRegression 1개 적합
배포 내보내기  단일 base 파이프라인 + isotonic → 평문 JSON(model_np.json):
                전처리(median/scale/onehot)·HistGB 트리·iso 임계값을 배열로 직렬화
                → 서빙은 numpy만으로 평가(api/flood_model.py). sklearn/scipy 불필요.
```

### 설계 결정 & 근거
- **검증을 시간분할이 아닌 Stratified CV로**: 침수는 여름에 집중·희소 → 2023 내부를 날짜로 자르면 검증 fold 양성이 0~1개가 되어 지표가 degenerate(PR-AUC=1.0). **계층 K-fold**가 모든 fold에 양성을 유지. 일반화의 최종 판정은 **2024 완전 홀드아웃**으로 별도 수행(연-단위 시간분할 → 누수 없음).
- **스케일링**: 강우(0~400mm)·횟수(0~10)·비율(0~1)·구코드(~1.1만) 스케일 상이. 트리는 스케일 불변이지만, 파이프라인을 **모델 교체(로지스틱 등)에 견고**하게 만들고 OneHot과 일관되게 처리하려 `StandardScaler` 포함.
- **PR-AUC 목적함수**: 1.1% 불균형에서 ROC-AUC는 낙관적 → 튜닝·모델선택은 **PR-AUC** 기준.
- **확률 보정**: balanced weight로 학습한 raw score는 빈도와 괴리(폭우인데 P≈0.02) → **isotonic 보정**으로 의사결정 가능한 확률로 변환(아래 효과 확인).
- **위치 인덱스 ColumnTransformer**: 문자열 컬럼명으로 학습하면 추론 시 DataFrame을 요구 → pandas 강제. 인덱스 기반으로 바꿔 numpy 배열만으로 전처리 재현(JSON 내보내기의 전제).
- **ONNX 대신 numpy 직접 평가**: `skl2onnx`+최신 `onnx`가 HistGB의 NaN-분기 bool 속성에서 변환 실패(py3.12). 트리 노드(`feature_idx/threshold/left/right/value`)를 직접 순회하는 게 더 견고하고, sklearn과 비트 동일.

---

## 3. 성능 (held-out 2024)

| | PR-AUC | ROC-AUC |
|---|---|---|
| **TEST 2024 — 배포 모델(numpy, 단일 base + OOF-iso)** | **0.074** | **0.879** |
| TEST 2024 — 참조(sklearn cv=3 보정, 미배포) | 0.109 | 0.910 |
| Optuna 검증(2023 5-fold) | 0.297 | — |
| 베이스라인 `rain_7d` 단독 | — | 0.723 |

(정확한 최신 수치는 `build/metrics.json`.)

- **배포 모델 ROC 0.879** (강우-only 0.72 대비 **+0.16**). 단일 base(전체 2023 적합) + OOF-isotonic 1개라 **numpy로 정확히 내보내기 가능**(`model_np.json`). sklearn cv=3 평균 보정(0.910)보다 살짝 낮지만 배포 가능성과 맞바꿈.
- **비트 단위 동일성**: numpy 추론기 vs sklearn `predict_proba` 최대 오차 **0.0** (train.py가 매 학습마다 검증).
- train(~0.98) vs test(0.88) 격차 = 약한 과적합. 2년치·양성 164개 한계 → 얕은 깊이·강한 정규화로 억제.
- **보정 해석**: OOF-iso는 base 점수를 실제 빈도로 매핑 → 폭우라도 P가 보수적(해당 동 침수빈도 수준). 랭킹(ROC) 보존이 핵심, 절대값은 임계값으로 사용(`FRONTEND.md §5`).

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
