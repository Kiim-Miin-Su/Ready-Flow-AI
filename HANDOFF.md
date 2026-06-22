# HANDOFF.md — 침수 모델 작업 인수인계

> 작성: 2026-06-23 · 작업자: AI 어시스턴트(위임 세션) · 다음 담당자/세션용
> 한 곳에서 "지금 상태 / 한 일 / 재현법 / 다음 할 일 / 함정"을 본다.

---

## 1. 지금 상태 (TL;DR)

- **배포됨.** commit `6d18570` push 완료 → Vercel 자동 배포, `GET /api/health` 200 OK 확인.
- 모델에 **monotonic 강우 제약** 적용 + 재학습 완료. 비단조/사강 결함 해결, 메트릭 개선.
- 근본원인 진단 + 개선 로드맵 문서화 완료 → [DIAGNOSIS.md](DIAGNOSIS.md).
- **상대 위험등급 `risk_level`/`risk_percentile` API 추가**(DIAGNOSIS §7-④, 비파괴). 보정확률이 폭우일에도 낮아 절대%가 오해를 주는 문제를, 학습분포 백분위 기반 등급으로 해결. 프론트(home_protector)도 이 등급을 우선 사용하도록 수정(폴백 포함). ⚠️ 프론트는 `flutter analyze`로 확인 필요(샌드박스에 dart 없음).
- **미배포 로컬 커밋이 있을 수 있음**(문서/도구/진단/risk_level). push 전 리뷰 권장.

## 2. 이번 세션에서 한 일

1. `train.py`에 `monotonic_cst` (강우 9개 피처 단조 비감소) — *코드는 이전 단계에서 작성돼 있었고*, 본 세션은 **재학습 실행**으로 산출물(`build/model_np.json`,`metrics.json`)을 갱신.
2. 검증: monotonicity 단위테스트(320케이스 위반 0), PREDICTION.md A~E 프로브 재현, numpy↔sklearn 비트 일치(0.0), FastAPI 스모크.
3. `build/model.pkl` 추적 해제(.gitignore 설계대로, 미배포·재생성 가능), `.claude/` ignore.
4. 진단·문서·도구 추가: `DIAGNOSIS.md`, `tools/diagnose.py`, `tools/validate_monotonicity.py`, `tools/dong_terrain.schema.csv`, 본 `HANDOFF.md`.

### 성능(배포 numpy 경로, 홀드아웃 2024)
| | old | new |
|---|---|---|
| PR-AUC | 0.0744 | **0.149** |
| ROC-AUC | 0.879 | **0.890** |
| monotonicity 위반 | (있음) | **0/320** |

## 3. 재현 방법

```bash
# 권장: 레포 핀 환경 (requirements-dev.txt: py3.12, sklearn 1.9, numpy 2.5, pandas 3.0)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py --trials 40           # build/model_np.json, metrics.json, serve_tables.json
python tools/validate_monotonicity.py # 단조성 회귀 테스트 (exit 0 = PASS)
python tools/diagnose.py              # 데이터/피처 진단 근거 재현
```

### ⚠️ 환경 정합성 주의 (중요)
본 세션 재학습은 샌드박스 제약(파이썬 3.12 다운로드 불가)으로 **Python 3.10 / scikit-learn 1.6.1 / numpy 2.2.6 / pandas 2.2.3**, **Optuna 20 trials**로 수행함. 레포 핀과 다르다.
- 서빙 산출물 `model_np.json`은 **평문 numpy 배열**이라 sklearn 버전과 무관 → 현재 배포는 안전(비트일치 확인).
- 그러나 **핀 환경(3.12/1.9.0)에서 40 trials로 재학습**해 정합성을 재확인하고 다시 커밋하는 것을 권장. (best_params·트리 구조가 미세하게 달라질 수 있음.)

## 4. 다음 할 일 (우선순위 — 근거는 DIAGNOSIS.md)

1. 🥇 **지형 피처 통합**(공간 변별 최대 레버리지). `tools/dong_terrain.schema.csv` 스키마대로 `data/dong_terrain_external.csv` 채우기 → `build_dataset.py`에 머지 훅 추가 → `train.py` FEATURES에 지형 컬럼 편입(monotonic은 강우에만). 데이터원: 국토지리정보원 DEM, 서울 GIS, 행안부 침수위험지구, 통계청 반지하.
2. 🥈 **구→동 강우 해상도**: 추가 게이지/기상청 격자·레이더.
3. 🥉 **inert 이력 피처 정리/교체**(corr≈0; §3). FEATURES 변경 시 §6 동기화 체크리스트 준수.
4. **확률→위험등급** 추가 응답 필드(`risk_level`, 비파괴).
5. **극단·과거(2020~2022, 8·8 포함) 데이터 보강**; 합성증강은 OFF-by-default 플래그 + 합성라벨.
6. **CI에 monotonicity 테스트** 추가(`tools/validate_monotonicity.py`, exit code 활용).

## 5. 환경/배포 메모
- 배포: GitHub `main` push → Vercel 자동(INFRA.md). 서빙 의존성은 fastapi+numpy뿐, 모델은 numpy 평문(JSON). 250MB 한도 회피 설계.
- 커밋 필수 산출물: `build/model_np.json`, `build/serve_tables.json`, `build/metrics.json`.
- 라이브 POST 점검(샌드박스에선 vercel POST 불가했음, 맥/브라우저에서):
  ```bash
  curl -X POST https://ready-flow-ai.vercel.app/api/predict \
    -H 'Content-Type: application/json' \
    -d '{"adm_cd":1162010200,"forecast_daily_rain":[10,30,80,180,381]}'
  # 새 모델이면 flood_probability ≈ 0.0548 (구 모델은 0.0765)
  ```

## 6. ⚠️ 함정 / FEATURES 변경 시 동기화 체크리스트
피처 목록을 바꾸면 아래를 **모두** 맞춰야 서빙이 안 깨진다:
- `train.py` `CONT`/`CAT`/`N_RAIN`/`monotone_cst()` (monotonic 벡터 길이)
- `api/flood_model.py` `FEATURES`, `N_CONT`
- `api/index.py` `rain_windows()` 출력 키, `HIST_KEYS`, `serve_tables` 생성부(train.py 하단)
- 재학습 후 `tools/validate_monotonicity.py`로 회귀 확인.

## 7. 정리(cleanup) 필요 — 작업자가 남긴 임시물
- **레포 외부(`/Users/kiim/workspaces/programming/`)** 에 학습용 임시 venv/cache 생성됨(약 640MB). FUSE 권한으로 샌드박스에서 삭제 불가 → 맥에서 수동 삭제:
  ```bash
  rm -rf /Users/kiim/workspaces/programming/{.sbenv,.uv,.sblibs,.sbvenv}
  ```
- `doubled_seven/.git/` 에 0바이트 잠금파일이 남았을 수 있음(`index.lock` 등). git이 불평하면:
  ```bash
  rm -f .git/index.lock .git/HEAD.lock .git/objects/maintenance.lock
  ```

## 8. AI 세션 컨텍스트(다음 AI 세션이 빠르게 이어받도록)
- 두 레포: `doubled_seven`(모델/FastAPI, 배포 ready-flow-ai.vercel.app) ↔ `home_protector`(Flutter 프론트). PREDICTION.md는 둘 다에 있음(프론트가 모델에 준 피드백).
- 핵심 사실: **모델 신호는 강우뿐**, 공간/이력 피처는 inert, 지형 피처는 미수집(NaN). "확률 낮음"은 대체로 올바른 보정(폭우일 실제 침수율 13%).
- 샌드박스 제약: github **release 다운로드**·**vercel POST**는 네트워크 허용목록 밖(단, git 프로토콜·PyPI·web_fetch GET은 가능). 백그라운드 프로세스는 호출 간 비유지, bash 호출 45s 상한, 마운트 폴더는 FUSE라 일부 unlink 불가.
