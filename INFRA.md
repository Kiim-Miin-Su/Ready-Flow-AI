# INFRA.md — 배포 (Vercel)

FastAPI 침수 예측 API를 **Vercel Python 서버리스**로 배포. GitHub 연동 시 push마다 자동 배포.

- API 사용법 → [`README.md`](README.md) · 모델 → [`MODEL.md`](MODEL.md)

---

## 1. 구성

```
api/index.py            ASGI(FastAPI) 엔트리포인트  — Vercel이 `app` 자동 인식
api/flood_model.py      순수 numpy 추론기 (sklearn 불필요) — index.py와 같은 디렉터리라 자동 번들
vercel.json             런타임/메모리/라우팅/포함파일
requirements.txt        서빙 의존성 = fastapi + numpy 뿐
build/model_np.json        학습모델을 평문 배열로 내보낸 JSON  ┐ includeFiles로 함수 번들에 포함
build/serve_tables.json    동별 이력 스냅샷                    ┘ (git에 커밋 필요)
```

`vercel.json` 핵심:
```json
{
  "functions": { "api/index.py": {
      "runtime": "@vercel/python@4.3.0",
      "memory": 1024, "maxDuration": 10,
      "includeFiles": "build/{model_np.json,serve_tables.json}" } },
  "rewrites": [ { "source": "/(.*)", "destination": "/api/index" } ]
}
```
- **rewrites**: 모든 경로를 단일 ASGI 앱으로 보냄 → 앱 내부 라우팅(`/api/health`, `/api/predict`)이 처리.
- **includeFiles**: `model_np.json`·`serve_tables.json`을 함수 번들에 동봉(런타임에서 `build/`로 읽음).

---

## 2. 번들 용량 (Vercel Hobby 한도 = 250MB 압축해제)

> ⚠️ **`scikit-learn`을 서빙에 넣으면 250MB를 초과**합니다(scipy 단독 ~130MB + numpy + sklearn).
> 실제 배포에서 `A Serverless Function has exceeded the unzipped maximum size of 250 MB` 발생.
> **해결: 모델을 ONNX가 아닌 평문 JSON(`model_np.json`)으로 내보내고, 추론을 순수 numpy로 재구현**
> (`api/flood_model.py`). sklearn `predict_proba`와 **비트 단위 동일**(오차 0.0) 검증.

| 패키지 | 대략 |
|---|---|
| numpy | ~40MB |
| fastapi+pydantic+starlette | ~25MB |
| **합계** | **~65MB** ✅ 한도 내 (여유 큼) |

추가 이점:
- **버전 결합 없음**: `model_np.json`은 평문 JSON이라 sklearn 버전과 무관(피클 언로드 호환성 문제 제거).
- ONNX는 `skl2onnx`+최신 `onnx`가 HistGB의 NaN-분기 속성에서 충돌(py3.12 빌드 실패)해 채택하지 않음 — numpy 직접 평가가 더 견고.
- `torch`(+800MB)·`pandas`(+70MB)·`scipy`(~130MB) 모두 불필요.
> 초과 시 대안: scipy 의존 축소, 또는 컨테이너 호스트(Fly.io/Render)로 이전.

---

## 3. GitHub → Vercel 연동 (권장)

```bash
git init && git add -A
git commit -m "flood-risk api: preprocess, model, serving"
git branch -M main
git remote add origin https://github.com/<you>/doubled_seven.git
git push -u origin main
```
1. vercel.com → **Add New… → Project → Import** 해당 GitHub 레포.
2. Framework Preset = **Other**, Root = 레포 루트(자동).
3. **Deploy**. 이후 `main` push마다 프로덕션, PR마다 Preview 배포.
4. 배포 후: `GET https://<app>.vercel.app/api/health` 로 확인.

> 커밋 필수 파일: `api/index.py`, `api/flood_model.py`, `vercel.json`, `requirements.txt`,
> `build/model_np.json`, `build/serve_tables.json`.
> `.gitignore`가 raw `data/`와 무거운 `build/*.parquet`·`build/*.pkl`는 제외하되 위 산출물은 포함하도록 설정됨.

### CLI 배포 (대안)
```bash
npm i -g vercel
vercel            # preview
vercel --prod     # production
```

---

## 4. CI (GitHub Actions)
`.github/workflows/ci.yml` — push/PR 시 서빙 의존성 설치 후 **헬스체크+예측 스모크**:
```python
TestClient(app).post("/api/predict", json={
  "address":"서울 노원구 중계동 23-28","forecast_daily_rain":[5,40,60,100]})
```
모델/코드 회귀를 배포 전에 차단.

---

## 5. 모델 재배포 워크플로
```bash
python build_dataset.py            # 데이터 갱신 시
python train.py --trials 40        # 재학습 -> build/model_np.json, serve_tables.json
git add build/model_np.json build/serve_tables.json build/metrics.json
git commit -m "retrain" && git push # Vercel 자동 재배포
```
> `model_np.json`은 평문 JSON이라 **sklearn 버전과 무관** — 서빙은 numpy만 필요(피클 호환성 걱정 없음).
> 학습 환경(`requirements-dev.txt`)에만 scikit-learn 필요.

---

## 6. 운영 체크리스트
- [ ] `geocode_to_admcd()` 실제 지오코더(VWorld/Kakao) 연동 — 스텁 교체 (`README.md §주의`)
- [ ] 기상청 예보 강우(일/시간) 연동 — `forecast_daily_rain` 자동 공급, 강도 프록시 교체
- [ ] CORS 필요 시 `fastapi.middleware.cors` 추가(웹 클라이언트)
- [ ] 시크릿(지오코더 API 키)은 Vercel **Environment Variables**로 주입
- [ ] `maxDuration`/`memory`는 트래픽에 맞춰 조정(현재 10s/1GB, 단건 추론 <50ms)
- [ ] 콜드스타트: 모델 263KB로 가벼움. 필요 시 Vercel Cron으로 워밍
