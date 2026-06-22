# FRONTEND.md — Flutter 연동 가이드

서울 침수 위험 예측 API를 Flutter 앱에서 바로 연동하기 위한 문서.
**기계 판독용 스펙은 [`openapi.json`](openapi.json)** (Swagger). 실행 중 서버의 `/docs`(Swagger UI)·`/redoc`도 동일.

---

## 1. Base URL & 공통

| 환경 | Base URL |
|---|---|
| 로컬 | `http://127.0.0.1:8000` |
| 프로덕션 | `https://<your-app>.vercel.app` |

- 모든 요청/응답 `application/json`, UTF-8.
- 인증 없음(공개). CORS 허용(`*`) — 웹/모바일에서 직접 호출 가능.
- 좌표/주소는 **서울 25개 구의 침수 이력 93개 법정동**만 커버. 그 외 → `404`.

---

## 2. 엔드포인트 요약

| 메서드 | 경로 | 용도 |
|---|---|---|
| `GET` | `/api/health` | 헬스체크(동 수/피처 수) |
| `GET` | `/api/dongs` | 커버리지 동 목록(선택 UI/자동완성용) |
| `POST` | `/api/predict` | 침수 확률 예측 |

### `POST /api/predict`

**Request**
```json
{
  "address": "서울 노원구 중계동 23-28",
  "forecast_daily_rain": [5, 40, 60, 100],
  "adm_cd": null
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `address` | string | ▲ | 유저 주소. 서버가 법정동으로 지오코딩. `adm_cd`를 주면 생략 가능 |
| `forecast_daily_rain` | number[] | ✅ | 일강우(mm), **과거→오늘**(오늘이 마지막). 1~30개 |
| `adm_cd` | int? | ✕ | 10자리 법정동코드. 주면 지오코딩 생략(가장 안정적) |

**Response 200**
```json
{ "adm_cd": 1135010600, "gu": "노원구", "dong": "중계동", "flood_probability": 0.7473 }
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `adm_cd` | int | 법정동코드 |
| `gu` | string | 구 |
| `dong` | string? | 동 라벨(null 가능) |
| `flood_probability` | number | 침수 확률 [0,1] |

**에러**
| status | detail 예 | 처리 권장 |
|---|---|---|
| 404 | `dong not resolved …` | 주소 해석 실패 → 동 직접 선택(`/api/dongs`) 유도 |
| 404 | `adm_cd … not in coverage` | 커버리지 밖 안내 |
| 422 | (pydantic 검증) | 입력값(빈 배열 등) 점검 |

> **권장 패턴**: 앱에서 `/api/dongs`로 동을 고르게 하고 `adm_cd`를 직접 전송하면 지오코더 의존/404를 피함. 주소 자유입력은 보조.

### `GET /api/dongs`
```json
[ { "adm_cd": 1135010500, "gu": "노원구", "dong": "상계동" }, … ]
```

---

## 3. Flutter 예제

### 3-1. 모델 클래스
```dart
class FloodRisk {
  final int admCd;
  final String gu;
  final String? dong;
  final double floodProbability;
  FloodRisk(this.admCd, this.gu, this.dong, this.floodProbability);
  factory FloodRisk.fromJson(Map<String, dynamic> j) => FloodRisk(
        j['adm_cd'] as int,
        j['gu'] as String,
        j['dong'] as String?,
        (j['flood_probability'] as num).toDouble(),
      );
}
```

### 3-2. `http` 패키지
```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const baseUrl = 'https://<your-app>.vercel.app';

Future<FloodRisk> predictFlood({
  String address = '',
  int? admCd,
  required List<double> forecastDailyRain,
}) async {
  final res = await http.post(
    Uri.parse('$baseUrl/api/predict'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'address': address,
      'adm_cd': admCd,
      'forecast_daily_rain': forecastDailyRain,
    }),
  );
  if (res.statusCode == 200) {
    return FloodRisk.fromJson(jsonDecode(utf8.decode(res.bodyBytes)));
  } else if (res.statusCode == 404) {
    throw Exception('커버리지 밖이거나 주소를 해석할 수 없습니다.');
  } else {
    throw Exception('예측 실패 (${res.statusCode}): ${utf8.decode(res.bodyBytes)}');
  }
}

// 사용
final risk = await predictFlood(
  admCd: 1135010600,
  forecastDailyRain: [5, 40, 60, 100],
);
print('${risk.dong} 침수확률 ${(risk.floodProbability * 100).toStringAsFixed(1)}%');
```

### 3-3. `dio` 패키지(인터셉터·타임아웃 선호 시)
```dart
final dio = Dio(BaseOptions(
  baseUrl: 'https://<your-app>.vercel.app',
  connectTimeout: const Duration(seconds: 5),
  headers: {'Content-Type': 'application/json'},
));

Future<FloodRisk> predict(int admCd, List<double> rain) async {
  final r = await dio.post('/api/predict',
      data: {'adm_cd': admCd, 'forecast_daily_rain': rain});
  return FloodRisk.fromJson(r.data);
}
```

### 3-4. 동 목록 로드(선택 UI)
```dart
Future<List<Map<String, dynamic>>> fetchDongs() async {
  final r = await http.get(Uri.parse('$baseUrl/api/dongs'));
  return (jsonDecode(utf8.decode(r.bodyBytes)) as List).cast<Map<String, dynamic>>();
}
```

---

## 4. OpenAPI 코드 생성(선택)
타입 안전 클라이언트를 자동 생성하려면 `openapi.json` 사용:
```bash
# openapi-generator
openapi-generator generate -i openapi.json -g dart-dio -o lib/api
# 또는 swagger_parser / openapi_generator(dart pub) 사용
```

---

## 5. UX 가이드
- `flood_probability`는 **상대 위험도**(보정했으나 2년치 학습이라 절대값은 보수적). 임계값 색상 예: `<0.2` 안전 / `0.2–0.5` 주의 / `>0.5` 경고.
- `forecast_daily_rain`은 기상청 단기예보(향후 3일 일강우)를 과거 누적과 이어 붙여 보내면 정확도↑(서버가 3/7/14/30일 누적을 계산).
- 동 미커버(404) 시: "이 지역은 아직 예측 대상이 아닙니다" + 인접 커버 동 안내.
