# FRONTEND.md — Flutter 연동 가이드 (API 계약)

이 문서는 **클라이언트(Flutter) 개발자를 위한 단일 API 계약서**입니다.
기계 판독용 스펙은 [`openapi.json`](openapi.json)(Swagger). 실행 중 서버의 `/docs`(Swagger UI)·`/redoc`도 동일.

> 데이터/모델 내부는 알 필요 없습니다. 이 문서의 **엔드포인트 + payload + SDK 정규화 규칙**만 보면 됩니다.

---

## 1. Base URL & 공통 규칙

| 환경     | Base URL                        |
| -------- | ------------------------------- |
| 로컬     | `http://127.0.0.1:8000`         |
| 프로덕션 | `https://ready-flow-ai.vercel.app` |

- 모든 요청/응답 `application/json`, UTF-8. 인증 없음. CORS 허용(`*`) — 모바일/웹 직접 호출 가능.
- 커버리지 = 서울 25개 구의 **침수 이력 93개 법정동**. 그 외 → `404`.

---

## 2. ⭐ SDK 연동 규칙 (입력 정규화 책임 = 클라이언트)

서버 API는 **특정 SDK에 묶이지 않은 정규화된 계약**입니다. 즉 어떤 SDK를 쓰든
**클라이언트가 SDK 출력을 아래 표준 payload로 변환**해서 보냅니다. (서버는 SDK 원본 JSON을 받지 않음.)

```
[SDK]                          [클라이언트 변환]                 [서버 payload]
주소/지오코딩 SDK  ──────▶  법정동코드(adm_cd) 추출   ──────▶  "adm_cd": 1135010600
(Kakao·VWorld 등)             (좌표→법정동)                     (또는 "address": "...")

날씨 예보 SDK     ──────▶  시간별 강우 → 일강우 합산  ──────▶  "forecast_daily_rain":
(기상청·OpenWeather)         (과거→오늘 순서, mm)               [5, 40, 60, 100]
```

**두 가지 변환만 책임지면 됩니다:**

| SDK 출력            | 변환 규칙                                                  | 비고                                                                                                               |
| ------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 주소/좌표           | → **`adm_cd`(10자리 법정동코드)** 로 보내는 게 가장 안전   | 지오코딩 SDK가 법정동코드를 주면 그대로 사용. 없으면 `address` 문자열 전송(서버 스텁 지오코더는 운영 시 교체 예정) |
| 시간별/3시간별 강우 | → **일강우(mm) 배열로 합산**, **과거→오늘**(오늘이 마지막) | 누적/선행 강우(3·7·14·30일)는 서버가 이 배열로 계산. 길이 1~30                                                     |

> 즉 "SDK에 맞췄냐"의 답: **API는 SDK 중립이고, 정규화는 클라이언트가 담당**합니다. SDK를 바꿔도 서버는 그대로입니다. (§4-5에 변환 코드 예시.)

---

## 3. 엔드포인트

| 메서드 | 경로           | 용도                                 |
| ------ | -------------- | ------------------------------------ |
| `GET`  | `/api/health`  | 헬스체크(동 수/피처 수)              |
| `GET`  | `/api/dongs`   | 커버리지 동 목록(선택 UI/자동완성용) |
| `POST` | `/api/predict` | 침수 확률 예측                       |

### `POST /api/predict`

**Request**

```json
{
  "adm_cd": 1135010600,
  "forecast_daily_rain": [5, 40, 60, 100],
  "address": ""
}
```

| 필드                  | 타입     | 필수 | 설명                                                      |
| --------------------- | -------- | ---- | --------------------------------------------------------- |
| `forecast_daily_rain` | number[] | ✅   | 일강우(mm), **과거→오늘**(오늘이 마지막). 1~30개          |
| `adm_cd`              | int?     | ▲    | 10자리 법정동코드. **이걸 보내는 걸 권장**(지오코딩 생략) |
| `address`             | string   | ▲    | `adm_cd`가 없을 때 주소로 지오코딩. 둘 중 하나는 필요     |
| `building_type`       | string?  | ✕    | (선택) 건물 유형 — 아래 허용값 중 하나. **안 보내도 됨**  |

**`building_type` 허용값** — 보낼 거면 아래 6개 중 하나만(그 외 값은 `422`):

| 값            | 의미        | 포함 예             |
| ------------- | ----------- | ------------------- |
| `residential` | 주거        | 단독주택·아파트·빌라 |
| `commercial`  | 상가/시설   | 상가·건물·시설      |
| `industrial`  | 공장        | 공장               |
| `underground` | 지하/반지하 | 지하·반지하         |
| `road`        | 도로        | 도로               |
| `etc`         | 기타        | 그 외               |

> ⚠️ **현재 모델은 `building_type`을 사용하지 않습니다**(예측값 불변). 누수·서빙 패리티 이슈로 v1 모델에서 제외(`DATA.md §6`). 이 필드는 **클라이언트가 지금부터 수집해 두면 향후 모델 버전에서 무중단 연동**되도록 한 예약 필드입니다. **안 보내는 게 기본**이며, 보내면 enum 검증만 합니다.

**Response 200**

```json
{ "adm_cd": 1135010600, "gu": "노원구", "dong": "중계동", "flood_probability": 0.0765 }
```

| 필드                | 타입    | 설명                                                      |
| ------------------- | ------- | --------------------------------------------------------- |
| `adm_cd`            | int     | 법정동코드                                                |
| `gu`                | string  | 구                                                        |
| `dong`              | string? | 동 라벨(null 가능)                                        |
| `flood_probability` | number  | 침수 확률 [0,1] — **base-rate 보정된 보수적 값**(§6 참고) |

**에러**
| status | detail 예 | 처리 권장 |
|---|---|---|
| 404 | `dong not resolved …` | 주소 해석 실패 → `/api/dongs`로 동 직접 선택 유도 |
| 404 | `adm_cd … not in coverage` | 커버리지 밖 안내 |
| 422 | (pydantic) | 입력값(빈 배열 등) 점검 |

### `GET /api/dongs`

```json
[ { "adm_cd": 1135010500, "gu": "노원구", "dong": "상계동" }, … ]
```

→ 동 선택 드롭다운/자동완성에 사용. 선택값의 `adm_cd`를 `/api/predict`에 전달하면 지오코더 불필요.

---

## 4. Flutter 예제

### 4-1. 모델 클래스

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

### 4-2. 예측 호출 (`http`)

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

const baseUrl = 'https://ready-flow-ai.vercel.app';

Future<FloodRisk> predictFlood({
  int? admCd,
  String address = '',
  required List<double> forecastDailyRain,
}) async {
  final res = await http.post(
    Uri.parse('$baseUrl/api/predict'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode({
      'adm_cd': admCd,
      'address': address,
      'forecast_daily_rain': forecastDailyRain,
    }),
  );
  if (res.statusCode == 200) {
    return FloodRisk.fromJson(jsonDecode(utf8.decode(res.bodyBytes)));
  } else if (res.statusCode == 404) {
    throw Exception('커버리지 밖이거나 주소를 해석할 수 없습니다.');
  }
  throw Exception('예측 실패 (${res.statusCode}): ${utf8.decode(res.bodyBytes)}');
}
```

### 4-3. SDK 정규화 어댑터 (§2 규칙의 구현)

```dart
// (a) 날씨 SDK의 시간별 강우 -> 일강우 배열(과거->오늘)
List<double> toDailyRain(List<Map<String, dynamic>> hourly) {
  final byDay = <String, double>{};
  for (final h in hourly) {
    final day = (h['time'] as String).substring(0, 10); // 'YYYY-MM-DD'
    byDay[day] = (byDay[day] ?? 0) + (h['rain_mm'] as num).toDouble();
  }
  final days = byDay.keys.toList()..sort();          // 과거 -> 오늘
  return [for (final d in days) byDay[d]!];
}

// (b) 지오코딩 SDK 결과 -> adm_cd (법정동코드). SDK가 b_code/법정동코드를 주면 그대로 사용.
int admCdFromGeocode(Map<String, dynamic> geo) => int.parse(geo['b_code'] as String);

// 사용
final risk = await predictFlood(
  admCd: admCdFromGeocode(geoResult),
  forecastDailyRain: toDailyRain(weatherHourly),
);
print('${risk.dong} 침수확률 ${(risk.floodProbability * 100).toStringAsFixed(1)}%');
```

### 4-4. 동 목록 (`dio`)

```dart
final dio = Dio(BaseOptions(baseUrl: 'https://ready-flow-ai.vercel.app'));
Future<List<dynamic>> fetchDongs() async => (await dio.get('/api/dongs')).data;
```

---

## 5. OpenAPI 코드 생성(선택)

타입 안전 클라이언트 자동 생성:

```bash
openapi-generator generate -i openapi.json -g dart-dio -o lib/api
# 또는 dart pub의 swagger_parser / openapi_generator 사용
```

---

## 6. UX 가이드

- `flood_probability`는 **base-rate에 보정된 보수적 확률**(폭우라도 해당 동 실제 침수빈도 수준). **절대값보다 상대 위험/임계값**으로 표현 권장. 예: `<0.05` 안전 / `0.05–0.15` 주의 / `>0.15` 경고 (운영 데이터로 임계값 재튜닝).
- `forecast_daily_rain`은 예보(향후 며칠)를 **과거 실측 강우와 이어 붙여** 보내면 누적(3·7·14·30일) 정확도↑.
- 동 미커버(404): "이 지역은 아직 예측 대상이 아닙니다" + 인접 커버 동 안내.
