# 🚀 실행 명령어 가이드

## 0단계. 초기 설정

```bash
# 프로젝트 클론 후 최초 1회
cp .env.example .env          # 환경변수 파일 생성
vi .env                        # API 키 입력 (data.go.kr에서 발급)
make install                   # 의존성 설치
```

---

## 1단계. 데이터 수집

```bash
make collect
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_collect.py` |
| 호출 클래스 | `src/data_collection/collector.py → DataCollector` |
| API | 소상공인진흥공단 상가정보 + 국세청 사업자 상태 |
| 출력 | `data/01_raw/stores_raw.csv` |
| 필수 조건 | `.env`에 `PUBLIC_DATA_SERVICE_KEY` 입력 |

> ⚠️ API 키가 없어도 2단계부터는 더미 데이터로 실행 가능합니다.

---

## 2단계. 모델 학습

```bash
# XGBoost (기본 — 빠르고 정형 데이터에 강함)
make train

# PyTorch 딥러닝 (데이터 10만건 이상일 때 유리)
make train-dl

# 직접 데이터 파일 지정
python scripts/run_train.py --model xgboost --data data/01_raw/my_data.csv
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_train.py` |
| 파이프라인 | `pipelines/train_pipeline.py → TrainPipeline` |
| 데이터 흐름 | 로드 → 정제 → 라벨 → 피처 → 분할 → 학습 → 평가 → 저장 |
| 출력 모델 | `models/registry/best_model.pkl` |
| 출력 전처리기 | `models/artifacts/scaler.pkl, label_encoders.pkl` |
| 출력 데이터 | `data/05_model_input/X_train.npy, y_train.npy` 등 |
| 평가 리포트 | `logs/eval_report.json` |
| 소요 시간 | XGBoost: ~4초 (5,000건), DL: ~30초 (5,000건) |

---

## 3단계. 모델 평가

```bash
make evaluate
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_evaluate.py` |
| 역할 | 저장된 모델을 테스트 데이터로 다시 평가 |
| 평가 메트릭 | 생존 정확도/AUC, 매출 MAE/R², 리스크 MAE |
| 출력 | `logs/eval_report.json` (갱신) |
| 전제 조건 | `make train`을 먼저 실행해야 모델이 있음 |

---

## 4단계. API 서버 실행

```bash
make serve
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_server.py` |
| 프레임워크 | FastAPI + Uvicorn |
| 주소 | `http://localhost:8000` |
| Swagger 문서 | `http://localhost:8000/docs` ← 여기서 테스트 가능 |
| 전제 조건 | `make train`으로 모델이 저장되어 있어야 함 |

### API 엔드포인트

```
GET  /health           → 서버 상태 확인
POST /api/v1/predict   → 창업 성공 예측
```

### 예측 API 호출 예시 (curl)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "founder_age": 35,
    "founder_gender": "M",
    "founder_education": "bachelor",
    "experience_years": 5,
    "has_related_experience": true,
    "business_category": "food",
    "business_sub_category": "cafe",
    "initial_investment": 50000000,
    "monthly_rent": 2000000,
    "store_size_sqm": 33.0,
    "employee_count": 2,
    "is_franchise": false,
    "district": "강남구 역삼동"
  }'
```

### 응답 예시

```json
{
  "success": true,
  "data": {
    "survival": {
      "one_year": 0.7234,
      "three_year": 0.4891
    },
    "financials": {
      "monthly_revenue": 15230000,
      "monthly_profit": 3120000,
      "break_even_months": 16
    },
    "risk": {
      "score": 0.3521,
      "level": "MEDIUM",
      "factors": ["경쟁 과밀 지역입니다"]
    },
    "recommendations": ["전반적으로 양호합니다. 마케팅 전략에 집중하세요"]
  }
}
```

---

## 5단계. 테스트

```bash
# 전체 테스트 (15개)
make test

# 단위 테스트만 (11개 — 함수/클래스 단위 검증)
make test-unit

# 통합 테스트만 (3개 — 파이프라인 전체 흐름 검증)
make test-integ
```

| 항목 | 내용 |
|------|------|
| 프레임워크 | pytest |
| 공통 픽스처 | `tests/conftest.py` (샘플 데이터, 학습된 모델) |
| 단위 테스트 | `tests/unit/test_features.py`, `test_models.py` |
| 통합 테스트 | `tests/integration/test_pipeline.py` |

---

## 기타 명령어

```bash
# 캐시/임시 파일 정리
make clean

# Docker 빌드 & 실행 (ECS Fargate 배포용)
docker build -t startup-consultant .
docker run -p 8000:8000 --env-file .env startup-consultant

# 도움말
make help
```

---

## 전체 실행 순서 요약

```
1. make install          # 최초 1회
2. make collect          # API 키 있으면 (없으면 건너뛰기)
3. make train            # 학습 (더미 데이터로도 가능)
4. make evaluate         # 평가 확인
5. make serve            # API 서버 → localhost:8000/docs
6. make test             # 테스트 확인
```

---

## 파일 흐름도

```
make collect
  └→ scripts/run_collect.py
      └→ src/data_collection/collector.py
          ├→ public_data_client.py (상가 데이터)
          └→ nts_client.py (사업자 상태)
      └→ 출력: data/01_raw/stores_raw.csv

make train
  └→ scripts/run_train.py
      └→ pipelines/train_pipeline.py
          ├→ src/preprocessing/cleaner.py   → data/02_interim/
          ├→ src/preprocessing/labeler.py   → data/03_processed/
          ├→ src/features/builder.py        → data/04_features/
          ├→ src/features/store.py          → data/05_model_input/
          ├→ src/models/xgboost_model.py    → models/registry/
          └→ src/evaluation/metrics.py      → logs/eval_report.json

make serve
  └→ scripts/run_server.py
      └→ src/serving/app.py
          ├→ src/serving/dependencies.py (모델 로드)
          ├→ src/serving/predictor.py (추론)
          └→ src/serving/schemas.py (요청/응답)
```