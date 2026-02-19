# 🚀 AI 기반 소상공인 창업 컨설팅 시스템

> ML 예측 + RAG + LLM 하이브리드 아키텍처
>
> 공공데이터 기반 상가 분석 → XGBoost/PyTorch 예측 → 로컬 LLM 자연어 컨설팅

---

## 전체 실행 순서 요약

```
1. make install           # 의존성 설치 (최초 1회)
2. make db-up             # MySQL Docker 시작
3. make collect           # 상가 데이터 수집 (API 키 필요)
4. make db-migrate        # CSV → MySQL 마이그레이션
5. make setup-ollama      # Ollama 한국어 모델 설치 (최초 1회)
6. make build-rag         # RAG 벡터DB 구축
7. make train             # XGBoost 모델 학습
8. make evaluate          # 모델 평가
9. make serve             # API 서버 → localhost:8000/docs
```

---

## 0단계. 초기 설정

```bash
# 프로젝트 클론 후 최초 1회
cp .env.example .env           # 환경변수 파일 생성
vi .env                         # API 키 입력 (data.go.kr에서 발급)
make install                    # 의존성 설치
```

### .env 설정

```env
PUBLIC_DATA_SERVICE_KEY=your_key_here     # 공공데이터 API 키 (필수)
NTS_API_KEY=your_key_here                 # 국세청 API 키 (선택)
REGION_CODE_API_KEY=your_key_here         # 행정동코드 API 키 (선택)
DATABASE_URL=mysql+pymysql://startup:startup1234@localhost:3306/startup_consultant
```

---

## 1단계. 데이터베이스 설정

```bash
# MySQL Docker 컨테이너 시작
make db-up

# 중지
make db-down
```

| 항목 | 내용 |
|------|------|
| 설정 파일 | `docker-compose.yml` |
| 초기화 SQL | `docker/init.sql` (테이블 자동 생성) |
| 접속 정보 | `localhost:3306`, user: `startup`, pw: `startup1234` |
| DB명 | `startup_consultant` |
| 테이블 | `stores` (상가), `region_codes` (행정동), `collection_logs` (수집 이력) |

---

## 2단계. 데이터 수집

```bash
# 행정동코드 수집 (API 키 불필요)
make collect-hdong

# 상가 데이터 수집 (서울 전체)
make collect

# 특정 행정동만 수집 (테스트)
python scripts/run_collect.py --codes 11680640 --limit 1

# CSV → MySQL 마이그레이션
make db-migrate
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_collect.py` |
| 호출 클래스 | `src/data_collection/collector.py → DataCollector` |
| API | 소상공인진흥공단 상가정보 (`adongCd` 8자리) + 국세청 사업자 상태 |
| 출력 (CSV) | `data/01_raw/stores_raw.csv` |
| 출력 (DB) | `stores` 테이블 (UPSERT, 중복 방지) |
| 행정동코드 | `data/00_region_codes/hdong_codes_11.csv` (PublicDataReader) |
| 필수 조건 | `.env`에 `PUBLIC_DATA_SERVICE_KEY` 입력 |

> ⚠️ API 키가 없어도 3단계부터는 더미 데이터로 실행 가능합니다.

---

## 3단계. 피처 엔지니어링

```bash
make feature
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_feature.py` |
| 파이프라인 | `DataCleaner → LabelGenerator → FeatureBuilder → FeatureStore` |
| 입력 | `data/01_raw/stores_raw.csv` 또는 MySQL `stores` 테이블 |
| 출력 | `data/05_model_input/X_train.npy, y_train.npy` 등 |
| 피처 수 | 22개 (업종 인코딩, 지역 통계, 경쟁업체 밀도, 창업자 프로필 등) |

---

## 4단계. 모델 학습

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
| 출력 모델 | `models/registry/best_model.pkl` (.pt for PyTorch) |
| 출력 전처리기 | `models/artifacts/scaler.pkl, label_encoders.pkl` |
| 평가 리포트 | `logs/eval_report.json` |
| 소요 시간 | XGBoost: ~4초, DL: ~30초 (5,000건 기준) |

---

## 5단계. 모델 평가

```bash
# 모든 모델 자동 탐지 & 평가
make evaluate

# 특정 모델만
python scripts/run_evaluate.py --model xgboost
python scripts/run_evaluate.py --model neural_net
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_evaluate.py` |
| 역할 | 저장된 모델을 테스트 데이터로 평가 + 모델 간 비교 |
| 평가 메트릭 | 생존 정확도/F1/AUC, 매출 MAE/R², 리스크 MAE |
| 출력 | `logs/eval_xgboost.json`, `logs/eval_neural_net.json`, `logs/eval_comparison.json` |
| 전제 조건 | `make train`을 먼저 실행해야 모델이 있음 |

---

## 6단계. LLM & RAG 설정

```bash
# Ollama 설치 (macOS)
brew install ollama

# 한국어 모델 설치 (5.4GB, 1회)
make setup-ollama
# 또는: ollama pull gemma2:9b

# 임베딩 모델 설치 (RAG용, 274MB)
ollama pull nomic-embed-text

# RAG 벡터DB 구축 (테스트: 1000건)
make build-rag-test

# RAG 벡터DB 구축 (전체)
make build-rag

# RAG 검색 테스트
python scripts/run_build_rag.py --query "강남구 카페"
```

| 항목 | 내용 |
|------|------|
| LLM 모델 | Ollama gemma2:9b (한국어 자동 탐지, 완전 로컬, 무료) |
| 임베딩 모델 | nomic-embed-text (274MB) |
| 벡터DB | ChromaDB (`data/06_vector_db/`) |
| 프롬프트 강화 | A) DataContext (업종/지역 통계) + B) RAGStore (유사 사례 검색) |
| 구축 스크립트 | `scripts/run_build_rag.py` |

### 한국어 모델 우선순위 (자동 탐지)

| 순위 | 모델 | 크기 | 한국어 |
|------|------|------|--------|
| 1 | EEVE-Korean-10.8B | 6.5GB | 최고 |
| 2 | gemma2:9b | 5.4GB | 좋음 |
| 3 | llama3.1:8b | 4.7GB | 보통 |

---

## 7단계. API 서버 실행

```bash
# Ollama 서버 시작 (터미널 1)
ollama serve

# API 서버 시작 (터미널 2)
make serve
```

| 항목 | 내용 |
|------|------|
| 실행 스크립트 | `scripts/run_server.py` |
| 프레임워크 | FastAPI + Uvicorn |
| 주소 | `http://localhost:8000` |
| Swagger 문서 | `http://localhost:8000/docs` ← 여기서 테스트 가능 |
| 전제 조건 | `make train` (모델) + `ollama serve` (LLM) |

### API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 + 활성 LLM 확인 |
| POST | `/api/v1/predict` | ML 예측만 (숫자 결과) |
| POST | `/api/v1/consult` | ML 예측 + LLM 종합 컨설팅 리포트 |
| POST | `/api/v1/strategy` | 맞춤형 전략 제안 |
| POST | `/api/v1/ask` | Q&A 대화 |
| POST | `/api/v1/competitors` | 경쟁업체 분석 |

### 컨설팅 API 호출 예시 (curl)

```bash
curl -X POST http://localhost:8000/api/v1/consult \
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
    "district": "역삼1동"
  }'
```

### 응답 예시

```json
{
  "success": true,
  "llm_provider": "Ollama (gemma2:9b)",
  "prediction": {
    "survival": { "one_year": 0.7234, "three_year": 0.4891 },
    "financials": {
      "monthly_revenue": 15230000,
      "monthly_profit": 3120000,
      "break_even_months": 16
    },
    "risk": {
      "score": 0.3521,
      "level": "MEDIUM",
      "factors": ["경쟁 과밀 지역입니다"]
    }
  },
  "analysis": "## 종합 평가\n역삼1동에서 카페 창업을 계획하고 계시군요. 해당 지역에는 현재 음식 업종 상가가 234개 운영 중이며..."
}
```

---

## 8단계. 테스트

```bash
# 전체 테스트
make test

# 단위 테스트만
make test-unit

# 통합 테스트만
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
make help               # 전체 명령어 도움말
make clean              # 캐시/임시 파일 정리
make collect-regions    # 법정동코드 수집 (행정표준코드 API)
make collect-hdong      # 행정동코드 수집 (PublicDataReader)
```

---

## 파일 흐름도

```
make collect
  └→ scripts/run_collect.py
      └→ src/data_collection/collector.py
          ├→ public_data_client.py  (상가 데이터, adongCd 8자리)
          └→ nts_client.py          (사업자 상태, 연속실패 3회 조기중단)
      └→ 출력: data/01_raw/stores_raw.csv

make db-migrate
  └→ scripts/run_migrate_to_db.py
      └→ src/database/repository.py
          ├→ StoreRepository.upsert_stores()   → stores 테이블
          └→ RegionRepository.upsert_regions() → region_codes 테이블

make train
  └→ scripts/run_train.py (OMP_NUM_THREADS=1)
      └→ pipelines/train_pipeline.py
          ├→ src/preprocessing/cleaner.py      → data/02_interim/
          ├→ src/preprocessing/labeler.py      → data/03_processed/
          ├→ src/features/builder.py           → data/04_features/
          ├→ src/features/store.py             → data/05_model_input/
          ├→ src/models/xgboost_model.py       → models/registry/
          └→ src/evaluation/metrics.py         → logs/eval_report.json

make build-rag
  └→ scripts/run_build_rag.py
      └→ src/llm/rag_store.py
          ├→ stores_raw.csv → 텍스트 변환
          ├→ Ollama nomic-embed-text 임베딩
          └→ ChromaDB 저장 → data/06_vector_db/

make serve
  └→ scripts/run_server.py
      └→ src/serving/app.py
          ├→ dependencies.py → Predictor (ML 모델 로드)
          ├→ dependencies.py → Consultant (LLM + DataContext + RAG)
          ├→ predictor.py    → ML 예측
          └→ consultant.py   → 통계(A) + RAG(B) + LLM → 자연어 리포트
```

---

## 시스템 아키텍처

```
┌─ 수집 ──────────────────────────────────────────┐
│ 공공데이터 API → CSV → MySQL (UPSERT 중복 방지)   │
│ - stores: 상가 원본 데이터                        │
│ - region_codes: 행정동 마스터                     │
│ - collection_logs: 수집 이력                     │
└──────────────────────────────────────────────────┘
         ↓
┌─ 전처리 & 학습 ──────────────────────────────────┐
│ MySQL → DataFrame → 피처/라벨 → XGBoost/PyTorch  │
│                               → ChromaDB (RAG)   │
└──────────────────────────────────────────────────┘
         ↓
┌─ LLM 서빙 ──────────────────────────────────────┐
│ API 요청 → ML 예측                               │
│          + A) DataContext (업종/지역 통계)         │
│          + B) RAGStore (ChromaDB 유사 사례)       │
│          → Ollama gemma2:9b → 자연어 답변         │
└──────────────────────────────────────────────────┘
```