# ============================================================
# 🚀 창업 컨설턴트 AI — 실행 명령어 모음
# ============================================================
# 사용법: make <명령어>
# ============================================================

.PHONY: install collect collect-regions feature train train-dl evaluate serve test clean help

# ── 도움말 (기본) ──
help:
	@echo ""
	@echo "  창업 컨설턴트 AI — 사용 가능한 명령어"
	@echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  초기 설정:"
	@echo "    make install         의존성 패키지 설치"
	@echo ""
	@echo "  데이터:"
	@echo "    make collect         공공데이터 API에서 상가 데이터 수집"
	@echo "    make collect-regions 법정동코드 수집 (서울시 기본)"
	@echo "    make feature         피처 엔지니어링 (정제된 데이터 → 학습용 변환)"
	@echo ""
	@echo "  학습:"
	@echo "    make train        XGBoost 모델 학습 (기본, 권장)"
	@echo "    make train-dl     PyTorch 딥러닝 모델 학습"
	@echo ""
	@echo "  평가:"
	@echo "    make evaluate     저장된 모델로 테스트 데이터 평가"
	@echo ""
	@echo "  서버:"
	@echo "    make serve        API 서버 실행 (localhost:8000)"
	@echo ""
	@echo "  테스트:"
	@echo "    make test         전체 테스트 실행"
	@echo "    make test-unit    단위 테스트만"
	@echo "    make test-integ   통합 테스트만"
	@echo ""
	@echo "  기타:"
	@echo "    make clean        캐시/임시 파일 정리"
	@echo "    make help         이 도움말 표시"
	@echo ""

# ── 초기 설정 ──
install:
	pip install -r requirements.txt

# ── 데이터 수집 ──
collect-hdong:
	python scripts/run_collect_hdong_codes.py

collect-regions:
	python scripts/run_collect_region_codes.py

collect:
	python scripts/run_collect.py

# ── 피처 엔지니어링 ──
feature:
	python scripts/run_feature.py

# ── 모델 학습 ──
train:
	python scripts/run_train.py --model xgboost

train-dl:
	python scripts/run_train.py --model neural_net

# ── 모델 평가 ──
evaluate:
	python scripts/run_evaluate.py

# ── API 서버 ──
serve:
	python scripts/run_server.py

# ── 테스트 ──
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

test-integ:
	python -m pytest tests/integration/ -v

# ── 정리 ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache

# ── 데이터베이스 ──
db-up:
	docker-compose up -d

db-down:
	docker-compose down

db-init:
	python -c "from src.database.connection import init_db; init_db()"

db-reset:
	docker-compose down -v && docker-compose up -d

db-migrate:
	python scripts/run_migrate_to_db.py

# ── LLM 설정 ──
setup-ollama:
	bash scripts/setup_ollama.sh

# ── RAG 벡터DB ──
build-rag:
	python scripts/run_build_rag.py

build-rag-test:
	python scripts/run_build_rag.py --max 1000