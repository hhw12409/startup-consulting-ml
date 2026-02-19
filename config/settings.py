"""
📁 config/settings.py
=====================
환경별 설정 관리.

[패턴] Singleton — @lru_cache로 앱 전체에서 하나의 인스턴스만 유지
[역할] .env 파일과 환경변수에서 설정값을 로드합니다.

사용법:
    from config.settings import get_settings
    s = get_settings()
    print(s.PUBLIC_DATA_SERVICE_KEY)
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings

# 프로젝트 루트 (이 파일 기준 한 단계 위)
ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """앱 전체 설정. 우선순위: 환경변수 > .env > 기본값"""

    # ── 기본 ──
    APP_NAME: str = "startup-consultant"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"          # development | production
    DEBUG: bool = True

    # ── 공공데이터 API 키 ──
    PUBLIC_DATA_SERVICE_KEY: str = ""  # data.go.kr 서비스 키
    NTS_API_KEY: str = ""             # 국세청 사업자 API 키
    REGION_CODE_API_KEY: str = ""     # 법정동코드 API 키

    # ── LLM API 키 ──
    ANTHROPIC_API_KEY: str = ""       # Claude API 키

    # ── 데이터베이스 ──
    DATABASE_URL: str = "mysql+pymysql://startup:startup1234@localhost:3306/startup_consultant"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE: int = 3600

    # ── 데이터 경로 ──
    DATA_RAW: str = str(ROOT / "data" / "01_raw")
    DATA_INTERIM: str = str(ROOT / "data" / "02_interim")
    DATA_PROCESSED: str = str(ROOT / "data" / "03_processed")
    DATA_FEATURES: str = str(ROOT / "data" / "04_features")
    DATA_MODEL_INPUT: str = str(ROOT / "data" / "05_model_input")

    # ── 모델 경로 ──
    MODEL_CHECKPOINTS: str = str(ROOT / "models" / "checkpoints")
    MODEL_REGISTRY: str = str(ROOT / "models" / "registry")
    MODEL_ARTIFACTS: str = str(ROOT / "models" / "artifacts")

    # ── 학습 하이퍼파라미터 (기본값, model_config.py에서 오버라이드) ──
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    MAX_EPOCHS: int = 200
    EARLY_STOPPING_PATIENCE: int = 20
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1

    # ── API 서버 ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── 로깅 ──
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = str(ROOT / "logs")

    class Config:
        env_file = str(ROOT / ".env")
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글턴. 테스트 시 get_settings.cache_clear() 호출."""
    return Settings()