"""
📁 src/utils/logger.py
=======================
구조화된 로깅 설정.

[역할] 앱 전체의 로깅 포맷과 핸들러를 통일합니다.

사용법:
    from src.utils.logger import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    logger.info("학습 시작", extra={"epoch": 1})
"""

import logging
import sys
from pathlib import Path
from config.settings import get_settings


def setup_logging() -> None:
    """앱 시작 시 1회 호출. 콘솔 + 파일 로깅을 설정합니다."""
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # 콘솔 핸들러
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # 파일 핸들러 (production)
    if settings.ENV != "development":
        fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # 외부 라이브러리 로그 억제
    for lib in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 반환. 관례: get_logger(__name__)"""
    return logging.getLogger(name)