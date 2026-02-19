"""
📁 src/utils/timer.py
======================
실행 시간 측정 유틸리티.

[패턴] Decorator — 함수에 @timer를 붙이면 실행시간을 자동 로깅합니다.

사용법:
    @timer("데이터 수집")
    def collect_data():
        ...
    # 출력: [TIMER] 데이터 수집 완료: 12.3초
"""

import time
import functools
from src.utils.logger import get_logger

logger = get_logger(__name__)


def timer(label: str = ""):
    """실행 시간 측정 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__name__
            start = time.time()
            logger.info("[TIMER] %s 시작...", name)

            result = func(*args, **kwargs)

            elapsed = time.time() - start
            logger.info("[TIMER] %s 완료: %.1f초", name, elapsed)
            return result
        return wrapper
    return decorator