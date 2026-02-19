"""
src/database/connection.py
===========================
SQLAlchemy 데이터베이스 연결 관리.

[패턴] Singleton — @lru_cache로 엔진 인스턴스 재사용
[역할] get_engine(), get_session() 함수를 통해 DB 연결을 제공합니다.

사용법:
    from src.database.connection import get_session
    session = get_session()
    try:
        result = session.query(Store).all()
    finally:
        session.close()
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import get_settings


@lru_cache()
def get_engine():
    """
    SQLAlchemy 엔진 싱글턴.

    @lru_cache로 앱 전체에서 하나의 엔진만 유지합니다.
    테스트 시 get_engine.cache_clear() 호출로 초기화 가능.
    """
    settings = get_settings()
    return create_engine(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_recycle=settings.DB_POOL_RECYCLE,
        echo=False,
    )


def get_session() -> Session:
    """새 데이터베이스 세션을 생성합니다."""
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def init_db():
    """
    ORM 모델 기반으로 테이블을 자동 생성합니다.

    개발/테스트 환경에서 사용. 프로덕션은 docker/init.sql 사용 권장.
    """
    from src.database.models import Base
    engine = get_engine()
    Base.metadata.create_all(engine)
