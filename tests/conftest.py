"""
📁 tests/conftest.py
=====================
pytest 공통 설정 파일.

[역할] 모든 테스트에서 공유하는 픽스처(fixture)를 정의합니다.
       pytest가 자동으로 이 파일을 로드합니다.

[패턴] Fixture — 테스트에 필요한 객체를 미리 생성하여 주입
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================
# 공통 데이터 픽스처
# ================================================================
@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """원본 데이터 형태의 테스트 DataFrame (5행)"""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [28, 35, 42, 55, 23],
        "gender": ["M", "F", "M", "F", "M"],
        "education_level": ["bachelor", "master", "high_school", "bachelor", "bachelor"],
        "experience_years": [3, 10, 15, 20, 1],
        "has_related_experience": [0, 1, 1, 0, 0],
        "has_startup_experience": [0, 0, 1, 1, 0],
        "initial_capital": [30_000_000, 100_000_000, 50_000_000, 200_000_000, 15_000_000],
        "business_category": ["food", "retail", "food", "service", "food"],
        "business_sub_category": ["cafe", "beauty", "chicken", "academy", "cafe"],
        "district": ["강남구", "마포구", "종로구", "서초구", "성동구"],
        "store_size_sqm": [33.0, 50.0, 66.0, 99.0, 20.0],
        "initial_investment": [50_000_000, 80_000_000, 40_000_000, 150_000_000, 20_000_000],
        "monthly_rent": [2_000_000, 3_000_000, 1_500_000, 5_000_000, 1_000_000],
        "employee_count": [2, 3, 1, 5, 0],
        "is_franchise": [0, 1, 0, 0, 1],
        "nearby_competitor_count": [8, 15, 5, 3, 20],
        "floating_population_level": ["high", "medium", "low", "medium", "high"],
    })


@pytest.fixture
def sample_labeled_df(sample_raw_df) -> pd.DataFrame:
    """라벨이 추가된 테스트 DataFrame"""
    df = sample_raw_df.copy()
    df["survival_1yr"] = [0.8, 0.6, 0.9, 0.4, 0.3]
    df["survival_3yr"] = [0.6, 0.4, 0.7, 0.2, 0.1]
    df["monthly_revenue"] = [15_000_000, 25_000_000, 12_000_000, 30_000_000, 8_000_000]
    df["monthly_profit"] = [3_000_000, 5_000_000, 2_000_000, 7_000_000, 1_000_000]
    df["risk_score"] = [0.3, 0.5, 0.2, 0.7, 0.8]
    df["break_even_months"] = [18, 16, 20, 24, 36]
    return df


@pytest.fixture
def sample_xy(sample_labeled_df) -> tuple[np.ndarray, np.ndarray]:
    """학습용 (X, y) numpy 배열"""
    from src.features.builder import FeatureBuilder
    builder = FeatureBuilder()
    return builder.fit_transform(sample_labeled_df)


# ================================================================
# DB 픽스처
# ================================================================
@pytest.fixture
def db_engine():
    """SQLite 인메모리 DB 엔진 (단위 테스트용)"""
    from sqlalchemy import create_engine
    from src.database.models import Base
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """SQLite 인메모리 DB 세션"""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_get_session(db_engine, monkeypatch):
    """get_session()을 SQLite 세션 팩토리로 패치.

    Repository 메서드들이 session.close()를 호출하므로,
    매 호출마다 새 세션을 생성해야 합니다.
    같은 engine을 공유하여 인메모리 DB 데이터를 유지합니다.
    """
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)

    monkeypatch.setattr(
        "src.database.repository.get_session",
        lambda: Session(),
    )
    return Session()


# ================================================================
# 모델 픽스처
# ================================================================
@pytest.fixture
def trained_xgboost(sample_xy):
    """학습된 XGBoost 모델"""
    from src.models.xgboost_model import XGBoostModel
    X, y = sample_xy
    model = XGBoostModel()
    model.train(X, y, X, y)  # 테스트용이므로 train=val
    return model