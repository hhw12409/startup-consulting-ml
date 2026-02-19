"""
📁 tests/unit/test_features.py
================================
피처 빌더 단위 테스트.

실행: pytest tests/unit/test_features.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.features.builder import FeatureBuilder


@pytest.fixture
def sample_df():
    """테스트용 샘플 DataFrame"""
    return pd.DataFrame({
        "age": [30, 45], "gender": ["M", "F"],
        "education_level": ["bachelor", "master"],
        "experience_years": [5, 10],
        "has_related_experience": [1, 0],
        "has_startup_experience": [0, 1],
        "initial_capital": [50_000_000, 100_000_000],
        "business_category": ["food", "retail"],
        "business_sub_category": ["cafe", "beauty"],
        "district": ["강남구", "마포구"],
        "store_size_sqm": [33.0, 50.0],
        "initial_investment": [50_000_000, 80_000_000],
        "monthly_rent": [2_000_000, 3_000_000],
        "employee_count": [2, 3],
        "is_franchise": [0, 1],
        "nearby_competitor_count": [5, 15],
        "floating_population_level": ["high", "medium"],
        # 타겟
        "survival_1yr": [0.8, 0.4],
        "survival_3yr": [0.6, 0.2],
        "monthly_revenue": [15_000_000, 25_000_000],
        "monthly_profit": [3_000_000, 5_000_000],
        "risk_score": [0.3, 0.7],
        "break_even_months": [18, 24],
    })


def test_fit_transform_returns_correct_shapes(sample_df):
    """fit_transform이 올바른 shape을 반환하는지"""
    builder = FeatureBuilder()
    X, y = builder.fit_transform(sample_df)

    assert X.shape[0] == 2, "행 수가 일치해야 함"
    assert y.shape == (2, 6), "타겟은 6개"
    assert X.dtype == np.float32


def test_transform_after_fit(sample_df):
    """fit 후 transform이 동일한 피처 수를 반환하는지"""
    builder = FeatureBuilder()
    X_train, _ = builder.fit_transform(sample_df)
    X_new = builder.transform(sample_df.drop(columns=[
        "survival_1yr", "survival_3yr", "monthly_revenue",
        "monthly_profit", "risk_score", "break_even_months",
    ]))

    assert X_new.shape[1] == X_train.shape[1], "피처 수가 동일해야 함"


def test_transform_before_fit_raises_error():
    """fit 없이 transform 호출 시 에러"""
    builder = FeatureBuilder()
    with pytest.raises(RuntimeError):
        builder.transform(pd.DataFrame({"age": [30]}))