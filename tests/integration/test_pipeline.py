"""
📁 tests/integration/test_pipeline.py
=======================================
파이프라인 통합 테스트.

실행: pytest tests/integration/test_pipeline.py -v

[통합 테스트 vs 단위 테스트]
  - 단위 테스트: 하나의 함수/클래스만 검증
  - 통합 테스트: 여러 컴포넌트가 연결되어 올바르게 동작하는지 검증
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path


class TestTrainPipeline:
    """학습 파이프라인 전체 흐름 테스트"""

    def test_full_pipeline_runs_without_error(self, sample_labeled_df):
        """전체 파이프라인이 에러 없이 실행되는지"""
        from src.models.xgboost_model import XGBoostModel
        from src.features.builder import FeatureBuilder
        from src.features.store import FeatureStore
        from src.evaluation.metrics import evaluate_model

        # 피처 생성
        builder = FeatureBuilder()
        X, y = builder.fit_transform(sample_labeled_df)

        # 분할
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FeatureStore(base_dir=tmpdir)
            sizes = store.save_splits(X, y, val_ratio=0.2, test_ratio=0.2)
            assert sizes["train"] > 0

            # 학습
            X_train, y_train = store.load("train")
            X_test, y_test = store.load("test")

            model = XGBoostModel()
            model.train(X_train, y_train)

            # 평가
            metrics = evaluate_model(model, X_test, y_test)
            assert "survival_1yr_acc" in metrics


class TestInferencePipeline:
    """추론 파이프라인 테스트"""

    def test_single_prediction(self, sample_labeled_df):
        """단건 추론이 올바른 형태를 반환하는지"""
        from src.models.xgboost_model import XGBoostModel
        from src.features.builder import FeatureBuilder
        from pipelines.inference_pipeline import InferencePipeline

        builder = FeatureBuilder()
        X, y = builder.fit_transform(sample_labeled_df)

        model = XGBoostModel()
        model.train(X, y)

        pipeline = InferencePipeline(model, builder)
        result = pipeline.predict_single({
            "age": 35, "gender": "M", "education_level": "bachelor",
            "experience_years": 5, "has_related_experience": 1,
            "has_startup_experience": 0, "initial_capital": 50_000_000,
            "business_category": "food", "business_sub_category": "cafe",
            "district": "강남구", "store_size_sqm": 33.0,
            "initial_investment": 50_000_000, "monthly_rent": 2_000_000,
            "employee_count": 2, "is_franchise": 0,
            "nearby_competitor_count": 8, "floating_population_level": "high",
        })

        assert "survival_1yr" in result
        assert "risk_level" in result
        assert 0 <= result["survival_1yr"] <= 1
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_batch_prediction(self, sample_labeled_df):
        """배치 추론이 입력 건수와 동일한 결과를 반환하는지"""
        from src.models.xgboost_model import XGBoostModel
        from src.features.builder import FeatureBuilder
        from pipelines.inference_pipeline import InferencePipeline

        builder = FeatureBuilder()
        X, y = builder.fit_transform(sample_labeled_df)

        model = XGBoostModel()
        model.train(X, y)

        pipeline = InferencePipeline(model, builder)
        input_df = sample_labeled_df.drop(columns=[
            "survival_1yr", "survival_3yr", "monthly_revenue",
            "monthly_profit", "risk_score", "break_even_months",
        ])
        results = pipeline.predict_batch(input_df)

        assert len(results) == len(input_df)