"""
tests/integration/test_db_pipeline.py
=========================================
DB 기반 파이프라인 통합 테스트.

SQLite 인메모리 DB를 사용하여 전체 파이프라인 흐름을 검증합니다.
"""

import uuid

import numpy as np
import pandas as pd
import pytest


class TestDBPipelineFlow:
    """DB 기반 파이프라인 전체 흐름 테스트"""

    def test_clean_label_flow(self, mock_get_session, sample_raw_df):
        """정제 -> 라벨링 -> DB 저장 흐름"""
        from src.preprocessing.cleaner import DataCleaner
        from src.preprocessing.labeler import LabelGenerator
        from src.database.repository import (
            CleanedStoreRepository, LabeledStoreRepository,
        )

        pipeline_run_id = str(uuid.uuid4())

        # 정제
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_df)

        cleaned_repo = CleanedStoreRepository()
        cleaned_repo.save_cleaned(cleaned, pipeline_run_id)

        # 라벨
        labeler = LabelGenerator()
        labeled = labeler.generate(cleaned)

        labeled_repo = LabeledStoreRepository()
        labeled_repo.save_labeled(labeled, pipeline_run_id)

        # 검증: DB에서 읽어도 라벨이 존재
        df = labeled_repo.to_dataframe(pipeline_run_id)
        assert not df.empty
        assert "survival_1yr" in df.columns
        assert "risk_score" in df.columns

    def test_feature_db_save(self, mock_get_session, sample_labeled_df):
        """피처 엔지니어링 -> DB 저장 흐름"""
        from src.features.builder import FeatureBuilder
        from src.database.repository import FeatureSetRepository

        builder = FeatureBuilder()
        X, y = builder.fit_transform(sample_labeled_df)

        pipeline_run_id = str(uuid.uuid4())
        repo = FeatureSetRepository()
        repo.save_feature_set(
            X, y,
            feature_columns=builder._feature_columns,
            target_columns=["survival_1yr", "survival_3yr",
                            "monthly_revenue", "monthly_profit",
                            "risk_score", "break_even_months"],
            pipeline_run_id=pipeline_run_id,
            scaler_params=builder.get_scaler_params(),
            encoder_classes=builder.get_encoder_classes(),
        )

        # 검증: DB에서 로드
        X_loaded, y_loaded, feat_cols, tgt_cols = repo.load_feature_set(pipeline_run_id)
        assert X_loaded.shape == X.shape
        assert y_loaded.shape == y.shape
        assert np.allclose(X, X_loaded)

    def test_training_run_lifecycle(self, mock_get_session):
        """학습 실행 기록 생명주기: 생성 -> 학습중 -> 평가중 -> 완료"""
        from src.database.repository import TrainingRunRepository

        repo = TrainingRunRepository()
        run_id = str(uuid.uuid4())

        # 생성
        repo.create_run(
            run_id=run_id,
            model_type="xgboost",
            train_size=100,
            val_size=10,
            test_size=10,
        )

        run = repo.get_run(run_id)
        assert run.status == "started"

        # 학습중
        repo.update_status(run_id, "training")
        run = repo.get_run(run_id)
        assert run.status == "training"

        # 평가중
        repo.update_status(run_id, "evaluating")
        run = repo.get_run(run_id)
        assert run.status == "evaluating"

        # 완료
        metrics = {"survival_1yr_acc": 0.85, "survival_3yr_acc": 0.72}
        repo.update_status(
            run_id, "completed",
            metrics=metrics,
            model_path="models/registry/best_model",
        )

        run = repo.get_run(run_id)
        assert run.status == "completed"
        assert run.completed_at is not None

    def test_full_pipeline_with_dummy(self, mock_get_session):
        """더미 데이터로 전체 파이프라인(정제->라벨->피처->DB저장) 흐름"""
        from src.preprocessing.cleaner import DataCleaner
        from src.preprocessing.labeler import LabelGenerator
        from src.features.builder import FeatureBuilder
        from src.database.repository import (
            CleanedStoreRepository, LabeledStoreRepository,
            FeatureSetRepository,
        )

        # 더미 데이터 생성
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "age": np.random.randint(20, 65, n),
            "gender": np.random.choice(["M", "F"], n),
            "education_level": np.random.choice(
                ["high_school", "bachelor", "master"], n),
            "experience_years": np.random.randint(0, 25, n),
            "has_related_experience": np.random.choice([0, 1], n),
            "has_startup_experience": np.random.choice([0, 1], n),
            "initial_capital": np.random.randint(10_000_000, 500_000_000, n),
            "business_category": np.random.choice(
                ["food", "retail", "service"], n),
            "business_sub_category": np.random.choice(
                ["cafe", "chicken", "beauty"], n),
            "district": np.random.choice(["강남구", "마포구", "종로구"], n),
            "store_size_sqm": np.random.uniform(10, 150, n),
            "initial_investment": np.random.randint(10_000_000, 300_000_000, n),
            "monthly_rent": np.random.randint(500_000, 10_000_000, n),
            "employee_count": np.random.randint(0, 10, n),
            "is_franchise": np.random.choice([0, 1], n),
            "nearby_competitor_count": np.random.randint(0, 30, n),
            "floating_population_level": np.random.choice(
                ["low", "medium", "high"], n),
        })

        pipeline_run_id = str(uuid.uuid4())

        # 1. 정제 -> DB
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df)
        cleaned_repo = CleanedStoreRepository()
        cleaned_repo.save_cleaned(cleaned, pipeline_run_id)

        # 2. 라벨 -> DB
        labeler = LabelGenerator()
        labeled = labeler.generate(cleaned)
        labeled_repo = LabeledStoreRepository()
        labeled_repo.save_labeled(labeled, pipeline_run_id)

        # 3. 피처 -> DB
        builder = FeatureBuilder()
        X, y = builder.fit_transform(labeled)

        feature_repo = FeatureSetRepository()
        feature_repo.save_feature_set(
            X, y,
            feature_columns=builder._feature_columns,
            target_columns=["survival_1yr", "survival_3yr",
                            "monthly_revenue", "monthly_profit",
                            "risk_score", "break_even_months"],
            pipeline_run_id=pipeline_run_id,
        )

        # 검증
        cleaned_loaded = cleaned_repo.to_dataframe(pipeline_run_id)
        labeled_loaded = labeled_repo.to_dataframe(pipeline_run_id)
        X_loaded, y_loaded, _, _ = feature_repo.load_feature_set(pipeline_run_id)

        assert len(cleaned_loaded) > 0
        assert len(labeled_loaded) > 0
        assert "survival_1yr" in labeled_loaded.columns
        assert X_loaded.shape[0] > 0
        assert np.allclose(X, X_loaded)
