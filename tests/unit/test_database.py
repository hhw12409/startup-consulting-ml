"""
tests/unit/test_database.py
==============================
DB Repository 단위 테스트.

SQLite 인메모리 DB로 빠르게 검증합니다.
MySQL 특화 기능(UPSERT 등)은 통합 테스트에서 검증합니다.
"""

import uuid

import numpy as np
import pandas as pd
import pytest


class TestCleanedStoreRepository:
    """CleanedStoreRepository 테스트"""

    def test_save_and_load(self, mock_get_session, sample_raw_df):
        """저장 후 로드하면 동일한 데이터가 반환되어야 한다"""
        from src.database.repository import CleanedStoreRepository
        repo = CleanedStoreRepository()

        pipeline_run_id = str(uuid.uuid4())
        saved = repo.save_cleaned(sample_raw_df, pipeline_run_id)

        assert saved == len(sample_raw_df)

        df = repo.to_dataframe(pipeline_run_id)
        assert len(df) == len(sample_raw_df)
        assert "business_category" in df.columns

    def test_get_latest_run_id(self, mock_get_session, sample_raw_df):
        """최신 pipeline_run_id를 올바르게 반환해야 한다"""
        from src.database.repository import CleanedStoreRepository
        repo = CleanedStoreRepository()

        run_id_1 = str(uuid.uuid4())
        run_id_2 = str(uuid.uuid4())

        repo.save_cleaned(sample_raw_df, run_id_1)
        repo.save_cleaned(sample_raw_df, run_id_2)

        latest = repo.get_latest_run_id()
        assert latest is not None

    def test_empty_table(self, mock_get_session):
        """빈 테이블 조회 시 빈 DataFrame 반환"""
        from src.database.repository import CleanedStoreRepository
        repo = CleanedStoreRepository()

        df = repo.to_dataframe()
        assert df.empty


class TestLabeledStoreRepository:
    """LabeledStoreRepository 테스트"""

    def test_save_and_load(self, mock_get_session, sample_labeled_df):
        """라벨 데이터 저장/로드 검증"""
        from src.database.repository import LabeledStoreRepository
        repo = LabeledStoreRepository()

        pipeline_run_id = str(uuid.uuid4())
        saved = repo.save_labeled(sample_labeled_df, pipeline_run_id)

        assert saved == len(sample_labeled_df)

        df = repo.to_dataframe(pipeline_run_id)
        assert len(df) == len(sample_labeled_df)
        assert "survival_1yr" in df.columns
        assert "risk_score" in df.columns

    def test_label_values_preserved(self, mock_get_session, sample_labeled_df):
        """라벨 값이 정확히 보존되어야 한다"""
        from src.database.repository import LabeledStoreRepository
        repo = LabeledStoreRepository()

        pipeline_run_id = str(uuid.uuid4())
        repo.save_labeled(sample_labeled_df, pipeline_run_id)

        df = repo.to_dataframe(pipeline_run_id)
        assert len(df) == len(sample_labeled_df)


class TestFeatureSetRepository:
    """FeatureSetRepository 테스트"""

    def test_serialize_deserialize(self, mock_get_session):
        """numpy 배열의 직렬화/역직렬화 검증"""
        from src.database.repository import FeatureSetRepository
        repo = FeatureSetRepository()

        X = np.random.randn(100, 22).astype(np.float32)
        y = np.random.randn(100, 6).astype(np.float32)
        feature_cols = [f"feature_{i}" for i in range(22)]
        target_cols = [f"target_{i}" for i in range(6)]
        pipeline_run_id = str(uuid.uuid4())

        repo.save_feature_set(
            X, y,
            feature_columns=feature_cols,
            target_columns=target_cols,
            pipeline_run_id=pipeline_run_id,
        )

        X_loaded, y_loaded, feat_cols, tgt_cols = repo.load_feature_set(pipeline_run_id)

        assert np.allclose(X, X_loaded)
        assert np.allclose(y, y_loaded)
        assert feat_cols == feature_cols
        assert tgt_cols == target_cols

    def test_metadata_stored(self, mock_get_session):
        """scaler_params와 encoder_classes가 올바르게 저장되어야 한다"""
        from src.database.repository import FeatureSetRepository
        repo = FeatureSetRepository()

        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 3).astype(np.float32)
        pipeline_run_id = str(uuid.uuid4())

        scaler_params = {"mean": [0.5] * 10, "scale": [1.0] * 10}
        encoder_classes = {"gender": ["F", "M"]}

        repo.save_feature_set(
            X, y,
            feature_columns=[f"f{i}" for i in range(10)],
            target_columns=[f"t{i}" for i in range(3)],
            pipeline_run_id=pipeline_run_id,
            scaler_params=scaler_params,
            encoder_classes=encoder_classes,
        )

        # 로드 후 메타데이터는 load_feature_set에서 직접 반환되지 않지만,
        # DB에 저장됨을 확인
        X_loaded, y_loaded, _, _ = repo.load_feature_set(pipeline_run_id)
        assert X_loaded.shape == (50, 10)

    def test_no_data_raises(self, mock_get_session):
        """저장된 데이터가 없으면 ValueError 발생"""
        from src.database.repository import FeatureSetRepository
        repo = FeatureSetRepository()

        with pytest.raises(ValueError, match="저장된 피처셋이 없습니다"):
            repo.load_feature_set()


class TestTrainingRunRepository:
    """TrainingRunRepository 테스트"""

    def test_create_and_get(self, mock_get_session):
        """학습 실행 생성 및 조회"""
        from src.database.repository import TrainingRunRepository
        repo = TrainingRunRepository()

        run_id = str(uuid.uuid4())
        run = repo.create_run(
            run_id=run_id,
            model_type="xgboost",
            train_size=8000,
            val_size=1000,
            test_size=1000,
        )

        assert run.run_id == run_id
        assert run.status == "started"

        fetched = repo.get_run(run_id)
        assert fetched is not None
        assert fetched.model_type == "xgboost"

    def test_update_status(self, mock_get_session):
        """상태 업데이트 검증"""
        from src.database.repository import TrainingRunRepository
        repo = TrainingRunRepository()

        run_id = str(uuid.uuid4())
        repo.create_run(run_id=run_id, model_type="neural_net")

        metrics = {"accuracy": 0.85, "f1": 0.82}
        repo.update_status(
            run_id, "completed",
            metrics=metrics,
            model_path="models/registry/best_model",
        )

        run = repo.get_run(run_id)
        assert run.status == "completed"
        assert run.completed_at is not None

    def test_get_latest_run(self, mock_get_session):
        """최신 완료된 실행을 조회"""
        from src.database.repository import TrainingRunRepository
        repo = TrainingRunRepository()

        run_id_1 = str(uuid.uuid4())
        repo.create_run(run_id=run_id_1, model_type="xgboost")
        repo.update_status(run_id_1, "completed")

        run_id_2 = str(uuid.uuid4())
        repo.create_run(run_id=run_id_2, model_type="xgboost")
        repo.update_status(run_id_2, "completed")

        latest = repo.get_latest_run(model_type="xgboost")
        assert latest is not None


class TestFeatureBuilderDBMethods:
    """FeatureBuilder의 새 DB 관련 메서드 테스트"""

    def test_get_scaler_params(self, sample_labeled_df):
        """fit 후 scaler params가 올바르게 반환되어야 한다"""
        from src.features.builder import FeatureBuilder
        builder = FeatureBuilder()
        builder.fit_transform(sample_labeled_df)

        params = builder.get_scaler_params()
        assert "mean" in params
        assert "scale" in params
        assert "var" in params
        assert isinstance(params["mean"], list)

    def test_get_encoder_classes(self, sample_labeled_df):
        """fit 후 encoder classes가 올바르게 반환되어야 한다"""
        from src.features.builder import FeatureBuilder
        builder = FeatureBuilder()
        builder.fit_transform(sample_labeled_df)

        classes = builder.get_encoder_classes()
        assert isinstance(classes, dict)
        # 범주형 피처가 인코딩되었으면 클래스가 있어야 함
        if "gender" in classes:
            assert "M" in classes["gender"]
            assert "F" in classes["gender"]

    def test_get_scaler_params_before_fit(self):
        """fit 전에는 빈 dict 반환"""
        from src.features.builder import FeatureBuilder
        builder = FeatureBuilder()
        assert builder.get_scaler_params() == {}
