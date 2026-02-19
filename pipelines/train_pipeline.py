"""
pipelines/train_pipeline.py
================================
학습 파이프라인 -- DB 기반 전처리부터 평가까지 한 번에 실행.

[패턴] Template Method -- 파이프라인의 골격을 정의
[역할] DB 로드 -> 전처리 -> 피처생성 -> 학습 -> 평가 -> 저장

데이터 흐름:
  stores 테이블 -> 정제 -> cleaned_stores
  -> 라벨 -> labeled_stores
  -> 피처 -> feature_sets + numpy 파일
  -> 학습 -> training_runs + 모델 파일

실행: python scripts/run_train.py
"""

import uuid

import pandas as pd

from config.settings import get_settings
from config.feature_config import FEATURE_CONFIG
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.labeler import LabelGenerator
from src.features.builder import FeatureBuilder
from src.features.store import FeatureStore
from src.models.base import BaseModel
from src.evaluation.metrics import evaluate_model
from src.evaluation.reporter import EvaluationReporter
from src.database.repository import (
    StoreRepository, CleanedStoreRepository,
    LabeledStoreRepository, TrainingRunRepository,
)
from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger(__name__)


class TrainPipeline:
    """
    학습 파이프라인.

    사용법:
        pipeline = TrainPipeline(model=XGBoostModel())
        pipeline.run()
    """

    def __init__(self, model: BaseModel):
        self._model = model
        self._settings = get_settings()
        self._cleaner = DataCleaner()
        self._labeler = LabelGenerator()
        self._builder = FeatureBuilder()
        self._store = FeatureStore()
        self._store_repo = StoreRepository()
        self._cleaned_repo = CleanedStoreRepository()
        self._labeled_repo = LabeledStoreRepository()
        self._training_repo = TrainingRunRepository()

    @timer("전체 학습 파이프라인")
    def run(self, data_path: str = None) -> dict:
        """
        전체 파이프라인 실행.

        단계:
        1. DB에서 데이터 로드
        2. 정제 -> cleaned_stores 저장
        3. 라벨 생성 -> labeled_stores 저장
        4. 피처 생성 -> feature_sets + numpy 저장
        5. Train/Val/Test 분할
        6. 모델 학습
        7. 평가
        8. 모델 + 전처리기 저장 + training_runs 기록

        Returns:
            {"metrics": {...}, "model_info": {...}, "data_sizes": {...}}
        """
        run_id = str(uuid.uuid4())
        pipeline_run_id = str(uuid.uuid4())
        logger.info("학습 실행 ID: %s / 파이프라인 ID: %s", run_id[:8], pipeline_run_id[:8])

        # ── 1. 기존 labeled 데이터가 있으면 재활용 ──
        df = self._try_load_labeled()

        if df is None:
            # ── 1-1. DB에서 원본 로드 ──
            df = self._load_data(data_path)

            # ── 1-2. 정제 ──
            df = self._cleaner.clean(df)
            self._cleaned_repo.delete_old_runs(keep_latest=0)
            self._cleaned_repo.save_cleaned(df, pipeline_run_id)

            # ── 1-3. 라벨 생성 ──
            df = self._labeler.generate(df)
            self._labeled_repo.delete_old_runs(keep_latest=0)
            self._labeled_repo.save_labeled(df, pipeline_run_id)
        else:
            logger.info("기존 labeled 데이터 재활용: %d행", len(df))

        # ── 4. 피처 생성 ──
        X, y = self._builder.fit_transform(df)
        target_columns = [t for t in FEATURE_CONFIG.targets if t in df.columns]

        # ── 5. 분할 & 저장 (파일 + DB) ──
        sizes = self._store.save_splits_to_db(
            X, y,
            feature_columns=self._builder._feature_columns,
            target_columns=target_columns,
            pipeline_run_id=pipeline_run_id,
            scaler_params=self._builder.get_scaler_params(),
            encoder_classes=self._builder.get_encoder_classes(),
        )
        X_train, y_train = self._store.load("train")
        X_val, y_val = self._store.load("val")
        X_test, y_test = self._store.load("test")

        # ── 6. 학습 실행 기록 생성 ──
        self._training_repo.create_run(
            run_id=run_id,
            model_type=self._model.name,
            pipeline_run_id=pipeline_run_id,
            train_size=sizes["train"],
            val_size=sizes["val"],
            test_size=sizes["test"],
            n_features=X_train.shape[1],
        )
        self._training_repo.update_status(run_id, "training")

        # ── 7. 학습 ──
        logger.info("━━━ 모델 학습: %s ━━━", self._model.name)
        self._model.train(X_train, y_train, X_val, y_val)

        # ── 8. 평가 ──
        self._training_repo.update_status(run_id, "evaluating")
        metrics = evaluate_model(self._model, X_test, y_test)

        reporter = EvaluationReporter()
        reporter.generate(
            metrics, self._model.get_info(),
            save_path=f"{self._settings.LOG_DIR}/eval_report.json",
        )

        # ── 9. 저장 ──
        model_path = f"{self._settings.MODEL_REGISTRY}/best_model"
        self._model.save(model_path)
        self._builder.save_artifacts(self._settings.MODEL_ARTIFACTS)

        # ── 10. 학습 실행 기록 완료 ──
        self._training_repo.update_status(
            run_id, "completed",
            metrics=metrics,
            model_path=model_path,
            artifacts_path=self._settings.MODEL_ARTIFACTS,
        )

        logger.info("파이프라인 완료 (run=%s)", run_id[:8])
        return {"metrics": metrics, "model_info": self._model.get_info(), "data_sizes": sizes}

    def _try_load_labeled(self) -> pd.DataFrame | None:
        """
        labeled_stores에 이미 데이터가 있으면 로드.
        없으면 None 반환하여 전체 파이프라인을 실행하도록 한다.
        """
        try:
            latest_id = self._labeled_repo.get_latest_run_id()
            if latest_id:
                df = self._labeled_repo.to_dataframe(latest_id)
                if not df.empty and "survival_1yr" in df.columns:
                    return df
        except Exception as e:
            logger.warning("labeled_stores 로드 실패: %s", e)
        return None

    def _load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        데이터 로드. DB를 우선으로, CSV fallback, 최후에 더미 데이터.

        Args:
            data_path: CSV 경로 (지정 시 해당 파일 사용)
        """
        # 1. CSV 경로가 직접 지정된 경우
        if data_path:
            from pathlib import Path
            from src.utils import io
            if Path(data_path).exists():
                logger.info("CSV 파일에서 로드: %s", data_path)
                return io.load_csv(data_path)

        # 2. DB에서 로드
        try:
            df = self._store_repo.to_dataframe()
            if not df.empty:
                logger.info("DB(stores)에서 로드: %d행", len(df))
                return df
        except Exception as e:
            logger.warning("DB 로드 실패: %s", e)

        # 3. 더미 데이터 (DB/CSV 모두 없을 때)
        logger.warning("원본 데이터 없음 -> 더미 데이터 사용")
        return self._create_dummy_data()

    def _create_dummy_data(self, n: int = 5000) -> pd.DataFrame:
        """테스트용 더미 데이터"""
        import numpy as np
        np.random.seed(42)

        return pd.DataFrame({
            "age": np.random.randint(20, 65, n),
            "gender": np.random.choice(["M", "F"], n),
            "education_level": np.random.choice(["high_school", "bachelor", "master"], n, p=[0.3, 0.55, 0.15]),
            "experience_years": np.random.randint(0, 25, n),
            "has_related_experience": np.random.choice([0, 1], n, p=[0.4, 0.6]),
            "has_startup_experience": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "initial_capital": np.random.randint(10_000_000, 500_000_000, n),
            "business_category": np.random.choice(["food", "retail", "service", "it", "education"], n),
            "business_sub_category": np.random.choice(["cafe", "chicken", "beauty", "academy", "online"], n),
            "district": np.random.choice(["강남구", "서초구", "마포구", "종로구", "영등포구", "성동구"], n),
            "store_size_sqm": np.random.uniform(10, 150, n),
            "initial_investment": np.random.randint(10_000_000, 300_000_000, n),
            "monthly_rent": np.random.randint(500_000, 10_000_000, n),
            "employee_count": np.random.randint(0, 10, n),
            "is_franchise": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "nearby_competitor_count": np.random.randint(0, 30, n),
            "floating_population_level": np.random.choice(["low", "medium", "high"], n),
        })
