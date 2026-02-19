"""
📁 pipelines/inference_pipeline.py
====================================
추론 파이프라인.

[패턴] Chain of Responsibility — 입력 → 검증 → 전처리 → 추론 → 후처리
[역할] API가 아닌 배치 추론이나 노트북에서 직접 사용할 때 편리합니다.

API 서빙(serving/predictor.py)은 HTTP 요청 처리에 특화되어 있고,
이 파이프라인은 순수 Python 함수로 사용할 수 있습니다.
"""

from typing import Optional
import pandas as pd
import numpy as np

from src.models.base import BaseModel
from src.features.builder import FeatureBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    추론 파이프라인.

    사용법:
        # 단건 추론
        pipeline = InferencePipeline(model, builder)
        result = pipeline.predict_single({"age": 35, "business_category": "food", ...})

        # 배치 추론
        results = pipeline.predict_batch(df)
    """

    def __init__(self, model: BaseModel, feature_builder: FeatureBuilder):
        self._model = model
        self._builder = feature_builder

    def predict_single(self, input_data: dict) -> dict:
        """
        단건 추론.

        Args:
            input_data: 원본 입력 딕셔너리

        Returns:
            예측 결과 딕셔너리
        """
        df = pd.DataFrame([input_data])
        results = self.predict_batch(df)
        return results[0] if results else {}

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        """
        배치 추론.

        Args:
            df: 여러 건의 입력 DataFrame

        Returns:
            각 건별 예측 결과 리스트
        """
        logger.info("배치 추론: %d건", len(df))

        # 1) 전처리
        X = self._builder.transform(df)

        # 2) 모델 추론
        raw = self._model.predict(X)

        # 3) 후처리: 각 행별로 결과 생성
        results = []
        for i in range(len(df)):
            result = {
                "survival_1yr": float(raw["survival"][i, 0]),
                "survival_3yr": float(raw["survival"][i, 1]),
                "monthly_revenue": int(raw["revenue"][i, 0]),
                "monthly_profit": int(raw["revenue"][i, 1]),
                "risk_score": float(raw["risk"][i, 0]),
                "break_even_months": max(1, int(raw["break_even"][i, 0])),
            }

            # 리스크 등급
            rs = result["risk_score"]
            result["risk_level"] = (
                "LOW" if rs < 0.3 else
                "MEDIUM" if rs < 0.6 else
                "HIGH" if rs < 0.8 else
                "CRITICAL"
            )

            results.append(result)

        logger.info("배치 추론 완료: %d건", len(results))
        return results

    @classmethod
    def from_saved(
            cls,
            model_path: str = None,
            artifact_path: str = None,
            model_type: str = "xgboost",
    ) -> "InferencePipeline":
        """
        [패턴] Factory Method -- 저장된 모델에서 파이프라인을 생성합니다.

        경로를 지정하지 않으면 DB의 training_runs에서 최신 모델을 조회합니다.

        사용법:
            # 직접 지정
            pipeline = InferencePipeline.from_saved(
                model_path="models/registry/best_model",
                artifact_path="models/artifacts/",
            )

            # DB에서 최신 모델 자동 조회
            pipeline = InferencePipeline.from_saved()
        """
        # DB에서 최신 학습 실행 조회 (경로 미지정 시)
        if not model_path or not artifact_path:
            try:
                from src.database.repository import TrainingRunRepository
                repo = TrainingRunRepository()
                latest = repo.get_latest_run(model_type=model_type)
                if latest:
                    model_path = model_path or latest.model_path
                    artifact_path = artifact_path or latest.artifacts_path
                    logger.info("DB에서 최신 모델 조회: run=%s", latest.run_id[:8])
            except Exception as e:
                logger.warning("DB 모델 조회 실패: %s", e)

        # 기본 경로 fallback
        from config.settings import get_settings
        settings = get_settings()
        model_path = model_path or f"{settings.MODEL_REGISTRY}/best_model"
        artifact_path = artifact_path or settings.MODEL_ARTIFACTS

        if model_type == "xgboost":
            from src.models.xgboost_model import XGBoostModel
            model = XGBoostModel()
        else:
            from src.models.neural_net import NeuralNetModel
            model = NeuralNetModel()

        model.load(model_path)
        builder = FeatureBuilder.load_artifacts(artifact_path)

        logger.info("추론 파이프라인 생성: model=%s", model.name)
        return cls(model, builder)