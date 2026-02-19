"""
📁 src/serving/dependencies.py
=================================
FastAPI 의존성 주입 설정.

[패턴] Factory — 모델, 전처리기, LLM 컨설턴트 인스턴스를 생성합니다.
"""

import os
from functools import lru_cache

from config.settings import get_settings
from src.models.base import BaseModel
from src.models.xgboost_model import XGBoostModel
from src.features.builder import FeatureBuilder
from src.serving.predictor import Predictor
from src.llm.consultant import StartupConsultant
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_predictor() -> Predictor:
    """Predictor 싱글턴. 모델 + 전처리기를 로드."""
    settings = get_settings()

    model: BaseModel = XGBoostModel()
    model_path = f"{settings.MODEL_REGISTRY}/best_model"
    if os.path.exists(f"{model_path}.pkl"):
        model.load(model_path)
        logger.info("모델 로드 완료: %s", model_path)
    else:
        logger.warning("학습된 모델 없음: %s (make train 실행 필요)", model_path)

    artifact_path = settings.MODEL_ARTIFACTS
    if os.path.exists(f"{artifact_path}/scaler.pkl"):
        builder = FeatureBuilder.load_artifacts(artifact_path)
    else:
        logger.warning("전처리기 없음: %s", artifact_path)
        builder = FeatureBuilder()

    return Predictor(model=model, feature_builder=builder)


@lru_cache()
def get_consultant() -> StartupConsultant:
    """LLM 컨설턴트 싱글턴. Ollama 로컬 모델 사용."""
    consultant = StartupConsultant()
    logger.info("LLM 컨설턴트 활성: %s", consultant.active_llm)
    return consultant