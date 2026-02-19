"""
📁 src/models/ensemble.py
===========================
앙상블 모델 — 여러 모델의 예측을 결합합니다.

[패턴] Composite + Strategy
[권장] Phase 3에서 개별 모델 검증 후 사용
"""

from pathlib import Path
from typing import Any, Optional
import numpy as np

from src.models.base import BaseModel
from src.utils.logger import get_logger
from src.utils import io

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """
    가중 평균 앙상블.

    사용법:
        ensemble = EnsembleModel(
            models=[xgb_model, nn_model],
            weights=[0.6, 0.4],   # XGBoost 60%, DL 40%
        )
        preds = ensemble.predict(X)
    """

    def __init__(self, models: list[BaseModel] = None, weights: list[float] = None):
        self._models = models or []
        self._weights = weights or [1.0 / len(self._models)] * len(self._models) if self._models else []

    @property
    def name(self) -> str:
        return "EnsembleModel"

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """각 모델을 개별 학습"""
        histories = {}
        for i, model in enumerate(self._models):
            logger.info("앙상블 %d/%d: %s 학습", i + 1, len(self._models), model.name)
            h = model.train(X_train, y_train, X_val, y_val)
            histories[model.name] = h
        return histories

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """각 모델 예측을 가중 평균"""
        all_preds = [m.predict(X) for m in self._models]
        keys = all_preds[0].keys()

        result = {}
        for key in keys:
            weighted = sum(p[key] * w for p, w in zip(all_preds, self._weights))
            result[key] = weighted

        return result

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self._models):
            model.save(f"{path}_sub{i}_{model.name}")
        io.save_pickle(self._weights, f"{path}_weights.pkl")

    def load(self, path: str) -> None:
        self._weights = io.load_pickle(f"{path}_weights.pkl")
        for i, model in enumerate(self._models):
            model.load(f"{path}_sub{i}_{model.name}")

    def get_info(self):
        return {
            "name": self.name,
            "sub_models": [m.get_info() for m in self._models],
            "weights": self._weights,
        }