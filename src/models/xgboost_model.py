"""
📁 src/models/xgboost_model.py
================================
XGBoost 기반 창업 예측 모델.

[패턴] Strategy — BaseModel 인터페이스를 구현
[역할] 정형 데이터에서 빠르고 강력한 baseline 모델
[권장] 데이터 1만건 이하일 때 이 모델부터 시작하세요

특징:
  - 태스크별 독립 모델 (분류 2개 + 회귀 4개)
  - Feature Importance 확인 가능
  - 학습 시간이 짧아 빠른 실험 가능
"""

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb

from src.models.base import BaseModel
from config.model_config import XGBOOST_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost Multi-task 모델.

    사용법:
        model = XGBoostModel()
        model.train(X_train, y_train, X_val, y_val)
        preds = model.predict(X_test)
        model.save("models/registry/v1/xgboost")
    """

    # 태스크 정의: (이름, 타겟 컬럼 인덱스, 유형)
    TASKS = [
        ("survival_1yr", 0, "classifier"),
        ("survival_3yr", 1, "classifier"),
        ("revenue",      2, "regressor"),
        ("profit",       3, "regressor"),
        ("risk",         4, "regressor"),
        ("break_even",   5, "regressor"),
    ]

    def __init__(self, config: Any = None):
        cfg = config or XGBOOST_CONFIG
        self._params = {
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "subsample": cfg.subsample,
            "colsample_bytree": cfg.colsample_bytree,
            "min_child_weight": cfg.min_child_weight,
            "random_state": cfg.random_state,
        }
        self._models: dict[str, Any] = {}
        self._is_trained = False

    @property
    def name(self) -> str:
        return "XGBoostModel"

    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("=== XGBoost 학습 시작 (%d 태스크) ===", len(self.TASKS))

        for task_name, col_idx, task_type in self.TASKS:
            logger.info("  학습 중: %s (%s)", task_name, task_type)

            if task_type == "classifier":
                model = xgb.XGBClassifier(**self._params, objective="binary:logistic", eval_metric="logloss")
                yt = (y_train[:, col_idx] > 0.5).astype(int)
                yv = (y_val[:, col_idx] > 0.5).astype(int) if y_val is not None else None
            else:
                model = xgb.XGBRegressor(**self._params, objective="reg:squarederror")
                yt = y_train[:, col_idx]
                yv = y_val[:, col_idx] if y_val is not None else None

            eval_set = [(X_val, yv)] if yv is not None else None
            model.fit(X_train, yt, eval_set=eval_set, verbose=False)
            self._models[task_name] = model

        self._is_trained = True
        logger.info("=== XGBoost 학습 완료 ===")
        return {"train_loss": [], "val_loss": []}

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self._is_trained:
            raise RuntimeError("학습되지 않은 모델")

        # 생존확률 (분류 → predict_proba)
        p1 = self._models["survival_1yr"].predict_proba(X)[:, 1:]
        p3 = self._models["survival_3yr"].predict_proba(X)[:, 1:]

        return {
            "survival":   np.hstack([p1, p3]),
            "revenue":    np.column_stack([
                self._models["revenue"].predict(X),
                self._models["profit"].predict(X),
            ]),
            "risk":       self._models["risk"].predict(X).reshape(-1, 1).clip(0, 1),
            "break_even": self._models["break_even"].predict(X).reshape(-1, 1).clip(1, None),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self._models, f)
        logger.info("모델 저장: %s.pkl", path)

    def load(self, path: str) -> None:
        with open(f"{path}.pkl", "rb") as f:
            self._models = pickle.load(f)
        self._is_trained = True
        logger.info("모델 로드: %s.pkl", path)

    def get_feature_importance(self, feature_names: list[str] = None) -> dict[str, list]:
        """피처 중요도 반환 (XGBoost 고유 기능)"""
        importance = {}
        for task_name, model in self._models.items():
            scores = model.feature_importances_
            if feature_names:
                importance[task_name] = sorted(
                    zip(feature_names, scores), key=lambda x: -x[1]
                )[:10]  # 상위 10개
            else:
                importance[task_name] = scores.tolist()
        return importance

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "xgboost",
            "tasks": list(self._models.keys()),
            "params": self._params,
            "is_trained": self._is_trained,
        }