"""
📁 src/evaluation/metrics.py
==============================
모델 평가 메트릭 계산.

[역할] 분류(생존 예측) + 회귀(매출 예측) 메트릭을 한 번에 계산합니다.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
)

from src.models.base import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
        model: BaseModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
) -> dict[str, float]:
    """
    모델 전체 평가.

    Args:
        model: 학습된 모델
        X_test: 테스트 피처
        y_test: 테스트 라벨 [N, 6]
               (survival_1yr, survival_3yr, revenue, profit, risk, break_even)

    Returns:
        메트릭 딕셔너리 (예: {"survival_1yr_accuracy": 0.85, ...})
    """
    preds = model.predict(X_test)
    metrics = {}

    # ── 생존 예측 (분류) ──
    if "survival" in preds:
        for i, tag in enumerate(["1yr", "3yr"]):
            y_true = (y_test[:, i] > 0.5).astype(int)
            y_pred = (preds["survival"][:, i] > 0.5).astype(int)
            y_prob = preds["survival"][:, i]

            metrics[f"survival_{tag}_acc"] = accuracy_score(y_true, y_pred)
            metrics[f"survival_{tag}_f1"] = f1_score(y_true, y_pred, zero_division=0)
            try:
                metrics[f"survival_{tag}_auc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics[f"survival_{tag}_auc"] = 0.0

    # ── 매출 예측 (회귀) ──
    if "revenue" in preds:
        metrics["revenue_mae"] = mean_absolute_error(y_test[:, 2], preds["revenue"][:, 0])
        metrics["revenue_r2"] = r2_score(y_test[:, 2], preds["revenue"][:, 0])
        metrics["profit_mae"] = mean_absolute_error(y_test[:, 3], preds["revenue"][:, 1])

    # ── 리스크 (회귀) ──
    if "risk" in preds:
        metrics["risk_mae"] = mean_absolute_error(y_test[:, 4], preds["risk"][:, 0])

    # ── 로깅 ──
    logger.info("━━━ 평가 결과: %s ━━━", model.name)
    for k, v in sorted(metrics.items()):
        logger.info("  %-25s %.4f", k, v)

    return metrics