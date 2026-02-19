"""
📁 src/features/scalers.py
============================
수치형 변수 스케일링 모듈.

[패턴] Strategy — 스케일링 방식을 교체 가능하게 분리합니다.

스케일링 방식:
  - StandardScaler: 평균=0, 표준편차=1 (가장 일반적, 딥러닝에 적합)
  - MinMaxScaler: 0~1 범위 (트리 모델에서는 불필요하지만 해도 무방)
  - RobustScaler: 중앙값/IQR 기반 (이상치에 강건, 투자금/매출 같은 컬럼에 적합)

실무 팁:
  - 트리 모델(XGBoost)은 스케일링이 필요 없지만 해도 성능에 해가 없음
  - 딥러닝 모델은 반드시 스케일링 필요 (안 하면 학습이 불안정)
  - 두 모델을 같이 쓸 거면 스케일링하는 게 안전
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseScaler(ABC):
    """스케일러 공통 인터페이스"""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseScaler":
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """스케일링을 되돌림 (예측값 → 원래 단위로 복원할 때)"""
        ...


class StandardScalerWrapper(BaseScaler):
    """StandardScaler (평균=0, 표준편차=1)"""

    def __init__(self):
        self._scaler = StandardScaler()

    def fit(self, X):
        self._scaler.fit(X)
        logger.debug("StandardScaler fit: %d 피처", X.shape[1])
        return self

    def transform(self, X):
        return self._scaler.transform(X)

    def inverse_transform(self, X):
        return self._scaler.inverse_transform(X)


class RobustScalerWrapper(BaseScaler):
    """
    RobustScaler (중앙값/IQR 기반).

    이상치가 많은 컬럼(투자금, 매출)에 적합합니다.
    """

    def __init__(self):
        self._scaler = RobustScaler()

    def fit(self, X):
        self._scaler.fit(X)
        logger.debug("RobustScaler fit: %d 피처", X.shape[1])
        return self

    def transform(self, X):
        return self._scaler.transform(X)

    def inverse_transform(self, X):
        return self._scaler.inverse_transform(X)


# ================================================================
# 스케일러 팩토리
# ================================================================
class ScalerFactory:
    """
    [패턴] Factory — 모델 타입에 따라 적절한 스케일러를 선택합니다.
    """

    @staticmethod
    def create(model_type: str = "standard") -> BaseScaler:
        """
        Args:
            model_type: "standard" | "robust"
        """
        if model_type == "robust":
            return RobustScalerWrapper()
        return StandardScalerWrapper()