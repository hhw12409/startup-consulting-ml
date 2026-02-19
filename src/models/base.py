"""
📁 src/models/base.py
======================
모델 추상 인터페이스.

[패턴] Strategy — 모든 모델이 이 인터페이스를 구현합니다.
                   코드 변경 없이 모델을 교체할 수 있습니다.

사용 예:
    model: BaseModel = XGBoostModel()     # XGBoost 사용
    model: BaseModel = NeuralNetModel()   # 딥러닝으로 교체 (코드 동일)
    model: BaseModel = EnsembleModel()    # 앙상블로 교체 (코드 동일)

    model.train(X, y)
    preds = model.predict(X_new)
    model.save("models/registry/v1")
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseModel(ABC):
    """
    ML 모델 추상 인터페이스.

    모든 모델(XGBoost, PyTorch, Ensemble)은 이 클래스를 상속합니다.
    """

    @abstractmethod
    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
    ) -> dict[str, list[float]]:
        """
        모델 학습.

        Returns:
            학습 히스토리 {"train_loss": [...], "val_loss": [...]}
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        예측 수행.

        Returns:
            태스크별 예측값:
            {
                "survival": [N, 2],    # 1년/3년 생존확률
                "revenue":  [N, 2],    # 월매출/월순이익
                "risk":     [N, 1],    # 리스크 점수
                "break_even": [N, 1],  # 손익분기 개월수
            }
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """모델을 파일로 저장"""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """저장된 모델을 로드"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름 (로깅용)"""
        ...

    def get_info(self) -> dict[str, Any]:
        """모델 메타정보 (기본 구현, 오버라이드 가능)"""
        return {"name": self.name}