"""
📁 src/features/encoders.py
=============================
범주형 변수 인코딩 모듈.

[패턴] Strategy — 인코딩 방식을 교체 가능하게 분리합니다.
[위치] builder.py에서 호출하여 사용합니다.

인코딩 방식:
  - LabelEncoding: 카테고리가 많을 때 (district 등) → 차원 증가 없음
  - OneHotEncoding: 카테고리가 적을 때 (gender 등) → 관계 없는 범주에 적합
  - TargetEncoding: 타겟과의 관계를 반영 (Phase 2에서 추가)

실무 팁:
  - 트리 모델(XGBoost)은 LabelEncoding으로 충분
  - 딥러닝 모델은 OneHot 또는 Embedding이 더 좋음
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.utils.logger import get_logger
from src.utils import io

logger = get_logger(__name__)


# ================================================================
# 인코더 인터페이스 (Strategy 패턴)
# ================================================================
class BaseEncoder(ABC):
    """인코더 공통 인터페이스. fit → transform 순서를 지킵니다."""

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseEncoder":
        """학습 데이터로 인코딩 규칙을 학습합니다."""
        ...

    @abstractmethod
    def transform(self, series: pd.Series) -> np.ndarray:
        """학습된 규칙으로 변환합니다."""
        ...

    @abstractmethod
    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        """원래 값으로 복원합니다."""
        ...


# ================================================================
# LabelEncoder 래퍼
# ================================================================
class SafeLabelEncoder(BaseEncoder):
    """
    안전한 LabelEncoder.

    sklearn LabelEncoder의 문제점을 보완합니다:
    - 학습 때 없던 새로운 카테고리 → -1로 처리 (에러 대신)
    - NaN → 'unknown'으로 처리

    사용법:
        enc = SafeLabelEncoder()
        enc.fit(df["district"])
        encoded = enc.transform(df["district"])
    """

    def __init__(self, unknown_value: int = -1):
        self._encoder = LabelEncoder()
        self._unknown_value = unknown_value
        self._classes: set = set()

    def fit(self, series: pd.Series) -> "SafeLabelEncoder":
        clean = series.fillna("unknown").astype(str)
        self._encoder.fit(clean)
        self._classes = set(self._encoder.classes_)
        logger.debug("LabelEncoder fit: %d 클래스", len(self._classes))
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        clean = series.fillna("unknown").astype(str)

        # 학습 때 없던 값 → unknown_value(-1)로 처리
        result = np.array([
            self._encoder.transform([v])[0] if v in self._classes else self._unknown_value
            for v in clean
        ])
        return result

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        # -1(unknown)은 "unknown"으로 복원
        mask = encoded != self._unknown_value
        result = np.full(len(encoded), "unknown", dtype=object)
        if mask.any():
            result[mask] = self._encoder.inverse_transform(encoded[mask].astype(int))
        return result


# ================================================================
# OneHotEncoder 래퍼
# ================================================================
class SafeOneHotEncoder(BaseEncoder):
    """
    안전한 OneHotEncoder.

    새로운 카테고리가 나타나면 무시합니다 (에러 대신).

    사용법:
        enc = SafeOneHotEncoder()
        enc.fit(df["gender"])                 # 학습: ["M", "F"]
        encoded = enc.transform(df["gender"]) # [1, 0] 또는 [0, 1]
    """

    def __init__(self):
        self._encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def fit(self, series: pd.Series) -> "SafeOneHotEncoder":
        clean = series.fillna("unknown").astype(str).values.reshape(-1, 1)
        self._encoder.fit(clean)
        logger.debug("OneHotEncoder fit: %d 클래스", len(self._encoder.categories_[0]))
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        clean = series.fillna("unknown").astype(str).values.reshape(-1, 1)
        return self._encoder.transform(clean)

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        return self._encoder.inverse_transform(encoded).ravel()


# ================================================================
# 인코더 팩토리 (어떤 컬럼에 어떤 인코더를 쓸지 결정)
# ================================================================
class EncoderFactory:
    """
    [패턴] Factory — 컬럼 특성에 따라 적절한 인코더를 생성합니다.

    규칙:
      - 카테고리 5개 이하 → OneHot (gender, floating_population_level)
      - 카테고리 6개 이상 → Label (district, business_category)
    """

    ONEHOT_THRESHOLD = 5  # 이 이하면 OneHot

    @staticmethod
    def create(series: pd.Series) -> BaseEncoder:
        n_unique = series.nunique()
        if n_unique <= EncoderFactory.ONEHOT_THRESHOLD:
            logger.debug("OneHot 선택: %s (%d 클래스)", series.name, n_unique)
            return SafeOneHotEncoder()
        else:
            logger.debug("Label 선택: %s (%d 클래스)", series.name, n_unique)
            return SafeLabelEncoder()