"""
모델별 하이퍼파라미터 정의.

모델을 교체하거나 튜닝할 때 이 파일만 수정하면 됩니다.
코드 안에 매직넘버가 흩어지는 것을 방지합니다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralNetConfig:
    """PyTorch 딥러닝 모델 하이퍼파라미터"""
    hidden_dims: tuple[int, ...] = (256, 128, 64)  # 히든 레이어 크기
    dropout_rate: float = 0.3                       # 드롭아웃 비율
    batch_norm: bool = True                         # 배치 정규화 사용
    # 손실 함수 태스크별 가중치 (합이 1이 아니어도 됨)
    loss_weight_survival: float = 1.0
    loss_weight_revenue: float = 0.3
    loss_weight_risk: float = 0.5
    loss_weight_break_even: float = 0.2


@dataclass(frozen=True)
class XGBoostConfig:
    """XGBoost 모델 하이퍼파라미터"""
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    random_state: int = 42


# ── 전역 인스턴스 ──
NEURAL_NET_CONFIG = NeuralNetConfig()
XGBOOST_CONFIG = XGBoostConfig()