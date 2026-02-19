"""
📁 src/features/store.py
=========================
피처 저장소.

[패턴] Repository — 피처 데이터의 저장/로드를 추상화
[역할] train/val/test split된 데이터를 numpy로 저장하고 로드합니다.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils import io

logger = get_logger(__name__)


class FeatureStore:
    """
    피처 저장소.

    사용법:
        store = FeatureStore()
        store.save_splits(X, y)            # 자동 split 후 저장
        X_train, y_train = store.load("train")  # 로드
    """

    def __init__(self, base_dir: str = None):
        self._dir = base_dir or get_settings().DATA_MODEL_INPUT

    def save_splits(
            self,
            X: np.ndarray,
            y: np.ndarray,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            random_state: int = 42,
    ) -> dict[str, int]:
        """
        Train/Val/Test로 분할하고 05_model_input/에 저장합니다.

        Args:
            X, y: 전체 피처와 라벨
            val_ratio, test_ratio: 검증/테스트 비율

        Returns:
            각 세트의 크기 {"train": 8000, "val": 1000, "test": 1000}
        """
        # Train+Val / Test 분할
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state,
        )

        # Train / Val 분할
        val_adjusted = val_ratio / (1 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_adjusted, random_state=random_state,
        )

        # 저장
        Path(self._dir).mkdir(parents=True, exist_ok=True)
        for name, X_split, y_split in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            io.save_numpy(X_split, f"{self._dir}/X_{name}.npy")
            io.save_numpy(y_split, f"{self._dir}/y_{name}.npy")

        sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
        logger.info("데이터 분할 저장: %s", sizes)
        return sizes

    def save_splits_to_db(
            self,
            X: np.ndarray,
            y: np.ndarray,
            feature_columns: list[str],
            target_columns: list[str],
            pipeline_run_id: str,
            scaler_params: dict = None,
            encoder_classes: dict = None,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            random_state: int = 42,
    ) -> dict[str, int]:
        """
        데이터를 분할하여 파일 + DB에 동시 저장합니다.

        파일: 모델 학습에 직접 사용 (numpy .npy)
        DB: 피처셋 메타데이터 + BLOB 추적 (재현성)

        Returns:
            각 세트의 크기 {"train": N, "val": N, "test": N}
        """
        # 1. 파일 저장 (기존 로직)
        sizes = self.save_splits(X, y, val_ratio, test_ratio, random_state)

        # 2. DB 저장 (전체 피처셋 메타데이터 + 배열)
        from src.database.repository import FeatureSetRepository
        repo = FeatureSetRepository()
        repo.save_feature_set(
            X, y,
            feature_columns=feature_columns,
            target_columns=target_columns,
            pipeline_run_id=pipeline_run_id,
            scaler_params=scaler_params,
            encoder_classes=encoder_classes,
            source_row_count=X.shape[0],
        )

        return sizes

    def load(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """
        저장된 데이터 로드.

        Args:
            split: "train" | "val" | "test"
        """
        X = io.load_numpy(f"{self._dir}/X_{split}.npy")
        y = io.load_numpy(f"{self._dir}/y_{split}.npy")
        return X, y