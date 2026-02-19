"""
📁 src/utils/io.py
===================
파일 읽기/쓰기 유틸리티.

[역할] CSV, numpy, pickle 파일의 저장/로드를 표준화합니다.
       경로 생성, 인코딩, 에러 처리를 한 곳에서 관리합니다.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """DataFrame → CSV 저장. 디렉토리 자동 생성."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", **kwargs)
    logger.info("CSV 저장: %s (%d행)", path, len(df))


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """CSV → DataFrame 로드."""
    df = pd.read_csv(path, **kwargs)
    logger.info("CSV 로드: %s (%d행 × %d열)", path, *df.shape)
    return df


def save_numpy(arr: np.ndarray, path: str) -> None:
    """numpy 배열 저장 (.npy)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info("Numpy 저장: %s (shape=%s)", path, arr.shape)


def load_numpy(path: str) -> np.ndarray:
    """numpy 배열 로드 (.npy)"""
    arr = np.load(path)
    logger.info("Numpy 로드: %s (shape=%s)", path, arr.shape)
    return arr


def save_pickle(obj: Any, path: str) -> None:
    """Python 객체 pickle 저장 (scaler, encoder 등)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Pickle 저장: %s", path)


def load_pickle(path: str) -> Any:
    """Pickle 로드"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Pickle 로드: %s", path)
    return obj