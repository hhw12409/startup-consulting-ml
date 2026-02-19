"""
📁 src/preprocessing/merger.py
===============================
여러 데이터 소스 병합 모듈.

[역할] 상가 데이터 + 상권 데이터 + 인구 데이터를 하나로 합칩니다.
[위치] 02_interim → 03_processed 단계 (labeler와 함께 사용)
"""

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataMerger:
    """여러 데이터프레임을 병합하는 유틸리티."""

    def merge_commercial_data(
            self, stores: pd.DataFrame, commercial: pd.DataFrame
    ) -> pd.DataFrame:
        """
        상가 데이터 + 상권분석 데이터 병합.

        Args:
            stores: 상가업소 DataFrame (dong_code 컬럼 필요)
            commercial: 상권 데이터 (dong_code, avg_sales, floating_pop 등)

        Returns:
            병합된 DataFrame
        """
        if commercial.empty:
            logger.warning("상권 데이터가 비어있어 병합 건너뜀")
            return stores

        merged = stores.merge(commercial, on="dong_code", how="left")
        logger.info("상권 데이터 병합: %d → %d열", stores.shape[1], merged.shape[1])
        return merged

    def merge_population_data(
            self, df: pd.DataFrame, population: pd.DataFrame
    ) -> pd.DataFrame:
        """거주인구 데이터 병합"""
        if population.empty:
            return df

        merged = df.merge(population, on="dong_code", how="left")
        logger.info("인구 데이터 병합: %d → %d열", df.shape[1], merged.shape[1])
        return merged