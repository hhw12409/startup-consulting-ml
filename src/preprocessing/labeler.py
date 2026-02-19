"""
📁 src/preprocessing/labeler.py
================================
타겟 라벨 생성 모듈.

[역할] 원본 데이터에서 ML 모델이 예측할 라벨(정답)을 만듭니다.
[위치] 02_interim → 03_processed 단계

라벨 종류:
  - survival_1yr: 1년 생존 여부 (0 or 1)
  - survival_3yr: 3년 생존 여부 (0 or 1)
  - risk_score: 리스크 점수 (규칙 기반 계산)
"""

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LabelGenerator:
    """
    타겟 라벨 생성기.

    사용법:
        labeler = LabelGenerator()
        df_labeled = labeler.generate(df_clean)
    """

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 라벨을 생성하여 컬럼으로 추가합니다."""
        df = df.copy()

        df = self._label_survival(df)
        df = self._label_revenue(df)
        df = self._label_risk(df)
        df = self._label_break_even(df)

        logger.info("라벨 생성 완료: %s", list(df.columns[-6:]))
        return df

    def _label_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        if "b_stt_cd" in df.columns:
            unique_vals = df["b_stt_cd"].unique()
            # 실제로 폐업 데이터가 있는 경우만 사용
            if "03" in unique_vals or "02" in unique_vals:
                df["survival_1yr"] = (df["b_stt_cd"] != "03").astype(float)
                df["survival_3yr"] = (df["b_stt_cd"] == "01").astype(float)
                return df

    # 폐업 데이터가 없으면 임시 라벨
        logger.warning("폐업 데이터 없음 → 임시 라벨 생성")
        df["survival_1yr"] = np.random.binomial(1, 0.7, len(df)).astype(float)
        df["survival_3yr"] = np.random.binomial(1, 0.5, len(df)).astype(float)
        return df

    def _label_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        매출/이익 라벨.

        실제 매출 데이터가 없으면 업종/지역 평균으로 대체합니다.
        Phase 2에서 카드사 데이터 연동 시 실제 값으로 교체.
        """
        if "monthly_revenue" not in df.columns:
            # 업종별 평균 매출 (통계청 기준 추정치)
            revenue_map = {
                "food": 15_000_000, "retail": 20_000_000,
                "service": 10_000_000, "it": 25_000_000,
                "education": 12_000_000,
            }
            if "business_category" in df.columns:
                df["monthly_revenue"] = df["business_category"].map(revenue_map).fillna(12_000_000)
            else:
                df["monthly_revenue"] = 12_000_000

            # 순이익 = 매출 × 이익률 (업종 평균 15~25%)
            profit_rate = np.random.uniform(0.10, 0.30, len(df))
            df["monthly_profit"] = (df["monthly_revenue"] * profit_rate).astype(int)

        return df

    def _label_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        리스크 점수 라벨 (규칙 기반).

        여러 리스크 요인을 가중 합산하여 0~1 점수를 계산합니다.
        """
        risk = np.zeros(len(df))

        # 1) 임대료 비율 리스크
        if "monthly_rent" in df.columns and "initial_investment" in df.columns:
            ratio = (df["monthly_rent"] * 12) / (df["initial_investment"] + 1)
            risk += np.clip(ratio, 0, 1) * 0.3

        # 2) 경쟁 밀집 리스크
        if "nearby_competitor_count" in df.columns:
            risk += np.clip(df["nearby_competitor_count"] / 30, 0, 1) * 0.2

        # 3) 경험 부족 리스크
        if "has_related_experience" in df.columns:
            risk += (1 - df["has_related_experience"]) * 0.2

        # 4) 나이 리스크 (25세 미만, 60세 이상)
        if "age" in df.columns:
            age_risk = np.where(df["age"] < 25, 0.15, np.where(df["age"] > 60, 0.1, 0))
            risk += age_risk

        df["risk_score"] = np.clip(risk, 0, 1)
        return df

    def _label_break_even(self, df: pd.DataFrame) -> pd.DataFrame:
        """손익분기 개월수 라벨 (투자금 / 월순이익)"""
        if "monthly_profit" in df.columns and "initial_investment" in df.columns:
            profit = df["monthly_profit"].replace(0, 1)  # 0 나누기 방지
            df["break_even_months"] = np.clip(
                (df["initial_investment"] / profit).astype(int), 1, 60
            )
        else:
            df["break_even_months"] = 18  # 기본값

        return df