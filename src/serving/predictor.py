"""
📁 src/serving/predictor.py
=============================
추론 파이프라인.

[패턴] Facade — 전처리 → 모델 추론 → 후처리를 하나의 인터페이스로 제공
[역할] API에서 받은 원본 데이터를 모델이 이해하는 형태로 변환하고,
       모델 출력을 사람이 이해하는 형태로 변환합니다.
"""

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.features.builder import FeatureBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    추론 파이프라인.

    사용법:
        predictor = Predictor(model=xgb, feature_builder=builder)
        result = predictor.predict({"age": 35, "business_category": "food", ...})
    """

    def __init__(self, model: BaseModel, feature_builder: FeatureBuilder):
        self._model = model
        self._builder = feature_builder

    def predict(self, input_data: dict) -> dict:
        """
        원본 입력 → 최종 결과.

        단계:
        1. dict → DataFrame
        2. 피처 변환 (builder.transform)
        3. 모델 추론
        4. 후처리 (비즈니스 로직)
        """
        # 1) DataFrame 변환
        df = pd.DataFrame([input_data])

        # 2) 피처 변환
        X = self._builder.transform(df)

        # 3) 모델 추론
        raw = self._model.predict(X)

        # 4) 후처리
        result = self._postprocess(raw, input_data)

        logger.info("예측 완료: survival_1yr=%.2f, risk=%s",
                    result["survival"]["one_year"], result["risk"]["level"])
        return result

    def _postprocess(self, raw: dict[str, np.ndarray], input_data: dict) -> dict:
        """모델 출력 → API 응답 형태로 변환 + 비즈니스 로직 추가"""
        surv = raw.get("survival", np.array([[0.5, 0.3]]))[0]
        rev = raw.get("revenue", np.array([[0, 0]]))[0]
        risk = raw.get("risk", np.array([[0.5]]))[0]
        be = raw.get("break_even", np.array([[12]]))[0]

        risk_score = float(risk[0])
        risk_factors = self._analyze_risk(input_data, risk_score)
        recommendations = self._generate_recs(input_data, risk_factors)

        # 리스크 등급
        if risk_score < 0.3:
            level = "LOW"
        elif risk_score < 0.6:
            level = "MEDIUM"
        elif risk_score < 0.8:
            level = "HIGH"
        else:
            level = "CRITICAL"

        return {
            "survival": {
                "one_year": round(float(surv[0]), 4),
                "three_year": round(float(surv[1]), 4),
            },
            "financials": {
                "monthly_revenue": int(rev[0]),
                "monthly_profit": int(rev[1]),
                "break_even_months": max(1, int(be[0])),
            },
            "risk": {
                "score": round(risk_score, 4),
                "level": level,
                "factors": risk_factors,
            },
            "recommendations": recommendations,
        }

    def _analyze_risk(self, data: dict, risk_score: float) -> list[str]:
        """규칙 기반 리스크 요인 분석"""
        factors = []
        rent = data.get("monthly_rent", 0)
        inv = data.get("initial_investment", 1)

        if inv > 0 and (rent * 12 / inv) > 0.5:
            factors.append("임대료가 투자금 대비 과도합니다")
        if data.get("nearby_competitor_count", 0) > 10:
            factors.append("경쟁 과밀 지역입니다")
        if not data.get("has_related_experience", 0):
            factors.append("해당 업종 경험이 부족합니다")
        if data.get("age", 30) < 25:
            factors.append("청년 창업은 통계적 생존율이 낮습니다")

        return factors

    def _generate_recs(self, data: dict, risk_factors: list) -> list[str]:
        """추천 생성"""
        recs = []
        if "임대료" in str(risk_factors):
            recs.append("임대료가 낮은 인근 지역을 검토하세요")
        if "경험" in str(risk_factors):
            recs.append("프랜차이즈 창업이나 현장 실습을 권장합니다")
        if not recs:
            recs.append("전반적으로 양호합니다. 마케팅 전략에 집중하세요")
        return recs