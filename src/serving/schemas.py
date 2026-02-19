"""
📁 src/serving/schemas.py
===========================
Pydantic 요청/응답 스키마.

Swagger UI 문서 자동 생성에 사용됩니다.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """창업 예측 API 요청"""

    # 창업자 정보
    founder_age: int = Field(..., ge=18, le=80, description="나이")
    founder_gender: str = Field(default="M", description="성별 (M/F)")
    founder_education: str = Field(default="bachelor", description="학력")
    experience_years: int = Field(default=0, ge=0, description="경력 연수")
    has_related_experience: bool = Field(default=False, description="업종 경험")

    # 사업 정보
    business_category: str = Field(..., description="업종 대분류")
    business_sub_category: str = Field(default="", description="업종 소분류")
    initial_investment: int = Field(..., ge=0, description="초기 투자금 (원)")
    monthly_rent: int = Field(default=0, ge=0, description="월 임대료 (원)")
    store_size_sqm: float = Field(default=0, ge=0, description="매장 크기 (㎡)")
    employee_count: int = Field(default=0, ge=0, description="종업원 수")
    is_franchise: bool = Field(default=False, description="프랜차이즈 여부")
    district: str = Field(..., description="지역 (행정동)")

    def to_dict(self) -> dict:
        """predictor에 전달할 딕셔너리"""
        return {
            "age": self.founder_age,
            "gender": self.founder_gender,
            "education_level": self.founder_education,
            "experience_years": self.experience_years,
            "has_related_experience": int(self.has_related_experience),
            "has_startup_experience": 0,
            "initial_capital": self.initial_investment,
            "business_category": self.business_category,
            "business_sub_category": self.business_sub_category,
            "district": self.district,
            "store_size_sqm": self.store_size_sqm,
            "initial_investment": self.initial_investment,
            "monthly_rent": self.monthly_rent,
            "employee_count": self.employee_count,
            "is_franchise": int(self.is_franchise),
            "nearby_competitor_count": 5,  # TODO: 실시간 조회
            "floating_population_level": "medium",
        }

    class Config:
        json_schema_extra = {
            "example": {
                "founder_age": 35, "founder_gender": "M",
                "business_category": "food", "business_sub_category": "cafe",
                "initial_investment": 50000000, "monthly_rent": 2000000,
                "store_size_sqm": 33.0, "employee_count": 2,
                "is_franchise": False, "district": "강남구 역삼동",
            }
        }


class PredictionResponse(BaseModel):
    """창업 예측 API 응답"""
    success: bool = True
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None