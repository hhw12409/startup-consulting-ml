"""
피처 정의서.

어떤 컬럼을 수치형/범주형으로 처리할지, 타겟은 무엇인지를 한 곳에서 관리합니다.
피처를 추가/제거할 때 이 파일만 수정하면 전체 파이프라인에 반영됩니다.

[패턴] Single Source of Truth — 피처 정보가 코드 곳곳에 흩어지는 것을 방지
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """피처 설정. frozen=True → 실수로 변경 방지"""

    # ── 수치형 피처 (StandardScaler 적용 대상) ──
    numerical: tuple[str, ...] = (
        "age",  # 창업자 나이
        "experience_years",  # 경력 연수
        "initial_investment",  # 초기 투자금 (원)
        "monthly_rent",  # 월 임대료 (원)
        "store_size_sqm",  # 매장 크기 (㎡)
        "employee_count",  # 종업원 수
        "nearby_competitor_count",  # 주변 경쟁업체 수
        "initial_capital",  # 보유 자본금 (원)
    )

    # ── 범주형 피처 (LabelEncoder 적용 대상) ──
    categorical: tuple[str, ...] = (
        "gender",  # 성별 (M/F)
        "education_level",  # 학력
        "business_category",  # 업종 대분류
        "business_sub_category",  # 업종 소분류
        "district",  # 지역 (행정동)
        "floating_population_level",  # 유동인구 수준 (low/medium/high)
    )

    # ── 이진 피처 (0 or 1, 인코딩 불필요) ──
    binary: tuple[str, ...] = (
        "has_related_experience",  # 업종 관련 경험 여부
        "has_startup_experience",  # 이전 창업 경험 여부
        "is_franchise",  # 프랜차이즈 여부
    )

    # ── 타겟 (모델이 예측할 값) ──
    targets: tuple[str, ...] = (
        "survival_1yr",  # 1년 생존확률 (0~1)
        "survival_3yr",  # 3년 생존확률 (0~1)
        "monthly_revenue",  # 예상 월매출 (원)
        "monthly_profit",  # 예상 월순이익 (원)
        "risk_score",  # 리스크 점수 (0~1)
        "break_even_months",  # 손익분기 도달 개월수
    )

    @property
    def all_features(self) -> list[str]:
        """전체 입력 피처 목록"""
        return list(self.numerical + self.categorical + self.binary)

    @property
    def n_features_raw(self) -> int:
        """원본 피처 수 (인코딩 전)"""
        return len(self.all_features)


# 전역 설정 인스턴스
FEATURE_CONFIG = FeatureConfig()
