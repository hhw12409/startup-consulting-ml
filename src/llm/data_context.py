"""
📁 src/llm/data_context.py
==============================
데이터 기반 컨텍스트 생성기.

[역할] 수집된 상가 데이터(stores_raw.csv)에서 업종별/지역별 통계를 계산하여
       LLM 프롬프트에 주입합니다.

[Before] LLM은 gemma2의 일반 지식만으로 답변
[After]  LLM은 실제 데이터 통계 + 일반 지식으로 답변

예시:
    context = DataContext()
    stats = context.get_context("food", "역삼1동")
    # → "역삼1동 음식점 234개, 평균 경쟁업체 12개, 폐업률 38%..."
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataContext:
    """
    수집된 데이터에서 통계를 추출하여 LLM 프롬프트에 제공합니다.

    사용법:
        ctx = DataContext()
        stats = ctx.get_context(category="음식", district="역삼1동")
        prompt = f"데이터 분석 결과:\\n{stats}\\n\\n위 통계를 참고하여..."
    """

    def __init__(self):
        self._df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """DB(stores 테이블)에서 상가 데이터 로드"""
        try:
            from src.database.repository import StoreRepository
            repo = StoreRepository()
            df = repo.to_dataframe()
            if not df.empty:
                df = df.astype(str)
                logger.info("데이터 컨텍스트 로드 (DB): %d건", len(df))
                return df
        except Exception as e:
            logger.warning("DB 로드 실패 (데이터 컨텍스트 비활성): %s", e)

        return pd.DataFrame()

    @property
    def is_available(self) -> bool:
        return not self._df.empty

    def get_context(self, category: str = None, district: str = None) -> str:
        """
        업종/지역 기반 통계 컨텍스트를 생성합니다.

        Args:
            category: 업종명 (예: "음식", "소매", "food")
            district: 행정동명 (예: "역삼1동", "서교동")

        Returns:
            프롬프트에 삽입할 통계 문자열
        """
        if not self.is_available:
            return ""

        sections = []

        # 1. 전체 개요
        sections.append(self._overall_stats())

        # 2. 업종별 통계
        if category:
            cat_stats = self._category_stats(category)
            if cat_stats:
                sections.append(cat_stats)

        # 3. 지역별 통계
        if district:
            dist_stats = self._district_stats(district)
            if dist_stats:
                sections.append(dist_stats)

        # 4. 업종+지역 교차 통계
        if category and district:
            cross_stats = self._cross_stats(category, district)
            if cross_stats:
                sections.append(cross_stats)

        # 5. 사업자 상태 통계 (생존/폐업)
        survival_stats = self._survival_stats(category, district)
        if survival_stats:
            sections.append(survival_stats)

        if not sections:
            return ""

        return "## 📊 실제 데이터 기반 분석 (참고용)\n\n" + "\n\n".join(sections)

    # ================================================================
    # 통계 계산 메서드
    # ================================================================

    def _overall_stats(self) -> str:
        """전체 데이터 개요"""
        df = self._df
        total = len(df)

        # 업종 분포
        cat_col = self._find_col(["category_large", "indsLclsCdNm", "business_category", "indsLclsNm"])
        if cat_col:
            top_cats = df[cat_col].value_counts().head(5)
            cat_text = ", ".join(f"{k}({v:,}개)" for k, v in top_cats.items())
        else:
            cat_text = "정보 없음"

        # 지역 분포
        dist_col = self._find_col(["adong_name", "adongNm", "district"])
        if dist_col:
            n_districts = df[dist_col].nunique()
        else:
            n_districts = 0

        return (
            f"### 전체 데이터 개요\n"
            f"- 분석 대상 상가: 총 {total:,}개\n"
            f"- 분석 지역: {n_districts}개 행정동\n"
            f"- 주요 업종: {cat_text}"
        )

    def _category_stats(self, category: str) -> str:
        """업종별 통계"""
        cat_col = self._find_col(["category_large", "indsLclsCdNm", "business_category", "indsLclsNm"])
        if not cat_col:
            return ""

        # 부분 매칭 (예: "food" → "음식", "카페" → "음식")
        mask = self._fuzzy_match(self._df[cat_col], category)
        subset = self._df[mask]

        if subset.empty:
            return ""

        total = len(self._df)
        count = len(subset)
        pct = count / total * 100

        # 중분류 분포
        sub_col = self._find_col(["category_mid", "indsMclsCdNm", "business_sub_category"])
        if sub_col:
            top_subs = subset[sub_col].value_counts().head(5)
            sub_text = ", ".join(f"{k}({v:,}개)" for k, v in top_subs.items())
        else:
            sub_text = "정보 없음"

        # 지역별 분포
        dist_col = self._find_col(["adong_name", "adongNm", "district"])
        if dist_col:
            top_dists = subset[dist_col].value_counts().head(5)
            dist_text = ", ".join(f"{k}({v:,}개)" for k, v in top_dists.items())
        else:
            dist_text = "정보 없음"

        matched_name = subset[cat_col].mode().iloc[0] if not subset.empty else category

        return (
            f"### '{matched_name}' 업종 분석\n"
            f"- 해당 업종 상가: {count:,}개 (전체의 {pct:.1f}%)\n"
            f"- 세부 업종 Top 5: {sub_text}\n"
            f"- 밀집 지역 Top 5: {dist_text}"
        )

    def _district_stats(self, district: str) -> str:
        """지역별 통계"""
        dist_col = self._find_col(["adong_name", "adongNm", "district"])
        if not dist_col:
            return ""

        mask = self._fuzzy_match(self._df[dist_col], district)
        subset = self._df[mask]

        if subset.empty:
            return ""

        count = len(subset)

        # 업종 분포
        cat_col = self._find_col(["category_large", "indsLclsCdNm", "business_category"])
        if cat_col:
            top_cats = subset[cat_col].value_counts().head(5)
            cat_text = ", ".join(f"{k}({v:,}개)" for k, v in top_cats.items())
        else:
            cat_text = "정보 없음"

        matched_name = subset[dist_col].mode().iloc[0] if not subset.empty else district

        return (
            f"### '{matched_name}' 지역 분석\n"
            f"- 해당 지역 상가: {count:,}개\n"
            f"- 업종 분포 Top 5: {cat_text}"
        )

    def _cross_stats(self, category: str, district: str) -> str:
        """업종+지역 교차 통계"""
        cat_col = self._find_col(["category_large", "indsLclsCdNm", "business_category"])
        dist_col = self._find_col(["adong_name", "adongNm", "district"])
        if not cat_col or not dist_col:
            return ""

        cat_mask = self._fuzzy_match(self._df[cat_col], category)
        dist_mask = self._fuzzy_match(self._df[dist_col], district)
        subset = self._df[cat_mask & dist_mask]

        if subset.empty:
            return ""

        count = len(subset)

        # 같은 지역 전체 대비 비율
        dist_total = self._df[dist_mask].sum() if dist_mask.any() else 0
        dist_total = int(dist_mask.sum())
        pct = count / dist_total * 100 if dist_total > 0 else 0

        # 세부 업종
        sub_col = self._find_col(["category_mid", "indsMclsCdNm", "business_sub_category"])
        if sub_col:
            top_subs = subset[sub_col].value_counts().head(3)
            sub_text = ", ".join(f"{k}({v}개)" for k, v in top_subs.items())
        else:
            sub_text = "정보 없음"

        matched_cat = subset[cat_col].mode().iloc[0] if not subset.empty else category
        matched_dist = subset[dist_col].mode().iloc[0] if not subset.empty else district

        return (
            f"### '{matched_dist}' × '{matched_cat}' 교차 분석\n"
            f"- 해당 지역+업종 상가: {count:,}개 (지역 내 {pct:.1f}%)\n"
            f"- 경쟁업체 수: 약 {count}개\n"
            f"- 세부 업종: {sub_text}"
        )

    def _survival_stats(self, category: str = None, district: str = None) -> str:
        """사업자 상태 통계 (생존/폐업)"""
        status_col = self._find_col(["biz_status_cd", "b_stt_cd", "b_stt"])
        if not status_col:
            return ""

        df = self._df.copy()

        # 필터
        if category:
            cat_col = self._find_col(["category_large", "indsLclsCdNm", "business_category"])
            if cat_col:
                df = df[self._fuzzy_match(df[cat_col], category)]

        if district:
            dist_col = self._find_col(["adong_name", "adongNm", "district"])
            if dist_col:
                df = df[self._fuzzy_match(df[dist_col], district)]

        if df.empty:
            return ""

        # 상태 코드: 01=계속, 02=휴업, 03=폐업
        status_counts = df[status_col].value_counts()
        total = len(df)

        active = int(status_counts.get("01", 0))
        suspended = int(status_counts.get("02", 0))
        closed = int(status_counts.get("03", 0))
        unknown = total - active - suspended - closed

        if active + closed == 0:
            return ""

        survival_rate = active / (active + closed) * 100 if (active + closed) > 0 else 0
        closure_rate = closed / (active + closed) * 100 if (active + closed) > 0 else 0

        return (
            f"### 사업자 생존 현황\n"
            f"- 영업 중: {active:,}개 ({survival_rate:.1f}%)\n"
            f"- 폐업: {closed:,}개 ({closure_rate:.1f}%)\n"
            f"- 휴업: {suspended:,}개\n"
            f"- 상태 미확인: {unknown:,}개"
        )

    # ================================================================
    # 유틸리티
    # ================================================================

    def _find_col(self, candidates: list[str]) -> str:
        """DataFrame에서 존재하는 첫 번째 컬럼명 반환"""
        for col in candidates:
            if col in self._df.columns:
                return col
        return ""

    def _fuzzy_match(self, series: pd.Series, keyword: str) -> pd.Series:
        """부분 매칭 + 한영 업종 매핑"""
        # 한영 업종 매핑
        category_map = {
            "food": "음식", "retail": "소매", "service": "서비스",
            "education": "교육", "it": "정보통신", "medical": "의료",
            "cafe": "음식", "restaurant": "음식", "카페": "음식",
        }

        keyword_kr = category_map.get(keyword.lower(), keyword)

        return series.fillna("").str.contains(keyword_kr, case=False, na=False)