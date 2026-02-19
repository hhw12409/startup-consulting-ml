"""
📁 src/preprocessing/cleaner.py
================================
데이터 정제 모듈.

[역할] API 원본 컬럼 매핑 → 결측치 처리 → 이상치 제거 → 타입 변환
[위치] 01_raw → 02_interim 단계

[핵심 수정사항]
공공데이터 API 원본 CSV는 'bizesNm', 'indsLclsCdNm' 같은 컬럼명을 사용합니다.
우리 모델은 'age', 'business_category' 같은 컬럼명을 기대합니다.
이 모듈에서 컬럼명을 매핑하고, 없는 피처는 기본값으로 채웁니다.
"""

import pandas as pd
import numpy as np

from config.feature_config import FEATURE_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ================================================================
# 공공데이터 API 원본 컬럼 → 우리 컬럼 매핑
# ================================================================
API_COLUMN_MAP = {
    # 공공데이터 API 컬럼명       → 우리 피처명
    "indsLclsCdNm":               "business_category",       # 업종 대분류명
    "indsLclsNm":                 "business_category",       # CSV 호환 (CdNm 없는 경우)
    "indsMclsCdNm":               "business_sub_category",   # 업종 중분류명
    "indsMclsNm":                 "business_sub_category",   # CSV 호환
    "adongNm":                    "district",                # 행정동명
    "adongCd":                    "dong_code",               # 행정동코드
    "lon":                        "longitude",               # 경도
    "lat":                        "latitude",                # 위도
    "bizesNm":                    "store_name",              # 상호명
    "bizesId":                    "biz_id",                  # 사업자번호
    "rdnmAdr":                    "road_address",            # 도로명주소
    "lnoAdr":                     "lot_address",             # 지번주소
}

# DB(stores 테이블) 컬럼명 → 우리 피처명
DB_COLUMN_MAP = {
    "category_large":             "business_category",       # 업종 대분류명
    "category_mid":               "business_sub_category",   # 업종 중분류명
    "adong_name":                 "district",                # 행정동명
    "adong_cd":                   "dong_code",               # 행정동코드
}


class DataCleaner:
    """
    데이터 정제기.

    공공데이터 원본 CSV와 더미 데이터 모두 처리 가능합니다.

    사용법:
        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 정제 파이프라인.

        단계:
        0. API 원본 컬럼 매핑 (공공데이터인 경우)
        1. 중복 제거
        2. 누락 피처 기본값 채우기
        3. 결측치 처리
        4. 이상치 제거
        5. 타입 변환
        """
        df = df.copy()
        original_len = len(df)

        # Step 0: API 원본 컬럼명이면 매핑
        df = self._map_api_columns(df)

        # Step 1: 중복 제거
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info("중복 제거: %d → %d행", original_len, len(df))

        # Step 2: 누락 피처 기본값 채우기
        df = self._fill_missing_features(df)

        # Step 3: 결측치 처리
        df = self._fill_missing(df)

        # Step 4: 이상치 제거
        df = self._remove_outliers(df)

        # Step 5: 타입 변환
        df = self._cast_types(df)

        logger.info("정제 완료: %d행 × %d열", *df.shape)
        return df

    def _map_api_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원본 컬럼명 → 우리 컬럼명으로 변환.

        공공데이터 API 컬럼(adongNm 등) 또는
        DB 컬럼(adong_name 등)을 감지하여 자동 매핑합니다.
        이미 우리 컬럼명이면 아무것도 안 합니다.
        """
        rename_map = {}

        # 1) 공공데이터 API 원본 컬럼 매핑
        api_cols = set(API_COLUMN_MAP.keys()) & set(df.columns)
        if api_cols:
            rename_map.update({k: v for k, v in API_COLUMN_MAP.items() if k in df.columns})
            logger.info("공공데이터 API 컬럼 감지 (%d개)", len(api_cols))

        # 2) DB(stores 테이블) 컬럼 매핑
        db_cols = set(DB_COLUMN_MAP.keys()) & set(df.columns)
        if db_cols:
            # 이미 매핑 대상 컬럼이 존재하면 건너뜀 (중복 방지)
            for k, v in DB_COLUMN_MAP.items():
                if k in df.columns and v not in df.columns and k not in rename_map:
                    rename_map[k] = v
            if db_cols - set(rename_map.keys()) != db_cols:
                logger.info("DB 컬럼 감지 (%d개)", len(db_cols))

        if not rename_map:
            logger.debug("매핑 대상 컬럼 없음 → 건너뜀")
            return df

        df = df.rename(columns=rename_map)
        logger.info("컬럼 매핑 완료: %s", list(rename_map.values()))
        return df

    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모델이 기대하는 피처가 df에 없으면 기본값으로 추가합니다.

        공공데이터 API에는 'age', 'experience_years' 같은 창업자 정보가 없습니다.
        이런 피처는 기본값으로 채워서 모델이 동작하도록 합니다.
        실제 서비스에서는 사용자가 직접 입력합니다.
        """
        defaults_numerical = {
            "age": 35,
            "experience_years": 3,
            "initial_investment": 50_000_000,
            "initial_capital": 50_000_000,
            "monthly_rent": 2_000_000,
            "store_size_sqm": 33.0,
            "employee_count": 1,
            "nearby_competitor_count": 5,
        }

        defaults_categorical = {
            "gender": "M",
            "education_level": "bachelor",
            "floating_population_level": "medium",
        }

        defaults_binary = {
            "has_related_experience": 0,
            "has_startup_experience": 0,
            "is_franchise": 0,
        }

        added = []
        for col, val in {**defaults_numerical, **defaults_categorical, **defaults_binary}.items():
            if col not in df.columns:
                df[col] = val
                added.append(col)

        if added:
            logger.info("누락 피처 기본값 추가 (%d개): %s", len(added), added)

        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리 (수치형: 중앙값, 범주형: 'unknown', 이진: 0)"""
        for col in FEATURE_CONFIG.numerical:
            if col in df.columns:
                median = df[col].median()
                nulls = df[col].isna().sum()
                if nulls > 0:
                    df[col] = df[col].fillna(median)
                    logger.debug("결측치 채움: %s (%d건, median=%.1f)", col, nulls, median)

        for col in FEATURE_CONFIG.categorical:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        for col in FEATURE_CONFIG.binary:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """IQR 방식 이상치 제거 (Q1 - 3*IQR ~ Q3 + 3*IQR)"""
        outlier_cols = ["initial_investment", "monthly_rent", "store_size_sqm"]

        for col in outlier_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr

            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            removed = before - len(df)
            if removed > 0:
                logger.info("이상치 제거: %s (%d건)", col, removed)

        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """컬럼 타입을 명시적으로 변환"""
        int_cols = ["age", "experience_years", "employee_count", "nearby_competitor_count"]
        float_cols = ["store_size_sqm", "initial_investment", "monthly_rent", "initial_capital"]

        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

        return df