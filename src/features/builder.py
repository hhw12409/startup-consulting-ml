"""
📁 src/features/builder.py
===========================
피처 빌더 — 전처리 파이프라인의 핵심.

[패턴] Pipeline Pattern — sklearn Pipeline처럼 여러 변환 단계를 체이닝
[역할] 인코딩 → 파생변수 → 스케일링을 순서대로 적용합니다.
[위치] 03_processed → 04_features 단계

핵심 원칙:
- fit_transform()은 학습 데이터에만 호출 (통계값 학습)
- transform()은 검증/테스트/추론에 호출 (학습된 통계값 적용)
- 이 구분을 지키지 않으면 데이터 누수(data leakage)가 발생합니다!
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.feature_config import FEATURE_CONFIG
from src.utils.logger import get_logger
from src.utils import io

logger = get_logger(__name__)


class FeatureBuilder:
    """
    피처 엔지니어링 파이프라인.

    사용법:
        builder = FeatureBuilder()

        # 학습 시 — 통계값(평균, 표준편차 등)을 학습
        X_train, y_train = builder.fit_transform(df_train)

        # 추론 시 — 학습된 통계값으로 변환만
        X_new = builder.transform(df_new)

        # 전처리기 저장/로드
        builder.save_artifacts("models/artifacts/")
        builder = FeatureBuilder.load_artifacts("models/artifacts/")
    """

    def __init__(self):
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler: StandardScaler = StandardScaler()
        self._feature_columns: list[str] = []
        self._is_fitted: bool = False

    # ================================================================
    # 공개 API
    # ================================================================

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터에 맞춰 전처리기를 학습(fit)하고 변환(transform)합니다.

        Args:
            df: 03_processed 데이터 (피처 + 타겟 포함)

        Returns:
            (X, y) — X: [N, feature_dim], y: [N, target_count]
        """
        df = df.copy()
        logger.info("fit_transform 시작: %d행 × %d열", *df.shape)

        # 1) 파생 변수 생성
        df = self._create_derived_features(df)

        # 2) 범주형 인코딩 (fit)
        for col in FEATURE_CONFIG.categorical:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le

        # 3) 피처/타겟 분리
        feature_cols = self._get_feature_columns(df)
        self._feature_columns = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[[t for t in FEATURE_CONFIG.targets if t in df.columns]].values.astype(np.float32)

        # 4) 수치 스케일링 (fit)
        X = self._scaler.fit_transform(X)

        self._is_fitted = True
        logger.info("fit_transform 완료: X=%s, y=%s", X.shape, y.shape)
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        새 데이터에 학습된 전처리를 적용합니다 (추론용).

        ⚠️ fit_transform() 또는 load_artifacts() 후에만 호출 가능
        """
        if not self._is_fitted:
            raise RuntimeError("fit_transform() 또는 load_artifacts()를 먼저 호출하세요")

        df = df.copy()
        df = self._create_derived_features(df)

        # 범주형 인코딩 (학습된 encoder 사용)
        for col, le in self._label_encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        X = df[self._feature_columns].values.astype(np.float32)
        X = self._scaler.transform(X)
        return X

    def save_artifacts(self, dir_path: str) -> None:
        """전처리기(scaler, encoder)를 파일로 저장합니다."""
        io.save_pickle(self._label_encoders, f"{dir_path}/label_encoders.pkl")
        io.save_pickle(self._scaler, f"{dir_path}/scaler.pkl")
        io.save_pickle(self._feature_columns, f"{dir_path}/feature_columns.pkl")
        logger.info("Artifacts 저장 완료: %s", dir_path)

    @classmethod
    def load_artifacts(cls, dir_path: str) -> "FeatureBuilder":
        """저장된 전처리기를 로드하여 새 인스턴스를 생성합니다."""
        builder = cls()
        builder._label_encoders = io.load_pickle(f"{dir_path}/label_encoders.pkl")
        builder._scaler = io.load_pickle(f"{dir_path}/scaler.pkl")
        builder._feature_columns = io.load_pickle(f"{dir_path}/feature_columns.pkl")
        builder._is_fitted = True
        logger.info("Artifacts 로드 완료: %s", dir_path)
        return builder

    # ================================================================
    # 파생 변수 생성 (도메인 지식 기반)
    # ================================================================

    def get_scaler_params(self) -> dict:
        """StandardScaler 파라미터를 JSON 직렬화 가능한 dict로 반환합니다."""
        if not self._is_fitted:
            return {}
        return {
            "mean": self._scaler.mean_.tolist(),
            "scale": self._scaler.scale_.tolist(),
            "var": self._scaler.var_.tolist(),
        }

    def get_encoder_classes(self) -> dict:
        """LabelEncoder 클래스를 JSON 직렬화 가능한 dict로 반환합니다."""
        return {
            col: le.classes_.tolist()
            for col, le in self._label_encoders.items()
        }

    # ================================================================
    # 파생 변수 생성 (도메인 지식 기반)
    # ================================================================

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        원본 피처에서 새로운 피처를 파생합니다.

        도메인 전문가의 지식을 코드로 표현하는 부분입니다.
        피처를 추가하려면 이 메서드에 추가하세요.
        """
        # 1) 임대료 부담률 = 연간 임대료 / 투자금
        if {"monthly_rent", "initial_investment"}.issubset(df.columns):
            df["rent_burden_ratio"] = (
                                              df["monthly_rent"] * 12
                                      ) / (df["initial_investment"].replace(0, 1))

        # 2) 평당 투자금 = 투자금 / 매장 크기
        if {"initial_investment", "store_size_sqm"}.issubset(df.columns):
            df["investment_per_sqm"] = (
                                           df["initial_investment"]
                                       ) / (df["store_size_sqm"].replace(0, 1))

        # 3) 1인당 투자금 = 투자금 / (직원수 + 1)
        if {"initial_investment", "employee_count"}.issubset(df.columns):
            df["investment_per_person"] = (
                                              df["initial_investment"]
                                          ) / (df["employee_count"] + 1)

        # 4) 경쟁 과밀 여부
        if "nearby_competitor_count" in df.columns:
            df["is_high_competition"] = (df["nearby_competitor_count"] > 10).astype(int)

        # 5) 청년/시니어 창업 여부
        if "age" in df.columns:
            df["is_young"] = (df["age"] < 30).astype(int)
            df["is_senior"] = (df["age"] >= 50).astype(int)

        # 6) 무경험 독립창업 (가장 리스크 높은 조합)
        if {"has_related_experience", "is_franchise"}.issubset(df.columns):
            df["inexperienced_independent"] = (
                    (df["has_related_experience"] == 0) & (df["is_franchise"] == 0)
            ).astype(int)

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        사용할 피처 컬럼만 추출 (화이트리스트 방식).

        [수정 이유]
        기존: 블랙리스트(제외할 것만 지정) → API 원본 컬럼(ctprvnCd 등)이 섞임
        수정: 화이트리스트(사용할 것만 지정) → feature_config에 정의된 것 + 파생변수만 사용

        이렇게 하면 공공데이터 원본 컬럼이 아무리 많아도
        우리가 정의한 피처만 사용합니다.
        """
        # 1) feature_config에 정의된 피처
        allowed = set(FEATURE_CONFIG.numerical + FEATURE_CONFIG.categorical + FEATURE_CONFIG.binary)

        # 2) _create_derived_features()에서 생성한 파생 피처
        derived = {
            "rent_burden_ratio", "investment_per_sqm", "investment_per_person",
            "is_high_competition", "is_young", "is_senior", "inexperienced_independent",
        }
        allowed |= derived

        # 3) df에 실제로 존재하고 + 수치형인 컬럼만 선택
        return [
            c for c in df.columns
            if c in allowed and df[c].dtype in ("int64", "float64", "int32", "float32")
        ]