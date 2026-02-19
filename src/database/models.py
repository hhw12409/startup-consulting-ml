"""
📁 src/database/models.py
============================
SQLAlchemy ORM 모델 정의.

stores 테이블: 상가 원본 데이터
region_codes 테이블: 행정동 코드 마스터
collection_logs 테이블: 수집 이력
cleaned_stores 테이블: 정제된 데이터 (02_interim)
labeled_stores 테이블: 라벨링 데이터 (03_processed)
feature_sets 테이블: 피처셋 (04_features)
training_runs 테이블: 학습 실행 이력
"""

from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, BigInteger, Float,
    DateTime, Text, Enum, DECIMAL, JSON, LargeBinary,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Store(Base):
    """상가 원본 데이터"""
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 사업자 정보
    biz_id = Column(String(20), unique=True, index=True)
    store_name = Column(String(200))
    branch_name = Column(String(100))

    # 업종 분류
    category_large_cd = Column(String(10))
    category_large = Column(String(50), index=True)
    category_mid_cd = Column(String(10))
    category_mid = Column(String(50))
    category_small_cd = Column(String(10))
    category_small = Column(String(100))

    # 표준산업분류
    ksic_cd = Column(String(10))
    ksic_name = Column(String(100))

    # 지역 정보
    sido_cd = Column(String(5))
    sido_name = Column(String(20))
    sgg_cd = Column(String(5), index=True)
    sgg_name = Column(String(20))
    adong_cd = Column(String(10), index=True)
    adong_name = Column(String(30))
    ldong_cd = Column(String(10))
    ldong_name = Column(String(30))

    # 주소
    lot_address = Column(String(300))
    road_address = Column(String(300))
    building_name = Column(String(100))
    zip_code = Column(String(10))

    # 위치
    longitude = Column(DECIMAL(11, 8))
    latitude = Column(DECIMAL(10, 8))

    # 층/호
    floor_info = Column(String(20))
    unit_info = Column(String(20))

    # 사업자 상태
    biz_status_cd = Column(String(5), index=True)
    biz_status = Column(String(10))
    closure_date = Column(String(10))

    # 메타
    data_ym = Column(String(6))
    collected_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Store(biz_id={self.biz_id}, name={self.store_name})>"


class RegionCode(Base):
    """행정동 코드 마스터"""
    __tablename__ = "region_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    region_cd = Column(String(10), unique=True, nullable=False)
    region_cd_8 = Column(String(8), nullable=False, index=True)
    sido_cd = Column(String(2), index=True)
    sgg_cd = Column(String(3))
    dong_cd = Column(String(3))
    sido_name = Column(String(20))
    sgg_name = Column(String(20))
    dong_name = Column(String(30))
    full_name = Column(String(80))

    def __repr__(self):
        return f"<RegionCode({self.region_cd_8}, {self.full_name})>"


class CollectionLog(Base):
    """수집 이력"""
    __tablename__ = "collection_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dong_cd = Column(String(10), nullable=False, index=True)
    dong_name = Column(String(30))
    store_count = Column(Integer, default=0)
    status = Column(String(10), default="success")
    error_msg = Column(Text)
    collected_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<CollectionLog({self.dong_cd}, {self.store_count}건, {self.status})>"


class CleanedStore(Base):
    """정제된 상가 데이터 (02_interim 단계)"""
    __tablename__ = "cleaned_stores"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 사업자 식별
    biz_id = Column(String(20), index=True)
    store_name = Column(String(200))

    # 업종
    business_category = Column(String(50), index=True)
    business_sub_category = Column(String(50))

    # 지역
    district = Column(String(30), index=True)
    dong_code = Column(String(10))
    longitude = Column(DECIMAL(11, 8))
    latitude = Column(DECIMAL(10, 8))
    road_address = Column(String(300))
    lot_address = Column(String(300))

    # 수치형 피처
    age = Column(Integer, default=35)
    experience_years = Column(Integer, default=3)
    initial_investment = Column(BigInteger, default=50_000_000)
    initial_capital = Column(BigInteger, default=50_000_000)
    monthly_rent = Column(BigInteger, default=2_000_000)
    store_size_sqm = Column(Float, default=33.0)
    employee_count = Column(Integer, default=1)
    nearby_competitor_count = Column(Integer, default=5)

    # 범주형 피처
    gender = Column(String(5), default="M")
    education_level = Column(String(20), default="bachelor")
    floating_population_level = Column(String(10), default="medium")

    # 이진 피처
    has_related_experience = Column(Integer, default=0)
    has_startup_experience = Column(Integer, default=0)
    is_franchise = Column(Integer, default=0)

    # 사업자 상태
    biz_status_cd = Column(String(5))
    biz_status = Column(String(10))
    closure_date = Column(String(10))

    # 파이프라인 메타
    pipeline_run_id = Column(String(36), index=True)
    cleaned_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CleanedStore(biz_id={self.biz_id}, category={self.business_category})>"


class LabeledStore(Base):
    """라벨링된 상가 데이터 (03_processed 단계)"""
    __tablename__ = "labeled_stores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cleaned_store_id = Column(Integer)

    # 사업자 식별
    biz_id = Column(String(20), index=True)
    store_name = Column(String(200))

    # 업종
    business_category = Column(String(50), index=True)
    business_sub_category = Column(String(50))

    # 지역
    district = Column(String(30))
    dong_code = Column(String(10))
    longitude = Column(DECIMAL(11, 8))
    latitude = Column(DECIMAL(10, 8))
    road_address = Column(String(300))
    lot_address = Column(String(300))

    # 수치형 피처
    age = Column(Integer)
    experience_years = Column(Integer)
    initial_investment = Column(BigInteger)
    initial_capital = Column(BigInteger)
    monthly_rent = Column(BigInteger)
    store_size_sqm = Column(Float)
    employee_count = Column(Integer)
    nearby_competitor_count = Column(Integer)

    # 범주형 피처
    gender = Column(String(5))
    education_level = Column(String(20))
    floating_population_level = Column(String(10))

    # 이진 피처
    has_related_experience = Column(Integer)
    has_startup_experience = Column(Integer)
    is_franchise = Column(Integer)

    # 사업자 상태
    biz_status_cd = Column(String(5))

    # 생성된 라벨 (타겟 변수)
    survival_1yr = Column(Float)
    survival_3yr = Column(Float)
    monthly_revenue = Column(BigInteger)
    monthly_profit = Column(BigInteger)
    risk_score = Column(Float)
    break_even_months = Column(Integer)

    # 파이프라인 메타
    pipeline_run_id = Column(String(36), index=True)
    labeled_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<LabeledStore(biz_id={self.biz_id}, survival_1yr={self.survival_1yr})>"


class FeatureSet(Base):
    """피처 엔지니어링 결과 (04_features 단계)"""
    __tablename__ = "feature_sets"

    id = Column(Integer, primary_key=True, autoincrement=True)

    pipeline_run_id = Column(String(36), nullable=False, index=True)

    # 피처 메타데이터
    feature_columns = Column(JSON, nullable=False)
    target_columns = Column(JSON, nullable=False)
    n_samples = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    n_targets = Column(Integer, nullable=False)

    # 직렬화된 numpy 배열 (LONGBLOB: 최대 4GB)
    feature_data = Column(LargeBinary(length=2**32 - 1), nullable=False)
    target_data = Column(LargeBinary(length=2**32 - 1), nullable=False)

    # 전처리기 파라미터 (재현성)
    scaler_params = Column(JSON)
    encoder_classes = Column(JSON)

    # 소스 추적
    source_table = Column(String(50), default="labeled_stores")
    source_row_count = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<FeatureSet(run={self.pipeline_run_id}, samples={self.n_samples}, features={self.n_features})>"


class TrainingRun(Base):
    """모델 학습 실행 이력"""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    run_id = Column(String(36), nullable=False, unique=True)
    pipeline_run_id = Column(String(36))

    # 모델 정보
    model_type = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100))

    # 데이터 분할
    train_size = Column(Integer)
    val_size = Column(Integer)
    test_size = Column(Integer)
    n_features = Column(Integer)

    # 아티팩트 경로
    model_path = Column(String(500))
    artifacts_path = Column(String(500))

    # 설정/결과 (JSON)
    hyperparameters = Column(JSON)
    metrics = Column(JSON)

    # 상태
    status = Column(String(20), default="started", index=True)
    error_message = Column(Text)

    # 시각
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    def __repr__(self):
        return f"<TrainingRun(id={self.run_id}, model={self.model_type}, status={self.status})>"