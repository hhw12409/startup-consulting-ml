"""
📁 src/database/repository.py
================================
데이터 저장소 (Repository 패턴).

[역할] 각 파이프라인 단계의 DB CRUD를 제공합니다.
[패턴] Repository — DB 접근을 하나의 클래스로 추상화

저장소 목록:
    StoreRepository         - stores 테이블 (원본 데이터)
    RegionRepository        - region_codes 테이블 (행정동 코드)
    CleanedStoreRepository  - cleaned_stores 테이블 (02_interim)
    LabeledStoreRepository  - labeled_stores 테이블 (03_processed)
    FeatureSetRepository    - feature_sets 테이블 (04_features)
    TrainingRunRepository   - training_runs 테이블 (실험 추적)
"""

import io as bio
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, text, desc
from sqlalchemy.dialects.mysql import insert as mysql_insert

from src.database.connection import get_session
from src.database.models import (
    Store, RegionCode, CollectionLog,
    CleanedStore, LabeledStore, FeatureSet, TrainingRun,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ================================================================
# API 원본 컬럼 → DB 컬럼 매핑
# ================================================================
API_TO_DB_MAP = {
    "bizesId": "biz_id",
    "bizesNm": "store_name",
    "brchNm": "branch_name",
    "indsLclsCd": "category_large_cd",
    "indsLclsCdNm": "category_large",
    "indsLclsNm": "category_large",       # CSV 호환 (CdNm 없는 경우)
    "indsMclsCd": "category_mid_cd",
    "indsMclsCdNm": "category_mid",
    "indsMclsNm": "category_mid",          # CSV 호환
    "indsSclsCd": "category_small_cd",
    "indsSclsCdNm": "category_small",
    "indsSclsNm": "category_small",        # CSV 호환
    "ksicCd": "ksic_cd",
    "ksicNm": "ksic_name",
    "ctprvnCd": "sido_cd",
    "ctprvnNm": "sido_name",
    "signguCd": "sgg_cd",
    "signguNm": "sgg_name",
    "adongCd": "adong_cd",
    "adongNm": "adong_name",
    "ldongCd": "ldong_cd",
    "ldongNm": "ldong_name",
    "lnoAdr": "lot_address",
    "rdnmAdr": "road_address",
    "bldNm": "building_name",
    "nwZipCd": "zip_code",
    "lon": "longitude",
    "lat": "latitude",
    "flrNo": "floor_info",
    "hoNo": "unit_info",
    "b_stt_cd": "biz_status_cd",
    "b_stt": "biz_status",
    "end_dt": "closure_date",
}


class StoreRepository:
    """상가 데이터 저장소"""

    def upsert_stores(self, df: pd.DataFrame) -> int:
        """
        DataFrame → stores 테이블에 UPSERT (중복 시 업데이트).

        Args:
            df: API 수집 결과 또는 CSV 데이터

        Returns:
            저장/업데이트된 행 수
        """
        # 컬럼 매핑
        rename_map = {k: v for k, v in API_TO_DB_MAP.items() if k in df.columns}
        df_mapped = df.rename(columns=rename_map)

        # DB 컬럼만 추출
        db_columns = [c.name for c in Store.__table__.columns if c.name not in ("id", "collected_at", "updated_at")]
        valid_cols = [c for c in db_columns if c in df_mapped.columns]
        df_insert = df_mapped[valid_cols].copy()

        # NaN → None
        df_insert = df_insert.where(pd.notnull(df_insert), None)

        session = get_session()
        saved = 0

        try:
            records = df_insert.to_dict("records")

            # 배치 UPSERT (1000건씩)
            for i in range(0, len(records), 1000):
                batch = records[i:i + 1000]

                stmt = mysql_insert(Store).values(batch)

                # 중복 시 업데이트할 컬럼
                update_cols = {
                    c: stmt.inserted[c] for c in valid_cols if c != "biz_id"
                }
                update_cols["updated_at"] = datetime.utcnow()

                stmt = stmt.on_duplicate_key_update(**update_cols)

                session.execute(stmt)
                saved += len(batch)

                if (i // 1000 + 1) % 10 == 0:
                    logger.info("  DB 저장 진행: %d/%d건", saved, len(records))

            session.commit()
            logger.info("✅ DB 저장 완료: %d건 (UPSERT)", saved)

        except Exception as e:
            session.rollback()
            logger.error("DB 저장 실패: %s", e)
            raise
        finally:
            session.close()

        return saved

    def get_store_count(self) -> int:
        """전체 상가 수"""
        session = get_session()
        try:
            return session.query(func.count(Store.id)).scalar()
        finally:
            session.close()

    def get_category_stats(self, category: str = None, district: str = None) -> dict:
        """
        업종/지역별 통계 조회.

        Returns:
            {
                "total": 90174,
                "category_count": 234,
                "category_pct": 36.1,
                "top_sub_categories": [("한식", 120), ("카페", 80)],
                "survival_rate": 62.0,
                "closure_rate": 38.0,
            }
        """
        session = get_session()
        try:
            total = session.query(func.count(Store.id)).scalar()

            # 필터 구성
            query = session.query(Store)
            if category:
                query = query.filter(Store.category_large.contains(category))
            if district:
                query = query.filter(Store.adong_name.contains(district))

            count = query.count()

            # 세부 업종 Top 5
            top_subs = (
                query.with_entities(Store.category_mid, func.count(Store.id))
                .group_by(Store.category_mid)
                .order_by(func.count(Store.id).desc())
                .limit(5)
                .all()
            )

            # 사업자 상태
            status_counts = (
                query.with_entities(Store.biz_status_cd, func.count(Store.id))
                .group_by(Store.biz_status_cd)
                .all()
            )
            status_dict = dict(status_counts)
            active = status_dict.get("01", 0)
            closed = status_dict.get("03", 0)

            survival = active / (active + closed) * 100 if (active + closed) > 0 else 0
            closure = closed / (active + closed) * 100 if (active + closed) > 0 else 0

            return {
                "total": total,
                "count": count,
                "pct": count / total * 100 if total > 0 else 0,
                "top_sub_categories": [(name, cnt) for name, cnt in top_subs if name],
                "survival_rate": survival,
                "closure_rate": closure,
            }
        finally:
            session.close()

    def get_district_stats(self, district: str) -> dict:
        """지역별 업종 분포"""
        session = get_session()
        try:
            query = session.query(Store).filter(Store.adong_name.contains(district))
            count = query.count()

            top_cats = (
                query.with_entities(Store.category_large, func.count(Store.id))
                .group_by(Store.category_large)
                .order_by(func.count(Store.id).desc())
                .limit(5)
                .all()
            )

            return {
                "count": count,
                "top_categories": [(name, cnt) for name, cnt in top_cats if name],
            }
        finally:
            session.close()

    def to_dataframe(self, category: str = None, district: str = None) -> pd.DataFrame:
        """
        DB → DataFrame 변환 (학습/피처링용).

        Args:
            category: 업종 필터
            district: 지역 필터
        """
        session = get_session()
        try:
            query = session.query(Store)
            if category:
                query = query.filter(Store.category_large.contains(category))
            if district:
                query = query.filter(Store.adong_name.contains(district))

            df = pd.read_sql(query.statement, session.bind)
            logger.info("DB → DataFrame: %d행 × %d열", *df.shape)
            return df
        finally:
            session.close()

    def log_collection(self, dong_cd: str, dong_name: str, count: int,
                       status: str = "success", error_msg: str = None):
        """수집 이력 저장"""
        session = get_session()
        try:
            log = CollectionLog(
                dong_cd=dong_cd,
                dong_name=dong_name,
                store_count=count,
                status=status,
                error_msg=error_msg,
            )
            session.add(log)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("수집 이력 저장 실패: %s", e)
        finally:
            session.close()


class RegionRepository:
    """행정동 코드 저장소"""

    def upsert_regions(self, df: pd.DataFrame) -> int:
        """행정동 코드 DataFrame → DB 저장"""
        col_map = {
            "region_cd": "region_cd",
            "region_cd_8": "region_cd_8",
            "sido_cd": "sido_cd",
            "sgg_cd": "sgg_cd",
            "dong_cd": "dong_cd",
            "sido_nm": "sido_name",
            "sgg_nm": "sgg_name",
            "dong_nm": "dong_name",
            "full_nm": "full_name",
        }

        rename_map = {k: v for k, v in col_map.items() if k in df.columns}
        df_mapped = df.rename(columns=rename_map)

        db_columns = [c.name for c in RegionCode.__table__.columns if c.name != "id"]
        valid_cols = [c for c in db_columns if c in df_mapped.columns]
        df_insert = df_mapped[valid_cols].where(pd.notnull(df_mapped[valid_cols]), None)

        session = get_session()
        saved = 0

        try:
            records = df_insert.to_dict("records")

            for i in range(0, len(records), 500):
                batch = records[i:i + 500]
                stmt = mysql_insert(RegionCode).values(batch)
                stmt = stmt.on_duplicate_key_update(
                    **{c: stmt.inserted[c] for c in valid_cols if c != "region_cd"}
                )
                session.execute(stmt)
                saved += len(batch)

            session.commit()
            logger.info("✅ 행정동 코드 저장: %d건", saved)
        except Exception as e:
            session.rollback()
            logger.error("행정동 코드 저장 실패: %s", e)
            raise
        finally:
            session.close()

        return saved

    def get_dong_codes(self, sido_cd: str = None) -> list[str]:
        """행정동 8자리 코드 목록 조회"""
        session = get_session()
        try:
            query = session.query(RegionCode.region_cd_8)
            if sido_cd:
                query = query.filter(RegionCode.sido_cd == sido_cd)
            # 시도/시군구 레벨 제외
            query = query.filter(~RegionCode.region_cd_8.endswith("0000"))
            return [r[0] for r in query.all()]
        finally:
            session.close()


# ================================================================
# numpy 직렬화 유틸리티
# ================================================================

def _serialize_numpy(arr: np.ndarray) -> bytes:
    """numpy 배열을 bytes로 직렬화 (DB BLOB 저장용)"""
    buf = bio.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _deserialize_numpy(data: bytes) -> np.ndarray:
    """bytes를 numpy 배열로 역직렬화"""
    buf = bio.BytesIO(data)
    return np.load(buf)


# ================================================================
# CleanedStoreRepository — 정제 데이터 (02_interim)
# ================================================================

class CleanedStoreRepository:
    """정제된 상가 데이터 저장소"""

    def save_cleaned(self, df: pd.DataFrame, pipeline_run_id: str) -> int:
        """
        정제된 DataFrame → cleaned_stores 테이블에 저장.

        Args:
            df: DataCleaner.clean() 결과
            pipeline_run_id: 파이프라인 실행 UUID

        Returns:
            저장된 행 수
        """
        # DB 컬럼에 매핑 가능한 컬럼만 추출
        db_columns = [
            c.name for c in CleanedStore.__table__.columns
            if c.name not in ("id", "cleaned_at")
        ]
        valid_cols = [c for c in db_columns if c in df.columns]

        df_insert = df[valid_cols].copy()
        df_insert["pipeline_run_id"] = pipeline_run_id

        # NaN → None
        df_insert = df_insert.where(pd.notnull(df_insert), None)

        session = get_session()
        saved = 0

        try:
            records = df_insert.to_dict("records")

            for i in range(0, len(records), 1000):
                batch = records[i:i + 1000]
                session.bulk_insert_mappings(CleanedStore, batch)
                saved += len(batch)

                if (i // 1000 + 1) % 10 == 0:
                    logger.info("  cleaned_stores 저장 진행: %d/%d건", saved, len(records))

            session.commit()
            logger.info("cleaned_stores 저장 완료: %d건 (run=%s)", saved, pipeline_run_id[:8])

        except Exception as e:
            session.rollback()
            logger.error("cleaned_stores 저장 실패: %s", e)
            raise
        finally:
            session.close()

        return saved

    def to_dataframe(self, pipeline_run_id: str = None) -> pd.DataFrame:
        """
        cleaned_stores → DataFrame 변환.

        Args:
            pipeline_run_id: 특정 실행 ID (None이면 최신)
        """
        session = get_session()
        try:
            query = session.query(CleanedStore)
            if pipeline_run_id:
                query = query.filter(CleanedStore.pipeline_run_id == pipeline_run_id)
            else:
                latest_id = self.get_latest_run_id()
                if latest_id:
                    query = query.filter(CleanedStore.pipeline_run_id == latest_id)

            df = pd.read_sql(query.statement, session.bind)

            # ORM 메타 컬럼 제거
            drop_cols = ["id", "cleaned_at"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            logger.info("cleaned_stores → DataFrame: %d행 × %d열", *df.shape)
            return df
        finally:
            session.close()

    def get_latest_run_id(self) -> Optional[str]:
        """최신 pipeline_run_id 조회"""
        session = get_session()
        try:
            result = (
                session.query(CleanedStore.pipeline_run_id)
                .order_by(desc(CleanedStore.cleaned_at))
                .first()
            )
            return result[0] if result else None
        finally:
            session.close()

    def delete_old_runs(self, keep_latest: int = 1):
        """오래된 파이프라인 run 데이터를 삭제하고 최신 N개만 유지"""
        session = get_session()
        try:
            runs = (
                session.query(CleanedStore.pipeline_run_id)
                .group_by(CleanedStore.pipeline_run_id)
                .order_by(desc(func.max(CleanedStore.cleaned_at)))
                .all()
            )
            run_ids = [r[0] for r in runs]

            if len(run_ids) <= keep_latest:
                return 0

            old_ids = run_ids[keep_latest:]
            deleted = (
                session.query(CleanedStore)
                .filter(CleanedStore.pipeline_run_id.in_(old_ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            logger.info("cleaned_stores 정리: %d건 삭제 (%d개 run 제거)", deleted, len(old_ids))
            return deleted
        except Exception as e:
            session.rollback()
            logger.error("cleaned_stores 정리 실패: %s", e)
            raise
        finally:
            session.close()


# ================================================================
# LabeledStoreRepository — 라벨링 데이터 (03_processed)
# ================================================================

class LabeledStoreRepository:
    """라벨링된 상가 데이터 저장소"""

    def save_labeled(self, df: pd.DataFrame, pipeline_run_id: str) -> int:
        """
        라벨링된 DataFrame → labeled_stores 테이블에 저장.

        Args:
            df: LabelGenerator.generate() 결과
            pipeline_run_id: 파이프라인 실행 UUID

        Returns:
            저장된 행 수
        """
        db_columns = [
            c.name for c in LabeledStore.__table__.columns
            if c.name not in ("id", "cleaned_store_id", "labeled_at")
        ]
        valid_cols = [c for c in db_columns if c in df.columns]

        df_insert = df[valid_cols].copy()
        df_insert["pipeline_run_id"] = pipeline_run_id

        # NaN → None
        df_insert = df_insert.where(pd.notnull(df_insert), None)

        session = get_session()
        saved = 0

        try:
            records = df_insert.to_dict("records")

            for i in range(0, len(records), 1000):
                batch = records[i:i + 1000]
                session.bulk_insert_mappings(LabeledStore, batch)
                saved += len(batch)

                if (i // 1000 + 1) % 10 == 0:
                    logger.info("  labeled_stores 저장 진행: %d/%d건", saved, len(records))

            session.commit()
            logger.info("labeled_stores 저장 완료: %d건 (run=%s)", saved, pipeline_run_id[:8])

        except Exception as e:
            session.rollback()
            logger.error("labeled_stores 저장 실패: %s", e)
            raise
        finally:
            session.close()

        return saved

    def to_dataframe(self, pipeline_run_id: str = None) -> pd.DataFrame:
        """
        labeled_stores → DataFrame 변환.

        Args:
            pipeline_run_id: 특정 실행 ID (None이면 최신)
        """
        session = get_session()
        try:
            query = session.query(LabeledStore)
            if pipeline_run_id:
                query = query.filter(LabeledStore.pipeline_run_id == pipeline_run_id)
            else:
                latest_id = self.get_latest_run_id()
                if latest_id:
                    query = query.filter(LabeledStore.pipeline_run_id == latest_id)

            df = pd.read_sql(query.statement, session.bind)

            # ORM 메타 컬럼 제거
            drop_cols = ["id", "cleaned_store_id", "labeled_at"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            logger.info("labeled_stores → DataFrame: %d행 × %d열", *df.shape)
            return df
        finally:
            session.close()

    def get_latest_run_id(self) -> Optional[str]:
        """최신 pipeline_run_id 조회"""
        session = get_session()
        try:
            result = (
                session.query(LabeledStore.pipeline_run_id)
                .order_by(desc(LabeledStore.labeled_at))
                .first()
            )
            return result[0] if result else None
        finally:
            session.close()

    def delete_old_runs(self, keep_latest: int = 1):
        """오래된 파이프라인 run 데이터를 삭제하고 최신 N개만 유지"""
        session = get_session()
        try:
            runs = (
                session.query(LabeledStore.pipeline_run_id)
                .group_by(LabeledStore.pipeline_run_id)
                .order_by(desc(func.max(LabeledStore.labeled_at)))
                .all()
            )
            run_ids = [r[0] for r in runs]

            if len(run_ids) <= keep_latest:
                return 0

            old_ids = run_ids[keep_latest:]
            deleted = (
                session.query(LabeledStore)
                .filter(LabeledStore.pipeline_run_id.in_(old_ids))
                .delete(synchronize_session=False)
            )
            session.commit()
            logger.info("labeled_stores 정리: %d건 삭제 (%d개 run 제거)", deleted, len(old_ids))
            return deleted
        except Exception as e:
            session.rollback()
            logger.error("labeled_stores 정리 실패: %s", e)
            raise
        finally:
            session.close()


# ================================================================
# FeatureSetRepository — 피처셋 (04_features)
# ================================================================

class FeatureSetRepository:
    """피처 엔지니어링 결과 저장소"""

    def save_feature_set(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
        target_columns: list[str],
        pipeline_run_id: str,
        scaler_params: dict = None,
        encoder_classes: dict = None,
        source_row_count: int = None,
    ) -> int:
        """
        피처 배열 → feature_sets 테이블에 저장.

        numpy 배열은 BLOB으로 직렬화하고,
        메타데이터(컬럼명, 스케일러 파라미터 등)는 JSON으로 저장합니다.

        Args:
            X: 피처 배열 [N, features]
            y: 타겟 배열 [N, targets]
            feature_columns: 피처 컬럼명 리스트
            target_columns: 타겟 컬럼명 리스트
            pipeline_run_id: 파이프라인 실행 UUID
            scaler_params: StandardScaler 파라미터 dict
            encoder_classes: LabelEncoder 클래스 dict

        Returns:
            저장된 FeatureSet ID
        """
        session = get_session()
        try:
            feature_set = FeatureSet(
                pipeline_run_id=pipeline_run_id,
                feature_columns=feature_columns,
                target_columns=target_columns,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                n_targets=y.shape[1] if y.ndim > 1 else 1,
                feature_data=_serialize_numpy(X),
                target_data=_serialize_numpy(y),
                scaler_params=scaler_params,
                encoder_classes=encoder_classes,
                source_row_count=source_row_count or X.shape[0],
            )
            session.add(feature_set)
            session.commit()

            logger.info(
                "feature_sets 저장: %d samples × %d features (run=%s)",
                X.shape[0], X.shape[1], pipeline_run_id[:8],
            )
            return feature_set.id

        except Exception as e:
            session.rollback()
            logger.error("feature_sets 저장 실패: %s", e)
            raise
        finally:
            session.close()

    def load_feature_set(
        self, pipeline_run_id: str = None
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        feature_sets → numpy 배열 + 메타데이터 로드.

        Args:
            pipeline_run_id: 특정 실행 ID (None이면 최신)

        Returns:
            (X, y, feature_columns, target_columns)
        """
        session = get_session()
        try:
            query = session.query(FeatureSet)
            if pipeline_run_id:
                query = query.filter(FeatureSet.pipeline_run_id == pipeline_run_id)

            feature_set = query.order_by(desc(FeatureSet.created_at)).first()

            if not feature_set:
                raise ValueError("저장된 피처셋이 없습니다")

            X = _deserialize_numpy(feature_set.feature_data)
            y = _deserialize_numpy(feature_set.target_data)

            logger.info(
                "feature_sets 로드: %d samples × %d features (run=%s)",
                X.shape[0], X.shape[1], feature_set.pipeline_run_id[:8],
            )
            return X, y, feature_set.feature_columns, feature_set.target_columns

        finally:
            session.close()

    def get_latest_run_id(self) -> Optional[str]:
        """최신 pipeline_run_id 조회"""
        session = get_session()
        try:
            result = (
                session.query(FeatureSet.pipeline_run_id)
                .order_by(desc(FeatureSet.created_at))
                .first()
            )
            return result[0] if result else None
        finally:
            session.close()


# ================================================================
# TrainingRunRepository — 학습 실행 이력
# ================================================================

class TrainingRunRepository:
    """학습 실행 이력 저장소"""

    def create_run(
        self,
        run_id: str,
        model_type: str,
        pipeline_run_id: str = None,
        model_name: str = None,
        train_size: int = None,
        val_size: int = None,
        test_size: int = None,
        n_features: int = None,
        hyperparameters: dict = None,
    ) -> TrainingRun:
        """
        학습 실행 기록을 생성합니다.

        Args:
            run_id: 학습 실행 UUID
            model_type: 모델 타입 (xgboost, neural_net)
            pipeline_run_id: 사용한 피처셋의 파이프라인 ID

        Returns:
            생성된 TrainingRun 객체
        """
        session = get_session()
        try:
            run = TrainingRun(
                run_id=run_id,
                pipeline_run_id=pipeline_run_id,
                model_type=model_type,
                model_name=model_name,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                n_features=n_features,
                hyperparameters=hyperparameters,
                status="started",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            session.expunge(run)

            logger.info("학습 실행 생성: %s (%s)", run_id[:8], model_type)
            return run

        except Exception as e:
            session.rollback()
            logger.error("학습 실행 생성 실패: %s", e)
            raise
        finally:
            session.close()

    def update_status(
        self,
        run_id: str,
        status: str,
        metrics: dict = None,
        model_path: str = None,
        artifacts_path: str = None,
        error_message: str = None,
    ):
        """
        학습 실행 상태를 업데이트합니다.

        Args:
            run_id: 학습 실행 UUID
            status: 새 상태 (training, evaluating, completed, failed)
            metrics: 평가 결과 dict
        """
        session = get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                logger.error("학습 실행 없음: %s", run_id)
                return

            run.status = status
            if metrics:
                run.metrics = metrics
            if model_path:
                run.model_path = model_path
            if artifacts_path:
                run.artifacts_path = artifacts_path
            if error_message:
                run.error_message = error_message
            if status in ("completed", "failed"):
                run.completed_at = datetime.utcnow()

            session.commit()
            logger.info("학습 실행 업데이트: %s → %s", run_id[:8], status)

        except Exception as e:
            session.rollback()
            logger.error("학습 실행 업데이트 실패: %s", e)
            raise
        finally:
            session.close()

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        """특정 학습 실행 조회"""
        session = get_session()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if run:
                session.expunge(run)
            return run
        finally:
            session.close()

    def get_latest_run(self, model_type: str = None) -> Optional[TrainingRun]:
        """최신 학습 실행 조회"""
        session = get_session()
        try:
            query = session.query(TrainingRun).filter(TrainingRun.status == "completed")
            if model_type:
                query = query.filter(TrainingRun.model_type == model_type)
            run = query.order_by(desc(TrainingRun.completed_at)).first()
            if run:
                session.expunge(run)
            return run
        finally:
            session.close()