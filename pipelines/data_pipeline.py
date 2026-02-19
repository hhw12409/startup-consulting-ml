"""
pipelines/data_pipeline.py
===============================
데이터 수집 -> 정제 -> 라벨 생성 파이프라인.

[패턴] Template Method -- 단계별 골격을 정의하고 각 단계는 교체 가능
[역할] 공공데이터 수집 -> 정제 -> 라벨 생성까지의 전체 흐름

데이터 흐름:
  수집 -> stores 테이블 (collector 내부)
  정제 -> cleaned_stores 테이블
  라벨 -> labeled_stores 테이블
"""

import uuid

import pandas as pd

from config.settings import get_settings
from src.data_collection.collector import DataCollector
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.labeler import LabelGenerator
from src.preprocessing.merger import DataMerger
from src.database.repository import CleanedStoreRepository, LabeledStoreRepository
from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger(__name__)


class DataPipeline:
    """
    데이터 수집 -> 정제 -> 라벨 생성 파이프라인.

    사용법:
        pipeline = DataPipeline()
        df = pipeline.run(dong_codes=["11680710", "11680720"])
    """

    def __init__(self):
        self._collector = DataCollector()
        self._cleaner = DataCleaner()
        self._labeler = LabelGenerator()
        self._merger = DataMerger()
        self._settings = get_settings()
        self._cleaned_repo = CleanedStoreRepository()
        self._labeled_repo = LabeledStoreRepository()

    @timer("데이터 파이프라인")
    def run(self, dong_codes: list[str] = None) -> pd.DataFrame:
        """
        전체 데이터 파이프라인 실행.

        단계:
        1. 수집 (collector) -> stores 테이블
        2. 정제 (cleaner)  -> cleaned_stores 테이블
        3. 라벨 (labeler)  -> labeled_stores 테이블
        """
        pipeline_run_id = str(uuid.uuid4())
        logger.info("파이프라인 실행 ID: %s", pipeline_run_id[:8])

        # Step 1: 수집 (collector가 내부적으로 stores 테이블에 저장)
        logger.info("━━━ Step 1: 데이터 수집 ━━━")
        raw_df = self._collector.collect(dong_codes)

        if raw_df.empty:
            logger.error("수집 데이터가 없습니다. API 키를 확인하세요.")
            return pd.DataFrame()

        # Step 2: 정제 -> cleaned_stores 테이블 (이전 run 정리 후 저장)
        logger.info("━━━ Step 2: 데이터 정제 ━━━")
        clean_df = self._cleaner.clean(raw_df)
        self._cleaned_repo.delete_old_runs(keep_latest=0)
        self._cleaned_repo.save_cleaned(clean_df, pipeline_run_id)

        # Step 3: 라벨 생성 -> labeled_stores 테이블 (이전 run 정리 후 저장)
        logger.info("━━━ Step 3: 라벨 생성 ━━━")
        labeled_df = self._labeler.generate(clean_df)
        self._labeled_repo.delete_old_runs(keep_latest=0)
        self._labeled_repo.save_labeled(labeled_df, pipeline_run_id)

        logger.info(
            "데이터 파이프라인 완료: %d행 → DB (pipeline_run=%s)",
            len(labeled_df), pipeline_run_id[:8],
        )
        return labeled_df
