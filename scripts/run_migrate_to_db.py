"""
📁 scripts/run_migrate_to_db.py
==================================
CSV 데이터 → MySQL 마이그레이션.

기존에 수집한 stores_raw.csv, hdong_codes_*.csv를 DB에 저장합니다.

실행:
  python scripts/run_migrate_to_db.py              # 전체 마이그레이션
  python scripts/run_migrate_to_db.py --stores      # 상가 데이터만
  python scripts/run_migrate_to_db.py --regions     # 행정동 코드만

필요:
  docker-compose up -d   (MySQL 먼저 실행)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from config.settings import get_settings
from src.database.repository import StoreRepository, RegionRepository
from src.utils.logger import setup_logging, get_logger
from src.utils.timer import timer

logger = get_logger(__name__)


@timer("CSV → DB 마이그레이션")
def migrate_stores(data_path: str):
    """stores_raw.csv → stores 테이블"""
    path = Path(data_path)
    if not path.exists():
        logger.error("파일 없음: %s (make collect 먼저 실행)", data_path)
        return

    logger.info("📥 CSV 로드: %s", data_path)
    df = pd.read_csv(data_path, dtype=str, low_memory=False)
    logger.info("  %d행 × %d열", *df.shape)

    repo = StoreRepository()
    saved = repo.upsert_stores(df)
    logger.info("✅ stores 테이블 저장: %d건", saved)

    # 저장 확인
    total = repo.get_store_count()
    logger.info("  DB 전체 상가 수: %d건", total)


@timer("행정동 코드 마이그레이션")
def migrate_regions():
    """hdong_codes_*.csv → region_codes 테이블"""
    settings = get_settings()
    region_dir = Path(settings.DATA_RAW).parent / "00_region_codes"

    csv_files = sorted(region_dir.glob("hdong_codes_*.csv"))
    if not csv_files:
        logger.warning("행정동 코드 CSV 없음: %s", region_dir)
        return

    repo = RegionRepository()

    for csv_path in csv_files:
        logger.info("📥 로드: %s", csv_path.name)
        df = pd.read_csv(csv_path, dtype=str)

        # 시도/시군구 레벨 제거
        if "region_cd_8" in df.columns:
            before = len(df)
            df = df[~df["region_cd_8"].str.endswith("0000")]
            logger.info("  필터: %d → %d건 (시도/시군구 제거)", before, len(df))

        saved = repo.upsert_regions(df)
        logger.info("  저장: %d건", saved)


@timer("정제 데이터 마이그레이션")
def migrate_cleaned(data_path: str, pipeline_run_id: str):
    """cleaned.csv → cleaned_stores 테이블"""
    path = Path(data_path)
    if not path.exists():
        logger.info("정제 데이터 없음 (건너뜀): %s", data_path)
        return

    logger.info("CSV 로드: %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    logger.info("  %d행 × %d열", *df.shape)

    from src.database.repository import CleanedStoreRepository
    repo = CleanedStoreRepository()
    saved = repo.save_cleaned(df, pipeline_run_id)
    logger.info("cleaned_stores 저장: %d건", saved)


@timer("라벨 데이터 마이그레이션")
def migrate_labeled(data_path: str, pipeline_run_id: str):
    """labeled.csv → labeled_stores 테이블"""
    path = Path(data_path)
    if not path.exists():
        logger.info("라벨 데이터 없음 (건너뜀): %s", data_path)
        return

    logger.info("CSV 로드: %s", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    logger.info("  %d행 × %d열", *df.shape)

    from src.database.repository import LabeledStoreRepository
    repo = LabeledStoreRepository()
    saved = repo.save_labeled(df, pipeline_run_id)
    logger.info("labeled_stores 저장: %d건", saved)


def main():
    parser = argparse.ArgumentParser(description="CSV → DB 마이그레이션")
    parser.add_argument("--stores", action="store_true", help="상가 데이터만")
    parser.add_argument("--regions", action="store_true", help="행정동 코드만")
    parser.add_argument("--cleaned", action="store_true", help="정제 데이터만")
    parser.add_argument("--labeled", action="store_true", help="라벨 데이터만")
    parser.add_argument("--all", action="store_true", help="전체 마이그레이션")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    # 아무것도 지정 안 하면 전체
    no_flags = not any([args.stores, args.regions, args.cleaned, args.labeled])
    do_all = args.all or no_flags

    logger.info("━━━ CSV → MySQL 마이그레이션 ━━━")
    logger.info("DB: %s", settings.DATABASE_URL.split("@")[-1])

    if do_all or args.regions:
        migrate_regions()

    if do_all or args.stores:
        migrate_stores(f"{settings.DATA_RAW}/stores_raw.csv")

    # 중간 데이터 마이그레이션 (공통 pipeline_run_id 사용)
    import uuid
    pipeline_run_id = str(uuid.uuid4())

    if do_all or args.cleaned:
        migrate_cleaned(f"{settings.DATA_INTERIM}/cleaned.csv", pipeline_run_id)

    if do_all or args.labeled:
        migrate_labeled(f"{settings.DATA_PROCESSED}/labeled.csv", pipeline_run_id)

    logger.info("")
    logger.info("마이그레이션 완료")


if __name__ == "__main__":
    main()