"""
📁 scripts/run_collect.py
===========================
상가 데이터 수집 실행 스크립트.

실행:
  python scripts/run_collect.py                                  # 서울 전체 (CSV 기반)
  python scripts/run_collect.py --sido 41                        # 경기도
  python scripts/run_collect.py --sgg "강남구,서초구"               # 특정 구만
  python scripts/run_collect.py --dong "역삼동,서교동"              # 특정 동만
  python scripts/run_collect.py --csv data/00_region_codes/dong_codes_all.csv  # CSV 직접 지정
  python scripts/run_collect.py --limit 5                        # 테스트 (5개 행정동만)
  python scripts/run_collect.py --codes 1168010100,1168010200    # 코드 직접 지정

필요 설정:
  1. 먼저 법정동코드 수집: make collect-regions
  2. .env에 PUBLIC_DATA_SERVICE_KEY 설정
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import setup_logging, get_logger
from src.data_collection.collector import DataCollector

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="상가 데이터 수집")
    parser.add_argument("--sido", type=str, default="11", help="시도코드 (기본: 11=서울)")
    parser.add_argument("--sgg", type=str, default=None, help="시군구 필터 (쉼표 구분, 예: 강남구,서초구)")
    parser.add_argument("--dong", type=str, default=None, help="읍면동 필터 (쉼표 구분, 예: 역삼동,서교동)")
    parser.add_argument("--csv", type=str, default=None, help="법정동코드 CSV 경로 직접 지정")
    parser.add_argument("--codes", type=str, default=None, help="행정동코드 직접 지정 (쉼표 구분)")
    parser.add_argument("--limit", type=int, default=None, help="최대 행정동 수 (테스트용)")
    args = parser.parse_args()

    setup_logging()

    collector = DataCollector()

    # 파라미터 가공
    sgg_filter = args.sgg.split(",") if args.sgg else None
    dong_filter = args.dong.split(",") if args.dong else None
    dong_codes = args.codes.split(",") if args.codes else None

    # 수집
    df = collector.collect(
        dong_codes=dong_codes,
        region_csv=args.csv,
        sido_cd=args.sido,
        sgg_filter=sgg_filter,
        dong_filter=dong_filter,
        limit=args.limit,
    )

    if not df.empty:
        logger.info("수집 결과: %d행 × %d열", *df.shape)
    else:
        logger.warning("수집 결과가 비어있습니다.")


if __name__ == "__main__":
    main()