"""
📁 scripts/run_collect_region_codes.py
========================================
법정동코드 수집 스크립트.

실행:
  python scripts/run_collect_region_codes.py                    # 서울시 전체
  python scripts/run_collect_region_codes.py --sido 11          # 서울시
  python scripts/run_collect_region_codes.py --sido 11 --sgg 680  # 서울 강남구만
  python scripts/run_collect_region_codes.py --sido all         # 전국 (시간 오래 걸림)
  python scripts/run_collect_region_codes.py --search "강남구"    # 지역명 검색

필요 설정:
  .env 파일에 REGION_CODE_API_KEY=your_key 추가
  (공공데이터포털 활용신청: https://www.data.go.kr/data/15077871/openapi.do)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.region_code_collector.region_code_collector import RegionCodeCollector
from src.utils.logger import setup_logging, get_logger
from src.utils.timer import timer

logger = get_logger(__name__)

# 주요 시도코드
SIDO_CODES = {
    "11": "서울특별시",     "26": "부산광역시",   "27": "대구광역시",
    "28": "인천광역시",     "29": "광주광역시",   "30": "대전광역시",
    "31": "울산광역시",     "36": "세종특별자치시",
    "41": "경기도",         "42": "강원특별자치도", "43": "충청북도",
    "44": "충청남도",       "45": "전북특별자치도", "46": "전라남도",
    "47": "경상북도",       "48": "경상남도",     "50": "제주특별자치도",
}


@timer("법정동코드 수집")
def main():
    parser = argparse.ArgumentParser(description="법정동코드 수집")
    parser.add_argument("--sido", type=str, default="11", help="시도코드 (기본: 11=서울, all=전국)")
    parser.add_argument("--sgg", type=str, default=None, help="시군구코드 3자리 (선택)")
    parser.add_argument("--search", type=str, default=None, help="지역명 검색 (예: 강남구)")
    parser.add_argument("--output", type=str, default="data/00_region_codes", help="출력 디렉토리")
    args = parser.parse_args()

    setup_logging()
    collector = RegionCodeCollector()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 지역명 검색 모드 ──
    if args.search:
        logger.info("🔍 지역명 검색: '%s'", args.search)
        df = collector.collect(locatadd_nm=args.search)
        if df.empty:
            logger.warning("검색 결과 없음: '%s'", args.search)
            return

        output_path = output_dir / f"search_{args.search}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("검색 결과 저장: %s (%d건)", output_path, len(df))
        print(df.to_string(index=False))
        return

    # ── 전국 수집 모드 ──
    if args.sido == "all":
        logger.info("🇰🇷 전국 법정동코드 수집 시작")
        all_dfs = []

        for sido_cd, sido_nm in SIDO_CODES.items():
            logger.info("  ▶ %s (%s) 수집 중...", sido_nm, sido_cd)
            df = collector.collect_dong(sido_cd=sido_cd)
            if not df.empty:
                all_dfs.append(df)
                logger.info("  ✅ %s: %d개 읍면동", sido_nm, len(df))

        if all_dfs:
            all_df = pd.concat(all_dfs, ignore_index=True)
            output_path = output_dir / "dong_codes_all.csv"
            all_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info("전국 수집 완료: %s (%d건)", output_path, len(all_df))
        return

    # ── 특정 시도 수집 모드 ──
    sido_nm = SIDO_CODES.get(args.sido, args.sido)
    logger.info("📍 %s (%s) 법정동코드 수집 시작", sido_nm, args.sido)

    if args.sgg:
        # 특정 시군구만
        df = collector.collect_dong(sido_cd=args.sido, sgg_cd=args.sgg)
        if not df.empty:
            output_path = output_dir / f"dong_codes_{args.sido}_{args.sgg}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info("저장: %s (%d건)", output_path, len(df))
            print(f"\n{df.to_string(index=False)}")
    else:
        # 시도 전체 (시도 + 시군구 + 읍면동)
        paths = collector.save_all(output_dir=str(output_dir), sido_cd=args.sido)
        logger.info("━━━ 수집 완료 ━━━")
        for level, path in paths.items():
            logger.info("  %s: %s", level, path)


# pandas import (전국 모드에서 사용)
import pandas as pd

if __name__ == "__main__":
    main()