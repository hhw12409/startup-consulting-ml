"""
📁 scripts/run_collect_hdong_codes.py
=======================================
행정동코드 수집 스크립트 (PublicDataReader 사용).

[중요] 소상공인 상가 API는 행정동코드(adongCd)를 사용합니다!
  - 행정동코드: 1168058000 (강남구 역삼1동) ← 상가 API에 사용
  - 법정동코드: 1168010100 (강남구 역삼동)  ← 상가 API에 안 됨!

실행:
  pip install PublicDataReader        # 1회만
  python scripts/run_collect_hdong_codes.py                # 서울 (기본)
  python scripts/run_collect_hdong_codes.py --sido 서울특별시
  python scripts/run_collect_hdong_codes.py --sido all      # 전국
  python scripts/run_collect_hdong_codes.py --sido 경기도

※ API 키 불필요! PublicDataReader가 행정안전부 데이터를 자동으로 가져옵니다.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

try:
    import PublicDataReader as pdr
except ImportError:
    print("❌ PublicDataReader가 필요합니다. 설치해주세요:")
    print("   pip install PublicDataReader")
    sys.exit(1)

from src.utils.logger import setup_logging, get_logger
from src.utils.timer import timer

logger = get_logger(__name__)

# 시도명 → 시도코드 매핑
SIDO_MAP = {
    "서울특별시": "11", "부산광역시": "26", "대구광역시": "27",
    "인천광역시": "28", "광주광역시": "29", "대전광역시": "30",
    "울산광역시": "31", "세종특별자치시": "36", "경기도": "41",
    "강원특별자치도": "42", "충청북도": "43", "충청남도": "44",
    "전북특별자치도": "45", "전라남도": "46", "경상북도": "47",
    "경상남도": "48", "제주특별자치도": "50",
}


@timer("행정동코드 수집")
def main():
    parser = argparse.ArgumentParser(description="행정동코드 수집")
    parser.add_argument("--sido", type=str, default="서울특별시",
                        help="시도명 (기본: 서울특별시, all=전국)")
    parser.add_argument("--output", type=str, default="data/00_region_codes",
                        help="출력 디렉토리")
    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 행정동코드 조회 (API 키 불필요) ──
    logger.info("📥 행정동코드 조회 중... (PublicDataReader)")
    hdong = pdr.code_hdong()
    logger.info("전국 행정동: %d개", len(hdong))
    logger.info("컬럼: %s", list(hdong.columns))

    # ── 2. 시도 필터 ──
    if args.sido == "all":
        df = hdong.copy()
        sido_cd = "all"
        logger.info("전국 행정동: %d개", len(df))
    else:
        # 시도명으로 필터
        sido_nm = args.sido
        df = hdong[hdong["시도명"] == sido_nm].copy()

        if df.empty:
            # 부분 매칭 시도
            matches = hdong[hdong["시도명"].str.contains(sido_nm)]
            if not matches.empty:
                sido_nm = matches["시도명"].iloc[0]
                df = hdong[hdong["시도명"] == sido_nm].copy()
                logger.info("'%s' → '%s' 매칭", args.sido, sido_nm)
            else:
                logger.error("시도명 '%s'를 찾을 수 없습니다.", args.sido)
                logger.error("사용 가능한 시도명: %s", list(hdong["시도명"].unique()))
                return

        sido_cd = SIDO_MAP.get(sido_nm, "00")
        logger.info("%s 행정동: %d개", sido_nm, len(df))

    # ── 3. 상가 API용 컬럼 정리 ──
    # PublicDataReader 출력: 행정동코드, 시도명, 시군구명, 읍면동명, (생성일)
    # collector에서 사용하는 형식으로 변환
    result = pd.DataFrame({
        "region_cd": df["행정동코드"],                              # 10자리 행정동코드
        "region_cd_8": df["행정동코드"].str[:8],                    # 8자리 (상가 API 호환)
        "sido_cd": df["행정동코드"].str[:2],                        # 시도코드
        "sgg_cd": df["행정동코드"].str[2:5],                        # 시군구코드
        "dong_cd": df["행정동코드"].str[5:8],                       # 읍면동코드
        "sido_nm": df["시도명"].values,
        "sgg_nm": df["시군구명"].values,
        "dong_nm": df["읍면동명"].values,
        "full_nm": df["시도명"].values + " " + df["시군구명"].values + " " + df["읍면동명"].values,
    })

    result = result.reset_index(drop=True)

    # ── 4. CSV 저장 ──
    filename = f"hdong_codes_{sido_cd}.csv"
    out_path = output_dir / filename
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info("━━━ 행정동코드 수집 완료 ━━━")
    logger.info("  저장: %s (%d건)", out_path, len(result))
    logger.info("  컬럼: %s", list(result.columns))
    logger.info("")
    logger.info("  다음 단계: make collect")
    logger.info("  (collector가 hdong_codes_*.csv를 자동으로 읽습니다)")
    logger.info("")

    # 미리보기
    print(f"\n📋 {sido_nm if args.sido != 'all' else '전국'} 행정동코드 미리보기:")
    print(result[["region_cd", "region_cd_8", "dong_nm", "full_nm"]].head(10).to_string(index=False))
    print(f"\n... 총 {len(result)}개")


if __name__ == "__main__":
    main()