"""
📁 src/data_collection/collector.py
====================================
데이터 수집 오케스트레이터.

[패턴] Facade — 여러 API 클라이언트를 조합하여 하나의 단순한 인터페이스로 제공
[역할] 법정동코드 CSV를 읽어 상가 데이터 + 사업자 상태를 수집합니다.

[수집 흐름]
  1. data/00_region_codes/dong_codes_*.csv 에서 법정동코드 로드
  2. 행정동코드별로 소상공인 상가 API 호출
  3. 사업자번호로 국세청 API 호출 (생존/폐업 라벨)
  4. data/01_raw/stores_raw.csv 저장

[사용법]
  collector = DataCollector()

  # CSV에서 법정동코드 읽어서 수집 (기본: 서울)
  df = collector.collect()

  # 특정 시도 CSV 지정
  df = collector.collect(region_csv="data/00_region_codes/dong_codes_41.csv")

  # 특정 구만 필터링
  df = collector.collect(sgg_filter=["강남구", "서초구"])

  # 행정동코드 직접 지정 (기존 방식도 가능)
  df = collector.collect(dong_codes=["1168010100"])
"""

import pandas as pd
from pathlib import Path

from config.settings import get_settings
from src.data_collection.public_data_client import PublicDataClient
from src.data_collection.nts_client import NtsClient
from src.utils.logger import get_logger
from src.utils.io import save_csv
from src.utils.timer import timer

logger = get_logger(__name__)


class DataCollector:
    """
    데이터 수집 Facade.

    법정동코드 CSV를 읽어 전국 어디든 수집할 수 있습니다.

    사용법:
        collector = DataCollector()

        # 방법 1: CSV 기반 (권장)
        df = collector.collect()                                   # 서울 전체
        df = collector.collect(sgg_filter=["강남구", "마포구"])       # 특정 구만
        df = collector.collect(region_csv="dong_codes_41.csv")     # 경기도

        # 방법 2: 행정동코드 직접 지정
        df = collector.collect(dong_codes=["1168010100"])
    """

    # 기본 법정동코드 CSV 경로
    DEFAULT_REGION_DIR = "data/00_region_codes"

    def __init__(self):
        self._store_client = PublicDataClient()
        self._nts_client = NtsClient()
        self._settings = get_settings()

    # ================================================================
    # 공개 메서드
    # ================================================================

    @timer("전체 데이터 수집")
    def collect(
            self,
            dong_codes: list[str] = None,
            region_csv: str = None,
            sido_cd: str = "11",
            sgg_filter: list[str] = None,
            dong_filter: list[str] = None,
            limit: int = None,
    ) -> pd.DataFrame:
        """
        상가 데이터를 수집하고 01_raw/에 저장합니다.

        Args:
            dong_codes: 행정동코드 직접 지정 (이 값이 있으면 CSV 무시)
            region_csv: 법정동코드 CSV 경로 (None이면 자동 탐색)
            sido_cd: 시도코드 (기본 "11"=서울, CSV 자동 탐색용)
            sgg_filter: 시군구명 필터 (예: ["강남구", "서초구"])
            dong_filter: 읍면동명 필터 (예: ["역삼동", "서교동"])
            limit: 최대 행정동 수 (테스트용)

        Returns:
            수집된 원본 DataFrame
        """
        # ── 1. 행정동코드 결정 ──
        if dong_codes:
            codes = dong_codes
            logger.info("직접 지정된 행정동코드: %d개", len(codes))
        else:
            codes = self._load_dong_codes(
                region_csv=region_csv,
                sido_cd=sido_cd,
                sgg_filter=sgg_filter,
                dong_filter=dong_filter,
            )

        if not codes:
            logger.error("수집할 행정동코드가 없습니다.")
            logger.error("먼저 'make collect-regions' 로 법정동코드를 수집하세요.")
            return pd.DataFrame()

        if limit:
            codes = codes[:limit]
            logger.info("limit=%d 적용 → %d개 행정동만 수집", limit, len(codes))

        logger.info("━━━ 수집 시작: %d개 행정동 ━━━", len(codes))

        # ── 2. 상가업소 조회 ──
        dfs = []
        for idx, code in enumerate(codes, 1):
            logger.info("  [%d/%d] 행정동 %s 수집 중...", idx, len(codes), code)
            df = self._store_client.get_stores_by_dong(code)
            if not df.empty:
                df["dong_code"] = code
                dfs.append(df)
                logger.info("  [%d/%d] %s → %d건", idx, len(codes), code, len(df))
            else:
                logger.info("  [%d/%d] %s → 0건 (데이터 없음)", idx, len(codes), code)

        if not dfs:
            logger.warning("수집된 데이터 없음. API 키를 확인하세요.")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        logger.info("상가 데이터 수집 완료: %d건", len(combined))

        # ── 3. 사업자 상태 병합 (생존/폐업 라벨용) ──
        try:
            if "bizesId" in combined.columns:
                biz_nums = combined["bizesId"].dropna().unique().tolist()
                if biz_nums:
                    logger.info("사업자 상태 조회: %d건", len(biz_nums))
                    status = self._nts_client.check_status(biz_nums)
                    if not status.empty:
                        combined = combined.merge(
                            status[["b_no", "b_stt_cd", "end_dt"]],
                            left_on="bizesId", right_on="b_no", how="left",
                        )
                        logger.info("사업자 상태 병합 완료")
        except Exception as e:
            logger.warning("⚠️ 국세청 API 실패 (상가 데이터는 정상 저장됩니다): %s", e)

        # ── 4. DB 저장 ──
        from src.database.repository import StoreRepository
        repo = StoreRepository()
        saved_count = repo.upsert_stores(combined)
        logger.info("━━━ 수집 완료: %d건 → DB (stores 테이블) ━━━", saved_count)
        return combined

    # ================================================================
    # 법정동코드 CSV 로드
    # ================================================================

    def _load_dong_codes(
            self,
            region_csv: str = None,
            sido_cd: str = "11",
            sgg_filter: list[str] = None,
            dong_filter: list[str] = None,
    ) -> list[str]:
        """
        법정동코드 CSV에서 행정동코드 목록을 로드합니다.

        탐색 순서:
          1. region_csv가 지정되면 그 파일 사용
          2. data/00_region_codes/dong_codes_{sido_cd}.csv 자동 탐색
          3. data/00_region_codes/dong_codes_all.csv (전국 파일)
          4. 모두 없으면 빈 리스트

        CSV 컬럼 규격 (RegionCodeCollector 출력):
          region_cd | sido_cd | sgg_cd | dong_cd | dong_nm | full_nm | flag
        """
        csv_path = self._find_region_csv(region_csv, sido_cd)

        if csv_path is None:
            # CSV 없으면 DB에서 행정동코드 로드 시도
            try:
                from src.database.repository import RegionRepository
                region_repo = RegionRepository()
                codes = region_repo.get_dong_codes(sido_cd=sido_cd)
                if codes:
                    logger.info("DB에서 행정동코드 로드: %d개 (sido_cd=%s)", len(codes), sido_cd)
                    if sgg_filter or dong_filter:
                        logger.warning("DB 로드 시 sgg_filter/dong_filter는 지원되지 않습니다")
                    return codes
            except Exception as e:
                logger.warning("DB 행정동코드 로드 실패: %s", e)

            logger.warning("법정동코드를 찾을 수 없습니다 (CSV/DB 모두 없음).")
            logger.warning("  먼저 실행: make collect-regions 또는 make db-migrate --regions")
            return []

        logger.info("법정동코드 CSV 로드: %s", csv_path)
        df = pd.read_csv(csv_path, dtype=str)
        logger.info("  전체 행정동: %d개", len(df))

        # sido_cd 필터 (CSV에 전국 데이터가 섞여있을 수 있음)
        if sido_cd and "sido_cd" in df.columns:
            before = len(df)
            df = df[df["sido_cd"] == sido_cd]
            if len(df) < before:
                logger.info("  시도 필터 (sido_cd=%s): %d → %d개", sido_cd, before, len(df))

        # hdong CSV는 이미 행정동 레벨만 포함 → 읍면동 필터 불필요
        is_hdong_csv = "region_cd_8" in df.columns

        if is_hdong_csv:
            # 시도/시군구 레벨 코드 제거 (끝 4자리가 0000인 것)
            before = len(df)
            df = df[~df["region_cd_8"].str.endswith("0000")]
            if len(df) < before:
                logger.info("  시도/시군구 레벨 제거: %d → %d개", before, len(df))
        elif "region_cd" in df.columns:
            # 법정동 CSV일 때만 읍면동 레벨 필터 (리 단위 제외)
            before = len(df)
            df = df[
                (df["region_cd"].str.len() == 10) &
                (df["region_cd"].str[5:8] != "000") &
                (df["region_cd"].str[8:10] == "00")
                ]
            if len(df) < before:
                logger.info("  읍면동 레벨 필터: %d → %d개", before, len(df))

        # 시군구명이 "소계"인 행 제거 (hdong CSV에 포함될 수 있음)
        if "dong_nm" in df.columns:
            df = df[~df["dong_nm"].isin(["소계", "합계", ""])]

        # 존재하는(폐지되지 않은) 법정동만 필터링
        if "flag" in df.columns:
            before = len(df)
            df = df[df["flag"] == "Y"]
            if len(df) < before:
                logger.info("  존재(flag=Y) 필터: %d → %d개", before, len(df))

        # 시군구 필터
        if sgg_filter and "full_nm" in df.columns:
            before = len(df)
            df = df[df["full_nm"].apply(
                lambda x: any(sgg in str(x) for sgg in sgg_filter)
            )]
            logger.info("  시군구 필터 %s: %d → %d개", sgg_filter, before, len(df))

        # 읍면동 필터
        if dong_filter and "dong_nm" in df.columns:
            before = len(df)
            df = df[df["dong_nm"].isin(dong_filter)]
            logger.info("  읍면동 필터 %s: %d → %d개", dong_filter, before, len(df))

        if df.empty:
            logger.warning("필터 결과 0건. 필터 조건을 확인하세요.")
            return []

        # 행정동코드 8자리 우선 (상가 API 호환), 없으면 region_cd 사용
        if "region_cd_8" in df.columns:
            codes = df["region_cd_8"].tolist()
            logger.info("  코드 형식: 행정동 8자리 (상가 API 호환)")
        else:
            codes = df["region_cd"].tolist()
            logger.info("  코드 형식: %d자리", len(codes[0]) if codes else 0)

        logger.info("  최종 수집 대상: %d개 행정동", len(codes))
        return codes

    def _find_region_csv(self, region_csv: str = None, sido_cd: str = "11") -> str | None:
        """행정동코드 CSV 파일을 탐색합니다. (hdong 우선)"""
        region_dir = Path(self.DEFAULT_REGION_DIR)

        # 1. 직접 지정
        if region_csv:
            p = Path(region_csv)
            if p.exists():
                return str(p)
            p = region_dir / region_csv
            if p.exists():
                return str(p)
            logger.warning("지정된 CSV 없음: %s", region_csv)

        # 2. 행정동코드 파일 우선 (상가 API 호환)
        hdong_file = region_dir / f"hdong_codes_{sido_cd}.csv"
        if hdong_file.exists():
            return str(hdong_file)

        hdong_all = region_dir / "hdong_codes_all.csv"
        if hdong_all.exists():
            return str(hdong_all)

        # 3. 법정동코드 파일 (폴백)
        dong_file = region_dir / f"dong_codes_{sido_cd}.csv"
        if dong_file.exists():
            logger.warning("⚠️ 법정동코드 CSV 사용 중. 상가 API와 호환되지 않을 수 있습니다.")
            logger.warning("  행정동코드 수집 권장: python scripts/run_collect_hdong_codes.py")
            return str(dong_file)

        # 4. 아무 hdong_codes_*.csv
        if region_dir.exists():
            candidates = list(region_dir.glob("hdong_codes_*.csv"))
            if candidates:
                return str(candidates[0])
            # 법정동 폴백
            candidates = list(region_dir.glob("dong_codes_*.csv"))
            if candidates:
                logger.warning("⚠️ 법정동코드 CSV 폴백: %s", candidates[0])
                return str(candidates[0])

        return None