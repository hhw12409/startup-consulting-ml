"""
📁 src/data/region_code_collector.py
=======================================
법정동코드 수집기 (공공데이터포털 API).

[API 정보]
  서비스명: 행정안전부_행정표준코드_법정동코드
  엔드포인트: http://apis.data.go.kr/1741000/StanReginCd/getStanReginCdList
  인증: 공공데이터포털 서비스키 (REGION_CODE_API_KEY)
  활용신청: https://www.data.go.kr/data/15077871/openapi.do

[법정동코드 체계 (10자리)]
  시도(2) + 시군구(3) + 읍면동(3) + 리(2)
  예) 1168010100 = 서울(11) 강남구(680) 역삼동(101) 00

[사용법]
  collector = RegionCodeCollector(api_key="your_key")

  # 서울시 전체 법정동 수집
  df = collector.collect(sido_cd="11")

  # 서울시 강남구만
  df = collector.collect(sido_cd="11", sgg_cd="680")

  # 전국 시도 목록
  df = collector.collect_sido()

  # CSV로 저장
  collector.save_all(output_dir="data/00_region_codes")
"""

import os
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# 서울시 시군구코드 매핑 (참고용)
SEOUL_SGG_CODES = {
    "110": "종로구",   "140": "중구",     "170": "용산구",
    "200": "성동구",   "215": "광진구",   "230": "동대문구",
    "260": "중랑구",   "290": "성북구",   "305": "강북구",
    "320": "도봉구",   "350": "노원구",   "380": "은평구",
    "410": "서대문구", "440": "마포구",   "470": "양천구",
    "500": "강서구",   "530": "구로구",   "545": "금천구",
    "560": "영등포구", "590": "동작구",   "620": "관악구",
    "650": "서초구",   "680": "강남구",   "710": "송파구",
    "740": "강동구",
}


class RegionCodeCollector:
    """
    공공데이터포털 법정동코드 수집기.

    API 스펙:
      - serviceKey: 인증키 (필수)
      - pageNo: 페이지 번호 (기본 1)
      - numOfRows: 한 페이지 결과 수 (기본 100, 최대 1000)
      - type: 응답 타입 (xml/json, 기본 xml)
      - locatadd_nm: 지역 주소명 검색 (선택)
      - flag: 사용 여부 (Y/N, 선택)
      - pg_yn: 하위 포함 여부 (N=해당 레벨만, Y=하위 포함)
      - up_cd: 상위 코드 (선택)
      - low_cd: 하위 코드 (선택)
    """

    BASE_URL = "http://apis.data.go.kr/1741000/StanReginCd/getStanReginCdList"

    def __init__(self, api_key: str = None):
        self._api_key = api_key or os.getenv("REGION_CODE_API_KEY", "")
        if not self._api_key:
            logger.warning("REGION_CODE_API_KEY 미설정. .env에 추가하세요.")

    # ================================================================
    # 공개 메서드
    # ================================================================

    def collect_sido(self) -> pd.DataFrame:
        """
        전국 시도(17개) 목록 수집.

        Returns:
            DataFrame [region_cd, sido_cd, sido_nm]
        """
        logger.info("시도 목록 수집 시작")
        rows = self._fetch_all(params={"pg_yn": "N"})
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 시도 레벨만 필터 (시군구 이하가 00000000인 것)
        df = df[df["region_cd"].str[2:] == "00000000"].copy()
        df["sido_cd"] = df["region_cd"].str[:2]
        df["sido_nm"] = df["locatadd_nm"]

        logger.info("시도 수집 완료: %d개", len(df))
        return df[["region_cd", "sido_cd", "sido_nm"]].reset_index(drop=True)

    def collect_sgg(self, sido_cd: str = "11") -> pd.DataFrame:
        """
        특정 시도의 시군구 목록 수집.

        Args:
            sido_cd: 시도코드 2자리 (기본 "11" = 서울)

        Returns:
            DataFrame [region_cd, sido_cd, sgg_cd, sgg_nm, full_nm]
        """
        logger.info("시군구 수집 시작: 시도=%s", sido_cd)
        up_cd = f"{sido_cd}00000000"
        rows = self._fetch_all(params={"up_cd": up_cd, "pg_yn": "N"})
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # 시군구 레벨 (읍면동 이하가 00000인 것)
        df = df[
            (df["region_cd"].str[:2] == sido_cd) &
            (df["region_cd"].str[5:] == "00000") &
            (df["region_cd"].str[2:5] != "000")
            ].copy()

        df["sido_cd"] = df["region_cd"].str[:2]
        df["sgg_cd"] = df["region_cd"].str[2:5]
        df["sgg_nm"] = df["locatadd_nm"].apply(lambda x: x.split()[-1] if " " in x else x)
        df["full_nm"] = df["locatadd_nm"]

        logger.info("시군구 수집 완료: %d개", len(df))
        return df[["region_cd", "sido_cd", "sgg_cd", "sgg_nm", "full_nm"]].reset_index(drop=True)

    def collect_dong(self, sido_cd: str = "11", sgg_cd: str = None) -> pd.DataFrame:
        """
        읍면동 수집 (핵심 메서드).

        Args:
            sido_cd: 시도코드 2자리
            sgg_cd: 시군구코드 3자리 (None이면 시도 전체)

        Returns:
            DataFrame [region_cd, sido_cd, sgg_cd, dong_cd, dong_nm, full_nm, flag]
        """
        if sgg_cd:
            logger.info("읍면동 수집: 시도=%s, 시군구=%s", sido_cd, sgg_cd)
            up_cd = f"{sido_cd}{sgg_cd}00000"
            rows = self._fetch_all(params={"up_cd": up_cd, "pg_yn": "N"})
        else:
            logger.info("읍면동 수집: 시도=%s (전체)", sido_cd)
            # 시도 전체 → 시군구별로 수집
            sgg_df = self.collect_sgg(sido_cd)
            if sgg_df.empty:
                return pd.DataFrame()

            all_rows = []
            for idx, sgg in sgg_df.iterrows():
                sgg_name = sgg.get("sgg_nm", sgg["sgg_cd"])
                logger.info(
                    "  [%d/%d] %s (%s%s) 수집 중...",
                    idx + 1, len(sgg_df), sgg_name, sido_cd, sgg["sgg_cd"],
                    )
                up_cd = f"{sido_cd}{sgg['sgg_cd']}00000"
                sgg_rows = self._fetch_all(params={"up_cd": up_cd, "pg_yn": "N"})
                all_rows.extend(sgg_rows)
                logger.info(
                    "  [%d/%d] %s → %d건",
                    idx + 1, len(sgg_df), sgg_name, len(sgg_rows),
                    )
                time.sleep(0.3)  # API 부하 방지

            rows = all_rows

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # 읍면동 레벨 필터 (리 코드가 00인 것)
        df = df[
            (df["region_cd"].str[5:8] != "000") &  # 읍면동이 000이 아닌 것
            (df["region_cd"].str[8:] == "00")       # 리 코드가 00인 것
            ].copy()

        df["sido_cd"] = df["region_cd"].str[:2]
        df["sgg_cd"] = df["region_cd"].str[2:5]
        df["dong_cd"] = df["region_cd"].str[5:8]
        df["dong_nm"] = df["locatadd_nm"].apply(lambda x: x.split()[-1] if " " in x else x)
        df["full_nm"] = df["locatadd_nm"]
        df["flag"] = df.get("flag", "Y")  # 존재 여부

        logger.info("읍면동 수집 완료: %d개", len(df))
        return df[["region_cd", "sido_cd", "sgg_cd", "dong_cd", "dong_nm", "full_nm", "flag"]].reset_index(drop=True)

    def collect(
            self,
            sido_cd: str = "11",
            sgg_cd: str = None,
            locatadd_nm: str = None,
    ) -> pd.DataFrame:
        """
        통합 수집 메서드.

        Args:
            sido_cd: 시도코드 (기본 "11" = 서울)
            sgg_cd: 시군구코드 (선택)
            locatadd_nm: 지역명 검색 (선택, 예: "강남구")

        Returns:
            법정동코드 DataFrame
        """
        if locatadd_nm:
            logger.info("지역명 검색: '%s'", locatadd_nm)
            rows = self._fetch_all(params={
                "locatadd_nm": locatadd_nm,
                "flag": "Y",
            })
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame()

        return self.collect_dong(sido_cd=sido_cd, sgg_cd=sgg_cd)

    def save_all(
            self,
            output_dir: str = "data/00_region_codes",
            sido_cd: str = "11",
    ) -> dict[str, str]:
        """
        시도/시군구/읍면동 코드를 CSV로 저장.

        Args:
            output_dir: 저장 디렉토리
            sido_cd: 시도코드 (기본 서울)

        Returns:
            {"sido": path, "sgg": path, "dong": path}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 시도
        sido_df = self.collect_sido()
        if not sido_df.empty:
            p = out / "sido_codes.csv"
            sido_df.to_csv(p, index=False, encoding="utf-8-sig")
            paths["sido"] = str(p)
            logger.info("저장: %s (%d건)", p, len(sido_df))

        # 시군구
        sgg_df = self.collect_sgg(sido_cd)
        if not sgg_df.empty:
            p = out / f"sgg_codes_{sido_cd}.csv"
            sgg_df.to_csv(p, index=False, encoding="utf-8-sig")
            paths["sgg"] = str(p)
            logger.info("저장: %s (%d건)", p, len(sgg_df))

        # 읍면동
        dong_df = self.collect_dong(sido_cd)
        if not dong_df.empty:
            p = out / f"dong_codes_{sido_cd}.csv"
            dong_df.to_csv(p, index=False, encoding="utf-8-sig")
            paths["dong"] = str(p)
            logger.info("저장: %s (%d건)", p, len(dong_df))

        return paths

    # ================================================================
    # 내부 메서드
    # ================================================================

    def _fetch_page(self, params: dict, page: int = 1, num_rows: int = 1000) -> tuple[list[dict], int]:
        """
        API 1페이지 호출.

        Returns:
            (rows: list[dict], total_count: int)
        """
        request_params = {
            "serviceKey": self._api_key,
            "pageNo": str(page),
            "numOfRows": str(num_rows),
            "type": "xml",      # XML이 더 안정적
            **params,
        }

        try:
            resp = requests.get(self.BASE_URL, params=request_params, timeout=30)
            resp.raise_for_status()

            return self._parse_xml(resp.text)

        except requests.RequestException as e:
            logger.error("API 호출 실패 (page=%d): %s", page, e)
            return [], 0

    def _fetch_all(self, params: dict, num_rows: int = 1000) -> list[dict]:
        """모든 페이지를 수집하여 합침."""
        all_rows = []
        page = 1

        while True:
            rows, total = self._fetch_page(params, page=page, num_rows=num_rows)

            if not rows:
                break

            all_rows.extend(rows)

            if total > num_rows:
                logger.info("    페이지 %d/%d 수집 (%d건 누적)",
                            page, (total + num_rows - 1) // num_rows, len(all_rows))

            if len(all_rows) >= total or len(rows) < num_rows:
                break

            page += 1
            time.sleep(0.2)  # API 부하 방지

        return all_rows

    def _parse_xml(self, xml_text: str) -> tuple[list[dict], int]:
        """
        XML 응답 파싱.

        응답 구조:
            <StanReginCd>
              <head>
                <totalCount>100</totalCount>
                <numOfRows>1000</numOfRows>
                <pageNo>1</pageNo>
              </head>
              <row>
                <region_cd>1168010100</region_cd>
                <sido_cd>11</sido_cd>
                <sgg_cd>680</sgg_cd>
                <umd_cd>101</umd_cd>
                <ri_cd>00</ri_cd>
                <locatjumin_cd>1168010100</locatjumin_cd>
                <locatjijuk_cd>1168010100</locatjijuk_cd>
                <locatadd_nm>서울특별시 강남구 역삼동</locatadd_nm>
                <locat_order>10</locat_order>
                <locat_rm></locat_rm>
                <locathigh_cd>1168000000</locathigh_cd>
                <locallow_nm>역삼동</locallow_nm>
                <adpt_de>19880423</adpt_de>
              </row>
              ...
            </StanReginCd>
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error("XML 파싱 실패: %s", e)
            # 에러 응답 확인
            if "SERVICE_KEY_IS_NOT_REGISTERED_ERROR" in xml_text:
                logger.error("❌ 서비스키가 등록되지 않았습니다. 공공데이터포털에서 활용신청을 확인하세요.")
            elif "INVALID_REQUEST_PARAMETER_ERROR" in xml_text:
                logger.error("❌ 요청 파라미터가 잘못되었습니다.")
            return [], 0

        # totalCount
        total = 0
        head = root.find(".//head")
        if head is not None:
            tc = head.find("totalCount")
            if tc is not None and tc.text:
                total = int(tc.text)

        # rows
        rows = []
        for row_elem in root.findall(".//row"):
            row_dict = {}
            for child in row_elem:
                row_dict[child.tag] = child.text or ""
            rows.append(row_dict)

        return rows, total