"""
📁 src/data_collection/public_data_client.py
=============================================
공공데이터포털 API 클라이언트.

[패턴] Adapter — 외부 API의 복잡한 응답을 내부에서 쓰기 쉬운 DataFrame으로 변환
[역할] 소상공인진흥공단 상가(상권)정보 API를 호출합니다.

[수정사항]
  divId: adongCd(행정동, 8자리) → ldongCd(법정동, 10자리)
  → RegionCodeCollector로 수집한 법정동코드를 그대로 사용 가능
"""

import time
import requests
import pandas as pd

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PublicDataClient:
    """
    공공데이터포털 API 클라이언트.

    사용법:
        client = PublicDataClient()
        df = client.get_stores_by_dong("1168010100")  # 강남구 역삼동 (법정동 10자리)
    """

    BASE_URL = "http://apis.data.go.kr/B553077/api/open/sdsc2"
    REQUEST_INTERVAL = 0.5  # API 호출 간격 (초)

    def __init__(self):
        self._key = get_settings().PUBLIC_DATA_SERVICE_KEY
        self._session = requests.Session()
        self._last_call = 0.0

    def get_stores_by_dong(
            self, dong_code: str, page: int = 1, size: int = 1000
    ) -> pd.DataFrame:
        """
        법정동 코드로 상가업소 조회.

        Args:
            dong_code: 법정동 코드 (10자리, 예: "1168010100")
            page: 페이지 번호
            size: 페이지당 건수

        Returns:
            상가업소 DataFrame (상호명, 업종, 주소, 경위도 등)
        """
        self._wait()

        resp = self._session.get(
            f"{self.BASE_URL}/storeListInDong",
            params={
                "serviceKey": self._key,
                "divId": "adongCd",
                "key": dong_code,
                "pageNo": page,
                "numOfRows": size,
                "type": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()

        items = resp.json().get("body", {}).get("items", [])
        if not items:
            logger.warning("데이터 없음: dong=%s", dong_code)
            return pd.DataFrame()

        df = pd.DataFrame(items)
        logger.info("조회 완료: dong=%s → %d건", dong_code, len(df))
        return df

    def get_stores_by_radius(
            self, lat: float, lng: float, radius: int = 500
    ) -> pd.DataFrame:
        """
        반경(m) 내 상가업소 조회.

        Args:
            lat: 위도
            lng: 경도
            radius: 반경 (미터)
        """
        self._wait()

        resp = self._session.get(
            f"{self.BASE_URL}/storeListInRadius",
            params={
                "serviceKey": self._key,
                "radius": radius,
                "cx": lng,
                "cy": lat,
                "numOfRows": 1000,
                "type": "json",
            },
            timeout=30,
        )
        resp.raise_for_status()

        items = resp.json().get("body", {}).get("items", [])
        return pd.DataFrame(items) if items else pd.DataFrame()

    def _wait(self):
        """Rate Limit 준수"""
        elapsed = time.time() - self._last_call
        if elapsed < self.REQUEST_INTERVAL:
            time.sleep(self.REQUEST_INTERVAL - elapsed)
        self._last_call = time.time()