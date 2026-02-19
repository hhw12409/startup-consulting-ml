"""
📁 src/data_collection/nts_client.py
=====================================
국세청 사업자등록 상태 조회 API 클라이언트.

[역할] 사업자등록번호로 계속/휴업/폐업 상태를 조회합니다.
       이 결과가 ML 모델의 '생존 라벨'이 됩니다.

[개선] 연속 실패 시 조기 중단하여 불필요한 API 호출과 에러 로그를 방지합니다.
"""

import time
import requests
import pandas as pd

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NtsClient:
    """
    국세청 사업자등록 상태 조회.

    사용법:
        client = NtsClient()
        df = client.check_status(["1234567890", "9876543210"])
        # b_stt_cd: "01"=계속, "02"=휴업, "03"=폐업
    """

    URL = "https://api.odcloud.kr/api/nts-businessman/v1/status"
    MAX_CONSECUTIVE_FAILURES = 3  # 연속 N번 실패하면 중단

    def __init__(self):
        self._key = get_settings().NTS_API_KEY

    def check_status(self, biz_numbers: list[str]) -> pd.DataFrame:
        """
        사업자 상태 일괄 조회 (자동 100건 배치).

        Returns:
            컬럼: b_no(사업자번호), b_stt(상태명), b_stt_cd(상태코드), end_dt(폐업일)
        """
        results = []
        total_batches = (len(biz_numbers) + 99) // 100
        consecutive_failures = 0

        logger.info("사업자 상태 조회 시작: %d건 (%d배치)", len(biz_numbers), total_batches)

        for i in range(0, len(biz_numbers), 100):
            batch_num = i // 100 + 1
            batch = biz_numbers[i:i + 100]

            try:
                resp = requests.post(
                    self.URL,
                    params={"serviceKey": self._key},
                    json={"b_no": batch},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json().get("data", [])
                results.extend(data)
                consecutive_failures = 0  # 성공하면 리셋
                time.sleep(0.5)

                # 진행률 (50배치마다)
                if batch_num % 50 == 0:
                    logger.info("  [%d/%d] 배치 진행 중... (%d건 수집)", batch_num, total_batches, len(results))

            except requests.RequestException as e:
                consecutive_failures += 1

                if consecutive_failures == 1:
                    logger.warning("국세청 API 실패 (batch %d): %s", batch_num, e)
                elif consecutive_failures == self.MAX_CONSECUTIVE_FAILURES:
                    remaining = total_batches - batch_num
                    logger.error(
                        "⚠️ 연속 %d회 실패 → 나머지 %d배치 스킵 (일일 한도 초과 가능성)",
                        self.MAX_CONSECUTIVE_FAILURES, remaining,
                    )
                    break

        logger.info(
            "사업자 상태 조회 완료: %d건 요청 → %d건 응답 (%.1f%%)",
            len(biz_numbers), len(results),
            len(results) / len(biz_numbers) * 100 if biz_numbers else 0,
        )
        return pd.DataFrame(results) if results else pd.DataFrame()