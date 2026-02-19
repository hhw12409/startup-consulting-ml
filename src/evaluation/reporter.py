"""
📁 src/evaluation/reporter.py
===============================
평가 리포트 생성.

[패턴] Template Method — 리포트 형식을 정의하고 내용만 교체
[역할] 메트릭 → 사람이 읽기 쉬운 텍스트/JSON 리포트로 변환
"""

import json
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReporter:
    """
    평가 리포트 생성기.

    사용법:
        reporter = EvaluationReporter()
        reporter.generate(metrics, model_info, save_path="logs/eval_report.json")
    """

    def generate(
            self,
            metrics: dict[str, float],
            model_info: dict,
            save_path: str = None,
    ) -> dict:
        """
        평가 리포트를 생성하고 선택적으로 파일로 저장합니다.

        Returns:
            리포트 딕셔너리
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": model_info,
            "metrics": metrics,
            "summary": self._summarize(metrics),
        }

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info("리포트 저장: %s", save_path)

        return report

    def _summarize(self, metrics: dict) -> str:
        """메트릭을 한 줄 요약으로 변환"""
        acc_1yr = metrics.get("survival_1yr_acc", 0)
        auc_1yr = metrics.get("survival_1yr_auc", 0)
        rev_mae = metrics.get("revenue_mae", 0)

        return (
            f"1년생존 정확도={acc_1yr:.1%}, AUC={auc_1yr:.3f}, "
            f"매출MAE={rev_mae:,.0f}원"
        )