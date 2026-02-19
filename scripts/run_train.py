"""
📁 scripts/run_train.py
========================
모델 학습 실행 스크립트.

실행: python scripts/run_train.py --model xgboost
      python scripts/run_train.py --model neural_net
"""

import os
# XGBoost + PyTorch 동시 사용 시 OpenMP 스레딩 충돌 방지
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from pipelines.train_pipeline import TrainPipeline

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="모델 학습")
    parser.add_argument("--model", choices=["xgboost", "neural_net"], default="xgboost")
    parser.add_argument("--data", type=str, default=None, help="학습 데이터 CSV 경로")
    args = parser.parse_args()

    setup_logging()

    # 모델 선택 (Strategy 패턴)
    if args.model == "xgboost":
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel()
    else:
        from src.models.neural_net import NeuralNetModel
        model = NeuralNetModel()

    # 파이프라인 실행
    pipeline = TrainPipeline(model=model)
    result = pipeline.run(data_path=args.data)

    logger.info("학습 결과: %s", result["metrics"])


if __name__ == "__main__":
    main()