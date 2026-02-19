"""
📁 scripts/run_evaluate.py
============================
저장된 모델을 로드하여 테스트 데이터로 평가합니다.

실행:
  python scripts/run_evaluate.py                     # 모든 모델 평가 (자동 탐색)
  python scripts/run_evaluate.py --model xgboost     # XGBoost만
  python scripts/run_evaluate.py --model neural_net  # PyTorch만
  python scripts/run_evaluate.py --model all         # 전체 + 비교 리포트
"""

import os
# XGBoost + PyTorch 동시 사용 시 OpenMP 스레딩 충돌 방지
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.features.store import FeatureStore
from src.evaluation.metrics import evaluate_model
from src.evaluation.reporter import EvaluationReporter
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def load_model(model_type: str, model_path: str):
    """모델 타입에 따라 적절한 클래스를 로드합니다."""
    if model_type == "xgboost":
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel()
    elif model_type == "neural_net":
        from src.models.neural_net import NeuralNetModel
        model = NeuralNetModel()
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")

    model.load(model_path)
    return model


def find_saved_models(registry_dir: str) -> list[tuple[str, str]]:
    """
    저장된 모델을 자동 탐색합니다.

    Returns:
        [(model_type, model_path), ...]
    """
    registry = Path(registry_dir)
    found = []

    # XGBoost (best_model.pkl 또는 xgboost_model.pkl)
    for name in ["best_model", "xgboost_model"]:
        if (registry / f"{name}.pkl").exists():
            found.append(("xgboost", str(registry / name)))
            break

    # PyTorch (neural_net_model.pt 또는 best_model.pt)
    for name in ["neural_net_model", "best_model"]:
        if (registry / f"{name}.pt").exists():
            found.append(("neural_net", str(registry / name)))
            break

    return found


def main():
    parser = argparse.ArgumentParser(description="모델 평가")
    parser.add_argument(
        "--model",
        choices=["xgboost", "neural_net", "all"],
        default="all",
        help="평가할 모델 (기본: all)",
    )
    args = parser.parse_args()

    setup_logging()
    s = get_settings()

    # ── 1. 테스트 데이터 로드 ──
    store = FeatureStore()
    try:
        X_test, y_test = store.load("test")
    except FileNotFoundError:
        logger.error("테스트 데이터 없음 (make train 먼저 실행하세요)")
        return

    logger.info("테스트 데이터: %d행 × %d열", *X_test.shape)

    # ── 2. 모델 로드 & 평가 ──
    all_metrics = {}
    reporter = EvaluationReporter()

    if args.model == "all":
        # 저장된 모델 자동 탐색
        saved = find_saved_models(s.MODEL_REGISTRY)
        if not saved:
            logger.error("저장된 모델 없음: %s (make train 먼저 실행하세요)", s.MODEL_REGISTRY)
            return
        logger.info("발견된 모델: %s", [m[0] for m in saved])
    else:
        # 특정 모델만
        model_path = f"{s.MODEL_REGISTRY}/best_model"
        saved = [(args.model, model_path)]

    for model_type, model_path in saved:
        logger.info("")
        logger.info("━" * 50)
        logger.info("📊 %s 평가 시작", model_type.upper())
        logger.info("━" * 50)

        try:
            model = load_model(model_type, model_path)
            metrics = evaluate_model(model, X_test, y_test)
            all_metrics[model_type] = metrics

            # 개별 리포트 저장
            reporter.generate(
                metrics, model.get_info(),
                save_path=f"{s.LOG_DIR}/eval_{model_type}.json",
            )
        except Exception as e:
            logger.error("%s 평가 실패: %s", model_type, e)

    # ── 3. 비교 리포트 (2개 이상 모델일 때) ──
    if len(all_metrics) >= 2:
        logger.info("")
        logger.info("━" * 50)
        logger.info("📊 모델 비교")
        logger.info("━" * 50)

        # 공통 메트릭으로 비교
        all_keys = set()
        for m in all_metrics.values():
            all_keys.update(m.keys())

        for key in sorted(all_keys):
            values = {}
            for model_name, metrics in all_metrics.items():
                if key in metrics:
                    values[model_name] = metrics[key]

            if len(values) >= 2:
                best = max(values, key=values.get) if "loss" not in key and "mae" not in key else min(values, key=values.get)
                comparison = " | ".join(f"{n}: {v:.4f}" for n, v in values.items())
                marker = " ← best" if len(values) > 1 else ""
                logger.info("  %-25s %s  [%s%s]", key, comparison, best, marker)

        # 비교 결과 저장
        import json
        compare_path = f"{s.LOG_DIR}/eval_comparison.json"
        with open(compare_path, "w") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        logger.info("비교 리포트 저장: %s", compare_path)

    logger.info("")
    logger.info("✅ 평가 완료")


if __name__ == "__main__":
    main()