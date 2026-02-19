"""
scripts/run_feature.py
===========================
피처 엔지니어링 실행 스크립트.

실행: python scripts/run_feature.py
      python scripts/run_feature.py --input data/03_processed/labeled.csv

[역할]
  DB(labeled_stores) -> 피처 변환 -> 05_model_input/ (train/val/test.npy) + DB(feature_sets)
  + models/artifacts/ (scaler, encoder 저장)
"""

import sys
import uuid
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from config.feature_config import FEATURE_CONFIG
from src.preprocessing.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.features.store import FeatureStore
from src.utils.logger import setup_logging, get_logger
from src.utils.timer import timer
from src.utils import io

logger = get_logger(__name__)


@timer("피처 엔지니어링")
def main():
    parser = argparse.ArgumentParser(description="피처 엔지니어링")
    parser.add_argument(
        "--input", type=str, default=None,
        help="입력 CSV 경로 (미지정 시 DB에서 로드)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="검증 데이터 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="테스트 데이터 비율 (기본: 0.1)",
    )
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    pipeline_run_id = str(uuid.uuid4())

    # ── 1. 데이터 로드 (DB 우선, CSV fallback) ──
    if args.input and Path(args.input).exists():
        df = io.load_csv(args.input)
        logger.info("CSV에서 로드: %s (%d행 × %d열)", args.input, *df.shape)
    else:
        try:
            from src.database.repository import LabeledStoreRepository
            labeled_repo = LabeledStoreRepository()
            df = labeled_repo.to_dataframe()
            if not df.empty:
                logger.info("DB(labeled_stores)에서 로드: %d행 × %d열", *df.shape)
            else:
                raise ValueError("labeled_stores 테이블이 비어있습니다")
        except Exception as e:
            logger.error("데이터 로드 실패: %s", e)
            logger.error("먼저 make train 또는 make collect를 실행하세요.")
            sys.exit(1)

    # ── 1-1. cleaner 통과 (API 원본 컬럼 매핑 + 누락 피처 채우기) ──
    cleaner = DataCleaner()
    df = cleaner.clean(df)

    # ── 2. 피처 변환 (fit + transform) ──
    builder = FeatureBuilder()
    X, y = builder.fit_transform(df)

    logger.info("피처 변환 완료: X=%s, y=%s", X.shape, y.shape)
    logger.info("피처 컬럼 수: %d개", X.shape[1])

    # ── 2-1. 04_features/에 중간 결과 저장 (디버깅/EDA용) ──
    import pandas as pd
    feature_cols = builder._feature_columns
    target_cols = [t for t in FEATURE_CONFIG.targets if t in df.columns]

    df_features = pd.DataFrame(X, columns=feature_cols)
    df_targets = pd.DataFrame(y, columns=target_cols)
    df_combined = pd.concat([df_features, df_targets], axis=1)
    io.save_csv(df_combined, f"{settings.DATA_FEATURES}/features.csv")

    feature_list_path = f"{settings.DATA_FEATURES}/feature_columns.txt"
    Path(feature_list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(feature_list_path, "w", encoding="utf-8") as f:
        f.write(f"# 피처 컬럼 목록 ({len(feature_cols)}개)\n")
        f.write(f"# 생성 시각: {pd.Timestamp.now()}\n\n")
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {col}\n")

    # ── 3. Train/Val/Test 분할 & 저장 (파일 + DB) ──
    store = FeatureStore()
    sizes = store.save_splits_to_db(
        X, y,
        feature_columns=feature_cols,
        target_columns=target_cols,
        pipeline_run_id=pipeline_run_id,
        scaler_params=builder.get_scaler_params(),
        encoder_classes=builder.get_encoder_classes(),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    logger.info("데이터 분할: train=%d, val=%d, test=%d",
                sizes["train"], sizes["val"], sizes["test"])

    # ── 4. 전처리기(scaler, encoder) 저장 ──
    builder.save_artifacts(settings.MODEL_ARTIFACTS)

    # ── 5. 요약 출력 ──
    logger.info("━━━ 피처 엔지니어링 완료 ━━━")
    logger.info("  피처 수: %d개", X.shape[1])
    logger.info("  출력 데이터: %s/", settings.DATA_MODEL_INPUT)
    logger.info("  DB 저장: feature_sets 테이블 (run=%s)", pipeline_run_id[:8])
    logger.info("  전처리기: %s/", settings.MODEL_ARTIFACTS)
    logger.info("  다음 단계: make train")


if __name__ == "__main__":
    main()
