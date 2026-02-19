"""
📁 tests/unit/test_models.py
===============================
모델 단위 테스트.

실행: pytest tests/unit/test_models.py -v

[테스트 원칙]
  - 각 테스트는 독립적 (다른 테스트 결과에 의존하지 않음)
  - conftest.py의 픽스처를 활용하여 중복 제거
  - 테스트명으로 '무엇을 검증하는지' 명확히 표현
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestXGBoostModel:
    """XGBoost 모델 테스트"""

    def test_predict_returns_all_tasks(self, trained_xgboost, sample_xy):
        """predict()가 4개 태스크 키를 모두 반환하는지"""
        X, _ = sample_xy
        preds = trained_xgboost.predict(X)

        assert "survival" in preds
        assert "revenue" in preds
        assert "risk" in preds
        assert "break_even" in preds

    def test_predict_shapes(self, trained_xgboost, sample_xy):
        """예측값의 shape이 올바른지"""
        X, _ = sample_xy
        n = X.shape[0]
        preds = trained_xgboost.predict(X)

        assert preds["survival"].shape == (n, 2), "survival: [N, 2]"
        assert preds["revenue"].shape == (n, 2), "revenue: [N, 2]"
        assert preds["risk"].shape == (n, 1), "risk: [N, 1]"
        assert preds["break_even"].shape == (n, 1), "break_even: [N, 1]"

    def test_survival_probability_range(self, trained_xgboost, sample_xy):
        """생존확률이 0~1 범위인지"""
        X, _ = sample_xy
        preds = trained_xgboost.predict(X)
        assert preds["survival"].min() >= 0.0
        assert preds["survival"].max() <= 1.0

    def test_risk_score_range(self, trained_xgboost, sample_xy):
        """리스크 점수가 0~1 범위인지"""
        X, _ = sample_xy
        preds = trained_xgboost.predict(X)
        assert preds["risk"].min() >= 0.0
        assert preds["risk"].max() <= 1.0

    def test_save_and_load(self, trained_xgboost, sample_xy):
        """저장 후 로드한 모델이 동일한 예측을 반환하는지"""
        X, _ = sample_xy

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/test_model"
            trained_xgboost.save(path)

            from src.models.xgboost_model import XGBoostModel
            loaded = XGBoostModel()
            loaded.load(path)

            pred_original = trained_xgboost.predict(X)
            pred_loaded = loaded.predict(X)

            np.testing.assert_array_almost_equal(
                pred_original["survival"], pred_loaded["survival"], decimal=5,
            )

    def test_predict_before_train_raises(self):
        """학습 전 predict 호출 시 에러가 발생하는지"""
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((1, 10)))

    def test_get_info(self, trained_xgboost):
        """모델 메타정보가 올바른지"""
        info = trained_xgboost.get_info()
        assert info["name"] == "XGBoostModel"
        assert info["is_trained"] is True
        assert len(info["tasks"]) == 6

    def test_feature_importance(self, trained_xgboost):
        """피처 중요도를 반환하는지"""
        importance = trained_xgboost.get_feature_importance()
        assert len(importance) > 0
        assert "survival_1yr" in importance


class TestBaseModel:
    """BaseModel 인터페이스 테스트"""

    def test_cannot_instantiate_abstract(self):
        """추상 클래스는 직접 생성할 수 없어야 함"""
        from src.models.base import BaseModel
        with pytest.raises(TypeError):
            BaseModel()