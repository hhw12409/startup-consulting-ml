"""
📁 src/models/neural_net.py
=============================
PyTorch 기반 Multi-Task 딥러닝 모델.

[패턴] Strategy — BaseModel 인터페이스를 구현
[아키텍처]
  Shared Backbone (FC → BN → ReLU → Dropout)
    ├── survival_head → Sigmoid (0~1)
    ├── revenue_head  → Linear (연속값)
    ├── risk_head     → Sigmoid (0~1)
    └── break_even_head → ReLU (양수)

[권장] 데이터 10만건 이상일 때 XGBoost보다 유리해지기 시작합니다.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel
from config.settings import get_settings
from config.model_config import NEURAL_NET_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ================================================================
# PyTorch 내부 네트워크 정의
# ================================================================
class _Net(nn.Module):
    """내부 PyTorch 모델 (외부에서 직접 사용하지 않음)"""

    def __init__(self, input_dim: int, cfg=NEURAL_NET_CONFIG):
        super().__init__()

        # Shared Backbone
        layers = []
        prev = input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(prev, h))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout_rate))
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Task-specific Heads
        last = cfg.hidden_dims[-1]
        self.survival_head  = nn.Sequential(nn.Linear(last, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid())
        self.revenue_head   = nn.Sequential(nn.Linear(last, 32), nn.ReLU(), nn.Linear(32, 2))
        self.risk_head      = nn.Sequential(nn.Linear(last, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.break_even_head = nn.Sequential(nn.Linear(last, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "survival":  self.survival_head(feat),
            "revenue":   self.revenue_head(feat),
            "risk":      self.risk_head(feat),
            "break_even": self.break_even_head(feat),
        }


# ================================================================
# BaseModel 구현체
# ================================================================
class NeuralNetModel(BaseModel):
    """
    PyTorch 딥러닝 모델.

    사용법:
        model = NeuralNetModel(input_dim=25)
        model.train(X_train, y_train, X_val, y_val)
        preds = model.predict(X_test)
    """

    def __init__(self, input_dim: int = 0, cfg=NEURAL_NET_CONFIG):
        self._input_dim = input_dim
        self._cfg = cfg
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._net: Optional[_Net] = None
        self._is_trained = False

        if input_dim > 0:
            self._net = _Net(input_dim, cfg).to(self._device)
            total = sum(p.numel() for p in self._net.parameters())
            logger.info("NeuralNet 생성: input=%d, params=%s, device=%s", input_dim, f"{total:,}", self._device)

    @property
    def name(self) -> str:
        return "NeuralNetModel"

    def train(self, X_train, y_train, X_val=None, y_val=None):
        settings = get_settings()
        cfg = self._cfg

        # 입력 차원 자동 감지
        if self._net is None:
            self._input_dim = X_train.shape[1]
            self._net = _Net(self._input_dim, cfg).to(self._device)

        # DataLoader
        train_loader = self._loader(X_train, y_train, settings.BATCH_SIZE, shuffle=True)
        val_loader = self._loader(X_val, y_val, settings.BATCH_SIZE) if X_val is not None else None

        # 옵티마이저
        optimizer = torch.optim.Adam(self._net.parameters(), lr=settings.LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        bce, mse = nn.BCELoss(), nn.MSELoss()

        # 학습 루프
        history = {"train_loss": [], "val_loss": []}
        best_val, patience_cnt, best_state = float("inf"), 0, None

        for epoch in range(settings.MAX_EPOCHS):
            # ---- Train ----
            self._net.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                out = self._net(xb)
                loss = self._loss(out, yb, bce, mse)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_train)

            # ---- Validate ----
            if val_loader:
                avg_val = self._validate(val_loader, bce, mse)
                history["val_loss"].append(avg_val)
                scheduler.step(avg_val)

                if avg_val < best_val:
                    best_val, patience_cnt = avg_val, 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_cnt += 1

                if epoch % 20 == 0:
                    logger.info("[Epoch %3d] train=%.4f val=%.4f lr=%.6f",
                                epoch, avg_train, avg_val, optimizer.param_groups[0]["lr"])

                if patience_cnt >= settings.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping @ epoch %d", epoch)
                    break

        if best_state:
            self._net.load_state_dict(best_state)

        self._is_trained = True
        logger.info("학습 완료. Best val_loss: %.4f", best_val)
        return history

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        self._net.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self._device)
            out = self._net(t)
        return {k: v.cpu().numpy() for k, v in out.items()}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self._net.state_dict(), "input_dim": self._input_dim}, f"{path}.pt")
        logger.info("모델 저장: %s.pt", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(f"{path}.pt", map_location=self._device)
        self._input_dim = ckpt["input_dim"]
        self._net = _Net(self._input_dim, self._cfg).to(self._device)
        self._net.load_state_dict(ckpt["state"])
        self._is_trained = True
        logger.info("모델 로드: %s.pt", path)

    # ---- Private ----

    def _loader(self, X, y, bs, shuffle=False):
        return DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)), batch_size=bs, shuffle=shuffle)

    def _loss(self, out, yb, bce, mse):
        c = self._cfg
        return (
                c.loss_weight_survival  * bce(out["survival"], yb[:, 0:2]) +
                c.loss_weight_revenue   * mse(out["revenue"],  yb[:, 2:4]) +
                c.loss_weight_risk      * bce(out["risk"],     yb[:, 4:5]) +
                c.loss_weight_break_even * mse(out["break_even"], yb[:, 5:6])
        )

    @torch.no_grad()
    def _validate(self, loader, bce, mse):
        self._net.eval()
        total = sum(self._loss(self._net(xb.to(self._device)), yb.to(self._device), bce, mse).item() for xb, yb in loader)
        return total / len(loader)

    def get_info(self):
        params = sum(p.numel() for p in self._net.parameters()) if self._net else 0
        return {"name": self.name, "type": "pytorch", "input_dim": self._input_dim,
                "params": f"{params:,}", "device": self._device, "is_trained": self._is_trained}