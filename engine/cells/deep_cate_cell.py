# -*- coding: utf-8 -*-
"""DeepCATECell — TARNet / DragonNet 기반 딥러닝 CATE 추정.

신경망 기반 이질적 처치효과(CATE) 추정기:
- **TARNet** (Shalit et al., 2017): 공유 표현 → 처치/통제 분기 헤드
- **DragonNet** (Shi et al., 2019): TARNet + 성향점수 헤드 + 타겟 정규화

기존 MetaLearnerCell(S/T/X/DR/R)과 동일 인터페이스를 유지하되,
End-to-End 신경망 학습으로 비선형 CATE를 직접 추정합니다.

학술적 참조:
  - Shalit, Johansson & Sontag (2017). "Estimating individual treatment
    effect: generalization bounds and algorithms." ICML.
  - Shi, Blei & Veitch (2019). "Adapting Neural Networks for the
    Estimation of Treatment Effects." NeurIPS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch 지연 임포트 (선택적 의존성)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch 미설치. DeepCATECell은 fallback 모드로 동작합니다.")


# nn.Module 기반 클래스들은 PyTorch가 있을 때만 정의
if HAS_TORCH:
    _nn_Module = nn.Module
else:
    # Stub: PyTorch 없이도 클래스 정의 자체는 가능하도록
    _nn_Module = object

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

@dataclass
class DeepCATEConfig:
    """DeepCATECell 하이퍼파라미터."""

    architecture: str = "dragonnet"      # "tarnet" | "dragonnet"
    shared_dims: Tuple[int, ...] = (200, 100)   # 공유 표현 레이어
    head_dims: Tuple[int, ...] = (100, 64)      # 결과 헤드 레이어
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    alpha: float = 1.0                  # DragonNet 타겟 정규화 가중치
    use_gpu: bool = True
    early_stopping_patience: int = 10
    val_ratio: float = 0.2              # 검증 세트 비율


# ──────────────────────────────────────────────
# TARNet 아키텍처
# ──────────────────────────────────────────────

class TARNetModule(_nn_Module):
    """TARNet: Treatment-Agnostic Representation Network.

    구조:
        X → [공유 표현 Φ(X)] → [Y₀ 헤드] → ŷ₀
                              → [Y₁ 헤드] → ŷ₁
    """

    def __init__(
        self,
        input_dim: int,
        shared_dims: Tuple[int, ...] = (200, 100),
        head_dims: Tuple[int, ...] = (100, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        # 공유 표현 네트워크 Φ(X)
        layers = []
        prev = input_dim
        for dim in shared_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.ELU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout),
            ])
            prev = dim
        self.shared = nn.Sequential(*layers)

        # Y₀ 헤드 (통제군)
        self.head_y0 = self._build_head(prev, head_dims, dropout)
        # Y₁ 헤드 (처치군)
        self.head_y1 = self._build_head(prev, head_dims, dropout)

    @staticmethod
    def _build_head(
        input_dim: int,
        head_dims: Tuple[int, ...],
        dropout: float,
    ) -> nn.Sequential:
        """결과 예측 헤드를 생성합니다."""
        layers = []
        prev = input_dim
        for dim in head_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파.

        Returns:
            (y0_pred, y1_pred): 각 (batch,) 크기.
        """
        phi = self.shared(x)
        y0 = self.head_y0(phi).squeeze(-1)
        y1 = self.head_y1(phi).squeeze(-1)
        return y0, y1


# ──────────────────────────────────────────────
# DragonNet 아키텍처
# ──────────────────────────────────────────────

class DragonNetModule(_nn_Module):
    """DragonNet: TARNet + 성향점수 헤드.

    구조:
        X → [공유 표현 Φ(X)] → [Y₀ 헤드] → ŷ₀
                              → [Y₁ 헤드] → ŷ₁
                              → [T 헤드]  → ê(X)  (성향점수)

    성향점수 헤드가 표현 학습을 정규화하여
    처치/통제 그룹 간 균형 잡힌 표현을 유도합니다.
    """

    def __init__(
        self,
        input_dim: int,
        shared_dims: Tuple[int, ...] = (200, 100),
        head_dims: Tuple[int, ...] = (100, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        # 공유 표현 네트워크
        layers = []
        prev = input_dim
        for dim in shared_dims:
            layers.extend([
                nn.Linear(prev, dim),
                nn.ELU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(dropout),
            ])
            prev = dim
        self.shared = nn.Sequential(*layers)

        # Y₀, Y₁ 헤드
        self.head_y0 = TARNetModule._build_head(prev, head_dims, dropout)
        self.head_y1 = TARNetModule._build_head(prev, head_dims, dropout)

        # 성향점수 헤드 (DragonNet 핵심)
        self.head_t = nn.Sequential(
            nn.Linear(prev, head_dims[-1] if head_dims else 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dims[-1] if head_dims else 64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """순전파.

        Returns:
            (y0_pred, y1_pred, t_pred): 각 (batch,) 크기.
        """
        phi = self.shared(x)
        y0 = self.head_y0(phi).squeeze(-1)
        y1 = self.head_y1(phi).squeeze(-1)
        t_pred = self.head_t(phi).squeeze(-1)
        return y0, y1, t_pred


# ──────────────────────────────────────────────
# DeepCATECell 메인
# ──────────────────────────────────────────────

class DeepCATECell:
    """딥러닝 기반 CATE 추정 셀.

    MetaLearnerCell과 동일한 인터페이스:
        cell = DeepCATECell(config)
        result = cell.execute(inputs)

    또는 직접 호출:
        cell.fit(X, T, Y)
        cate = cell.predict_cate(X)
    """

    # 메타러너 인터페이스 호환
    name = "DeepCATE"

    def __init__(self, config=None, deep_config: Optional[DeepCATEConfig] = None):
        self.config = config
        self.deep_config = deep_config or DeepCATEConfig()
        self.model = None
        self.device = None
        self.scaler_x_mean = None
        self.scaler_x_std = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _get_device(self) -> torch.device:
        """가용한 디바이스를 감지합니다."""
        if self.deep_config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info("GPU 사용: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            self.logger.info("CPU 사용")
        return device

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """입력 정규화 (Z-score)."""
        if fit:
            self.scaler_x_mean = X.mean(axis=0)
            self.scaler_x_std = X.std(axis=0) + 1e-8
        return (X - self.scaler_x_mean) / self.scaler_x_std

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "DeepCATECell":
        """모델 학습.

        Args:
            X: (n, p) 공변량 행렬.
            T: (n,) 이진 처치 벡터 (0/1).
            Y: (n,) 관측 결과.

        Returns:
            self (체이닝).
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch가 필요합니다: pip install torch")

        cfg = self.deep_config
        self.device = self._get_device()

        # 소표본 자동 튜닝 (n < 2000)
        n_samples = len(X)
        if n_samples < 2000:
            self.logger.info("소표본 감지 (n=%d < 2000): 정규화 강화 모드 적용", n_samples)
            # 설정 복제 및 수정
            cfg = replace(
                cfg,
                dropout=max(cfg.dropout, 0.5),
                weight_decay=max(cfg.weight_decay, 1e-2),
                batch_size=min(cfg.batch_size, 32),
                epochs=min(cfg.epochs, 50),
                learning_rate=min(cfg.learning_rate, 5e-4),
                # 모델 용량 축소
                shared_dims=(64, 32) if cfg.shared_dims[0] > 64 else cfg.shared_dims,
                head_dims=(32, 16) if cfg.head_dims[0] > 32 else cfg.head_dims,
            )

        # 정규화
        X_norm = self._normalize(X, fit=True)
        n, p = X_norm.shape

        # T 이진화
        T_binary = (T > np.median(T)).astype(np.float32) if len(np.unique(T)) > 2 else T.astype(np.float32)

        # 검증 세트 분할
        val_size = int(n * cfg.val_ratio)
        perm = np.random.RandomState(42).permutation(n)
        idx_train, idx_val = perm[val_size:], perm[:val_size]

        # 텐서 변환
        X_t = torch.FloatTensor(X_norm).to(self.device)
        T_t = torch.FloatTensor(T_binary).to(self.device)
        Y_t = torch.FloatTensor(Y.astype(np.float32)).to(self.device)

        train_ds = TensorDataset(X_t[idx_train], T_t[idx_train], Y_t[idx_train])
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

        val_x = X_t[idx_val]
        val_t = T_t[idx_val]
        val_y = Y_t[idx_val]

        # 모델 생성
        if cfg.architecture == "dragonnet":
            self.model = DragonNetModule(
                input_dim=p,
                shared_dims=cfg.shared_dims,
                head_dims=cfg.head_dims,
                dropout=cfg.dropout,
            ).to(self.device)
        else:
            self.model = TARNetModule(
                input_dim=p,
                shared_dims=cfg.shared_dims,
                head_dims=cfg.head_dims,
                dropout=cfg.dropout,
            ).to(self.device)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.logger.info(
            "학습 시작: arch=%s, epochs=%d, lr=%.4f, params=%d",
            cfg.architecture, cfg.epochs, cfg.learning_rate,
            sum(p.numel() for p in self.model.parameters()),
        )

        for epoch in range(cfg.epochs):
            self.model.train()
            train_loss_sum = 0.0

            for batch_x, batch_t, batch_y in train_loader:
                optimizer.zero_grad()

                if cfg.architecture == "dragonnet":
                    y0_pred, y1_pred, t_pred = self.model(batch_x)
                    loss = self._dragonnet_loss(
                        y0_pred, y1_pred, t_pred,
                        batch_t, batch_y, alpha=cfg.alpha,
                    )
                else:
                    y0_pred, y1_pred = self.model(batch_x)
                    loss = self._tarnet_loss(y0_pred, y1_pred, batch_t, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item()

            # 검증
            self.model.eval()
            with torch.no_grad():
                if cfg.architecture == "dragonnet":
                    vy0, vy1, vt_pred = self.model(val_x)
                    val_loss = self._dragonnet_loss(
                        vy0, vy1, vt_pred, val_t, val_y, alpha=cfg.alpha,
                    ).item()
                else:
                    vy0, vy1 = self.model(val_x)
                    val_loss = self._tarnet_loss(vy0, vy1, val_t, val_y).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                self.logger.info(
                    "  Epoch %d/%d: train=%.4f, val=%.4f (best=%.4f)",
                    epoch + 1, cfg.epochs,
                    train_loss_sum / len(train_loader),
                    val_loss, best_val_loss,
                )

            if patience_counter >= cfg.early_stopping_patience:
                self.logger.info("  조기 중단 (epoch %d)", epoch + 1)
                break

        # 최적 가중치 복원
        if best_state:
            self.model.load_state_dict(best_state)

        self.logger.info("학습 완료: best_val_loss=%.4f", best_val_loss)
        return self

    @staticmethod
    def _tarnet_loss(
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """TARNet 관측 결과 손실.

        L = Σ [ T·(Y - ŷ₁)² + (1-T)·(Y - ŷ₀)² ]
        """
        loss_0 = ((1 - t) * (y - y0_pred) ** 2).mean()
        loss_1 = (t * (y - y1_pred) ** 2).mean()
        return loss_0 + loss_1

    @staticmethod
    def _dragonnet_loss(
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        t_pred: torch.Tensor,
        t_true: torch.Tensor,
        y_true: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """DragonNet 결합 손실.

        L = L_outcome + α · L_propensity
        L_outcome = TARNet 손실
        L_propensity = BCE(ê(X), T)
        """
        outcome_loss = DeepCATECell._tarnet_loss(y0_pred, y1_pred, t_true, y_true)
        propensity_loss = nn.functional.binary_cross_entropy(
            t_pred, t_true, reduction="mean",
        )
        return outcome_loss + alpha * propensity_loss

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """CATE τ̂(x) = ŷ₁(x) - ŷ₀(x) 예측.

        Args:
            X: (n, p) 공변량 행렬.

        Returns:
            (n,) CATE 예측값.
        """
        if self.model is None:
            raise RuntimeError("먼저 fit()을 호출하세요.")

        X_norm = self._normalize(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, DragonNetModule):
                y0, y1, _ = self.model(X_t)
            else:
                y0, y1 = self.model(X_t)

        return (y1 - y0).cpu().numpy()

    def predict_outcomes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Y₀, Y₁, CATE, 성향점수 예측.

        Returns:
            {"y0": ..., "y1": ..., "cate": ..., "propensity": ...}
        """
        if self.model is None:
            raise RuntimeError("먼저 fit()을 호출하세요.")

        X_norm = self._normalize(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, DragonNetModule):
                y0, y1, t_pred = self.model(X_t)
                propensity = t_pred.cpu().numpy()
            else:
                y0, y1 = self.model(X_t)
                propensity = np.full(len(X), 0.5)

        y0_np = y0.cpu().numpy()
        y1_np = y1.cpu().numpy()

        return {
            "y0": y0_np,
            "y1": y1_np,
            "cate": y1_np - y0_np,
            "propensity": propensity,
        }

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 셀 인터페이스.

        MetaLearnerCell과 호환되는 execute 메서드.
        """
        df = inputs.get("dataframe")
        feature_names = inputs.get("feature_names", [])
        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")

        if df is None:
            self.logger.warning("데이터프레임이 없습니다.")
            return {**inputs, "deep_cate": None}

        # 피처 추출
        features = feature_names or [
            c for c in df.columns if c not in [treatment_col, outcome_col]
        ]
        X = df[features].values.astype(np.float64)
        T = df[treatment_col].values
        Y = df[outcome_col].values

        # 학습 및 예측
        self.fit(X, T, Y)
        outcomes = self.predict_outcomes(X)

        self.logger.info(
            "DeepCATE 완료: arch=%s, ATE=%.4f, CATE std=%.4f",
            self.deep_config.architecture,
            np.mean(outcomes["cate"]),
            np.std(outcomes["cate"]),
        )

        return {
            **inputs,
            "deep_cate": {
                "architecture": self.deep_config.architecture,
                "cate": outcomes["cate"],
                "y0": outcomes["y0"],
                "y1": outcomes["y1"],
                "propensity": outcomes["propensity"],
                "ate": float(np.mean(outcomes["cate"])),
                "cate_std": float(np.std(outcomes["cate"])),
            },
        }
