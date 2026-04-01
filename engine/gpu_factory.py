# -*- coding: utf-8 -*-
"""GPU 가속 LightGBM 팩토리.

WhyLabConfig의 GPU 설정에 따라 LGBMRegressor/LGBMClassifier를
GPU 또는 CPU 모드로 생성합니다.

RTX 4070 SUPER 12GB 최적화.
"""

from __future__ import annotations

import logging
from typing import Optional

from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)

_GPU_AVAILABLE: Optional[bool] = None


def _check_gpu() -> bool:
    """LightGBM GPU 사용 가능 여부를 한 번만 확인합니다."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        from lightgbm import LGBMRegressor
        test_model = LGBMRegressor(
            n_estimators=2, device="gpu", gpu_device_id=0, verbose=-1,
        )
        import numpy as np
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        test_model.fit(X, y)
        _GPU_AVAILABLE = True
        logger.info("✅ LightGBM GPU 가속 활성화 (RTX 4070 SUPER)")
    except Exception as e:
        _GPU_AVAILABLE = False
        logger.warning("⚠️ LightGBM GPU 사용 불가, CPU 폴백: %s", e)

    return _GPU_AVAILABLE


def create_lgbm_regressor(
    config: WhyLabConfig,
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    num_leaves: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lightweight: bool = False,
):
    """GPU 가속 LGBMRegressor를 생성합니다.

    Args:
        config: WhyLab 설정. GPU 플래그 참조.
        n_estimators: 오버라이드. None이면 config 사용.
        max_depth: 오버라이드.
        num_leaves: 오버라이드.
        learning_rate: 오버라이드.
        lightweight: True면 경량 설정 (반증/CV용).

    Returns:
        LGBMRegressor 인스턴스 (GPU 또는 CPU).
    """
    from lightgbm import LGBMRegressor

    cfg = config.dml

    if lightweight:
        n_est = n_estimators or 100
        md = max_depth or 3
        nl = num_leaves or 15
        lr = learning_rate or 0.1
    else:
        n_est = n_estimators or cfg.lgbm_n_estimators
        md = max_depth or cfg.lgbm_max_depth
        nl = num_leaves or cfg.lgbm_num_leaves
        lr = learning_rate or cfg.lgbm_learning_rate

    params = {
        "n_estimators": n_est,
        "max_depth": md,
        "num_leaves": nl,
        "learning_rate": lr,
        "verbose": cfg.lgbm_verbose,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # GPU 가속 적용
    if cfg.use_gpu and _check_gpu():
        params["device"] = "gpu"
        params["gpu_device_id"] = cfg.gpu_device_id

    return LGBMRegressor(**params)
