# -*- coding: utf-8 -*-
"""Public API for dt_gate."""

# الإصدار (طابق آخر tag لديك)
__version__ = "0.1.1"

# إعادة تصدير الدوال/الكلاسات الأساسية
from .simulate import run_all, save_summary_and_csv
from .ekf import EKFScalar
from .tube import tube_radius_from_nis
from .risk import RiskMLP, isotonic_calibrate, smooth_samples
from .queueing import beta_cantelli, bootstrap_ci
from .optimizer import raso_solve
from .telemetry import init_classes, attack, plant_epoch
from . import config as config  # للوصول إلى الثوابت بسهولة: dt_gate.config.EPOCHS ...

__all__ = [
    "__version__",
    "run_all",
    "save_summary_and_csv",
    "EKFScalar",
    "tube_radius_from_nis",
    "RiskMLP",
    "isotonic_calibrate",
    "smooth_samples",
    "beta_cantelli",
    "bootstrap_ci",
    "raso_solve",
    "init_classes",
    "attack",
    "plant_epoch",
    "config",
]
