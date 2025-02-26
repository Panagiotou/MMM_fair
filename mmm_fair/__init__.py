# mmm_fair/__init__.py
from .mmm_fair import MMM_Fair
from .mmm_fair_gb import MMM_Fair_GradientBoostedClassifier
from .data_process import data_uci
__all__ = ["MMM_Fair","data_uci","MMM_Fair_GradientBoostedClassifier"]