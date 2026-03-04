"""
Classification models for imbalanced entrance detection.

- Weighted Random Forest (WRF): cost-sensitive, heavier penalty on minority
- Balanced Random Forest (BRF): under-samples majority per tree
- SmoteBoost: SMOTE resampling + AdaBoost ensemble
- Random Forest (RF): standard baseline
"""
from typing import Dict

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from config import ModelConfig


def build_wrf(cfg: ModelConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.wrf_n_trees,
        max_depth=cfg.wrf_max_depth,
        class_weight={0: 1, 1: cfg.wrf_class_weight_ratio},
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def build_brf(cfg: ModelConfig) -> BalancedRandomForestClassifier:
    return BalancedRandomForestClassifier(
        n_estimators=cfg.brf_n_trees,
        max_depth=cfg.brf_max_depth,
        sampling_strategy="auto",
        replacement=True,
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def build_rf(cfg: ModelConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf_n_trees,
        max_depth=cfg.rf_max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def build_smoteboost(cfg: ModelConfig):
    """SMOTE + AdaBoost pipeline (approximation of true SmoteBoost)."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    smote = SMOTE(
        k_neighbors=min(cfg.smote_k_neighbors, 3),
        random_state=cfg.random_state,
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=cfg.smote_n_estimators,
        random_state=cfg.random_state,
        algorithm="SAMME",
    )
    return Pipeline([("smote", smote), ("adaboost", ada)])


def build_all_models(cfg: ModelConfig) -> Dict[str, object]:
    return {
        "WRF": build_wrf(cfg),
        "BRF": build_brf(cfg),
        "SmoteBoost": build_smoteboost(cfg),
        "RF": build_rf(cfg),
    }


def predict_entrance_proba(model, X: np.ndarray) -> np.ndarray:
    """Return P(positive) for each sample."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
    return model.decision_function(X)