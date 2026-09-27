"""
Training module for Business Entity Resolution.
Accepts pre-computed numpy arrays for memory efficiency.
"""

import os
import sys
import time
import json
import gc
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb

from src.config import Config
from src.features import FEATURE_NAMES
from src.evaluation import evaluate_macro_f05
from src.singleton_detection import apply_decision_policy
from src.threshold_optimization import optimize_thresholds

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       val_pairs_by_s1: Dict[str, List[dict]],
                       val_gt: Dict[str, Set[str]], 
                       model_type: str = 'lightgbm',
                       reports_dir: str = 'reports') -> Tuple[Any, dict]:
    """
    Trains specified model on pre-computed arrays and evaluates entity-level Macro F_0.5.
    """
    os.makedirs(reports_dir, exist_ok=True)
    print(f"\n--- Training {model_type.upper()} Model ---")
    t0 = time.time()

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    print(f"Train matrix: {X_train.shape} (Positives: {pos_count:,}, Negatives: {neg_count:,})")

    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED, class_weight='balanced')
        model.fit(X_train, y_train)
    elif model_type == 'hist_gradient_boosting':
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1,
            random_state=Config.RANDOM_SEED
        )
        model.fit(X_train, y_train)
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(**Config.LGBM_PARAMS)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_time = time.time() - t0
    print(f"Model trained in {train_time:.1f}s")

    # Evaluate on Validation
    print("Scoring validation candidates...")
    val_scored_candidates = {}
    for s1_id, pairs in val_pairs_by_s1.items():
        if not pairs:
            val_scored_candidates[s1_id] = []
            continue
        X_val = np.array([p['features'] for p in pairs], dtype=np.float32)
        probs = model.predict_proba(X_val)[:, 1]
        cands_with_prob = [(pairs[i]['target_id'], float(probs[i])) for i in range(len(pairs))]
        cands_with_prob.sort(key=lambda x: x[1], reverse=True)
        val_scored_candidates[s1_id] = cands_with_prob

    # Also add val S1 entities that had no pairs (pure singletons)
    for s1_id in val_gt:
        if s1_id not in val_scored_candidates:
            val_scored_candidates[s1_id] = []

    # Optimize decision thresholds
    print("Optimizing thresholds for Macro F_0.5...")
    best_config = optimize_thresholds(val_scored_candidates, val_gt)
    print(f"\nResults for {model_type.upper()}:")
    print(f"  Macro F_0.5:        {best_config['macro_f05']:.4f}")
    print(f"  Macro Precision:    {best_config['macro_precision']:.4f}")
    print(f"  Macro Recall:       {best_config['macro_recall']:.4f}")
    print(f"  Singleton Accuracy: {best_config['singleton_accuracy']:.4f}")
    print(f"  Decision Threshold: {best_config['decision_threshold']}")
    print(f"  Singleton Thresh:   {best_config['singleton_threshold']}")

    # Feature importances for tree models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'feature': FEATURE_NAMES,
            'importance': importances
        }).sort_values('importance', ascending=False)
        fi_path = os.path.join(reports_dir, f'feature_importance_{model_type}.csv')
        fi_df.to_csv(fi_path, index=False)
        print(f"Feature importances saved to {fi_path}")
        print("Top 10 features:")
        for _, row in fi_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, best_config
