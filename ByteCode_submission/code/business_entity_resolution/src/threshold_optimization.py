"""
Threshold optimization module for Macro F_0.5.
Performs 2D grid search over decision and singleton thresholds.
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from src.evaluation import evaluate_macro_f05
from src.singleton_detection import apply_decision_policy

def optimize_thresholds(val_scored_candidates: Dict[str, List[Tuple[str, float]]], 
                        val_ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Grid searches over (decision_threshold, singleton_threshold) to maximize Macro F_0.5.
    val_scored_candidates: {s1_id: [(candidate_id, prob), ...]}
    val_ground_truth: {s1_id: set(matched_ids)}
    """
    best_f05 = -1.0
    best_config = {
        'decision_threshold': 0.65,
        'singleton_threshold': 0.55,
        'macro_f05': 0.0,
        'macro_precision': 0.0,
        'macro_recall': 0.0,
        'singleton_accuracy': 0.0
    }

    decision_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    singleton_grid = [0.45, 0.50, 0.55, 0.60, 0.65]

    for d_thresh in decision_grid:
        for s_thresh in singleton_grid:
            if s_thresh > d_thresh:
                continue

            predictions = {}
            for s1_id in val_ground_truth.keys():
                cand_scores = val_scored_candidates.get(s1_id, [])
                pred_set = apply_decision_policy(
                    cand_scores,
                    decision_threshold=d_thresh,
                    singleton_threshold=s_thresh
                )
                predictions[s1_id] = pred_set

            metrics = evaluate_macro_f05(predictions, val_ground_truth)
            f05 = metrics['macro_f05']

            if f05 > best_f05:
                best_f05 = f05
                best_config = {
                    'decision_threshold': d_thresh,
                    'singleton_threshold': s_thresh,
                    'macro_f05': f05,
                    'macro_precision': metrics['macro_precision'],
                    'macro_recall': metrics['macro_recall'],
                    'singleton_accuracy': metrics['singleton_accuracy']
                }

    return best_config
