"""
Evaluation module implementing the exact competition metric:
Entity-level Macro F_0.5 with singleton handling.
"""

from typing import Dict, Set, List, Tuple
import numpy as np

def compute_entity_f05(pred_set: Set[str], gt_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, f0_5) for a single Source 1 entity.
    Singletons:
      - GT empty & Pred empty -> 1.0 (precision=1.0, recall=1.0)
      - GT empty & Pred non-empty -> 0.0 (precision=0.0, recall=0.0)
    Non-singletons:
      - GT non-empty & Pred empty -> 0.0
      - GT non-empty & Pred non-empty -> standard F_0.5 formula
    """
    is_gt_singleton = len(gt_set) == 0
    is_pred_singleton = len(pred_set) == 0

    if is_gt_singleton:
        if is_pred_singleton:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if is_pred_singleton:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return precision, recall, f05

def evaluate_macro_f05(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth for all S1 entities.
    Returns:
      - macro_f05: Average F_0.5 across all entities
      - macro_precision: Average precision across all entities
      - macro_recall: Average recall across all entities
      - singleton_accuracy: Accuracy on true singletons
      - non_singleton_f05: Average F_0.5 on non-singletons
      - total_entities: Total S1 evaluated
      - false_merges_on_singletons: Count of singletons incorrectly assigned matches
    """
    precisions = []
    recalls = []
    f05_scores = []

    singleton_scores = []
    non_singleton_f05_scores = []
    false_merges_on_singletons = 0

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r, f05 = compute_entity_f05(pred_set, gt_set)

        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)

        if len(gt_set) == 0:
            singleton_scores.append(f05)
            if len(pred_set) > 0:
                false_merges_on_singletons += 1
        else:
            non_singleton_f05_scores.append(f05)

    return {
        'macro_f05': float(np.mean(f05_scores)) if f05_scores else 0.0,
        'macro_precision': float(np.mean(precisions)) if precisions else 0.0,
        'macro_recall': float(np.mean(recalls)) if recalls else 0.0,
        'singleton_accuracy': float(np.mean(singleton_scores)) if singleton_scores else 1.0,
        'non_singleton_f05': float(np.mean(non_singleton_f05_scores)) if non_singleton_f05_scores else 0.0,
        'total_entities': len(ground_truth),
        'false_merges_on_singletons': false_merges_on_singletons
    }
