"""
Singleton detection and multi-match decision policy module.
Filters candidate probabilities into final entity matching predictions.
"""

from typing import List, Tuple, Set

def apply_decision_policy(candidate_scores: List[Tuple[str, float]], 
                          decision_threshold: float = 0.65,
                          singleton_threshold: float = 0.55,
                          margin_threshold: float = 0.20) -> Set[str]:
    """
    Applies precision-oriented decision logic to candidate scores.
    candidate_scores: List of (candidate_id, probability) sorted descending by probability.
    
    Returns:
      Set of matched entity IDs, or empty set if entity is identified as a singleton.
    """
    if not candidate_scores:
        return set()

    top_id, top_score = candidate_scores[0]

    # Singleton check: if top candidate does not meet the minimum confidence threshold,
    # safely predict empty to avoid false merge penalties.
    if top_score < singleton_threshold:
        return set()

    # Collect candidates meeting the decision threshold
    accepted = set()
    for cid, score in candidate_scores:
        if score >= decision_threshold:
            accepted.add(cid)

    # If top score is between singleton_threshold and decision_threshold,
    # accept top_id ONLY if margin over second candidate is wide
    if not accepted and top_score >= singleton_threshold:
        second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0
        if (top_score - second_score) >= margin_threshold and top_score >= 0.60:
            accepted.add(top_id)

    return accepted
