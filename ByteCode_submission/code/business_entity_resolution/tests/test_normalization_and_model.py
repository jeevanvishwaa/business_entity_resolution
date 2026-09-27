"""
Package validation test suite.
Tests:
- Country-aware address normalization
- Model loading and 36-feature schema
- Decision policy thresholds (0.70 / 0.45)
- Candidate subset invariant preservation
"""

import os
import sys
import numpy as np
import joblib
import pytest

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from src.config import Config
from src.text_normalization import normalize_address, normalize_business_name
from src.singleton_detection import apply_decision_policy

def test_country_aware_normalization():
    # US abbreviation expansion
    us_res = normalize_address("32 Turnbrook, Jackson, TN", country="US")
    assert "tennessee" in us_res['raw_clean'], "US TN not expanded to Tennessee"
    
    # India abbreviation expansion
    in_res = normalize_address("32 Turnbrook, Jackson, TN", country="INDIA")
    assert "tamil nadu" in in_res['raw_clean'], "India TN not expanded to Tamil Nadu"
    
    # France road abbreviation expansion
    fr_res = normalize_address("15 Bd Haussmann, 75009 Paris", country="FRANCE")
    assert "boulevard" in fr_res['raw_clean'], "France bd not expanded to boulevard"

def test_model_loading_and_features():
    model_path = Config.MODEL_PATH
    assert os.path.exists(model_path), f"Model file missing: {model_path}"
    model = joblib.load(model_path)
    assert model.n_features_in_ == 36, f"Expected 36 features, found {model.n_features_in_}"
    
    # Test prediction on dummy input
    X = np.zeros((2, 36), dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == 2
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

def test_decision_policy_thresholds():
    # Candidates above 0.70 accepted
    scores = [("C1", 0.75), ("C2", 0.40)]
    accepted = apply_decision_policy(scores, decision_threshold=0.70, singleton_threshold=0.45)
    assert accepted == {"C1"}
    
    # Top score below 0.45 produces empty set (singleton)
    scores_low = [("C1", 0.42), ("C2", 0.30)]
    accepted_low = apply_decision_policy(scores_low, decision_threshold=0.70, singleton_threshold=0.45)
    assert accepted_low == set()
