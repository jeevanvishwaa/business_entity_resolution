"""
Configuration and constants for Amazon ML Challenge 2026: Business Entity Resolution.
Phase 6 Promoted Production Configuration:
- LightGBM Model E4-B (250 estimators, feature_fraction=0.65, min_data_in_leaf=30)
- Decision Threshold: 0.70, Singleton Threshold: 0.45
- Max Candidates: 60, Posting Cap: 500
- Fully portable relative paths.
"""

import os

class Config:
    # Portable relative paths:
    # PKG_DIR is code/business_entity_resolution/
    # ROOT_DIR is package root (where output/ and Documentation_template.md live)
    PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ROOT_DIR = os.path.abspath(os.path.join(PKG_DIR, "..", ".."))
    BASE_DIR = ROOT_DIR

    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    MODELS_DIR = os.path.join(PKG_DIR, "models")
    REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
    EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    # Data paths
    TRAIN_S1 = os.path.join(DATA_DIR, "train", "train_source1.tsv")
    TRAIN_S2 = os.path.join(DATA_DIR, "train", "train_source2.tsv")
    TRAIN_S3 = os.path.join(DATA_DIR, "train", "train_source3.tsv")
    TRAIN_GT = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

    TEST_S1 = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    TEST_S2 = os.path.join(DATA_DIR, "test", "test_source2.tsv")
    TEST_S3 = os.path.join(DATA_DIR, "test", "test_source3.tsv")

    # Output paths
    MATCHING_RESULTS_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_PAIRS_TSV = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    # Final promoted model and threshold paths
    MODEL_PATH = os.path.join(MODELS_DIR, "final_model.joblib")
    CONFIG_PATH = os.path.join(MODELS_DIR, "final_threshold_config.json")
    VALIDATE_SCRIPT = os.path.join(ROOT_DIR, "utils", "validate_submission.py")

    # Reproducibility
    RANDOM_SEED = 42

    # Blocking hyperparameters
    MAX_CANDIDATES_PER_BLOCK = 500  # inverted index posting list length cap
    MAX_FINAL_CANDIDATES = 60       # max candidates passed to pairwise model per S1 entity
    MIN_TOKEN_LEN_FOR_BLOCK = 4     # min token length for distinctive token blocking

    # Sampling for training pairs
    NEGATIVE_RATIO = 4

    # Phase 6 Promoted Decision Policy
    DECISION_THRESHOLD = 0.70
    SINGLETON_THRESHOLD = 0.45
    MARGIN_THRESHOLD = 0.20

    # Phase 6 Promoted Model Hyperparameters (LightGBM E4-B)
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': 8,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'n_estimators': 250,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
