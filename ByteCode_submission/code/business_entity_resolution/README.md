# Business Entity Resolution System (Phase 6 Final Production Release)
## Amazon ML Challenge 2026

---

### 1. Project Title & Business Problem
**Challenge**: Multi-Source Business Entity Resolution under Scale & Memory Constraints.
**Objective**: Accurately resolve Reference Source-1 queries (1,732,544 records) against target business entities across Source 2 and Source 3 (7,799,589 total target pool) across France, the United States, and India.
**Evaluation Metric**: Precision-weighted Macro F_0.5 score:
$$F_{0.5} = \frac{(1 + 0.5^2) \cdot \text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

---

### 2. End-to-End System Architecture
The production system is built as a two-stage streaming entity resolution architecture:

1. **Stage 1: Inverted Index Blocking & Candidate Generation** (`src/blocking.py`)
   - 6 canonical blocking strategies:
     1. `block_name_exact` (bit 1)
     2. `block_name_compact` (bit 2)
     3. `block_name_prefix` (bit 4)
     4. `block_name_token` (bit 8)
     5. `block_addr_postal` / `block_addr_street` (bit 16 / 32)
     6. `block_addr_compact` (bit 64)
   - Country-Conditioned Address Normalization (`src/text_normalization.py`):
     - Resolves state abbreviation collisions (US TN -> Tennessee vs India TN -> Tamil Nadu).
     - Expands French road prefixes (`bd` -> `boulevard`, `av` -> `avenue`, `r` -> `rue`).
   - Bounded posting lists (`max_post_len = 500`).
   - Popcount-based candidate ranking with deterministic tie-breaking.
   - Strict candidate cap: `MAX_FINAL_CANDIDATES = 60`.

2. **Stage 2: Pairwise Feature Engineering & Regularized LightGBM** (`src/features.py`, `models/final_model.joblib`)
   - 36-dimensional numeric feature vector:
     - 12 Name string similarity and token overlap features (token sort, token set, partial, compact, jaccard).
     - 11 Address similarity and numeric token agreement features.
     - 3 Cross-field interaction features.
     - 3 Source provenance indicator flags.
     - 7 Blocking strategy provenance indicators.
   - Model: **LightGBM E4-B (Regularized Tree Subsampling)**:
     - `n_estimators = 250`
     - `feature_fraction = 0.65`
     - `min_data_in_leaf = 30`
     - `num_leaves = 63`

3. **Stage 3: Precision-Weighted Decision Policy & Singleton Detection** (`src/singleton_detection.py`)
   - Decision threshold: $\tau_{\text{decision}} = 0.70$
   - Singleton threshold: $\tau_{\text{singleton}} = 0.45$
   - Enforces candidate subset invariant: $\text{matched\_ids} \subseteq \text{candidate\_ids}$.

---

### 3. Repository Structure
```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration with portable paths
│   ├── text_normalization.py     # Country-aware name and address normalization
│   ├── blocking.py               # 6-prong SQLite & in-memory candidate generation
│   ├── features.py               # 36 pairwise feature extraction engine
│   ├── singleton_detection.py    # Precision-oriented decision threshold policy
│   ├── threshold_optimization.py # Threshold tuning and evaluation
│   ├── data_loader.py            # TSV streaming data loaders
│   ├── evaluation.py             # Macro F0.5 evaluation metrics
│   ├── predict.py                # Disk-backed streaming test inference
│   ├── train.py                  # LightGBM model training pipeline
│   └── dataset_audit.py          # Data inventory and profiling
├── models/
│   ├── final_model.joblib        # Promoted LightGBM E4-B production model
│   └── final_threshold_config.json # Threshold settings (0.70 / 0.45)
├── tests/
│   ├── test_candidate_generation.py
│   └── test_normalization_and_model.py
├── run_pipeline.py               # CLI master pipeline entry point
├── requirements.txt              # Pinned environment dependencies
└── README.md                     # Technical system documentation
```

---

### 4. Installation & Environment Setup
Requires Python 3.10 or 3.11:
```bash
pip install -r requirements.txt
```

---

### 5. Execution Commands
To reproduce pipeline stages:
```bash
# 1. Run unit and integration tests
pytest -v tests/

# 2. Run inference using pre-trained Phase 6 model
python run_pipeline.py --mode predict

# 3. Validate submission formatting using official validator
python run_pipeline.py --mode validate_submission
```

---

### 6. Official Competition Outputs
The finalized, competition-ready outputs are located at:
- `output/matching_results.tsv`: Exactly 1,732,544 rows (matches for leaderboard scoring).
- `output/candidate_pairs.tsv`: Exactly 1,732,544 rows (blocking candidate set).

---

### 7. Limitations & Technical Notes
- Ground truth for the competition test set is sequestered; development performance gains (+0.008039 Macro F0.5) were measured on the 5,000-query frozen development set.
- All candidate generation and streaming inference operations are memory-bounded (< 600 MB RAM for single worker; < 6 GB RAM across parallel workers).
