# ML Challenge 2026: Business Entity Resolution Solution Documentation

**Team Name:** AmazonMLChallenge2026  
**Submission Date:** 2026-09-27  

---

## 1. Executive Summary
Our entity resolution architecture employs a high-throughput, memory-bounded two-stage pipeline designed for multi-million-record scale under strict memory constraints (< 12 GB RAM). The system combines a 6-prong inverted-index blocking engine with country-conditioned address normalization, a regularized 36-feature LightGBM gradient boosted decision tree classifier (E4-B), and a precision-weighted decision policy tuned for the competition's Macro F_0.5 metric. Evaluated on the 5,000-query frozen labeled development validation set, this promoted configuration achieved Macro F_0.5 = 0.892791 (Precision: 0.939930, Recall: 0.809188, Candidate Recall: 0.804490), delivering a +0.008039 Macro F_0.5 gain over the baseline while maintaining 100% submission validity across 1,732,544 test entities.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across Source 1 reference records and Source 2/3 target pools revealed four fundamental challenges:
1. **Transliteration and OCR noise**: Punctuation variances, casing differences, and noise tokens in business names.
2. **Abbreviation collisions across countries**: Common abbreviations have conflicting semantics across partitions (e.g., "TN" expands to "Tennessee" in the US, but "Tamil Nadu" in India; "bd" and "av" represent "boulevard" and "avenue" in France).
3. **Severe candidate explosion**: Brute-force Cartesian matching across 1.73M queries and 7.8M targets requires > 1.35e13 comparisons, necessitating aggressive sub-linear blocking.
4. **Precision-weighted metric**: Macro F_0.5 penalizes false positives (incorrect merges) twice as heavily as false negatives (missed merges), requiring conservative decision thresholds and explicit singleton detection.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Disk-Backed Inverted Index Blocking + Pairwise GBDT Classifier + Precision-Tuned Margin Policy.  
**Core Technical Contributions:**
- **Country-Conditioned Address Normalization**: Partition-aware abbreviation mapping eliminating cross-country entity confusion.
- **Provenance Bitmask Accumulation**: 6 canonical blocking strategies tracked via bitwise flags, enabling popcount candidate ranking and deterministic tie-breaking.
- **Regularized Tree Subsampling (LightGBM E4-B)**: Subsampling feature space (`feature_fraction=0.65`, `min_data_in_leaf=30`, `n_estimators=250`) preventing model overfitting on dominant token features and boosting singleton accuracy to 93.17%.
- **Precision-Optimized Decision Gating**: Dual-threshold filtering ($	au_{	ext{decision}}=0.70$, $	au_{	ext{singleton}}=0.45$) prioritizing high precision while retaining > 80% recall.

---

## 3. Candidate Generation (Blocking)

### Blocking Strategies Used:
1. `block_name_exact` (bit 1): Exact clean normalized business name match.
2. `block_name_compact` (bit 2): Alphanumeric-only compact name match (stripping all punctuation/whitespace).
3. `block_name_prefix` (bit 4): First 1–2 distinctive core tokens (excluding legal suffixes like Inc, Ltd, LLC, Pvt).
4. `block_name_token` (bit 8): Inverted index of distinctive name tokens ($\ge 4$ characters).
5. `block_addr_postal` / `block_addr_street` (bit 16 / 32): Street number combined with postal code or street token.
6. `block_addr_compact` (bit 64): Alphanumeric compact address match ($\ge 10$ characters).

### Candidate Pool Management & True Match Retention:
- **Posting List Cap**: Inverted index posting lists are capped at `max_post_len = 500` to prevent stopword explosion.
- **Candidate Cap**: Maximum final candidates per query strictly capped at `MAX_FINAL_CANDIDATES = 60`.
- **Popcount Ranking**: Candidates are ranked by the number of matching blocking blocks (popcount) descending, tie-breaking by candidate ID for strict determinism.
- **Candidate Recall**: 80.45% candidate recall verified on labeled development data, generating candidates for 99.66% of test entities (only 5,869 / 1,732,544 queries had zero candidates).

---

## 4. Matching Model

### Pairwise Features (36 Numeric Features):
- **Name Similarities (12 features)**: RapidFuzz ratio, partial ratio, token sort ratio, token set ratio, W-ratio, compact ratio, exact match flag, compact match flag, character Jaccard similarity, string length difference, token count difference, shared token count.
- **Address Similarities (11 features)**: Address ratio, partial ratio, token sort ratio, token set ratio, address Jaccard similarity, shared number count, shared number flag, postal code exact match, has_addr_s1, has_addr_s2, both_have_addr.
- **Cross-Field Interactions (3 features)**: Strong name + strong address, strong name + null address, weak name + strong address.
- **Source Indicators (2 features)**: Source 2 target indicator, Source 3 target indicator.
- **Provenance Features (8 features)**: Provenance block match count, and 7 individual block presence flags.

### Model Architecture & Training:
- **Model Type**: LightGBM Binary Classifier (`LGBMClassifier`, GBDT).
- **Hyperparameters**:
  - `n_estimators`: 250
  - `learning_rate`: 0.08
  - `num_leaves`: 63
  - `max_depth`: 8
  - `feature_fraction`: 0.65 (regularized tree subsampling)
  - `min_data_in_leaf`: 30
  - `bagging_fraction`: 0.85
- **Threshold Selection**:
  - 2D grid search optimizing Macro F_0.5 on frozen development set.
  - Decision threshold: $	au_{	ext{decision}} = 0.70$
  - Singleton threshold: $	au_{	ext{singleton}} = 0.45$

---

## 5. Results & Error Analysis

### Development Validation Performance (5,000 Labeled Validation Queries):
- **Macro F_0.5 Score**: **0.892791** (+0.008039 over Phase 2/3 baseline 0.884752)
- **Macro Precision**: **0.939930** (+0.005390 over baseline 0.934540)
- **Macro Recall**: **0.809188** (+0.013241 over baseline 0.795947)
- **Candidate Recall**: **0.804490** (+0.006006 over baseline 0.798484)
- **Singleton Accuracy**: **0.931655** (+0.017986 over baseline 0.913669)

### Error Diagnosis:
- **Common False Merges**: Occur primarily when distinct branches of franchise chains share identical corporate names and postal districts but differ in minor suite numbers. The elevated 0.70 threshold effectively eliminated 18 false positives.
- **Common False Negatives**: Arise primarily from extreme address omissions (e.g. queries providing only a city name without street or postal codes).

---

## 6. Conclusion
The promoted Phase 6 Business Entity Resolution system demonstrates robust precision, sub-linear computational scaling, and deterministic execution over 1.73M queries. By combining country-aware normalization, multi-pronged inverted index candidate generation, regularized gradient boosted feature trees, and precision-oriented threshold gating, the system ensures high-quality entity resolution while adhering to strict memory (< 6 GB peak across 3 workers) and runtime budgets (< 47 minutes full test inference).

---

## Appendix

### A. Code Artefacts
- **Runnable Code**: Contained entirely in `code/business_entity_resolution/` with `src/`, `models/`, `tests/`, `README.md`, and `requirements.txt`.
- **Pre-Trained Model**: Pre-packaged in `code/business_entity_resolution/models/final_model.joblib`.
- **Validation**: Verified against official validator `utils/validate_submission.py` with 0 errors.

### B. Output Files
- `output/matching_results.tsv`: 1,732,544 rows (matches for leaderboard evaluation).
- `output/candidate_pairs.tsv`: 1,732,544 rows (candidate sets).
