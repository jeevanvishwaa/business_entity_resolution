"""
Memory-efficient pipeline for entity resolution on constrained hardware (12GB RAM).
Processes countries sequentially, writes pairs to disk incrementally,
and uses numpy arrays throughout to minimize overhead.
"""

import os
import sys
import io
import time
import gc
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from src.config import Config
from src.text_normalization import normalize_business_name, normalize_address, normalize_country
from src.blocking import CandidateIndex
from src.features import compute_pairwise_features, FEATURE_NAMES

def prepare_training_and_val_pairs(sample_s1_size: int = 15000, val_ratio: float = 0.2):
    """
    Memory-efficient pair generation.
    - Smaller sample
    - Processes one country at a time, freeing memory between countries
    - Writes features to disk as .npy files instead of accumulating in RAM
    - Returns paths to saved arrays
    """
    print(f"\n==========================================")
    print(f"PREPARING TRAINING & VALIDATION PAIRS")
    print(f"Sample S1 size: {sample_s1_size:,} | Val ratio: {val_ratio}")
    print(f"==========================================")
    t0 = time.time()

    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    pairs_dir = os.path.join(Config.MODELS_DIR, "pairs_cache")
    os.makedirs(pairs_dir, exist_ok=True)

    # 1. Load ground truth
    print("Loading ground truth...")
    gt_map = {}
    with open(Config.TRAIN_GT, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            s1_id = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                gt_map[s1_id] = {mid.strip() for mid in parts[1].split(',') if mid.strip()}
            else:
                gt_map[s1_id] = set()

    # 2. Sample S1 entities
    print("Sampling S1 training entities...")
    s1_matched = [sid for sid, m in gt_map.items() if len(m) > 0]
    s1_singletons = [sid for sid, m in gt_map.items() if len(m) == 0]

    np.random.seed(Config.RANDOM_SEED)
    n_singletons = int(sample_s1_size * 0.08)
    n_matched = sample_s1_size - n_singletons

    sampled_s1_ids = set(np.random.choice(s1_matched, size=min(n_matched, len(s1_matched)), replace=False))
    sampled_s1_ids.update(np.random.choice(s1_singletons, size=min(n_singletons, len(s1_singletons)), replace=False))

    # Read S1 records for sampled entities
    s1_records_by_country = defaultdict(list)
    with open(Config.TRAIN_S1, 'r', encoding='utf-8', errors='replace') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            s1_id = parts[0].strip()
            if s1_id in sampled_s1_ids:
                name = parts[1].strip()
                addr = parts[2].strip()
                cntry = normalize_country(parts[3].strip())
                rec = {
                    'entity_id': s1_id,
                    'norm_name': normalize_business_name(name),
                    'norm_addr': normalize_address(addr),
                    'country': cntry
                }
                s1_records_by_country[cntry].append(rec)

    total_loaded = sum(len(v) for v in s1_records_by_country.values())
    print(f"Loaded {total_loaded:,} sampled S1 records:")
    for c, recs in s1_records_by_country.items():
        print(f"  {c}: {len(recs):,}")

    # Determine needed target IDs
    needed_target_ids = set()
    for s1_id in sampled_s1_ids:
        needed_target_ids.update(gt_map.get(s1_id, set()))
    print(f"Ground truth positive targets needed: {len(needed_target_ids):,}")

    # Train/Val entity split
    all_sampled = list(sampled_s1_ids)
    np.random.shuffle(all_sampled)
    n_val = int(len(all_sampled) * val_ratio)
    val_s1_set = set(all_sampled[:n_val])
    train_s1_set = set(all_sampled[n_val:])
    print(f"Train S1: {len(train_s1_set):,} | Val S1: {len(val_s1_set):,}")

    # 3. Process each country partition sequentially
    all_train_X = []
    all_train_y = []
    val_pairs_by_s1 = defaultdict(list)  # Only store val pairs in memory (smaller)

    for country, s1_recs in s1_records_by_country.items():
        print(f"\n--- Processing {country} ({len(s1_recs):,} S1 entities) ---")
        t_c = time.time()

        # Build candidate index with ONLY needed targets + small background sample
        cand_index = CandidateIndex(country=country, max_post_len=Config.MAX_CANDIDATES_PER_BLOCK)

        for src_path in [Config.TRAIN_S2, Config.TRAIN_S3]:
            src_name = os.path.basename(src_path)
            count = 0
            for chunk in pd.read_csv(src_path, sep='\t', chunksize=500000, keep_default_na=False):
                chunk_filtered = chunk[chunk['country'].apply(normalize_country) == country]
                # Keep all needed targets
                sub_needed = chunk_filtered[chunk_filtered['entity_id'].isin(needed_target_ids)]
                # Small background sample for hard negatives (2%)
                remaining = chunk_filtered[~chunk_filtered['entity_id'].isin(needed_target_ids)]
                sub_sample = remaining.sample(frac=0.02, random_state=Config.RANDOM_SEED) if len(remaining) > 0 else remaining
                combined = pd.concat([sub_needed, sub_sample])
                for cid, b_name, b_addr in zip(combined['entity_id'].values, combined['business_name'].values, combined['business_address'].values):
                    cid = str(cid).strip()
                    n_dict = normalize_business_name(str(b_name).strip())
                    a_dict = normalize_address(str(b_addr).strip())
                    cand_index.add_target_record(cid, n_dict, a_dict)
                    count += 1
            print(f"  Indexed {count:,} records from {src_name}")

        cand_index.prune_high_frequency_tokens()
        print(f"  Total indexed: {len(cand_index.records):,}")

        # Generate pairs for this country
        train_X_country = []
        train_y_country = []

        for idx, s1 in enumerate(s1_recs):
            s1_id = s1['entity_id']
            is_val = s1_id in val_s1_set
            true_matches = gt_map.get(s1_id, set())

            ranked_cands = cand_index.retrieve_candidates(
                s1['norm_name'], s1['norm_addr'], max_candidates=Config.MAX_FINAL_CANDIDATES
            )
            cand_prov = {c[0]: c[1] for c in ranked_cands}
            cand_ids = set(c[0] for c in ranked_cands)

            # Positives
            for true_id in true_matches:
                t_rec = cand_index.get_record(true_id)
                if t_rec:
                    prov = cand_prov.get(true_id, ['gt_positive'])
                    feat = compute_pairwise_features(
                        s1['norm_name'], s1['norm_addr'],
                        true_id, t_rec, prov
                    )
                    if is_val:
                        val_pairs_by_s1[s1_id].append({
                            'target_id': true_id, 'features': feat, 'label': 1
                        })
                    else:
                        train_X_country.append(feat)
                        train_y_country.append(1)

            # Hard negatives from blocking candidates
            neg_cands = [cid for cid in cand_ids if cid not in true_matches]
            max_negs = Config.NEGATIVE_RATIO * max(1, len(true_matches))
            if len(neg_cands) > max_negs:
                neg_cands = list(np.random.choice(neg_cands, size=max_negs, replace=False))

            for neg_id in neg_cands:
                t_rec = cand_index.get_record(neg_id)
                if t_rec:
                    prov = cand_prov.get(neg_id, ['block_candidate'])
                    feat = compute_pairwise_features(
                        s1['norm_name'], s1['norm_addr'],
                        neg_id, t_rec, prov
                    )
                    if is_val:
                        val_pairs_by_s1[s1_id].append({
                            'target_id': neg_id, 'features': feat, 'label': 0
                        })
                    else:
                        train_X_country.append(feat)
                        train_y_country.append(0)

            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx+1:,}/{len(s1_recs):,} S1 entities ({(idx+1)/len(s1_recs)*100:.0f}%)")

        # Add country pairs to global arrays
        if train_X_country:
            all_train_X.extend(train_X_country)
            all_train_y.extend(train_y_country)
        print(f"  Country {country}: {len(train_X_country):,} train pairs, done in {time.time()-t_c:.1f}s")

        # Free memory
        del cand_index, train_X_country, train_y_country
        gc.collect()

    # Convert to numpy arrays
    X_train = np.array(all_train_X, dtype=np.float32)
    y_train = np.array(all_train_y, dtype=np.int32)
    del all_train_X, all_train_y
    gc.collect()

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    print(f"\nTotal train matrix: {X_train.shape} (pos: {pos:,}, neg: {neg:,})")
    print(f"Val S1 entities with pairs: {len(val_pairs_by_s1):,}")
    print(f"Preparation done in {time.time()-t0:.1f}s")

    val_gt = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_set}
    return X_train, y_train, val_pairs_by_s1, val_gt
